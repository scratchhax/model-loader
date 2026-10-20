from __future__ import annotations

import threading
import time

import httpx
from docker.errors import DockerException, NotFound

from .. import strata as strata_engine
from ..config import settings

from .backends import (_effective_container_names, _extract_ports,
                      discover_llama_containers, engine_for)
from .dockerbase import _docker_client
from .gpu import gpu_holders


# ---------- Strata load control ----------

# Strata's server keeps its model between two states the rest of this app has no word for:
# loaded, and parked - port open, VRAM free, model files PERHAPS still in the OS page cache.
# POST /unload moves it to parked in ~0.3 s and gives back BOTH the VRAM and the pinned expert
# arena; POST /load brings it back, and blocks until the model answers - so the caller either
# waits on it or polls /health, never a browser request.
#
# Parked is cheap to ENTER and NOT cheap to leave. A /load after an engine swap was measured at
# 104 s on 2026-10-15, against the ~90 s a cold container start costs - no saving at all. An
# earlier version of this comment claimed parked was "seconds from serving again"; that was
# reasoning from a warm page cache, never a measurement, and it does not survive the situation
# it was written for, because the other engine reads its own 25 GB model in between and evicts
# the cache the argument depended on.
#
# The Content-Type is not decoration: the server rejects anything else on these two paths
# (415) as a CSRF guard, and checks Origin for the same reason - a plain form POST from any
# web page must not be able to unload the model. This app sends JSON with no Origin header,
# which is exactly what that guard lets through.

_strata_errors: dict[str, str] = {}
_strata_errors_lock = threading.Lock()

# /metrics is read on the speedometer's 2 s cadence; a 2 s TTL means each poll costs at
# most one request even when several panels want it in the same tick.
_strata_metrics_cache: dict[str, tuple[float, "dict | None"]] = {}


def strata_metrics(name: str) -> dict | None:
    """GET /metrics as a dict, cached for 2 seconds; None when it does not answer.

    The speedometer used to scrape Strata's log because that was the only place the rates
    lived. /metrics carries the same numbers as data - tok_s while generating,
    decode_tok_s and hit_rate for the last request, a monotonic completed count - and
    unlike a log regex it cannot drift out of date when the log format moves.
    """
    now = time.time()
    t, val = _strata_metrics_cache.get(name, (0.0, None))
    if now - t < 2.0:
        return val
    val = None
    client = _docker_client()
    if client is not None:
        try:
            attrs = client.containers.get(name).attrs or {}
            _, port = _extract_ports(attrs)
            if port is not None:
                with httpx.Client(timeout=httpx.Timeout(2.0)) as c:
                    r = c.get(f"http://{name}:{port}/metrics")
                if r.status_code == 200:
                    val = r.json()
        except (NotFound, DockerException, httpx.HTTPError, ValueError):
            val = None
    _strata_metrics_cache[name] = (now, val)
    return val


def strata_model_ids(name: str) -> list[str]:
    """The model ids a running Strata serves, from its own /v1/models. [] when unreachable."""
    client = _docker_client()
    if client is None:
        return []
    try:
        attrs = client.containers.get(name).attrs or {}
        _, port = _extract_ports(attrs)
        if port is None:
            return []
        with httpx.Client(timeout=httpx.Timeout(3.0)) as c:
            r = c.get(f"http://{name}:{port}/v1/models")
        if r.status_code != 200:
            return []
        return [str(m.get("id")) for m in (r.json() or {}).get("data") or [] if m.get("id")]
    except (NotFound, DockerException, httpx.HTTPError, ValueError):
        return []


def _strata_control(name: str, path: str, timeout: float) -> tuple[bool, str]:
    """POST one of Strata's control endpoints. (ok, status-or-reason).

    The refusals are the interesting part and each has its own shape upstream: 409 when a
    request is running or queued (it will not unload mid-request), 503 when the GPU it would
    load onto is busy (min_free_vram_mib) or the engine is stuck. Both are worth relaying
    verbatim - "the GPU is in use by another program" is the coexistence answer this whole
    feature exists to make possible.
    """
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    try:
        attrs = client.containers.get(name).attrs or {}
    except (NotFound, DockerException) as e:
        return False, f"{name}: {e}"
    _, port = _extract_ports(attrs)
    if port is None:
        return False, f"{name} exposes no API port"
    url = f"http://{name}:{port}{path}"
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout)) as c:
            r = c.post(url, headers={"Content-Type": "application/json"}, content=b"{}")
    except httpx.TimeoutException:
        return False, f"{url}: {_TIMED_OUT} after {timeout:.0f}s"
    except httpx.HTTPError as e:
        return False, f"{url}: {e}"
    if r.status_code == 200:
        try:
            return True, str((r.json() or {}).get("status") or "ok")
        except ValueError:
            return True, "ok"
    if r.status_code == 409:
        return False, "a request is running or queued - Strata will not unload mid-request"
    if r.status_code == 503:
        try:
            msg = str(((r.json() or {}).get("error") or {}).get("message"))
        except ValueError:
            msg = "GPU busy or engine stuck"
        return False, msg
    return False, f"HTTP {r.status_code} from {url}"


_TIMED_OUT = "timed out"

# How long /unload may take to ANSWER. It used to be 10 s on the belief that an unload takes
# ~0.3 s. It does not always: on 2026-10-20 a park of the UD-IQ4_XS pack at 262K context ran
# past 10 s and still succeeded - /health said loaded:false and both cards were back to 60 MB -
# while the Park button reported "Failed: .../unload: timed out", and the engine switch, which
# calls this same function, would have aborted a perfectly good switch on it.
_UNLOAD_TIMEOUT_S = 60.0
# After a timeout, how long to watch /health for the park to land before calling it a failure.
_UNLOAD_CONFIRM_S = 60.0


def strata_loaded(name: str) -> bool | None:
    """/health's loaded bool for one Strata backend; None when it does not answer."""
    client = _docker_client()
    if client is None:
        return None
    try:
        attrs = client.containers.get(name).attrs or {}
        _, port = _extract_ports(attrs)
        if port is None:
            return None
        with httpx.Client(timeout=httpx.Timeout(3.0)) as c:
            r = c.get(f"http://{name}:{port}/health")
        if r.status_code != 200:
            return None
        return bool((r.json() or {}).get("loaded"))
    except (NotFound, DockerException, httpx.HTTPError, ValueError, AttributeError):
        return None


def strata_unload(name: str) -> tuple[bool, str]:
    """Park a running Strata: VRAM and pinned RAM back now, container and page cache warm.

    Synchronous on purpose, because its two refusals (a request in flight, a stuck engine) are
    only useful if the caller hears them right now.

    A TIMEOUT is not one of those refusals and is not reported as one. The server releases the
    model before it answers, and how long that takes grows with the pack and the context arena,
    so "no answer yet" says nothing about whether the park happened. /health does: loaded:false
    means it landed, however late the answer to the POST. Only a model still loaded after the
    confirm window is a failure.
    """
    with _strata_errors_lock:
        _strata_errors.pop(name, None)
    ok, msg = _strata_control(name, "/unload", timeout=_UNLOAD_TIMEOUT_S)
    if ok or _TIMED_OUT not in msg:
        return ok, msg
    deadline = time.time() + _UNLOAD_CONFIRM_S
    while True:
        if strata_loaded(name) is False:
            return True, "parked (Strata answered slowly, /health confirms it is unloaded)"
        if time.time() >= deadline:
            return False, f"{msg}, and /health still reports the model loaded"
        time.sleep(1.0)


def strata_load(name: str) -> tuple[bool, str]:
    """Load a parked Strata. Fire-and-forget.

    The endpoint blocks until the model answers - seconds from the page cache, up to two
    minutes cold - and a request that waited on that would time out in the browser long
    before it finished. The card's own 2 s poll shows the flip; a failure (usually the 503
    "the GPU is in use") lands in strata_error() for the card to name.
    """
    with _strata_errors_lock:
        _strata_errors.pop(name, None)

    def _do() -> None:
        ok, msg = _strata_control(name, "/load", timeout=420.0)
        with _strata_errors_lock:
            if ok:
                _strata_errors.pop(name, None)
            else:
                _strata_errors[name] = msg

    threading.Thread(target=_do, daemon=True, name=f"strata-load-{name}").start()
    return True, ("loading — seconds from the page cache, up to two minutes cold; "
                  "the card flips when the model answers")


def strata_error(name: str, loaded: bool = False) -> str:
    """The last failed load/unload for this backend, "" when none. Clears itself once the
    model is visibly loaded again, so a stale 503 cannot outlive the state that caused it."""
    with _strata_errors_lock:
        if loaded:
            _strata_errors.pop(name, None)
        return _strata_errors.get(name, "")


def strata_held_cards() -> tuple[int, ...]:
    """The cards a RUNNING Strata backend holds, per the pin file. () when no Strata runs,
    or it runs unpinned.

    The file, not the container's baked environment: the documented entry point sources
    this file on every start, so after any restart the file is what the process is on,
    while docker inspect keeps reporting create-time values until the container is
    recreated. (Measured on this box: a re-pinned engine held one card while its baked env
    still said 1,0.) Empty on any failure - whole cards are the conservative answer when
    the truth is unknown, and it is the answer the app gave before Strata existed at all.
    """
    try:
        client = _docker_client()
        if client is None:
            return ()
        for c in client.containers.list(filters={"label": f"ai-lab.engine={strata_engine.ENGINE}"}):
            sel = strata_engine.read_pin()
            if sel:
                return tuple(int(x) for x in sel.split(","))
    except Exception:  # noqa: BLE001
        return ()
    return ()


def strata_timeshare_keys() -> dict:
    """The time-share keys the Strata card's settings ask for, None where switched off.

    The None entries are the point as much as the values: the merge writer deletes them
    from the run config, so switching idle unload OFF actually takes `idle_unload_s` out
    of the file instead of leaving a parked-server timer nobody asked for anymore.
    """
    from .. import db
    keys: dict = {}
    v = db.get_setting("strata_idle_unload_s", "")
    keys["idle_unload_s"] = int(v) if v.isdigit() and int(v) > 0 else None
    v = db.get_setting("strata_min_free_vram_mib", "")
    keys["min_free_vram_mib"] = int(v) if v.isdigit() and int(v) > 0 else None
    # The hook is a shell command Strata runs (timeout 120 s) before every load, and a
    # failed hook is a warning in its log, not a refused load - so the curl gets a timeout
    # shorter than Strata's own and the whole thing degrades to "load anyway", which is
    # what a box with nothing to yield wants.
    keys["before_load"] = (f"curl -fsS -m 110 -X POST {settings.strata_yield_url}"
                           if db.get_setting("strata_yield_on_load", "") == "1" else None)
    return keys


def strata_desired_keys() -> dict:
    """Everything this app wants in the run config: the env baseline, then the card's
    time-share settings on top - the UI is the more recent, more deliberate writer."""
    return {**settings.strata_config_key_map, **strata_timeshare_keys()}


def strata_yield() -> str:
    """Stop every running backend that is not Strata, so its reload gets the cards.
    One line back.

    Called by Strata's before_load hook, which runs before the engine starts and waits
    15 s for the VRAM to actually appear - so this only has to stop things, not prove they
    are gone. Stopping (not unloading) is the only thing a llama router at --models-max 1
    answers to, and that is the same fact the Engine switch was built on.
    """
    from .. import gpu_procs
    stopped = []
    for r in gpu_holders():
        if not r["running"] or not r["backend"]:
            continue
        if r["engine"] == strata_engine.ENGINE:
            continue          # never touch Strata from Strata's own hook
        ok, msg = gpu_procs.set_container_running(r["name"], start=False)
        if not ok:
            return f"could not stop {r['name']}: {msg}"
        stopped.append(r["name"])
    return "stopped " + ", ".join(stopped) if stopped else "nothing to stop"


def declared_reserves_gb() -> float:
    """Sum of `ai-lab.vram-reserve-gb` labels over every container, running or not.

    A declared tenant (chatterbox, comfyui) is on-demand: it holds nothing most of the
    time, and the label's whole point is that its share stays reserved while it is stopped,
    so llama's fit never takes the last free GB and then has to evict the tenant that
    starts next. The container declares what it needs; the fit math believes it.
    """
    client = _docker_client()
    if client is None:
        return 0.0
    total = 0.0
    try:
        for c in client.containers.list(all=True):
            raw = ((c.labels or {}).get("ai-lab.vram-reserve-gb") or "").strip()
            if not raw:
                continue
            try:
                total += max(0.0, float(raw))
            except ValueError:
                continue
    except DockerException:
        return 0.0
    return round(total, 1)


def _fit_backends() -> dict[str, float]:
    """{backend_name: total VRAM GiB} for everything we can actually plan against.

    Prefers the live probe over settings.gpu_vram_map. The static map is an optional
    override and is empty by default — relying on it alone silently disabled the fit
    chips entirely once the hardcoded example values were removed.

    Declared tenant reserves (ai-lab.vram-reserve-gb) come off the pool: the number a
    plan may spend, not the number the card has.
    """
    from .. import hw
    reserves = declared_reserves_gb()
    out: dict[str, float] = {}
    for name in _effective_container_names():
        vram = float(settings.gpu_vram_map.get(name, 0) or 0)
        if vram <= 0:
            st = hw.stats_for(name)
            if st.ok and st.gpu and st.gpu.vram_total_gb > 0:
                vram = float(st.gpu.vram_total_gb)
        if vram > 0:
            out[name] = round(max(0.0, vram - reserves), 1)
    return out


# Above this, a CPU-resident DENSE model is technically loadable and practically unusable.
# Calibrated on measured tok/s on a dual-channel DDR5 box: a 6.6 GB dense model managed
# 4.4 tok/s, implying roughly 29 GB/s of effective read bandwidth, so 30 GB of dense weights
# lands near 1 tok/s.
#
# KNOWN LIMITATION: this is size-based, and size is the wrong axis for MoE. A mixture-of-
# experts model reads only its active experts per token, so a 30 GB MoE can be several times
# faster than a 30 GB dense one — measured here, a 15.8 GB MoE beat a 6.6 GB dense by 2x.
# Telling them apart needs expert_count/expert_used_count from the GGUF header, which the
# chips do not have (they receive a file size and nothing else). The tooltip says so rather
# than pretending the number applies to both.
_CPU_CRAWL_GB = 30.0


def _cpu_backends() -> list[str]:
    """Names of discovered llama containers with no GPU — they run on the CPU.

    _fit_backends() keys on VRAM and so cannot see these at all. They are still real places
    to run a model: the constraint is system RAM rather than VRAM, and ctx-size is the only
    fit lever, since ngl, n-cpu-moe and tensor-split all presuppose a GPU.
    """
    names = set(_effective_container_names())
    out: list[str] = []
    for d in discover_llama_containers():
        n = d.get("name") or ""
        if n in names and (d.get("vendor") or "").lower() in ("cpu", "", "unknown"):
            from .. import hw
            st = hw.stats_for(n)
            if not (st.ok and st.gpu and st.gpu.vram_total_gb > 0):
                out.append(n)
    return out


def vram_fit_chips(size_bytes: int, summary: dict | None = None) -> list[dict]:
    """Per-backend fit verdicts: {'name', 'vram_gb', 'verdict', 'ratio_pct'}.

    verdict is one of {fits, tight, oom, streams, impossible}.

    `oom` and `impossible` are genuinely different answers and must not look alike. `oom`
    means "not on the GPU alone" — offload moves layers into system RAM and it runs, slower.
    `impossible` means the model exceeds VRAM **plus** RAM, so there is nowhere for those
    layers to go and no setting rescues it. Rendering both as a red cross against a backend
    name tells the user the GPU is the constraint, when for `impossible` the machine is.

    An impossible model returns a SINGLE machine-level chip rather than one per backend,
    because naming backends implies picking a different one would help.

    `summary` is the repo's GGUF header (optional — gated repos and header-fetch failures
    fall back to size-only verdicts). It exists for one question the size cannot answer:
    is this a mixture-of-experts model on Strata? Strata's design mode is to keep attention
    and the hot experts resident and STREAM the routed experts from host RAM, so an MoE
    larger than VRAM is not an offload-candidate-slower — it is the normal case. Measured
    on this box: a 78.1 GiB IQ3_S Flash-Next running on Strata at ~120 tok/s while the
    size-only chip called it oom against 63.7 GiB of VRAM. Dense models keep the VRAM-only
    verdicts on every engine, because for them there is nothing to stream.
    """
    from .. import hw
    gb = size_bytes / (1024 ** 3)
    moe = bool(((summary or {}).get("model") or {}).get("expert_count"))

    # usable, not total: the OS reserve is already spoken for (see hw.usable_ram_gb).
    ram = hw.usable_ram_gb()
    # Strata's own pinned arena is streaming budget, not competition: loading a different
    # model on strata reclaims it, so for an MoE the machine's real ceiling includes it.
    # Other engines must not count on it — pinned anonymous memory does not give itself
    # back to a llama-server process.
    arena = hw.strata_arena_gb() if moe else 0.0
    ceiling = ram + arena
    pooled = max((v for v in _fit_backends().values() if v > 0), default=0.0)
    if ram > 0 and pooled > 0 and gb > (pooled + ceiling):
        return [{
            "name": "this machine",
            "vram_gb": round(pooled, 1),
            "host_ram_gb": round(ceiling, 1),
            "verdict": "impossible",
            "ratio_pct": round(gb / (pooled + ceiling) * 100),
            "needs_gb": round(gb),
            "ceiling_gb": round(pooled + ceiling),
        }]

    out: list[dict] = []
    for name, vram in _fit_backends().items():
        if vram <= 0:
            continue
        ratio = gb / vram
        if ratio < 0.65:
            verdict = "fits"
        elif ratio < 0.9:
            verdict = "tight"
        elif moe and ceiling > 0 and engine_for(name) == "strata":
            verdict = "streams"
        else:
            verdict = "oom"
        chip = {
            "name": name,
            "vram_gb": vram,
            "verdict": verdict,
            "ratio_pct": round(ratio * 100),
        }
        if verdict == "streams":
            # The honest denominator for a streaming verdict is the combined pool the
            # model actually spans; the VRAM-only ratio would read >100% and re-look
            # like the oom this verdict exists to correct.
            chip["host_ram_gb"] = round(ceiling, 1)
            chip["ratio_pct"] = round(gb / (vram + ceiling) * 100)
        out.append(chip)

    # CPU backends are sized against usable RAM. "Fits" and "usable" diverge sharply here:
    # generation is RAM-bandwidth-bound, so a large dense model can occupy memory perfectly
    # well and still produce well under a token per second. A plain green tick would be
    # true and misleading, so anything past a threshold gets its own 'slow' verdict.
    for name in _cpu_backends():
        if ram <= 0:
            continue
        ratio = gb / ram
        if ratio >= 1.0:
            verdict = "oom"
        elif gb >= _CPU_CRAWL_GB:
            verdict = "slow"
        elif ratio < 0.75:
            verdict = "fits"
        else:
            verdict = "tight"
        out.append({
            "name": name,
            "vram_gb": ram,
            "verdict": verdict,
            "ratio_pct": round(ratio * 100),
            "is_cpu": True,
        })
    return out
