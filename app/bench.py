"""Benchmark harness: run a fixed prompt set against models and record what happened.

Deliberately a dead end. Nothing here feeds back into autoconfig or models.ini — the output
is for a person to look at and understand their hardware. That is why every measurement is
stored raw rather than reduced to a score, and why there is no "apply these findings" path.

Two things follow from that being the whole purpose:

  * The machine must be exactly as it was afterwards. This phase does not write models.ini at
    all — it benchmarks sections as they already exist — so the only lasting effect is which
    model happens to be resident at the end, which the router would change on the next request
    anyway.
  * A run is disruptive and slow. The router holds one model at a time (--models-max 1), so
    benchmarking several means evicting and reloading each in turn, and anything else using
    the box will stall or fail for the duration. The UI warns before starting; this module
    makes sure the warning is true by refusing to run two at once and by supporting cancel
    between every unit of work.

Measurements come from llama-server's own response rather than being timed from outside where
possible. A streaming request with stream_options.include_usage ends with a `timings` object
carrying prompt_n/prompt_ms, predicted_n/predicted_ms and draft_n/draft_n_accepted, which are
the server's view of its own work. Only two things have to be measured from here: time to
first token, which is a property of the wire and not visible to the server, and load time,
which happens before any request exists to report it.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

import httpx

from . import db, hw

# Generation is capped so a run has a predictable duration and every variant does the same
# amount of work. Without a cap, one model rambling to its context limit turns a five-minute
# suite into an hour and makes the per-variant numbers incomparable.
# Reasoning models spend a large part of their budget thinking before they answer, and a cap
# that runs out mid-thought measures only how fast a model thinks - never whether it can write
# the code or the prose the prompt was asking for.
#
# Measured against the seeded prompts, letting both local models run to a natural stop:
#
#   gemma-4-E4B   coding 1376 (406 thinking)   reasoning 2394 (1247 thinking)
#   gemma-4-12b   coding 1500 (716 thinking)   reasoning 2115 (1292 thinking)
#
# So 512 truncated three prompts out of four and 1024 would still have cut the two hardest.
# 3072 clears the worst observed case with room to spare.
#
# Raising the cap costs less time than it looks: a cap is only paid when it is reached, and
# these finish on their own between 550 and 2400 tokens. What it does end is every request
# costing exactly 512 tokens because every request was being cut off.
DEFAULT_MAX_TOKENS = 3072

# Sampling temperature is pinned to 0 for every benchmark request. Output length drives almost
# every derived number here, so letting it vary run to run would put noise into the one place
# it cannot be averaged out.
_TEMPERATURE = 0.0

_CONNECT_TIMEOUT_S = 15.0
# Generous: this read spans a cold model load on a 27B, which is tens of seconds before a
# single byte comes back.
_READ_TIMEOUT_S = 900.0

_LOCK = threading.Lock()
_CANCEL = threading.Event()
_THREAD: threading.Thread | None = None


@dataclass
class JobState:
    """Live progress of the one benchmark that may be running."""
    run_id: int = 0
    status: str = "idle"          # idle | running | cancelling | done | cancelled | error
    backend: str = ""
    total: int = 0
    done: int = 0
    current: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    lines: list[str] = field(default_factory=list)

    @property
    def pct(self) -> int:
        return int(round(100.0 * self.done / self.total)) if self.total else 0

    @property
    def elapsed_s(self) -> float:
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at) if self.started_at else 0.0

    @property
    def eta_s(self) -> float:
        """Remaining seconds, projected from work already completed.

        Only meaningful once something has finished; before that there is no rate to project
        from and the honest answer is "unknown", not a number derived from one guess.
        """
        if self.done <= 0 or self.status != "running":
            return 0.0
        per = self.elapsed_s / self.done
        return max(0.0, per * (self.total - self.done))

    @property
    def active(self) -> bool:
        return self.status in ("running", "cancelling")


_STATE = JobState()


def state() -> JobState:
    with _LOCK:
        return _STATE


def cancel() -> bool:
    """Ask the run to stop. It finishes the request in flight, then unwinds."""
    with _LOCK:
        if not _STATE.active:
            return False
        _STATE.status = "cancelling"
        _STATE.lines.append("cancel requested — finishing the request in flight")
    _CANCEL.set()
    return True


def _log(msg: str) -> None:
    with _LOCK:
        _STATE.lines.append(msg)
        # The panel shows the tail; an unbounded list would grow for the life of the process.
        if len(_STATE.lines) > 200:
            del _STATE.lines[:-200]


def _resolve_endpoint(container: str) -> tuple[str, str]:
    """(base_url, error). Containers are addressed by name on the docker network, the same way
    the rest of the app reaches them."""
    from . import services
    client = services._docker_client()
    if client is None:
        return "", "docker unreachable"
    try:
        c = client.containers.get(container)
    except Exception as e:  # noqa: BLE001
        return "", f"{type(e).__name__}: {e}"
    try:
        _, port = services._extract_ports(c.attrs or {})
    except Exception:  # noqa: BLE001
        port = None
    if not port:
        return "", "no internal port exposed"
    return f"http://{container}:{port}", ""


def loaded_aliases(base_url: str) -> set[str]:
    try:
        r = httpx.get(f"{base_url}/v1/models", timeout=10.0)
        r.raise_for_status()
        return {str(m.get("id") or "") for m in (r.json().get("data") or [])
                if (m.get("status") or {}).get("value") == "loaded"}
    except (httpx.HTTPError, ValueError):
        return set()


def measure(base_url: str, alias: str, prompt: str, max_tokens: int) -> dict:
    """One streamed completion. Returns the server's own timings plus wire-measured TTFT.

    Two first-token times, because reasoning models have two.

    `ttft_ms` is the first VISIBLE token of any kind, reasoning included - that is the pause a
    person actually experiences, since a client streaming the thinking shows it immediately.
    `ttft_answer_ms` is the first token of the answer proper, which on a thinking model can be
    thousands of tokens later and is the wait before anything useful appears. Reporting only
    the first would misdescribe one kind of model or the other.

    Neither counts the initial role-only delta llama.cpp emits, which carries no text: nothing
    is on screen until a character is.
    """
    payload = {
        "model": alias,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": _TEMPERATURE,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    out: dict = {"err": ""}
    t0 = time.perf_counter()
    ttft: float | None = None
    ttft_answer: float | None = None
    finish_reason = ""
    timings: dict = {}
    try:
        with httpx.Client(timeout=httpx.Timeout(_CONNECT_TIMEOUT_S, read=_READ_TIMEOUT_S)) as cl:
            with cl.stream("POST", f"{base_url}/v1/chat/completions", json=payload) as r:
                if r.status_code != 200:
                    r.read()
                    return {"err": f"HTTP {r.status_code}: {r.text[:200]}"}
                for line in r.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    body = line[6:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except ValueError:
                        continue
                    if chunk.get("timings"):
                        timings = chunk["timings"]
                    for ch in (chunk.get("choices") or []):
                        d = ch.get("delta") or {}
                        if ch.get("finish_reason"):
                            finish_reason = str(ch["finish_reason"])
                        visible = (d.get("content") or "") or (d.get("reasoning_content") or "")
                        if visible and ttft is None:
                            ttft = time.perf_counter() - t0
                        if (d.get("content") or "") and ttft_answer is None:
                            ttft_answer = time.perf_counter() - t0
    except httpx.HTTPError as e:
        return {"err": f"{type(e).__name__}: {e}"}

    total = time.perf_counter() - t0
    draft_n = int(timings.get("draft_n") or 0)
    draft_ok = int(timings.get("draft_n_accepted") or 0)
    out.update({
        "ttft_ms": round(ttft * 1000, 2) if ttft is not None else None,
        "ttft_answer_ms": round(ttft_answer * 1000, 2) if ttft_answer is not None else None,
        # "length" means the token cap cut it off, so this row describes an unfinished
        # response and its content-based numbers should be read with that in mind.
        "truncated": 1 if finish_reason == "length" else 0,
        "total_ms": round(total * 1000, 2),
        "prompt_n": int(timings.get("prompt_n") or 0),
        "prompt_tps": timings.get("prompt_per_second"),
        "gen_n": int(timings.get("predicted_n") or 0),
        "gen_tps": timings.get("predicted_per_second"),
        "draft_n": draft_n or None,
        "draft_acc": round(draft_ok / draft_n, 4) if draft_n else None,
    })
    return out


def _peak_vram(backend: str) -> list[float]:
    """Per-card VRAM in GB, from the sampler the dashboard already runs.

    Its cache refreshes on a couple of seconds, so this is a sample rather than a true peak.
    That is honest for what it is used for — how close a config runs to the edge — and far
    cheaper than a dedicated nvidia-smi poll for the duration of every request.
    """
    try:
        st = hw.stats_for(backend)
        gpu = getattr(st, "gpu", None)
        if gpu and getattr(gpu, "cards", None):
            return [round(c.vram_used_gb, 2) for c in gpu.cards]
        if gpu:
            return [round(gpu.vram_used_gb, 2)]
    except Exception:  # noqa: BLE001
        pass
    return []


def _read_load_ms(container: str, alias: str) -> float | None:
    """How long the router took to bring `alias` up, from its own log.

    Anchored on the router announcing the load and the spawned instance announcing it is
    listening. Measured here rather than inferred from a cold request's latency, which would
    fold prompt processing and first-token time into the number.
    """
    from . import services
    try:
        client = services._docker_client()
        if client is None:
            return None
        c = client.containers.get(container)
        raw = c.logs(tail=800, stdout=True, stderr=True, timestamps=True)
        text = raw.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None

    from . import telemetry
    start_ts = None
    for line in text.splitlines():
        if f"ensure_model: model name={alias} is not loaded" in line:
            start_ts = telemetry._parse_ts(line)
        elif start_ts and "llama_server: listening on http://127.0.0.1:" in line:
            end_ts = telemetry._parse_ts(line)
            if end_ts and end_ts >= start_ts:
                return round((end_ts - start_ts) * 1000, 1)
            start_ts = None
    return None


def _other_traffic(container: str, aliases: set[str], since: float) -> bool:
    """Did anything OTHER than the benchmark hit this backend during the window?

    Warning the user is not the same as preventing them, and someone else on the LAN can still
    fire a request. A contended measurement is still worth keeping — it just must not be
    mistaken for a clean one.
    """
    from . import services, telemetry
    try:
        client = services._docker_client()
        if client is None:
            return False
        c = client.containers.get(container)
        text = c.logs(tail=400, stdout=True, stderr=True, timestamps=True).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return False
    samples, _ = telemetry.parse_log(text)
    return any(s.ts >= since and s.alias and s.alias not in aliases for s in samples)


def start(*, backend: str, aliases: list[str], prompt_ids: list[int],
          reps: int = 3, max_tokens: int = DEFAULT_MAX_TOKENS) -> tuple[bool, str]:
    """Kick off a run in the background. One at a time, refused otherwise."""
    global _THREAD, _STATE
    aliases = [a for a in aliases if a]
    if not (backend or "").strip():
        return False, "no backend selected"
    if not aliases:
        return False, "no models selected"
    prompts = db.prompts_by_ids(prompt_ids)
    if not prompts:
        return False, "no prompts selected"
    reps = max(1, min(int(reps or 1), 10))
    max_tokens = max(16, min(int(max_tokens or DEFAULT_MAX_TOKENS), 4096))

    with _LOCK:
        if _STATE.active:
            return False, "a benchmark is already running"

    base_url, err = _resolve_endpoint(backend)
    if err:
        return False, err

    _CANCEL.clear()
    started = time.time()
    run_id = db.bench_create_run(backend, reps, max_tokens, started)
    with _LOCK:
        # A fresh object rather than mutating the last run's, so no field can survive by
        # being one someone forgot to reset.
        _STATE = JobState(run_id=run_id, status="running", backend=backend,
                          total=len(aliases) * len(prompts) * reps, started_at=started)

    _THREAD = threading.Thread(
        target=_run, name="benchmark",
        args=(run_id, backend, base_url, aliases,
              [(p["name"], p["body"]) for p in prompts], reps, max_tokens),
        daemon=True)
    _THREAD.start()
    return True, ""


def _run(run_id: int, backend: str, base_url: str, aliases: list[str],
         prompts: list[tuple[str, str]], reps: int, max_tokens: int) -> None:
    status, err = "done", ""
    alias_set = set(aliases)
    try:
        for alias in aliases:
            if _CANCEL.is_set():
                status = "cancelled"
                break
            variant_id = db.bench_add_variant(run_id, alias)
            was_loaded = alias in loaded_aliases(base_url)
            _log(f"{alias}: starting{' (already resident)' if was_loaded else ''}")
            first_request = True

            for prompt_name, prompt_body in prompts:
                for rep in range(1, reps + 1):
                    if _CANCEL.is_set():
                        status = "cancelled"
                        break
                    # The first request of a variant pays for the model load and starts with an
                    # empty cache. Recorded, but marked cold so it can be read separately
                    # instead of dragging the median of everything else.
                    cold = first_request and not was_loaded
                    first_request = False
                    with _LOCK:
                        _STATE.current = f"{alias} · {prompt_name} · rep {rep}/{reps}"
                    t_req = time.time()
                    m = measure(base_url, alias, prompt_body, max_tokens)
                    db.bench_add_result(
                        variant_id, prompt_name=prompt_name, rep=rep, cold=1 if cold else 0,
                        ttft_ms=m.get("ttft_ms"), ttft_answer_ms=m.get("ttft_answer_ms"),
                        truncated=m.get("truncated") or 0, total_ms=m.get("total_ms"),
                        prompt_n=m.get("prompt_n"), prompt_tps=m.get("prompt_tps"),
                        gen_n=m.get("gen_n"), gen_tps=m.get("gen_tps"),
                        draft_n=m.get("draft_n"), draft_acc=m.get("draft_acc"),
                        peak_vram_json=json.dumps(_peak_vram(backend)),
                        contended=1 if _other_traffic(backend, alias_set, t_req) else 0,
                        err=m.get("err") or "")
                    with _LOCK:
                        _STATE.done += 1
                    if m.get("err"):
                        _log(f"{alias} · {prompt_name} rep {rep}: {m['err']}")
                    elif cold:
                        db.bench_set_load_ms(variant_id, _read_load_ms(backend, alias))
                if _CANCEL.is_set():
                    break
            if not _CANCEL.is_set():
                _log(f"{alias}: done")
    except Exception as e:  # noqa: BLE001
        status, err = "error", f"{type(e).__name__}: {e}"
        _log(f"failed: {err}")

    if _CANCEL.is_set() and status != "error":
        status = "cancelled"
    finished = time.time()
    try:
        db.bench_finish_run(run_id, status, finished, err)
    except Exception:  # noqa: BLE001
        pass
    with _LOCK:
        _STATE.status = status
        _STATE.finished_at = finished
        _STATE.error = err
        _STATE.current = ""
    _CANCEL.clear()


def estimate_seconds(n_aliases: int, n_prompts: int, reps: int, max_tokens: int) -> int:
    """Rough duration for the confirmation dialog.

    Deliberately pessimistic. A number that undersells the disruption is worse than one that
    oversells it: the whole point of showing it is so nobody starts an hour-long run thinking
    it will take five minutes.
    """
    load_s = 25.0                        # cold load of a mid-size model, per variant
    # Assume a request uses roughly 60% of its cap rather than all of it: models stop when they
    # are done, and measured completions ran 550-2400 tokens against a 4096 cap. Costing every
    # request at the full cap made the estimate grow with a number that is rarely reached.
    per_req_s = 4.0 + (max_tokens * 0.6) / 35.0
    return int(n_aliases * (load_s + n_prompts * reps * per_req_s))


# ---------------------------------------------------------------- throughput sweeps
#
# Prompt-processing and generation throughput are llama.cpp's own `llama bench` territory, and
# it does them better than driving the server would: it warms up, repeats, reports a standard
# deviation, sweeps parameters natively, and measures the model rather than the HTTP path.
#
# What it cannot do is measure the configuration actually in use. It has no speculative
# decoding, no mmproj and no server slots, so on a model running draft-mtp it reports the
# unaccelerated speed - 133 tok/s against the 208 the server really delivers. That is why both
# engines exist rather than one: this for clean throughput under known parameters, the prompt
# suite for what the running configuration does to a real request.
#
# What this adds over running llama-bench by hand is that a sweep starts from a models.ini
# section, so it measures the settings that section actually uses rather than llama-bench's
# defaults (f16 cache, flash-attn auto), which would describe a configuration nobody runs.

# ini keys worth carrying into a sweep, mapped to their llama-bench flag. Anything llama-bench
# does not understand is dropped rather than guessed at.
_SWEEP_FROM_INI = (
    ("cache-type-k", "-ctk"),
    ("cache-type-v", "-ctv"),
    ("flash-attn", "-fa"),
    ("ngl", "-ngl"),
    ("n-gpu-layers", "-ngl"),
    ("n-cpu-moe", "-ncmoe"),
    ("split-mode", "-sm"),
    ("tensor-split", "-ts"),
    ("main-gpu", "-mg"),
    ("ubatch-size", "-ub"),
    ("batch-size", "-b"),
)


def sweep_args_for_section(name: str) -> list[str]:
    """llama-bench flags reflecting what this models.ini section is configured to do."""
    from . import ini
    vals = ini.get_section(name) or {}
    out: list[str] = []
    seen: set[str] = set()
    for key, flag in _SWEEP_FROM_INI:
        v = str(vals.get(key, "")).strip()
        if not v or flag in seen:
            continue
        if flag == "-ts":
            # models.ini writes tensor-split comma-separated; llama-bench wants slashes.
            v = v.replace(",", "/")
        out += [flag, v]
        seen.add(flag)
    return out


def _sweep_runtime(backend: str) -> tuple[str, dict, bool, str]:
    """(image, volume binds, wants_gpu, error) mirrored from the llama backend container.

    Derived from the container rather than configured separately, so a sweep runs against the
    same image, the same models directory and the same devices as the server. A sweep that
    measured a different build or a different mount would quietly be answering about something
    other than what is deployed.
    """
    from . import services
    if not (backend or "").strip():
        return "", {}, False, "no backend selected"
    client = services._docker_client()
    if client is None:
        return "", {}, False, "docker unreachable"
    try:
        c = client.containers.get(backend)
    except Exception as e:  # noqa: BLE001
        return "", {}, False, f"{type(e).__name__}: {e}"
    attrs = c.attrs or {}
    try:
        image = ((c.image.tags or []) or [""])[0]
    except Exception:  # noqa: BLE001
        image = ""
    if not image:
        return "", {}, False, "cannot determine backend image"
    binds = {}
    for m in (attrs.get("Mounts") or []):
        src, dst = m.get("Source"), m.get("Destination")
        if src and dst:
            binds[src] = {"bind": dst, "mode": "ro"}
    wants_gpu = bool((attrs.get("HostConfig") or {}).get("DeviceRequests"))
    return image, binds, wants_gpu, ""


def run_sweep_once(backend: str, model_path: str, extra: list[str],
                   n_prompt: int, n_gen: int, depths: str, reps: int,
                   timeout_s: int = 1800) -> tuple[list[dict], str]:
    """Run `llama bench` in a throwaway container. Returns (entries, error)."""
    import docker as _docker
    from . import services

    image, binds, wants_gpu, err = _sweep_runtime(backend)
    if err:
        return [], err
    cmd = ["bench", "-m", model_path, "-p", str(n_prompt), "-n", str(n_gen),
           "-r", str(reps), "-o", "json"]
    if depths.strip():
        cmd += ["-d", depths.strip()]
    cmd += extra

    client = services._docker_client()
    if client is None:
        return [], "docker unreachable"
    kwargs: dict = {
        "image": image, "command": cmd, "entrypoint": "/app/llama",
        "volumes": binds, "detach": True, "remove": False,
        "network_mode": "none",   # nothing here needs the network, so keep it off the LAN
    }
    if wants_gpu:
        kwargs["device_requests"] = [_docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]

    try:
        cont = client.containers.run(**kwargs)
    except Exception as e:  # noqa: BLE001
        return [], f"{type(e).__name__}: {e}"

    try:
        deadline = time.time() + timeout_s
        while True:
            cont.reload()
            if cont.status not in ("running", "created"):
                break
            if _CANCEL.is_set():
                try:
                    cont.kill()
                except Exception:  # noqa: BLE001
                    pass
                return [], "cancelled"
            if time.time() > deadline:
                try:
                    cont.kill()
                except Exception:  # noqa: BLE001
                    pass
                return [], f"timed out after {timeout_s}s"
            time.sleep(1.0)
        raw = cont.logs(stdout=True, stderr=False).decode("utf-8", "replace")
        # llama-bench writes progress and backend chatter to stderr; stdout is the JSON array.
        start = raw.find("[")
        if start >= 0:
            try:
                return json.loads(raw[start:]), ""
            except ValueError:
                pass  # fall through and report what llama-bench actually said

        # Every failure here lands on the same path, because the interesting information is
        # always on stderr and never on stdout. A truncated stdout is what a failed run looks
        # like - llama-bench opens the JSON array before it loads the model, so a model that
        # will not load leaves exactly "[\n" behind. Reporting that as a parse error blamed the
        # parser for something it had no part in; the actual message is "failed to load model".
        errtxt = cont.logs(stdout=False, stderr=True).decode("utf-8", "replace")
        line = ""
        for cand in reversed(errtxt.strip().splitlines()):
            c = cand.strip()
            if c and ("error" in c.lower() or "failed" in c.lower()):
                line = c
                break
        if not line:
            line = (errtxt.strip().splitlines() or ["no output"])[-1].strip()
        return [], line[:200]
    finally:
        try:
            cont.remove(force=True)
        except Exception:  # noqa: BLE001
            pass


def start_sweep(*, backend: str, aliases: list[str], n_prompt: int = 512, n_gen: int = 128,
                depths: str = "0,4096,16384", reps: int = 3) -> tuple[bool, str]:
    global _THREAD, _STATE
    aliases = [a for a in aliases if a]
    if not (backend or "").strip():
        return False, "no backend selected"
    if not aliases:
        return False, "no models selected"
    with _LOCK:
        if _STATE.active:
            return False, "a benchmark is already running"

    from . import ini
    targets: list[tuple[str, str, list[str]]] = []
    for a in aliases:
        vals = ini.get_section(a) or {}
        rel = str(vals.get("model", "")).strip()
        if not rel:
            return False, f"section '{a}' has no model path"
        # models.ini paths are what llama-server is given, and the sweep container mounts the
        # same models directory at the same place, so they resolve unchanged.
        targets.append((a, rel, sweep_args_for_section(a)))

    _CANCEL.clear()
    started = time.time()
    run_id = db.bench_create_run(backend, reps, n_gen, started)
    with _LOCK:
        _STATE = JobState(run_id=run_id, status="running", backend=backend,
                          total=len(targets), started_at=started)
    _THREAD = threading.Thread(target=_run_sweep, name="benchmark-sweep",
                               args=(run_id, backend, targets, n_prompt, n_gen, depths, reps),
                               daemon=True)
    _THREAD.start()
    return True, ""


def _free_gpu(backend: str, base_url: str) -> None:
    """Evict whatever the router is holding, so a sweep container has the cards to itself.

    llama-bench loads the model into its OWN process, so anything the server still has resident
    is competing for the same VRAM - and a large model then simply fails to load, which is what
    "failed to load model" means when there is apparently plenty of memory. The router has no
    unload endpoint, so restarting the backend is the available lever. Nothing is lost by it:
    models are loaded on demand, so the next request brings back whatever it needs.
    """
    try:
        if not loaded_aliases(base_url):
            return
    except Exception:  # noqa: BLE001
        return
    from . import services
    _log("freeing GPU: restarting the backend so the sweep has the cards to itself")
    try:
        services.restart_llama_backend(backend)
    except Exception as e:  # noqa: BLE001
        _log(f"could not restart {backend}: {e}")
        return
    # Wait for the router to answer again, so the rest of the app is not left talking to a
    # container that is still coming up.
    for _ in range(60):
        if _CANCEL.is_set():
            return
        try:
            if httpx.get(f"{base_url}/v1/models", timeout=3.0).status_code == 200:
                _log("backend back up, GPUs clear")
                return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    _log("backend did not answer within 60s; continuing anyway")


def _run_sweep(run_id: int, backend: str, targets: list, n_prompt: int, n_gen: int,
               depths: str, reps: int) -> None:
    status, err = "done", ""
    try:
        base_url, _e = _resolve_endpoint(backend)
        if base_url:
            _free_gpu(backend, base_url)
        for alias, model_path, extra in targets:
            if _CANCEL.is_set():
                status = "cancelled"
                break
            with _LOCK:
                _STATE.current = f"{alias} - llama bench"
            _log(f"{alias}: {' '.join(extra) or 'section defaults'}")
            entries, e = run_sweep_once(backend, model_path, extra, n_prompt, n_gen, depths, reps)
            if e:
                _log(f"{alias}: {e}")
                if e == "cancelled":
                    status = "cancelled"
                    break
            for entry in entries:
                db.bench_add_sweep(run_id, alias, entry)
            if entries:
                _log(f"{alias}: {len(entries)} measurements")
            with _LOCK:
                _STATE.done += 1
    except Exception as e:  # noqa: BLE001
        status, err = "error", f"{type(e).__name__}: {e}"
        _log(f"failed: {err}")

    if _CANCEL.is_set() and status != "error":
        status = "cancelled"
    finished = time.time()
    try:
        db.bench_finish_run(run_id, status, finished, err)
    except Exception:  # noqa: BLE001
        pass
    with _LOCK:
        _STATE.status = status
        _STATE.finished_at = finished
        _STATE.error = err
        _STATE.current = ""
    _CANCEL.clear()
