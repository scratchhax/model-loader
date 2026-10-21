from __future__ import annotations

import asyncio
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from docker.errors import DockerException, NotFound

from ..config import settings
from ..utils import container_env

from .dockerbase import _docker_client
from .hotpath import _http


# ---------- llama backends: richer view + restart + probe ----------

# error-of-last-restart per container, shown in the card until it clears
_restart_errors: dict[str, str] = {}
_restart_errors_lock = threading.Lock()


@dataclass
class LlamaBackend:
    name: str
    found: bool
    status: str = ""           # running | exited | restarting | not_found | error
    image: str = ""
    short_id: str = ""
    started_at: str = ""
    uptime: str = ""
    host_ports: list[str] = field(default_factory=list)
    internal_port: int | None = None
    loaded_model: str | None = None
    probe_error: str | None = None
    # A genuine load failure reported by the router — a model whose /v1/models status is an
    # error value, with its message. Distinct from probe_error (a probe/HTTP problem) and from
    # mere idleness: under --models-preset with --models-max 1, "nothing loaded" is the normal
    # resting state, so nothing-loaded must never read as a failure. None = no reported failure.
    load_failed: str | None = None
    # How long ago this app first saw that failure (persisted, so a restart does not make an old
    # failure look new). llama.cpp keeps a model's failed flag until the NEXT attempt to load
    # that model, which on a backend that was never meant to serve it is never: llama-voice
    # carried a gemma-4-31B failure from a misrouted request for days under a red "load failed"
    # chip while serving wheatley-voice perfectly. See load_failed_current.
    load_failed_age: float = 0.0
    # The model the router is bringing up right now, and how long it has been at it. Both None /
    # 0.0 whenever nothing is loading, which is almost always. loading_since is wall-clock
    # seconds measured by US, not by the router: /v1/models reports the state but not when it
    # started, so the clock begins the first time a probe sees it. The probe runs every 1.5 s
    # (_HERO_BACKENDS_TTL), so the figure can read up to ~1.5 s short of the truth - fine for a
    # counter whose job is to prove the box is busy, not to time anything.
    loading_model: str | None = None
    loading_since: float = 0.0
    # A model the server still owns but whose VRAM it has released under --sleep-idle-seconds.
    # Never folded into loaded_model: see _probe_loaded_model for why that would stop sleep
    # working at all. The next request reloads it, measured at 8.2-9.4 s on this box.
    sleeping_model: str | None = None
    # A parked single-model server (Strata): the port answers, the model says "unloaded", the
    # VRAM is free and the files are in the page cache - seconds from serving, one request or
    # one POST /load away. Distinct from sleeping (which is llama's --sleep-idle-seconds, where
    # the server still owns the model) and from an idle router (which lists every models.ini
    # section as unloaded and is not one button from serving). See _probe_loaded_model.
    unloaded_model: str | None = None
    last_restart_error: str | None = None
    # --sleep-idle-seconds: after this many seconds without a request the server releases its
    # model's VRAM and reloads on the next one. 0 = the flag is absent, which is llama.cpp's
    # default and means the model holds its allocation until something evicts it.
    sleep_idle_s: int = 0
    # Whether it is sleeping RIGHT NOW. Only meaningful when sleep_idle_s > 0.
    asleep: bool = False
    # True when launched with --models-preset/--models-dir, i.e. it serves models.ini and can
    # swap models. False for a -m server such as the Orpheus TTS backend, which serves exactly
    # one file and is not a chat endpoint.
    router: bool = True
    # "rocm" | "cuda" | "cpu" | "unknown", from the image tag. Whether this backend can touch
    # the cards at all, which is what decides who leads the overview.
    vendor: str = ""
    # Which inference stack this is: "llama" for every llama.cpp image, or whatever the
    # container declares in its `ai-lab.engine` label. Strata is the first non-llama one.
    #
    # It exists because almost everything this app knows is llama.cpp-specific: models.ini,
    # autoconfig's KV and compute-buffer budget, the argv vocabulary, --models-max eviction.
    # A backend of another engine is a real backend - it holds a card, serves an OpenAI API
    # and answers /v1/models - but none of that machinery describes it, so the code that
    # would otherwise size it or write its arguments has to be able to ask.
    engine: str = "llama"

    @property
    def load_failed_current(self) -> bool:
        """Is the load failure something to act on now, or history?

        Current for FAILURE_FRESH_S after it happened, history after that - still shown on the
        Containers page, dimmed, with its age, never as a red alarm.

        Not "current while the backend holds nothing", which was the first version: an idle
        router at --models-max 1 holds nothing most of the time, so a five-day-old failure on
        idle llama-voice stayed red exactly as before.
        """
        return bool(self.load_failed) and self.load_failed_age < FAILURE_FRESH_S


def _parse_started_at(iso: str) -> tuple[str, str]:
    # docker returns e.g. "2025-08-11T00:51:00.123456789Z"
    if not iso or iso.startswith("0001"):
        return "", ""
    try:
        # trim nanoseconds to microseconds
        core, _, frac = iso.partition(".")
        if frac:
            frac = frac.rstrip("Z")[:6]
            iso_norm = f"{core}.{frac}+00:00"
        else:
            iso_norm = core.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_norm)
    except ValueError:
        return iso, ""
    local = dt.astimezone().strftime("%Y-%m-%d %H:%M")
    delta = datetime.now(timezone.utc) - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        up = f"{secs}s"
    elif secs < 3600:
        up = f"{secs // 60}m"
    elif secs < 86400:
        up = f"{secs // 3600}h {(secs % 3600) // 60}m"
    else:
        up = f"{secs // 86400}d {(secs % 86400) // 3600}h"
    return local, up


def _extract_ports(attrs: dict) -> tuple[list[str], int | None]:
    """Return (['8082->8080/tcp'], 8080).

    The internal port is one that ACTUALLY has a host binding. Multi-port
    containers used to get whichever port docker enumerated last - including an
    unbound one - and the probes then targeted a closed port and reported
    "nothing loaded" for a server that was fine.
    """
    ports_map = (attrs.get("NetworkSettings") or {}).get("Ports") or {}
    result: list[str] = []
    internal: int | None = None
    any_port: int | None = None
    for cont_port, bindings in ports_map.items():
        # cont_port like "8080/tcp"
        try:
            port = int(cont_port.split("/")[0])
        except ValueError:
            continue
        if any_port is None:
            any_port = port
        if not bindings:
            continue
        if internal is None:
            internal = port
        for b in bindings:
            hp = b.get("HostPort")
            if hp:
                result.append(f"{hp}->{cont_port}")
    return result, (internal if internal is not None else any_port)


def _api_key_from_attrs(attrs: dict) -> str:
    """The llama-server --api-key, if the container was started with one.

    Compose `command:` lands in Config.Cmd, an image entrypoint in Config.Entrypoint; scan both
    so a key set either way is found. Returns "" when the server runs open (the common LAN case).
    """
    cfg = attrs.get("Config") or {}
    argv = [str(t) for t in (cfg.get("Entrypoint") or [])] + [str(t) for t in (cfg.get("Cmd") or [])]
    for i, tok in enumerate(argv):
        if tok == "--api-key" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--api-key="):
            return tok.split("=", 1)[1]
    return ""


def browser_endpoints(browser_host: str) -> list[dict]:
    """Per-backend, browser-reachable /v1 base URL + api key, for client snippets.

    Probing uses docker-internal http://<container>:<port>, which the browser cannot resolve.
    Here we take the published host port that llama-server's API port is bound to and pair it
    with the address the browser actually used to reach Model Loader, so a snippet works from
    any machine on the LAN rather than only from inside the compose network.
    """
    client = _docker_client()
    if client is None or not browser_host:
        return []
    out: list[dict] = []
    for name in _effective_container_names():
        try:
            attrs = client.containers.get(name).attrs or {}
        except (NotFound, DockerException):
            continue
        host_ports, internal = _extract_ports(attrs)
        host_port = None
        if internal is not None:
            suffix = f"{internal}/"
            for hp in host_ports:  # each is "8082->8080/tcp"
                left, _, cont = hp.partition("->")
                if cont.startswith(suffix) and left.isdigit():
                    host_port = left
                    break
        if host_port is None:  # no binding matched the API port; fall back to first numeric
            for hp in host_ports:
                left, _, _ = hp.partition("->")
                if left.isdigit():
                    host_port = left
                    break
        if host_port is None:
            continue
        out.append({
            "name": name,
            "base_url": f"http://{browser_host}:{host_port}/v1",
            "api_key": _api_key_from_attrs(attrs),
        })
    return out


def _load_failure(status: dict) -> tuple[bool, int | None]:
    """(failed, exit_code) for one model's /v1/models status object.

    Measured on this llama.cpp build by loading a section whose model file does not exist:

        before the attempt : {"value": "unloaded"}
        after it failed    : {"value": "unloaded", "exit_code": 1, "failed": true}

    There is no distinct error `value` and no message; the model returns to unloaded and gains a
    flag, which persists. A model evicted normally reads plain {"value": "unloaded"} with neither
    key, so the flag is specific to failure and an ordinary model swap never trips it. The check
    this replaced looked for an error in `value`, which this router never sends, so the dashboard
    alarm could not fire. A non-zero exit code alone is also treated as a failure, since a clean
    stop leaves no exit code at all.
    """
    raw = status.get("exit_code")
    try:
        code = int(raw) if raw is not None else None
    except (TypeError, ValueError):
        code = None
    return (status.get("failed") is True or (code is not None and code != 0)), code


async def _probe_loaded_model(
    container_name: str, internal_port: int | None
) -> tuple[str | None, str | None, str | None, str | None, str | None, str | None]:
    """(loaded_model_csv, probe_error, load_failed, loading_model, sleeping_model, unloaded_model).

    loaded_model  — comma list of ids whose status is 'loaded'.
    probe_error   — a problem talking to the endpoint (HTTP/JSON), or an idle-state note.
    load_failed   — the first model whose last load failed, as "name (exit N)"; None otherwise.
                    The only field the dashboard alarm keys off: a real failure, not the normal
                    unloaded state. See _load_failure for what the router actually reports.
    loading_model — the model the router is bringing up RIGHT NOW, or None.
    sleeping_model — a model held under --sleep-idle-seconds: the server still owns it and will
                    reload it on the next request, but its VRAM is released. Measured: 50.1 GB
                    awake, 7.3 GB asleep on this box.
    unloaded_model — a single-model server (Strata) whose port answers while its one model says
                    "unloaded": the VRAM is free and a request - or POST /load - brings it back
                    in seconds, because the model files stay in the OS page cache. Only set when
                    the server lists EXACTLY ONE model and that one says unloaded. A llama router
                    lists every models.ini section that way, and for it "N configured, none
                    loaded" is the honest note - a parked Strata and an idle router are different
                    states and only one of them is one button away from serving.

    sleeping is kept OUT of loaded_model on purpose, and it is not a stylistic choice. Callers
    treat loaded_model as "there is something here worth probing", and the probe is /slots -
    which both resets the idle timer and wakes a sleeper. Folding the two together would mean
    the overview page, polling every 500 ms, silently prevented any model from ever sleeping
    and woke the one that already had. Display and liveness have to stay separate fields.

    The last of those is why this function was wrong for as long as it has existed. Measured on
    b11206 by polling this endpoint at 120 ms through a real swap:

        0.00  gemma-4-31B-it-Q6_K=loaded
        1.11  Qwen3.8-27B-UD-Q4_K_M=loading
        10.51 Qwen3.8-27B-UD-Q4_K_M=loaded

    The outgoing model flips straight to 'unloaded' and the incoming one sits on 'loading', so
    for the whole window NOTHING reports as loaded. Collecting only 'loaded' therefore made a
    model swap indistinguishable from an idle box, and the overview said "nothing loaded" for
    the entire load - 9.6 s on a 27B, 40 s on gemma-4-31B at Q6_K. The status was always on the
    wire; it was being thrown away here.
    """
    if internal_port is None:
        return None, None, None, None, None, None
    url = f"http://{container_name}:{internal_port}/v1/models"
    try:
        r = await _http().get(url, timeout=httpx.Timeout(2.0, read=3.0),
                              headers=auth_headers(container_name))
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}", None, None, None, None
        data = r.json()
        items = data.get("data") or []
        if not items:
            return None, "no models configured", None, None, None, None
        loaded_ids: list[str] = []
        unloaded_ids: list[str] = []
        failure: str | None = None
        loading: str | None = None
        sleeping: str | None = None
        for it in items:
            mid = str(it.get("id") or "")
            # A ROUTER reports status.value per model because it loads and evicts them. A
            # server started with -m has no status field at all, because it has nothing to
            # load or unload: the one model named on its command line is resident for the
            # life of the process. Reading an absent status as "not loaded" made Orpheus
            # TTS report "1 configured, none loaded" while it was actively speaking, and
            # left the overview claiming nothing was running whenever it was the only
            # backend holding a model. Its id is the container path, so show the stem.
            if "status" not in it:
                stem = mid.rsplit("/", 1)[-1]
                if stem.lower().endswith(".gguf"):
                    stem = stem[:-5]
                if stem:
                    loaded_ids.append(stem)
                continue
            status = it.get("status") or {}
            val = str(status.get("value") or "").lower()
            if val == "loaded":
                loaded_ids.append(mid)
            elif val == "sleeping":
                sleeping = mid
            elif val == "loading":
                # At --models-max 1 there can only be one, and it flips within ~0.3 s of the
                # request - early enough to cover the eviction of its predecessor, so this
                # one field spans the whole episode the user is staring at.
                loading = mid
            elif val == "unloaded" and not _load_failure(status)[0]:
                # A FAILED llama model reads value "unloaded" too, with the flag set -
                # so the failure check has to keep seeing those, and only a clean
                # unloaded lands here.
                unloaded_ids.append(mid)
            elif failure is None:
                failed, code = _load_failure(status)
                if failed:
                    failure = f"{mid} (exit {code})" if code is not None else mid
        if failure is not None:
            return (", ".join(i for i in loaded_ids if i) or None, None, failure,
                    loading, sleeping, None)
        if loaded_ids:
            return ", ".join(i for i in loaded_ids if i), None, None, loading, sleeping, None
        # "none loaded" is the honest note only when nothing is on its way in and nothing is
        # merely asleep. Either of those read as an idle box, which is the confusion this
        # whole function keeps being wrong about.
        if loading or sleeping:
            return None, None, None, loading, sleeping, None
        # One model, and it says unloaded: that is not an idle router with an ini full of
        # candidates, it is a parked single-model server - VRAM free, files in the page
        # cache, seconds from serving. Saying "1 configured, none loaded" made it read as
        # the router case and hid the one fact that distinguishes them.
        if len(items) == 1 and len(unloaded_ids) == 1:
            return None, None, None, None, None, unloaded_ids[0]
        return None, f"{len(items)} configured, none loaded", None, None, None, None
    except (httpx.HTTPError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}", None, None, None, None


def _engine_label(c) -> str:
    """The engine a container declares in its `ai-lab.engine` label, lowercased, or "".

    Backends were found purely by image name, which holds exactly as long as every backend is
    a llama.cpp tag. Strata is the first one that is not: a different stack, its own locally
    built image, and nothing in models.ini applying to it. An explicit label is the
    declaration. Sniffing for "strata" in an image name would be the same substring guess one
    rung further down, and image tags are not ours to depend on.
    """
    try:
        labels = ((c.attrs or {}).get("Config") or {}).get("Labels") or {}
    except DockerException:
        return ""
    if not isinstance(labels, dict):
        return ""
    return str(labels.get("ai-lab.engine") or "").strip().lower()


def discover_llama_containers() -> list[dict]:
    """Return metadata for every container that is a backend: a llama.cpp:server-* image, or
    any container declaring an `ai-lab.engine` label. Includes stopped containers so the user
    can see and start them.
    """
    client = _docker_client()
    if client is None:
        return []
    out: list[dict] = []
    try:
        containers = client.containers.list(all=True)
    except DockerException:
        return []
    for c in containers:
        try:
            tags = c.image.tags or [c.image.short_id]
            img = (tags[0] if tags else "").lower()
        except DockerException:
            img = ""
        engine = _engine_label(c)
        if not engine and "ghcr.io/ggml-org/llama.cpp" not in img and "llama.cpp" not in img:
            continue
        if c.name == "model-loader":
            continue
        engine = engine or "llama"
        vendor = "rocm" if "rocm" in img else "cuda" if "cuda" in img else ("cpu" if "server" in img else "unknown")
        # Router or single-model server? A router is given --models-preset/--models-dir and
        # serves whatever models.ini holds; a server given -m serves exactly one file and has
        # no relationship to models.ini at all. They are both llama.cpp:server images, so the
        # image tag cannot tell them apart - only the command line can. Orpheus TTS is the
        # case that forced the distinction: a -m server on port 5006 that this app would
        # otherwise offer to register with Open WebUI as a chat endpoint, at :8080.
        try:
            cmd = (c.attrs or {}).get("Config", {}).get("Cmd") or []
        except DockerException:
            cmd = []
        joined = " ".join(str(t) for t in cmd) if isinstance(cmd, list) else str(cmd)
        # Only a llama.cpp image can be a router. Another engine has no models.ini to serve
        # and no eviction to do, so it is a one-model server by construction - the same shape
        # as a `-m` llama server, which is exactly what `router = False` already means here.
        is_router = engine == "llama" and ("--models-preset" in joined or "--models-dir" in joined)
        out.append({"name": c.name, "image": img, "vendor": vendor,
                    "router": is_router, "engine": engine, "api_key": api_key_of(c.attrs)})
    return out


# The environment variables a backend reads its API key from. Strata 0.2 requires one on every
# endpoint but /health (`strata launch --access lan`); upstream's image takes it as API_KEY, and
# STRATA_API_KEY is the server's own name for it. llama.cpp and Maya set neither, so for them
# auth_headers() is {} and their requests are exactly what they were.
_KEY_ENV = ("API_KEY", "STRATA_API_KEY")


def api_key_of(attrs: dict | None) -> str:
    """A container's API key, from its own environment (compose env_file lands there). "" if none."""
    vals = container_env(attrs)
    return next((vals[k] for k in _KEY_ENV if vals.get(k)), "")


def auth_headers(name: str) -> dict[str, str]:
    """{"Authorization": "Bearer <key>"} for a backend that has a key, {} for one that does not.

    Every request this app makes to a backend goes through here - the probes, the slot and
    metrics reads, park and load, the benchmark - so a backend that starts requiring a key needs
    nothing but the key in its own environment. Read from the engine map's cache (10 s), so it
    costs no docker call of its own.
    """
    key = _key_map().get(name, "")
    return {"Authorization": f"Bearer {key}"} if key else {}


# The engine map, cached. An engine cannot change without a container being recreated, so this
# is near-static data - but the first version of engine_for() re-enumerated EVERY container over
# the docker socket on every call, and it is called from the GPU strip (once per backend), from
# inference_speed (twice per backend) and from telemetry. The overview polls the strip every
# 500 ms, and _gpu_strip_context is synchronous, so all of that ran ON the event loop.
#
# The symptom was not slowness, it was wrong answers: the loop stalled for over two seconds at a
# time, the async probes in snapshot_llama_backends hit their 2 s CONNECT timeout, and every
# backend reported ConnectTimeout while host curl answered the same endpoints in 0.7 ms. The
# overview showed "nothing loaded" on roughly three polls in four. Worth remembering as a shape:
# a blocked event loop presents as unreachable services, not as a slow page.
_ENGINE_TTL = 10.0
_engine_cache: tuple[float, dict[str, str]] | None = None
_engine_lock = threading.Lock()


_key_cache: dict[str, str] = {}


def _refresh_maps() -> dict[str, str]:
    """One docker enumeration fills both caches: engine per backend, and API key per backend."""
    global _engine_cache, _key_cache
    found = discover_llama_containers()
    fresh = {d["name"]: str(d.get("engine") or "llama") for d in found}
    keys = {d["name"]: d.get("api_key") or "" for d in found}
    with _engine_lock:
        _engine_cache = (time.time(), fresh)
        _key_cache = keys
    return fresh


def engine_map() -> dict[str, str]:
    """{container name: engine} for every discovered backend, cached for _ENGINE_TTL."""
    with _engine_lock:
        if _engine_cache and (time.time() - _engine_cache[0]) < _ENGINE_TTL:
            return _engine_cache[1]
    return _refresh_maps()


def _key_map() -> dict[str, str]:
    """{container name: API key} for every discovered backend, on the engine map's cache."""
    with _engine_lock:
        if _engine_cache and (time.time() - _engine_cache[0]) < _ENGINE_TTL:
            return _key_cache
    _refresh_maps()
    with _engine_lock:
        return _key_cache


def engine_for(name: str) -> str:
    """The engine of one backend by container name; "llama" when nothing says otherwise.

    Defaulting to "llama" rather than "" is deliberate: every caller is llama.cpp machinery
    asking "is this mine?", and a container that has gone missing between the snapshot and
    the question should not silently acquire a new engine. A backend declares its way OUT of
    llama, never into it.
    """
    return engine_map().get(name, "llama")


def _effective_container_names() -> list[str]:
    """Union of LLAMA_CONTAINERS env (retained even if not currently running) and any
    live-discovered llama.cpp containers on the docker socket. Order: env names first, then
    newly-discovered ones. Duplicates removed while preserving order."""
    whitelist = settings.llama_container_names
    discovered = [c["name"] for c in discover_llama_containers()]
    seen: set[str] = set()
    result: list[str] = []
    for n in list(whitelist) + discovered:
        if n and n not in seen:
            seen.add(n)
            result.append(n)
    return result


# What each backend reported as loaded on the last probe, so SYNCHRONOUS callers can ask.
# The router's /v1/models is the only authority on this and reaching it is async; every page
# that draws a VRAM meter also renders the hero, which probes, so this is refreshed in the
# same request rather than being a cache with a life of its own.
_LOADED_BY_BACKEND: dict[str, str] = {}
_LOADED_LOCK = threading.Lock()


# When each backend's current load was FIRST seen, keyed by (backend, model). The model is
# part of the key so that a swap straight into another swap restarts the clock instead of
# inheriting the previous model's start time. Pruned on every observation, so it holds at
# most one entry per backend and never needs a sweep.
_LOADING_SINCE: dict[tuple[str, str], float] = {}


FAILURE_FRESH_S = 1800.0
_FAILED_KEY = "backends.failed_since"


_failed_seen: dict[str, float] | None = None     # loaded from the db once, then kept here


_EXIT_LINE = "exited with status"


def _failure_time_from_log(backend: str, failure: str) -> float | None:
    """When the failure actually happened, from llama.cpp's own log line, or None.

    A failure is first SEEN whenever this app first probes after it - a deploy, a restart - which
    can be days late: llama-voice's gemma-4-31B failure happened 2026-10-15 18:34 and was first
    "seen" five days later. The router logs `instance name=<model> exited with status N` with a
    docker timestamp, so the first sighting is dated from that where it exists. Read once per
    new failure, never on the steady-state path.
    """
    from datetime import datetime
    model = failure.split(" (exit", 1)[0]
    needle = f"instance name={model} {_EXIT_LINE}"
    try:
        client = _docker_client()
        raw = client.containers.get(backend).logs(timestamps=True, tail=20000) if client else b""
    except Exception:  # noqa: BLE001 - no log is "date it from now", not an error
        return None
    for line in reversed(raw.decode("utf-8", "replace").splitlines()):
        if needle in line:
            stamp = line.split(" ", 1)[0]
            try:
                # 2026-10-15T18:34:26.984590009Z - fromisoformat takes microseconds, not nanos
                return datetime.fromisoformat(stamp[:26] + "+00:00").timestamp()
            except ValueError:
                return None
    return None


def _note_failure(backend: str, failure: str | None) -> float:
    """Seconds since this backend's current load failure was first seen; 0.0 when none.

    Persisted in the settings db rather than only in memory: llama.cpp holds a failed flag for
    days, and an in-memory clock would restart at every deploy and call a week-old failure new.
    Read from the db ONCE and written only on change, because this runs on every probe (1.5 s,
    on the event loop) and a blocked loop is what made every backend look unreachable before.
    """
    global _failed_seen
    from .. import db
    import json as _json
    with _LOADED_LOCK:
        if _failed_seen is None:
            try:
                got = _json.loads(db.get_setting(_FAILED_KEY, "") or "{}")
                _failed_seen = {str(k): float(v) for k, v in got.items()} if isinstance(got, dict) else {}
            except Exception:  # noqa: BLE001 - bad JSON or no db yet: start a fresh record
                _failed_seen = {}
        seen = _failed_seen
        key_prefix = backend + "	"
        want = key_prefix + failure if failure else ""
        changed = False
        for k in [k for k in seen if k.startswith(key_prefix) and k != want]:
            del seen[k]
            changed = True
        age = 0.0
        if failure:
            if want not in seen:
                seen[want] = _failure_time_from_log(backend, failure) or time.time()
                changed = True
            age = max(0.0, time.time() - seen[want])
        snapshot = _json.dumps(seen) if changed else None
    if snapshot is not None:
        try:
            db.set_setting(_FAILED_KEY, snapshot)
        except Exception:  # noqa: BLE001 - a lost record costs a mis-aged chip, not a page
            pass
    return age


def _note_loading(backend: str, model: str | None) -> float:
    """Seconds this backend has been loading `model`. 0.0 when it is not loading."""
    with _LOADED_LOCK:
        for k in [k for k in _LOADING_SINCE
                  if k[0] == backend and (model is None or k[1] != model)]:
            del _LOADING_SINCE[k]
        if not model:
            return 0.0
        t0 = _LOADING_SINCE.setdefault((backend, model), time.time())
    return max(0.0, time.time() - t0)


def last_loaded_ids(name: str) -> set[str]:
    """Model ids the last probe saw loaded on this backend. Empty when never probed."""
    with _LOADED_LOCK:
        csv = _LOADED_BY_BACKEND.get(name, "")
    return {p.strip() for p in csv.split(",") if p.strip()}


def _sleep_idle_seconds(attrs: dict) -> int:
    """The container's --sleep-idle-seconds, or 0 when absent or disabled.

    Read off the container's own command line rather than from a setting of ours, because the
    command line is what llama-server is actually running. llama.cpp spells "off" as -1 and as
    the flag simply not being there; both come back 0 here so callers have one thing to test.
    """
    cmd = (attrs.get("Config") or {}).get("Cmd") or []
    if not isinstance(cmd, list):
        return 0
    for i, tok in enumerate(cmd):
        if tok == "--sleep-idle-seconds" and i + 1 < len(cmd):
            try:
                v = int(str(cmd[i + 1]).strip())
            except (TypeError, ValueError):
                return 0
            return v if v > 0 else 0
        if isinstance(tok, str) and tok.startswith("--sleep-idle-seconds="):
            try:
                v = int(tok.split("=", 1)[1].strip())
            except (TypeError, ValueError):
                return 0
            return v if v > 0 else 0
    return 0


# llama-server brackets a sleep with exactly these two, and nothing else reports the state:
# /props and /v1/models read identically asleep or awake, so the log is the only signal.
_RE_SLEEP_MARK = re.compile(r"(entering|exiting) sleeping state")


def _is_asleep(container) -> bool:
    """True when the last sleep marker in the log says it went to sleep and never came back.

    Deliberately NOT probed over HTTP. Waking is what a request does - /slots alone both
    resets the idle timer and wakes a sleeper - so a liveness check would be the thing that
    prevents the sleep it is trying to observe.
    """
    try:
        raw = container.logs(tail=400, stdout=True, stderr=True)
    except Exception:  # noqa: BLE001 - a log we cannot read is not a sleeping model
        return False
    last = ""
    for line in raw.decode("utf-8", errors="replace").splitlines():
        m = _RE_SLEEP_MARK.search(line)
        if m:
            last = m.group(1)
    return last == "entering"


def _snapshot_backends_sync() -> tuple[list[LlamaBackend], list[tuple[int, str, int | None]]]:
    """The blocking half: docker enumeration, inspect and the sleeper log read.

    Split out of snapshot_llama_backends because it is all synchronous docker SDK
    work, and the async caller runs on the event loop - the hero poll hits this
    every 500 ms, and a container's `logs(tail=400)` for the sleeper check is a
    round trip per running backend. The caller runs this in a thread.
    """
    client = _docker_client()
    effective = _effective_container_names()
    if client is None:
        return [LlamaBackend(name=n, found=False, status="docker unreachable")
                for n in effective], []

    out: list[LlamaBackend] = []
    probe_targets: list[tuple[int, str, int | None]] = []  # (idx, name, internal_port)
    _discovered = discover_llama_containers()
    _routers = {d["name"]: bool(d.get("router")) for d in _discovered}
    _vendors = {d["name"]: str(d.get("vendor") or "") for d in _discovered}
    _engines = {d["name"]: str(d.get("engine") or "llama") for d in _discovered}

    for i, name in enumerate(effective):
        b = LlamaBackend(name=name, found=False, status="not_found")
        with _restart_errors_lock:
            b.last_restart_error = _restart_errors.get(name)
        try:
            c = client.containers.get(name)
            b.found = True
            b.status = c.status
            b.image = (c.image.tags or [c.image.short_id])[0]
            b.short_id = c.short_id
            attrs = c.attrs or {}
            state = (attrs.get("State") or {})
            started, up = _parse_started_at(state.get("StartedAt", ""))
            b.started_at = started
            b.uptime = up
            b.host_ports, b.internal_port = _extract_ports(attrs)
            b.sleep_idle_s = _sleep_idle_seconds(attrs)
            b.router = _routers.get(name, True)
            b.vendor = _vendors.get(name, "")
            b.engine = _engines.get(name, "llama")
            if b.sleep_idle_s > 0 and b.status == "running":
                b.asleep = _is_asleep(c)
        except NotFound:
            pass
        except DockerException as e:
            b.status = f"error: {e}"
        out.append(b)
        if b.found and b.status == "running":
            probe_targets.append((i, name, b.internal_port))

    return out, probe_targets


async def snapshot_llama_backends() -> list[LlamaBackend]:
    out, probe_targets = await asyncio.to_thread(_snapshot_backends_sync)

    if probe_targets:
        results = await asyncio.gather(
            *[_probe_loaded_model(n, p) for _, n, p in probe_targets],
            return_exceptions=False,
        )
        for (i, _n, _p), (loaded, err, failed, loading, sleeping, unloaded) in zip(probe_targets, results):
            out[i].loaded_model = loaded
            out[i].probe_error = err
            out[i].load_failed = failed
            # In a thread: a NEW failure reads the backend's log once to date it.
            out[i].load_failed_age = await asyncio.to_thread(_note_failure, out[i].name, failed)
            out[i].loading_model = loading
            out[i].loading_since = _note_loading(out[i].name, loading)
            out[i].sleeping_model = sleeping
            out[i].unloaded_model = unloaded
            # The router's own word beats the log scrape. _is_asleep reads the last sleep marker
            # out of `docker logs`, which is a guess that goes stale the moment the log rotates
            # or the marker scrolls past the tail; this is the server stating its current status.
            if sleeping:
                out[i].asleep = True
            with _LOADED_LOCK:
                _LOADED_BY_BACKEND[out[i].name] = loaded or ""
    return out
