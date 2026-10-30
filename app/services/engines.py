from __future__ import annotations

import threading
import time

import httpx
from docker.errors import APIError, DockerException, NotFound


from .. import engine_registry
from .dockerbase import _docker_client
from .backends import (_effective_container_names, _extract_ports, _restart_errors, auth_headers,
                     _restart_errors_lock, engine_for)
from .gpu import gpu_engines, gpu_holders
from .stratactl import strata_load, strata_unload


# ---------- engine switching ----------

# Two inference engines cannot share these cards. A llama.cpp router with --models-max 1 holds
# its model until something evicts it, and Strata sizes its expert cache against free VRAM at
# startup, so whichever starts second gets the scraps or dies ("mtp: buffers do not fit").
# Making them exclusive is the only arrangement where either gets the hardware it was set up for.
#
# The switch stops every OTHER BACKEND that was given a GPU, plus any non-backend GPU tenant
# that is actually holding VRAM right now.
#
# The two halves have different tests for a reason. A backend is stopped even when it holds
# nothing, because holding nothing is its idle state and the next request would take the cards
# straight back. A tenant is stopped only when it is measurably in the way, because plenty of
# containers are handed /dev/kfd and allocate nothing (gpu-monitor is one); stopping those frees
# nothing and loses a service.

# What the current switch is doing, for the panel to render. One at a time by construction -
# the lock is held for the whole run, so a second request is rejected rather than interleaved.
_switch_lock = threading.Lock()
_switch_state: dict = {}

def engine_switch_state() -> dict:
    """A snapshot of the running (or last) switch, for the panel. {} when none has run.

    No lock: the worker only ever replaces whole values through dict.update, so a copy taken
    here is a consistent-enough view for a status line, and taking the switch lock to read it
    would block the panel for the whole sixty seconds the switch is allowed to run.
    """
    return dict(_switch_state)


def _vram_by_container() -> dict[str, float]:
    """{container name: GB of VRAM its processes hold right now}.

    From the kernel's own per-process accounting under /sys/class/kfd, which is world-readable
    from any container. This app has no /dev/kfd and cannot run rocm-smi, and the container it
    would otherwise exec into for that is often the one about to be stopped.
    """
    from .. import gpu_procs
    out: dict[str, float] = {}
    try:
        for t in gpu_procs.tenants():
            if t.container:
                out[t.container] = out.get(t.container, 0.0) + t.total_gb
    except Exception:  # noqa: BLE001 - an unreadable topology must not wedge the switch
        return {}
    return out


def _vram_held_gb(exclude: set[str]) -> float:
    """GB of VRAM held by processes belonging to containers NOT in `exclude`.

    Read from the kernel's own per-process accounting under /sys/class/kfd, which is
    world-readable from any container - this app has no /dev/kfd and cannot run rocm-smi, and
    the container it would normally exec into for that is the one being stopped.
    """
    from .. import gpu_procs
    try:
        return sum(t.total_gb for t in gpu_procs.tenants()
                   if (t.container or "") not in exclude and t.total_gb > 0.005)
    except Exception:  # noqa: BLE001 - an unreadable topology must not wedge the switch
        return 0.0


# How long each engine gets to answer after `docker start` (engine_registry; Maya's is the long
# one). router_api._NARRATE_MAX_S is sized past the longest.
READY_TIMEOUT_S = {n: e.ready_timeout_s for n, e in engine_registry.ENGINES.items()}
_READY_TIMEOUT_DEFAULT_S = engine_registry.DEFAULT_READY_TIMEOUT_S


def _wait_ready(name: str, timeout: float | None = None, quiet: bool = False) -> tuple[bool, str]:
    """Poll one backend until it is actually serving. (ready, note). `quiet` leaves the switch's
    status alone (the slow-start watcher must not turn "slow" back into "loading").

    `docker start` returning is not "serving", and the gap is not small: Strata reads ~55 GB of
    experts into RAM before its port opens, and a llama router takes 10-40 s depending on the
    model. What "serving" means per engine (which path, what a 200 must say) is
    engine_registry.serving().
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
        return True, f"{name} exposes no API port — not waiting"
    engine = engine_for(name)
    if timeout is None:
        timeout = READY_TIMEOUT_S.get(engine, _READY_TIMEOUT_DEFAULT_S)
    strata = engine_registry.speaks_strata(engine)
    spec = engine_registry.get(engine)
    url = f"http://{name}:{port}{spec.ready_path if spec else '/v1/models'}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=httpx.Timeout(3.0)) as c:
                r = c.get(url, headers=auth_headers(name))
                if engine_registry.serving(engine, r.status_code,
                                           r.json() if r.status_code == 200 and spec and spec.single_model
                                           else None):
                    return True, ""
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        if quiet:
            time.sleep(2.0)
            continue
        _switch_state.update(phase="loading",
                             msg=f"{name} up {int(time.time() - _switch_state.get('t0', time.time()))}s, "
                                 + ("model not loaded yet" if strata
                                    else "loading the model" if engine == "maya"
                                    else "port not answering yet"))
        time.sleep(2.0)
    return False, f"{name} started but {url} did not answer within {int(timeout)}s"


# How long a "slow" start is still watched for, after the switch itself has given up waiting.
_SLOW_WATCH_S = 3600.0


def _watch_slow_start(name: str, t0) -> None:
    """Keep watching a start the switch called slow, and close it out as done when it answers.

    "Slow" ended the switch's wait but not the start: DeepSeek re-tuning after an engine update
    took 15 min (2026-10-30), served, and the banner said "still loading" for good. This runs
    after the switch lock is released, so another switch is never blocked by it, and it touches
    the state only while that state is still THIS switch (same t0, still slow).
    """
    def _run() -> None:
        deadline = time.time() + _SLOW_WATCH_S
        while time.time() < deadline:
            if _switch_state.get("t0") != t0 or _switch_state.get("phase") != "slow":
                return
            ready, _ = _wait_ready(name, timeout=30.0, quiet=True)
            if ready:
                if _switch_state.get("t0") == t0 and _switch_state.get("phase") == "slow":
                    _switch_state.update(phase="done", msg="", t1=time.time())
                return
    threading.Thread(target=_run, daemon=True, name=f"slow-start-{name}").start()


def activate_engine(engine: str, soft: bool = False, share: bool = False) -> tuple[bool, str]:
    """Make `engine` the only thing holding the cards. Fire-and-forget; progress in _switch_state.

    Returns immediately because the whole sequence takes tens of seconds - stopping a router
    with 30 GB of weights, waiting for the driver to actually release it, then a start that
    reads ~55 GB of experts into RAM. A request that blocked on all that would time out in the
    browser long before it finished, and the GPU strip is already polling, so the panel shows
    the phases on its own.

    `soft` asks each displaced backend that supports it to UNLOAD instead of stopping: today
    that is Strata alone (a llama router at --models-max 1 has no unload endpoint, which is
    the whole reason this switch exists). Unload gives back the same VRAM and RAM in ~0.3 s, so
    the RELEASE is immediate and the container keeps its settings.

    It does not make the way back quick: a parked Strata asked to load again takes about as long
    as a cold container start, because the other engine reads its own model in between and
    evicts the page cache. A parked-but-running server can also be woken by a stray request, so
    hard remains the default and soft is what the checkbox asks for.

    `share` displaces nothing: it starts the engine's backends and leaves every other holder
    exactly where it is. That is only honest when the cards are actually dividable - the
    caller decides when to offer it (a running Strata pinned to one card leaves the other
    for llama) - and the fit table has already been sized against the remaining cards, so
    the start is not a gamble.
    """
    engines = {e["engine"]: e for e in gpu_engines()}
    if engine not in engines:
        return False, f"no GPU backend of engine '{engine}'"
    if not _switch_lock.acquire(blocking=False):
        cur = _switch_state.get("engine") or "another engine"
        return False, f"already switching to {cur}"

    targets = set(engines[engine]["names"])
    live = _vram_by_container()
    holders = {r["name"]: r for r in gpu_holders()}
    others = []
    skipped = []
    shared = []
    for r in holders.values():
        if r["name"] in targets or not r["running"]:
            continue
        if share:
            shared.append(r["name"])
            continue
        if r["backend"] or live.get(r["name"], 0.0) > 0.005:
            others.append(r)
        else:
            skipped.append(r["name"])
    # A target that is already running is a PARKED Strata being asked back: docker start is a
    # no-op there and /load is the ask. (A running llama target needs nothing; _wait_ready
    # confirms it either way.)
    running_targets = {n for n in targets if holders.get(n, {}).get("running")}
    plan_unload = [r for r in others if soft and engine_registry.speaks_strata(r["engine"]) and r["running"]]
    plan_stop = [r for r in others if r not in plan_unload]
    _switch_state.clear()
    _switch_state.update(engine=engine, phase="stopping", msg="", t0=time.time(),
                         stopped=[r["name"] for r in plan_stop],
                         unloaded=[r["name"] for r in plan_unload],
                         shared=shared, skipped=skipped, started=[])

    def _run() -> None:
        try:
            from .. import gpu_procs
            for r in plan_unload:
                # An unload waits for an answer in flight, and an answer can run minutes (a
                # DeepSeek reply of 5,740 tokens held a switch for ~3 min on 2026-10-30 while the
                # page said only "stopping"). Say so, so a wait reads as a wait.
                _switch_state.update(phase="unloading",
                                     msg=f"{r['name']} unloads once any answer in progress finishes")
                ok, msg = strata_unload(r["name"])
                if not ok:
                    _switch_state.update(phase="failed",
                                         msg=f"could not unload {r['name']}: {msg}")
                    return
            for r in plan_stop:
                _switch_state.update(phase="stopping", msg=f"stopping {r['name']}")
                ok, msg = gpu_procs.set_container_running(r["name"], start=False)
                if not ok:
                    _switch_state.update(phase="failed", msg=f"could not stop {r['name']}: {msg}")
                    return

            # Wait for the DRIVER to release it, not merely for the process to exit. Strata
            # sizes its expert cache against free VRAM at startup and llama.cpp's own fitter
            # does the same, and the drop is not instant after the process is gone, so starting
            # the moment docker returns reads a stale figure and gets a fraction of the card.
            #
            # Only when something was actually released. The drain asks "has the VRAM the
            # switch is reclaiming come back", and share reclaims nothing - the other engine's
            # 30+ GB is the point of the mode, so waiting for it would wait sixty seconds and
            # then fail.
            if not (plan_unload or plan_stop):
                held = 0.0
            else:
                deadline = time.time() + 60
                while time.time() < deadline:
                    held = _vram_held_gb(targets)
                    if held < 0.5:
                        break
                    _switch_state.update(phase="draining", msg=f"{held:.1f} GB still held")
                    time.sleep(1.0)
                else:
                    _switch_state.update(
                        phase="failed",
                        msg=f"{_vram_held_gb(targets):.1f} GB still on the cards after 60 s — "
                            "something outside docker is holding them")
                    return

            _switch_state.update(phase="starting", msg=f"starting {', '.join(sorted(targets))}")
            for name in sorted(targets):
                if name in running_targets:
                    # Up but parked (the only way a target is both this engine's and already
                    # running): /load is the ask, and _wait_ready below - which polls /health
                    # on Strata - is what proves it.
                    if engine_registry.speaks_strata(engine_for(name)):
                        ok, msg = strata_load(name)
                        if not ok:
                            _switch_state.update(phase="failed", msg=f"could not load {name}: {msg}")
                            return
                    _switch_state["started"].append(name)
                    continue
                ok, msg = gpu_procs.set_container_running(name, start=True)
                if not ok:
                    _switch_state.update(phase="failed", msg=f"could not start {name}: {msg}")
                    return
                _switch_state["started"].append(name)
            # Not done until it ANSWERS. See _wait_ready: `docker start` returning is 90 s short
            # of serving on Strata, and reporting done there is what made a working switch look
            # like a broken button.
            for name in sorted(targets):
                ready, note = _wait_ready(name)
                if not ready:
                    # "slow", not "failed": the container is up and may still come good. Saying
                    # it failed would send someone looking for a crash that has not happened.
                    _switch_state.update(phase="slow", msg=note)
                    _watch_slow_start(name, _switch_state.get("t0"))
                    return
            _switch_state.update(phase="done", msg="", t1=time.time())
        except Exception as e:  # noqa: BLE001 - a thread that dies silently leaves the panel lying
            _switch_state.update(phase="failed", msg=f"{type(e).__name__}: {e}")
        finally:
            _switch_lock.release()

    threading.Thread(target=_run, daemon=True).start()
    giving = " and ".join(x for x in (
        ("unloading " + ", ".join(r["name"] for r in plan_unload)) if plan_unload else "",
        ("stopping " + ", ".join(r["name"] for r in plan_stop)) if plan_stop else "",
    ) if x) or "nothing"
    note = f" (left {', '.join(skipped)} alone — holding no VRAM)" if skipped else ""
    if share:
        with_ = f", leaving {', '.join(sorted(shared))} up" if shared else ""
        return True, f"sharing the cards: starting {', '.join(sorted(targets))}{with_}"
    return True, (f"switching to {engine}: {giving}, then starting "
                  f"{', '.join(sorted(targets))}{note}")


def set_backend_running(name: str, start: bool) -> tuple[bool, str]:
    """Start or stop one backend. The control the Containers page was missing.

    Restart was the only lifecycle button a backend had, which is fine until two engines have
    to take turns on the same cards: freeing them meant an ssh session, and a control that
    needs a terminal is not a control. Stopping is also the ONLY way to make a llama router
    give up its model - at --models-max 1 it holds it until a request for another one arrives,
    and there is no unload endpoint on this build.
    """
    if name not in _effective_container_names():
        return False, f"{name} is not a configured backend"
    from .. import gpu_procs
    return gpu_procs.set_container_running(name, start=start)


def restart_llama_backend(name: str) -> tuple[bool, str]:
    """Fire-and-forget restart. Errors captured in _restart_errors."""
    if name not in _effective_container_names():
        return False, "container not in configured list"
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    try:
        c = client.containers.get(name)
    except NotFound:
        return False, "container not found"
    except DockerException as e:
        return False, f"docker error: {e}"

    with _restart_errors_lock:
        _restart_errors.pop(name, None)

    def _do() -> None:
        try:
            c.restart(timeout=15)
        except (DockerException, APIError) as e:
            with _restart_errors_lock:
                _restart_errors[name] = f"{type(e).__name__}: {e}"

    threading.Thread(target=_do, daemon=True, name=f"restart-{name}").start()
    return True, "restart queued"
