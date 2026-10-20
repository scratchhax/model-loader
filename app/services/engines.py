from __future__ import annotations

import threading
import time

import httpx
from docker.errors import APIError, DockerException, NotFound


from .dockerbase import _docker_client
from .backends import (_effective_container_names, _extract_ports, _restart_errors, auth_headers,
                     _restart_errors_lock, engine_for)
from .gpu import gpu_engines, gpu_holders
from .stratactl import strata_load, strata_unload


# ---------- engine switching ----------

# Two inference engines cannot share these cards. A llama.cpp router with --models-max 1 holds
# its model until something evicts it, and Strata sizes its expert cache against free VRAM at
# startup, so whichever starts second gets the scraps - or, as measured on this box, dies on
# "mtp: buffers do not fit (0 MiB of 32624 MiB VRAM free on this GPU)". Making them exclusive
# is the only arrangement where either one gets the hardware it was configured for.
#
# The switch stops every OTHER BACKEND that was given a GPU, plus any non-backend GPU tenant
# that is actually holding VRAM right now.
#
# The two halves have different tests for a reason. A backend is stopped even when it holds
# nothing, because holding nothing is its idle state and the next request would take the cards
# straight back. A tenant is stopped only when it is measurably in the way, because plenty of
# containers are handed /dev/kfd and allocate nothing - gpu-monitor is one, and the first
# version of this stopped it, SIGKILLed it when its shutdown ran past the grace period, and
# left the box without its monitoring sidecar to free zero bytes.

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


# How long each engine gets to answer after `docker start`. Maya's is the long one: measured
# 2026-10-20, a start from a cold page cache took ~5 min (GLM-5.3-Flash Maya-S warms 43 + 46 GB
# of expert tiers before its port opens, one card at 0.17 GB/s), so 420 s would report a healthy
# start as slow with a minute to spare at best. router_api._NARRATE_MAX_S is sized past this.
READY_TIMEOUT_S = {"maya": 900.0}
_READY_TIMEOUT_DEFAULT_S = 420.0


def _wait_ready(name: str, timeout: float | None = None) -> tuple[bool, str]:
    """Poll one backend until it is actually serving. (ready, note).

    `docker start` returning is not "serving", and the gap is not small: Strata reads ~55 GB of
    experts into RAM before its port opens - measured at 90 s here - and a llama router is 10-40 s
    depending on the model. Reporting the switch done at `docker start` meant clicking the button,
    seeing "done", finding nothing on the port and concluding it had not worked. That is exactly
    what happened.

    The question is per-engine, because "200" means different things. On llama it is /v1/models:
    llama.cpp's /health returns 503 "Loading model" while loading (and curl does not fail on a
    503, so a naive poll measures a model mid-load), while /v1/models only answers once the
    router is up. On Strata it is /health, because /v1/models answers 200 with the model listed
    as "unloaded" - a 200 there can mean "up and serving nothing", which is precisely the state a
    POST /load passes through on its way to loaded. /health carries the one bool that settles it
    ({"loaded": true}), and during a cold start the port is closed until the model is in, so the
    same poll covers cold start and warm /load alike. On Maya it is /health with status "ok":
    there is no load/unload pair, the port opens only once the model is in, and the statuses it
    reports otherwise ("updating", "reloading") mean up but not taking requests.
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
    strata = engine == "strata"
    health = engine in ("strata", "maya")
    url = f"http://{name}:{port}/{'health' if health else 'v1/models'}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=httpx.Timeout(3.0)) as c:
                r = c.get(url, headers=auth_headers(name))
                if r.status_code == 200:
                    if strata:
                        if bool(r.json().get("loaded")):
                            return True, ""
                    elif engine == "maya":
                        if str(r.json().get("status") or "") == "ok":
                            return True, ""
                    else:
                        return True, ""
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        _switch_state.update(phase="loading",
                             msg=f"{name} up {int(time.time() - _switch_state.get('t0', time.time()))}s, "
                                 + ("model not loaded yet" if strata
                                    else "loading the model" if engine == "maya"
                                    else "port not answering yet"))
        time.sleep(2.0)
    return False, f"{name} started but {url} did not answer within {int(timeout)}s"


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

    It does not make the way back quick. Measured 2026-10-15: a parked Strata asked to load
    again took 104 s, against ~90 s for a cold container start. This docstring used to claim
    "the way back costs seconds instead of a 90 s cold start", which was inferred from the page
    cache staying warm and never measured - and the inference fails in exactly the case it was
    written for, since the other engine reads its own 25 GB model in between and evicts that
    cache. The trade is that a parked-but-running server can be woken by a stray request, so
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
    plan_unload = [r for r in others if soft and r["engine"] == "strata" and r["running"]]
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
                ok, msg = strata_unload(r["name"])
                if not ok:
                    _switch_state.update(phase="failed",
                                         msg=f"could not unload {r['name']}: {msg}")
                    return
            for r in plan_stop:
                ok, msg = gpu_procs.set_container_running(r["name"], start=False)
                if not ok:
                    _switch_state.update(phase="failed", msg=f"could not stop {r['name']}: {msg}")
                    return

            # Wait for the DRIVER to release it, not merely for the process to exit. Strata
            # sizes its expert cache against free VRAM at startup and llama.cpp's own fitter
            # does the same; starting the moment docker returns means reading a stale figure
            # and silently getting a fraction of the card. Measured on this box, the drop is
            # not instant after the process is gone.
            #
            # Only when something was actually released. The drain asks "has the VRAM the
            # switch is reclaiming come back", and share reclaims nothing - the other
            # engine's 30+ GB is the point of the mode, so waiting for it to drop waits
            # sixty seconds and then fails with "something outside docker is holding them".
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

            _switch_state.update(phase="starting", msg="")
            for name in sorted(targets):
                if name in running_targets:
                    # Up but parked (the only way a target is both this engine's and already
                    # running): /load is the ask, and _wait_ready below - which polls /health
                    # on Strata - is what proves it.
                    if engine_for(name) == "strata":
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
