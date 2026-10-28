from __future__ import annotations

import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import db, gpu_activity, hw, services
from . import strata as strata_engine
from . import router_api
from .config import settings
from .routers import badges as badges_router
from .routers import benchmark as benchmark_router
from .routers import config as config_router
from .routers import containers as containers_router
from .routers import downloads as downloads_router
from .routers import models as models_router
from .routers import overview as overview_router
from .routers import palette as palette_router
from .routers import power as power_router
from .routers import prompts as prompts_router
from .routers import router_tab as router_tab_router
from .routers import search as search_router
from .routers import settings as settings_router
from .routers import strata as strata_router
from .routers import maya as maya_router
from .routers import updates as updates_router

app = FastAPI(title="Model Loader")


# Paths a read-only instance may still write to: its own router settings, which live in its own
# database and drive nothing on the host.
_READONLY_ALLOW = ("/router/",)


@app.middleware("http")
async def _readonly_guard(request: Request, call_next):
    """Refuse every mutating request when this instance is not the one in charge.

    Model Loader owns real lifecycle: it starts and stops engines, writes models.ini, writes the
    Strata pin file and drives the engine switch. Running a second instance for development is
    only safe if it cannot do any of that - otherwise two schedulers fight over the same two
    cards, and the symptom is a container that will not start, which is exactly the failure this
    setup keeps producing. The guard is here rather than in compose because a read-only VOLUME
    does not stop a container restart through the docker socket.
    """
    if (settings.router_readonly
            and request.method in ("POST", "PUT", "PATCH", "DELETE")
            and not request.url.path.startswith(_READONLY_ALLOW)):
        return JSONResponse(
            status_code=403,
            content={"detail": "read-only instance: the live Model Loader owns this action"},
        )
    return await call_next(request)


_UNSAFE_METHODS = ("POST", "PUT", "PATCH", "DELETE")


def cross_site(headers) -> str:
    """Why a mutating request is refused as coming from another site, or "".

    This app holds the docker socket, so a form post is as good as root on the host, and with no
    check any page open in a browser on the LAN could make one (a cross-site form POST needs no
    script and no CORS). Browsers name the page a request came from in Origin, and in Referer
    when they leave Origin out; either one naming a different host is refused, and so is
    Origin: null (a sandboxed frame or a file:// page). A request with neither is a script or
    the router process asking for a switch, and is let through: this is not authentication.
    """
    from urllib.parse import urlsplit
    src = headers.get("origin") or headers.get("referer") or ""
    if not src:
        return ""
    if src == "null":
        return "a page with no origin"
    host = urlsplit(src).netloc.lower()
    mine = {h.lower() for h in (headers.get("host"), headers.get("x-forwarded-host")) if h}
    return "" if host in mine else f"a page on {host or src}"


@app.middleware("http")
async def _same_site_only(request: Request, call_next):
    if request.method in _UNSAFE_METHODS:
        why = cross_site(request.headers)
        if why:
            return JSONResponse(status_code=403,
                                content={"detail": f"refused: this request came from {why}, "
                                                   f"not from Model Loader"})
    return await call_next(request)


@app.on_event("startup")
def _startup() -> None:
    db.init()
    # BOTH router tables, here as well as in the router process. CREATE TABLE IF NOT EXISTS
    # makes the overlap safe, and it is not redundant: the deployed container runs a single
    # uvicorn (app.main), so this is the only startup that runs there. The Router tab reads the
    # rollup through totals(), series() and retention_state(), and creating only the request
    # table left all three raising "no such table: router_rollup" on a fresh database - it
    # worked in the two-process dev container and nowhere else.
    router_api._init_db()
    router_api._init_rollup()
    db.seed_bench_prompts()
    # Downloads a deploy or restart interrupted; a single-stream temp resumes where it stopped.
    downloads_router.manager.resume_pending()
    hw.start_sampler()
    gpu_activity.start()
    threading.Thread(target=_reconcile_strata_config, daemon=True, name="strata-config").start()


def _reconcile_strata_config() -> None:
    """Once at startup, ensure the coexistence keys are in every Strata backend's run config.

    Startup rather than every poll: the engine reads the config at start, so a key that
    appeared since then only matters for the next start anyway - and the card's note says
    exactly that meanwhile. A thread because the docker calls must not hold up the first
    request, and silent because the card reports the delta either way.
    """
    keys = services.strata_desired_keys()
    if not keys:
        return
    try:
        client = services._docker_client()
        if client is None:
            return
        own = services.strata_own_attrs(client)
        for c in client.containers.list(all=True, filters={"label": f"ai-lab.engine={strata_engine.ENGINE}"}):
            path = strata_engine.run_config_path(c.attrs, own)
            if path is not None:
                strata_engine.merge_run_config(path, keys)
    except Exception:  # noqa: BLE001 - no docker socket is a normal state for a dev checkout
        pass


app.include_router(palette_router.router)


# ---------- Overview ----------

app.include_router(overview_router.router)


# ---------- Power and thermals ----------

app.include_router(power_router.router)


# ---------- Models ----------

app.include_router(models_router.router)


# ---------- Settings ----------

app.include_router(settings_router.router)


# ---------- Search ----------

app.include_router(search_router.router)


# ---------- Downloads ----------

app.include_router(downloads_router.router)


# ---------- models.ini ----------

app.include_router(config_router.router)


# ---------- Containers ----------

app.include_router(containers_router.router)


# ---------- Maya ----------

app.include_router(maya_router.router)


# ---------- Engine updates ----------

app.include_router(updates_router.router)


# ---------- Prompt library ----------

app.include_router(prompts_router.router)


# ---------------------------------------------------------------- benchmark

app.include_router(benchmark_router.router)


app.include_router(badges_router.router)


# ===================================================================== Strata models

app.include_router(strata_router.router)


# ---------- Router ----------

app.include_router(router_tab_router.router)
