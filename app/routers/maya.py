from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from .. import maya as maya_engine
from .. import services
from ..web import templates, _blocked_chip, _QUEUED_CHIP

router = APIRouter()

# The port compose publishes Maya's own dashboard on (its Monitor and Chat), for the link. The
# page cannot learn it from inside: from here the container is maya:8080.
DASHBOARD_PORT = 8087


def _maya_state() -> dict:
    client = services._docker_client()
    run_cfg = maya_engine.run_config_name(client) if client else ""
    cfg = maya_engine.read_config(run_cfg) if run_cfg else None
    metrics = services.strata_metrics(maya_engine.CONTAINER)    # Maya serves Strata's /metrics
    try:
        status = client.containers.get(maya_engine.CONTAINER).status if client else ""
    except Exception:  # noqa: BLE001 - no container is a state the page explains, not an error
        status = ""
    return {
        "status": status,
        "health": maya_engine.health(client) if client else None,
        "metrics_engine": (metrics or {}).get("engine") or {},
        "installed": maya_engine.installed(client) if client else [],
        "run_config": run_cfg,
        "writable": maya_engine.CONF_DIR.is_dir(),
        "tunables": maya_engine.TUNABLES,
        "tuning": maya_engine.tuning_state(cfg, metrics) if cfg else {},
        "engine_running": metrics is not None,
        "dashboard_port": DASHBOARD_PORT,
    }


@router.get("/maya", response_class=HTMLResponse)
def maya_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("maya.html", {"request": request, **_maya_state()})


@router.post("/maya/tuning", response_class=HTMLResponse)
def maya_tuning(name: str = Form(...), value: str = Form(""), config: str = Form(...)) -> HTMLResponse:
    ok, msg = maya_engine.set_tunable(name, value, config)
    if not ok:
        return _blocked_chip(msg)
    return HTMLResponse(_QUEUED_CHIP.format(label="Saved", title=msg))
