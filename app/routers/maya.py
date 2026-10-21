from __future__ import annotations

import asyncio

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from .. import engine_controls
from .. import maya as maya_engine
from .. import services
from ..web import templates, _form_note

router = APIRouter()

def host_port(attrs: dict | None, container_port: str = "8080/tcp") -> str:
    """The host port compose publishes a container port on, for a link a browser can follow.

    From inside this app Maya is maya:8080, which a browser cannot reach; docker inspect has the
    binding (NetworkSettings while running, HostConfig.PortBindings either way). "" if unpublished.
    """
    attrs = attrs or {}
    for ports in ((attrs.get("NetworkSettings") or {}).get("Ports") or {},
                  (attrs.get("HostConfig") or {}).get("PortBindings") or {}):
        for b in ports.get(container_port) or []:
            if (b or {}).get("HostPort"):
                return str(b["HostPort"])
    return ""


def _maya_state() -> dict:
    client = services._docker_client()
    run_cfg = maya_engine.run_config_name(client) if client else ""
    cfg = maya_engine.read_config(run_cfg) if run_cfg else None
    metrics = services.strata_metrics(maya_engine.CONTAINER)    # Maya serves Strata's /metrics
    try:
        c = client.containers.get(maya_engine.CONTAINER) if client else None
        status, attrs = (c.status, c.attrs) if c else ("", {})
    except Exception:  # noqa: BLE001 - no container is a state the page explains, not an error
        status, attrs = "", {}
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
        "dashboard_port": host_port(attrs),
    }


@router.get("/maya", response_class=HTMLResponse)
async def maya_page(request: Request) -> HTMLResponse:
    state = await asyncio.to_thread(_maya_state)
    return templates.TemplateResponse("maya.html", {"request": request, **state,
                                                    **await engine_controls.rows_context_async()})


@router.post("/maya/tuning", response_class=HTMLResponse)
def maya_tuning(name: str = Form(...), value: str = Form(""), config: str = Form(...)) -> HTMLResponse:
    ok, msg = maya_engine.set_tunable(name, value, config)
    return _form_note(ok, msg, label="saved, applies on next start")
