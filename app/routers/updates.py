from __future__ import annotations

import time

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from .. import deepseek_engine, services, updates
from ..web import templates

router = APIRouter()


def _ago(ts: float) -> str:
    s = max(0, int(time.time() - ts))
    if s < 90:
        return "just now"
    if s < 5400:
        return f"{s // 60} min ago"
    if s < 172800:
        return f"{s // 3600} h ago"
    return f"{s // 86400} days ago"


def _ctx(request: Request, ds_note: str = "", ds_ok: bool = True) -> dict:
    state = updates.cached()
    return {"request": request, "rows": state.get("engines") or [],
            "checked": _ago(float(state["checked_at"])) if state.get("checked_at") else "",
            "checking": updates.in_progress(),
            "ds": deepseek_engine.status(services._docker_client(), state.get("deepseek_engine_latest") or ""),
            "ds_note": ds_note, "ds_ok": ds_ok}


@router.get("/updates", response_class=HTMLResponse)
def updates_card(request: Request) -> HTMLResponse:
    """The Overview's engine-update card. Starts a background check when the cached one is stale,
    and renders whatever is cached now - the page never waits on GitHub."""
    updates.refresh(services._docker_client())
    return templates.TemplateResponse("_engine_updates.html", _ctx(request))


@router.post("/updates/check", response_class=HTMLResponse)
def updates_check(request: Request) -> HTMLResponse:
    updates.refresh(services._docker_client(), force=True)
    return templates.TemplateResponse("_engine_updates.html", _ctx(request))


@router.post("/updates/deepseek-pin", response_class=HTMLResponse)
def deepseek_pin(request: Request, folder: str = Form("")) -> HTMLResponse:
    """Pin DeepSeek's engine to an installed package, or follow upstream (folder ""). Writes
    deepseek.env only; it applies at DeepSeek's next start, and nothing is restarted here."""
    ok, note = deepseek_engine.set_pin(folder.strip())
    return templates.TemplateResponse("_engine_updates.html", _ctx(request, ds_note=note, ds_ok=ok))
