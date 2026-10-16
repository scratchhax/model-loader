from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from .. import db, hf
from ..web import templates

router = APIRouter()


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("settings.html", {"request": request, "token": db.get_setting("hf_token", ""), "flash": None})


@router.post("/settings", response_class=HTMLResponse)
async def settings_save(request: Request, hf_token: str = Form(""), action: str = Form("save")) -> HTMLResponse:
    hf_token = hf_token.strip()
    db.set_setting("hf_token", hf_token)
    flash = {"ok": True, "msg": "Saved."}
    if action == "test":
        try:
            ok, msg = await hf.validate_token(hf_token)
            flash = {"ok": ok, "msg": f"Saved. {msg}"}
        except Exception as e:  # noqa: BLE001
            flash = {"ok": False, "msg": f"Saved, but test failed: {e}"}
    return templates.TemplateResponse("settings.html", {"request": request, "token": hf_token, "flash": flash})
