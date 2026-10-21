"""A model's badges (fast, smart, ...): set and clear, answered with the badge box re-rendered."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from .. import db
from ..web import templates

router = APIRouter()


@router.post("/badge", response_class=HTMLResponse)
def badge_assign(request: Request,
                 alias: str = Form(...),
                 category: str = Form(...),
                 rating: int = Form(...),
                 note: str = Form(""),
                 run_id: int = Form(0),
                 collapsible: int = Form(0)) -> HTMLResponse:
    ok, err = db.badge_set(alias, category, rating, note, run_id or None)
    # open=True on the way back: you just used the form, so you may well want it again. Without
    # it the re-rendered partial would come back collapsed after every single change.
    return templates.TemplateResponse("_badges.html", {
        "request": request, "alias": alias, "badges": db.badges_for(alias),
        "run_id": run_id, "badge_err": "" if ok else err,
        "collapsible": bool(collapsible), "open": True,
    })


@router.post("/badge/clear", response_class=HTMLResponse)
def badge_remove(request: Request,
                 alias: str = Form(...),
                 category: str = Form(...),
                 run_id: int = Form(0),
                 collapsible: int = Form(0)) -> HTMLResponse:
    db.badge_clear(alias, category)
    return templates.TemplateResponse("_badges.html", {
        "request": request, "alias": alias, "badges": db.badges_for(alias),
        "run_id": run_id, "badge_err": "",
        "collapsible": bool(collapsible), "open": True,
    })
