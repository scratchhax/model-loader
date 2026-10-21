"""Cmd+K: the command palette's list of everything it can jump to or trigger."""

from __future__ import annotations

from fastapi import APIRouter

from .. import db, ini, services

router = APIRouter()


@router.get("/palette.json")
def palette() -> dict:
    """Everything Cmd+K can jump to or trigger."""
    snap = services.snapshot_models_dir()
    items: list[dict] = []
    for g in snap.ggufs:
        items.append({
            "kind": "model",
            "title": g.display_name,
            "hint": g.human_size,
            "url": f"/model/{g.display_name}",
            "icon": "layers-3",
        })
    for s in ini.list_sections():
        items.append({
            "kind": "section",
            "title": s.name,
            "hint": f"{len(s.items)} opts",
            "url": f"/config/section/{s.name}/edit",
            "icon": "file-cog",
        })
    for name in services._effective_container_names():
        items.append({
            "kind": "container",
            "title": name,
            "hint": "view details",
            "url": "/containers",
            "icon": "server",
        })
        items.append({
            "kind": "action",
            "title": f"Restart {name}",
            "hint": "container",
            "action": "post",
            "url": f"/containers/{name}/restart",
            "icon": "rotate-cw",
        })
    for p in db.list_prompts():
        items.append({
            "kind": "prompt",
            "title": p["name"],
            "hint": "saved prompt",
            "url": "/prompts",
            "icon": "message-square-quote",
        })
    # global actions
    items += [
        {"kind": "action", "title": "Check for HF updates", "hint": "compare local vs HF", "action": "post", "url": "/models/check-updates", "icon": "refresh-cw"},
        {"kind": "action", "title": "Clear finished downloads", "hint": "history clear", "action": "post", "url": "/downloads/clear", "icon": "check"},
        {"kind": "page", "title": "Search Hugging Face", "url": "/search", "icon": "search"},
        {"kind": "page", "title": "Downloads", "url": "/downloads", "icon": "download"},
        {"kind": "page", "title": "models.ini editor", "url": "/config", "icon": "file-cog"},
        {"kind": "page", "title": "Containers", "url": "/containers", "icon": "server"},
        {"kind": "page", "title": "Prompts", "url": "/prompts", "icon": "message-square-quote"},
        {"kind": "page", "title": "Settings", "url": "/settings", "icon": "settings"},
        {"kind": "page", "title": "Overview / dashboard", "url": "/", "icon": "gauge"},
    ]
    return {"items": items}
