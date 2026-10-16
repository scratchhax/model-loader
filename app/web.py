"""Shared web plumbing: the Jinja environment every route and router renders through.

Lives apart from main.py so router modules can import it without importing the app back.
"""

from __future__ import annotations

import html
from pathlib import Path

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import db
from . import strata_models

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.globals["hue"] = lambda s: sum(ord(c) for c in (s or "")) % 360
templates.env.globals["strata_only"] = strata_models.strata_only
templates.env.globals["badge_categories"] = db.BADGE_CATEGORIES
templates.env.globals["badge_labels"] = db.BADGE_LABELS


def _fmt_ts(v) -> str:
    """Epoch seconds -> local "2026-09-09 01:23". Empty for anything unparseable, so a
    missing timestamp renders as nothing rather than as 1970."""
    from datetime import datetime
    try:
        return datetime.fromtimestamp(float(v)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return ""


templates.env.filters["ts"] = _fmt_ts


def _blocked_chip(reason: str) -> HTMLResponse:
    """What a refused download button turns into. The reason rides in the tooltip."""
    return HTMLResponse(
        '<span class="shrink-0 inline-flex items-center rounded-md bg-red-100 dark:bg-red-950 '
        'text-red-800 dark:text-red-300 px-3 py-1.5 text-xs font-medium" '
        f'title="{html.escape(reason)}">Blocked: llama.cpp can&rsquo;t load this</span>')
