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
from .engine_colors import engine_tint

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.globals["hue"] = lambda s: sum(ord(c) for c in (s or "")) % 360
templates.env.globals["strata_only"] = strata_models.strata_only
templates.env.globals["badge_categories"] = db.BADGE_CATEGORIES
templates.env.globals["badge_labels"] = db.BADGE_LABELS
templates.env.globals["engine_tint"] = engine_tint


def _fmt_ts(v) -> str:
    """Epoch seconds -> local "2026-09-09 01:23". Empty for anything unparseable, so a
    missing timestamp renders as nothing rather than as 1970."""
    from datetime import datetime
    try:
        return datetime.fromtimestamp(float(v)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return ""


templates.env.filters["ts"] = _fmt_ts


def _blocked_chip(reason: str, label: str = "Blocked: llama.cpp can&rsquo;t load this") -> HTMLResponse:
    """What a refused action button turns into. The reason rides in the tooltip.

    The default label is the download refusal it was written for; anything else passes its own,
    because a Strata install that fails is not llama.cpp refusing a file.
    """
    return HTMLResponse(
        '<span class="shrink-0 inline-flex items-center rounded-md bg-red-100 dark:bg-red-950 '
        'text-red-800 dark:text-red-300 px-3 py-1.5 text-xs font-medium" '
        f'title="{html.escape(reason)}">{label}</span>')


def _form_note(ok: bool, msg: str, label: str = "saved") -> HTMLResponse:
    """The result of a settings form, swapped into the form's own `.form-note` span.

    Settings forms used to swap themselves for a "Saved" chip, so the control was gone until a
    reload. They now target `find .form-note` instead: the select keeps the value just saved,
    and this says whether it took. A failure shows its reason, not just a tooltip.
    """
    if ok:
        return HTMLResponse(f'<span class="text-[11px] text-emerald-700 dark:text-emerald-400" '
                            f'title="{html.escape(msg)}">&check; {html.escape(label)}</span>')
    return HTMLResponse(f'<span class="inline-block max-w-[18rem] truncate align-middle text-[11px] '
                        f'text-red-700 dark:text-red-400" title="{html.escape(msg)}">'
                        f'Failed: {html.escape(msg)}</span>')


def _toast(ok: bool, msg: str) -> HTMLResponse:
    """The banner the models page swaps into #models-owui-toast. Shared so three routes that
    report into the same strip cannot drift apart in markup."""
    cls = ("bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300" if ok
           else "bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300")
    return HTMLResponse(f'<div class="rounded-md {cls} px-3 py-2 text-sm">'
                        f'{"&check;" if ok else "Failed:"} {html.escape(msg)}</div>')




_QUEUED_CHIP = (
    '<a href="/downloads" title="{title}" '
    'class="shrink-0 inline-flex items-center gap-1 rounded-md bg-emerald-100 dark:bg-emerald-950 '
    'text-emerald-800 dark:text-emerald-300 px-3 py-1.5 text-xs font-medium">'
    '✓ {label}</a>'
)
