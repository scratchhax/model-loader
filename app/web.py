"""Shared web plumbing: the Jinja environment every route and router renders through.

Lives apart from main.py so router modules can import it without importing the app back.
"""

from __future__ import annotations

import html
from pathlib import Path

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import db, engine_registry
from . import strata_models
from .engine_colors import engine_tint
from .gpu_activity import curve

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.globals["hue"] = lambda s: sum(ord(c) for c in (s or "")) % 360
templates.env.globals["strata_only"] = strata_models.strata_only
templates.env.globals["badge_categories"] = db.BADGE_CATEGORIES
templates.env.globals["badge_labels"] = db.BADGE_LABELS
templates.env.globals["engine_tint"] = engine_tint
templates.env.globals["curve"] = curve

# The HEAT ramp: how hard something works against its ceiling (GPU compute, power, heat, load,
# RAM, disk), slate when idle through indigo and violet to orange at the limit. Kept apart from
# the tenant colours, which say WHO is on a card and never HOW MUCH. _live.html draws the charts
# from these same stops, so a ring and the line beside it agree.
HEAT_STOPS = ((0.0, (71, 85, 105)), (0.35, (99, 102, 241)), (0.65, (192, 132, 252)), (1.0, (251, 146, 60)))


def heat(frac) -> str:
    """A fraction of a ceiling (clamped to 0..1) as an rgb() colour on the heat ramp."""
    try:
        f = min(1.0, max(0.0, float(frac or 0.0)))
    except (TypeError, ValueError):
        f = 0.0
    for (f0, c0), (f1, c1) in zip(HEAT_STOPS, HEAT_STOPS[1:]):
        if f <= f1:
            t = (f - f0) / (f1 - f0)
            return "rgb(%d,%d,%d)" % tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))
    return "rgb(%d,%d,%d)" % HEAT_STOPS[-1][1]


templates.env.globals["heat"] = heat
templates.env.globals["heat_stops"] = [[f, list(c)] for f, c in HEAT_STOPS]
# Engines that are Strata's server (Load / Unload, the soft handover): the Qwen install and DeepSeek.
templates.env.globals["strata_protocol"] = engine_registry.STRATA_PROTOCOL
# Engines that pin most of the RAM, so a take-over by them stops a Strata rather than parking it.
templates.env.globals["needs_ram"] = tuple(n for n, e in engine_registry.ENGINES.items() if e.needs_ram)
templates.env.globals["engine_label"] = lambda n:(engine_registry.get(n).label if engine_registry.get(n) else (n or ""))


def _fmt_ts(v) -> str:
    """Epoch seconds -> local "2026-09-09 01:23". Empty for anything unparseable, so a
    missing timestamp renders as nothing rather than as 1970."""
    from datetime import datetime
    try:
        return datetime.fromtimestamp(float(v)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return ""


templates.env.filters["ts"] = _fmt_ts


def _fmt_wh(wh) -> str:
    from .energy import fmt_wh
    return fmt_wh(wh)


templates.env.filters["wh"] = _fmt_wh


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

    Settings forms target `find .form-note` rather than replacing themselves, so the control
    stays and the select keeps the value just saved; this says whether it took. A failure shows
    its reason, not just a tooltip.
    """
    if ok:
        return HTMLResponse(f'<span class="text-[11px] text-emerald-700 dark:text-emerald-400" '
                            f'title="{html.escape(msg)}">&check; {html.escape(label)}</span>')
    return HTMLResponse(f'<span class="inline-block max-w-[18rem] truncate align-middle text-[11px] '
                        f'text-red-700 dark:text-red-400" title="{html.escape(msg)}">'
                        f'Failed: {html.escape(msg)}</span>')


_BANNER_TONES = {
    "ok": "bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300",
    "fail": "bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300",
    "warn": "bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-300",
}


def _banner(tone: str, text: str, prefix: str = "") -> HTMLResponse:
    """The one result banner, for every action that reports into a strip on its page.

    `text` is escaped here, so an error message, a container name or a model's reply cannot
    inject markup; `prefix` is trusted markup (a check mark, "Failed:").
    """
    return HTMLResponse(f'<div class="rounded-md {_BANNER_TONES[tone]} px-3 py-2 text-sm">'
                        f'{prefix}{html.escape(text)}</div>')


def _toast(ok: bool, msg: str, fail: str = "Failed:") -> HTMLResponse:
    """A success or failure banner: a check mark, or `fail` ("" for none), then the message."""
    return _banner("ok" if ok else "fail", msg, "&check; " if ok else (f"{fail} " if fail else ""))




_QUEUED_CHIP = (
    '<a href="/downloads" title="{title}" '
    'class="shrink-0 inline-flex items-center gap-1 rounded-md bg-emerald-100 dark:bg-emerald-950 '
    'text-emerald-800 dark:text-emerald-300 px-3 py-1.5 text-xs font-medium">'
    '✓ {label}</a>'
)
