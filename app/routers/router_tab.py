"""The Router tab: per-engine routing controls, request totals and charts, conversation history."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from .. import db, engine_controls, router_api, strata_models
from ..config import settings
from ..web import templates, _form_note, _toast
from .strata import STRATA_ENV_FILE

router = APIRouter()


# ===================================================================== Router
#
# These routes are exempt from the read-only guard (see main._READONLY_ALLOW): they write only to this
# instance's own database and drive nothing on the host - no container, no models.ini, no engine.


# Windows the page offers. "all" is deliberately last and unbounded: with retention in place it
# is a real question again rather than an unbounded scan of everything ever recorded.
ROUTER_RANGES = (("24h", "last 24 hours", 86400.0), ("7d", "last 7 days", 7 * 86400.0),
                 ("30d", "last 30 days", 30 * 86400.0), ("90d", "last 90 days", 90 * 86400.0),
                 ("1y", "last year", 365 * 86400.0), ("all", "everything kept", 0.0))


def _router_since(rng: str) -> float:
    for key, _label, secs in ROUTER_RANGES:
        if key == rng:
            return (time.time() - secs) if secs else 0.0
    return time.time() - 86400.0


def _router_rows(since: float = 0.0, limit: int = 100) -> list[dict]:
    with db._conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT ts, conv, client, asked, engine, backend, model, shape, stream, status, "
            "error, prompt_tokens, completion_tokens, cached_tokens, ttft_ms, total_ms "
            "FROM router_request WHERE ts >= ? ORDER BY ts DESC LIMIT ?", (since, limit))]


def _router_conversations(since: float = 0.0, limit: int = 50) -> list[dict]:
    """One row per conversation, newest first.

    A conversation is the unit a person thinks in - "that chat got slow", "that one burned a lot
    of context" - and it is the thing the OpenAI protocol refuses to give you. Grouping on the
    prefix hash is what makes these questions answerable at all.
    """
    with db._conn() as c:
        # engine and model come from the conversation's LATEST row, not from MAX() of each:
        # independent aggregates pick independently and can pair an engine with a model it
        # never served.
        return [dict(r) for r in c.execute(
            "SELECT conv, COUNT(*) AS turns, MAX(ts) AS last_ts, "
            "       SUM(prompt_tokens) AS prompt_tokens, SUM(completion_tokens) AS completion_tokens, "
            "       SUM(cached_tokens) AS cached_tokens, "
            "       AVG(NULLIF(ttft_ms,0)) AS avg_ttft, AVG(total_ms) AS avg_total, "
            "       SUM(status >= 400) AS failures, "
            "       COUNT(DISTINCT CASE WHEN model <> '' THEN model END) AS models_used, "
            "       (SELECT engine FROM router_request x WHERE x.conv = r.conv AND x.engine <> '' "
            "         ORDER BY x.ts DESC LIMIT 1) AS engine, "
            "       (SELECT model  FROM router_request x WHERE x.conv = r.conv AND x.model  <> '' "
            "         ORDER BY x.ts DESC LIMIT 1) AS model, "
            "       MAX(client) AS client "
            "FROM router_request r WHERE conv <> '' AND ts >= ? "
            "GROUP BY conv ORDER BY last_ts DESC LIMIT ?", (since, limit))]




# Chart geometry is computed here rather than in template expressions: it is arithmetic with
# edge cases (an all-zero window, a single bucket, a stack whose segments round to sub-pixel
# heights) and those are worth being able to read and test.
_CHART_W = 720.0
_CHART_H = 132.0
_CHART_PAD_B = 16.0        # room for the time axis
_BAR_GAP = 2.0             # the surface gap the mark spec asks for between adjacent bars
_SEG_GAP = 2.0             # and between stacked segments, so a stack reads as parts


def _router_charts(ser: dict) -> dict:
    pts = ser.get("points") or []
    n = len(pts)
    out = {"grain": ser.get("grain", "hour"), "w": _CHART_W, "h": _CHART_H,
           "tokens": [], "requests": [], "token_max": 0, "request_max": 0,
           "ticks": [], "empty": n == 0}
    if not n:
        return out

    plot_h = _CHART_H - _CHART_PAD_B
    # The gaps between stacked segments take vertical space too, so the VALUES are scaled into
    # what is left. Scaling to the full height and then inserting gaps pushed the tallest stack
    # off the top of the plot - caught by checking that no segment had a negative y.
    stack_h = plot_h - 2 * _SEG_GAP          # at most three segments, so at most two gaps
    bar_w = max(1.0, (_CHART_W / n) - _BAR_GAP)

    token_max = max((p["read"] + p["output"] + p["reused"]) for p in pts) or 1
    req_max = max(p["requests"] for p in pts) or 1
    out["token_max"], out["request_max"] = token_max, req_max

    for i, p in enumerate(pts):
        x = (i * (_CHART_W / n)) + (_BAR_GAP / 2)
        # Stack order is fixed, so a series never changes colour between renders: reused at the
        # bottom (it is free), then the work that cost something.
        segs, y = [], plot_h
        for key, slot in (("reused", 3), ("read", 1), ("output", 2)):
            v = p[key]
            if v <= 0:
                continue
            h = (v / token_max) * stack_h
            if h < 0.6:        # keep a non-zero value visible rather than rounding it away
                h = 0.6
            y -= h
            segs.append({"key": key, "slot": slot, "x": round(x, 2), "y": round(y, 2),
                         "w": round(bar_w, 2), "h": round(max(0.6, h - _SEG_GAP), 2), "v": v})
            y -= _SEG_GAP
        out["tokens"].append({"x": round(x, 2), "w": round(bar_w, 2), "t": p["t"],
                              "segs": segs, "total": p["read"] + p["output"] + p["reused"],
                              "read": p["read"], "reused": p["reused"], "output": p["output"]})

        ok = p["requests"] - p["failures"]
        h_ok = (ok / req_max) * plot_h if ok > 0 else 0.0
        h_bad = (p["failures"] / req_max) * plot_h if p["failures"] else 0.0
        out["requests"].append({
            "x": round(x, 2), "w": round(bar_w, 2), "t": p["t"],
            "ok_y": round(plot_h - h_ok, 2), "ok_h": round(h_ok, 2),
            "bad_y": round(plot_h - h_ok - h_bad - (_SEG_GAP if h_ok and h_bad else 0), 2),
            "bad_h": round(h_bad, 2),
            "requests": p["requests"], "failures": p["failures"]})

    # A handful of ticks, never one per bar: a label under every bucket is unreadable at an
    # hour's resolution and tells you nothing a few anchors do not.
    step = max(1, -(-n // 6))     # ceil: n//6 gives a label under EVERY bar for small n
    out["ticks"] = [{"x": round((i * (_CHART_W / n)) + bar_w / 2, 2), "t": pts[i]["t"]}
                    for i in range(0, n, step)]
    return out


@router.get("/router/conversation/{conv}", response_class=HTMLResponse)
def router_conversation(request: Request, conv: str) -> HTMLResponse:
    """Every turn of one conversation.

    Only possible inside the raw window - once a conversation has been folded into hourly totals
    its individual turns are gone, by design. The page says so rather than showing an empty
    table that looks like a bug.
    """
    with db._conn() as c:
        turns = [dict(r) for r in c.execute(
            "SELECT ts, asked, engine, backend, model, stream, status, error, prompt_tokens, "
            "cached_tokens, completion_tokens, ttft_ms, total_ms "
            "FROM router_request WHERE conv = ? ORDER BY ts", (conv,))]
    return templates.TemplateResponse("router_conversation.html", {
        "request": request, "conv": conv, "turns": turns,
        "retention": router_api.retention_state(),
    })


@router.get("/router", response_class=HTMLResponse)
async def router_page(request: Request, range: str = "7d") -> HTMLResponse:
    # Discovery is async; everything after it (docker listings, Maya's configs through docker's
    # archive API, install configs on disk, the history queries) is blocking, so it runs in a
    # thread. On the event loop it stalled every other request this process serves, the
    # Overview's half-second poll included.
    since = _router_since(range)
    bs, by_model = await router_api._live_map()
    ctx = await asyncio.to_thread(_router_page_context, bs, range, since)
    return templates.TemplateResponse("router.html", {"request": request, **ctx})


def _router_page_context(bs: list[dict], range: str, since: float) -> dict:
    return {
        **engine_controls.rows_context(bs),
        "backends": bs,
        "recording": db.get_setting("router.record", "1") != "0",
        "conversations": _router_conversations(since),
        "rows": _router_rows(since),
        "port": settings.router_port,
        "ranges": ROUTER_RANGES,
        "range": range,
        "totals": router_api.totals(since),
        "charts": _router_charts(router_api.series(since)),
        "retention": router_api.retention_state(),
    }


@router.post("/router/default", response_class=HTMLResponse)
def router_set_default(engine: str = Form(...), model: str = Form("")) -> HTMLResponse:
    if engine not in router_api.ENGINES:
        return _form_note(False, f"unknown engine {engine!r}")
    if engine == "maya":
        return _form_note(False, "Maya starts the model its compose command names; it is not set here")
    db.set_setting(f"router.default.{engine}", model.strip())
    if engine == "strata" and model.strip():
        # For Strata the value is an INSTALL TAG, because that is the unit you can choose - the
        # loaded model cannot be changed without a restart. Applying it means writing the same
        # pin file the Strata tab writes, which a read-only instance must not do: two instances
        # driving the same engine is how this setup produced crash loops.
        tag = model.strip()
        fam = "qwen"
        for f in ("unsloth-", "swift-", "coder-"):
            if tag.startswith(f):
                fam = f[:-1]
                break
        mdl = tag[len(fam) + 1:].upper() if fam != "qwen" else tag.upper()
        if settings.router_readonly:
            return _form_note(True, f"{tag} recorded as the Strata default. This instance is "
                                    f"read-only, so apply it from the live Model Loader's "
                                    f"Strata tab ({fam} / {mdl})", label="noted, not applied")
        ok, msg = strata_models.set_selection(STRATA_ENV_FILE, fam, mdl, "")
        return _form_note(ok, msg + ("; restart Strata to apply" if ok else ""),
                          label="saved, applies on next start")
    return _form_note(True, f"{engine} now defaults to {model or '(none)'}")


@router.post("/router/backend", response_class=HTMLResponse)
def router_set_backend(engine: str = Form(...), backend: str = Form("")) -> HTMLResponse:
    """Pin an engine's traffic to one container, or clear it back to automatic."""
    if engine not in router_api.ENGINES:
        return _form_note(False, f"unknown engine {engine!r}")
    db.set_setting(f"router.backend.{engine}", backend.strip())
    return _form_note(True, f"{engine} pinned to {backend}" if backend.strip()
                      else f"{engine} back to automatic (CPU last, purpose-named servers after the "
                           f"general router)")


@router.post("/router/recording", response_class=HTMLResponse)
def router_set_recording(request: Request, on: str = Form("1")) -> HTMLResponse:
    """The kill switch. Off, the router still routes but writes nothing.

    It exists so that if accounting ever misbehaves - a lock, a slow write, a bad migration -
    turning it off is one click rather than editing config and restarting under pressure, while
    inference keeps working.
    """
    db.set_setting("router.record", "1" if on == "1" else "0")
    return templates.TemplateResponse("_recording_toggle.html", {
        "request": request, "recording": db.get_setting("router.record", "1") != "0"})


@router.post("/router/autoswitch", response_class=HTMLResponse)
def router_set_autoswitch(request: Request, engine: str = Form(...), on: str = Form("")) -> HTMLResponse:
    """Let a request for `engine` hand it the cards, instead of being refused.

    Per engine and off by default, because this is the one control here that moves hardware on
    its own: turning it on means a client picking a name in a model dropdown can stop the engine
    somebody else is using. The guards live in router_api.may_autoswitch() - it refuses while
    anything is in flight on the other engine, and for a settle period after the last switch -
    but the decision to allow it at all is this switch, and it is yours.
    """
    if engine not in router_api.ENGINES:
        return _toast(False, f"unknown engine {engine}")
    db.set_setting(f"router.autoswitch.{engine}", "1" if on == "1" else "")
    # The switch itself, re-rendered in its new state - not a "Saved" chip, which replaced the
    # control and left nothing to click until a reload.
    return templates.TemplateResponse("_autoswitch_toggle.html", {
        "request": request, "e": engine, "on": router_api.autoswitch_enabled(engine),
        "dwell": int(router_api.dwell_s())})
