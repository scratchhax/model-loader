from __future__ import annotations

import asyncio
import threading
import time

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import db, hw, ini, services
from . import maya as maya_engine
from . import strata as strata_engine
from . import strata_models
from . import router_api
from .config import settings
from .fit import _backend_list, _predicted_vram_gb
from .routers import benchmark as benchmark_router
from .routers.benchmark import _bench_ctx
from .routers import config as config_router
from .routers import containers as containers_router
from .routers import downloads as downloads_router
from .routers import models as models_router
from .routers import overview as overview_router
from .routers import power as power_router
from .routers import prompts as prompts_router
from .routers import search as search_router
from .routers import settings as settings_router
from .routers import strata as strata_router
from .routers import maya as maya_router
from .routers import updates as updates_router
from .routers.strata import STRATA_DATA, STRATA_ENV_FILE, _strata_selected_tag
from .web import templates, _QUEUED_CHIP, _form_note, _toast

app = FastAPI(title="Model Loader")


# Paths a read-only instance may still write to: its own router settings, which live in its own
# database and drive nothing on the host.
_READONLY_ALLOW = ("/router/",)


@app.middleware("http")
async def _readonly_guard(request: Request, call_next):
    """Refuse every mutating request when this instance is not the one in charge.

    Model Loader owns real lifecycle: it starts and stops engines, writes models.ini, writes the
    Strata pin file and drives the engine switch. Running a second instance for development is
    only safe if it cannot do any of that - otherwise two schedulers fight over the same two
    cards, and the symptom is a container that will not start, which is exactly the failure this
    setup keeps producing. The guard is here rather than in compose because a read-only VOLUME
    does not stop a container restart through the docker socket.
    """
    if (settings.router_readonly
            and request.method in ("POST", "PUT", "PATCH", "DELETE")
            and not request.url.path.startswith(_READONLY_ALLOW)):
        return JSONResponse(
            status_code=403,
            content={"detail": "read-only instance: the live Model Loader owns this action"},
        )
    return await call_next(request)


_UNSAFE_METHODS = ("POST", "PUT", "PATCH", "DELETE")


def cross_site(headers) -> str:
    """Why a mutating request is refused as coming from another site, or "".

    This app holds the docker socket, so a form post is as good as root on the host, and with no
    check any page open in a browser on the LAN could make one (a cross-site form POST needs no
    script and no CORS). Browsers name the page a request came from in Origin, and in Referer
    when they leave Origin out; either one naming a different host is refused, and so is
    Origin: null (a sandboxed frame or a file:// page). A request with neither is a script or
    the router process asking for a switch, and is let through: this is not authentication.
    """
    from urllib.parse import urlsplit
    src = headers.get("origin") or headers.get("referer") or ""
    if not src:
        return ""
    if src == "null":
        return "a page with no origin"
    host = urlsplit(src).netloc.lower()
    mine = {h.lower() for h in (headers.get("host"), headers.get("x-forwarded-host")) if h}
    return "" if host in mine else f"a page on {host or src}"


@app.middleware("http")
async def _same_site_only(request: Request, call_next):
    if request.method in _UNSAFE_METHODS:
        why = cross_site(request.headers)
        if why:
            return JSONResponse(status_code=403,
                                content={"detail": f"refused: this request came from {why}, "
                                                   f"not from Model Loader"})
    return await call_next(request)


@app.on_event("startup")
def _startup() -> None:
    db.init()
    # BOTH router tables, here as well as in the router process. CREATE TABLE IF NOT EXISTS
    # makes the overlap safe, and it is not redundant: the deployed container runs a single
    # uvicorn (app.main), so this is the only startup that runs there. The Router tab reads the
    # rollup through totals(), series() and retention_state(), and creating only the request
    # table left all three raising "no such table: router_rollup" on a fresh database - it
    # worked in the two-process dev container and nowhere else.
    router_api._init_db()
    router_api._init_rollup()
    db.seed_bench_prompts()
    # Downloads a deploy or restart interrupted; a single-stream temp resumes where it stopped.
    downloads_router.manager.resume_pending()
    hw.start_sampler()
    threading.Thread(target=_reconcile_strata_config, daemon=True, name="strata-config").start()


def _reconcile_strata_config() -> None:
    """Once at startup, ensure the coexistence keys are in every Strata backend's run config.

    Startup rather than every poll: the engine reads the config at start, so a key that
    appeared since then only matters for the next start anyway - and the card's note says
    exactly that meanwhile. A thread because the docker calls must not hold up the first
    request, and silent because the card reports the delta either way.
    """
    keys = services.strata_desired_keys()
    if not keys:
        return
    try:
        client = services._docker_client()
        if client is None:
            return
        own = services.strata_own_attrs(client)
        for c in client.containers.list(all=True, filters={"label": f"ai-lab.engine={strata_engine.ENGINE}"}):
            path = strata_engine.run_config_path(c.attrs, own)
            if path is not None:
                strata_engine.merge_run_config(path, keys)
    except Exception:  # noqa: BLE001 - no docker socket is a normal state for a dev checkout
        pass


@app.get("/palette.json")
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


# ---------- Overview ----------

app.include_router(overview_router.router)


# ---------- Power and thermals ----------

app.include_router(power_router.router)


# ---------- Models ----------

app.include_router(models_router.router)


# ---------- Settings ----------

app.include_router(settings_router.router)


# ---------- Search ----------

app.include_router(search_router.router)


# ---------- Downloads ----------

app.include_router(downloads_router.router)


# ---------- models.ini ----------

app.include_router(config_router.router)


# ---------- Containers ----------

app.include_router(containers_router.router)


# ---------- Maya ----------

app.include_router(maya_router.router)


# ---------- Engine updates ----------

app.include_router(updates_router.router)


# ---------- Prompt library ----------

app.include_router(prompts_router.router)


# ---------------------------------------------------------------- benchmark

app.include_router(benchmark_router.router)


@app.post("/badge", response_class=HTMLResponse)
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


@app.post("/badge/clear", response_class=HTMLResponse)
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


def _median(xs: list[float]) -> float:
    """Median, not mean. Same reasoning as everywhere else here: one request that queued behind
    a model load reads as a fraction of the true rate and drags an average with it."""
    xs = sorted(x for x in xs if x)
    if not xs:
        return 0.0
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def _bench_charts(run_id: int, results: list, sweeps: list) -> dict:
    """Series for the charts, shaped so the template just hands them to Chart.js.

    Cold and contended rows are excluded from every aggregate: one includes a model load, the
    other ran while something else had the GPU, and neither describes steady-state speed. They
    stay in the table, where the badges explain them.
    """
    import json as _json
    clean = [r for r in results if not r["cold"] and not r["contended"] and not r["err"]]

    aliases = sorted({r["alias"] for r in clean})
    prompts = sorted({r["prompt_name"] for r in clean})

    def _pick(alias, prompt, field, digits):
        """Median for one cell, or None when there is nothing clean to report.

        None rather than 0.0: Chart.js skips a null point but draws a zero, and a zero bar on
        a speed chart reads as "this model produced nothing" when the truth is "every row for
        it was cold, contended or failed". A model that would not load looked like a model
        that ran infinitely slowly.
        """
        vals = [r[field] for r in clean
                if r["alias"] == alias and r["prompt_name"] == prompt and r[field]]
        return round(_median(vals), digits) if vals else None

    gen_series = [{"label": p, "data": [_pick(a, p, "gen_tps", 1) for a in aliases]}
                  for p in prompts]
    ttft_series = [{"label": p, "data": [_pick(a, p, "ttft_ms", 0) for a in aliases]}
                   for p in prompts]

    # Depth decay, straight from llama bench. One line per model per test kind, x = depth.
    depths = sorted({w["n_depth"] or 0 for w in sweeps})
    decay: list[dict] = []
    for a in sorted({w["alias"] for w in sweeps}):
        for kind, want_gen in (("tg", True), ("pp", False)):
            pts = []
            for d in depths:
                m = [w["avg_ts"] for w in sweeps
                     if w["alias"] == a and (w["n_depth"] or 0) == d
                     and bool(w["n_gen"]) == want_gen and w["avg_ts"]]
                pts.append(round(m[0], 1) if m else None)
            if any(p is not None for p in pts):
                decay.append({"label": f"{a} {kind}", "data": pts, "kind": kind})

    # Measured VRAM against what autoconfig predicted. The one chart that can falsify the fit
    # maths: everything else here measures speed, which the estimator never claimed to know.
    vram_labels, vram_measured, vram_predicted = [], [], []
    capacity = 0.0
    try:
        for b in _backend_list():
            capacity = max(capacity, float(b.get("vram_gb") or 0))
    except Exception:  # noqa: BLE001
        capacity = 0.0
    for a in aliases:
        peaks = []
        for r in clean:
            if r["alias"] != a:
                continue
            try:
                peaks.append(sum(_json.loads(r["peak_vram_json"] or "[]")))
            except ValueError:
                pass
        if not peaks:
            continue
        vram_labels.append(a)
        vram_measured.append(round(max(peaks), 2))
        vram_predicted.append(_predicted_vram_gb(a))

    return {
        "capacity_gb": round(capacity, 2),
        "aliases": aliases,
        "gen": gen_series,
        "ttft": ttft_series,
        "depths": depths,
        "decay": decay,
        "vram_labels": vram_labels,
        "vram_measured": vram_measured,
        "vram_predicted": vram_predicted,
    }


@app.get("/benchmark/run/{run_id}", response_class=HTMLResponse)
def benchmark_run(request: Request, run_id: int) -> HTMLResponse:
    import json as _json
    ctx = _bench_ctx(request)
    results = db.bench_results(run_id)
    sweeps = db.bench_sweeps(run_id)
    variants = db.bench_variants(run_id)
    ctx.update({
        "run": db.bench_run(run_id),
        "variants": variants,
        "results": results,
        "sweeps": sweeps,
        "charts_json": _json.dumps(_bench_charts(run_id, results, sweeps)),
        "badges": {v["alias"]: db.badges_for(v["alias"]) for v in variants},
    })
    return templates.TemplateResponse("benchmark.html", ctx)


# ===================================================================== Strata models

app.include_router(strata_router.router)


# ===================================================================== Router
#
# These routes are exempt from the read-only guard (see _READONLY_ALLOW): they write only to this
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
        # engine and model come from the conversation's LATEST row, not from MAX() of each.
        # Independent aggregates pick independently: a conversation that hit Strata and llama
        # reported MAX(engine)="strata" with MAX(model)="wheatley-voice" - a pair that never
        # happened, and which read as Strata having loaded a voice model.
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


def _strata_choices() -> list[tuple[str, str]]:
    """[(install tag, label)] for the Strata default — the INSTALLED packs, not the loaded one.

    Strata serves exactly one model at a time, so asking it what it offers returns one id, and a
    dropdown built from that can only ever say "the thing already running". But a Strata default
    is a statement about the NEXT start, so the set to choose from is what is installed. Those
    come from the install configs on the volume, which is also why this still lists them while
    Strata is stopped.
    """
    import json as _json
    out: list[tuple[str, str]] = []
    prov = strata_models.provenance()
    for tag, d in strata_models.installed(STRATA_DATA).items():
        name = tag
        try:
            name = _json.loads(open(d["config"], encoding="utf-8").read()).get("model_name") or tag
        except (OSError, ValueError):
            pass
        owner = (prov.get(tag, {}).get("repo") or "").split("/")[0]
        real = strata_models.install_label(strata_models.describe_install(tag, prov))
        label = f"{real}  ({owner}, served as {name})" if real else (f"{name}  ({owner})" if owner else name)
        if d.get("missing"):
            label += "  — source files missing"
        out.append((tag, label))
    return out


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


@app.get("/router/conversation/{conv}", response_class=HTMLResponse)
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


@app.get("/router", response_class=HTMLResponse)
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
    per_engine: dict[str, list] = {}
    for b in bs:
        if b["engine"] == "strata":
            continue                       # built from installed packs below, not from /v1/models
        per_engine.setdefault(b["engine"], [])
        for m in b["models"]:
            if m not in per_engine[b["engine"]]:
                per_engine[b["engine"]].append(m)
    # (value, label) pairs, which is what the template unpacks. Every engine but Strata offers
    # its model ids as both; Strata's are built from installed packs below. Converting llama
    # alone left any other engine's list as bare strings, and the Router tab 500'd ("too many
    # values to unpack") the first time Maya was running.
    per_engine.setdefault("llama", [])     # always present: the unloaded sections join it below
    for e in list(per_engine):
        per_engine[e] = [(m, m) for m in per_engine[e]]
    # ini sections no running llama backend serves (added after its last start) are offered
    # too, labelled - a dropdown that silently omits them reads as the app losing the model.
    unloaded = set(router_api.unloaded_sections([m for m, _ in per_engine.get("llama", [])]))
    per_engine["llama"] += [(n, n) for n in sorted(unloaded)]
    per_engine["strata"] = _strata_choices()
    # Maya's list is its INSTALLED models, read from its config volume, so it is there while the
    # container is stopped - which, the engines taking turns, is most of the time you look.
    # Falls back to whatever a running Maya advertises if the configs cannot be read.
    maya_choices = maya_engine.choices(services._docker_client())
    if maya_choices:
        per_engine["maya"] = maya_choices
    defaults = {e: router_api.default_for(e) for e in router_api.ENGINES}
    # Maya's row shows what it WILL serve: the model of the config its compose command names.
    # router.default.maya was never read by anything, so a choice there was a no-op.
    maya_run_config = maya_engine.run_config_name(services._docker_client())
    maya_runs = next((r["model"] for r in maya_engine.installed(services._docker_client())
                      if r["file"] == maya_run_config), "")
    if maya_runs:
        defaults["maya"] = maya_runs
    elif len(per_engine.get("maya", [])) == 1:
        defaults["maya"] = per_engine["maya"][0][0]
    # Strata's row, likewise: the install strata.env names, which the Strata tab can also change.
    # The stored router.default.strata went stale the first time it did.
    strata_next = _strata_selected_tag(strata_models.read_selection(STRATA_ENV_FILE))
    if strata_next:
        defaults["strata"] = strata_next
    return {
        "engines": router_api.ENGINES,
        "per_engine": per_engine,
        "unloaded": unloaded,
        "defaults": defaults,
        "maya_run_config": maya_run_config,
        "pins": {e: router_api.pinned_backend(e) for e in router_api.ENGINES},
        "autoswitch": {e: router_api.autoswitch_enabled(e) for e in router_api.ENGINES},
        "autoswitch_dwell": int(router_api.dwell_s()),
        # Stopped ones included ON PURPOSE: the backend worth pinning is usually the one that
        # is not running, because the engines take turns.
        "backends_by_engine": {
            e: [(b["name"], b.get("running", True))
                for b in sorted(router_api._backends(include_stopped=True), key=router_api._rank)
                if b["engine"] == e]
            for e in router_api.ENGINES},
        "live_engines": sorted({b["engine"] for b in bs}),
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


@app.post("/router/default", response_class=HTMLResponse)
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


@app.post("/router/backend", response_class=HTMLResponse)
def router_set_backend(engine: str = Form(...), backend: str = Form("")) -> HTMLResponse:
    """Pin an engine's traffic to one container, or clear it back to automatic."""
    if engine not in router_api.ENGINES:
        return _form_note(False, f"unknown engine {engine!r}")
    db.set_setting(f"router.backend.{engine}", backend.strip())
    return _form_note(True, f"{engine} pinned to {backend}" if backend.strip()
                      else f"{engine} back to automatic (CPU last, purpose-named servers after the "
                           f"general router)")


@app.post("/router/recording", response_class=HTMLResponse)
def router_set_recording(request: Request, on: str = Form("1")) -> HTMLResponse:
    """The kill switch. Off, the router still routes but writes nothing.

    It exists so that if accounting ever misbehaves - a lock, a slow write, a bad migration -
    turning it off is one click rather than editing config and restarting under pressure, while
    inference keeps working.
    """
    db.set_setting("router.record", "1" if on == "1" else "0")
    return templates.TemplateResponse("_recording_toggle.html", {
        "request": request, "recording": db.get_setting("router.record", "1") != "0"})


@app.post("/router/autoswitch", response_class=HTMLResponse)
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
