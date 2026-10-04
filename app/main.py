from __future__ import annotations

import html
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from . import autoconfig
from . import telemetry
from . import bench, db, gguf_meta, gpu_procs, hf, hw, ini, services, vram_live
from .config import settings
from .downloader import manager
from .utils import human_bytes, shard_key

app = FastAPI(title="Model Loader")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.globals["hue"] = lambda s: sum(ord(c) for c in (s or "")) % 360
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


@app.on_event("startup")
def _startup() -> None:
    db.init()
    db.seed_bench_prompts()
    hw.start_sampler()


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


@app.get("/classic", response_class=HTMLResponse)
async def dashboard_classic(request: Request) -> HTMLResponse:
    """The counter-led overview this replaced. Kept reachable for one release in case the new
    one is missing something; delete it, and dashboard_classic.html, once nobody wants it."""
    snap = services.snapshot_models_dir()
    backends = await services.snapshot_llama_backends()
    sections = ini.list_sections()
    # recent downloads: successful ones from history, deduped by filename
    rows = db.recent_downloads(20)
    seen: set[str] = set()
    recent: list[dict] = []
    owners: set[str] = set()
    for r in rows:
        if r["status"] != "done":
            continue
        if r["filename"] in seen:
            continue
        seen.add(r["filename"])
        owner = (r["repo_id"].split("/", 1)[0] if "/" in r["repo_id"] else "") if r["repo_id"] else ""
        if owner:
            owners.add(owner)
        recent.append({
            "filename": r["filename"],
            "size_h": human_bytes(int(r["total_bytes"] or 0)),
            "owner": owner,
        })
        if len(recent) >= 5:
            break
    avatars = await hf.owner_avatars(list(owners)) if owners else {}
    for r in recent:
        r["avatar_url"] = avatars.get(r["owner"], "")
    active = sum(1 for j in manager.snapshot() if j.status in ("queued", "downloading"))
    return templates.TemplateResponse("dashboard_classic.html", {
        "request": request,
        "snap": snap,
        "backends": backends,
        "stats": _stats_by_name(),
        "ini_sections": sections,
        "recent_downloads": recent,
        "active_downloads": active,
    })


# ---------- Mission control (v2 overview) ----------

def _host_line() -> str:
    """One line of context under the page title: what this box is, in its own terms."""
    bits: list[str] = []
    names: list[str] = []
    total_vram = 0.0
    isa = ""
    for name in services._effective_container_names():
        st = hw.stats_for(name)
        if st.ok and st.gpu and st.gpu.vram_total_gb > 0:
            total_vram = max(total_vram, st.gpu.vram_total_gb)
            names.append(name)
            for c in st.gpu.cards:
                if "gfx" in (c.name or "") and not isa:
                    isa = c.name.split("(")[-1].rstrip(")")
    if total_vram:
        bits.append(f"{total_vram:.0f} GB VRAM")
    if isa:
        bits.append(isa)
    ram = hw.host_ram_gb()
    if ram:
        bits.append(f"{ram:.0f} GB RAM")
    return " · ".join(bits) if bits else "Everything live at a glance."


# Which backends exist and what they hold changes on the timescale of a model load, not of a
# 500 ms poll, and discovering it costs ~50 ms of docker API and HTTP probes - the most
# expensive thing on the hero's path. Cached briefly so the fast poll spends its time on the
# ~1 ms slot read that actually moves.
_HERO_BACKENDS_TTL = 1.5
_hero_backends: tuple[float, list] | None = None


async def _hero_backends_cached() -> list:
    global _hero_backends
    if _hero_backends and (time.time() - _hero_backends[0]) < _HERO_BACKENDS_TTL:
        return _hero_backends[1]
    backends = await services.snapshot_llama_backends()
    _hero_backends = (time.time(), backends)
    return backends


# The MTP/spec box reads the telemetry rollup, and the hero polls at 500 ms. The rollup itself
# only moves when telemetry re-ingests (every 20 s), so the box is recomputed on a 10 s TTL and
# the ingest call inside it is self-throttled the same way.
_MTP_TTL_S = 10.0
_mtp_cache: dict[str, tuple[float, "telemetry.Stats"]] = {}


def _hero_mtp(backend: str, model_id: str) -> dict | None:
    """Draft-acceptance facts for the running model, or None when it does not speculate.

    What the running instance was SPECULATING with comes off its own argv (the telemetry db's
    newest spawn record), not the ini: the box claims what the server is actually doing. The
    numbers come from the print_timing draft-acceptance lines the telemetry module scrapes -
    measured from real traffic, which is the only honest source, since llama.cpp reports the
    acceptance for the requests it has already served and nothing at all until one finishes.
    """
    argv = db.latest_server_config(backend).get("argv") or {}
    stype = str(argv.get("--spec-type") or "").strip()
    if not stype or stype == "none":
        return None
    key = f"{backend}/{model_id}"
    now = time.time()
    ent = _mtp_cache.get(key)
    if ent is None or (now - ent[0]) > _MTP_TTL_S:
        try:
            telemetry.ingest([backend])
            ent = (now, telemetry.stats_for(model_path="", alias=model_id))
        except Exception:  # noqa: BLE001 - a missing box, never a broken hero
            ent = (now, telemetry.Stats())
        _mtp_cache[key] = ent
    tel = ent[1]
    return {
        "spec_type": stype,
        "label": "MTP" if "mtp" in stype else ("NGRAM" if stype.startswith("ngram") else "SPEC"),
        "acc_pct": (100.0 * tel.draft_acc_p50) if tel.draft_acc_p50 is not None else None,
        "mean_len": tel.draft_len_p50 if tel.draft_len_p50 else 0.0,
        "n": tel.draft_n,
        "age": tel.age_str,
        "n_max": tel.draft_n_max,
    }


async def _hero_context() -> dict:
    """The loaded model, its throughput, and enough of its config to read the panel.

    Picks the busiest backend rather than the first: with several backends, the one actually
    working is the one worth leading with. Falls back to any backend holding a model, then to
    an empty standby panel.
    """
    backends = await _hero_backends_cached()
    best: tuple[int, object, object] | None = None      # (rank, backend, speed)
    scored: list[tuple[int, object, object]] = []
    for b in backends:
        if not b.loaded_model:
            continue
        sp = await services.inference_speed(b.name, b.internal_port, b.loaded_model)
        # Three questions, in the order they matter on a box whose whole point is the cards:
        #
        #   generating now   the thing actually working always leads, GPU or not
        #   touches a GPU    an idle CPU backend is not the headline while a card holds a model
        #   is a chat router a TTS server holds its model permanently and so always counts as
        #                    loaded; it must not outrank the model you are talking to
        #
        # Ranking on `router` alone tied three backends that share one models.ini - the GPU
        # router, llama-cpu and llama-voice - so the winner was whichever docker happened to
        # enumerate first, and the hero appeared to pick at random. In practice it would show
        # llama-cpu serving wheatley-voice while a 25 GB model sat on the cards.
        _gpu = b.vendor in ("rocm", "cuda")
        rank = (4 if (sp and sp.live) else 0) + (2 if _gpu else 0) + (1 if b.router else 0)
        scored.append((rank, b, sp))
        if best is None or rank > best[0]:
            best = (rank, b, sp)
    if best is None:
        return {"speed": None, "hero_model": "", "hero_backend": "",
                "hero_shape": None, "hero_quant": "", "hero_size_h": "",
                "hero_ctx_cfg": "", "hero_vram_used": None, "hero_vram_total": 0.0,
                "tps_spark": "", "tps_peak": 0.0, "tps_samples": 0, "hero_mtp": None,
                "hero_others": []}

    _rank, b, sp = best

    # Everything else holding a model. The hero can only lead with one, and on this box three
    # backends are routinely resident at once - a chat model on the cards, a TTS model beside
    # it, and the Home Assistant voice model on the CPU. Picking one and silently dropping the
    # rest is what made the headline look arbitrary; showing them removes the question.
    hero_others = []
    for _r, ob, osp in sorted(scored, key=lambda t: -t[0]):
        if ob.name == b.name:
            continue
        hero_others.append({
            "backend": ob.name,
            "model": (ob.loaded_model or "").split(",")[0].strip(),
            "gpu": ob.vendor in ("rocm", "cuda"),
            "live": bool(osp and osp.live),
            "state": (osp.state if osp else "idle"),
            "tps": round(osp.gen_tps, 1) if (osp and osp.live and osp.gen_tps) else 0.0,
            "asleep": bool(getattr(ob, "asleep", False)),
        })

    model_id = (b.loaded_model or "").split(",")[0].strip()

    # Fold in this backend's log before anything reads its spawn record. Rate-limited to once
    # every 20 s internally, so an HTMX poll does not re-read the log every two seconds.
    #
    # This used to happen only inside _hero_mtp(), which returns early when the RECORDED argv
    # declares no --spec-type - and that record is exactly the thing that goes stale. Load a
    # model whose predecessor did not speculate and nothing on this page ever refreshed it:
    # observed four minutes after a swap, with the VRAM meter costing a 25 GB dense gemma
    # against the 90 GB MoE record it replaced. A page that renders a backend's state should
    # not depend on what the previous model happened to have configured.
    try:
        telemetry.ingest([b.name])
    except Exception:  # noqa: BLE001 - stale telemetry must never cost the hero
        pass

    # Config and shape come from the section and the GGUF header - the same sources the models
    # page uses, so the hero cannot disagree with the rest of the app.
    ctx_cfg = ""
    quant = ""
    size_h = ""
    shape = None
    try:
        sec = ini.get_section(model_id) or {}
        raw_ctx = str(sec.get("ctx-size") or "").strip()
        if raw_ctx.isdigit():
            ctx_cfg = f"{int(raw_ctx) // 1024}k"
        rel = str(sec.get("model") or "").replace("/models/", "", 1)
        if rel:
            path = settings.models_dir / rel
            quant = hf.infer_quant(path.name) or ""
            if path.exists():
                size = path.stat().st_size
                if "/" in rel:
                    size = sum(x.stat().st_size for x in path.parent.iterdir()
                               if x.is_file() and x.suffix.lower() == ".gguf"
                               and not ini._is_companion(x.name))
                size_h = human_bytes(size)
                shape = services.model_shape(path)
    except Exception:  # noqa: BLE001 - the hero degrades to fewer facts, it never 500s
        pass

    st = hw.stats_for(b.name)
    vram_used = st.gpu.vram_used_gb if (st.ok and st.gpu) else None
    vram_total = st.gpu.vram_total_gb if (st.ok and st.gpu) else 0.0

    # How much of this model is NOT on the cards, for the one-line readout under the context
    # bar. Only the numbers: the hero is glanced at, so the reasoning lives in the line's
    # tooltip and the full treatment stays on the autoconfig panel, where someone is reading
    # deliberately. None whenever nothing is offloaded, which is the common case.
    hero_weights_split = None
    try:
        if st.ok and st.gpu:
            _t = _gpu_tenants()
            hero_weights_split = (vram_live.breakdown(
                b.name, st.gpu,
                frozenset(i.strip() for i in (b.loaded_model or "").split(",") if i.strip()),
                foreign=[(t.label, t.total_gb) for t in _t if t.foreign and t.total_gb > 0.005],
            ) or {}).get("weights_split")
    except Exception:  # noqa: BLE001 - a missing line must never cost the hero
        hero_weights_split = None

    # The sparkline is scaled to the peak in its own window, and that peak is printed beside it.
    # A fixed scale cannot work here: 27 tok/s is a busy dense model and a slow MoE one, so the
    # only honest options are a stated scale or a meaningless one.
    tps = services.tps_history(b.name)
    tps_peak = max(tps) if tps else 0.0
    return {"speed": sp, "hero_model": model_id, "hero_backend": b.name,
            "hero_shape": shape, "hero_quant": quant, "hero_size_h": size_h,
            "hero_ctx_cfg": ctx_cfg, "hero_vram_used": vram_used, "hero_vram_total": vram_total,
            "tps_spark": hw.sparkline(tps, tps_peak or None) if len(tps) > 2 else "",
            "tps_peak": tps_peak, "tps_samples": len(tps),
            # Drive the per-slot strip's row height and column wrap. Computed here rather than
            # in the template so the thresholds sit with the dataclass they describe.
            "slot_density": services.slot_density(len(sp.slots) if sp else 0),
            "slot_columns": services.slot_columns(len(sp.slots) if sp else 0),
            "hero_weights_split": hero_weights_split,
            "hero_others": hero_others,
            "hero_mtp": _hero_mtp(b.name, model_id)}


def _gpu_tenants() -> list:
    """Every process on the cards, ours and not ours. [] on a box this cannot be read on.

    Lives here rather than in gpu_procs so the module stays a probe with no opinion about
    which containers the app considers its own.
    """
    try:
        return gpu_procs.tenants(set(services._effective_container_names()))
    except Exception:  # noqa: BLE001 - a tenant table that fails degrades to not being drawn
        return []


def _solo_backend() -> str:
    """The single running backend that sees GPUs, or "" when zero or several do.

    The live VRAM breakdown estimates the components of ONE backend's load against the card,
    so several GPU-bearing backends means the residual segment would silently absorb the
    second one's model - the meter falls back to the plain measured bar instead. Backends
    without GPU cards (a CPU build, a voice pipeline) cannot put anything on the device, so
    they do not break sole ownership, and the "other" residual would catch them if they did.

    Note what this does NOT cover, and never could: a GPU process the app did not start. It
    enumerates BACKENDS, and a TTS server or an image generator is not one. That gap is what
    gpu_procs closes - sole ownership among backends is still the condition for drawing the
    meter, and the meter is now told about the tenants that are not backends.
    """
    gpu_backends = [n for n in services._effective_container_names()
                    if (st := hw.stats_for(n)).ok and st.gpu and st.gpu.cards]
    return gpu_backends[0] if len(gpu_backends) == 1 else ""


def _gpu_strip_context() -> dict:
    """Per-card stats plus per-card sparklines, for whichever backend has GPUs.

    Per card on purpose: llama.cpp places each layer on one device, and the allocations that are
    not layer-split all land on the main GPU, so a pooled bar can read comfortable while one card
    is full.

    When exactly one backend owns the cards, each card's bar is additionally split into the
    live part-to-whole breakdown (vram_live) - the same meter the autoconfig panel draws for
    the recommendation, built here from what was ACTUALLY loaded against measured usage.
    """
    cards: list = []
    history: dict[int, dict] = {}
    card_breakdowns: dict[int, dict] = {}
    pooled_vb: dict | None = None
    solo = _solo_backend()
    tenants = _gpu_tenants()
    foreign_cards = gpu_procs.foreign_by_card(tenants)
    foreign_pool = [(t.label, t.total_gb) for t in tenants if t.foreign and t.total_gb > 0.005]
    for name in services._effective_container_names():
        st = hw.stats_for(name)
        if not (st.ok and st.gpu and st.gpu.cards):
            continue
        cards = st.gpu.cards
        if name == solo:
            # The pooled meter is kept as well as the per-card ones. A per-card row has no
            # room for a legend, so it collapses weights, KV and compute into one apportioned
            # segment - fine for "is this card full", useless for "how much of that is
            # context". The pooled meter is where those are separable, because the fixed costs
            # are known per card and only the PLACEMENT is not, so it is drawn once underneath
            # with the full legend and the spill bar, exactly as the container card draws it.
            pooled_vb = vram_live.breakdown(name, st.gpu,
                                            frozenset(services.last_loaded_ids(name)),
                                            foreign=foreign_pool)
            card_breakdowns = vram_live.per_card(pooled_vb, cards, foreign_cards)
        pts = hw.history_for(name)
        if pts:
            span_s = (pts[-1].ts - pts[0].ts) if len(pts) > 1 else 0.0
            mins = int(span_s // 60)
            label = f"last {mins}m" if mins >= 1 else f"last {int(span_s)}s"
            for c in cards:
                util = [p.per_gpu_util[c.index] for p in pts
                        if len(p.per_gpu_util) > c.index]
                vram = [p.per_gpu_vram_used_gb[c.index] for p in pts
                        if len(p.per_gpu_vram_used_gb) > c.index]
                history[c.index] = {
                    "window_label": label,
                    "util": hw.sparkline(util, 100.0) if util else "",
                    "vram": hw.sparkline(vram, c.vram_total_gb or None) if vram else "",
                }
        break
    # The estimate-based meter is all-or-nothing on the backend's model and says None whenever
    # it cannot size one - idle router, sleeping model, no telemetry. A foreign tenant's VRAM
    # is measured, not estimated, so it should not disappear with it. Fall back to drawing what
    # IS known rather than an anonymous fill.
    if not card_breakdowns and foreign_cards and cards:
        card_breakdowns = vram_live.tenants_only_per_card(cards, foreign_cards)
    return {"cards": cards, "gpu_history": history, "card_breakdowns": card_breakdowns,
            "tenants": _tenant_rows(tenants, cards)}


def _tenant_rows(tenants: list, cards: list) -> list[dict]:
    """The "on the cards" table: one row per GPU process, in the card order the strip uses.

    Drawn whenever there is anything to draw, including when the per-card meters are not -
    this table needs no estimate and no sole owner, only the kernel's own accounting, so it
    is the one readout that keeps working in exactly the situations the meter gives up on.
    """
    if not tenants or not cards:
        return []
    idx = [c.index for c in cards]
    rows = []
    for t in tenants:
        rows.append({
            "label": t.label,
            "comm": t.comm,
            "pid": t.pid,
            "foreign": t.foreign,
            "total_gb": t.total_gb,
            # None, not 0.0: a card this process has registered with but allocated nothing on
            # is not the same claim as a card it is using lightly, and a column of dashes next
            # to a column of numbers says which card a tenant is actually on at a glance.
            "per_card": [(t.per_card_gb.get(i) or None) for i in idx],
        })
    return rows


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    snap = services.snapshot_models_dir()
    backends = await services.snapshot_llama_backends()
    # What downloaded last does not tell you anything about what the box is doing now, so the
    # list is gone from this page; only a download still in flight is worth a line.
    active = sum(1 for j in manager.snapshot() if j.status in ("queued", "downloading"))
    ctx = {"request": request, "snap": snap, "backends": backends,
           "ini_sections": ini.list_sections(),
           "active_downloads": active, "host_line": _host_line()}
    ctx.update(await _hero_context())
    ctx.update(_gpu_strip_context())
    ctx.update(_power_context())
    ctx.update(_power_settings_context())
    return templates.TemplateResponse("dashboard.html", ctx)


@app.get("/v2")
def dashboard_v2_redirect() -> RedirectResponse:
    """This lived at /v2 while it was being built; send stale tabs and bookmarks to the front."""
    return RedirectResponse("/", status_code=308)


@app.get("/hero", response_class=HTMLResponse)
async def hero_partial(request: Request) -> HTMLResponse:
    ctx = {"request": request}
    ctx.update(await _hero_context())
    return templates.TemplateResponse("_hero.html", ctx)


@app.get("/gpu-strip", response_class=HTMLResponse)
async def gpu_strip_partial(request: Request) -> HTMLResponse:
    ctx = {"request": request}
    ctx.update(_gpu_strip_context())
    return templates.TemplateResponse("_gpu_strip.html", ctx)


# ---------- Power and thermals ----------

# Defaults, overridable per box from the panel (stored in the kv table). The power ones are
# deliberately generic: only the owner knows what UPS is under the desk, and a number invented
# from the hardware would look authoritative while meaning nothing.
_POWER_DEFAULTS = {
    "power_warn_w": 700.0,
    "power_crit_w": 900.0,
    "power_baseline_w": 55.0,    # board, RAM, fans, drives, NIC - nothing here meters them
    "temp_warn_c": 85.0,
    "temp_crit_c": 95.0,
}


def _power_settings() -> dict:
    """Thresholds, with env as the first override and the kv table as the second."""
    out = dict(_POWER_DEFAULTS)
    for key in out:
        env = os.environ.get(key.upper())
        raw = db.get_setting(key, env or "")
        if raw:
            try:
                out[key] = float(raw)
            except ValueError:
                pass
    try:
        out["psu_efficiency"] = float(os.environ.get("POWER_PSU_EFFICIENCY", "0.90"))
    except ValueError:
        out["psu_efficiency"] = 0.90
    return out


def _power_context() -> dict:
    cfg = _power_settings()
    roll = hw.power_rollup(cfg["power_baseline_w"], cfg["psu_efficiency"])
    watts, temps = hw.power_history()
    # Pinned to the configured ceilings, so the height of each line is the headroom. Scaled to
    # its own window instead, a quiet box and a box about to trip the UPS would look identical.
    return {
        "p": roll,
        "power_warn_w": cfg["power_warn_w"], "power_crit_w": cfg["power_crit_w"],
        "temp_warn_c": cfg["temp_warn_c"], "temp_crit_c": cfg["temp_crit_c"],
        "power_spark": hw.sparkline(watts, cfg["power_crit_w"]) if len(watts) > 2 else "",
        "temp_spark": hw.sparkline(temps, cfg["temp_crit_c"]) if len(temps) > 2 else "",
    }


@app.get("/power", response_class=HTMLResponse)
def power_partial(request: Request) -> HTMLResponse:
    ctx = {"request": request}
    ctx.update(_power_context())
    return templates.TemplateResponse("_power.html", ctx)


def _power_settings_context(saved: bool = False) -> dict:
    cfg = _power_settings()
    return {"power_warn_w": cfg["power_warn_w"], "power_crit_w": cfg["power_crit_w"],
            "baseline_w": cfg["power_baseline_w"], "temp_warn_c": cfg["temp_warn_c"],
            "temp_crit_c": cfg["temp_crit_c"], "saved": saved, "reopen": saved}


@app.post("/power/settings", response_class=HTMLResponse)
async def power_settings_save(request: Request) -> HTMLResponse:
    """Save thresholds and re-render just this form.

    Returns the form rather than the readings panel: the readings are on a 1s poll and would
    replace whatever came back within the second anyway.
    """
    form = await request.form()
    mapping = {"power_warn_w": "power_warn_w", "power_crit_w": "power_crit_w",
               "baseline_w": "power_baseline_w", "temp_warn_c": "temp_warn_c",
               "temp_crit_c": "temp_crit_c"}
    for field, key in mapping.items():
        raw = str(form.get(field) or "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        # A limit of zero would divide by zero in the bar and make every reading "over limit".
        if key.endswith("_w") and value <= 0:
            continue
        db.set_setting(key, str(value))
    ctx = {"request": request}
    ctx.update(_power_settings_context(saved=True))
    return templates.TemplateResponse("_power_settings.html", ctx)


# ---------- Models directory ----------

async def _loaded_map() -> dict[str, list[str]]:
    """stem -> [backend_name, ...] for currently-loaded models."""
    backends = await services.snapshot_llama_backends()
    m: dict[str, list[str]] = {}
    for b in backends:
        if not b.loaded_model:
            continue
        for mid in [s.strip() for s in b.loaded_model.split(",")]:
            if mid:
                m.setdefault(mid, []).append(b.name)
    return m


def _update_status_map(snap) -> dict[str, dict]:
    """display_name -> what the last Hugging Face check says about this model.

    {'status': 'up-to-date'|'stale'|'unknown', 'reason': 'size'|'date'|'',
     'remote': 'YYYY-MM-DD', 'checked_at': ts, 'delta_days': int|None,
     'repo_id': str, 'parts': [(local rel path, repo path)]}

    Two joins had to be got right and neither was. The check records one row per DOWNLOADED
    file - a local relative path, per shard - while this map is keyed by the entry the list
    renders, whose display_name is the shard BASE and carries no subdirectory. So
    `checks.get(g.display_name)` matched nothing for any sharded model and nothing for any
    model stored in a subdirectory, which since the layout change is all of them: every model
    on the box scored "unknown" and no staleness was ever actually evaluated. The other join,
    local name against repo path, is fixed in hf._match_repo_path.

    The verdict itself is now size first. A commit date moves whenever anything in the repo's
    metadata is touched, so "the remote commit is newer than my file" is true of almost every
    model almost always and says nothing about the weights. A differing byte count is not
    ambiguous: the GGUF was rebuilt. Only when the size is unknown - an old row, or a repo the
    tree API would not describe - does this fall back to the date, and it says which test it
    used so the UI can be honest about how much the answer is worth.
    """
    from datetime import datetime, timezone
    checks = db.all_update_checks()

    def _parse(iso: str):
        try:
            core, _, frac = iso.partition(".")
            if frac:
                return datetime.fromisoformat(f"{core}.{frac.rstrip('Z')[:6]}+00:00")
            return datetime.fromisoformat(core.replace("Z", "+00:00"))
        except ValueError:
            return None

    out: dict[str, dict] = {}
    for g in snap.ggufs:
        # Every local path this entry covers, in the same form the download recorded it.
        # Companions included: an mmproj is folded into its model's row, so a projector that
        # was rebuilt has nowhere else to be reported and no other way to be refreshed.
        covered = list(g.parts) + list(g.companion_parts)
        rels = [f"{g.subdir}/{p.name}" if g.subdir else p.name for p in covered]
        rows = [(rel, checks[rel]) for rel in rels if rel in checks]
        if not rows:
            continue
        repo_id = rows[0][1].get("repo_id") or ""
        parts = [(rel, r.get("hf_path") or "") for rel, r in rows]
        checked_at = max(r.get("checked_at") or 0 for _rel, r in rows)

        # Size is per shard and decisive on its own: one shard of a different length means
        # the model was rebuilt, whatever the rest of them say.
        size_known = False
        size_differs = False
        by_name = {p.name: p for p in covered}
        for rel, r in rows:
            remote_size = r.get("hf_size")
            if remote_size is None:
                continue
            p = by_name.get(rel.rsplit("/", 1)[-1])
            if p is None:
                continue
            try:
                local_size = p.stat().st_size
            except OSError:
                continue
            size_known = True
            if int(remote_size) != local_size:
                size_differs = True

        remote_dts = [d for d in (_parse(r.get("hf_last_modified") or "") for _rel, r in rows) if d]
        newest = max(remote_dts) if remote_dts else None
        local_dt = datetime.fromtimestamp(g.mtime, tz=timezone.utc)
        delta_days = (newest - local_dt).days if newest else None
        remote_s = newest.strftime("%Y-%m-%d") if newest else ""

        if size_known:
            status, reason = ("stale", "size") if size_differs else ("up-to-date", "size")
        elif newest:
            status, reason = ("stale" if newest > local_dt else "up-to-date"), "date"
        else:
            status, reason = "unknown", ""

        out[g.display_name] = {
            "status": status, "reason": reason, "remote": remote_s,
            "checked_at": checked_at, "delta_days": delta_days,
            "repo_id": repo_id, "parts": parts,
        }
    return out


async def _models_avatar_map(snap) -> tuple[dict[str, str], dict[str, str]]:
    """(filename -> owner, owner -> avatar_url) for the files currently in the models dir."""
    owner_by_file = db.owner_by_filename()
    file_to_owner: dict[str, str] = {}
    for g in snap.ggufs:
        for p in g.parts:
            # look up by plain basename and by subdir-qualified name (for shards in subdirs)
            keys = [p.name]
            if g.subdir:
                keys.append(f"{g.subdir}/{p.name}")
            for k in keys:
                if k in owner_by_file:
                    file_to_owner[g.display_name] = owner_by_file[k]
                    break
            if g.display_name in file_to_owner:
                break
    avatars = await hf.owner_avatars(list(set(file_to_owner.values()))) if file_to_owner else {}
    return file_to_owner, avatars


def _owui_visibility() -> dict:
    """Per-connection model visibility for the Models page.

    Returns {"conns": [{host, url, prefix_id, explicit}], "visible": {model_id: {url: bool}}}.
    Only populated when there is more than one connection — with a single backend every
    model is on it and a toggle would be pure noise.
    """
    st = services.openwebui_state()
    conns = [c for c in (st.get("connections") or []) if not c.get("stale")]
    if not st.get("found"):
        return {"conns": [], "visible": {}, "caps": {}}

    # Capabilities are computed regardless of connection count. Unlike the visibility
    # toggles -- which are noise with a single backend, since everything is on it -- a wrong
    # vision flag is just as broken with one connection as with five.
    want = services.openwebui_capability_plan()
    have = services.openwebui_capability_state()
    mods = services.section_modalities()
    spec = services.sections_with_speculative()
    caps: dict[str, dict] = {}
    # Match the id back to its section by NAME, never by splitting on dots: OpenWebUI ids are
    # "<prefix>.<section>" but section names contain dots too (Qwen3.8-27B-Q4_K_M), so
    # rsplit(".", 1) yields "8-27B-Q4_K_M" and silently matches nothing.
    _sections = set(ini.section_names())
    for mid, should in want.items():
        section = next((n for n in _sections if mid == n or mid.endswith("." + n)), None)
        if section is None:
            continue
        row = caps.setdefault(section, {"should": should, "current": [], "mismatch": False,
                                        "modalities": mods.get(section, []),
                                        "speculative": spec.get(section, "")})
        cur = have.get(mid)
        row["current"].append((mid, cur))
        if cur != should:
            row["mismatch"] = True

    if len(conns) < 2:
        return {"conns": [], "visible": {}, "caps": caps}
    ids = sorted(ini.section_names())
    visible: dict[str, dict] = {}
    for mid in ids:
        row: dict[str, bool] = {}
        for c in conns:
            filt = c.get("model_ids") or []
            row[c["url"]] = (mid in filt) if filt else True  # empty filter = offers everything
        visible[mid] = row
    return {
        "conns": [{"host": c["host"], "url": c["url"], "prefix_id": c.get("prefix_id") or "",
                   "explicit": bool(c.get("model_ids"))} for c in conns],
        "visible": visible,
        "caps": caps,
    }


async def _models_list_ctx(request: Request, snap, flash=None) -> dict:
    """Everything _models_list.html needs, built in ONE place.

    The partial is rendered from four routes - the page, two deletes and the update check -
    and the three POST routes each assembled their own context by hand. They were missing
    `owui`, `badges` and `shapes`, so deleting one file silently stripped the shape chips, the
    per-connection availability toggles and every rating off every OTHER row until the next
    full reload. A partial swapped into a page has to come back with the same shape it had.
    """
    file_to_owner, avatars = await _models_avatar_map(snap)
    loaded_map = await _loaded_map()
    update_status = _update_status_map(snap)
    owui = _owui_visibility()
    shapes = _shapes_for_files(snap)
    # What a section can DO is derived from its own config - GGUF header, its mmproj, its spec
    # profile - by the same helper the models.ini cards use. OpenWebUI's record is a copy of
    # that and can lag or have never seen a section at all, which is exactly what happened to
    # voice-cpu: its file showed image and audio, its own row showed an unreadable projector.
    try:
        section_caps = _section_caps(ini.list_sections())
    except Exception:  # noqa: BLE001 - an unreadable ini must not take the models page down
        section_caps = {}
    return {
        "request": request, "snap": snap, "flash": flash,
        "file_to_owner": file_to_owner, "avatars": avatars,
        "owui": owui,
        "rows": _model_rows(snap, shapes, loaded_map, update_status, owui,
                            db.badges_by_alias(), section_caps),
    }


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request) -> HTMLResponse:
    snap = services.snapshot_models_dir()
    return templates.TemplateResponse(
        "models.html", await _models_list_ctx(request, snap))


def _model_rows(snap, shapes: dict, loaded_map: dict, update_status: dict,
                owui: dict, badges_by_alias: dict, section_caps: dict | None = None) -> list[dict]:
    """One row per SERVED MODEL, not per file, plus the facets the list groups and filters on.

    A file and a model are not the same thing. gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf backs two
    models.ini sections - its own and `voice-cpu` - with different settings over the same
    weights, and `voice-qwen` is the only name Qwen3.5-4B is served under at all. Keyed by
    file, those meta-models were a comma-separated footnote on somebody else's row: no state
    of their own, no ratings of their own, nowhere to click through to the section that
    defines them. They are the thing you actually call from OpenWebUI, so they get a row.

    What belongs to the FILE stays on the file's row and appears once: the byte count, the
    checkbox, delete. A second section over the same weights frees nothing when you remove it
    and must not double the group totals, so it carries `primary: False`, no size of its own
    and no delete button - its sort key is still the file's size, so the two sort together
    rather than the alias sinking to the bottom of a size sort.

    What belongs to the SECTION is per row and is why the split is worth making: whether it is
    resident right now, whether speculation is configured for it, which backends offer it, and
    what you have rated it. All four differ between two sections over one file.

    Family and tier come from the file either way - same weights, same placement - so an alias
    sorts into the same group as the file it is built on rather than off under its own name.
    """
    card_gb = services.largest_card_gb()
    caps = (owui or {}).get("caps") or {}
    split = {g.display_name: services.split_quant(g.stem) for g in snap.ggufs}
    # Needs the whole set at once: whether a variant has a base to fold onto is a fact about
    # what else is on disk, not about its own filename.
    canon = services.canonical_families({fam for fam, _q in split.values()})
    rows: list[dict] = []
    for g in snap.ggufs:
        family, quant = split[g.display_name]
        family = canon.get(family, family)
        tier, tier_label, tier_short = services.size_tier(g.total_bytes, card_gb)
        shape = (shapes or {}).get(g.display_name)
        us = (update_status or {}).get(g.display_name)

        # The section named after the file leads, so the row that owns the byte count is the
        # one whose name matches it; a file served only under aliases has no such row and the
        # first alias owns it instead.
        ids = sorted(g.aliases or [], key=lambda a: (a != g.stem, a.lower()))
        if not ids:
            ids = [g.stem]

        for n, served_id in enumerate(ids):
            cap = caps.get(served_id) or {}
            # The section's own config wins over OpenWebUI's copy of it, and is the only source
            # that exists for a section OpenWebUI has never been told about. The mismatch flag
            # still comes from OpenWebUI - disagreement between the two is the thing it reports.
            sect = (section_caps or {}).get(served_id) or {}
            mods = sect.get("mods") if sect else (cap.get("modalities") or [])
            mods = list(mods or [])
            spec = sect.get("spec") if sect else ("head" if cap.get("speculative") else "")
            proj_unknown = sect.get("proj_unknown") if sect else bool(g.companion_name and not mods)
            loaded = (loaded_map or {}).get(served_id) or []
            configured = bool(g.aliases)

            if g.is_companion:
                status, status_label = "orphan", "Orphan companion"
            elif loaded:
                status, status_label = "loaded", "Loaded now"
            elif configured:
                status, status_label = "configured", "In models.ini"
            else:
                status, status_label = "unconfigured", "Not configured"

            flags = []
            if shape and shape.is_moe:
                flags.append("moe")
            elif shape:
                flags.append("dense")
            if "vision" in mods:
                flags.append("vision")
            if "audio" in mods:
                flags.append("audio")
            if mods or g.companion_name:
                flags.append("multimodal")
            if spec:
                flags.append("draft")
            if (us or {}).get("status") == "stale":
                flags.append("update")
            if served_id != g.stem:
                flags.append("alias")
            flags.append(status)

            rows.append({
                "g": g,
                "id": served_id,
                # True when this name is not the filename's: a model that exists only in
                # models.ini, which is what gets the headline instead of the filename.
                "is_alias": served_id != g.stem,
                "primary": n == 0,
                "configured": configured,
                "siblings": [i for i in ids if i != served_id],
                "shape": shape,
                "mods": mods,
                "spec": spec,
                "proj_unknown": proj_unknown,
                "head_idle": bool(sect.get("head_idle")),
                "cap": cap,
                "loaded": loaded,
                # NOT named "update": Jinja resolves r.update on a dict to the dict's own
                # update METHOD, which is truthy, so `{% if r.update %}` fired on every row
                # and painted an amber "update" chip across the whole list while the tooltip
                # it filled in read "dated  ( days newer)". Anything colliding with a dict
                # method - update, items, keys, values, get, copy, pop - must not be a key here.
                "stale": us if (us or {}).get("status") == "stale" else None,
                "badges": (badges_by_alias or {}).get(served_id) or [],
                "family": family, "quant": quant,
                "tier": tier, "tier_label": tier_label, "tier_short": tier_short,
                "status": status, "status_label": status_label,
                "flags": " ".join(flags),
                "card_gb": card_gb,
                # Sort by the file's size on every row so an alias sorts beside its file;
                # count only the owning row, so group totals stay the bytes on disk.
                "sort_bytes": g.total_bytes,
                "count_bytes": g.total_bytes if n == 0 else 0,
                "search": " ".join([g.display_name, served_id, family, quant,
                                    (shape.arch if shape else ""), " ".join(g.aliases or [])]).lower(),
            })
    return rows


def _shapes_for_files(snap) -> dict:
    """{display_name: ModelShape} for the models list.

    Companions are skipped: an mmproj declares itself as `clip`, which would render a
    "dense" chip on a projector that has no layers to offload in the first place.
    """
    out: dict = {}
    for g in snap.ggufs:
        if g.is_companion or not g.parts:
            continue
        shape = services.model_shape(g.parts[0])
        if shape.known:
            out[g.display_name] = shape
    return out



@app.get("/model/{filename:path}", response_class=HTMLResponse)
async def model_local_detail(request: Request, filename: str) -> HTMLResponse:
    from datetime import datetime, timezone
    from fastapi import HTTPException

    if ".." in filename or filename.startswith("/") or filename.count("/") > 1:
        raise HTTPException(status_code=400, detail="bad filename")

    # Resolve: filename may be "flat.gguf" OR "subdir/flat.gguf" OR "subdir" (routes to first shard).
    snap = services.snapshot_models_dir()
    entry = None
    if "/" in filename:
        sub, base = filename.split("/", 1)
        entry = next((g for g in snap.ggufs if g.subdir == sub and g.display_name == base), None)
    else:
        entry = next((g for g in snap.ggufs if g.display_name == filename and g.subdir == ""), None)
        if entry is None:
            # try as a subdir name pointing at the group inside
            entry = next((g for g in snap.ggufs if g.subdir == filename), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="not found")

    path = entry.parts[0]  # first shard is where metadata lives
    st = path.stat()
    raw = gguf_meta.read_raw(path)
    summary = gguf_meta.summarize(raw)
    stem = entry.stem
    filename = entry.route_key

    owner_by_file = db.owner_by_filename()
    # try subdir-qualified name first, then plain basename (for flat downloads)
    owner = owner_by_file.get(filename, "") or owner_by_file.get(entry.parts[0].name, "")
    avatars = await hf.owner_avatars([owner]) if owner else {}
    avatar_url = avatars.get(owner, "") if owner else ""

    # The section that owns this file may be named something else entirely (renamed to give
    # the model a short API id), so resolve file -> section rather than assuming stem == id.
    owning = ini.sections_by_file().get(entry.first_shard_rel) or []
    model_id = stem if stem in owning else (owning[0] if owning else stem)

    lm = await _loaded_map()
    loaded_on = lm.get(model_id, [])

    in_ini = model_id in ini.section_names()
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

    return templates.TemplateResponse("model_local.html", {
        "request": request,
        "filename": filename,
        "path": str(path),
        "stem": stem,
        "model_id": model_id,
        "size_h": human_bytes(st.st_size),
        "mtime": mtime,
        "raw": raw,
        "summary": summary,
        "owner": owner,
        "avatar_url": avatar_url,
        "loaded_on": loaded_on,
        "in_ini": in_ini,
        "badges": db.badges_for(model_id),
    })


@app.post("/models/delete", response_class=HTMLResponse)
async def models_delete(request: Request, name: str = Form(...)) -> HTMLResponse:
    ok, msg, freed = services.delete_gguf(name)
    # Prune here catches ids orphaned EARLIER -- it cannot catch the model just deleted.
    # delete_gguf removes files, not the models.ini section, so at this point the section is
    # still there and the id still resolves; `unknown_ids` means "no section provides this".
    # What makes an id dead is deleting its section, and that path prunes for itself.
    # A no-op (and no restart) when nothing is actually stale.
    if ok:
        try:
            pruned = services.prune_openwebui_unknown_ids()
            if pruned:
                msg = f"{msg}; removed {len(pruned)} stale id(s) from OpenWebUI"
            services.sync_openwebui_capabilities()
        except Exception:  # noqa: BLE001 -- deletion must succeed even if OpenWebUI is down
            pass
    snap = services.snapshot_models_dir()
    flash = {"ok": ok, "msg": msg, "freed_h": human_bytes(freed) if freed else None}
    return templates.TemplateResponse(
        "_models_list.html", await _models_list_ctx(request, snap, flash))


@app.post("/models/delete-bulk", response_class=HTMLResponse)
async def models_delete_bulk(request: Request) -> HTMLResponse:
    form = await request.form()
    names = form.getlist("names") if hasattr(form, "getlist") else form.get("names") or []
    if isinstance(names, str):
        names = [names]
    total_freed = 0
    ok_count = 0
    errors: list[str] = []
    for n in names:
        ok, msg, freed = services.delete_gguf(str(n))
        if ok:
            ok_count += 1
            total_freed += freed
        else:
            errors.append(f"{n}: {msg}")
    # One prune after the whole batch, not per file -- each whitelist write restarts
    # open-webui, so doing it inside the loop would restart it once per deleted model.
    if ok_count:
        try:
            services.prune_openwebui_unknown_ids()
        except Exception:  # noqa: BLE001
            pass
    snap = services.snapshot_models_dir()
    if errors:
        flash = {"ok": False, "msg": f"deleted {ok_count}, {len(errors)} failed: " + "; ".join(errors[:3])}
    else:
        flash = {"ok": True, "msg": f"deleted {ok_count} file(s)", "freed_h": human_bytes(total_freed) if total_freed else None}
    return templates.TemplateResponse(
        "_models_list.html", await _models_list_ctx(request, snap, flash))


@app.post("/models/check-updates", response_class=HTMLResponse)
async def models_check_updates(request: Request) -> HTMLResponse:
    records = db.download_records()
    if records:
        await hf.check_updates_for(records)
    snap = services.snapshot_models_dir()
    flash = {"ok": True, "msg": f"checked {len(records)} record(s) against HF"}
    return templates.TemplateResponse(
        "_models_list.html", await _models_list_ctx(request, snap, flash))


# ---------- Settings ----------

@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("settings.html", {"request": request, "token": db.get_setting("hf_token", ""), "flash": None})


@app.post("/settings", response_class=HTMLResponse)
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


# ---------- Search ----------

@app.get("/search", response_class=HTMLResponse)
def search_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("search.html", {"request": request, "query": ""})


@app.get("/search/results", response_class=HTMLResponse)
async def search_results(request: Request, q: str = "", sort: str = "downloads",
                         limit: int = 30) -> HTMLResponse:
    error: str | None = None
    results = []
    avatars: dict[str, str] = {}
    browse_mode = not q.strip()  # empty query = "show me trending" browse view
    try:
        # Empty query is now valid — HF returns the top-N gguf models under `sort`
        results = await hf.search_models(q.strip(), sort=sort, limit=limit)
        owners = [(m.id.split("/", 1)[0] if "/" in m.id else m.id) for m in results]
        avatars = await hf.owner_avatars(owners)
    except httpx.HTTPStatusError as e:
        error = f"HF returned HTTP {e.response.status_code}"
    except httpx.HTTPError as e:
        error = f"network error: {e}"
    # Cross-check against local downloads so we can flag repos the user already has.
    # Must be checked against the FILESYSTEM, not just the history log — see the helper.
    downloaded = _downloaded_and_still_present()
    return templates.TemplateResponse("_search_results.html", {
        "request": request, "results": results, "error": error, "avatars": avatars,
        "browse_mode": browse_mode, "browse_sort": sort,
        "downloaded_by_repo": downloaded,
    })


def _files_present_on_disk() -> set[str]:
    """Every GGUF currently in the models directory, keyed the way download_history stores it.

    History rows use two shapes: bare "model.gguf" for downloads that predate the per-model
    subdirectory layout, and "stem/model.gguf" for everything since. Both are indexed so a
    caller can match either.
    """
    names: set[str] = set()
    try:
        snap = services.snapshot_models_dir()
    except OSError:
        return names
    for g in snap.ggufs:
        for part in list(g.parts) + list(g.companion_parts):
            names.add(part.name)
            if g.subdir:
                names.add(f"{g.subdir}/{part.name}")
    return names


def _downloaded_and_still_present() -> dict[str, list[str]]:
    """{repo_id: [filename, ...]} for repos whose files are ACTUALLY on disk right now.

    download_history is a log, not an inventory: a row stays `done` forever, so a model that
    was downloaded and later deleted keeps being reported as owned. Search then tells you that
    you already have something you do not, which is exactly backwards from the point of the
    flag — it exists to stop you re-downloading, and a false positive stops you downloading
    at all. Intersect the log with the filesystem.
    """
    present = _files_present_on_disk()
    out: dict[str, list[str]] = {}
    for repo, files in db.downloaded_files_by_repo().items():
        kept = [f for f in files if f in present or f.rsplit("/", 1)[-1] in present]
        if kept:
            out[repo] = kept
    return out


@app.get("/search/repo/{repo_id:path}", response_class=HTMLResponse)
async def search_repo(request: Request, repo_id: str) -> HTMLResponse:
    error: str | None = None
    gated: str = ""
    foreign: str = ""
    groups: list[dict] = []
    try:
        detail = await hf.repo_detail(repo_id)
        # group by shard_base
        by_base: dict[str, list[hf.HfFile]] = {}
        for f in detail.files:
            by_base.setdefault(f.shard_base, []).append(f)
        for base, files in by_base.items():
            files.sort(key=lambda x: (x.shard_index or 0, x.path))
            shard_total = files[0].shard_total if files[0].shard_index else None
            total_size = sum(x.size for x in files)
            groups.append({
                "shard_base": base,
                "shard_total": shard_total,
                "total_size_h": human_bytes(total_size),
                "total_size": total_size,
                "quant": next((x.quant for x in files if x.quant), None),
                "fit_chips": services.vram_fit_chips(total_size),
                "files": [{"path": x.path, "size": x.size, "size_h": human_bytes(x.size), "quant": x.quant} for x in files],
            })
        groups.sort(key=lambda g: (0 if g["files"][0]["path"].lower().endswith(".gguf") else 1, g["shard_base"].lower()))

        # Real context estimates need the model's layer/head counts, which only exist in the
        # GGUF header — so range-fetch ONE header for the repo. Those fields are properties of
        # the model and identical across its quants, so a single fetch (~1 MB) covers every
        # group; only file size varies, and we already know that per group.
        probe = next(
            (g for g in groups
             if g["files"][0]["path"].lower().endswith(".gguf")
             and "mmproj" not in g["files"][0]["path"].lower()),
            None,
        )
        # A repo's projector ships alongside its quants; it will be loaded with the model,
        # so its VRAM has to come off the budget or every multimodal estimate reads high.
        # Pick the smallest, matching what _find_mmproj does locally.
        mmproj_sizes = [
            g["total_size"] for g in groups
            if "mmproj" in g["files"][0]["path"].lower()
        ]
        mmproj_gb = (min(mmproj_sizes) / (1024 ** 3)) if mmproj_sizes else 0.0

        if probe:
            summary = await hf.gguf_header(repo_id, probe["files"][0]["path"])
            # A gated repo lists its files publicly but refuses the weights, so estimates
            # would silently vanish with no explanation. Surface the reason instead.
            gated = hf.gated_reason(repo_id) if not summary else ""
            # Warn BEFORE the download, not after: the header already tells us llama.cpp
            # can't load this, and without this the estimates just silently didn't appear.
            foreign = autoconfig.foreign_gguf_reason(summary)
            if summary and not foreign:
                for g in groups:
                    first = g["files"][0]["path"].lower()
                    if not first.endswith(".gguf") or "mmproj" in first:
                        continue
                    g["estimates"] = _preset_estimates(summary, g["total_size"], mmproj_gb)
                    g["native_ctx"] = (summary.get("model") or {}).get("context_length") or 0
    except httpx.HTTPStatusError as e:
        error = f"HF returned HTTP {e.response.status_code} for {repo_id}"
    except httpx.HTTPError as e:
        error = f"network error: {e}"
    return templates.TemplateResponse("_repo_files.html", {"request": request, "repo_id": repo_id, "groups": groups, "error": error, "gated": gated, "foreign": foreign})


# ---------- Downloads ----------

_QUEUED_CHIP = (
    '<a href="/downloads" title="{title}" '
    'class="shrink-0 inline-flex items-center gap-1 rounded-md bg-emerald-100 dark:bg-emerald-950 '
    'text-emerald-800 dark:text-emerald-300 px-3 py-1.5 text-xs font-medium">'
    '✓ {label}</a>'
)


def _preset_estimates(summary: dict, size_bytes: int, mmproj_gb: float = 0.0) -> list[dict]:
    """Fast / Balanced / Long-ctx context estimates for a model we have NOT downloaded.

    Runs the very same autoconfig fit math used on local models, so a search-page estimate
    and the eventual Config recommendation agree instead of being two different guesses.
    Returns [] when there is nothing meaningful to show (no GPU backend, or the GGUF header
    lacks the fields needed to size a KV cache).
    """
    backends = []
    for name, vram in services._fit_backends().items():
        backends.append({
            "name": name, "vendor": "cuda", "vram_gb": vram,
            "gpu_count": hw.gpu_count_for(name), "card_vram_gb": hw.card_vram_gb_for(name),
            "host_ram_gb": hw.host_ram_gb(),
            "baseline": {},
        })
    if not backends or not summary:
        return []
    out: list[dict] = []
    for key, label in (("fast", "Fast"), ("balanced", "Balanced"), ("long-ctx", "Long ctx")):
        try:
            rec = autoconfig.analyze(
                summary=summary, file_size=size_bytes, backends=backends,
                preset=key, models_dir=None, section_name="",
                mmproj_gb_override=(mmproj_gb or None),
            )
        except Exception:  # noqa: BLE001 — an estimate must never break the search page
            return []
        if rec.error or not rec.recommended_ctx:
            continue
        chosen = next((p for p in rec.presets if p.key == rec.active_preset), None)
        out.append({
            "key": key,
            "label": label,
            "ctx": rec.recommended_ctx,
            "ctx_h": autoconfig.format_ctx(rec.recommended_ctx),
            "gpu_layers": chosen.gpu_layers if chosen else 0,
            "total_layers": chosen.total_layers if chosen else 0,
            "speed_pct": round((chosen.speed_score if chosen else 1.0) * 100),
            "offload": bool(chosen and chosen.offload_kind),
        })
        # Collapse duplicates: a model that fits fully at native has one real answer, and
        # three identical chips would imply choices that don't exist.
        if len(out) > 1 and out[-1]["ctx"] == out[0]["ctx"] and not out[-1]["offload"]:
            out.pop()
    return out


def _model_stem(filename: str) -> str:
    """Basename minus .gguf extension. Used as the subdir name for one-dir-per-model layout."""
    b = Path(filename).name
    return b[:-5] if b.lower().endswith(".gguf") else b


def _dest_for_main(main_filename: str) -> tuple[str, str]:
    """(subdir, dest_filename) for a MAIN model file. Subdir = filename stem."""
    stem = _model_stem(main_filename)
    return stem, f"{stem}/{Path(main_filename).name}"


def _dest_for_companion(main_stem: str, companion_filename: str) -> str:
    """Place a companion file (mmproj, tokenizer.model, chat_template) inside the main model's subdir."""
    return f"{main_stem}/{Path(companion_filename).name}"


def _blocked_chip(reason: str) -> HTMLResponse:
    """What a refused download button turns into. The reason rides in the tooltip."""
    return HTMLResponse(
        '<span class="shrink-0 inline-flex items-center rounded-md bg-red-100 dark:bg-red-950 '
        'text-red-800 dark:text-red-300 px-3 py-1.5 text-xs font-medium" '
        f'title="{html.escape(reason)}">Blocked: llama.cpp can&rsquo;t load this</span>')


async def _foreign_repo(repo_id: str, main_path: str) -> str:
    """Why a download from this repo is refused, or "". Checked server-side as well as hidden
    in the page, so a stale page or a hand-made request can't queue a file llama.cpp won't load.
    An unreadable header (network, gated) is not a verdict and never blocks."""
    return autoconfig.foreign_gguf_reason(await hf.gguf_header(repo_id, main_path))


@app.post("/download", response_class=HTMLResponse)
async def download_single(repo_id: str = Form(...), path: str = Form(...), size: int = Form(0)) -> HTMLResponse:
    base = Path(path).name
    if base.lower().endswith(".gguf") and "mmproj" not in base.lower():
        reason = await _foreign_repo(repo_id, path)
        if reason:
            return _blocked_chip(reason)
    if "mmproj" in base.lower():
        # Standalone mmproj download: put in a subdir named after its own stem
        stem = _model_stem(base)
        filename = f"{stem}/{base}"
        manager.enqueue(repo_id=repo_id, hf_path=path, filename=filename, total_bytes=size)
        return HTMLResponse(_QUEUED_CHIP.format(label="Queued", title=f"Queued {filename} — see Downloads"))

    # Main GGUF: subdir = stem. Companions from the same repo are queued alongside it.
    main_stem, filename = _dest_for_main(base)
    manager.enqueue(repo_id=repo_id, hf_path=path, filename=filename, total_bytes=size)
    extras = 0
    try:
        detail = await hf.repo_detail(repo_id)
        for f in detail.files:
            fp_name = Path(f.path).name
            if "mmproj" in fp_name.lower():
                manager.enqueue(
                    repo_id=repo_id,
                    hf_path=f.path,
                    filename=_dest_for_companion(main_stem, fp_name),
                    total_bytes=f.size,
                )
                extras += 1
                if extras >= 4:
                    break  # cap: some repos have many mmproj variants

        # A speculative-decoding head, when the repo ships one for this model. Only the
        # SMALLEST is taken: repos commonly publish BF16/F16/Q8_0/Q4_0 of the same head, they
        # are 60-170 MB, they sit in VRAM all session, and draft quality below Q8 barely moves
        # the acceptance rate. Queueing all four would waste bandwidth and disk to no purpose.
        #
        # These used to be skipped entirely after community draft models segfaulted
        # llama-server. That was a generic draft paired with an unrelated model; a head shipped
        # in the model's own repo is trained against it, and llama.cpp drives it through
        # draft-mtp rather than draft-simple. Autoconfig still only PROPOSES enabling it.
        heads = [f for f in detail.files
                 if f.path.lower().endswith(".gguf")
                 and "mmproj" not in Path(f.path).name.lower()
                 and autoconfig._looks_like_draft(Path(f.path).name)]
        if heads:
            head = min(heads, key=lambda f: f.size or 0)
            manager.enqueue(
                repo_id=repo_id,
                hf_path=head.path,
                filename=_dest_for_companion(main_stem, Path(head.path).name),
                total_bytes=head.size,
            )
            extras += 1
    except httpx.HTTPError:
        pass  # non-fatal — companions can be fetched manually later

    label = "Queued" if not extras else f"Queued (+ {extras} companion)"
    return HTMLResponse(_QUEUED_CHIP.format(label=label, title=f"Queued {filename}" + (f" and {extras} companion file(s)" if extras else "")))


@app.post("/download/multi", response_class=HTMLResponse)
async def download_multi(repo_id: str = Form(...), shard_base: str = Form(...)) -> HTMLResponse:
    try:
        detail = await hf.repo_detail(repo_id)
    except httpx.HTTPError as e:
        return HTMLResponse(
            f'<span class="shrink-0 inline-flex items-center rounded-md bg-red-100 dark:bg-red-950 '
            f'text-red-800 dark:text-red-300 px-3 py-1.5 text-xs font-medium" title="{e}">HF error</span>'
        )
    shards = sorted((f for f in detail.files if f.shard_base == shard_base),
                    key=lambda f: (f.shard_index or 0, f.path))
    if shards:
        reason = await _foreign_repo(repo_id, shards[0].path)
        if reason:
            return _blocked_chip(reason)
    # All shards + companion mmproj go into the same subdir named after the shard base.
    subdir = Path(shard_base).stem  # strip .gguf
    count = 0
    mmproj_count = 0
    for f in detail.files:
        if f.shard_base == shard_base:
            base = Path(f.path).name
            manager.enqueue(repo_id=repo_id, hf_path=f.path, filename=f"{subdir}/{base}", total_bytes=f.size)
            count += 1
    # Auto-queue any mmproj files in the same repo into the same subdir
    for f in detail.files:
        fp_name = Path(f.path).name
        if "mmproj" in fp_name.lower():
            manager.enqueue(repo_id=repo_id, hf_path=f.path, filename=_dest_for_companion(subdir, fp_name), total_bytes=f.size)
            mmproj_count += 1
            if mmproj_count >= 4:
                break
    label = f"Queued {count} shards"
    if mmproj_count:
        label += f" + {mmproj_count} mmproj"
    return HTMLResponse(_QUEUED_CHIP.format(label=label, title=f"Queued {count} shards + {mmproj_count} companion files from {repo_id} into subdir '{subdir}/'"))


@app.post("/download/url", response_class=HTMLResponse)
async def download_url(url: str = Form(...), filename: str = Form("")) -> HTMLResponse:
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return HTMLResponse('<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">URL must start with http:// or https://</div>')
    if not filename.strip():
        from urllib.parse import urlparse, unquote
        p = urlparse(url).path
        filename = unquote(p.rsplit("/", 1)[-1]) if p else ""
    filename = filename.strip()
    if not filename:
        return HTMLResponse('<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">could not derive filename — set one explicitly</div>')
    if "/" in filename or ".." in filename:
        return HTMLResponse('<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">filename must be a plain basename</div>')

    if filename.lower().endswith(".gguf") and "mmproj" not in filename.lower():
        reason = autoconfig.foreign_gguf_reason(await hf.url_gguf_header(url))
        if reason:
            return HTMLResponse(
                '<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">'
                f'Not queued. {html.escape(reason)}</div>')

    total = 0
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            r = await client.head(url)
            if r.status_code < 400:
                total = int(r.headers.get("content-length") or 0)
    except httpx.HTTPError:
        pass  # size will be filled in during download

    # URL imports also get their own subdir (per-filename-stem, so main + optional mmproj coexist if named right)
    stem = _model_stem(filename)
    final_filename = f"{stem}/{filename}"
    manager.enqueue_url(url=url, filename=final_filename, total_bytes=total)
    return HTMLResponse(
        f'<div class="rounded-md bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 px-3 py-2 text-sm">'
        f'Queued <span class="font-mono">{filename}</span>. See progress below.</div>'
    )


@app.get("/downloads", response_class=HTMLResponse)
def downloads_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("downloads.html", {"request": request, "jobs": manager.snapshot()})


@app.get("/downloads/rows", response_class=HTMLResponse)
def downloads_rows(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("_download_rows.html", {"request": request, "jobs": manager.snapshot()})


@app.post("/downloads/{job_id}/cancel")
def downloads_cancel(job_id: str) -> Response:
    manager.cancel(job_id)
    return Response(status_code=204)


@app.post("/downloads/clear", response_class=HTMLResponse)
def downloads_clear(request: Request) -> HTMLResponse:
    manager.clear_finished()
    return templates.TemplateResponse("_download_rows.html", {"request": request, "jobs": manager.snapshot()})


# ---------- models.ini ----------

def _section_caps(sections: list) -> dict[str, dict]:
    """{section name: capability facts} for the chips on the section cards.

    Everything is derived locally — shape from the GGUF header, modalities from the section's
    own projector, speculation from the saved profile plus the heads that exist — so the ini
    page says what the CONFIGured model can do even when OpenWebUI (whose records drive the
    models list) has never seen it. Header reads are the cached kind; this is cheap per render.
    """
    caps: dict[str, dict] = {}
    for s in sections:
        if not s.matched_file:
            continue
        vals = dict(s.items)
        shape = services.model_shape(settings.models_dir / s.matched_file)
        mmproj = (vals.get("mmproj") or "").strip()
        mods = services.projector_modalities(mmproj) if mmproj else frozenset()
        subdir = s.matched_file.split("/", 1)[0] if "/" in s.matched_file else ""
        # A draft head counts when it is named by the section, sits beside these weights, or
        # is baked into them (MTP layers) - the same three ways autoconfig resolves one.
        head_rel = (vals.get("spec-draft-model") or "").strip()
        head_ext = bool(head_rel) or bool(autoconfig._find_mtp(settings.models_dir, s.name, subdir))
        head_avail = head_ext or shape.internal_mtp
        prof_key = autoconfig.match_spec_profile(vals)
        prof = autoconfig.SPEC_PROFILE_BY_KEY.get(prof_key)
        if prof_key == "custom":
            drafts = bool((vals.get("spec-type") or "").strip())
        else:
            drafts = bool(prof and prof.spec_type and prof.needs_head)
        spec_on = head_avail and drafts
        mtp_mode = shape.internal_mtp or bool(prof and "mtp" in (prof.spec_type or ""))
        caps[s.name] = {
            "shape": shape if shape.known else None,
            "mods": mods,
            "proj_unknown": bool(mmproj) and not mods,
            "spec": ("mtp" if mtp_mode else "head") if spec_on else "",
            "head_idle": head_avail and not spec_on,
        }
    return caps


def _config_context(request: Request, flash: dict | None = None) -> dict:
    sections = ini.list_sections()
    # Client snippets need the address the browser used to reach us (not the docker-internal
    # host the prober uses), so snippets work from any LAN machine. Compute endpoints once.
    browser_host = request.url.hostname or ""
    endpoints = services.browser_endpoints(browser_host)
    for s in sections:
        s.client_snippets = [
            {"name": ep["name"], "base_url": ep["base_url"], "api_key": ep["api_key"],
             **ini.to_client_config(s.name, ep["base_url"], ep["api_key"])}
            for ep in endpoints
        ]
    return {
        "request": request,
        "sections": sections,
        "caps": _section_caps(sections),
        "unregistered": ini.unregistered_gguf_stems(),
        "raw": ini.raw_text(),
        "backups": ini.list_backups(),
        "ini_path": str(settings.models_ini_path),
        "llama_backends": services._effective_container_names(),
        "flash": flash,
    }


@app.get("/config", response_class=HTMLResponse)
def config_page(request: Request, saved: str = "", deleted: str = "", err: str = "", edit: str = "") -> HTMLResponse:
    flash = None
    if saved:
        flash = {"ok": True, "msg": f"Saved section [{saved}]."}
    elif deleted:
        flash = {"ok": True, "msg": f"Deleted section [{deleted}]."}
    elif err:
        flash = {"ok": False, "msg": err}
    ctx = _config_context(request, flash)
    # ?edit=<section> makes the page open straight into that section's form, which is how
    # direct links from elsewhere (e.g. a model's "edit models.ini") arrive here.
    ctx["edit_section"] = edit if (edit and edit in ini.section_names()) else ""
    return templates.TemplateResponse("config.html", ctx)


@app.get("/config/section/new", response_class=HTMLResponse)
def config_section_new(request: Request, name: str = "") -> HTMLResponse:
    warn = None
    if not name:
        warn = "Pick a filename first."
    elif not ini.valid_section_name(name):
        warn = f"Invalid section name: {name!r}"
    elif name in ini.section_names():
        warn = f"Section [{name}] already exists — use Edit instead."

    values: dict[str, str] = {}
    hints: list[str] = []
    if not warn:
        vals2, hints2 = _gguf_hints_for(name)
        values, hints = vals2, hints2

    return templates.TemplateResponse("_section_form.html", {
        "request": request,
        "mode": "new",
        "section_name": name,
        "values": values,
        "extras_text": "",
        "form_tiers": ini.FORM_TIERS,
        "warning": warn,
        "smart_hints": hints,
        "auto_filled": bool(values),
    })


def _pick_main_gguf(files: list[Path], section: str) -> Path:
    """Which of a subdir's GGUFs is THIS section's model. Companions are already filtered out.

    Not the first alphabetically. A folder routinely holds several quants of one model, and the
    section name is what says which: `[...-Q8_0]` wants the Q8_0 file, not whichever sorts first.
    Shards are grouped before choosing so the answer is always part 1, the shard carrying the
    metadata and the only one llama-server should be handed.
    """
    groups: dict[str, list[tuple[int, Path]]] = {}
    for q in files:
        base, idx, _total = shard_key(q.name)
        stem = base[:-5] if base.lower().endswith(".gguf") else base
        groups.setdefault(stem, []).append((idx or 1, q))

    def first_shard(stem: str) -> Path:
        return min(groups[stem], key=lambda t: t[0])[1]

    low = section.lower()
    for stem in groups:
        if stem.lower() == low:
            return first_shard(stem)
    for stem in groups:
        if low.startswith(stem.lower()) or stem.lower().startswith(low):
            return first_shard(stem)
    # The section name matches nothing here, so fall back to the largest model present rather
    # than to sort order. Size is a far better guess at "the main one" than the alphabet.
    biggest = max(groups, key=lambda s: sum(
        autoconfig.file_size_or_none(q) or 0 for _i, q in groups[s]))
    return first_shard(biggest)


def _resolve_section_gguf(name: str) -> tuple[Path | None, str, str | None]:
    """(gguf_path, model_rel, rel) for a section. rel is models-dir-relative, model_rel is
    the container-absolute /models/... path llama-server wants, or "" for flat layouts.

    Resolution order matters. An explicit `model =` wins over deriving the file from the
    section name, because (a) a section can be renamed to give the model a short API id, and
    (b) a subdir can hold several quants, where re-deriving would silently pick whichever
    sorts first rather than the one this section is actually configured for.
    """
    rel = ini.section_file_rel(name)
    explicit = ((ini.get_section(name) or {}).get("model") or "").strip()
    if explicit and rel and (settings.models_dir / rel).is_file():
        return settings.models_dir / rel, (explicit if "/" in rel else ""), rel

    if rel is not None and "/" in rel:
        subdir_name = rel.split("/", 1)[0]
        subdir_path = settings.models_dir / subdir_name
        if subdir_path.is_dir():
            gguf_files = sorted([q for q in subdir_path.iterdir() if q.is_file() and q.suffix.lower() == ".gguf"])
            # Every companion, not just mmproj. This used to test `"mmproj" not in name` and
            # then take main_files[0] - the alphabetically first survivor - which on a real
            # folder picked ...-MAX-MTP-IQ2_M.gguf over ...-MAX-Q8_0.gguf because M sorts before
            # Q. The section named ...-Q8_0 was therefore configured to run an 11.3 GiB IQ2_M
            # quant while the 27.7 GiB Q8_0 beside it was never loaded. ini._is_companion
            # already knew about draft heads; this was a third copy of the narrower mmproj-only
            # test, and the one missed when the other two were unified.
            main_files = [q for q in gguf_files
                          if not ini._is_companion(q.name, autoconfig.file_size_or_none(q))
                          ] or gguf_files
            if main_files:
                pick = _pick_main_gguf(main_files, name)
                return pick, f"/models/{subdir_name}/{pick.name}", rel
        return None, "", rel
    if rel is not None:
        return settings.models_dir / rel, "", rel
    return settings.models_dir / f"{name}.gguf", "", rel


def _gguf_hints_for(name: str) -> tuple[dict[str, str], list[str]]:
    gguf_path, model_rel, rel = _resolve_section_gguf(name)

    if gguf_path is None or not gguf_path.is_file():
        return {}, [f"No GGUF found for `{name}` — cannot suggest defaults from metadata."]
    try:
        summary = gguf_meta.summarize(gguf_meta.read_raw(gguf_path))
    except (gguf_meta.GgufMetaError, OSError) as e:
        return {}, [f"Could not read GGUF: {e}"]
    values, hints = ini.suggest_defaults(summary)
    if model_rel:
        values["model"] = model_rel
        # Auto-detect a companion mmproj (multimodal projector — vision, audio, etc.)
        # in the same subdir so multimodal models load out of the box.
        subdir_path = Path(model_rel).parent  # e.g. /models/<stem>
        try:
            for p in Path(str(subdir_path).replace("/models", str(settings.models_dir), 1)).iterdir():
                if p.is_file() and p.suffix.lower() == ".gguf" and "mmproj" in p.name.lower():
                    values["mmproj"] = f"/models/{Path(model_rel).parts[-2]}/{p.name}"
                    hints.insert(0, f"Companion mmproj found → `mmproj = {values['mmproj']}` pre-filled.")
                    break
        except OSError:
            pass
        # Distinguish real sharding (multi-part files) from "single file in a subdir"
        _, part_idx, part_total = shard_key(Path(model_rel).name)
        if part_idx is not None and part_total and part_total > 1:
            hints.insert(0, f"Sharded model ({part_total} parts) → `model = {model_rel}` pre-filled to the first shard; llama-server auto-loads the rest.")
        else:
            hints.insert(0, f"Model lives in a subdir → `model = {model_rel}` pre-filled with the absolute path.")
    return values, hints


@app.get("/config/section/{name}/edit", response_class=HTMLResponse)
def config_section_edit(request: Request, name: str, reset: int = 0) -> HTMLResponse:
    # This endpoint returns a PARTIAL, designed to be swapped into /config by HTMX. Reaching
    # it by ordinary navigation (the "edit models.ini" link on a model page, a bookmark, a
    # refresh) renders the fragment with no base.html — so no stylesheet, no nav, just raw
    # form controls on white. Bounce those requests to the real page and let it open the
    # section instead.
    if request.headers.get("HX-Request", "").lower() != "true":
        from urllib.parse import quote
        return Response(status_code=303, headers={"Location": f"/config?edit={quote(name)}"})
    vals = ini.get_section(name)
    if vals is None and not reset:
        return HTMLResponse(f"unknown section: {name}", status_code=404)
    if reset:
        form_vals, hints = _gguf_hints_for(name)
        extras_text = ""
        hints = ["Reset to GGUF-derived defaults. Nothing saved yet — click Save to apply."] + hints
    else:
        form_vals, extras_text = ini.split_section_for_form(vals or {})
        hints = []
    return templates.TemplateResponse("_section_form.html", {
        "request": request,
        "mode": "edit",
        "section_name": name,
        "values": form_vals,
        "extras_text": extras_text,
        "form_tiers": ini.FORM_TIERS,
        "warning": None,
        "smart_hints": hints,
        "auto_filled": reset == 1,
    })


@app.post("/config/section/{name}")
async def config_section_save(request: Request, name: str) -> Response:
    if not ini.valid_section_name(name):
        return Response(status_code=200, headers={"HX-Redirect": f"/config?err=invalid+section+name+{name}"})
    form = await request.form()
    values: dict[str, str] = {}
    for f in ini.ALL_FIELDS:
        raw = form.get(f"fld_{f.key}", "")
        if f.kind == "bool":
            values[f.key] = "true" if raw else ""
        else:
            values[f.key] = str(raw).strip()
    extras = str(form.get("extras", ""))
    # Checked BEFORE the write: a section that did not exist a moment ago is a new model,
    # and only a new model gets a default backend. Re-running this on an ordinary save would
    # undo a deliberate removal from a connection.
    is_new_section = name not in ini.section_names()
    try:
        ini.upsert_section(name, values, extras)
    except Exception as e:  # noqa: BLE001
        return Response(status_code=200, headers={"HX-Redirect": f"/config?err=save+failed:+{e}"})
    # Whether a model can accept an image is decided here (by the presence of `mmproj`) but
    # enforced in OpenWebUI, which otherwise shows the image-upload control on everything and
    # only fails at inference with "image input is not supported". Push it across on every
    # save, so adding or removing a projector updates the UI that people actually click.
    # No restart: the `model` table is ordinary app data, not PersistentConfig.
    try:
        services.sync_openwebui_capabilities()
    except Exception:  # noqa: BLE001 -- saving the section must not depend on OpenWebUI
        pass
    # A brand-new model is offered on the GPU backends by default. Without this it lands in
    # models.ini, works, and is invisible in OpenWebUI because every connection carries an
    # explicit whitelist that predates it.
    note = ""
    if is_new_section:
        try:
            ok, msg = services.assign_new_model_to_gpu(name)
            if ok and msg:
                note = f"&note={msg}"
        except Exception:  # noqa: BLE001
            pass
    return Response(status_code=200, headers={"HX-Redirect": f"/config?saved={name}{note}"})


def _container_baseline(name: str) -> list[str]:
    """Return the Config.Cmd list for a running container, or [] on failure."""
    try:
        client = services._docker_client()
        if client is None:
            return []
        c = client.containers.get(name)
        return list((c.attrs or {}).get("Config", {}).get("Cmd") or [])
    except Exception:  # noqa: BLE001
        return []


def _backend_list() -> list[dict]:
    """Backend descriptors for autoconfig, from the env whitelist or auto-discovery.

    Extracted so the benchmark results page can ask for the same fit estimate the autoconfig
    panel shows, rather than growing a second, drifting copy of this.
    """
    out = []
    for bn in services._effective_container_names():
        vram = hw.vram_gb_for(bn)
        if vram <= 0:
            continue  # CPU backends have no VRAM budget; autoconfig can't do KV math on them yet
        # detect vendor via hw sampler cache (or docker inspect image)
        vendor = "unknown"
        try:
            client = services._docker_client()
            if client is not None:
                img = ((client.containers.get(bn).image.tags or [""]) or [""])[0].lower()
                if "rocm" in img:
                    vendor = "rocm"
                elif "cuda" in img:
                    vendor = "cuda"
        except Exception:  # noqa: BLE001
            pass
        cmd = _container_baseline(bn)
        base = autoconfig.parse_baseline(cmd) if cmd else {}
        out.append({"name": bn, "vendor": vendor, "vram_gb": float(vram),
                    "gpu_count": hw.gpu_count_for(bn), "card_vram_gb": hw.card_vram_gb_for(bn),
                    "host_ram_gb": hw.host_ram_gb(),
                    "baseline": base})
    return out


def _predicted_vram_gb(section: str) -> float | None:
    """Autoconfig's budget for this section: weights + KV only, in GB.

    NOT the total the cards will hold. Compute buffers and the projector are budgeted elsewhere
    and are not small - llama.cpp's own estimator puts gemma-4-E4B's compute buffers at 4 GB
    against 2.4 GB of weights - so a measured peak reads several GB above this by construction.
    The chart labels both bars accordingly; presenting them as rivals would make a correct
    estimate look 5 GB wrong.
    """
    try:
        gguf_path, model_rel, rel = _resolve_section_gguf(section)
        if gguf_path is None or not gguf_path.is_file():
            return None
        summary = gguf_meta.summarize(gguf_meta.read_raw(gguf_path))
        backends = _backend_list()
        if not backends:
            return None
        subdir = rel.split("/", 1)[0] if rel and "/" in rel else ""
        vals = ini.get_section(section) or {}
        rec = autoconfig.analyze(
            summary=summary, file_size=gguf_path.stat().st_size, backends=backends,
            model_rel=model_rel, current_section=vals, models_dir=settings.models_dir,
            section_name=section, model_subdir=subdir)
        if not rec.frontier:
            return None
        # Compare at the context the section is actually configured for, not at whatever
        # autoconfig would recommend - the measurement was taken under the former.
        try:
            want = int(str(vals.get("ctx-size", "")).strip() or 0)
        except ValueError:
            want = 0
        want = want or rec.recommended_total_ctx or rec.recommended_ctx
        row = min(rec.frontier, key=lambda r: abs(r.ctx - want)) if want else rec.frontier[0]
        return round(row.gpu_gb + row.kv_gb, 2)
    except Exception:  # noqa: BLE001
        return None


@app.get("/config/section/{name}/autoconfig", response_class=HTMLResponse)
async def config_autoconfig(request: Request, name: str, preset: str = "",
                            sessions: int = 0, spec: str = "", vision: str = "") -> HTMLResponse:
    import json as _json
    # 0 means the UI did not specify, so fall back to what is SAVED rather than to 1. Defaulting
    # to 1 made every fresh open of the panel propose resetting a `parallel = 3` section to one
    # slot, and Fill+Save would then quarter its per-conversation context without anyone
    # choosing that - the same silent-downgrade trap _preset_for_saved_ctx exists to prevent for
    # ctx-size. The panel's own buttons always pass sessions explicitly, so they still
    # round-trip whatever is on screen.
    if not sessions:
        try:
            sessions = int(str((ini.get_section(name) or {}).get("parallel") or "1").strip())
        except (TypeError, ValueError):
            sessions = 1
    sessions = max(1, min(int(sessions or 1), 8))

    gguf_path, model_rel, rel = _resolve_section_gguf(name)

    if gguf_path is None or not gguf_path.is_file():
        return templates.TemplateResponse("_autoconfig_panel.html", {
            "request": request,
            "rec": autoconfig.Recommendation(plans=[], recommended_backend="", recommended_ctx=0,
                                             values={}, values_minimal={}, baseline_redundant={},
                                             quirks=[], unavailable=[], current_diff=[],
                                             error=f"No GGUF found for '{name}'."),
        })

    try:
        summary = gguf_meta.summarize(gguf_meta.read_raw(gguf_path))
    except (gguf_meta.GgufMetaError, OSError) as e:
        return templates.TemplateResponse("_autoconfig_panel.html", {
            "request": request,
            "rec": autoconfig.Recommendation(plans=[], recommended_backend="", recommended_ctx=0,
                                             values={}, values_minimal={}, baseline_redundant={},
                                             quirks=[], unavailable=[], current_diff=[],
                                             error=f"Failed to read GGUF: {e}"),
        })

    file_size = gguf_path.stat().st_size
    # If sharded, sum THIS model's own shards. Matching on the shard base rather than on
    # "every non-companion GGUF in the folder" matters as soon as a folder holds two quants of
    # one model: the turbo folder has an 11.3 GiB IQ2_M beside a 27.7 GiB Q8_0, and the old
    # rule reported 28.3 GB of weights whichever of them was selected - for the IQ2_M that is
    # nearly triple. An unsharded file is simply its own size, so the sum only runs when this
    # file actually declares itself part of a set.
    if gguf_path is not None:
        _base, _idx, _parts = shard_key(gguf_path.name)
        if _parts:
            try:
                file_size = sum(
                    q.stat().st_size for q in gguf_path.parent.iterdir()
                    if q.is_file() and q.suffix.lower() == ".gguf"
                    and shard_key(q.name)[0] == _base
                )
            except OSError:
                pass

    backend_list = _backend_list()

    # Existing section (for diff)
    current_section = ini.get_section(name)

    # Detect subdir (if the resolved rel had one)
    model_subdir = ""
    if rel and "/" in rel:
        model_subdir = rel.split("/", 1)[0]

    # Fold in whatever llama-server has logged since the last look. Rate-limited internally and
    # wrapped so a log-format change or a docker hiccup costs the panel its measurements rather
    # than costing the user the page.
    try:
        telemetry.ingest(services._effective_container_names())
        tel = telemetry.stats_for(model_path=model_rel, alias=name)
        cfgh = telemetry.config_history(model_path=model_rel, alias=name)
    except Exception:  # noqa: BLE001
        tel, cfgh = telemetry.Stats(), []

    rec = autoconfig.analyze(
        summary=summary,
        file_size=file_size,
        backends=backend_list,
        model_rel=model_rel,
        current_section=current_section,
        preset=preset,
        n_sessions=sessions,
        models_dir=settings.models_dir,
        section_name=name,
        model_subdir=model_subdir,
        spec_profile=spec,
        vision=vision,
    )

    # Vision costs context, and the panel should say how much rather than leave it to be found
    # out. When a projector exists, size the other state as well with everything else held
    # equal - same preset, sessions and spec profile - so both figures come from the same fit
    # maths instead of one real number and one guess.
    vision_cost = None
    if rec.has_projector and not rec.error:
        def _vision_state(r):
            p = next((x for x in r.presets if x.key == r.active_preset), None)
            return {"ctx": r.recommended_ctx,
                    "gpu_layers": p.gpu_layers if p else 0,
                    "total_layers": p.total_layers if p else 0}
        try:
            alt = autoconfig.analyze(
                summary=summary, file_size=file_size, backends=backend_list,
                model_rel=model_rel, current_section=current_section,
                preset=rec.active_preset, n_sessions=sessions,
                models_dir=settings.models_dir, section_name=name,
                model_subdir=model_subdir, spec_profile=spec,
                vision=("off" if rec.vision else "on"),
            )
            vision_cost = {("on" if rec.vision else "off"): _vision_state(rec),
                           ("off" if rec.vision else "on"): _vision_state(alt)}
        except Exception:  # noqa: BLE001 - the comparison is a nicety; it must not cost the panel
            vision_cost = None

    # Prepare per-plan row map for the template + which ctx columns to show
    plan_row_map = {p.name: {r.ctx: r for r in p.rows} for p in rec.plans}

    # Where the recommended configuration's VRAM actually goes, split into the segments the
    # panel draws. Built here rather than in the template so the bar shows the same figures the
    # fit maths used instead of a re-derivation that could drift from them. Every byte of the
    # card is accounted for, free space included, so the bar sums to the whole card - a
    # breakdown that silently omits a cost is how the compute buffer stayed invisible for so
    # long. Ordered weights -> context -> overhead -> compute: that order keeps the two hues
    # the palette validator flags as a weak pair for protanopia from touching.
    vram_breakdown = None
    _is_moe_model = isinstance((summary.get("model") or {}).get("expert_count"), int)         and ((summary.get("model") or {}).get("expert_count") or 0) > 1
    _rec_plan = next((p for p in rec.plans if p.name == rec.recommended_backend), None)
    _rec_row = (plan_row_map.get(rec.recommended_backend) or {}).get(rec.recommended_ctx)
    if _rec_plan and _rec_row and _rec_plan.vram_gb > 0 and _rec_row.fits:
        _total = float(_rec_plan.vram_gb)
        _parts = [
            # Say the offloaded share outright. On a model larger than the card this segment is
            # only the resident part - Flash-Next shows 51 GB here against an 84 GB file - and a
            # reader comparing it to the file size has no way to tell that is correct.
            ("model", "model weights", float(_rec_row.model_gb),
             ("GPU-resident weights: %d%% of the model, the other %d%% streams from host RAM"
              % (_rec_row.gpu_pct, 100 - _rec_row.gpu_pct)) if _rec_row.gpu_pct < 100
             else "the whole model, resident on the GPU"),
            ("ctx", "context (KV)", float(_rec_row.kv_gb),
             "KV cache for %s tokens" % f"{_rec_row.total_ctx:,}"),
            ("overhead", "overhead", float(_rec_row.reserve_gb) + float(_rec_row.aux_gb),
             "driver and runtime context, plus any vision projector and draft head"),
            ("compute", "compute buffers", float(_rec_row.compute_gb),
             "per-card graph scratch, which grows with context"),
        ]
        _used = sum(v for _k, _l, v, _n in _parts)
        _free = max(0.0, _total - _used)
        _segments = [{"key": k, "label": lbl, "gb": v, "pct": 100.0 * v / _total, "note": note}
                     for k, lbl, v, note in _parts if v > 0.005]
        if _free > 0.005:
            _segments.append({"key": "free", "label": "free", "gb": _free,
                              "pct": 100.0 * _free / _total,
                              "note": "unallocated on the card"})
        # Where the model's WEIGHTS end up, which the card bar above cannot show: anything
        # offloaded is not on the card at all, so it has no place in a bar scaled to the card.
        # Its own bar, its own denominator (the model), stated as such - two scales in one bar
        # would be the chart equivalent of a second y-axis.
        #
        # This is the number that answers "have I left VRAM". For a MoE it is specifically the
        # experts that move: attention stays resident, so counting layers would understate it,
        # which is why gpu_pct is measured in bytes.
        _weights_total_gb = file_size / (1024 ** 3)
        _weights_gpu_gb = float(_rec_row.model_gb)
        _weights_host_gb = max(0.0, _weights_total_gb - _weights_gpu_gb)
        weights_split = None
        if _weights_total_gb > 0 and _weights_host_gb > 0.05:
            weights_split = {
                "total_gb": _weights_total_gb,
                "gpu_gb": _weights_gpu_gb,
                "host_gb": _weights_host_gb,
                "gpu_pct": 100.0 * _weights_gpu_gb / _weights_total_gb,
                "host_pct": 100.0 * _weights_host_gb / _weights_total_gb,
                "is_moe": bool(_is_moe_model),
                "offload_kind": _rec_row.offload_kind,
                "n_cpu_moe": _rec_row.n_cpu_moe,
            }
        vram_breakdown = {
            "total_gb": _total, "used_gb": _used, "free_gb": _free,
            "backend": rec.recommended_backend, "ctx": rec.recommended_ctx,
            "total_ctx": _rec_row.total_ctx, "segments": _segments,
            "weights_split": weights_split,
            "subtitle": "%s at %s%s" % (
                rec.recommended_backend, autoconfig.format_ctx(rec.recommended_ctx),
                " \u00d7 %d" % rec.n_sessions if rec.n_sessions > 1 else ""),
        }
    all_ctx = sorted({r.ctx for p in rec.plans for r in p.rows})
    # pick a compact set: some small, some near the max/native
    interesting = set()
    for target in (8192, 32768, 65536, 131072, 196608, 262144):
        if target in all_ctx:
            interesting.add(target)
    for p in rec.plans:
        if p.max_ctx:
            interesting.add(p.max_ctx)
    native = summary.get("model", {}).get("context_length")
    if isinstance(native, int) and native in all_ctx:
        interesting.add(native)
    ctx_columns = sorted(interesting)[:6]  # cap to 6 columns for width

    return templates.TemplateResponse("_autoconfig_panel.html", {
        "request": request,
        "rec": rec,
        "section_name": name,
        "model_display": model_rel or (rel or f"{name}.gguf"),
        "file_size": file_size,
        "arch": summary.get("arch"),
        "params": (summary.get("general") or {}).get("params"),
        "ctx_columns": ctx_columns,
        "plan_row_map": plan_row_map,
        "vram_breakdown": vram_breakdown,
        "format_ctx": autoconfig.format_ctx,
        "vision_cost": vision_cost,
        # Carried on every link that re-renders this panel, so picking a preset or a session
        # count does not silently flip vision back to the saved state.
        "vision_q": ("on" if rec.vision else "off") if rec.has_projector else "",
        "tel": tel,
        "cfgh": cfgh,
        "values_json": _json.dumps(rec.values),
        "values_minimal_json": _json.dumps(rec.values_minimal),
        "displaced_json": _json.dumps(rec.displaced),
        # Full offload frontier for the custom slider. Every entry is an achievable
        # config, so the slider snaps to real points rather than interpolating.
        "frontier_json": _json.dumps([
            {
                "ctx": p.ctx,
                "total_ctx": p.ctx * sessions,
                "ngl": p.ngl,
                "n_cpu_moe": p.n_cpu_moe,
                "kind": p.offload_kind,
                "gpu_layers": p.gpu_layers,
                "total_layers": p.total_layers,
                "gpu_gb": p.gpu_gb,
                "kv_gb": p.kv_gb,
                "speed": p.speed_score,
            }
            for p in rec.frontier
        ]),
    })


@app.post("/config/section/{name}/rename")
async def config_section_rename(request: Request, name: str) -> Response:
    """Rename a preset. The section name IS the model id llama-server serves, so this is how
    you give a model a short, human name — `model =` keeps pointing at the same file."""
    form = await request.form()
    new = (form.get("new_name") or "").strip()
    if not new:
        return Response(status_code=200, headers={"HX-Redirect": "/config?err=new+name+required"})
    if new == name:
        return Response(status_code=200, headers={"HX-Redirect": f"/config?edit={new}"})
    if not ini.valid_section_name(new):
        return Response(status_code=200,
                        headers={"HX-Redirect": f"/config?err=invalid+name:+{new}"})
    if new in ini.section_names():
        return Response(status_code=200,
                        headers={"HX-Redirect": f"/config?err=section+already+exists:+{new}"})
    if not ini.rename_section(name, new):
        return Response(status_code=200, headers={"HX-Redirect": "/config?err=rename+failed"})

    # Reconcile OpenWebUI's per-connection whitelists in ONE pass (each write restarts
    # open-webui, so doing this as two passes would cost two restarts).
    #
    # Two things have to happen together. Carry the old id across, or the model silently
    # vanishes from the picker. And drop ids no section provides any more: OpenWebUI RENDERS
    # its whitelist rather than intersecting it with what the backend reports, so a dead id
    # stays visible in the model picker and fails with "model not found" only when someone
    # tries to use it — the one failure mode neither end shows you.
    try:
        known = set(ini.section_names())
        for c in (services.openwebui_state().get("connections") or []):
            ids = c.get("model_ids") or []
            if not ids:
                continue  # empty whitelist = "offer everything"; nothing to reconcile
            fixed = [new if m == name else m for m in ids]
            fixed = [m for m in fixed if m in known]
            # Never write an empty list: that would flip the connection from a filter to
            # "offer every model", quietly widening what this backend exposes.
            if fixed and fixed != ids:
                services.set_openwebui_model_filter(c["url"], fixed)
    except Exception:  # noqa: BLE001 — renaming must succeed even if OpenWebUI is unreachable
        pass

    return Response(status_code=200, headers={"HX-Redirect": f"/config?saved={new}&edit={new}"})


@app.post("/config/section/{name}/delete")
def config_section_delete(name: str) -> Response:
    ok = ini.delete_section(name)
    if not ok:
        return Response(status_code=200, headers={"HX-Redirect": f"/config?err=no+such+section:+{name}"})

    # Removing the section is the moment the id stops being servable -- llama-server resolves
    # model ids out of models.ini, so from here OpenWebUI is offering something no backend can
    # answer. And it RENDERS its whitelist rather than intersecting it with what the backend
    # reports, so the model sits in the picker looking fine and fails with "model not found"
    # only when somebody selects it.
    #
    # The rename path already reconciles for exactly this reason; delete did not, which is how
    # a deleted model kept appearing. Deleting the GGUF does not help either: that leaves the
    # section in place, so the id is still "known" at that point and nothing is pruned.
    #
    # Two writes, one restart. The whitelist lives in OpenWebUI's config table, which it reads
    # only at boot, so changing it costs a restart. The stale per-model record lives in its
    # `model` table, which is read per request, so clearing that one is free.
    try:
        services.prune_openwebui_unknown_ids()
        services.align_openwebui_capabilities()
    except Exception:  # noqa: BLE001 -- deleting a section must succeed even if OpenWebUI is down
        pass
    return Response(status_code=200, headers={"HX-Redirect": f"/config?deleted={name}"})


# ---------- Containers ----------

def _stats_by_name() -> dict:
    return {name: hw.stats_for(name) for name in services._effective_container_names()}


def _perf_by_name() -> dict:
    """Sparkline-ready view of the rolling history the hw sampler already keeps.

    Percentages are pinned to a 0-100 scale and VRAM to the card total, so the lines stay
    comparable between refreshes instead of auto-rescaling to whatever the current window
    happens to contain.
    """
    out: dict[str, dict] = {}
    for name in services._effective_container_names():
        pts = hw.history_for(name)
        if not pts:
            continue
        st = hw.stats_for(name)
        vram_total = st.gpu.vram_total_gb if (st.ok and st.gpu) else 0.0
        span_s = (pts[-1].ts - pts[0].ts) if len(pts) > 1 else 0.0
        mins = int(span_s // 60)
        cards = (st.gpu.cards if (st.ok and st.gpu) else []) or []
        frees = [c.vram_free_gb for c in cards]
        out[name] = {
            "points": len(pts),
            "has_gpu": bool(st.ok and st.gpu),
            "window_label": (f"last {mins}m" if mins >= 1 else f"last {int(span_s)}s"),
            "gpu_util": hw.sparkline([p.gpu_util for p in pts], 100.0),
            "vram": hw.sparkline([p.vram_used_gb for p in pts], vram_total or None),
            "cpu": hw.sparkline([p.cpu_pct for p in pts], None),
            "mem": hw.sparkline([p.mem_used_gb for p in pts], None),
            "cur_gpu_util": pts[-1].gpu_util,
            "cur_vram": pts[-1].vram_used_gb,
            "vram_total": vram_total,
            "cur_cpu": pts[-1].cpu_pct,
            "cur_mem": pts[-1].mem_used_gb,
            # spread between the most and least free card — the number that explains
            # "OOM on one GPU while the other looks fine"
            "imbalance_gb": round(max(frees) - min(frees), 2) if len(frees) > 1 else 0.0,
        }
    return out


@app.get("/containers", response_class=HTMLResponse)
async def containers_page(request: Request) -> HTMLResponse:
    backends = await services.snapshot_llama_backends()
    return templates.TemplateResponse("containers.html", {
        "request": request, "backends": backends, "stats": _stats_by_name(), "perf": _perf_by_name(),
        "prompts": db.list_prompts(),
        "openwebui": services.openwebui_state(),
        # Raw model ids a backend reports — what OpenWebUI's model_ids whitelist matches on.
        # Every llama backend serves the same models.ini, so the section names are the list.
        "all_model_ids": sorted(ini.section_names()),
    })


@app.post("/containers/sync-openwebui", response_class=HTMLResponse)
def containers_sync_openwebui() -> HTMLResponse:
    ok, msg = services.sync_openwebui_endpoints()
    if ok:
        return HTMLResponse(
            f'<div class="rounded-md bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 px-3 py-2 text-sm">✓ {msg}</div>'
        )
    return HTMLResponse(
        f'<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">Sync failed: {msg}</div>'
    )


@app.post("/containers/openwebui-align-capabilities", response_class=HTMLResponse)
def containers_openwebui_align() -> HTMLResponse:
    """Make OpenWebUI's per-model capabilities match models.ini, and drop stale records."""
    ok, msg = services.align_openwebui_capabilities()
    cls = ("bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300" if ok
           else "bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300")
    return HTMLResponse(f'<div class="rounded-md {cls} px-3 py-2 text-sm">{"✓" if ok else "Align failed:"} {msg}</div>')


@app.post("/containers/openwebui-filter", response_class=HTMLResponse)
async def containers_openwebui_filter(request: Request) -> HTMLResponse:
    """Set which models one OpenWebUI connection offers. No checked boxes = offer all."""
    form = await request.form()
    url = (form.get("url") or "").strip()
    if not url:
        return HTMLResponse(
            '<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">No connection specified.</div>'
        )
    model_ids = [str(v) for v in form.getlist("model_ids") if str(v).strip()]
    ok, msg = services.set_openwebui_model_filter(url, model_ids)
    cls = ("bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300" if ok
           else "bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300")
    mark = "✓" if ok else "Filter failed:"
    return HTMLResponse(f'<div class="rounded-md {cls} px-3 py-2 text-sm">{mark} {msg}</div>')


def _toast(ok: bool, msg: str) -> HTMLResponse:
    """The banner the models page swaps into #models-owui-toast. Shared so three routes that
    report into the same strip cannot drift apart in markup."""
    cls = ("bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300" if ok
           else "bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300")
    return HTMLResponse(f'<div class="rounded-md {cls} px-3 py-2 text-sm">'
                        f'{"&check;" if ok else "Failed:"} {html.escape(msg)}</div>')


@app.post("/models/redownload", response_class=HTMLResponse)
async def models_redownload(name: str = Form(...)) -> HTMLResponse:
    """Queue a fresh copy of every part of one model from the repo it came from.

    The stale chip used to be a span, so the page could tell you a model was out of date and
    then offer nothing to do about it. This is what the chip posts to.

    It re-downloads the parts the update check actually resolved - local path to repo path,
    one row per shard - rather than guessing a repo layout, and leaves the existing files in
    place until each download finishes: the downloader writes to a temp file and os.replace()s
    it, so a model that is loaded right now keeps serving from the old inode until it is
    reloaded.
    """
    snap = services.snapshot_models_dir()
    g = next((e for e in snap.ggufs if e.display_name == name), None)
    if g is None:
        return _toast(False, f"{name} is not in the models directory")
    info = _update_status_map(snap).get(name) or {}
    repo_id = info.get("repo_id") or ""
    parts = [(rel, rp) for rel, rp in (info.get("parts") or []) if rp]
    if not repo_id or "/" not in repo_id:
        return _toast(False, f"no Hugging Face repo recorded for {name} — re-download it from Search HF")
    if not parts:
        return _toast(False, f"could not work out which files in {repo_id} {name} came from; "
                             "run Check for updates first")
    by_name = {p.name: p for p in list(g.parts) + list(g.companion_parts)}
    for rel, repo_path in parts:
        p = by_name.get(rel.rsplit("/", 1)[-1])
        try:
            size = p.stat().st_size if p is not None else 0
        except OSError:
            size = 0
        manager.enqueue(repo_id=repo_id, hf_path=repo_path, filename=rel, total_bytes=size)
    n = len(parts)
    return _toast(True, f"queued {n} file{'s' if n != 1 else ''} from {repo_id} — see Downloads")


@app.post("/models/align-capability", response_class=HTMLResponse)
def models_align_capability(section: str = Form(...)) -> HTMLResponse:
    """Align OpenWebUI capabilities for a single models.ini section."""
    ok, msg = services.align_openwebui_capability_for(section)
    return _toast(ok, msg)


@app.post("/models/openwebui-visibility", response_class=HTMLResponse)
async def models_openwebui_visibility(request: Request) -> HTMLResponse:
    """Toggle one model's visibility on one OpenWebUI connection, from the Models page."""
    form = await request.form()
    url = (form.get("url") or "").strip()
    model_id = (form.get("model_id") or "").strip()
    show = str(form.get("show") or "").lower() in ("1", "true", "on", "yes")
    if not url or not model_id:
        return HTMLResponse(
            '<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">Missing model or connection.</div>'
        )
    ok, msg = services.toggle_openwebui_model(url, model_id, show, sorted(ini.section_names()))
    cls = ("bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300" if ok
           else "bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300")
    return HTMLResponse(f'<div class="rounded-md {cls} px-3 py-2 text-sm">{"✓" if ok else "Failed:"} {msg}</div>')


@app.get("/containers/{name}/dashboard", response_class=HTMLResponse)
async def containers_dashboard(request: Request, name: str) -> HTMLResponse:
    backends = await services.snapshot_llama_backends()
    b = next((x for x in backends if x.name == name), None)
    if b is None:
        return HTMLResponse(f"<span class='text-xs text-slate-500'>unknown backend: {name}</span>")
    # Throughput rides along with the rest of the 2 s poll rather than having its own timer:
    # one refresh, one consistent picture, and no second interval to reason about.
    speed = await services.inference_speed(name, b.internal_port, b.loaded_model)
    # Live "where the VRAM goes" meter, from what this backend actually loaded against
    # measured usage. Only when this backend owns the cards alone (see _solo_backend);
    # None otherwise, and the card draws no meter. A backend with no GPU gets the RAM
    # variant instead: the whole is its own process footprint, not a shared device, so
    # there is no sole-ownership question to fail.
    st = _stats_by_name().get(name)
    vram_breakdown = None
    vb_heading = None
    if st and st.ok:
        if st.gpu and _solo_backend() == name:
            tenants = _gpu_tenants()
            vram_breakdown = vram_live.breakdown(
                name, st.gpu,
                frozenset(i.strip() for i in (b.loaded_model or "").split(",") if i.strip()),
                foreign=[(t.label, t.total_gb) for t in tenants if t.foreign and t.total_gb > 0.005])
        elif not st.gpu and st.container:
            vram_breakdown = vram_live.ram_breakdown(name, st.container)
            if vram_breakdown:
                vb_heading = "Where the RAM goes"
    return templates.TemplateResponse("_container_dashboard.html", {
        "request": request, "b": b, "stats": _stats_by_name(), "perf": _perf_by_name(),
        "speed": speed, "vram_breakdown": vram_breakdown, "vb_heading": vb_heading,
    })


@app.post("/containers/{name}/restart", response_class=HTMLResponse)
def containers_restart(name: str) -> HTMLResponse:
    services.restart_llama_backend(name)
    # small chip: replaces the button when hx-swap=outerHTML; ignored when hx-swap=none
    return HTMLResponse(
        f'<span class="inline-flex items-center rounded-md bg-amber-100 dark:bg-amber-950 '
        f'text-amber-800 dark:text-amber-300 px-2.5 py-1 text-xs font-mono">restarting {name}…</span>'
    )


@app.post("/containers/{name}/test", response_class=HTMLResponse)
async def containers_test(name: str, prompt: str = Form(...), max_tokens: int = Form(256)) -> HTMLResponse:
    result = await services.test_prompt(name, prompt, max_tokens=max_tokens)
    if not result.get("ok"):
        return HTMLResponse(
            f'<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">'
            f'{result.get("err", "test failed")}</div>'
        )
    reply = (result.get("reply") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    tps = result.get("tokens_per_s")
    tps_bit = f" · {tps} tok/s" if tps else ""
    meta = (
        f'{result["model"]} · {result["completion_tokens"]} tok in {result["elapsed_s"]}s{tps_bit}'
        f' · prompt {result["prompt_tokens"]} tok'
    )
    return HTMLResponse(
        f'<div class="rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/40 p-3">'
        f'<pre class="text-xs whitespace-pre-wrap font-mono">{reply}</pre>'
        f'<div class="mt-2 text-[10px] text-slate-500 dark:text-slate-400 font-mono">{meta}</div>'
        f'</div>'
    )


@app.get("/containers/{name}/logs", response_class=HTMLResponse)
def containers_logs(name: str, q: str = "", level: str = "") -> HTMLResponse:
    ok, out = services.container_logs(name)
    if not ok:
        return HTMLResponse(f"<span class='text-red-400'>{out}</span>")

    if q or level:
        needle = q.lower()
        keep_error = level == "error"
        keep_warn = level in ("warn", "warning")
        filtered: list[str] = []
        for line in out.splitlines():
            low = line.lower()
            if needle and needle not in low:
                continue
            if keep_error and ("err" not in low and "error" not in low and "fatal" not in low):
                continue
            if keep_warn and ("err" not in low and "warn" not in low and "wrn" not in low and "error" not in low and "fatal" not in low):
                continue
            filtered.append(line)
        out = "\n".join(filtered)
        if not out:
            out = f"(no lines match q={q!r} level={level!r})"

    safe = out.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return HTMLResponse(safe or "(no output)")


@app.get("/containers/{name}/diagnose", response_class=HTMLResponse)
def containers_diagnose(request: Request, name: str) -> HTMLResponse:
    """Parse a backend's recent log for a failed-load signature and propose a fix.

    Rendered as a Jinja partial for consistency with every other fragment here — Jinja
    autoescapes the log lines, so no hand-rolled escaping is needed.
    """
    ok, findings, err = services.diagnose_container(name)
    return templates.TemplateResponse("_diagnose_result.html", {
        "request": request, "ok": ok, "err": err, "findings": findings,
    })


# ---------- Prompt library ----------

@app.get("/prompts", response_class=HTMLResponse)
def prompts_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("prompts.html", {
        "request": request, "prompts": db.list_prompts(), "flash": None,
    })


@app.post("/prompts", response_class=HTMLResponse)
def prompts_add(request: Request, name: str = Form(...), body: str = Form(...)) -> HTMLResponse:
    name = name.strip()
    body = body.strip()
    if not name or not body:
        return templates.TemplateResponse("prompts.html", {
            "request": request, "prompts": db.list_prompts(),
            "flash": {"ok": False, "msg": "name and body are required"},
        })
    db.add_prompt(name, body)
    return templates.TemplateResponse("prompts.html", {
        "request": request, "prompts": db.list_prompts(),
        "flash": {"ok": True, "msg": f"saved '{name}'"},
    })


@app.post("/prompts/{pid}/delete", response_class=HTMLResponse)
def prompts_delete(request: Request, pid: int) -> HTMLResponse:
    ok = db.delete_prompt(pid)
    return templates.TemplateResponse("prompts.html", {
        "request": request, "prompts": db.list_prompts(),
        "flash": {"ok": ok, "msg": "deleted" if ok else "not found"},
    })


@app.get("/prompts/{pid}/body", response_class=HTMLResponse)
def prompts_body(pid: int) -> HTMLResponse:
    p = db.get_prompt(pid)
    if not p:
        return HTMLResponse("", status_code=404)
    return HTMLResponse(p["body"])


# ---------------------------------------------------------------- benchmark

def _bench_ctx(request: Request, flash: str = "") -> dict:
    """Everything the benchmark page needs. Sections are offered rather than GGUF files:
    a benchmark runs against an alias the router can serve, which is what a section is."""
    return {
        "request": request,
        "sections": ini.section_names(),
        "sweep_args": {n: " ".join(bench.sweep_args_for_section(n)) for n in ini.section_names()},
        "prompts": db.list_prompts(),
        "backends": services._effective_container_names(),
        "max_tokens_default": bench.DEFAULT_MAX_TOKENS,
        "max_tokens_ceiling": bench.MAX_TOKENS_CEILING,
        "job": bench.state(),
        "runs": db.bench_runs(limit=15),
        "flash": flash,
    }


@app.get("/benchmark", response_class=HTMLResponse)
def benchmark_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("benchmark.html", _bench_ctx(request))


@app.post("/benchmark/start", response_class=HTMLResponse)
def benchmark_start(request: Request,
                    backend: str = Form(...),
                    aliases: list[str] = Form(default=[]),
                    prompt_ids: list[int] = Form(default=[]),
                    reps: int = Form(3),
                    max_tokens: int = Form(bench.DEFAULT_MAX_TOKENS)) -> HTMLResponse:
    ok, err = bench.start(backend=backend, aliases=aliases, prompt_ids=prompt_ids,
                          reps=reps, max_tokens=max_tokens)
    return templates.TemplateResponse("_bench_progress.html", {
        "request": request, "job": bench.state(), "flash": "" if ok else err,
    })


@app.post("/benchmark/sweep", response_class=HTMLResponse)
def benchmark_sweep(request: Request,
                    backend: str = Form(...),
                    aliases: list[str] = Form(default=[]),
                    n_prompt: int = Form(512),
                    n_gen: int = Form(128),
                    depths: str = Form("0,4096,16384"),
                    reps: int = Form(3)) -> HTMLResponse:
    ok, err = bench.start_sweep(backend=backend, aliases=aliases, n_prompt=n_prompt,
                                n_gen=n_gen, depths=depths, reps=reps)
    return templates.TemplateResponse("_bench_progress.html", {
        "request": request, "job": bench.state(), "flash": "" if ok else err,
    })


@app.get("/benchmark/progress", response_class=HTMLResponse)
def benchmark_progress(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("_bench_progress.html", {
        "request": request, "job": bench.state(), "flash": "",
    })


@app.post("/benchmark/cancel", response_class=HTMLResponse)
def benchmark_cancel(request: Request) -> HTMLResponse:
    bench.cancel()
    return templates.TemplateResponse("_bench_progress.html", {
        "request": request, "job": bench.state(), "flash": "",
    })


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


@app.get("/benchmark/banner", response_class=HTMLResponse)
def benchmark_banner(request: Request) -> HTMLResponse:
    """Site-wide "a benchmark is running" strip. Polled from every page, so it stays cheap:
    it reads in-memory job state and touches neither docker nor the database."""
    return templates.TemplateResponse("_bench_banner.html", {
        "request": request, "job": bench.state(),
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
