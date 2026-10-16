from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from . import autoconfig
from . import telemetry
from . import db, gpu_procs, hf, hw, ini, services, vram_live
from . import strata as strata_engine
from . import strata_models
from . import router_api
from .config import settings
from .downloader import manager
from .fit import _backend_list, _predicted_vram_gb, _preset_estimates, _solo_backend
from .routers import benchmark as benchmark_router
from .routers import config as config_router
from .routers import containers as containers_router
from .routers import downloads as downloads_router
from .routers import models as models_router
from .routers import settings as settings_router
from .utils import human_bytes
from .web import templates, _blocked_chip, _QUEUED_CHIP, _toast

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
    hw.start_sampler()
    threading.Thread(target=_reconcile_strata_config, daemon=True, name="strata-config").start()


def _strata_own_attrs(client) -> dict:
    """This app's own inspect data. Its mounts are the second half of the path translation:
    a host path only means something here once it has been mapped through what this
    container mounts where. Outside Docker there is nothing to translate."""
    try:
        return client.containers.get(socket.gethostname()).attrs or {}
    except Exception:  # noqa: BLE001 - a name docker does not know just means "no mapping"
        return {}


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
        own = _strata_own_attrs(client)
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
        arena = hw.strata_arena_gb()
        bits.append(f"{ram:.0f} GB RAM ({arena:.0f} pinned by strata)" if arena > 0
                    else f"{ram:.0f} GB RAM")
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


async def _hero_mtp(backend: str, model_id: str) -> dict | None:
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
            await asyncio.to_thread(telemetry.ingest, [backend])
            ent = (now, telemetry.stats_for(model_path="", alias=model_id))
        except Exception:  # noqa: BLE001 - a missing box, never a broken hero
            ent = (now, telemetry.Stats())
        _mtp_cache[key] = ent
    tel = ent[1]
    # Two shapes of draft wear the same spec-type. With a draft FILE the head is a second model
    # with its own weights and its own compute buffer - measured at 7.77 GB on the Q4_K_M. With
    # the MTP layers inside the target GGUF there is no file, no second set of weights, and
    # measured here no second buffer either: the per-card meter predicted card 0 at 28.80 GB
    # against 28.8 measured, residual 0.00, while draft-mtp was running at 72% acceptance.
    # Telling them apart matters, because the honest answer to "what does speculation cost me"
    # is 7.8 GB for one and approximately nothing for the other.
    draft_file = (argv.get("--model-draft") or argv.get("-md")
                  or argv.get("--spec-draft-model") or "").strip()
    return {
        "spec_type": stype,
        "builtin": not draft_file,
        "label": "MTP" if "mtp" in stype else ("NGRAM" if stype.startswith("ngram") else "SPEC"),
        "acc_pct": (100.0 * tel.draft_acc_p50) if tel.draft_acc_p50 is not None else None,
        "mean_len": tel.draft_len_p50 if tel.draft_len_p50 else 0.0,
        "n": tel.draft_n,
        "age": tel.age_str,
        "n_max": tel.draft_n_max,
    }


def _hero_foreign_only() -> list[dict]:
    """Foreign tenants that are working, for the standby panel.

    Nothing of ours loaded is precisely the case where a foreign tenant is the only thing on
    the cards, so the standby panel is the one that must not stay silent about it.
    """
    try:
        return [{"label": t.label, "cu_pct": t.cu_pct, "cu_peak": t.cu_peak,
                 "cu_total": t.cu_total, "gb": round(t.total_gb, 1),
                 "card": (max(t.cu_by_card, key=t.cu_by_card.get) if t.cu_by_card else None)}
                for t in _gpu_tenants() if t.foreign and t.working]
    except Exception:  # noqa: BLE001
        return []


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
        if not b.loaded_model and not b.loading_model and not b.sleeping_model:
            continue
        # Only a genuinely loaded model is probed. inference_speed reads /slots, and /slots
        # both resets the idle timer and wakes a sleeper - so probing a sleeping backend would
        # mean this page, polling every 500 ms, prevented sleep and woke whatever had managed
        # it. A loading model has no slots to read either.
        sp = (await services.inference_speed(b.name, b.internal_port, b.loaded_model)
              if b.loaded_model else None)
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
        #   is it loading        a swap in progress is the most interesting thing on the box
        #                           after something actually generating, and it is the state
        #                           that used to render as a blank panel for up to 40 seconds
        #
        # 3 puts it above any idle backend (2 + 1 at most) and below anything live (4 and up),
        # which is the order you would read them in out loud.
        #   is it asleep         it still owns a model and will serve it; it outranks nothing
        #                            and loses to everything, but it beats not being shown at
        #                            all, which is what used to happen
        _gpu = b.vendor in ("rocm", "cuda")
        rank = ((4 if (sp and sp.live) else 0) + (3 if b.loading_model else 0)
                + (2 if _gpu else 0) + (1 if b.router else 0))
        scored.append((rank, b, sp))
        if best is None or rank > best[0]:
            best = (rank, b, sp)
    if best is None:
        return {"speed": None, "hero_model": "", "hero_backend": "",
                "hero_shape": None, "hero_quant": "", "hero_size_h": "",
                "hero_ctx_cfg": "", "hero_vram_used": None, "hero_vram_total": 0.0,
                "tps_spark": "", "tps_peak": 0.0, "tps_samples": 0,
                "pp_spark": "", "pp_peak": 0.0, "pp_samples": 0, 
                "cx_spark": "", "cx_peak": 0.0, "cx_samples": 0, "hero_mtp": None,
                "hero_loading": False, "hero_loading_s": 0.0,
                "hero_foreign": _hero_foreign_only(), "hero_asleep": False,
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
            "model": (ob.loading_model or ob.loaded_model
                      or ob.sleeping_model or "").split(",")[0].strip(),
            "loading": bool(ob.loading_model),
            "gpu": ob.vendor in ("rocm", "cuda"),
            "live": bool(osp and osp.live),
            "state": (osp.state if osp else "idle"),
            "tps": round(osp.gen_tps, 1) if (osp and osp.live and osp.gen_tps) else 0.0,
            "asleep": bool(getattr(ob, "asleep", False)),
        })

    # Mid-swap the interesting model is the one arriving, not the nothing it replaced. Every
    # fact below keys off model_id, so naming it here means the size, quant and context all come
    # from the ini and the GGUF header exactly as they do for a resident model - which is the
    # whole of what the loading panel has to say, obtained for free.
    hero_loading = bool(b.loading_model)
    hero_asleep = bool(b.sleeping_model) and not b.loaded_model and not hero_loading
    model_id = (b.loading_model or b.loaded_model or b.sleeping_model or "").split(",")[0].strip()

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
        await asyncio.to_thread(telemetry.ingest, [b.name])
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

    # Anything on the cards that is NOT one of ours and is actually running kernels right now.
    # This is the whole answer to "the panel says idle while the box is clearly working": our
    # own backends can be idle while a TTS model or an image pipeline has a card at 70%.
    hero_foreign = []
    try:
        for t in _gpu_tenants():
            if t.foreign and t.working:
                hero_foreign.append({
                    "label": t.label, "cu_pct": t.cu_pct, "cu_peak": t.cu_peak,
                    "cu_total": t.cu_total, "gb": round(t.total_gb, 1),
                    "card": (max(t.cu_by_card, key=t.cu_by_card.get) if t.cu_by_card else None),
                })
    except Exception:  # noqa: BLE001 - a box without KFD simply has no tenants
        hero_foreign = []

    st = hw.stats_for(b.name)
    vram_used = st.gpu.vram_used_gb if (st.ok and st.gpu) else None
    vram_total = st.gpu.vram_total_gb if (st.ok and st.gpu) else 0.0

    # How much of this model is NOT on the cards, for the one-line readout under the context
    # bar. Only the numbers: the hero is glanced at, so the reasoning lives in the line's
    # tooltip and the full treatment stays on the autoconfig panel, where someone is reading
    # deliberately. None whenever nothing is offloaded, which is the common case.
    hero_weights_split = None
    try:
        # Not mid-load: the breakdown apportions VRAM against the RUNNING model's config, and
        # during a load the allocation on the cards belongs to neither the model leaving nor the
        # one arriving. A number assembled from both is worse than no number.
        if st.ok and st.gpu and not hero_loading and not hero_asleep:
            _t = _gpu_tenants()
            hero_weights_split = (vram_live.breakdown(
                b.name, st.gpu,
                frozenset(i.strip() for i in (b.loaded_model or "").split(",") if i.strip()),
                foreign=[(t.label, t.total_gb) for t in _t if t.foreign and t.total_gb > 0.005],
            ) or {}).get("weights_split")
    except Exception:  # noqa: BLE001 - a missing line must never cost the hero
        hero_weights_split = None

    # What speculation costs, merged into the MTP box so the price sits beside the payoff.
    # Only meaningful once there is a GPU to measure against, which is why it is here and not
    # in _hero_mtp - that function sees telemetry and argv, never the cards.
    hero_mtp = None if hero_loading else await _hero_mtp(b.name, model_id)
    if hero_mtp is not None and st.ok and st.gpu and not hero_asleep:
        try:
            hero_mtp["cost"] = vram_live.draft_cost(
                b.name, st.gpu,
                frozenset(i.strip() for i in (b.loaded_model or "").split(",") if i.strip()))
        except Exception:  # noqa: BLE001 - a missing cost must never cost the hero
            hero_mtp["cost"] = None

    # The sparkline is scaled to the peak in its own window, and that peak is printed beside it.
    # A fixed scale cannot work here: 27 tok/s is a busy dense model and a slow MoE one, so the
    # only honest options are a stated scale or a meaningless one.
    tps = services.tps_history(b.name)
    tps_peak = max(tps) if tps else 0.0
    # Sparse by construction: one point per completed prefill, not one per poll.
    pp = services.prefill_history(b.name)
    pp_peak = max(pp) if pp else 0.0
    cx = services.ctx_history(b.name)
    cx_peak = max(cx) if cx else 0.0
    return {"speed": sp, "hero_model": model_id, "hero_backend": b.name,
            "hero_shape": shape, "hero_quant": quant, "hero_size_h": size_h,
            "hero_ctx_cfg": ctx_cfg, "hero_vram_used": vram_used, "hero_vram_total": vram_total,
            # Blank mid-load. tps_history is per BACKEND, so the line drawn during a swap is
            # the outgoing model's throughput sitting under the incoming model's name - the one
            # reading of this panel that would be actively false rather than merely absent.
            "tps_spark": ("" if hero_loading
                          else hw.sparkline(tps, tps_peak or None) if len(tps) > 2 else ""),
            "tps_peak": 0.0 if hero_loading else tps_peak,
            "tps_samples": 0 if hero_loading else len(tps),
            "pp_spark": ("" if hero_loading
                         else hw.sparkline(pp, pp_peak or None) if len(pp) > 2 else ""),
            "pp_peak": 0.0 if hero_loading else pp_peak,
            "pp_samples": 0 if hero_loading else len(pp),
            # Against its own window peak, like the two above it: a 15k conversation in a 256k
            # window drawn against the CONFIGURED ceiling would be a flat line on the floor.
            "cx_spark": ("" if hero_loading
                         else hw.sparkline(cx, cx_peak or None) if len(cx) > 2 else ""),
            "cx_peak": 0.0 if hero_loading else cx_peak,
            "cx_samples": 0 if hero_loading else len(cx),
            "hero_loading": hero_loading,
            "hero_loading_s": b.loading_since if hero_loading else 0.0,
            "hero_foreign": hero_foreign,
            "hero_asleep": hero_asleep,
            "hero_loading_gpu": hero_loading and b.vendor in ("rocm", "cuda"),
            # Drive the per-slot strip's row height and column wrap. Computed here rather than
            # in the template so the thresholds sit with the dataclass they describe.
            "slot_density": services.slot_density(len(sp.slots) if sp else 0),
            "slot_columns": services.slot_columns(len(sp.slots) if sp else 0),
            "hero_weights_split": hero_weights_split,
            "hero_others": hero_others,
            # Speculation's scorecard is scraped from completed requests. There are none for a
            # model that has not finished loading, and the rollup would hand back its
            # predecessor's figures under the new model's name.
            "hero_mtp": hero_mtp}


def _gpu_tenants() -> list:
    """Every process on the cards, ours and not ours. [] on a box this cannot be read on.

    Lives here rather than in gpu_procs so the module stays a probe with no opinion about
    which containers the app considers its own.
    """
    try:
        ts = gpu_procs.tenants(set(services._effective_container_names()))
        gpu_procs.note_seen(ts)
        return ts
    except Exception:  # noqa: BLE001 - a tenant table that fails degrades to not being drawn
        return []


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
    live = {t.container: t.total_gb for t in tenants if t.foreign and t.container}
    free_gb = sum(max(0.0, c.vram_total_gb - c.vram_used_gb) for c in cards) if cards else 0.0
    ejectable = []
    # Exclude the LLAMA backends only, not every backend. A llama backend is excluded because
    # it has its own controls and its VRAM is the thing being broken down; a backend of another
    # engine is exactly what this button is for - it holds a whole card against a model the
    # router cannot evict, so taking the card back means stopping the container. Strata is the
    # first of those, and passing the full backend list here would have silently dropped it.
    _llama = [n for n in services._effective_container_names()
              if services.engine_for(n) == "llama"]
    for row in gpu_procs.gpu_capable_containers(_llama):
        # A RUNNING container earns a row only by actually holding VRAM. Plenty of things are
        # given /dev/kfd and allocate nothing - gpu-monitor is one - and offering to eject them
        # is noise that makes the real control harder to find. A STOPPED one always shows,
        # because being stopped is the whole reason you would want the button.
        if row["running"] and live.get(row["name"], 0.0) <= 0.005:
            continue
        want = live.get(row["name"]) or row["last_gb"]
        ejectable.append(dict(
            row, gb=live.get(row["name"], 0.0), want_gb=want,
            # Restoring needs the VRAM back. Saying so BEFORE the click is the whole value of
            # putting this in the app rather than leaving it to two curl commands: a chat model
            # with fit = on will have taken the freed space, and the only symptom otherwise is
            # a 503 from a service that looks like it started fine.
            tight=bool(not row["running"] and want > 0.005 and free_gb < want * 1.15),
            free_gb=free_gb))
    engines = services.gpu_engines()
    return {"cards": cards, "gpu_history": history, "card_breakdowns": card_breakdowns,
            "tenants": _tenant_rows(tenants, cards), "ejectable": ejectable,
            "engines": engines, "engine_switch": services.engine_switch_state(),
            "share": _share_options(engines, cards)}


def _share_options(engines: list, cards: list) -> dict[str, str]:
    """Engines that could start without displacing anyone, each with its honest one-liner.

    Sharing is on the table only when the cards demonstrably divide:

      * a running Strata pinned to a strict subset of the cards leaves the rest for llama -
        the pin is read from the container's own environment, so it names what the process
        actually holds, not what a file once said it would;
      * a stopped Strata whose pin file names one card can join a running llama - it sizes
        its expert cache against free VRAM at start, so the pin is the whole contract, and
        until min_free_vram_mib is set a wrong pin means a failed start, not a squeeze.

    Anything else - Strata across both cards, an unpinned Strata, a llama whose card
    membership is not readable from outside - shares nothing and gets no button, because a
    Share that silently OOMs one engine is worse than no button.
    """
    running = [e for e in engines if e["running"]]
    n = len(cards)
    if len(running) != 1 or n < 2:
        return {}
    out: dict[str, str] = {}
    if running[0]["engine"] == strata_engine.ENGINE:
        held = services.strata_held_cards()
        if held and len(held) < n and all(0 <= h < n for h in held):
            note = f"card {'+'.join(str(h) for h in held)} stays with strata"
            for e in engines:
                if not e["running"]:
                    out[e["engine"]] = note
    elif running[0]["engine"] == "llama":
        pin = strata_engine.read_pin()
        if pin.isdigit() and "," not in pin and int(pin) < n:
            for e in engines:
                if not e["running"] and e["engine"] == strata_engine.ENGINE:
                    out[e["engine"]] = (f"strata takes card {pin} and sizes itself against "
                                        "that card's free VRAM at start")
    return out


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
            # Who is WORKING, as opposed to merely resident. cu_total is 0 on a box whose
            # stats nodes cannot be read, and the template must then say nothing rather than
            # draw an idle badge - unknown and idle are different claims.
            "working": t.working,
            "cu_peak": t.cu_peak,
            "cu_total": t.cu_total,
            "cu_pct": t.cu_pct,
            "cu_card": (max(t.cu_by_card, key=t.cu_by_card.get) if t.cu_by_card else None),
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
    ctx.update(await asyncio.to_thread(_gpu_strip_context))
    ctx.update(_power_context())
    ctx.update(_host_stats_context())
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
    ctx.update(await asyncio.to_thread(_gpu_strip_context))
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


# Thresholds for the host row. Chosen against what this box actually does rather than generic
# sysadmin numbers: the things that go wrong here are a model sized past available RAM (an OOM
# kill, because there is no swap) and a models volume filling up mid-download (weights are
# 18-120 GB a piece, so "a few GB free" is already too late).
_LOAD_WARN_PCT = 80.0
_LOAD_CRIT_PCT = 100.0
_RAM_WARN_FREE_GB = 16.0
_RAM_CRIT_FREE_GB = 8.0
_DISK_WARN_FREE_GB = 150.0
_DISK_CRIT_FREE_GB = 60.0


def _host_stats_context() -> dict:
    """The host row: load, memory, swap, disk, at the cadence of the other meters.

    Four cheap reads (two /proc files, one statvfs, one cached cgroup probe), so this can sit
    on the same 1 s poll as the power row without costing anything. Deliberately NOT
    snapshot_models_dir(), which walks every GGUF to size it.

    Each card carries the figure a decision is made on, not the biggest number available:
    memory shows what is AVAILABLE rather than what is used, because the question is always
    "will the next model fit"; disk shows free rather than used, for the same reason.
    """
    load = hw.host_load()
    swap_used, swap_total = hw.host_swap_gb()
    free_gb, total_gb = hw.host_disk_gb(settings.models_dir)
    ram_total = hw.host_ram_gb()
    ram_avail = hw.host_ram_available_gb()
    arena = hw.strata_arena_gb()

    # Both drawn against a REAL ceiling rather than their own window: load against core count,
    # free RAM against total. The height of the line is then the headroom, which is the thing
    # being asked. Scaled to its own range, a quiet box and one about to run out would draw the
    # same picture - the argument the power panel already makes for its own sparklines.
    lh = hw.host_load_history()
    rh = hw.host_ram_history()
    return {"host": {
        "load": load,
        "load_spark": hw.sparkline(lh, 100.0) if len(lh) > 2 else "",
        "ram_spark": hw.sparkline(rh, ram_total or None) if len(rh) > 2 else "",
        "hist_n": len(lh),
        "load_warn": load["pct"] >= _LOAD_WARN_PCT and load["pct"] < _LOAD_CRIT_PCT,
        "load_crit": load["pct"] >= _LOAD_CRIT_PCT,

        "ram_total": ram_total,
        "ram_avail": ram_avail,
        "ram_used_pct": round(100.0 * (ram_total - ram_avail) / ram_total, 1) if ram_total else 0.0,
        "arena": arena,
        "ram_warn": 0 < ram_avail < _RAM_WARN_FREE_GB and ram_avail >= _RAM_CRIT_FREE_GB,
        "ram_crit": 0 < ram_avail < _RAM_CRIT_FREE_GB,

        "swap_used": swap_used,
        "swap_total": swap_total,
        # Swapping at all on this box is a problem rather than a cushion: the pinned expert
        # arena cannot be paged out, so pressure lands on everything else.
        "swap_warn": swap_total > 0 and swap_used > 0.5,

        "disk_free": free_gb,
        "disk_total": total_gb,
        "disk_used_pct": round(100.0 * (total_gb - free_gb) / total_gb, 1) if total_gb else 0.0,
        "disk_warn": 0 < free_gb < _DISK_WARN_FREE_GB and free_gb >= _DISK_CRIT_FREE_GB,
        "disk_crit": 0 < free_gb < _DISK_CRIT_FREE_GB,
        "disk_path": str(settings.models_dir),
    }}


@app.get("/hoststats", response_class=HTMLResponse)
def host_stats_partial(request: Request) -> HTMLResponse:
    ctx = {"request": request}
    ctx.update(_host_stats_context())
    return templates.TemplateResponse("_host_stats.html", ctx)


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

app.include_router(models_router.router)


# ---------- Settings ----------

app.include_router(settings_router.router)


# ---------- Search ----------

@app.get("/search", response_class=HTMLResponse)
def search_page(request: Request, engine: str = "") -> HTMLResponse:
    return templates.TemplateResponse("search.html",
                                      {"request": request, "query": "", "engine": engine})


@app.get("/search/results", response_class=HTMLResponse)
async def search_results(request: Request, q: str = "", sort: str = "downloads",
                         limit: int = 30, engine: str = "") -> HTMLResponse:
    # One search, two engines. Strata needs a file list per result to answer exactly, which is
    # why it is a chosen filter rather than something every search pays for: the llama path is
    # one Hub request, the Strata path is one per result.
    if engine == "strata":
        return await _strata_results_response(request, q)

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
    # A HINT, never a verdict. Deciding properly needs the repo's file list, and doing that for
    # every result is exactly the cost this page refuses to pay by default - so the name is used
    # to say "worth a look" and the Strata filter remains the authority. Worded as a question in
    # the UI for the same reason.
    maybe_strata = {
        m.id for m in results
        if strata_models.ARCH_MARKER in m.id.lower() and strata_models.GSQ_MARKER in m.id.lower()
    }
    return templates.TemplateResponse("_search_results.html", {
        "request": request, "results": results, "error": error, "avatars": avatars,
        "browse_mode": browse_mode, "browse_sort": sort,
        "downloaded_by_repo": downloaded, "maybe_strata": maybe_strata, "engine": engine,
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

app.include_router(downloads_router.router)


# ---------- models.ini ----------

app.include_router(config_router.router)


# ---------- Containers ----------

app.include_router(containers_router.router)


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
#
# The Strata half of Search HF. It is a SEPARATE page rather than a filter on the llama search
# because the two engines disagree about what a model is: llama takes any GGUF file, Strata
# takes a complete shard set of one architecture in one quantization format and prepares an
# expert pack from it. Mixing them into one result list would mean most rows carry a "not for
# this engine" badge, which is noise, not information.

STRATA_CONTAINER = os.environ.get("STRATA_CONTAINER", "strata")
STRATA_ENV_FILE = os.environ.get("STRATA_ENV_PATH", "/strata/strata.env")
STRATA_DATA = os.environ.get("STRATA_DATA_DIR", "/strata/data")


def _strata_state() -> dict:
    """Catalog, what is installed, and which pairs still have a free pack name."""
    tiers = strata_models.catalog(STRATA_CONTAINER)
    inst = strata_models.installed(STRATA_DATA)
    sel = strata_models.read_selection(STRATA_ENV_FILE)
    prov = strata_models.provenance()
    for tag, d in inst.items():
        d["about"] = strata_models.describe_install(tag, prov)
    return {
        "tiers": tiers,
        "installed": inst,
        "free": {t.install_tag for t in strata_models.free_slots(set(inst), tiers)},
        "selection": sel,
        "engine_up": bool(tiers),
        # {install_tag: repo} so a search row can say whether the slot it wants is held by THIS
        # repo (already installed) or by a different one (genuinely taken).
        "slot_repo": {t: (v.get("repo") or "") for t, v in prov.items()},
        # Which repos anything was installed FROM, independent of the catalog. Deriving this by
        # matching installed tags against catalog tiers looked equivalent and is not: the
        # catalog comes from the RUNNING engine, so rolling the engine back can drop a tier
        # that an installed pack still uses (0.1.40.2 has no swift/IQ3_S; 0.1.43 added it).
        # The pack still runs, so "you have this" must not stop being true because the engine
        # changed underneath it.
        "installed_repos": {(v.get("repo") or "") for v in prov.values() if v.get("repo")},
        # Tuning applies to the model that will RUN next, which is the one the selection names -
        # not whatever happens to be loaded, because a change made now lands on the next start.
        "tuning_tag": _strata_selected_tag(sel),
        "tunables": strata_models.TUNABLES,
    }


def _strata_selected_tag(sel: dict) -> str:
    """The install tag the pin file's FAMILY/MODEL names — the same tag the entry point builds."""
    fam, mod = (sel.get("FAMILY") or "").strip(), (sel.get("MODEL") or "").strip()
    if not mod:
        return ""
    prefix = "" if fam in ("", "qwen") else f"{fam}-"
    return f"{prefix}{mod.lower()}"


@app.post("/strata/tuning", response_class=HTMLResponse)
def strata_tuning(install_tag: str = Form(...), flag: str = Form(...),
                  value: str = Form(...)) -> HTMLResponse:
    ok, msg = strata_models.set_tunable(install_tag, flag, value, STRATA_CONTAINER, STRATA_DATA)
    if not ok:
        return _blocked_chip(msg)
    return HTMLResponse(_QUEUED_CHIP.format(label="Saved", title=msg))


@app.get("/strata/models", response_class=HTMLResponse)
async def strata_models_page(request: Request) -> HTMLResponse:
    st = _strata_state()
    owners = [d["about"]["owner"] for d in st["installed"].values() if d["about"]["owner"]]
    try:
        avatars = await hf.owner_avatars(owners) if owners else {}
    except httpx.HTTPError:
        avatars = {}      # an icon is decoration; never fail the page for one
    metrics = services.strata_metrics(STRATA_CONTAINER)
    st["tuning"] = (strata_models.tuning_state(st["tuning_tag"], metrics, STRATA_DATA)
                    if st["tuning_tag"] else {})
    st["engine_running"] = metrics is not None
    return templates.TemplateResponse("strata_models.html",
                                      {"request": request, "query": "", "avatars": avatars, **st})


@app.get("/strata/search/results", response_class=HTMLResponse)
async def strata_search_results(request: Request, q: str = "") -> HTMLResponse:
    """Kept so an existing link or bookmark still works; the page itself now searches through
    /search?engine=strata, so there is one search box in the app rather than two."""
    return await _strata_results_response(request, q)


async def _strata_results_response(request: Request, q: str = "") -> HTMLResponse:
    """HF search, with every repo classified against the engine's catalog.

    Repos that match nothing are kept and shown as refused WITH THE REASON. Hiding them would
    be worse: the common case is a Qwen3.8-Flash-Next repo in a quantization Strata cannot
    read, and a user who searched for it deserves to be told that rather than shown an empty
    page that looks like a broken search.
    """
    st = _strata_state()
    error: str | None = None
    rows: list[dict] = []
    avatars: dict[str, str] = {}
    if not st["engine_up"]:
        error = ("The Strata container is not running, so its model catalog cannot be read. "
                 "Start Strata from the overview and search again.")
    else:
        try:
            found = await hf.search_models(q.strip() or "Qwen3.8-Flash-Next", limit=25)

            # Classifying a repo needs its FILE LIST, which is a second request per result, and
            # done one at a time that is the whole latency of this page: 25 repos x ~0.5 s was
            # ~13 s of mostly-idle waiting. They are independent, so they go together.
            #
            # Bounded, not unbounded: a bare gather would open 25 simultaneous connections to
            # the Hub per keystroke-driven search, which is how an app earns a 429 and then
            # shows nothing at all. Eight keeps it one round of waiting without being rude.
            sem = asyncio.Semaphore(8)

            async def _detail(m):
                async with sem:
                    try:
                        return m, await hf.repo_detail(m.id)
                    except httpx.HTTPError:
                        return m, None

            # Order is the Hub's relevance order, and gather preserves it. Sorting by
            # installability instead was tempting and wrong: it would bury the repo someone
            # searched for by name under whatever happens to be installable.
            for m, detail in await asyncio.gather(*(_detail(m) for m in found)):
                if detail is None:
                    continue
                files = [{"path": f.path, "size": f.size} for f in detail.files]
                matches = strata_models.classify_repo(files, st["tiers"])
                rows.append({"repo": m, "matches": matches,
                             "reason": "" if matches else _strata_refusal(files)})
            # Owner avatars, the same call and the same cache the llama search uses, so a repo
            # looks like itself on both pages instead of like two different things.
            avatars = await hf.owner_avatars([r["repo"].id.split("/", 1)[0] for r in rows])
        except httpx.HTTPStatusError as e:
            error = f"HF returned HTTP {e.response.status_code}"
        except httpx.HTTPError as e:
            error = f"network error: {e}"
    return templates.TemplateResponse("_strata_results.html", {
        "request": request, "rows": rows, "error": error, "avatars": avatars,
        "installed": st["installed"], "free": st["free"], "slot_repo": st["slot_repo"],
        "installed_repos": st["installed_repos"],
    })


def _strata_refusal(files: list[dict]) -> str:
    """Why a repo matched nothing — named precisely, because "unsupported" teaches nothing."""
    names = [f["path"].rsplit("/", 1)[-1].lower() for f in files if f["path"].lower().endswith(".gguf")]
    if not names:
        return "no GGUF files"
    if not any(strata_models.ARCH_MARKER in n for n in names):
        return "not Qwen3.8-Flash-Next — Strata serves only that architecture"
    if not any(strata_models.GSQ_MARKER in n for n in names):
        return ("Qwen3.8-Flash-Next, but not GSQ-RCO quantized — Strata's pack builder reads "
                "GSQ-RCO files (and Unsloth's own UD-IQ4_XS / UD-Q4_K_XL, which it downloads itself)")
    return "GSQ-RCO, but no complete shard set for a quant tier this engine lists"


@app.post("/strata/install", response_class=HTMLResponse)
async def strata_install(repo_id: str = Form(...), family: str = Form(...),
                         model: str = Form(...)) -> HTMLResponse:
    """Plan the install, place what we already have, and queue only what is missing."""
    st = _strata_state()
    tier = next((t for t in st["tiers"] if t.family == family and t.model == model), None)
    if tier is None:
        return _blocked_chip("that family/tier is not in the engine's catalog")
    if tier.install_tag in st["installed"]:
        return _blocked_chip(f"{tier.install_tag} is already installed — remove it first")

    try:
        detail = await hf.repo_detail(repo_id)
    except httpx.HTTPError as e:
        return _blocked_chip(f"cannot read {repo_id}: {e}")
    files = [{"path": f.path, "size": f.size} for f in detail.files]
    match = next((m for m in strata_models.classify_repo(files, [tier])), None)
    if match is None:
        return _blocked_chip("this repo no longer matches that tier")

    # Content hashes first: without them every reuse check degrades to "same size", which is
    # not good enough to act on (see plan_install).
    paths = [f["src"] for f in match.files] + ([match.mmproj["src"]] if match.mmproj else [])
    etags = await strata_models.fetch_etags(repo_id, paths)
    for f in match.files:
        f["sha"] = etags.get(f["src"], "")
    if match.mmproj:
        match.mmproj["sha"] = etags.get(match.mmproj["src"], "")

    subdir = f"strata-{repo_id.split('/')[-1]}-{model}"
    plan = await asyncio.to_thread(strata_models.plan_install, match,
                                   str(settings.models_dir), subdir)
    await asyncio.to_thread(strata_models.materialize_reuse, plan)

    for f in plan["fetch"]:
        manager.enqueue(repo_id=repo_id, hf_path=f["src"],
                        filename=f"{subdir}/{f['dst']}", total_bytes=f["size"])

    # Recorded at QUEUE time, not on completion: the slot is spoken for from here on, and a
    # download that is interrupted still leaves files whose origin someone will need to know.
    strata_models.record_provenance(tier.install_tag, repo_id, family, model,
                                   tier.canonical_repo)

    saved = plan["saved_bytes"]
    note = f", reusing {human_bytes(saved)} already on disk" if saved else ""
    return HTMLResponse(_QUEUED_CHIP.format(
        label=f"Queued {human_bytes(plan['fetch_bytes'])}",
        title=f"{len(plan['fetch'])} file(s) queued into {subdir}{note}"))


@app.post("/strata/select", response_class=HTMLResponse)
def strata_select(family: str = Form(...), model: str = Form(...),
                  gguf_dir: str = Form("")) -> HTMLResponse:
    """Point the next Strata start at an installed pair. Takes effect on restart, not now."""
    ok, msg = strata_models.set_selection(STRATA_ENV_FILE, family, model, gguf_dir)
    if not ok:
        return _blocked_chip(msg)
    return HTMLResponse(_QUEUED_CHIP.format(label="Saved", title=msg + " — restart Strata to apply"))


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
        label = f"{name}  ({owner})" if owner else name
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
    since = _router_since(range)
    bs, by_model = await router_api._live_map()
    per_engine: dict[str, list] = {}
    for b in bs:
        if b["engine"] == "strata":
            continue                       # built from installed packs below, not from /v1/models
        per_engine.setdefault(b["engine"], [])
        for m in b["models"]:
            if m not in per_engine[b["engine"]]:
                per_engine[b["engine"]].append(m)
    per_engine["llama"] = [(m, m) for m in per_engine.get("llama", [])]
    per_engine["strata"] = _strata_choices()
    return templates.TemplateResponse("router.html", {
        "request": request,
        "engines": router_api.ENGINES,
        "per_engine": per_engine,
        "defaults": {e: router_api.default_for(e) for e in router_api.ENGINES},
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
    })


@app.post("/router/default", response_class=HTMLResponse)
def router_set_default(engine: str = Form(...), model: str = Form("")) -> HTMLResponse:
    if engine not in router_api.ENGINES:
        return _blocked_chip(f"unknown engine {engine!r}")
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
            return HTMLResponse(_QUEUED_CHIP.format(
                label="Noted", title=f"{tag} recorded as the Strata default. This instance is "
                                     f"read-only, so apply it from the live Model Loader's "
                                     f"Strata tab ({fam} / {mdl})"))
        ok, msg = strata_models.set_selection(STRATA_ENV_FILE, fam, mdl, "")
        return HTMLResponse(_QUEUED_CHIP.format(
            label="Saved" if ok else "Failed",
            title=msg + (" — restart Strata to apply" if ok else "")))
    return HTMLResponse(_QUEUED_CHIP.format(
        label="Saved", title=f"{engine} now defaults to {model or '(none)'}"))


@app.post("/router/backend", response_class=HTMLResponse)
def router_set_backend(engine: str = Form(...), backend: str = Form("")) -> HTMLResponse:
    """Pin an engine's traffic to one container, or clear it back to automatic."""
    if engine not in router_api.ENGINES:
        return _blocked_chip(f"unknown engine {engine!r}")
    db.set_setting(f"router.backend.{engine}", backend.strip())
    return HTMLResponse(_QUEUED_CHIP.format(
        label="Saved",
        title=f"{engine} pinned to {backend}" if backend.strip()
              else f"{engine} back to automatic (CPU last, purpose-named servers after the "
                   f"general router)"))


@app.post("/router/recording", response_class=HTMLResponse)
def router_set_recording(on: str = Form("1")) -> HTMLResponse:
    """The kill switch. Off, the router still routes but writes nothing.

    It exists so that if accounting ever misbehaves - a lock, a slow write, a bad migration -
    turning it off is one click rather than editing config and restarting under pressure, while
    inference keeps working.
    """
    db.set_setting("router.record", "1" if on == "1" else "0")
    return HTMLResponse(_QUEUED_CHIP.format(
        label="Saved", title="recording on" if on == "1" else "recording off — routing continues"))


@app.post("/router/autoswitch", response_class=HTMLResponse)
def router_set_autoswitch(engine: str = Form(...), on: str = Form("")) -> HTMLResponse:
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
    return HTMLResponse(_QUEUED_CHIP.format(
        label="Saved",
        title=(f"requests for {engine} may switch the engines"
               if on == "1" else f"requests for {engine} will be refused instead of switching")))
