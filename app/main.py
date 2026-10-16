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
from . import db, gguf_meta, gpu_procs, hf, hw, ini, services, vram_live
from . import strata as strata_engine
from . import strata_models
from . import router_api
from .config import settings
from .downloader import manager
from .fit import _fit_card_vram, _fit_host_ram, _preset_estimates
from .routers import benchmark as benchmark_router
from .routers import downloads as downloads_router
from .routers import models as models_router
from .routers import settings as settings_router
from .routers.models import _section_caps, _update_status_map
from .utils import human_bytes, shard_key
from .web import templates, _blocked_chip

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
        await asyncio.to_thread(services.sync_openwebui_capabilities)
    except Exception:  # noqa: BLE001 -- saving the section must not depend on OpenWebUI
        pass
    # A brand-new model is offered on the GPU backends by default. Without this it lands in
    # models.ini, works, and is invisible in OpenWebUI because every connection carries an
    # explicit whitelist that predates it.
    note = ""
    if is_new_section:
        try:
            ok, msg = await asyncio.to_thread(services.assign_new_model_to_gpu, name)
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
        # Autoconfig's whole budget is llama.cpp's: GPU-resident weights, its KV cache layout,
        # its per-card compute buffers, its argv. A backend of another engine holds a card and
        # is a real backend, but none of those numbers describe it - Strata streams experts
        # from host RAM against a cache it sizes itself - so listing it here would produce a
        # confident fit table for a stack that does not work that way, and an "assign to" that
        # writes llama arguments nothing will read.
        if services.engine_for(bn) != "llama":
            continue
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
                    "gpu_count": hw.gpu_count_for(bn), "card_vram_gb": _fit_card_vram(bn),
            "host_ram_gb": _fit_host_ram(),
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
        await asyncio.to_thread(telemetry.ingest,
                                await asyncio.to_thread(services._effective_container_names))
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
    def _reconcile() -> None:
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
    try:
        # Each whitelist write restarts open-webui (up to 30 s) - off the event loop.
        await asyncio.to_thread(_reconcile)
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
        "openwebui": await asyncio.to_thread(services.openwebui_state),
        # Raw model ids a backend reports — what OpenWebUI's model_ids whitelist matches on.
        # Every llama backend serves the same models.ini, so the section names are the list.
        "all_model_ids": sorted(ini.section_names()),
        "strata_pin": _strata_pin_state(backends),
    })


def _strata_pin_state(backends) -> dict | None:
    """Which card Strata is pinned to, which one it is actually running on, and the choices.

    None when there is no Strata backend at all, which is the normal case on a box without
    one and is what keeps the card out of the template.

    `file` and `live` are read from different places deliberately: the file is what the next
    start will use, `live` is what the running container was created with. They are equal
    except between a pin change and the restart that applies it, and saying so is the whole
    point of showing both.
    """
    b = next((x for x in backends if getattr(x, "engine", "llama") == strata_engine.ENGINE), None)
    if b is None:
        return None
    attrs = {}
    own = {}
    try:
        client = services._docker_client()
        if client is not None:
            attrs = client.containers.get(b.name).attrs or {}
            own = _strata_own_attrs(client)
    except Exception:  # noqa: BLE001 - a missing container just means no live pin to compare
        attrs = {}
    # Exec candidates for the topology read: this app has no /dev/kfd, so the card count comes
    # from a container that does. Any GPU-bearing backend will answer, Strata included.
    cands = [x.name for x in backends if x.status == "running" and x.vendor in ("rocm", "cuda")]
    n = strata_engine.card_count(tuple(cands))
    # The choices, as the values the form posts. Single cards first, then the split - which is
    # offered rather than recommended: upstream measured it LOSING on this exact pair, because
    # one 32 GB R9700 already holds every expert and the split only pays when no single card
    # does. Offering it anyway because that is a claim worth checking on hardware whose memory
    # bandwidth is half the box it was measured on.
    choices = [{"value": str(i), "label": f"card {i}"} for i in range(n)]
    if n > 1:
        split = ",".join(str(i) for i in range(n - 1, -1, -1))   # highest first = main card
        choices.append({"value": split, "label": f"both ({split} split)"})
    ts_idle = db.get_setting("strata_idle_unload_s", "")
    arena = hw.strata_arena_gb()
    avail = hw.host_ram_available_gb()
    return {
        "backend": b.name,
        "file": strata_engine.read_pin(),
        # The file IS the live pin while running: the documented entry point sources it on
        # every start, so any restart applies it, while docker inspect keeps reporting the
        # create-time environment until a recreate. (Measured: a re-pinned engine ran on
        # one card while its baked env still said 1,0 - showing that as "live" made the
        # card claim a restart was needed forever.)
        "live": strata_engine.read_pin() if b.status == "running" else "",
        "choices": choices,
        "running": b.status == "running",
        "config_note": strata_engine.config_note(
            strata_engine.run_config_path(attrs, own), services.strata_desired_keys()),
        "timeshare": {
            "idle_s": ts_idle,
            "min_free": db.get_setting("strata_min_free_vram_mib", ""),
            "yield_on": db.get_setting("strata_yield_on_load", "") == "1",
        },
        # The reclaim-stall zone: with the arena pinned, RAM the kernel can still take back
        # without touching GPU queues is what MemAvailable says it has. Under a few GB,
        # the next allocation of anyone else on the box is reclaimed FROM THE ENGINE, and
        # ROCm's reaction to that is to suspend KFD queues - the `verify: timed out` stall.
        "reclaim": {"arena": arena, "available": avail,
                    "warn": arena > 0 and 0 < avail < 8.0},
    }


@app.post("/containers/{name}/strata-pin", response_class=HTMLResponse)
def containers_strata_pin(name: str, card: str = Form(...)) -> HTMLResponse:
    """Pin Strata to one card, and restart it so the pin takes effect.

    The restart is part of the action rather than a second button because the file alone
    changes nothing: the engine reads it at start. Leaving the user to notice that is how a
    setting looks applied while the process is still on the old card.
    """
    if services.engine_for(name) != strata_engine.ENGINE:
        return HTMLResponse(
            '<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 '
            'px-3 py-2 text-sm">not a Strata backend</div>'
        )
    ok, msg = strata_engine.set_pin(card)
    if not ok:
        return HTMLResponse(
            f'<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 '
            f'px-3 py-2 text-sm">{msg}</div>'
        )
    services.restart_llama_backend(name)
    return HTMLResponse(
        f'<div class="rounded-md bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-300 '
        f'px-3 py-2 text-sm">{msg} — restarting {name}. It reloads ~55 GB into RAM, so give it '
        f'a minute or two before the port answers.</div>'
    )


def _strata_config_toast(ok: bool, msg: str) -> HTMLResponse:
    cls = ("bg-emerald-50 dark:bg-emerald-950/40 text-emerald-800 dark:text-emerald-300" if ok
           else "bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300")
    return HTMLResponse(
        f'<div class="rounded-md {cls} px-3 py-2 text-sm">{msg}</div>'
    )


@app.post("/containers/{name}/strata-timeshare", response_class=HTMLResponse)
def containers_strata_timeshare(name: str, idle_s: str = Form(""), min_free_mib: str = Form(""),
                                yield_on: str = Form("")) -> HTMLResponse:
    """Save the time-share settings and write them into the run config now.

    The write happens here, the effect happens at Strata's next start - server.py reads
    these keys when it boots. Nothing restarts Strata from this route, on purpose: the
    button that would apply the settings immediately is also the button that would drop
    whatever model is answering on that box right now, and that is the pin route's
    deliberate, warned-about job, not a side effect of Save.
    """
    if services.engine_for(name) != strata_engine.ENGINE:
        return _strata_config_toast(False, "not a Strata backend")
    vals = {}
    for key, raw in (("strata_idle_unload_s", idle_s.strip()),
                     ("strata_min_free_vram_mib", min_free_mib.strip())):
        if raw and (not raw.isdigit() or int(raw) <= 0):
            return _strata_config_toast(False, f"{key.split('strata_')[-1]}: a positive whole number, or empty for off")
        vals[key] = raw if raw and int(raw) > 0 else ""
    for key, val in vals.items():
        db.set_setting(key, val)
    db.set_setting("strata_yield_on_load", "1" if yield_on == "1" else "")

    keys = services.strata_desired_keys()
    try:
        client = services._docker_client()
        attrs = client.containers.get(name).attrs or {}
        path = strata_engine.run_config_path(attrs, _strata_own_attrs(client))
    except Exception as e:  # noqa: BLE001 - saved in the db either way; the note on the card covers the rest
        return _strata_config_toast(True, "saved — but the run config could not be reached "
                                          f"({e}); it will be applied at app start")
    if path is None:
        return _strata_config_toast(True, "saved — run config path unknown; it will be applied at app start")
    ok, msg = strata_engine.merge_run_config(path, keys)
    if not ok:
        return _strata_config_toast(True, f"saved — but not in the run config yet: {msg}")
    return _strata_config_toast(True, f"saved — run config: {msg}. Takes effect on {name}'s next start.")


@app.post("/internal/yield")
def internal_yield() -> Response:
    """Strata's before_load endpoint: give up the cards, because a reload is coming.

    No auth, like everything here (LAN-only tool, see the README) - and the worst a stray
    POST does is stop the llama backends, which is a pressable button elsewhere. The hook
    only exists at all while the Strata card has yield switched on, and a failed hook is a
    warning in Strata's log, not a refused load: a box with nothing to yield degrades to
    "load anyway".
    """
    return Response(services.strata_yield() + "\n", media_type="text/plain")


@app.post("/containers/{name}/strata-load", response_class=HTMLResponse)
def containers_strata_load(name: str) -> HTMLResponse:
    """Load Strata's parked model without touching the container.

    The whole point of the park/load pair: a container restart re-reads ~55 GB of experts
    and costs 90 s, while /load comes back from the OS page cache in seconds. Fire-and-
    forget - the endpoint blocks until the model answers - so the card's own poll is what
    shows the flip, and a refusal (usually "the GPU is in use by another program") surfaces
    on the card via services.strata_error().
    """
    if services.engine_for(name) != strata_engine.ENGINE:
        return _toast(False, "not a Strata backend")
    ok, msg = services.strata_load(name)
    return _toast(ok, msg)


@app.post("/containers/{name}/strata-unload", response_class=HTMLResponse)
def containers_strata_unload(name: str) -> HTMLResponse:
    """Park Strata: VRAM and pinned RAM back now, container and page cache left warm.

    The soft counterpart of Stop. It refuses (409) while a request is in flight rather than
    pulling the model out from under it, which is the right answer and worth seeing.
    """
    if services.engine_for(name) != strata_engine.ENGINE:
        return _toast(False, "not a Strata backend")
    ok, msg = services.strata_unload(name)
    return _toast(ok, msg)


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
    # Writes webui.db through docker exec and restarts open-webui (up to 30 s) - off the loop.
    ok, msg = await asyncio.to_thread(services.set_openwebui_model_filter, url, model_ids)
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
    # Same as the filter route: webui.db write + open-webui restart behind it.
    ok, msg = await asyncio.to_thread(
        services.toggle_openwebui_model, url, model_id, show, sorted(ini.section_names()))
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
        # A failed /load or /unload (usually the 503 "the GPU is in use by another program")
        # has nowhere else to surface: the load itself is fire-and-forget, so the card that
        # asked is the one that has to say why it is still parked. Clears itself once the
        # model is visibly loaded again.
        "strata_error": services.strata_error(name, loaded=bool(b.loaded_model)),
    })


@app.post("/gpu-tenants/{name}/{action}", response_class=HTMLResponse)
def gpu_tenant_power(name: str, action: str) -> HTMLResponse:
    """Eject (stop) or restore (start) a GPU-capable container that is not a llama backend.

    Stopping rather than calling the service's own unload endpoint, deliberately: every such
    service has a different API, or none, and docker is the one control that works for all of
    them. It also frees MORE - chatterbox's own unload leaves 1.34 GB of HIP context behind
    where stopping the container returns the whole 5.88 GB. The cost is a slower restore,
    measured ~10 s against ~5, which is a trade worth making for something ejected precisely
    because it is not being used.

    Refuses to touch a llama backend: those are the router's, and the Containers page owns them.
    """
    if name in set(services._effective_container_names()):
        return _toast(False, f"{name} is a llama backend - use the Containers page")
    if action not in ("eject", "restore"):
        return _toast(False, f"unknown action {action}")
    ok, msg = gpu_procs.set_container_running(name, start=(action == "restore"))
    return _toast(ok, msg)


@app.get("/engines/switch")
def engines_switch_state() -> dict:
    """What the running (or last) engine switch is doing. {} when none has run.

    JSON rather than a rendered panel, and a GET so the read-only guard is indifferent to it,
    because the caller is not a browser: the router process polls this to narrate a swap into a
    streaming chat response while the client waits.

    It has to be an HTTP read and not an import. `services._switch_state` is a module-level dict
    and `_switch_lock` a threading.Lock, so each process gets its OWN copy - and this container
    runs two, `app.main` on 8090 and `app.router_api` on 8086. The router importing
    activate_engine() directly would mean two schedulers, each holding the only lock it can see,
    racing over the same two cards: the exact failure the read-only guard was written to stop,
    reintroduced inside one container. So the UI process owns the switch and this is how anyone
    else finds out about it.
    """
    return services.engine_switch_state()


@app.post("/engines/{engine}/activate", response_class=HTMLResponse)
def engines_activate(engine: str, soft: str = Form(""), share: str = Form("")) -> HTMLResponse:
    """Hand the cards to one engine: release everything else holding a GPU, then start it.

    This exists because the two engines here cannot share the hardware and the only way to
    make one let go was an ssh session. A control that needs a terminal is not a control, and
    a llama router at --models-max 1 has no unload endpoint on this build - stopping the
    container is the only thing that makes it give the cards back.

    `soft` (the checkbox on the Engine row) asks backends that support it to unload instead:
    Strata gives the same VRAM and RAM back in ~0.3 s and keeps its container and settings. It
    does NOT make the return trip quick - measured at 104 s on 2026-10-15, no better than a
    cold start, because the other engine reads its own model in between and evicts the page
    cache the old claim here assumed would still be warm. Hard remains the default because a
    parked server can still be woken by a stray request, and "who owns the cards" deserves an
    answer that cannot change its mind.

    `share` (the Share button, offered only when a running engine demonstrably holds a
    strict subset of the cards) starts this engine and displaces nothing - the cards are
    divided, not handed over.
    """
    ok, msg = services.activate_engine(engine, soft=soft == "1", share=share == "1")
    return _toast(ok, msg)


@app.post("/containers/{name}/power", response_class=HTMLResponse)
def containers_power(name: str, action: str = Form(...)) -> HTMLResponse:
    """Start or stop one backend. The lifecycle control a backend card was missing.

    Restart was the only button, which is fine until two engines take turns on the same cards.
    Kept separate from the engine switch on purpose: this one touches exactly what it is told
    to and nothing else, which is what you want when the question is "is it this container".
    """
    if action not in ("start", "stop"):
        return _toast(False, f"unknown action {action}")
    ok, msg = services.set_backend_running(name, start=(action == "start"))
    return _toast(ok, msg)


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
