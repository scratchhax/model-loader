from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db, gpu_activity, gpu_procs, hf, hw, ini, services, telemetry, vram_live
from .. import strata as strata_engine
from .. import strata_models
from ..config import settings
from ..downloader import manager
from ..fit import _gpu_tenants, _solo_backend
from ..utils import human_bytes
from ..web import templates
from .power import _host_stats_context, _power_context, _power_settings_context

router = APIRouter()


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
    # A backend sees only the cards it was given; on a box with one container per card the
    # host's own list is the whole of it.
    total_vram = max(total_vram, sum(c.vram_total_gb for c in hw.host_nvidia_cards()))
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
                 "cu_total": t.cu_total, "util_src": t.util_src, "gb": round(t.total_gb, 1),
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
        # Ranked by what matters on a box whose whole point is the cards:
        #
        #   generating now (4)   the thing actually working always leads, GPU or not
        #   loading (3)          a swap in progress is the most interesting thing after that
        #   touches a GPU (2)    an idle CPU backend is not the headline while a card holds a model
        #   chat router (1)      a TTS server holds its model permanently and so always counts as
        #                        loaded; it must not outrank the model you are talking to
        #
        # A sleeping model gets no points: it still owns a model and is shown, but outranks
        # nothing. Without the GPU and router terms the three backends that share one models.ini
        # (the GPU router, llama-cpu, llama-voice) tie, and docker's list order picks the hero.
        _gpu = b.vendor in ("rocm", "cuda")
        rank = ((4 if (sp and sp.live) else 0) + (3 if b.loading_model else 0)
                + (2 if _gpu else 0) + (1 if b.router else 0))
        scored.append((rank, b, sp))
        if best is None or rank > best[0]:
            best = (rank, b, sp)
    if best is None:
        return {"speed": None, "hero_model": "", "hero_backend": "", "hero_engine": "",
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
    # every 20 s internally, so an HTMX poll does not re-read the log every two seconds. Done
    # here, not only inside _hero_mtp(), because that returns early when the RECORDED argv
    # declares no --spec-type, and the record is exactly what goes stale after a swap.
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
                    "cu_total": t.cu_total, "util_src": t.util_src, "gb": round(t.total_gb, 1),
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
    hero_engine = services.engine_for(b.name)
    # A Strata model in a borrowed catalog slot is served under that slot's name; say what it is.
    hero_alias = strata_models.served_labels().get(model_id) if hero_engine == "strata" else None
    return {"speed": sp, "hero_model": model_id, "hero_backend": b.name,
            "hero_engine": hero_engine, "hero_alias": hero_alias,
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
    host_cards = hw.host_nvidia_cards()
    to_host: dict[int, int] = {}
    for name in services._effective_container_names():
        st = hw.stats_for(name)
        if not (st.ok and st.gpu and st.gpu.cards):
            continue
        cards = st.gpu.cards
        to_host = _host_index_map(cards, host_cards)
        # Tenants are numbered by the host; the backend's meter by its own view. Hand the meter
        # the tenants on ITS cards under ITS numbers, or a container pinned to the host's second
        # card would be told about whatever sits on the first.
        if to_host:
            foreign_cards_local = {i: foreign_cards.get(h, []) for i, h in to_host.items()}
        else:
            foreign_cards_local = foreign_cards
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
            card_breakdowns = vram_live.per_card(pooled_vb, cards, foreign_cards_local)
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
    # Widen to the host's cards when the backends between them see fewer. Everything keyed by
    # the backend's numbering is re-keyed by the host's, so the strip, the tenant table and the
    # foreign tenants all count cards the same way. A card no backend meters gets the
    # measurement-only meter below.
    if len(host_cards) > len(cards):
        if to_host:
            card_breakdowns = {to_host[i]: v for i, v in card_breakdowns.items() if i in to_host}
            history = {to_host[i]: v for i, v in history.items() if i in to_host}
        elif cards:
            card_breakdowns, history = {}, {}
        cards = host_cards
        unmetered = [c for c in cards if c.index not in card_breakdowns]
        card_breakdowns.update(vram_live.tenants_only_per_card(unmetered, foreign_cards))
    elif not card_breakdowns and foreign_cards and cards:
        # The estimate-based meter is all-or-nothing on the backend's model and says None
        # whenever it cannot size one - idle router, sleeping model, no telemetry. A foreign
        # tenant's VRAM is measured, not estimated, so it should not disappear with it. Fall
        # back to drawing what IS known rather than an anonymous fill.
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


def _host_index_map(cards: list, host_cards: list) -> dict[int, int]:
    """{backend's card index: host's card index}, matched by PCI bus. {} when nothing matches -
    on AMD, where both number the cards the same way, or without host-sensors' NVIDIA table."""
    by_bus = {c.bus: c.index for c in host_cards if c.bus}
    return {c.index: by_bus[c.bus] for c in cards if getattr(c, "bus", "") in by_bus}


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
            "util_src": t.util_src,
            "cu_card": (max(t.cu_by_card, key=t.cu_by_card.get) if t.cu_by_card else None),
        })
    return rows


async def _llm_activity() -> dict[str, dict]:
    """What each llama backend is doing, keyed by container name, for its row on a card.

    The same probe and the same cache the hero reads, so the card rows and the hero can never
    disagree, and asking twice costs nothing: inference_speed is cached per backend and touches
    /slots only while a task is in flight (an idle model is left alone so it can sleep).
    """
    out: dict[str, dict] = {}
    for b in await _hero_backends_cached():
        model = (b.loading_model or b.loaded_model or b.sleeping_model or "").split(",")[0].strip()
        sp = (await services.inference_speed(b.name, b.internal_port, b.loaded_model)
              if b.loaded_model else None)
        live = bool(sp and sp.live)
        out[b.name] = {
            "model": model,
            "loading": bool(b.loading_model),
            "asleep": bool(b.sleeping_model) and not b.loaded_model and not b.loading_model,
            "state": sp.state if live else "idle",
            "tps": round(sp.gen_tps, 1) if (live and sp.gen_tps) else 0.0,
            "prefill_pct": sp.prefill_pct if live else 0,
            "slots_busy": sum(1 for s in sp.slots if s.state != "idle") if live else 0,
        }
    return out


def _cards_context(llm: dict[str, dict]) -> dict:
    """One panel per card: who holds it, who is working on it, and what that work is.

    Built on _gpu_strip_context rather than beside it, so the card list, the host renumbering,
    the eject/restore rows and the engine switch all come from the one place that already gets
    them right; this adds the per-tenant view and the timeline on top.
    """
    strip = _gpu_strip_context()
    cards = strip["cards"]
    tenants = _gpu_tenants()
    ejectable = {e["name"]: e for e in strip["ejectable"]}
    panels = []
    for c in cards:
        rows: dict[str, dict] = {}
        for t in tenants:
            gb = t.per_card_gb.get(c.index, 0.0)
            cu = t.cu_by_card.get(c.index, 0)
            if gb < 0.01 and not cu:
                continue
            # One row per container, not per pid: a llama router runs a child per model, and
            # two pids of one tenant are one thing to the person reading the card.
            r = rows.setdefault(t.label, {
                "label": t.label, "kind": gpu_activity.kind_of(t.label, not t.foreign),
                "backend": t.backend, "gb": 0.0, "pct": 0, "known": False, "pids": []})
            r["gb"] += gb
            r["pids"].append(t.pid)
            if t.cu_total:
                r["known"] = True
                r["pct"] = max(r["pct"], round(100.0 * cu / t.cu_total))
        for r in rows.values():
            r["working"] = r["pct"] > 0
            r["llm"] = llm.get(r["backend"]) if r["backend"] else None
            r["comfy"] = (gpu_activity.comfy_status(r["label"])
                          if r["kind"] == "image" and "comfy" in r["label"].lower() else None)
            # The tool's own account beats the compute sample: a render between two pmon
            # samples is still a render, and a chat model mid-generation is working even on
            # the one sample that caught it waiting on the CPU.
            if r["comfy"] and r["comfy"].busy:
                r["working"] = True
            if r["llm"] and (r["llm"]["state"] != "idle" or r["llm"]["loading"]):
                r["working"] = True
            r["eject"] = ejectable.get(r["label"])
        ordered = sorted(rows.values(), key=lambda r: (not r["working"], -r["gb"]))
        held = sum(r["gb"] for r in ordered)
        other = max(0.0, c.vram_used_gb - held)
        total = c.vram_total_gb or 1.0
        segs = [{"label": r["label"], "kind": r["kind"], "gb": r["gb"],
                 "pct": 100.0 * r["gb"] / total} for r in sorted(ordered, key=lambda r: -r["gb"])]
        # Driver context and anything nvidia-smi did not attribute. Shown only when it is big
        # enough to see, and named for what it nearly always is rather than as a mystery.
        if other > 0.15:
            segs.append({"label": "other", "kind": "other", "gb": other, "pct": 100.0 * other / total})
        panels.append({
            "card": c, "rows": ordered, "segs": segs,
            "free_gb": max(0.0, c.vram_total_gb - c.vram_used_gb),
            "working": [r for r in ordered if r["working"]],
            "timeline": gpu_activity.stacked_bands(gpu_activity.timeline(c.index)),
        })
    stopped = [e for e in strip["ejectable"] if not e["running"]]
    return {"panels": panels, "stopped": stopped, "now": time.time(), "engines": strip["engines"],
            "engine_switch": strip["engine_switch"], "share": strip["share"]}


@router.get("/cards", response_class=HTMLResponse)
async def cards_partial(request: Request) -> HTMLResponse:
    ctx = {"request": request}
    llm = await _llm_activity()
    ctx.update(await asyncio.to_thread(_cards_context, llm))
    backends = await _hero_backends_cached()
    ctx["parked_engines"] = {services.engine_for(b.name) for b in backends
                             if getattr(b, "unloaded_model", None)}
    return templates.TemplateResponse("_cards.html", ctx)


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    snap = services.snapshot_models_dir()
    backends = await services.snapshot_llama_backends()
    # What downloaded last does not tell you anything about what the box is doing now, so the
    # list is gone from this page; only a download still in flight is worth a line.
    active = sum(1 for j in manager.snapshot() if j.status in ("queued", "downloading"))
    ctx = {"request": request, "snap": snap, "backends": backends,
           "ini_sections": ini.list_sections(),
           "active_downloads": active, "host_line": _host_line(), "cards_above": True}
    ctx.update(await _hero_context())
    ctx.update(await asyncio.to_thread(_cards_context, await _llm_activity()))
    ctx["parked_engines"] = {services.engine_for(b.name) for b in backends
                             if getattr(b, "unloaded_model", None)}
    ctx.update(_power_context())
    ctx.update(_host_stats_context())
    ctx.update(_power_settings_context())
    return templates.TemplateResponse("dashboard.html", ctx)


@router.get("/v2")
def dashboard_v2_redirect() -> RedirectResponse:
    """This lived at /v2 while it was being built; send stale tabs and bookmarks to the front."""
    return RedirectResponse("/", status_code=308)


@router.get("/hero", response_class=HTMLResponse)
async def hero_partial(request: Request) -> HTMLResponse:
    ctx = {"request": request, "cards_above": True}
    ctx.update(await _hero_context())
    return templates.TemplateResponse("_hero.html", ctx)


@router.get("/gpu-strip", response_class=HTMLResponse)
async def gpu_strip_partial(request: Request) -> HTMLResponse:
    ctx = {"request": request}
    ctx.update(await asyncio.to_thread(_gpu_strip_context))
    # A PARKED engine's container is running but holds no cards (Strata after /unload). The
    # engine row read "running" as "holding the cards", so a parked Strata said "active" beside
    # the engine that actually held them, and its button was DISABLED - the one way back to it
    # from the strip was greyed out. The backend probe already knows: a single-model server whose
    # one model reads unloaded. Cached (1.5 s), so this adds no probe of its own.
    backends = await _hero_backends_cached()
    ctx["parked_engines"] = {services.engine_for(b.name) for b in backends
                             if getattr(b, "unloaded_model", None)}
    return templates.TemplateResponse("_gpu_strip.html", ctx)
