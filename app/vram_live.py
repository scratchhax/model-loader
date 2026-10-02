"""Live, per-component VRAM breakdown for a running backend.

The drivers only report a per-card TOTAL: nvidia-smi has never once said "this many bytes are
the KV cache". What makes a breakdown honest anyway is that llama-server allocates the weights,
the whole-context KV buffer and the compute buffers AT LOAD TIME and they do not move after
that. So the static components are estimated from what the router ACTUALLY loaded — the argv
telemetry already stores, the GGUF metadata, file sizes — while free space and an "other"
residual come from the measured usage. The bar therefore sums to the card exactly: the residual
absorbs what the estimate cannot know (allocator fragmentation, extra library workspaces), and
when the estimate itself is not trustworthy the caller falls back to the plain measured bar.

This is the live sibling of the autoconfig panel's recommended-configuration meter; both draw
from the same segments and the same palette, and both are built here rather than in templates
so a re-derivation cannot drift from the fit maths.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from . import autoconfig, db, gguf_meta, ini
from .config import settings

# Components are allocated at load, so they only change on respawn. A short TTL picks up a
# respawn (new argv, new model) without re-reading GGUF metadata on every 1-2 s poll, and it
# caches the failure modes too — a model whose metadata cannot be sized must not re-parse
# the whole GGUF every tick just to keep saying nothing.
_TTL_S = 30.0
_NEG_TTL_S = 10.0
_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, "dict | None"]] = {}


def _host_path(container_path: str) -> Path:
    """Map a container-side /models path onto this app's view of the same directory."""
    p = (container_path or "").strip()
    if p.startswith("/models"):
        p = str(settings.models_dir) + p[len("/models"):]
    return Path(p)


def _argv_int(argv: dict, *flags: str) -> int:
    for f in flags:
        v = argv.get(f)
        if v not in (None, "true"):
            try:
                return int(str(v).strip())
            except (TypeError, ValueError):
                continue
    return 0


def _components(name: str, vendor: str, gpu_count: int) -> dict | None:
    """Static per-load VRAM components for what this backend has loaded, or None when any
    input needed for an honest estimate is missing or contradictory."""
    cfg = db.latest_server_config(name)
    argv = cfg.get("argv") or {}
    path_s = cfg.get("model_path") or argv.get("--model") or argv.get("-m") or ""
    alias = cfg.get("alias") or ""
    if not path_s and alias:
        path_s = (ini.get_section(alias) or {}).get("model", "").strip()
    if not path_s:
        return None
    model = _host_path(path_s)
    try:
        file_size = model.stat().st_size
    except OSError:
        return None

    # n-cpu-moe offloads a per-layer share of the experts, which cannot be costed from the
    # file alone (the expert weight fraction is not in the metadata). Say nothing rather
    # than draw a weights segment that is wrong by an unknown ratio.
    if _argv_int(argv, "--n-cpu-moe", "-cmoe", "-ncmoe") > 0:
        return None

    ctx_total = _argv_int(argv, "--ctx-size", "-c")
    if ctx_total <= 0:
        return None

    try:
        summary = gguf_meta.summarize(gguf_meta.read_raw(model))
    except (gguf_meta.GgufMetaError, OSError, ValueError):
        return None
    g = autoconfig.model_geometry(summary)
    layers = g["layers"]
    if layers <= 0 or g["head_dim"] <= 0 or g["kv_heads"] <= 0:
        return None

    # GPU-resident share of the weights. llama loads everything on the GPU unless the argv
    # capped n-gpu-layers; an explicit lower cap is read straight off what was loaded.
    ngl = _argv_int(argv, "--n-gpu-layers", "-ngl", "-ngld")
    gpu_pct = 100 if (ngl <= 0 or ngl >= layers) else max(0, min(100, round(100.0 * ngl / layers)))
    overhead_mul = autoconfig._MODEL_OVERHEAD_SPLIT if gpu_count > 1 else autoconfig._MODEL_OVERHEAD_SINGLE
    model_gb = file_size / (1024 ** 3) * overhead_mul * gpu_pct / 100.0

    bytes_per = autoconfig._cache_dtype_bytes(argv.get("--cache-type-k") or autoconfig._CACHE_DEFAULT)
    kv_gb = autoconfig.kv_gb_for_geometry(g, ctx_total, bytes_per)
    if kv_gb <= 0:
        return None

    try:
        ubatch = int(str(argv.get("--ubatch-size") or argv.get("-ub") or 512).strip() or 512)
    except (TypeError, ValueError):
        ubatch = 512
    compute_gb = autoconfig.compute_buffer_gb(ctx_total, ubatch, vendor=vendor) * gpu_count

    reserve_gb = autoconfig._RESERVE_PER_GPU * gpu_count

    # Non-layer-split extras that live on the main GPU: exactly the ones the running server
    # was told to load — flags off its own argv, not what the ini merely allows.
    aux_gb = 0.0
    mmproj_s = (argv.get("--mmproj") or "").strip()
    if mmproj_s:
        try:
            mmproj_gb = _host_path(mmproj_s).stat().st_size / (1024 ** 3)
            aux_gb += mmproj_gb * autoconfig._MMPROJ_VRAM_MULT + autoconfig._MMPROJ_COMPUTE_GB
            hidden = int(g["embed"] or 0)
            if hidden > 0:
                # Vision ubatch bump, the same calibrated increase the fit maths charges.
                aux_gb += 7.0 * max(0, 1024 - 512) * layers * hidden / 1e9
        except OSError:
            pass
    draft_s = (argv.get("--spec-draft-model") or "").strip()
    if draft_s:
        try:
            aux_gb += _host_path(draft_s).stat().st_size / (1024 ** 3) * 1.15
        except OSError:
            pass

    if model_gb <= 0:
        return None
    return {"model_gb": model_gb, "kv_gb": kv_gb, "compute_gb": compute_gb,
            "reserve_gb": reserve_gb, "aux_gb": aux_gb,
            "gpu_pct": gpu_pct, "ctx_total": ctx_total, "model_label": model.name}


def _cached_components(name: str, vendor: str, gpu_count: int) -> dict | None:
    with _LOCK:
        entry = _CACHE.get(name)
        if entry and (time.time() - entry[0]) < (_TTL_S if entry[1] else _NEG_TTL_S):
            return entry[1]
    try:
        comps = _components(name, vendor, gpu_count)
    except Exception:  # noqa: BLE001 - a meter that fails must degrade to the plain bar
        comps = None
    with _LOCK:
        _CACHE[name] = (time.time(), comps)
    return comps


def _segments(model_gb, kv_gb, overhead_gb, compute_gb, other_gb, free_gb, total_gb,
              gpu_pct=100, ctx_total=0):
    """The part-to-whole segment list, ordered weights -> context -> overhead -> compute ->
    other -> free: the same order the autoconfig panel validated (it keeps the two hues the
    palette validator flags as a weak protanopic pair apart), with the two residual buckets
    neutrals at the end."""
    parts = [
        ("model", "model weights", model_gb,
         ("GPU-resident weights: %d%% of the model, the rest streams from host RAM" % gpu_pct)
         if gpu_pct < 100 else "the whole model, resident on the GPU"),
        ("ctx", "context (KV)", kv_gb,
         "KV cache for %s tokens (the whole pool across slots)" % f"{ctx_total:,}" if ctx_total
         else "KV cache"),
        ("overhead", "overhead", overhead_gb,
         "driver and runtime context per card, plus any vision projector and draft head"),
        ("compute", "compute buffers", compute_gb, "per-card graph scratch, which grows with context"),
    ]
    if other_gb > 0.005:
        parts.append(("other", "other", other_gb,
                      "in the measured total but not in the estimate: allocator fragmentation, "
                      "library workspaces, anything else sharing the device"))
    if free_gb > 0.005:
        parts.append(("free", "free", free_gb, "unallocated on the card"))
    return [{"key": k, "label": lbl, "gb": round(v, 2), "pct": 100.0 * v / total_gb, "note": note}
            for k, lbl, v, note in parts if v > 0.005]


def breakdown(name: str, gpu) -> dict | None:
    """Live part-to-whole VRAM breakdown for one backend, or None.

    None is the answer far more often than for the recommended-configuration meter, and every
    reason is a case where drawing an estimate would lie: no model loaded, no spawn telemetry
    (so the actual ctx and offload flags are unknown), the model file unreadable, an n-cpu-moe
    offload whose weights cost cannot be derived, a sleeping model whose measured usage has
    legitimately dropped below the allocation, or more than one backend on the cards (the
    estimate covers one of them and the residual would silently absorb the other).
    """
    if gpu is None or not getattr(gpu, "cards", None) or gpu.gpu_count <= 0:
        return None
    total_gb = float(gpu.vram_total_gb)
    used_gb = float(gpu.vram_used_gb)
    if total_gb <= 0 or used_gb <= 0:
        return None

    comps = _cached_components(name, gpu.vendor, gpu.gpu_count)
    if comps is None:
        return None

    est = comps["model_gb"] + comps["kv_gb"] + comps["reserve_gb"] + comps["aux_gb"] + comps["compute_gb"]
    # The allocation cannot exceed what the device measures for it. A wide overshoot means the
    # model went to sleep, the respawn has not been re-ingested yet, or the geometry estimate
    # is wrong — none of which a bar can distinguish from a truthful one, so show none.
    if est > total_gb * 1.02:
        return None
    scale = 1.0
    if est > used_gb:
        if used_gb < est * 0.85:
            return None
        scale = used_gb / est

    model_gb = comps["model_gb"] * scale
    kv_gb = comps["kv_gb"] * scale
    overhead_gb = (comps["reserve_gb"] + comps["aux_gb"]) * scale
    compute_gb = comps["compute_gb"] * scale
    other_gb = max(0.0, used_gb - est * scale)
    free_gb = max(0.0, total_gb - used_gb)

    return {
        "total_gb": total_gb, "used_gb": used_gb, "free_gb": free_gb,
        "subtitle": "%s @ %s, as loaded" % (comps["model_label"], autoconfig.format_ctx(comps["ctx_total"])),
        "segments": _segments(model_gb, kv_gb, overhead_gb, compute_gb, other_gb, free_gb,
                              total_gb, gpu_pct=comps["gpu_pct"], ctx_total=comps["ctx_total"]),
        "raw": dict(comps, scale=scale),
    }


def ram_breakdown(name: str, cont) -> dict | None:
    """Live part-to-whole meter of one CPU backend's OWN process footprint, or None.

    The CPU sibling of breakdown(), and it deliberately measures a different whole. A GPU
    meter's whole is the card - a fixed budget one load competes for. There is no such thing
    for RAM: it is the machine's shared pool, and llama.cpp mmaps the GGUF by default, so the
    weights are resident only by degree, evictable and shared. The one whole that means
    something is the model's own footprint, and that is what this draws: the whole is the
    llama-server process's measured RSS (hw's /proc probe), the weights segment is the GGUF's
    measured resident mmap, and KV/compute/other split the rest the same estimate+residual
    way as the GPU meter. No free segment - the bar adds up to the process, and what is not
    in it is not the process's to show.
    """
    rss = float(getattr(cont, "proc_rss_gb", 0) or 0)
    mm = float(getattr(cont, "model_resident_gb", 0) or 0)
    if rss <= 0 or mm < 0:
        return None

    comps = _cached_components(name, "cpu", 1)
    if comps is None:
        return None
    # No driver reserve on CPU - that 0.5 GB/card is a CUDA-context charge and belongs to
    # neither the process's anon memory nor its claim to honesty.
    kv_gb, compute_gb, aux_gb = comps["kv_gb"], comps["compute_gb"], comps["aux_gb"]
    weights_total = comps["model_gb"]

    # Same contract as the GPU meter: the anon estimates cannot exceed the anon memory
    # actually measured. Overshoot inside 15% rescales (estimates are calibrations, not
    # accounts); past that the meter stays home.
    anon_est = kv_gb + compute_gb + aux_gb
    anon_used = max(0.0, rss - mm)
    if anon_est > anon_used * 1.15 and anon_est > 0.05:
        return None
    scale = min(1.0, anon_used / anon_est) if anon_est > 0 else 1.0
    kv_gb, compute_gb, aux_gb = kv_gb * scale, compute_gb * scale, aux_gb * scale
    other_gb = max(0.0, rss - mm - (kv_gb + compute_gb + aux_gb))

    if mm >= weights_total * 0.99:
        w_note = "the whole model, resident in RAM"
        w_label = "model weights"
    else:
        w_note = ("%d%% of the model resident - it is mmap'd, the rest pages in from disk "
                  "as it is touched, and the OS takes it back under pressure"
                  % round(100.0 * mm / weights_total) if weights_total > 0 else "mmap'd model")
        w_label = "model weights (resident)"

    segments = _segments(mm, kv_gb, aux_gb, compute_gb, other_gb, 0.0, rss, ctx_total=comps["ctx_total"])
    # Always rewrite the weights segment: _segments' default note speaks of a GPU, and this
    # meter's weights are mmap'd RAM pages, resident by degree - a different claim entirely.
    for s in segments:
        if s["key"] == "model":
            s["label"], s["note"] = w_label, w_note

    return {
        "total_gb": rss, "used_gb": rss, "free_gb": 0.0,
        "subtitle": "%s @ %s, process footprint" % (comps["model_label"],
                                                    autoconfig.format_ctx(comps["ctx_total"])),
        "segments": segments,
    }


def per_card(vb: dict, cards: list) -> dict[int, dict]:
    """Split a pooled breakdown into per-card meters, keyed by card index.

    Known exactly per card: the driver reserve (one per card) and the non-split extras, which
    land on the main GPU. What is not known is where llama placed each layer — the sampler
    sees usage, not placement — so the weights+KV+compute pool is apportioned across the
    cards by their measured usage and drawn as ONE combined segment per card. Naming it
    "weights + KV + compute" is honest about what it is; inventing a per-card split the
    device never reported is not. Each card's segments still sum to that card, and the free
    share stays measured rather than modelled.
    """
    out: dict[int, dict] = {}
    if not vb or not cards:
        return out
    if len(cards) == 1:
        return {0: vb}

    raw = vb["raw"]
    pool = (raw["model_gb"] + raw["kv_gb"] + raw["compute_gb"]) * raw["scale"]
    fixed = []
    for i, c in enumerate(cards):
        # Card 0 also carries the projector/draft-head allocation; a card's own overhead can
        # never exceed what is measured on it.
        want = raw["reserve_gb"] + (raw["aux_gb"] if i == 0 else 0.0)
        fixed.append(min(want * raw["scale"], c.vram_used_gb))
    slack = [max(0.0, c.vram_used_gb - fixed[i]) for i, c in enumerate(cards)]
    slack_sum = sum(slack)
    for i, c in enumerate(cards):
        share = slack[i] / slack_sum if slack_sum > 0 else 1.0 / len(cards)
        alloc = min(pool * share, max(0.0, c.vram_used_gb - fixed[i]))
        other_gb = max(0.0, c.vram_used_gb - fixed[i] - alloc)
        free_gb = max(0.0, c.vram_total_gb - c.vram_used_gb)
        card_total = float(c.vram_total_gb or c.vram_used_gb)
        if card_total <= 0:
            continue
        out[i] = {
            "total_gb": card_total, "used_gb": c.vram_used_gb, "free_gb": free_gb,
            "subtitle": vb["subtitle"],
            "segments": [s for s in [
                {"key": "model", "label": "weights + KV + compute", "gb": round(alloc, 2),
                 "pct": 100.0 * alloc / card_total,
                 "note": "this card's share of the layer-split allocation, apportioned by usage"},
                {"key": "overhead", "label": "overhead", "gb": round(fixed[i], 2),
                 "pct": 100.0 * fixed[i] / card_total,
                 "note": "driver and runtime context"
                         + (", plus the projector and draft head, which pin to this card" if i == 0 else "")},
                {"key": "other", "label": "other", "gb": round(other_gb, 2),
                 "pct": 100.0 * other_gb / card_total,
                 "note": "in the measured total but not in the estimate"},
                {"key": "free", "label": "free", "gb": round(free_gb, 2),
                 "pct": 100.0 * free_gb / card_total,
                 "note": "unallocated on the card"},
            ] if s["gb"] > 0.005],
        }
    return out
