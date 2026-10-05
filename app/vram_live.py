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
from .utils import shard_key

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


def kfd_process_vram_gb() -> float:
    """Total VRAM held by the largest single GPU process, in GB, or 0.0 when unreadable.

    The amdgpu kernel driver publishes exactly this per process and per card, which is the one
    number llama.cpp will not tell you and the drivers' per-card totals cannot separate:

        /sys/class/kfd/kfd/proc/<pid>/vram_<gpuid>   bytes, one file per card

    Readable straight from this container - /sys is already mounted, no rocm-smi binary and no
    docker exec needed - and it agrees with `rocm-smi --showpids` to the byte. PIDs there are
    the host's and this process cannot map them to a container, but it does not need to: under
    `--models-max 1` the router keeps one child resident, so the largest consumer IS the loaded
    model. Taking the largest rather than the sum is what makes that safe - a second tenant on
    the cards inflates the total but not the maximum.
    """
    root = Path("/sys/class/kfd/kfd/proc")
    best = 0
    try:
        for proc in root.iterdir():
            total = 0
            for f in proc.glob("vram_*"):
                try:
                    total += int(f.read_text().strip() or 0)
                except (OSError, ValueError):
                    continue
            best = max(best, total)
    except OSError:
        return 0.0
    return best / (1024 ** 3)


def _components(name: str, vendor: str, gpu_count: int,
                loaded_ids: frozenset[str] | None = None) -> dict | None:
    """Static per-load VRAM components for what this backend has loaded, or None when any
    input needed for an honest estimate is missing or contradictory.

    `loaded_ids` is what the router says is resident RIGHT NOW. It has to be checked, because
    everything here describes a SPAWN RECORD scraped from the log and those two can disagree:
    telemetry ingests on a 20 s timer, so for the first seconds of a new model the newest
    recorded spawn is still the previous one. Pairing a live measurement with the wrong
    model's file does not degrade gracefully - it produces a confident, specific, wrong
    answer. Observed: a dense 25 GB gemma fully resident on the cards, measured at 48.7 GB of
    VRAM, costed against the 90.8 GB Flash-Next record that was 32 seconds older, and reported
    as "60% of experts in system RAM" on a model that has no experts and spilled nothing.
    """
    cfg = db.latest_server_config(name)
    argv = cfg.get("argv") or {}
    path_s = cfg.get("model_path") or argv.get("--model") or argv.get("-m") or ""
    alias = cfg.get("alias") or ""
    if not path_s and alias:
        path_s = (ini.get_section(alias) or {}).get("model", "").strip()
    if not path_s:
        return None
    # Identity gate. An empty set means nobody asked or the probe has not run, and the caller
    # gets the old behaviour; a non-empty set that does not contain this record's alias means
    # the record is for a model that is not the one on the cards.
    if loaded_ids and alias and alias not in loaded_ids:
        return None
    model = _host_path(path_s)
    try:
        file_size = model.stat().st_size
    except OSError:
        return None
    # Sum the whole shard set. The first shard of a split GGUF holds the metadata and almost no
    # tensors - 11 MB of Flash-Next's 83.8 GB - so statting only the named file valued the
    # weights at essentially nothing, and everything that was really weights fell into the
    # "other" residual: 24.46 GB of a 31.9 GB card, 77%, which is the opposite of a breakdown.
    _base, _idx, _parts = shard_key(model.name)
    if _parts:
        try:
            file_size = sum(q.stat().st_size for q in model.parent.iterdir()
                            if q.is_file() and q.suffix.lower() == ".gguf"
                            and shard_key(q.name)[0] == _base)
        except OSError:
            pass

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
    # Split mode off the RUNNING server's own argv, not the ini: tensor-parallel allocates a
    # measured 0.63x the per-card compute buffer that layer split does (see the table in
    # autoconfig). Charging the layer figure for a tensor load inflates this segment by ~37%
    # and hides the same amount inside "other", which is the one segment nobody can explain.
    # The meter's job is to show what is actually unaccounted for, so it gets the real number.
    split_mode = (argv.get("--split-mode") or argv.get("-sm") or "").strip()
    compute_gb = autoconfig.compute_buffer_gb(ctx_total, ubatch, vendor=vendor,
                                              split_mode=split_mode) * gpu_count

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
    # A speculative draft head costs FAR more than its file. Measured 2026-10-04 on
    # gemma-4-31B-it-Q6_K, tensor split, ctx 262144, ubatch 1024, same mmproj in both runs:
    #
    #     without draft   43.27 GiB      with draft   49.43 GiB      difference 6.16 GiB
    #
    # against the 0.56 GB this used to charge - eleven times under, and the single largest
    # error in the whole meter. The buffer inventory says why: the draft runs its OWN context,
    # so it allocates a SECOND compute buffer sized from the same ctx and ubatch as the target.
    # Measured per card, 2735.29 MiB for the draft against 2735.32 for the target - identical.
    # Its own weights (382.31 MiB/card) and KV (425.00) are the small part.
    #
    # The flag is `--model-draft`. This used to look for `--spec-draft-model`, which llama.cpp
    # does not emit, so a draft head was charged NOTHING AT ALL - the bug behind the ~3 GB/card
    # "other" residual that prompted the measurement.
    draft_s = (argv.get("--model-draft") or argv.get("-md")
               or argv.get("--spec-draft-model") or "").strip()
    # What the head costs IN TOTAL, reported but never apportioned. The two halves land in
    # different buckets below because they live in different places on the cards - the weights
    # pin to the main GPU, the compute buffer is per card - and a single number cannot be split
    # correctly across cards. So the per-card meters keep using aux/compute exactly as they did,
    # and this is carried alongside purely so the panel can answer "what is speculation costing
    # me", which no segment could answer once the cost had been divided between two of them.
    draft_gb = 0.0
    if draft_s:
        try:
            w = _host_path(draft_s).stat().st_size / (1024 ** 3) * 1.15
            aux_gb += w
            draft_gb += w
        except OSError:
            pass
        # The duplicate compute buffer is per card, like the target's, so it belongs with
        # compute rather than in aux - aux is apportioned to the main GPU, which would put
        # the whole of it on card 0 and skew every per-card figure beside it.
        draft_gb += compute_gb      # the doubling below IS the draft's own buffer
        compute_gb *= 2.0

    if model_gb <= 0:
        return None
    return {"model_gb": model_gb, "kv_gb": kv_gb, "compute_gb": compute_gb,
            "reserve_gb": reserve_gb, "aux_gb": aux_gb, "draft_gb": draft_gb,
            "gpu_pct": gpu_pct, "ctx_total": ctx_total, "model_label": model.name,
            # Raw weight bytes and the split multiplier, kept apart so the offloaded path can
            # work backwards from measured VRAM to "how much of the model is actually here".
            "model_total_gb": file_size / (1024 ** 3), "overhead_mul": overhead_mul,
            "is_moe": bool(isinstance(g.get("experts"), int) and (g.get("experts") or 0) > 1)}


def _cached_components(name: str, vendor: str, gpu_count: int,
                       loaded_ids: frozenset[str] | None = None) -> dict | None:
    # The loaded model is part of the key, not just an argument: caching on the backend name
    # alone would serve the previous model's components for the whole TTL after a swap, which
    # is the same mismatch the gate exists to prevent.
    key = (name, loaded_ids or frozenset())
    with _LOCK:
        entry = _CACHE.get(key)
        if entry and (time.time() - entry[0]) < (_TTL_S if entry[1] else _NEG_TTL_S):
            return entry[1]
    try:
        comps = _components(name, vendor, gpu_count, loaded_ids)
    except Exception:  # noqa: BLE001 - a meter that fails must degrade to the plain bar
        comps = None
    with _LOCK:
        _CACHE[key] = (time.time(), comps)
    return comps


def _segments(model_gb, kv_gb, overhead_gb, compute_gb, other_gb, free_gb, total_gb,
              gpu_pct=100, ctx_total=0, foreign=()):
    """The part-to-whole segment list, ordered weights -> context -> overhead -> compute ->
    [other tenants] -> other -> free: the same order the autoconfig panel validated (it keeps
    the two hues the palette validator flags as a weak protanopic pair apart), with the two
    residual buckets neutrals at the end.

    `foreign` is [(label, GB)] for processes on the card that are NOT this backend - see
    gpu_procs. They sit after everything this backend owns and before the residual, because
    that is the reading order of the claim being made: here is what the model costs, here is
    who else is on the card, here is what is left over. Their hue (fuchsia-600) was validated
    against the four it joins and the surfaces it is drawn on; the worst adjacent CVD pair is
    unchanged by its addition.
    """
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
    for label, gb in foreign:
        parts.append(("foreign", label, gb,
                      "another process on this card, not this backend: measured straight from "
                      "the kernel's per-process VRAM accounting, not estimated"))
    if other_gb > 0.005:
        parts.append(("other", "other", other_gb,
                      "in the measured total but not in the estimate: allocator fragmentation, "
                      "library workspaces, anything else sharing the device"))
    if free_gb > 0.005:
        parts.append(("free", "free", free_gb, "unallocated on the card"))
    return [{"key": k, "label": lbl, "gb": round(v, 2), "pct": 100.0 * v / total_gb, "note": note}
            for k, lbl, v, note in parts if v > 0.005]


def draft_cost(name: str, gpu, loaded_ids: frozenset[str] | None = None) -> dict | None:
    """What the speculative draft head costs, and what that VRAM would otherwise buy.

    None when the running instance is not speculating. The second figure is the point: on a
    box that is VRAM-bound, the head is not paid for in gigabytes, it is paid for in CONTEXT.
    Converting its cost at this model's own KV rate turns "6.2 GB" - a number nobody can judge
    - into "94k of context", which is the trade actually being made and can be weighed against
    the acceptance rate sitting next to it.

    The head is far more than its file: it runs its own context and so allocates a SECOND
    compute buffer sized from the same ctx and ubatch as the target. Measured on
    gemma-4-31B-it-Q6_K, 43.27 GiB without against 49.43 GiB with - 6.16 GiB, of which the
    weights were 382 MiB per card and the duplicate buffer 2735 MiB.
    """
    if gpu is None or getattr(gpu, "gpu_count", 0) <= 0:
        return None
    comps = _cached_components(name, gpu.vendor, gpu.gpu_count, loaded_ids)
    if not comps:
        return None
    gb = float(comps.get("draft_gb") or 0.0)
    if gb <= 0:
        return None
    kv_gb, ctx_total = float(comps.get("kv_gb") or 0.0), int(comps.get("ctx_total") or 0)
    # GB per token of context, from this model's own KV figure rather than any rule of thumb -
    # quantised KV, head count and layer count all move it, so a shared constant would be
    # wrong by a lot more than this number is worth.
    ctx_tokens = int(gb / (kv_gb / ctx_total)) if (kv_gb > 0 and ctx_total > 0) else 0
    return {"gb": round(gb, 2), "ctx_tokens": ctx_tokens,
            "ctx_pct": round(100.0 * gb / kv_gb, 1) if kv_gb > 0 else 0.0}


def breakdown(name: str, gpu, loaded_ids: frozenset[str] | None = None,
              foreign: list[tuple[str, float]] | None = None) -> dict | None:
    """Live part-to-whole VRAM breakdown for one backend, or None.

    None is the answer far more often than for the recommended-configuration meter, and every
    reason is a case where drawing an estimate would lie: no model loaded, no spawn telemetry
    (so the actual ctx and offload flags are unknown), the model file unreadable, an n-cpu-moe
    offload whose weights cost cannot be derived, a sleeping model whose measured usage has
    legitimately dropped below the allocation, or more than one backend on the cards (the
    estimate covers one of them and the residual would silently absorb the other).

    `foreign` is [(label, GB)] for OTHER processes on the cards, measured per process by
    gpu_procs. Passing it does two things, and the second matters more than the first: the
    tenant gets a named segment instead of swelling the residual, AND every comparison below
    is made against what is left for THIS backend. Without that, somebody else's 6 GB counts
    as this model's measured usage - which both hides a genuine shortfall (the `used_gb <
    est * 0.85` guard stops firing) and makes a spill look like a fit.
    """
    if gpu is None or not getattr(gpu, "cards", None) or gpu.gpu_count <= 0:
        return None
    total_gb = float(gpu.vram_total_gb)
    used_gb = float(gpu.vram_used_gb)
    if total_gb <= 0 or used_gb <= 0:
        return None

    comps = _cached_components(name, gpu.vendor, gpu.gpu_count, loaded_ids)
    if comps is None:
        return None

    # What is this backend's, and what is this backend's to spend. Everything from here on is
    # reasoned in those terms; only free_gb stays measured against the whole card, because
    # free is free whoever was not using it.
    foreign = [(l, gb) for l, gb in (foreign or []) if gb > 0.005]
    foreign_gb = min(sum(gb for _l, gb in foreign), used_gb)
    own_used_gb = max(0.0, used_gb - foreign_gb)
    own_total_gb = max(0.0, total_gb - foreign_gb)
    if own_used_gb <= 0 or own_total_gb <= 0:
        return None

    fixed_gb = comps["kv_gb"] + comps["reserve_gb"] + comps["aux_gb"] + comps["compute_gb"]
    est = comps["model_gb"] + fixed_gb

    # The whole model cannot fit, so llama.cpp has put some of it in host RAM - and the argv
    # does not say how much, because `--fit` decides that at load time and writes no ngl. This
    # is precisely the case the meter exists for, so solve for the weights instead of bailing.
    #
    # It is sound because the unknown is only ever the PLACEMENT. The fixed costs are allocated
    # at load from the context and the card count and do not move, so whatever else the device
    # is holding is weights. Measurement supplies the one thing the config cannot, which is the
    # opposite of guessing: nothing here is estimated that could have been measured.
    if est > own_total_gb * 1.02:
        resident_gb = kfd_process_vram_gb() or own_used_gb
        weights_gb = resident_gb - fixed_gb
        total_weights = comps["model_total_gb"]
        if weights_gb <= 0.05 or total_weights <= 0:
            return None
        on_gpu_raw = weights_gb / (comps["overhead_mul"] or 1.0)
        host_gb = total_weights - on_gpu_raw
        # Under a twentieth of a gigabyte apart is the estimate being pessimistic about a model
        # that did fit, not a spill worth drawing a second bar for.
        if host_gb <= 0.05:
            return None
        gpu_pct = max(0, min(100, int(round(100.0 * on_gpu_raw / total_weights))))
        free_gb = max(0.0, total_gb - used_gb)
        return {
            "total_gb": total_gb, "used_gb": used_gb, "free_gb": free_gb,
            "subtitle": "%s @ %s, as loaded" % (comps["model_label"],
                                                autoconfig.format_ctx(comps["ctx_total"])),
            "segments": _segments(weights_gb, comps["kv_gb"],
                                  comps["reserve_gb"] + comps["aux_gb"], comps["compute_gb"],
                                  max(0.0, own_used_gb - resident_gb), free_gb, total_gb,
                                  gpu_pct=gpu_pct, ctx_total=comps["ctx_total"],
                                  foreign=foreign),
            "weights_split": {
                "total_gb": total_weights, "gpu_gb": on_gpu_raw, "host_gb": host_gb,
                "gpu_pct": 100.0 * on_gpu_raw / total_weights,
                "host_pct": 100.0 * host_gb / total_weights,
                "is_moe": comps["is_moe"], "offload_kind": "", "n_cpu_moe": 0,
                "measured": True,
            },
            "raw": dict(comps, scale=1.0, measured_resident_gb=resident_gb,
                        drawn_model_gb=weights_gb, foreign_gb=foreign_gb),
        }
    scale = 1.0
    if est > own_used_gb:
        if own_used_gb < est * 0.85:
            return None
        scale = own_used_gb / est

    model_gb = comps["model_gb"] * scale
    kv_gb = comps["kv_gb"] * scale
    overhead_gb = (comps["reserve_gb"] + comps["aux_gb"]) * scale
    compute_gb = comps["compute_gb"] * scale
    other_gb = max(0.0, own_used_gb - est * scale)
    free_gb = max(0.0, total_gb - used_gb)

    return {
        "total_gb": total_gb, "used_gb": used_gb, "free_gb": free_gb,
        "subtitle": "%s @ %s, as loaded" % (comps["model_label"], autoconfig.format_ctx(comps["ctx_total"])),
        "segments": _segments(model_gb, kv_gb, overhead_gb, compute_gb, other_gb, free_gb,
                              total_gb, gpu_pct=comps["gpu_pct"], ctx_total=comps["ctx_total"],
                              foreign=foreign),
        "raw": dict(comps, scale=scale, drawn_model_gb=model_gb, foreign_gb=foreign_gb),
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


def tenants_only_per_card(cards: list,
                          foreign_by_card: dict[int, list[tuple[str, float]]]) -> dict[int, dict]:
    """Per-card meter built from MEASUREMENT ALONE: who else is on the card, and what is left.

    This exists because breakdown() is all-or-nothing on the BACKEND's model, and it says None
    for a long list of good reasons - no model loaded, one asleep, no spawn telemetry, an
    offload it cannot cost. In every one of those cases the card falls back to a plain fill,
    and a foreign tenant's VRAM goes unexplained even though it is the one quantity here that
    was never estimated in the first place.

    Observed 2026-10-04: router idle with nothing loaded, chatterbox-tts holding 5.88 GB of
    card 0. The tenant table said "chatterbox-tts 5.88 GB" and the bar directly above it showed
    an anonymous fill - two readouts on one screen disagreeing about the same card.

    So when the estimate cannot be drawn, draw what was measured. Every segment here is the
    kernel's own per-process figure; "other" is the remainder, which is honest because nothing
    in this meter claims to know what that remainder is.
    """
    out: dict[int, dict] = {}
    for i, c in enumerate(cards):
        idx = getattr(c, "index", i)
        fgn = [(l, gb) for l, gb in (foreign_by_card or {}).get(idx, []) if gb > 0.005]
        if not fgn:
            continue
        total = float(c.vram_total_gb or 0)
        used = float(c.vram_used_gb or 0)
        if total <= 0:
            continue
        spent = 0.0
        segs = []
        for label, gb in fgn:
            take = min(gb, max(0.0, used - spent))
            if take <= 0.005:
                continue
            spent += take
            segs.append({"key": "foreign", "label": label, "gb": round(take, 2),
                         "pct": 100.0 * take / total,
                         "note": "another process on this card, measured per process by the "
                                 "kernel. No backend model is sized here - this meter is "
                                 "measurement only"})
        other = max(0.0, used - spent)
        free = max(0.0, total - used)
        if other > 0.005:
            segs.append({"key": "other", "label": "other", "gb": round(other, 2),
                         "pct": 100.0 * other / total,
                         "note": "allocated on this card but not attributable to a process "
                                 "this app can see - driver and display context, mostly"})
        if free > 0.005:
            segs.append({"key": "free", "label": "free", "gb": round(free, 2),
                         "pct": 100.0 * free / total, "note": "unallocated on the card"})
        out[idx] = {"total_gb": total, "used_gb": used, "free_gb": free,
                    "subtitle": "measured per process; no backend model sized",
                    "segments": segs}
    return out


def per_card(vb: dict, cards: list,
             foreign_by_card: dict[int, list[tuple[str, float]]] | None = None) -> dict[int, dict]:
    """Split a pooled breakdown into per-card meters, keyed by card index.

    Three of the four costs are known per card and only one is not, so they are itemised
    rather than lumped:

      reserve  - one driver/runtime context per card, exact
      aux      - projector and draft head, which pin to the main GPU, exact
      compute  - graph scratch, allocated IDENTICALLY on every card by construction
                 (compute_gb is a per-card figure multiplied by the card count), so dividing
                 it back out is exact, not apportioned
      weights  - the only unknown: llama reports usage, not which card holds which layer
      KV       -

    Weights and KV therefore share what is left of each card's measured usage, split by the
    POOLED weights:KV ratio. That is an apportionment and is labelled as one, but it is a
    defensible one: both scale with the number of layers a card holds, so a card with more
    layers holds proportionally more of each. It beats the previous single
    "weights + KV + compute" lump, which was honest but answered "is this card full" when the
    question is "full of what".

    Each card's segments still sum to that card, and the free share stays measured.

    `foreign_by_card` is the fifth itemised cost and it is EXACT, measured per process and per
    card by gpu_procs. It has to be subtracted before the apportionment, not merely drawn: the
    apportionment splits each card's unexplained usage by how much of it there is, so a tenant
    sitting entirely on one card used to be spread across both. Measured on this box with
    chatterbox-tts holding 6.28 GB on card 0 and nothing on card 1, the strip reported "other
    3.42 GB" and "other 2.58 GB" - the second of those describing a card the container had not
    allocated a byte on, and both of them stealing from the weights figure beside them.
    """
    out: dict[int, dict] = {}
    if not vb or not cards:
        return out
    if len(cards) == 1:
        return {0: vb}

    raw = vb["raw"]
    n = len(cards)
    scale = raw["scale"]
    weights_pool = raw.get("drawn_model_gb", raw["model_gb"] * scale)
    kv_pool = raw["kv_gb"] * scale
    wk_pool = weights_pool + kv_pool
    w_frac = (weights_pool / wk_pool) if wk_pool > 0 else 1.0
    compute_each = raw["compute_gb"] * scale / n
    fixed, overhead, foreign = [], [], []
    for i, c in enumerate(cards):
        # Card 0 also carries the projector/draft-head allocation; a card's own overhead can
        # never exceed what is measured on it.
        want = raw["reserve_gb"] / n + (raw["aux_gb"] if i == 0 else 0.0)
        ov = min(want * scale, c.vram_used_gb)
        overhead.append(ov)
        # Another tenant's share of THIS card, clamped to what is left after the overhead so a
        # sampling skew between the two sources can never push the segments past the card.
        room = max(0.0, c.vram_used_gb - ov)
        fgn, spent = [], 0.0
        for label, gb in (foreign_by_card or {}).get(i, []):
            take = min(gb, room - spent)
            if take > 0.005:
                fgn.append((label, take))
                spent += take
        foreign.append(fgn)
        fixed.append(min(ov + spent + compute_each, c.vram_used_gb))
    slack = [max(0.0, c.vram_used_gb - fixed[i]) for i, c in enumerate(cards)]
    slack_sum = sum(slack)
    for i, c in enumerate(cards):
        share = slack[i] / slack_sum if slack_sum > 0 else 1.0 / n
        alloc = min(wk_pool * share, max(0.0, c.vram_used_gb - fixed[i]))
        model_gb = alloc * w_frac
        kv_gb = alloc - model_gb
        foreign_gb = sum(gb for _l, gb in foreign[i])
        compute_gb = min(compute_each, max(0.0, c.vram_used_gb - overhead[i] - foreign_gb))
        other_gb = max(0.0, c.vram_used_gb - overhead[i] - foreign_gb - compute_gb - alloc)
        free_gb = max(0.0, c.vram_total_gb - c.vram_used_gb)
        card_total = float(c.vram_total_gb or c.vram_used_gb)
        if card_total <= 0:
            continue
        out[i] = {
            "total_gb": card_total, "used_gb": c.vram_used_gb, "free_gb": free_gb,
            "subtitle": vb["subtitle"],
            "segments": [s for s in [
                {"key": "model", "label": "model weights", "gb": round(model_gb, 2),
                 "pct": 100.0 * model_gb / card_total,
                 "note": "this card's share of the layer-split weights, apportioned by usage: "
                         "llama reports what a card holds, not which layers it holds"},
                {"key": "ctx", "label": "context (KV)", "gb": round(kv_gb, 2),
                 "pct": 100.0 * kv_gb / card_total,
                 "note": "this card's share of the KV cache, apportioned the same way"},
                {"key": "overhead", "label": "overhead", "gb": round(overhead[i], 2),
                 "pct": 100.0 * overhead[i] / card_total,
                 "note": "driver and runtime context"
                         + (", plus the projector and draft head, which pin to this card" if i == 0 else "")},
                {"key": "compute", "label": "compute buffers", "gb": round(compute_gb, 2),
                 "pct": 100.0 * compute_gb / card_total,
                 "note": "graph scratch, allocated identically on every card"},
            ] + [
                {"key": "foreign", "label": label, "gb": round(gb, 2),
                 "pct": 100.0 * gb / card_total,
                 "note": "another process on this card, not this backend - measured per "
                         "process by the kernel, so this one is not apportioned"}
                for label, gb in foreign[i]
            ] + [
                {"key": "other", "label": "other", "gb": round(other_gb, 2),
                 "pct": 100.0 * other_gb / card_total,
                 "note": "in the measured total but not in the estimate"},
                {"key": "free", "label": "free", "gb": round(free_gb, 2),
                 "pct": 100.0 * free_gb / card_total,
                 "note": "unallocated on the card"},
            ] if s["gb"] > 0.005],
        }
    return out
