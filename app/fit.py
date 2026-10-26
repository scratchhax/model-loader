"""Fit-math inputs shared by the download estimates and the autoconfig panel.

Moved out of main.py so the downloads router and the config routes can both reach it
without either importing the other.
"""

from __future__ import annotations

from . import autoconfig, gguf_meta, gpu_procs, hw, ini, services
from pathlib import Path

from .config import settings
from .utils import shard_key


def _fit_host_ram() -> float:
    """RAM the fit math may count on: the host total minus a running Strata's pinned arena.

    The static HOST_RAM_RESERVE_GB covers the OS and its habits; the arena is a measured
    tenant on top of that, and 35-55 GB of pinned experts is well past what a 32 GB reserve
    was ever meant to absorb. Unchanged when no Strata runs, so a llama-only box sees
    exactly what it always saw.
    """
    return round(max(0.0, hw.host_ram_gb() - hw.strata_arena_gb()), 1)


def _fit_card_vram(bn: str) -> list[float]:
    """Per-card VRAM for the fit table, with the cards a running Strata holds zeroed out.

    The table is a promise about placement, and a card whose VRAM is spoken for by another
    engine's expert cache is not available for llama layers no matter what the free figure
    says - "free" on that card is Strata's next expert, not llama's next layer. Zeroing the
    cap rather than shrinking it keeps the arithmetic honest downstream: _split_feasible
    fills cards in order and a zero-cap card simply takes no layers, which is exactly the
    placement llama's own fitter will also arrive at once it sees the card is full.
    """
    caps = hw.card_vram_gb_for(bn)
    held = services.strata_held_cards()
    if not held or not caps:
        return caps
    return [0.0 if i in held else c for i, c in enumerate(caps)]


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
            "gpu_count": hw.gpu_count_for(name), "card_vram_gb": _fit_card_vram(name),
                    "host_ram_gb": _fit_host_ram(),
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




def _container_status(name: str) -> str:
    """Docker status string ('running', 'exited', ...) or '' when unknowable."""
    try:
        client = services._docker_client()
        if client is None:
            return ""
        return str(client.containers.get(name).status or "")
    except Exception:  # noqa: BLE001
        return ""


def _gpu_image(name: str) -> bool:
    """True when the container runs a GPU build of llama.cpp (rocm / cuda / vulkan image tag)."""
    try:
        client = services._docker_client()
        if client is None:
            return False
        tags = client.containers.get(name).image.tags or []
    except Exception:  # noqa: BLE001
        return False
    return any(v in t.lower() for t in tags for v in ("rocm", "cuda", "vulkan"))


def _backend_list() -> list[dict]:
    """Backend descriptors for autoconfig, from the env whitelist or auto-discovery.

    Shared so the benchmark results page asks for the same fit estimate the autoconfig panel
    shows rather than growing a second copy.

    A discovered GPU backend whose probe reads 0 is offered with the sysfs card facts and an
    `offline` flag the panel discloses: the card totals are a property of the machine. That
    covers a STOPPED backend (the plan describes the state after the engine switch) and a
    RUNNING GPU-image backend the sampler has not probed yet. The second used to be dropped,
    and then the panel had no multi-GPU backend to size against: a section saved from it kept
    the `split-mode = layer` every new section starts with, and on 2026-10-26 Qwen3.8-27B-Q8 ran
    on both R9700s in sequence at 15 tok/s instead of in parallel. A running CPU build is still
    dropped, since the machine's cards do not describe its devices.
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
        offline = False
        cards: list[float] = []
        if vram <= 0:
            cards = hw.sysfs_card_vram_gb()
            if not cards:
                continue
            if _container_status(bn) == "running" and not _gpu_image(bn):
                continue  # a running CPU build has no VRAM budget; autoconfig can't do KV math on it
            offline = True
            vram = sum(cards)
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
        if vendor == "unknown":
            vendor = hw.sysfs_gpu_vendor() or "unknown"
        cmd = _container_baseline(bn)
        base = autoconfig.parse_baseline(cmd) if cmd else {}
        entry = {"name": bn, "vendor": vendor, "vram_gb": float(vram),
                 "gpu_count": len(cards) if offline else hw.gpu_count_for(bn),
                 "card_vram_gb": list(cards) if offline else _fit_card_vram(bn),
                 "host_ram_gb": _fit_host_ram(),
                 "baseline": base}
        if offline:
            # The strata-held zeroing is deliberately NOT applied to an offline entry:
            # strata holds the cards only until the engine switch, and the plan is for
            # after it.
            entry["offline"] = True
        out.append(entry)
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
            # Every companion, not just mmproj (ini._is_companion knows draft heads too).
            # Otherwise the alphabetically first survivor wins, which picks a
            # ...-MAX-MTP-IQ2_M.gguf over the ...-MAX-Q8_0.gguf the section is named for.
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
