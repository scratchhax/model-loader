"""Fit-math inputs shared by the download estimates and the autoconfig panel.

Moved out of main.py so the downloads router and the config routes can both reach it
without either importing the other.
"""

from __future__ import annotations

from . import autoconfig, hw, services


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
