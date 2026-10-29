from __future__ import annotations

import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import db, hw, power_cap
from ..config import settings
from ..gpu_activity import curve
from ..web import templates

router = APIRouter()


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
        "gpu_cap": power_cap.status(),
        # The same series as curves for the overview's smooth sparklines, on the same ceilings.
        "power_curve": curve(watts, cfg["power_crit_w"]),
        "temp_curve": curve(temps, cfg["temp_crit_c"]),
    }


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
        "load_curve": curve(lh, 100.0),
        # Drawn as USED, so the line rises as the box fills, the same direction as the ring.
        "ram_curve": curve([ram_total - r for r in rh], ram_total or None) if ram_total else None,
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


@router.get("/boxbar", response_class=HTMLResponse)
def box_bar_partial(request: Request) -> HTMLResponse:
    """The strip across the top of the overview: power, heat, load, RAM, swap, disk.

    One partial for what used to be two (/power and /hoststats), because the strip is one
    line and two polls racing to fill halves of it would redraw it twice a second."""
    ctx = {"request": request}
    ctx.update(_power_context())
    ctx.update(_host_stats_context())
    return templates.TemplateResponse("_box_bar.html", ctx)


def _power_settings_context(saved: bool = False, cap_note: str = "", cap_ok: bool = False) -> dict:
    cfg = _power_settings()
    return {"power_warn_w": cfg["power_warn_w"], "power_crit_w": cfg["power_crit_w"],
            "baseline_w": cfg["power_baseline_w"], "temp_warn_c": cfg["temp_warn_c"],
            "temp_crit_c": cfg["temp_crit_c"], "saved": saved, "reopen": saved or bool(cap_note),
            "gpu_cap": power_cap.status(), "cap_note": cap_note, "cap_ok": cap_ok}


@router.post("/power/cap", response_class=HTMLResponse)
async def power_cap_save(request: Request) -> HTMLResponse:
    """Ask the host for a new GPU board-power cap (see power_cap.py for how it gets applied).

    Only a whole number inside the range every card accepts is written: the host script would
    clamp anything else, and a cap that silently became a different number is the confusion the
    Overview warning exists to end.
    """
    form = await request.form()
    lo, hi = power_cap.allowed(power_cap.cards())
    raw = str(form.get("cap_w") or "").strip()
    ok = False
    if not lo or not hi:
        note = "No card reports a power cap range, so nothing was changed."
    elif not raw.isdigit() or not lo <= int(raw) <= hi:
        note = f"The cards take {lo}-{hi} W. Nothing was changed."
    else:
        power_cap.set_desired(int(raw))
        note, ok = f"Asked for {int(raw)} W. The host applies it within a few seconds.", True
    ctx = {"request": request}
    ctx.update(_power_settings_context(cap_note=note, cap_ok=ok))
    return templates.TemplateResponse("_power_settings.html", ctx)


@router.post("/power/settings", response_class=HTMLResponse)
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
