"""Read-only JSON for everything the web UI shows, for dashboards and scripts outside this app.

Every endpoint is built from the SAME context functions its page renders from, so the API can never
disagree with the UI and adds no probe of its own: calling /api/v1/status costs what one open
Overview tab costs on its own polls, and nothing at all when nobody calls it.

Two kinds of thing are left out on purpose:

  * Presentation. The pages ship pre-drawn SVG paths (sparklines, smooth curves, the ring's dash
    runs) and layout hints. Those are dropped; the numbers they were drawn from are in
    /api/v1/series as raw (timestamp, value) pairs, which is what another app can actually use.
  * Anything that is configuration rather than state, or could carry a secret: Open WebUI's
    connection list, the prompt library's bodies, a download's URL and paths.

GET only, so neither the read-only guard nor the same-site guard is in the way.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import math
import os
import sqlite3
import time
from pathlib import Path

from fastapi import APIRouter

from .. import bench, db, energy, gpu_activity, hw, ini, router_api, services, updates
from ..config import settings
from ..downloader import manager
from . import containers as containers_page
from . import maya as maya_page
from . import models as models_page
from . import overview, power
from . import router_tab

router = APIRouter(prefix="/api/v1")

API_VERSION = 1

# Keys whose value is drawing, not data. Matched exactly or by suffix.
_DRAWN_SUFFIXES = ("_spark", "_curve")
_DRAWN_KEYS = frozenset({"spark", "ring", "ring_c", "request", "slot_density", "slot_columns",
                         "search", "sort_bytes", "count_bytes"})
# The Containers page's per-backend sparklines, whose names are too generic to drop everywhere
# (HistoryPoint has a gpu_util FIELD); removed from that one dict instead.
_PERF_DRAWN = ("gpu_util", "vram", "cpu", "mem")

# What a download job may say about itself. The URL and the two paths are left out: a URL
# import can be any address the owner typed, and the paths say nothing a dashboard needs.
_DOWNLOAD_FIELDS = ("id", "repo_id", "filename", "total_bytes", "downloaded_bytes", "status",
                    "error", "started_at", "completed_at", "parallel", "expected_sha256",
                    "verified", "pct", "speed_bps")


def _drawn(key: str) -> bool:
    return key in _DRAWN_KEYS or key.endswith(_DRAWN_SUFFIXES)


def _properties(obj) -> dict:
    """A dataclass's computed facts (vram_free_gb, working, progress_pct ...), evaluated now.

    They are half of what the templates print, so leaving them out would make a client
    re-derive them. One that raises is skipped rather than failing the whole response."""
    out = {}
    for name, member in inspect.getmembers(type(obj), lambda m: isinstance(m, property)):
        if name.startswith("_"):
            continue
        try:
            out[name] = getattr(obj, name)
        except Exception:  # noqa: BLE001 - a property is a convenience, never a reason to 500
            continue
    return out


def jsonable(obj, _depth: int = 0):
    """Anything a page context holds, as plain JSON: dataclasses with their properties, sqlite
    rows, paths, sets. Drawn values are dropped, private fields skipped, NaN and infinity become
    null (JSON has neither)."""
    if _depth > 24:
        return None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, sqlite3.Row):
        obj = dict(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)
             if not f.name.startswith("_")}
        d.update(_properties(obj))
        obj = d
    if isinstance(obj, dict):
        return {str(k): jsonable(v, _depth + 1) for k, v in obj.items()
                if not (isinstance(k, str) and (k.startswith("_") or _drawn(k)))}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj, key=str) if isinstance(obj, (set, frozenset)) else obj
        return [jsonable(v, _depth + 1) for v in items]
    # Anything else as text, never by walking its attributes: a context that ever carried a
    # docker object would otherwise serialise the client behind it.
    return str(obj)


def _pairs(series) -> list[list[float]]:
    """(ts, value) tuples as [ts, value] lists, rounded to what a chart can show."""
    return [[round(t, 3), round(v, 3)] for t, v in series]


# ------------------------------------------------------------------------------ the overview

async def _hero() -> dict:
    return jsonable(await overview._hero_context())


async def _cards_and_strip() -> tuple[dict, dict]:
    """The card panels and the GPU strip, computing the strip once for both."""
    llm = await overview._llm_activity()

    def build():
        strip = overview._gpu_strip_context()
        return overview._cards_context(llm, strip=strip), strip

    cards, strip = await asyncio.to_thread(build)
    backends = await overview._hero_backends_cached()
    parked = {services.engine_for(b.name) for b in backends if getattr(b, "unloaded_model", None)}
    cards = {**cards, "parked_engines": parked, "llm": llm}
    # gpu_history is pre-drawn per card; its numbers are in /series under "containers".
    strip = {k: v for k, v in strip.items() if k != "gpu_history"}
    return jsonable(cards), jsonable(strip)


def _box() -> dict:
    out = power._power_context()
    out.update(power._host_stats_context())
    out["settings"] = power._power_settings_context()
    return jsonable(out)


def _host() -> dict:
    snap = overview._models_snapshot_cached()
    return jsonable({
        "line": overview._host_line(),
        "models_dir": {"exists": snap.exists, "error": snap.error, "disk": snap.disk,
                       "gguf_count": sum(1 for g in snap.ggufs if not g.is_companion),
                       "ini_sections": len(snap.ini_aliases)},
        "active_downloads": sum(1 for j in manager.snapshot()
                                if j.status in ("queued", "downloading")),
    })


@router.get("")
def index() -> dict:
    """What is here, so a client can discover it without reading the source."""
    return {"version": API_VERSION, "endpoints": [
        "/api/v1/status", "/api/v1/hero", "/api/v1/cards", "/api/v1/strip", "/api/v1/box",
        "/api/v1/series", "/api/v1/router?range=24h|7d|30d|90d|1y|all", "/api/v1/energy",
        "/api/v1/containers",
        "/api/v1/downloads", "/api/v1/models", "/api/v1/benchmarks",
        "/api/v1/benchmarks/{run_id}", "/api/v1/updates", "/api/v1/maya",
        "/api/v1/engines/switch"]}


@router.get("/status")
async def status() -> dict:
    """Everything the Overview shows, now, in one call. Series are in /series."""
    hero = await _hero()
    cards, strip = await _cards_and_strip()
    host = await asyncio.to_thread(_host)
    box = await asyncio.to_thread(_box)
    return {"version": API_VERSION, "ts": time.time(), "host": host, "hero": hero,
            "cards": cards, "strip": strip, "box": box,
            "engine_switch": jsonable(services.engine_switch_state())}


@router.get("/hero")
async def hero() -> dict:
    return await _hero()


@router.get("/cards")
async def cards() -> dict:
    return (await _cards_and_strip())[0]


@router.get("/strip")
async def strip() -> dict:
    return (await _cards_and_strip())[1]


@router.get("/box")
async def box() -> dict:
    return await asyncio.to_thread(_box)


@router.get("/series")
async def series() -> dict:
    """Every live history the pages draw, as raw [timestamp, value] pairs, oldest first.

    backends    per llama backend: tok/s and context in use (tokens), sampled on the Overview's
                poll; prefill is one tok/s reading per completed prompt, with no timestamps;
                slots is the same tok/s and context per slot, keyed by slot index ("solo" for
                Strata's request outside the batch slots)
    cards       per card (host numbering): compute %, VRAM GB and each tenant's compute share
    containers  per backend, the hw sampler's own history: GPU util and VRAM per card, CPU, RAM
    power       wall watts and the hottest temperature
    host        load % and available RAM GB
    """
    def build() -> dict:
        names = services._effective_container_names()
        backends = {n: {"tps": _pairs(services.tps_series(n)),
                        "ctx": _pairs(services.ctx_series(n)),
                        "prefill": [round(v, 2) for v in services.prefill_history(n)],
                        "slots": {("solo" if i < 0 else str(i)): {k: _pairs(v) for k, v in sl.items()}
                                  for i, sl in services.slot_series(n).items()}}
                    for n in names}
        cards: dict[str, list] = {}
        for c in gpu_activity._cards():
            cards[str(c.index)] = [{"ts": round(s.ts, 3), "util": s.util, "vram_gb": s.vram_gb,
                                    "share": s.share} for s in gpu_activity.timeline(c.index)]
        conts = {n: jsonable(hw.history_for(n)) for n in names}
        watts, temps = hw.power_series()
        load, ram = hw.host_series()
        return {"backends": backends, "cards": cards, "containers": conts,
                "power": {"watts": _pairs(watts), "temp_c": _pairs(temps)},
                "host": {"load_pct": _pairs(load), "ram_avail_gb": _pairs(ram)}}
    return {"version": API_VERSION, "ts": time.time(), **await asyncio.to_thread(build)}


@router.get("/energy")
async def energy_stats() -> dict:
    """Electricity: this box's wall energy and the routed requests' share of it.

    windows         last 24 h / 7 d / 30 d: box_wh (whole box at the wall, from the meter's
                    hourly record), box_metered_s (how much of the window the meter covered:
                    restarts are gaps, not zeros), routed_wh / routed_n (energy of the requests
                    the router served, each its share of the box while it ran), requests
    days            box Wh per 24 h ending now, the last 30, oldest first
    by_model_24h    routed Wh per engine and model, with Wh per 1k output tokens
    by_model_7d     the same over 7 days
    conversations   the 15 most recent conversations (agent runs) with their Wh
    now_w           the meter's newest wall reading
    """
    def build() -> dict:
        now = time.time()
        wins = {}
        for key, secs in (("24h", 86400.0), ("7d", 7 * 86400.0), ("30d", 30 * 86400.0)):
            s = energy.summary(now, window_s=secs)
            wins[key] = {k: s[k] for k in ("box_wh", "box_metered_s", "box_coverage",
                                           "routed_wh", "routed_n", "requests")}
        convs = router_tab._router_conversations(now - 86400.0, limit=15)
        return {"now_w": energy.summary(now)["now_w"], "windows": wins,
                "days": energy.box_days(30, now),
                "by_model_24h": energy.by_model(now - 86400.0),
                "by_model_7d": energy.by_model(now - 7 * 86400.0),
                "conversations": [{"conv": c["conv"], "engine": c["engine"], "model": c["model"],
                                   "turns": c["turns"], "last_ts": c["last_ts"],
                                   "output_tokens": c["completion_tokens"] or 0,
                                   "wh": round(c["energy_wh"] or 0.0, 4) if c.get("energy_n") else None,
                                   "measuring": c.get("energy_pending") or 0} for c in convs]}
    return {"version": API_VERSION, "ts": time.time(), **jsonable(await asyncio.to_thread(build))}


# ------------------------------------------------------------------------------ other pages

@router.get("/router")
async def router_stats(range: str = "24h") -> dict:
    """The Router tab: totals, bucketed activity, conversations and recent requests."""
    since = router_tab._router_since(range)

    def build() -> dict:
        return {"range": range, "since": since,
                "recording": db.get_setting("router.record", "1") != "0",
                "totals": router_api.totals(since),
                "series": router_api.series(since),
                "conversations": router_tab._router_conversations(since),
                "requests": router_tab._router_rows(since),
                "retention": router_api.retention_state()}
    return jsonable(await asyncio.to_thread(build))


@router.get("/containers")
async def containers() -> dict:
    """The Containers page: every backend, its live stats and the current readings behind its
    sparklines. Open WebUI's connection state is not included (it holds API keys)."""
    backends = await services.snapshot_llama_backends()
    try:
        ini_mtime = os.path.getmtime(settings.models_ini_path)
    except OSError:
        ini_mtime = 0.0

    def build() -> dict:
        perf = {n: {k: v for k, v in p.items() if k not in _PERF_DRAWN}
                for n, p in containers_page._perf_by_name().items()}
        return {"backends": backends, "stats": containers_page._stats_by_name(),
                "perf": perf,
                "ini_stale": {b.name for b in backends
                              if b.status == "running" and getattr(b, "engine", "llama") == "llama"
                              and containers_page.ini_stale_since(b.started_at, ini_mtime)},
                "strata_pin": containers_page._strata_pin_state(backends)}
    out = await asyncio.to_thread(build)
    speeds = {}
    for b in backends:
        if b.loaded_model:
            speeds[b.name] = await services.inference_speed(b.name, b.internal_port, b.loaded_model)
    out["speed"] = speeds
    return jsonable(out)


@router.get("/downloads")
def downloads() -> dict:
    jobs = []
    for j in manager.snapshot():
        jobs.append({k: jsonable(getattr(j, k, None)) for k in _DOWNLOAD_FIELDS})
    return {"jobs": jobs}


@router.get("/models")
async def models() -> dict:
    """The Models page's rows: one per served model, with its file, shape, state and ratings.

    Built on demand only - it reads every GGUF header - so poll /status, not this."""
    snap = await asyncio.to_thread(overview._models_snapshot_cached)
    loaded = await models_page._loaded_map()

    def build() -> dict:
        try:
            caps = models_page._section_caps(ini.list_sections())
        except Exception:  # noqa: BLE001 - an unreadable ini must not take the list down
            caps = {}
        owui = models_page._owui_visibility()
        rows = models_page._model_rows(snap, models_page._shapes_for_files(snap), loaded,
                                       models_page._update_status_map(snap), owui,
                                       db.badges_by_alias(), caps)
        return {"disk": snap.disk, "error": snap.error, "rows": rows}
    return jsonable(await asyncio.to_thread(build))


@router.get("/benchmarks")
def benchmarks(limit: int = 15) -> dict:
    return jsonable({"job": bench.state(), "runs": db.bench_runs(limit=max(1, min(limit, 200)))})


@router.get("/benchmarks/{run_id}")
def benchmark_run(run_id: int) -> dict:
    run = db.bench_run(run_id)
    if run is None:
        return {"run": None}
    return jsonable({"run": run, "variants": db.bench_variants(run_id),
                     "results": db.bench_results(run_id)})


@router.get("/updates")
def engine_updates() -> dict:
    """The engine-updates card: installed against latest, from the last check (no new check)."""
    state = updates.cached()
    return jsonable({"engines": state.get("engines") or [], "checked_at": state.get("checked_at"),
                     "checking": updates.in_progress()})


@router.get("/maya")
async def maya() -> dict:
    state = await asyncio.to_thread(maya_page._maya_state)
    return jsonable({k: v for k, v in state.items() if k not in ("tunables", "writable")})


@router.get("/engines/switch")
def engine_switch() -> dict:
    return jsonable(services.engine_switch_state())
