"""What each routed request cost in electricity: the box's wall energy while it ran.

A hosted API bills a run in money. Here the bill is the electricity, so the router's rows get
an energy figure instead: the wall power this box drew (measured GPUs + CPU package, plus the
board allowance, through the PSU efficiency, the same figure as the power panel) integrated
over the seconds each request was being served. When requests overlap they split it evenly,
so the requests of a stretch add up to what the box drew for them, not a multiple of it.

It is the WHOLE BOX's draw, idle included, because that is what the meter on the wall sees
while a request runs. The power panel's own history only grew while somebody had a page open,
so this keeps a meter of its own: a sample every couple of seconds in memory, and every few
seconds the requests that have finished get their share written onto their row
(router_request.energy_wh). Samples live only in this process; a request that overlaps a
restart is measured from what was seen, and one with nothing seen at all is marked unknown
(-1) rather than given a guess.
"""
from __future__ import annotations

import bisect
import threading
import time
from collections import deque

from . import db

SAMPLE_S = 2.0
ATTRIBUTE_EVERY_S = 10.0
# A request is settled this long after it ends, so a sample taken just after its last second is
# in hand and the requests that overlapped it have been recorded too (the router writes a row
# when a request ENDS).
SETTLE_S = 6.0
# Long enough for any request this box will serve; older samples are no use to anyone.
KEEP_S = 6 * 3600.0
# A request is unknown (-1) when samples cover less than this share of it.
MIN_COVERAGE = 0.5

_samples: deque[tuple[float, float]] = deque()     # (ts, wall watts), oldest first
_lock = threading.Lock()
_started = False


def _settings() -> tuple[float, float]:
    from .routers.power import _power_settings
    cfg = _power_settings()
    return cfg["power_baseline_w"], cfg["psu_efficiency"]


def sample(now: float | None = None) -> float | None:
    """Take one wall-power reading into the meter. None when nothing could be read."""
    from . import hw
    try:
        baseline, eff = _settings()
        w = float(hw.power_rollup(baseline, eff)["wall_w"])
    except Exception:  # noqa: BLE001 - a missed reading is a gap, never a crash
        return None
    add_sample(now or time.time(), w)
    return w


def add_sample(ts: float, watts: float) -> None:
    with _lock:
        if _samples and ts <= _samples[-1][0]:
            return
        if _samples:
            # The box's own meter: every stretch between two readings goes into its hour, unless
            # the gap is so long (a stall, a restart) that a straight line across it is a guess.
            t0, w0 = _samples[-1]
            dt = ts - t0
            if dt <= MAX_GAP_S:
                b = int((t0 + dt / 2) // 3600) * 3600.0
                acc = _unflushed.setdefault(b, [0.0, 0.0])
                acc[0] += (w0 + watts) / 2 * dt / 3600.0
                acc[1] += dt
        _samples.append((ts, watts))
        while _samples and _samples[0][0] < ts - KEEP_S:
            _samples.popleft()


# ---------------------------------------------------------------- the box's own running total
#
# Whole-box energy per clock hour, kept in the database so a rolling day (and a history of days)
# survives restarts. Hours rather than calendar days because the boxes run on UTC: a "today" that
# resets mid-afternoon reads as a bug. The seconds the meter actually covered are kept beside it,
# so a restart's missing minutes are a stated gap rather than a silent undercount.

MAX_GAP_S = 30.0
KEEP_HOURS_DAYS = 400
_unflushed: dict[float, list[float]] = {}    # {hour start: [Wh, metered seconds]}


def _init_table(c) -> None:
    c.execute("CREATE TABLE IF NOT EXISTS energy_hour (bucket REAL PRIMARY KEY, "
              "wh REAL NOT NULL DEFAULT 0, metered_s REAL NOT NULL DEFAULT 0)")


def flush(now: float | None = None) -> int:
    """Add the meter's unwritten hours to energy_hour. Returns how many hours were touched."""
    with _lock:
        pending = dict(_unflushed)
        _unflushed.clear()
    if not pending:
        return 0
    now = now or time.time()
    try:
        with db._LOCK, db._conn() as c:
            _init_table(c)
            c.executemany(
                "INSERT INTO energy_hour(bucket, wh, metered_s) VALUES(?,?,?) "
                "ON CONFLICT(bucket) DO UPDATE SET wh = wh + excluded.wh, "
                "metered_s = metered_s + excluded.metered_s",
                [(b, v[0], v[1]) for b, v in pending.items()])
            c.execute("DELETE FROM energy_hour WHERE bucket < ?", (now - KEEP_HOURS_DAYS * 86400.0,))
    except Exception:  # noqa: BLE001 - put it back for the next try rather than lose it
        with _lock:
            for b, v in pending.items():
                acc = _unflushed.setdefault(b, [0.0, 0.0])
                acc[0] += v[0]
                acc[1] += v[1]
        return 0
    return len(pending)


def box_energy(since: float) -> tuple[float, float]:
    """(Wh, metered seconds) the whole box drew from the hour holding `since` to now."""
    start = int(since // 3600) * 3600.0
    try:
        with db._conn() as c:
            _init_table(c)
            r = c.execute("SELECT SUM(wh), SUM(metered_s) FROM energy_hour WHERE bucket >= ?",
                          (start,)).fetchone()
    except Exception:  # noqa: BLE001
        r = None
    with _lock:
        extra = [v for b, v in _unflushed.items() if b >= start]
    wh = (r[0] or 0.0) if r else 0.0
    sec = (r[1] or 0.0) if r else 0.0
    return wh + sum(v[0] for v in extra), sec + sum(v[1] for v in extra)


def box_days(days: int = 30, now: float | None = None) -> list[dict]:
    """Whole-box energy per 24 h ending now, oldest first: [{"start", "wh", "metered_s"}]."""
    now = now or time.time()
    out = []
    try:
        with db._conn() as c:
            _init_table(c)
            rows = c.execute("SELECT bucket, wh, metered_s FROM energy_hour WHERE bucket >= ?",
                             (now - days * 86400.0 - 3600,)).fetchall()
    except Exception:  # noqa: BLE001
        rows = []
    for i in range(days, 0, -1):
        lo, hi = now - i * 86400.0, now - (i - 1) * 86400.0
        sel = [r for r in rows if lo <= r[0] + 1800 < hi]
        out.append({"start": round(lo), "wh": round(sum(r[1] for r in sel), 2),
                    "metered_s": round(sum(r[2] for r in sel))})
    return out


_summary_cache: tuple[float, dict] | None = None


def summary(now: float | None = None, window_s: float = 86400.0) -> dict:
    """The running totals the Overview and the API show: the last `window_s` of whole-box energy
    and of routed requests' share of it, and the meter's newest reading. Cached 5 s: the
    Overview's power cell polls every second."""
    global _summary_cache
    now = now or time.time()
    if _summary_cache and window_s == 86400.0 and now - _summary_cache[0] < 5.0:
        return _summary_cache[1]
    from . import router_api
    box_wh, metered = box_energy(now - window_s)
    t = router_api.totals(now - window_s)
    pts = samples()
    out = {"window_s": window_s, "box_wh": round(box_wh, 3), "box_metered_s": round(metered),
           "box_coverage": round(min(1.0, metered / window_s), 3) if window_s else 0.0,
           "routed_wh": round(t.get("energy_wh") or 0.0, 3), "routed_n": t.get("energy_n") or 0,
           "requests": t.get("requests") or 0,
           "now_w": round(pts[-1][1], 1) if pts and now - pts[-1][0] < 30 else None}
    if window_s == 86400.0:
        _summary_cache = (now, out)
    return out


def samples() -> list[tuple[float, float]]:
    with _lock:
        return list(_samples)


def _energy_shares(reqs: list[tuple[int, float, float]], pts: list[tuple[float, float]]) -> dict[int, tuple[float, float]]:
    """{id: (watt-hours, covered seconds)} for requests given as (id, start, end).

    Power is taken as constant between samples at their average (trapezoid), and each slice of
    time is shared evenly among the requests running in it. Slices with no sample on both sides
    are not counted, which is what coverage measures.
    """
    out = {rid: [0.0, 0.0] for rid, _s, _e in reqs}
    if len(pts) < 2 or not reqs:
        return {k: (v[0], v[1]) for k, v in out.items()}
    # Every boundary where either the power or the set of running requests can change.
    edges = sorted({t for t, _w in pts} | {s for _r, s, _e in reqs} | {e for _r, _s, e in reqs})
    ts = [t for t, _w in pts]

    def watts_at(t: float) -> float:
        i = bisect.bisect_right(ts, t)
        if i >= len(ts) and t == ts[-1]:
            return pts[-1][1]                          # exactly on the newest sample
        if i <= 0 or i >= len(ts):
            return -1.0
        (t0, w0), (t1, w1) = pts[i - 1], pts[i]
        return w0 + (w1 - w0) * (t - t0) / (t1 - t0) if t1 > t0 else w0

    for a, b in zip(edges, edges[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2
        running = [rid for rid, s, e in reqs if s <= mid < e]
        if not running:
            continue
        wa, wb = watts_at(a), watts_at(b)
        if wa < 0 or wb < 0:
            continue                                   # outside what the meter saw
        wh = (wa + wb) / 2 * (b - a) / 3600.0 / len(running)
        for rid in running:
            out[rid][0] += wh
            out[rid][1] += b - a
    return {k: (v[0], v[1]) for k, v in out.items()}


def attribute(now: float | None = None) -> int:
    """Write energy_wh onto every finished request that has none yet. Returns how many."""
    now = now or time.time()
    pts = samples()
    with db._conn() as c:
        pending = [(r[0], r[1], r[1] + (r[2] or 0) / 1000.0) for r in c.execute(
            "SELECT id, ts, total_ms FROM router_request "
            "WHERE energy_wh IS NULL AND ts + total_ms / 1000.0 <= ? AND ts >= ?",
            (now - SETTLE_S, now - KEEP_S))]
        if not pending:
            return 0
        lo = min(s for _r, s, _e in pending)
        hi = max(e for _r, _s, e in pending)
        # Everything that overlapped them, settled or not, so a shared second is shared.
        overlap = [(r[0], r[1], r[1] + (r[2] or 0) / 1000.0) for r in c.execute(
            "SELECT id, ts, total_ms FROM router_request WHERE ts < ? AND ts + total_ms / 1000.0 > ?",
            (hi, lo))]
    window = [p for p in pts if lo - 2 * SAMPLE_S - 60 <= p[0] <= hi + 2 * SAMPLE_S + 60]
    shares = _energy_shares(sorted(set(overlap) | set(pending)), window)
    updates = []
    for rid, s, e in pending:
        wh, covered = shares.get(rid, (0.0, 0.0))
        dur = e - s
        if dur <= 0:
            updates.append((0.0, rid))
        elif covered >= MIN_COVERAGE * dur:
            updates.append((round(wh * dur / covered, 5), rid))   # scale up the unseen part
        elif not pts or pts[0][0] > s or e < now - 120:
            updates.append((-1.0, rid))                # never going to be seen: unknown
    if updates:
        with db._LOCK, db._conn() as c:
            c.executemany("UPDATE router_request SET energy_wh = ? WHERE id = ? AND energy_wh IS NULL",
                          updates)
    return len(updates)


def _loop() -> None:
    last = 0.0
    while True:
        sample()
        now = time.time()
        if now - last >= ATTRIBUTE_EVERY_S:
            last = now
            try:
                attribute(now)
            except Exception:  # noqa: BLE001 - accounting must never take the meter down
                pass
            flush(now)
        time.sleep(SAMPLE_S)


def start() -> None:
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_loop, daemon=True, name="energy-meter").start()


def by_model(since: float) -> list[dict]:
    """Routed energy per engine and model from `since`, raw rows and rollups both, biggest first."""
    acc: dict[tuple[str, str], dict] = {}
    with db._conn() as c:
        for raw, sql in (
            (True, "SELECT engine, model, SUM(CASE WHEN energy_wh >= 0 THEN energy_wh ELSE 0 END), "
                   "       SUM(energy_wh >= 0), SUM(CASE WHEN energy_wh >= 0 THEN completion_tokens ELSE 0 END) "
                   "FROM router_request WHERE ts >= ? GROUP BY engine, model"),
            (False, "SELECT engine, model, SUM(energy_wh_sum), SUM(energy_n), 0 "
                    "FROM router_rollup WHERE bucket >= ? GROUP BY engine, model"),
        ):
            for e, m, wh, n, toks in c.execute(sql, (since,)):
                d = acc.setdefault((e or "", m or ""), {"engine": e or "", "model": m or "",
                                                        "wh": 0.0, "requests": 0, "output_tokens": 0,
                                                        "_raw_wh": 0.0})
                d["wh"] += wh or 0.0
                d["requests"] += n or 0
                d["output_tokens"] += toks or 0
                if raw:
                    d["_raw_wh"] += wh or 0.0
    out = [d for d in acc.values() if d["requests"]]
    for d in out:
        d["wh"] = round(d["wh"], 3)
        # Per 1k output tokens from the individually kept requests only: a rollup row has the
        # energy of its measured requests but not their tokens, so mixing them in would skew it.
        raw_wh = d.pop("_raw_wh")
        d["wh_per_1k_output"] = round(raw_wh * 1000 / d["output_tokens"], 3) if d["output_tokens"] else None
    return sorted(out, key=lambda d: -d["wh"])


def fmt_wh(wh) -> str:
    """'—' for unknown, else Wh with sensible precision, kWh past a thousand."""
    if wh is None or wh < 0:
        return "—"
    if wh >= 1000:
        return f"{wh / 1000:.2f} kWh"
    if wh >= 100:
        return f"{wh:.0f} Wh"
    if wh >= 10:
        return f"{wh:.1f} Wh"
    return f"{wh:.2f} Wh"
