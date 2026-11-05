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
        _samples.append((ts, watts))
        while _samples and _samples[0][0] < ts - KEEP_S:
            _samples.popleft()


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
        time.sleep(SAMPLE_S)


def start() -> None:
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_loop, daemon=True, name="energy-meter").start()


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
