"""Each routed request gets the wall energy the box drew while it ran (app.energy).

The local answer to a hosted API's per-run cost: watt-hours instead of money, per request,
per conversation (one agent run) and in the Router tab's totals.
"""
from __future__ import annotations

import pytest

from app import db, energy
from app import router_api as ra
from tests.test_routes_smoke import web  # noqa: F401 - the fixture: real app, empty box

T0 = 1_800_000_000.0


@pytest.fixture
def rdb(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "router.db")
    db.init()
    ra._init_db()
    ra._init_rollup()
    monkeypatch.setattr(energy, "_samples", energy.deque())
    monkeypatch.setattr(db, "get_setting", lambda k, d="": d)   # recording on
    return tmp_path


def flat(watts, start, end, step=2.0):
    t = start
    while t <= end:
        energy.add_sample(t, watts)
        t += step


def rec(ts, secs, conv="c1", out=100):
    ra._record(ts=ts, conv=conv, engine="strata", model="m", status=200,
               completion_tokens=out, total_ms=secs * 1000)


def energies():
    with db._conn() as c:
        return [r[0] for r in c.execute("SELECT energy_wh FROM router_request ORDER BY ts")]


# ---------------------------------------------------------------- the arithmetic

def test_one_request_at_a_steady_draw():
    pts = [(T0 + i * 2.0, 360.0) for i in range(30)]
    (wh, covered), = energy._energy_shares([(1, T0 + 2, T0 + 38)], pts).values()
    assert wh == pytest.approx(360 * 36 / 3600)            # 3.6 Wh
    assert covered == pytest.approx(36)


def test_overlapping_requests_share_and_add_up_to_the_box():
    pts = [(T0 + i * 2.0, 400.0) for i in range(40)]
    sh = energy._energy_shares([(1, T0, T0 + 40), (2, T0 + 20, T0 + 60)], pts)
    # 0-20 s alone, 20-40 s shared, 40-60 s alone
    assert sh[1][0] == pytest.approx(400 * (20 + 10) / 3600)
    assert sh[2][0] == pytest.approx(400 * (10 + 20) / 3600)
    assert sh[1][0] + sh[2][0] == pytest.approx(400 * 60 / 3600)   # never more than the wall saw


def test_power_is_interpolated_between_samples():
    pts = [(T0, 100.0), (T0 + 10, 300.0)]
    (wh, _), = energy._energy_shares([(1, T0, T0 + 10)], pts).values()
    assert wh == pytest.approx(200 * 10 / 3600)


# ---------------------------------------------------------------- written onto the rows

def test_finished_requests_get_their_energy_and_running_ones_wait(rdb):
    flat(360.0, T0 - 10, T0 + 100)
    rec(T0, 36)                                  # ended at T0+36
    rec(T0 + 90, 5)                              # ends at T0+95: not settled at T0+97
    assert energy.attribute(now=T0 + 97) == 1
    first, second = energies()
    assert first == pytest.approx(3.6, rel=1e-3) and second is None


def test_a_request_the_meter_never_saw_is_unknown_not_guessed(rdb):
    flat(300.0, T0 + 1000, T0 + 1100)            # the meter started after a restart
    rec(T0, 30)
    energy.attribute(now=T0 + 1100)
    assert energies() == [-1.0]


def test_a_settled_request_is_not_measured_twice(rdb):
    flat(360.0, T0 - 10, T0 + 100)
    rec(T0, 36)
    energy.attribute(now=T0 + 60)
    assert energy.attribute(now=T0 + 99) == 0


# ---------------------------------------------------------------- totals and rollups

def test_totals_and_compaction_carry_energy(rdb):
    flat(360.0, T0 - 10, T0 + 100)
    rec(T0, 36)
    rec(T0 + 40, 36, conv="c2")
    energy.attribute(now=T0 + 100)
    before = ra.totals(0)
    assert before["energy_wh"] == pytest.approx(7.2, rel=1e-3) and before["energy_n"] == 2
    ra.compact(now=T0 + 30 * 86400)              # folded into hourly rows
    after = ra.totals(0)
    assert after["energy_wh"] == pytest.approx(7.2, rel=1e-3) and after["energy_n"] == 2
    with db._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM router_request").fetchone()[0] == 0


def test_an_old_database_gains_the_columns(tmp_path, monkeypatch):
    import sqlite3
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE router_request (id INTEGER PRIMARY KEY, ts REAL NOT NULL, "
                  "conv TEXT NOT NULL DEFAULT '', total_ms REAL NOT NULL DEFAULT 0)")
        c.execute("CREATE TABLE router_rollup (grain TEXT NOT NULL, bucket REAL NOT NULL, "
                  "engine TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '', "
                  "PRIMARY KEY (grain, bucket, engine, model))")
    monkeypatch.setattr(db, "_DB_PATH", path)
    ra._init_db()
    ra._init_rollup()
    with db._conn() as c:
        assert "energy_wh" in {r[1] for r in c.execute("PRAGMA table_info(router_request)")}
        assert {"energy_wh_sum", "energy_n"} <= {r[1] for r in c.execute("PRAGMA table_info(router_rollup)")}


# ---------------------------------------------------------------- the pages

def test_router_tab_and_conversation_show_energy(web):  # noqa: F811
    import time
    now = time.time()
    energy.add_sample(now - 200, 360.0)
    ra._record(ts=now - 100, conv="pi-run", engine="strata", model="m", status=200,
               completion_tokens=500, total_ms=36_000)
    with db._LOCK, db._conn() as c:
        c.execute("UPDATE router_request SET energy_wh = 3.6")
    ra._record(ts=now - 20, conv="pi-run", engine="strata", model="m", status=200,
               completion_tokens=10, total_ms=1000)     # still being measured
    page = web.get("/router").text
    assert ">Energy<" in page and "3.60 Wh" in page and "still being measured" in page
    conv = web.get("/router/conversation/pi-run").text
    assert "3.60 Wh at the wall" in conv and "1 of 2 turns measured" in conv


def test_formatting():
    assert energy.fmt_wh(None) == "—" and energy.fmt_wh(-1) == "—"
    assert energy.fmt_wh(0.123) == "0.12 Wh" and energy.fmt_wh(12.34) == "12.3 Wh"
    assert energy.fmt_wh(456) == "456 Wh" and energy.fmt_wh(2500) == "2.50 kWh"
