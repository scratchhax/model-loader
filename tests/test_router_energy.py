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


def test_a_conversation_with_nothing_measured_says_so_not_zero(web):  # noqa: F811
    import re
    import time
    ra._record(ts=time.time() - 500, conv="old-run", engine="llama", model="m", status=200,
               completion_tokens=5, total_ms=1000)
    with db._LOCK, db._conn() as c:
        c.execute("UPDATE router_request SET energy_wh = -1")
    page = web.get("/router").text
    row = page.split("/router/conversation/old-run")[1].split("</tr>")[0]
    assert "0.00 Wh" not in row and re.search(r"whitespace-nowrap\">—", row)


def test_formatting():
    assert energy.fmt_wh(None) == "—" and energy.fmt_wh(-1) == "—"
    assert energy.fmt_wh(0.123) == "0.12 Wh" and energy.fmt_wh(12.34) == "12.3 Wh"
    assert energy.fmt_wh(456) == "456 Wh" and energy.fmt_wh(2500) == "2.50 kWh"


# ---------------------------------------------------------------- the box's running total

def test_the_box_meter_fills_hours_and_survives_a_flush(rdb, monkeypatch):
    monkeypatch.setattr(energy, "_unflushed", {})
    h = 1_800_003_600.0                          # an hour boundary
    flat(360.0, h - 60, h + 60)                  # two minutes straddling it, 2 s apart
    energy.flush(now=h + 60)
    with db._conn() as c:
        rows = dict(c.execute("SELECT bucket, wh FROM energy_hour").fetchall())
    assert rows[h - 3600] == pytest.approx(360 * 60 / 3600, rel=0.05)
    assert rows[h] == pytest.approx(360 * 60 / 3600, rel=0.05)
    wh, metered = energy.box_energy(h - 3600)
    assert wh == pytest.approx(12.0, rel=0.01) and metered == pytest.approx(120, abs=1)


def test_a_long_gap_is_not_drawn_across(rdb, monkeypatch):
    monkeypatch.setattr(energy, "_unflushed", {})
    energy.add_sample(T0, 300.0)
    energy.add_sample(T0 + 600, 300.0)           # ten minutes of nothing: a restart
    energy.flush(now=T0 + 600)
    assert energy.box_energy(T0 - 3600) == (0.0, 0.0)


def test_summary_puts_routed_beside_the_box(rdb, monkeypatch):
    import time
    monkeypatch.setattr(energy, "_unflushed", {})
    monkeypatch.setattr(energy, "_summary_cache", None)
    now = time.time()
    flat(360.0, now - 100, now - 2)
    rec(now - 80, 36)
    energy.attribute(now=now)
    s = energy.summary(now)
    assert s["box_wh"] == pytest.approx(360 * 98 / 3600, rel=0.05)
    assert s["routed_wh"] == pytest.approx(3.6, rel=0.01) and s["routed_n"] == 1
    assert s["now_w"] == 360.0


def test_by_model_keeps_per_token_to_the_measured_rows(rdb):
    flat(360.0, T0 - 10, T0 + 100)
    rec(T0, 36, out=1000)
    energy.attribute(now=T0 + 100)
    (m,) = energy.by_model(0)
    assert m["model"] == "m" and m["requests"] == 1
    assert m["wh"] == pytest.approx(3.6, rel=1e-3) and m["wh_per_1k_output"] == pytest.approx(3.6, rel=1e-3)


def test_overview_strip_and_api_carry_the_running_total(web, monkeypatch):  # noqa: F811
    monkeypatch.setattr(energy, "summary", lambda *a, **k: {
        "window_s": 86400.0, "box_wh": 2140.0, "box_metered_s": 86400, "box_coverage": 1.0,
        "routed_wh": 380.0, "routed_n": 12, "requests": 12, "now_w": 300.0})
    strip = web.get("/boxbar").text
    assert "24 h" in strip and "2.14 kWh" in strip and "380 Wh routed" in strip
    api = web.get("/api/v1/energy").json()
    assert api["windows"]["24h"]["box_wh"] == 2140.0 and len(api["days"]) == 30
    assert "/api/v1/energy" in web.get("/api/v1").json()["endpoints"]
    assert web.get("/api/v1/box").json()["energy"]["routed_wh"] == 380.0


def test_a_part_day_box_total_is_not_set_beside_a_full_day_routed_one(web, monkeypatch):  # noqa: F811
    # The box meter started 48 s ago, the routed figures go back further: "3.18 Wh of 0.97 Wh"
    # would read as impossible. The strip shows the routed figure alone until the day is covered.
    monkeypatch.setattr(energy, "summary", lambda *a, **k: {
        "window_s": 86400.0, "box_wh": 0.97, "box_metered_s": 48, "box_coverage": 0.001,
        "routed_wh": 3.18, "routed_n": 2, "requests": 2, "now_w": 70.0})
    line = web.get("/boxbar").text.split('id="bar-power-num"')[1].split("</div>")[1]
    assert "3.18 Wh</span> routed, 24 h" in line and "0.97" not in line
