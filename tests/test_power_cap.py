"""The GPU power cap: the wanted value in /data/power-cap.env, and whether the cards hold it.

From 2026-10-22 amd-power-cap.service failed at every boot and both cards sat at 300 W with nothing
on screen. These pin the check that says so, and the setting that asks the host for a new cap.
"""
from __future__ import annotations

import os
import time

import pytest

from app import power_cap
from app.config import settings
from tests.test_routes_smoke import web  # noqa: F401  (the fixture)


def _card(root, pci, cap, lo=210, hi=300):
    hw = root / pci / "hwmon" / "hwmon4"
    hw.mkdir(parents=True)
    (hw / "power1_cap").write_text(str(cap * 1_000_000))
    (hw / "power1_cap_min").write_text(str(lo * 1_000_000))
    (hw / "power1_cap_max").write_text(str(hi * 1_000_000))
    return hw


@pytest.fixture
def box(tmp_path, monkeypatch):
    sysfs = tmp_path / "amdgpu"
    sysfs.mkdir()
    data = tmp_path / "data"     # the same folder the web fixture uses, when both are in play
    data.mkdir(exist_ok=True)
    monkeypatch.setattr(power_cap, "SYSFS", str(sysfs))
    monkeypatch.setattr(settings, "data_dir", data)
    return sysfs


def _age(seconds):
    t = time.time() - seconds
    os.utime(power_cap.env_path(), (t, t))


def test_the_cards_are_read_in_pci_order_with_their_range(box):
    _card(box, "0000:08:00.0", 300)
    _card(box, "0000:03:00.0", 210)
    cs = power_cap.cards()
    assert [(c.pci, c.cap_w, c.min_w, c.max_w) for c in cs] == [
        ("0000:03:00.0", 210, 210, 300), ("0000:08:00.0", 300, 210, 300)]
    assert power_cap.allowed(cs) == (210, 300)


@pytest.mark.parametrize("text, want", [
    ("AMD_POWER_CAP_W=230\n", 230),
    ("# note\nAMD_POWER_CAP_W = 250\n", 250),
    ("AMD_POWER_CAP_W=1;reboot\n", 210),     # the script treats this as 210, so must we
    ("SOMETHING_ELSE=5\n", 210),
])
def test_the_wanted_cap_is_read_the_way_the_host_script_reads_it(box, text, want):
    power_cap.env_path().write_text(text)
    assert power_cap.desired() == want


def test_no_file_means_the_services_own_default(box):
    assert power_cap.desired() == 210


def test_set_desired_writes_the_one_line_the_unit_reads(box):
    power_cap.set_desired(240)
    assert power_cap.env_path().read_text() == "AMD_POWER_CAP_W=240\n"


def test_a_card_above_the_cap_is_reported_once_the_host_has_had_time(box):
    _card(box, "0000:03:00.0", 210)
    _card(box, "0000:08:00.0", 300)
    power_cap.set_desired(210)
    _age(60)
    s = power_cap.status()
    assert [c.pci for c in s["off"]] == ["0000:08:00.0"] and not s["settling"]


def test_a_fresh_change_is_settling_not_an_alarm(box):
    _card(box, "0000:03:00.0", 210)
    power_cap.set_desired(250)
    s = power_cap.status()
    assert s["settling"] and s["off"] == []


def test_a_cap_below_the_floor_is_held_at_the_floor_without_an_alarm(box):
    # The script clamps to the card's range, so 210 on a card asked for 200 is what was asked.
    _card(box, "0000:03:00.0", 210)
    power_cap.env_path().write_text("AMD_POWER_CAP_W=200\n")
    _age(60)
    assert power_cap.status()["off"] == []


def test_the_route_writes_a_cap_inside_the_range(web, box):  # noqa: F811
    _card(box, "0000:03:00.0", 210)
    r = web.post("/power/cap", data={"cap_w": "250"})
    assert r.status_code == 200 and "Asked for 250 W" in r.text
    assert power_cap.desired() == 250


@pytest.mark.parametrize("value", ["150", "400", "abc", ""])
def test_the_route_refuses_anything_outside_the_range(web, box, value):  # noqa: F811
    _card(box, "0000:03:00.0", 210)
    power_cap.set_desired(210)
    r = web.post("/power/cap", data={"cap_w": value})
    assert r.status_code == 200 and "Nothing was changed" in r.text
    assert power_cap.desired() == 210


def test_the_power_card_names_the_card_that_is_not_capped(web, box):  # noqa: F811
    _card(box, "0000:03:00.0", 210)
    _card(box, "0000:08:00.0", 300)
    power_cap.set_desired(210)
    _age(60)
    html = web.get("/power").text
    assert "GPU power cap not held" in html and "card 1 at 300 W" in html and "set to 210 W" in html


def test_no_warning_when_the_cards_hold_the_cap(web, box):  # noqa: F811
    _card(box, "0000:03:00.0", 210)
    _card(box, "0000:08:00.0", 210)
    power_cap.set_desired(210)
    _age(60)
    assert "GPU power cap not held" not in web.get("/power").text
