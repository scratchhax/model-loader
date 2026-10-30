"""DeepSeek's engine pin (2026-10-30): the container's own setup swapped 0.1.52 for 0.1.54 at a start,
unasked, and re-tuned for ~13 min. STRATA_DSV41_EXE in deepseek.env is upstream's off switch."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import deepseek_engine as de
from app import updates
from tests.test_routes_smoke import web  # noqa: F401  (the fixture)

FLAVOR = "linux-x86_64-hip-rx9070"


@pytest.fixture
def box(tmp_path, monkeypatch):
    engines = tmp_path / "data-ds" / "install" / "engine-dsv41"
    for v in ("0.1.52", "0.1.54"):
        d = engines / f"{v}-{FLAVOR}"
        d.mkdir(parents=True)
        (d / de.EXE).write_text("")
    (engines / "not-a-package").mkdir()
    monkeypatch.setattr(de, "ENGINE_DIR", engines)
    monkeypatch.setattr(de, "PIN_PATH", tmp_path / "deepseek.env")
    return tmp_path


def test_installed_newest_first(box):
    assert [d["version"] for d in de.installed()] == ["0.1.54", "0.1.52"]
    assert de.installed()[0]["flavor"] == FLAVOR


def test_pin_and_unpin_round_trip(box):
    assert de.read_pin() == ""                                   # no file yet: following upstream
    ok, _ = de.set_pin(f"0.1.54-{FLAVOR}")
    text = (box / "deepseek.env").read_text()
    assert ok and f"STRATA_DSV41_EXE=/data/install/engine-dsv41/0.1.54-{FLAVOR}/strata-dsv41" in text
    assert de.read_pin() == f"0.1.54-{FLAVOR}"
    ok, _ = de.set_pin("")
    assert ok and de.read_pin() == "" and "STRATA_DSV41_EXE=\n" in (box / "deepseek.env").read_text()


def test_only_an_installed_package_can_be_pinned(box):
    ok, msg = de.set_pin(f"0.1.99-{FLAVOR}")
    assert not ok and "not installed" in msg and de.read_pin() == ""


def test_status_says_when_upstream_is_ahead_and_when_the_pin_is_gone(box):
    de.set_pin(f"0.1.54-{FLAVOR}")
    s = de.status(None, latest="0.1.55")
    assert s["pinned"]["version"] == "0.1.54" and s["behind"] and not s["pin_missing"]
    (box / "deepseek.env").write_text(f"STRATA_DSV41_EXE=/data/install/engine-dsv41/0.1.40-{FLAVOR}/strata-dsv41\n")
    s = de.status(None)
    assert s["pinned"] is None and s["pin_missing"]


def test_wired_reads_the_containers_entrypoint():
    def docker(entry):
        return SimpleNamespace(containers=SimpleNamespace(
            get=lambda n: SimpleNamespace(attrs={"Config": {"Entrypoint": entry}})))
    assert de.wired(docker(["/usr/bin/tini", "--", "/bin/sh", "/strata-pin/entrypoint-ds.sh"])) is True
    assert de.wired(docker(["/usr/bin/tini", "--", "/opt/strata/entrypoint.sh"])) is False
    assert de.wired(None) is None


def test_latest_dsv41_picks_this_flavors_newest_package():
    releases = [
        {"assets": [{"name": f"strata-dsv41-0.1.54-{FLAVOR}.tar.gz"},
                    {"name": "strata-dsv41-0.1.60-linux-x86_64-cuda-cuda13.tar.gz"}]},
        {"assets": [{"name": f"strata-dsv41-0.1.52-{FLAVOR}.tar.gz"}]},
        {"prerelease": True, "assets": [{"name": f"strata-dsv41-0.1.99-{FLAVOR}.tar.gz"}]},
    ]

    class Client:
        def get(self, url, params=None):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: releases)
    assert updates.latest_dsv41(Client(), "Niko1221/Strata", FLAVOR) == "0.1.54"


def test_the_card_pins_and_unpins(box, web):  # noqa: F811
    html = web.post("/updates/deepseek-pin", data={"folder": f"0.1.54-{FLAVOR}"}).text
    assert "DeepSeek engine" in html and ">pinned<" in html and "Unpin" in html
    assert de.read_pin() == f"0.1.54-{FLAVOR}"
    html = web.post("/updates/deepseek-pin", data={"folder": ""}).text
    assert "follows upstream" in html and "Pin 0.1.54" in html and de.read_pin() == ""
