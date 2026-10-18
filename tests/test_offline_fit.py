"""Tests for sizing autoconfig while the llama backend is stopped.

The card totals are a property of the machine, not of a running container: sysfs
publishes them, and _backend_list now offers a stopped GPU backend with those facts
(flagged offline, disclosed by the panel) instead of hard-stopping.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app import hw


CARD_BYTES = 34208743424  # one R9700: 31.9 GiB


def _mk_sysfs(root: Path, cards: dict[str, object], driver: str | None = None):
    for name, content in cards.items():
        d = root / name / "device"
        d.mkdir(parents=True)
        f = d / "mem_info_vram_total"
        f.write_text(str(content))
        if driver:
            drv = root / "drivers" / driver
            drv.mkdir(parents=True, exist_ok=True)
            os.symlink(drv, d / "driver")
    return root


class TestSysfsFacts:
    def test_card_totals_in_card_order(self, tmp_path, monkeypatch):
        _mk_sysfs(tmp_path, {"card0": CARD_BYTES, "card1": CARD_BYTES})
        monkeypatch.setattr(hw, "_DRM_CARDS", tmp_path)
        assert hw.sysfs_card_vram_gb() == [31.9, 31.9]

    def test_unreadable_card_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        _mk_sysfs(tmp_path, {"card0": CARD_BYTES, "card1": "garbage"})
        monkeypatch.setattr(hw, "_DRM_CARDS", tmp_path)
        assert hw.sysfs_card_vram_gb() == [31.9]

    def test_missing_drm_is_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hw, "_DRM_CARDS", tmp_path / "gone")
        assert hw.sysfs_card_vram_gb() == []

    def test_vendor_from_driver_symlink(self, tmp_path, monkeypatch):
        _mk_sysfs(tmp_path, {"card0": CARD_BYTES}, driver="amdgpu")
        monkeypatch.setattr(hw, "_DRM_CARDS", tmp_path)
        assert hw.sysfs_gpu_vendor() == "rocm"

    def test_vendor_unknown_without_symlinks(self, tmp_path, monkeypatch):
        _mk_sysfs(tmp_path, {"card0": CARD_BYTES})
        monkeypatch.setattr(hw, "_DRM_CARDS", tmp_path)
        assert hw.sysfs_gpu_vendor() == ""


class FakeContainer:
    def __init__(self, tags, status="exited", cmd=None):
        self.image = type("I", (), {"tags": tags})()
        self.status = status
        self.attrs = {"Config": {"Cmd": cmd or ["--models-preset", "/models/models.ini"]}}


class FakeClient:
    def __init__(self, container):
        self._c = container

    class _Get:
        def __init__(self, container):
            self.container = container

        def get(self, name):
            return self.container

    @property
    def containers(self):
        return self._Get(self._c)


@pytest.fixture
def offline_env(monkeypatch):
    """A discovered llama backend, stopped, probe dead, two cards in sysfs."""
    from app import fit
    from app import services
    monkeypatch.setattr(services, "_effective_container_names", lambda: ["llama-9700x2"])
    monkeypatch.setattr(services, "engine_for", lambda n: "llama")
    monkeypatch.setattr(services, "_docker_client",
                        lambda: FakeClient(FakeContainer(["ghcr.io/x/llama.cpp:server-rocm"])))
    monkeypatch.setattr(hw, "vram_gb_for", lambda n: 0.0)
    monkeypatch.setattr(hw, "sysfs_card_vram_gb", lambda: [31.9, 31.9])
    monkeypatch.setattr(hw, "sysfs_gpu_vendor", lambda: "rocm")
    monkeypatch.setattr(fit, "_container_status", lambda n: "exited")
    monkeypatch.setattr(fit, "_fit_host_ram", lambda: 93.6)
    return fit


class TestBackendListOffline:
    def test_stopped_gpu_backend_is_offered_with_machine_facts(self, offline_env):
        [b] = offline_env._backend_list()
        assert b["name"] == "llama-9700x2"
        assert b["offline"] is True
        assert b["vram_gb"] == pytest.approx(63.8)
        assert b["gpu_count"] == 2
        assert b["card_vram_gb"] == [31.9, 31.9]
        assert b["vendor"] == "rocm"
        # The strata-held zeroing is skipped offline: the plan is for after the switch.
        assert 0.0 not in b["card_vram_gb"]

    def test_running_backend_with_dead_probe_is_still_dropped(self, offline_env, monkeypatch):
        # Machine facts may not describe a running CPU build's devices; unchanged behavior.
        monkeypatch.setattr(offline_env, "_container_status", lambda n: "running")
        assert offline_env._backend_list() == []

    def test_no_sysfs_facts_still_hard_stops(self, offline_env, monkeypatch):
        monkeypatch.setattr(hw, "sysfs_card_vram_gb", lambda: [])
        assert offline_env._backend_list() == []

    def test_running_backend_path_is_unchanged(self, monkeypatch):
        from app import fit
        from app import services
        monkeypatch.setattr(services, "_effective_container_names", lambda: ["llama-9700x2"])
        monkeypatch.setattr(services, "engine_for", lambda n: "llama")
        monkeypatch.setattr(services, "_docker_client",
                            lambda: FakeClient(FakeContainer(
                                ["ghcr.io/x/llama.cpp:server-rocm"], status="running")))
        monkeypatch.setattr(hw, "vram_gb_for", lambda n: 63.7)
        monkeypatch.setattr(hw, "gpu_count_for", lambda n: 2)
        monkeypatch.setattr(fit, "_fit_card_vram", lambda n: [31.9, 0.0])
        monkeypatch.setattr(fit, "_fit_host_ram", lambda: 60.0)
        [b] = fit._backend_list()
        assert "offline" not in b
        assert b["vram_gb"] == 63.7
        assert b["gpu_count"] == 2
        assert b["card_vram_gb"] == [31.9, 0.0]   # held-card zeroing still applies live


class TestRecommendationFlag:
    def test_offline_fit_defaults_false_and_round_trips(self):
        from app.autoconfig import Recommendation
        r = Recommendation(plans=[], recommended_backend="", recommended_ctx=0)
        assert r.offline_fit is False
        assert Recommendation(plans=[], recommended_backend="x", recommended_ctx=0,
                              offline_fit=True).offline_fit is True
