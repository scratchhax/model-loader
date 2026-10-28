"""Which view of the cards feeds "W at the wall".

The figure exists to keep the box inside a UPS or a breaker, so a GPU missing from the sum is
the worst input it can take - it reads as headroom. Measured on a box with two RTX 5070s pinned
at 175 W and hammered by ComfyUI and two training runs: `41 W measured (0 W GPUs + 41 W CPU)`,
because the roll-up summed the llama backends and none was running. These pin the host publisher
feeding it instead, and the fallback that has to survive where there is no host GPU view at all.
"""
from __future__ import annotations

from app import hw

# Two cards, both busy, neither held by a llama backend. Values are the shape of what
# host-sensors publishes on this box; the fan/VRAM fields are here so the parser sees a
# complete entry.
HOST = {
    "ts": 1790567427.9, "cpu_package_w": 40.6, "cpu_temp_c": 64.0, "nvme_temp_max_c": 75.8,
    "nvidia": {"gpus": [
        {"index": 0, "bus": "00000000:01:00.0", "name": "NVIDIA GeForce RTX 5070",
         "mem_used_mib": 4462, "mem_total_mib": 12227, "util_pct": 100, "power_w": 182.3,
         "power_cap_w": 175, "temp_c": 81, "fan_pct": 77},
        {"index": 1, "bus": "00000000:08:00.0", "name": "NVIDIA GeForce RTX 5070",
         "mem_used_mib": 5129, "mem_total_mib": 12227, "util_pct": 100, "power_w": 115.97,
         "power_cap_w": 175, "temp_c": 66, "fan_pct": 33}],
        "procs": [{"pid": 2051365, "gpu": 0, "mem_mib": 4428, "sm_pct": 98}]},
}

# An AMD box: the publisher carries CPU and drives only, by design.
AMD = {"ts": 1790567427.9, "cpu_package_w": 90.0, "cpu_temp_c": 70.0, "nvme_temp_max_c": 55.0}


def _card(index: int, watts: float, temp_c: float = 60.0) -> hw.GpuCard:
    return hw.GpuCard(index=index, name="AMD Radeon", util_pct=90.0, vram_used_gb=8.0,
                      vram_total_gb=12.0, temp_c=temp_c, power_w=watts, power_cap_w=210.0)


class TestNvidiaHostView:
    def test_no_backend_running_still_counts_both_cards(self, monkeypatch):
        monkeypatch.setattr(hw, "host_sensors", lambda: HOST)
        monkeypatch.setattr(hw, "_backend_cards", lambda: [])
        roll = hw.power_rollup(55.0, 0.9)
        assert roll["gpu_w"] == 298.3                      # 182.3 + 115.97, both cards
        assert roll["gpu_src"] == "host"
        assert roll["wall_w"] == round((298.3 + 40.6 + 55.0) / 0.9, 1)
        # The temperature row rides on the same source: with 0 W GPUs it had nothing to name.
        assert roll["hottest_label"] == "GPU" and roll["hottest_c"] == 81.0

    def test_host_view_is_used_instead_of_a_backend_subset(self, monkeypatch):
        # A container pinned to one card reports one card. Adding its figure to the host's two
        # would double count the shared card, so the host's view is used on its own.
        monkeypatch.setattr(hw, "host_sensors", lambda: HOST)
        monkeypatch.setattr(hw, "_backend_cards", lambda: [_card(0, 182.3)])
        assert hw.power_rollup(55.0, 0.9)["gpu_w"] == 298.3

    def test_gpu_cards_are_per_card_for_the_panel(self, monkeypatch):
        monkeypatch.setattr(hw, "host_sensors", lambda: HOST)
        monkeypatch.setattr(hw, "_backend_cards", lambda: [])
        assert hw.power_rollup(55.0, 0.9)["gpu_cards"] == [
            {"index": 0, "watts": 182.3, "temp_c": 81.0},
            {"index": 1, "watts": 116.0, "temp_c": 66.0}]


class TestFallbackWithoutHostGpus:
    def test_amd_uses_the_backends_own_vendor_tool(self, monkeypatch):
        monkeypatch.setattr(hw, "host_sensors", lambda: AMD)
        monkeypatch.setattr(hw, "_backend_cards", lambda: [_card(0, 150.0), _card(1, 150.0)])
        roll = hw.power_rollup(55.0, 0.9)
        assert roll["gpu_w"] == 300.0 and roll["gpu_src"] == "backends"

    def test_two_backends_are_one_view_not_a_sum(self, monkeypatch):
        # Both containers were handed the same two cards, so each reports the same 300 W. The
        # roll-up takes one container's honest view rather than a total of 600 W.
        from app import services
        monkeypatch.setattr(hw, "host_sensors", lambda: {})
        monkeypatch.setattr(services, "_effective_container_names", lambda: ["llama-a", "llama-b"])
        monkeypatch.setattr(hw, "stats_for", lambda name: hw.BackendStats(
            ok=True, gpu=hw.GpuStats(vendor="amd", name="2 x GPU", util_pct=90.0,
                                     vram_used_gb=16.0, vram_total_gb=24.0, temp_c=60.0,
                                     power_w=300.0, gpu_count=2,
                                     cards=[_card(0, 150.0), _card(1, 150.0)]), container=None))
        assert hw.power_rollup(55.0, 0.9)["gpu_w"] == 300.0

    def test_stale_publisher_reports_no_gpus_and_says_which_source(self, monkeypatch):
        monkeypatch.setattr(hw, "host_sensors", lambda: {"stale": True, "age_s": 22.0})
        monkeypatch.setattr(hw, "_backend_cards", lambda: [])
        roll = hw.power_rollup(55.0, 0.9)
        assert roll["gpu_w"] == 0.0 and roll["gpu_src"] == ""
        assert roll["host_stale"] and not roll["host_ok"]
