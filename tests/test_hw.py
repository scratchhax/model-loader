from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from app import hw


@pytest.fixture(autouse=True)
def no_vendor_cache():
    hw._VENDOR_CACHE.clear()
    yield
    hw._VENDOR_CACHE.clear()


def card(name):
    return NS(name=name, image=NS(tags=[name]))


class TestDetectVendor:
    def test_image_tag_decides_vendor(self):
        assert hw._detect_vendor(card("ghcr.io/ggml-org/llama-server:rocm")) == "amd"
        assert hw._detect_vendor(card("hip-base:1")) == "amd"
        assert hw._detect_vendor(card("nvcc/cuda:12")) == "nvidia"
        assert hw._detect_vendor(card("nvidia/llama:latest")) == "nvidia"
        assert hw._detect_vendor(card("vulkan-server:1")) == "vulkan"
        assert hw._detect_vendor(card("plain-cpu:1")) == "unknown"

    def test_result_is_cached_per_name(self):
        assert hw._detect_vendor(card("x-rocm:1")) == "amd"
        assert hw._detect_vendor(card("x-rocm:1")) == "amd"  # even from a fresh object


class TestGpuCardMath:
    def cardobj(self, used, total):
        return hw.GpuCard(index=0, name="c", util_pct=0.0, vram_used_gb=used,
                          vram_total_gb=total, temp_c=0.0, power_w=0.0)

    def test_free_and_pct(self):
        c = self.cardobj(10.0, 24.0)
        assert c.vram_free_gb == 14.0 and c.vram_pct == 41.7

    def test_overdrawn_and_zero_total(self):
        assert self.cardobj(30.0, 24.0).vram_free_gb == 0.0
        assert self.cardobj(5.0, 0.0).vram_pct == 0.0


class TestSparkline:
    def test_needs_two_points(self):
        assert hw.sparkline([]) == "" and hw.sparkline([5.0]) == ""

    def test_auto_scale_and_hint(self):
        assert hw.sparkline([0.0, 1.0]) == "0.0,20.0 100.0,2.0"
        assert hw.sparkline([0.0, 1.0], max_hint=2.0) == "0.0,20.0 100.0,11.0"

    def test_values_over_hint_clamp(self):
        assert hw.sparkline([5.0, 5.0], max_hint=1.0) == "0.0,2.0 100.0,2.0"

    def test_all_zero_does_not_divide_by_zero(self):
        assert hw.sparkline([0.0, 0.0]) == "0.0,20.0 100.0,20.0"


class TestAmdParsing:
    def test_substring_match_with_exclusion(self):
        c = {"VRAM Total Memory (B)": "68719476736",
             "Current Socket Graphics Package Power (W)": "123",
             "Average Graphics Package Power (W)": "99"}
        assert hw._amd_num(c, ("graphics", "power"), ("average",)) == 123.0
        assert hw._amd_num(c, ("vram", "total", "memory")) == 68719476736.0
        assert hw._amd_num(c, ("nothing",)) == 0.0

    def test_non_numeric_values_are_skipped(self):
        c = {"power (W)": "N/A", "other power (W)": "5"}
        assert hw._amd_num(c, ("power",)) == 5.0

    def test_name_prefers_series_and_appends_gfx(self):
        c = {"Card Series": "AMD Radeon Graphics", "GFX Version": "gfx1201"}
        assert hw._amd_name("card0", c) == "AMD Radeon Graphics (gfx1201)"

    def test_na_values_fall_through(self):
        c = {"Card Series": "N/A", "Card Model": "R9700"}
        assert hw._amd_name("card0", c) == "R9700"

    def test_nothing_useful_returns_card_key(self):
        assert hw._amd_name("card0", {}) == "card0"
