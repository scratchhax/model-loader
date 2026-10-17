"""Characterization tests for app/vram_live.py: the live VRAM meter math.

Path translation, flag parsing, kernel kfd accounting, the segment list and the
measurement-only fallback meter are pure or filesystem-stubbable; the model
component estimates come through _cached_components, which these tests stub.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

from app import vram_live as vl
from app.config import settings


class TestHostPath:
    def test_models_prefix_maps_to_the_app_view(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "models_dir", tmp_path)
        assert vl._host_path("/models/deep/x.gguf") == tmp_path / "deep/x.gguf"

    def test_other_paths_pass_through(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "models_dir", tmp_path)
        assert str(vl._host_path("/data/y.json")) == "/data/y.json"
        assert str(vl._host_path("")) == "."   # Path normalizes empty to cwd


class TestArgvInt:
    def test_first_usable_flag_wins(self):
        assert vl._argv_int({"--ctx-size": "8192"}, "--ctx-size", "--context") == 8192
        assert vl._argv_int({"--context": " 4096 "}, "--ctx-size", "--context") == 4096

    def test_true_and_garbage_are_not_numbers(self):
        # llama flag dicts mark valueless flags as "true"; those must not become 0-and-stop.
        assert vl._argv_int({"--a": "true", "--b": "7"}, "--a", "--b") == 7
        assert vl._argv_int({"--a": "12.5"}, "--a") == 0
        assert vl._argv_int({}, "--a") == 0


class TestKfdProcessVram:
    @pytest.fixture
    def kfd(self, tmp_path, monkeypatch):
        root = tmp_path / "proc"
        real = pathlib.Path

        def shim(p):
            return real(str(p).replace("/sys/class/kfd/kfd/proc", str(root))) \
                if str(p).startswith("/sys") else real(p)
        monkeypatch.setattr(vl, "Path", shim)
        return root

    def test_largest_single_process_not_the_sum(self, kfd):
        real = pathlib.Path
        for pid, cards in (("111", {"vram_0": 1 << 30}),
                           ("222", {"vram_0": 2 << 30, "vram_1": 1 << 30})):
            d = kfd / pid
            d.mkdir(parents=True)
            for f, v in cards.items():
                (d / f).write_text(str(v))
        assert vl.kfd_process_vram_gb() == 3.0

    def test_missing_root_is_zero(self, kfd):
        assert vl.kfd_process_vram_gb() == 0.0


class TestSegments:
    def test_order_thresholds_and_percentages(self):
        segs = vl._segments(model_gb=10.0, kv_gb=5.0, overhead_gb=1.0, compute_gb=2.0,
                            other_gb=0.5, free_gb=4.0, total_gb=24.0, ctx_total=8192)
        assert [s["key"] for s in segs] == \
            ["model", "ctx", "overhead", "compute", "other", "free"]
        assert segs[0]["pct"] == pytest.approx(100.0 * 10 / 24)
        assert "8,192 tokens" in segs[1]["note"]

    def test_tiny_residuals_are_dropped(self):
        segs = vl._segments(10, 5, 1, 2, 0.004, 0.001, 20)
        assert [s["key"] for s in segs] == ["model", "ctx", "overhead", "compute"]

    def test_foreign_tenants_sit_after_ours_and_before_residuals(self):
        segs = vl._segments(10, 5, 1, 2, 0.5, 4, 24,
                            foreign=[("chatterbox-tts", 5.88)])
        assert [s["key"] for s in segs] == \
            ["model", "ctx", "overhead", "compute", "foreign", "other", "free"]
        assert segs[4]["label"] == "chatterbox-tts"

    def test_partial_offload_note(self):
        segs = vl._segments(10, 5, 1, 2, 0, 0, 20, gpu_pct=60)
        assert "60%" in segs[0]["note"]


class TestDraftCost:
    GPU = SimpleNamespace(gpu_count=2, vendor="amd")

    def test_cost_converted_to_context_at_the_models_own_rate(self, monkeypatch):
        monkeypatch.setattr(vl, "_cached_components", lambda *a, **k: {
            "draft_gb": 6.16, "kv_gb": 20.0, "ctx_total": 262144})
        d = vl.draft_cost("x", self.GPU)
        assert d["gb"] == 6.16
        assert d["ctx_tokens"] == int(6.16 / (20.0 / 262144))
        assert d["ctx_pct"] == 30.8

    def test_none_when_not_speculating(self, monkeypatch):
        assert vl.draft_cost("x", None) is None
        assert vl.draft_cost("x", SimpleNamespace(gpu_count=0)) is None
        monkeypatch.setattr(vl, "_cached_components", lambda *a, **k: {})
        assert vl.draft_cost("x", self.GPU) is None
        monkeypatch.setattr(vl, "_cached_components", lambda *a, **k: {"draft_gb": 0})
        assert vl.draft_cost("x", self.GPU) is None


def _card(i, total, used):
    return SimpleNamespace(index=i, vram_total_gb=total, vram_used_gb=used)


class TestTenantsOnlyPerCard:
    def test_measurement_only_meter(self):
        cards = [_card(0, 32.0, 10.0)]
        out = vl.tenants_only_per_card(cards, {0: [("chatterbox-tts", 5.88)]})
        m = out[0]
        assert [s["key"] for s in m["segments"]] == ["foreign", "other", "free"]
        assert m["segments"][0]["gb"] == 5.88
        assert m["segments"][1]["gb"] == pytest.approx(4.12)   # 10 - 5.88
        assert m["free_gb"] == 22.0

    def test_cards_without_foreign_tenants_get_no_meter(self):
        assert vl.tenants_only_per_card([_card(0, 32.0, 10.0)], {}) == {}

    def test_tenant_clamped_to_what_the_card_actually_holds(self):
        out = vl.tenants_only_per_card([_card(0, 32.0, 3.0)],
                                       {0: [("ghost", 6.28)]})
        assert out[0]["segments"][0]["gb"] == 3.0

    def test_zero_total_card_skipped(self):
        assert vl.tenants_only_per_card([_card(0, 0, 0)], {0: [("x", 1.0)]}) == {}


class TestPerCard:
    def test_empty_inputs(self):
        assert vl.per_card({}, []) == {}
        assert vl.per_card({"raw": {}}, []) == {}

    def test_single_card_passes_the_pooled_meter_through(self):
        vb = {"raw": {}}
        assert vl.per_card(vb, [_card(0, 32.0, 10.0)]) == {0: vb}
