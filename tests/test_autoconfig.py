from __future__ import annotations

import json

import pytest

from app import autoconfig as ac

GiB = 1024 ** 3


def dense_summary(**over):
    base = dict(block_count=32, attention_head_count=32, embedding_length=4096,
                attention_head_count_kv=8, context_length=131072)
    base.update(over)
    return {"arch": "llama", "model": base, "chat_template": ""}


def backend(vram=24.0, **over):
    d = dict(name="llama", vendor="cuda", vram_gb=vram, gpu_count=1,
             card_vram_gb=[vram], host_ram_gb=64.0, baseline={})
    d.update(over)
    return [d]


class TestComputeBuffer:
    # Formula pinned from the module's own calibration: (128 + 7.81 MiB/1K ctx)
    # x 0.63 no-pipeline-parallel fallback, per card, at ubatch 512.
    def test_default_nvidia(self):
        assert ac.compute_buffer_gb(32768) == pytest.approx((128 + 7.81 * 32) * 0.63 / 1024)

    def test_zero_ctx_is_zero(self):
        assert ac.compute_buffer_gb(0) == 0.0

    def test_pipeline_parallel_charges_the_larger_ask(self):
        assert ac.compute_buffer_gb(32768, pipeline_parallel=True) == \
            pytest.approx((128 + 7.81 * 32) / 1024)

    def test_tensor_split_discount(self):
        assert ac.compute_buffer_gb(32768, split_mode="tensor") == \
            pytest.approx((128 + 7.81 * 32) * 0.63 * 0.63 / 1024)

    def test_ubatch_scales_the_ctx_term(self):
        assert ac.compute_buffer_gb(32768, ubatch=1024) == \
            pytest.approx((128 + 7.81 * 32 * 2) * 0.63 / 1024)

    def test_ubatch_floor(self):
        assert ac.compute_buffer_gb(32768, ubatch=1) == \
            pytest.approx((128 + 7.81 * 32 * 0.25) * 0.63 / 1024)

    def test_amd_matches_nvidia_numbers(self):
        assert ac.compute_buffer_gb(65536, vendor="rocm") == ac.compute_buffer_gb(65536)

    def test_unknown_vendor_falls_back_to_nvidia(self):
        assert ac.compute_buffer_gb(65536, vendor="vulkan") == ac.compute_buffer_gb(65536)


class TestCacheDtypeBytes:
    @pytest.mark.parametrize("k,want", [
        ("q8_0", 1.0625), ("f16", 2.0), ("", 2.0), ("f32", 4.0),
        ("q4_0", 0.625), ("Q5_0", 0.75), ("nonsense", 2.0),
    ])
    def test_map(self, k, want):
        assert ac._cache_dtype_bytes(k) == want


class TestKvHeadsExtraction:
    def test_plain_int(self):
        assert ac._kv_first_int(8) == 8

    def test_array_mode_value(self):
        assert ac._kv_first_int({"_array": True, "sample": [8, 8, 1, 8]}) == 8

    def test_list_first(self):
        assert ac._kv_first_int([4, 8]) == 4

    def test_default_when_unparseable(self):
        assert ac._kv_first_int({}, default=12) == 12


class TestPatterns:
    def test_period(self):
        assert ac._period_of([1, 1, 1, 1]) == 1
        assert ac._period_of([1, 0, 1, 0]) == 2
        assert ac._period_of([0, 1, 1, 1, 1, 1, 1]) == 7

    def test_sample_only_when_count_matches_layers(self):
        arr = {"_array": True, "count": 24, "sample": [0, 1, 1]}
        assert ac._pattern_sample(arr, 24) == [0, 1, 1]
        assert ac._pattern_sample(arr, 48) == []

    def test_plain_list_passthrough(self):
        assert ac._pattern_sample([1, 0], 2) == [1, 0]
        assert ac._pattern_sample("junk", 2) == []


class TestKvCacheBytes:
    def test_plain_dense(self):
        # 32 layers x 8 kv heads x (128+128) bytes/token x 4096 ctx = 256 MiB
        assert ac.kv_cache_bytes("llama", 4096, 32, 8, 128, 1.0) == 268435456

    def test_asymmetric_key_value_lengths(self):
        assert ac.kv_cache_bytes("llama", 4096, 32, 8, 128, 1.0,
                                 key_length=128, value_length=96) == 234881024

    def test_hybrid_attention_ssm(self):
        # 48 layers, interval 4 -> 12 full-attention + 36 SSM at 4 MiB each
        assert ac.kv_cache_bytes("qwen3next", 8192, 48, 8, 128, 1.0,
                                 full_attention_interval=4,
                                 ssm_state_size=128) == 352321536

    def test_gemma_legacy_approximation(self):
        # no declared window: 1-in-6 global at full ctx, rest at 4096
        assert ac.kv_cache_bytes("gemma2", 8192, 26, 8, 128, 1.0) == 251658240

    def test_declared_sliding_window_with_pattern(self):
        assert ac.kv_cache_bytes("gemma3", 8192, 24, 8, 128, 1.0,
                                 sliding_window=512,
                                 sliding_window_pattern=[0, 1, 1, 1, 1, 1],
                                 key_length_swa=256,
                                 value_length_swa=256) == 109051904

    def test_shared_kv_layers(self):
        assert ac.kv_cache_bytes("llama", 4096, 32, 8, 128, 1.0,
                                 shared_kv_layers=8) == 201326592

    def test_zero_guards(self):
        assert ac.kv_cache_bytes("llama", 0, 32, 8, 128, 1.0) == 0
        assert ac.kv_cache_bytes("llama", 4096, 32, 0, 128, 1.0) == 0


class TestParseBaseline:
    def test_router_command_parses_to_known_keys_only(self):
        out = ac.parse_baseline([
            "--models-preset", "/models/models.ini", "--host", "0.0.0.0",
            "--port", "8080", "--models-max", "1",
            "-ngl", "999", "--ctx-size", "8192", "--jinja",
        ])
        assert out == {"ngl": "999", "ctx-size": "8192", "jinja": "true"}


class TestModelGeometry:
    def test_dense_geometry(self):
        g = ac.model_geometry(dense_summary())
        assert g["layers"] == 32 and g["head_dim"] == 128
        assert g["kv_heads"] == 8 and g["native_ctx"] == 131072

    def test_kv_heads_array_collapses_to_mode(self):
        g = ac.model_geometry(dense_summary(
            attention_head_count_kv={"_array": True, "sample": [8, 8, 1]}))
        assert g["kv_heads"] == 8

    def test_kv_gb_for_geometry_matches_raw_call(self):
        g = ac.model_geometry(dense_summary())
        raw = ac.kv_cache_bytes(g["arch"], 65536, g["layers"], g["kv_heads"],
                                g["head_dim"], 1.0625)
        assert ac.kv_gb_for_geometry(g, 65536, 1.0625) == raw / GiB


class TestSpecProfiles:
    def test_none_section_has_no_opinion(self):
        assert ac.match_spec_profile(None) == ""

    def test_absent_or_none_type_is_deliberate_off(self):
        assert ac.match_spec_profile({}) == "off"
        assert ac.match_spec_profile({"spec-type": "none"}) == "off"

    def test_exact_profile_match(self):
        sec = {"spec-type": "draft-mtp", "spec-draft-n-max": "4", "spec-draft-p-min": "0.25"}
        assert ac.match_spec_profile(sec) == "balanced"

    def test_knob_mismatch_is_custom(self):
        sec = {"spec-type": "draft-mtp", "spec-draft-n-max": "6", "spec-draft-p-min": "0.25"}
        assert ac.match_spec_profile(sec) == "custom"

    def test_extra_owned_knob_is_custom(self):
        sec = {"spec-type": "draft-mtp", "spec-draft-n-max": "4",
               "spec-draft-p-min": "0.25", "spec-draft-n-min": "2"}
        assert ac.match_spec_profile(sec) == "custom"

    def test_profile_keys_are_in_the_domain(self):
        assert set(ac.SPEC_PROFILE_KEYS) <= ac.AUTOCONFIG_DOMAIN


class TestVisionSwitch:
    def test_explicit_pick_wins(self):
        assert ac._vision_enabled("on", {"mmproj-auto": "off"}) is True
        assert ac._vision_enabled("off", {"mmproj": "/models/x-mmproj.gguf"}) is False

    def test_saved_mmproj_is_an_explicit_yes(self):
        assert ac._vision_enabled("", {"mmproj": "/models/x-mmproj.gguf"}) is True

    def test_mmproj_auto_off_is_an_explicit_no(self):
        assert ac._vision_enabled("", {"mmproj-auto": "off"}) is False

    def test_default_is_on(self):
        assert ac._vision_enabled("", {}) is True


class TestDomainGaps:
    def test_model_never_reported_and_extras_sorted(self):
        assert ac._domain_gaps({"ngl": "999", "model": "/models/x.gguf",
                                "zzz-new": "1", "aaa-new": "2"}) == ["aaa-new", "zzz-new"]


class TestForeignGguf:
    def test_missing_block_count_is_a_verdict(self):
        assert "wasn't made for llama.cpp" in ac.foreign_gguf_reason(
            {"arch": "qwen3_asr", "model": {}})

    def test_present_metadata_is_fine(self):
        assert ac.foreign_gguf_reason(dense_summary()) == ""

    def test_no_arch_is_a_later_shard_not_a_verdict(self):
        assert ac.foreign_gguf_reason({"arch": "", "model": {}}) == ""


class TestMoeRatio:
    @pytest.mark.parametrize("n,want", [
        (None, 0.0), (1, 0.0), (4, 0.65), (8, 0.78), (16, 0.85), (32, 0.90), (64, 0.92),
    ])
    def test_ladder(self, n, want):
        assert ac._moe_ratio(n) == want


class TestFormatCtx:
    def test_round_kilobytes(self):
        assert ac.format_ctx(8192) == "8K"
        assert ac.format_ctx(131072) == "128K"

    def test_otherwise_comma(self):
        assert ac.format_ctx(90112) == "88K"  # 90112 = 88 * 1024 exactly
        assert ac.format_ctx(90000) == "90,000"
        assert ac.format_ctx(512) == "512"


class TestAnalyzeGolden:
    """Pin analyze()'s end-to-end output for representative shapes. These are
    characterization goldens: they record what the fit maths does TODAY so the
    planned refactor can prove it changes nothing."""

    def test_small_dense_fits_whole(self):
        r = ac.analyze(summary=dense_summary(context_length=8192),
                       file_size=4 * GiB, backends=backend())
        assert r.error == ""
        assert r.recommended_backend == "llama"
        assert r.recommended_ctx == 8192
        assert r.fits_full_gpu is True
        assert r.values == {
            "cache-ram": "8192", "cache-reuse": "1", "cache-type-k": "q8_0",
            "cache-type-v": "q8_0", "ctx-size": "8192", "flash-attn": "on",
            "ngl": "999", "parallel": "1",
        }
        assert [p.key for p in r.presets] == ["balanced", "long-ctx"]

    def test_dense_overflow_picks_fast_ngl999(self):
        r = ac.analyze(summary=dense_summary(), file_size=20 * GiB, backends=backend())
        assert r.recommended_ctx == 49152
        assert r.fits_full_gpu is False
        assert [(p.key, p.ctx, p.offload_kind, p.ngl) for p in r.presets] == [
            ("fast", 49152, "", 999),
            ("balanced", 90112, "ngl", 27),
            ("long-ctx", 131072, "ngl", 22),
        ]
        assert r.values["ctx-size"] == "49152" and r.values["ngl"] == "999"

    def test_moe_overflow_hands_placement_to_fit(self):
        r = ac.analyze(summary=dense_summary(block_count=48, context_length=262144,
                                             expert_count=64),
                       file_size=60 * GiB, backends=backend())
        assert r.recommended_ctx == 90112
        # The overflow path emits fit=on and leaves ngl / tensor-split /
        # n-cpu-moe UNSET - --fit only adjusts unset args, and pinning them is
        # what disabled it. They appear in `displaced` (Fill must clear them).
        assert r.values["fit"] == "on"
        assert "ngl" not in r.values and "tensor-split" not in r.values
        assert {"ngl", "tensor-split", "n-cpu-moe", "split-mode"} <= set(r.displaced)
        assert [p.offload_kind for p in r.presets] == ["n-cpu-moe", "n-cpu-moe", "cpu-moe"]

    def test_multi_gpu_defaults_to_tensor_split(self):
        r = ac.analyze(
            summary=dense_summary(), file_size=30 * GiB,
            backends=backend(vram=48.0, gpu_count=2, card_vram_gb=[24.0, 24.0]))
        assert r.recommended_ctx == 131072
        assert r.values["split-mode"] == "tensor"

    def test_reasoning_effort_defaults_to_high(self):
        s = dense_summary()
        s["chat_template_features"] = {"accepts_reasoning_effort": True}
        r = ac.analyze(summary=s, file_size=4 * GiB, backends=backend())
        assert json.loads(r.values["chat-template-kwargs"]) == {"reasoning_effort": "high"}

    def test_tensor_split_keeps_the_chosen_mtp_profile(self):
        # Under tensor split the draft used to be wiped (an old fattn abort with long KV). On
        # build 11804 it held through 128K-token prompts at 38-45 tok/s against 25.6 without it.
        r = ac.analyze(
            summary=dense_summary(nextn_predict_layers=1), file_size=30 * GiB,
            backends=backend(vram=48.0, gpu_count=2, card_vram_gb=[24.0, 24.0]),
            section_name="Qwen-Q8", spec_profile="balanced")
        assert r.values["split-mode"] == "tensor"
        assert r.values["spec-type"] == "draft-mtp"
        assert r.values["spec-draft-n-max"] == "4" and r.values["spec-draft-p-min"] == "0.25"

    def test_tensor_split_with_speculation_off_stays_off(self):
        r = ac.analyze(
            summary=dense_summary(nextn_predict_layers=1), file_size=30 * GiB,
            backends=backend(vram=48.0, gpu_count=2, card_vram_gb=[24.0, 24.0]),
            section_name="Qwen-Q8", spec_profile="off")
        assert r.values["split-mode"] == "tensor" and r.values["spec-type"] == ""

    def test_sessions_scale_total_ctx_and_parallel(self):
        r = ac.analyze(summary=dense_summary(context_length=8192),
                       file_size=4 * GiB, backends=backend(), n_sessions=4)
        assert r.n_sessions == 4
        assert r.recommended_ctx == 8192
        assert r.recommended_total_ctx == 32768
        assert r.values["parallel"] == "4"

    def test_no_backends_error(self):
        r = ac.analyze(summary=dense_summary(), file_size=4 * GiB, backends=[])
        assert r.error.startswith("No GPU backend available")

    def test_zero_vram_backend_error(self):
        r = ac.analyze(summary=dense_summary(), file_size=4 * GiB,
                       backends=backend(vram=0.0))
        assert r.error.startswith("Backend(s) found (llama)")
        assert "GPU_VRAM=" in r.error

    def test_model_bigger_than_vram_plus_ram_error(self):
        r = ac.analyze(summary=dense_summary(), file_size=200 * GiB, backends=backend())
        assert r.error.startswith("This model needs about 200 GB")

    def test_foreign_gguf_error(self):
        r = ac.analyze(summary={"arch": "qwen3_asr", "model": {}},
                       file_size=4 * GiB, backends=backend())
        assert "wasn't made for llama.cpp" in r.error

    def test_n_sessions_clamped(self):
        r = ac.analyze(summary=dense_summary(context_length=8192),
                       file_size=4 * GiB, backends=backend(), n_sessions=99)
        assert r.n_sessions == 8
