from __future__ import annotations

import sys
from datetime import datetime, timezone

import pytest

from app import telemetry as t


class TestArgvToDict:
    def test_values_bare_flags_and_volatile_removal(self):
        out = t.argv_to_dict([
            "--model", "/models/a/a.gguf", "--jinja", "--ctx-size", "8192",
            "--port", "8081", "--alias", "voice", "-p", "9", "--host", "0.0.0.0",
        ])
        assert out == {"--model": "/models/a/a.gguf", "--jinja": "true",
                       "--ctx-size": "8192"}

    def test_bare_flag_at_end(self):
        assert t.argv_to_dict(["--jinja"]) == {"--jinja": "true"}


class TestDifferingKeys:
    def test_single_config_has_nothing_to_differ(self):
        assert t.differing_keys([{"--ctx-size": "8192"}]) == []

    def test_only_differing_keys_reported(self):
        out = t.differing_keys([
            {"--ctx-size": "8192", "--flash-attn": "on"},
            {"--ctx-size": "65536", "--flash-attn": "on"},
        ])
        assert out == ["--ctx-size"]

    def test_missing_vs_present_counts_as_differing(self):
        out = t.differing_keys([{"--a": "1"}, {"--a": "1", "--b": "2"}])
        assert out == ["--b"]


class TestPercentile:
    def test_nearest_rank_no_interpolation(self):
        vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert t._pct(vals, 0.5) == 5
        assert t._pct(vals, 0.25) == 3
        assert t._pct(vals, 0.75) == 8

    def test_empty(self):
        assert t._pct([], 0.5) == 0.0


class TestParseTs:
    def test_rfc3339_z_microseconds(self):
        ts = t._parse_ts("2026-10-16T01:23:45.123456Z [60279] hello")
        assert ts == datetime(2026, 10, 16, 1, 23, 45, 123456,
                              tzinfo=timezone.utc).timestamp()

    def test_nanoseconds_parse_only_on_py311_plus(self):
        # docker logs -t emits 9-digit fractions; 3.10's fromisoformat rejects them (prod is
        # 3.12, where they parse). Pinned both ways so the host-vs-prod split is visible.
        ts = t._parse_ts("2026-10-16T01:23:45.123456789Z [60279] hello")
        if sys.version_info >= (3, 11):
            assert ts == pytest.approx(
                datetime(2026, 10, 16, 1, 23, 45, 123456,
                         tzinfo=timezone.utc).timestamp())
        else:
            assert ts == 0.0

    def test_no_prefix_or_garbage(self):
        assert t._parse_ts("[60279] 0.05.690 I no timestamp here") == 0.0
        assert t._parse_ts("2026-13-99T99:99:99Z junk") == 0.0


TS = "2026-10-16T01:00:00.000000Z"
ARGV = lambda tok, ts=TS: f"{ts} [100] I srv    load:   {tok}"
LOG = "\n".join([
    f"{TS} [100] I srv    load: spawning server instance with name=voice-cpu on port 8082",
    f"{TS} [100] I srv    load: spawning server instance with args:",
    ARGV("--model"),
    ARGV("/models/gemma/gemma.gguf"),
    ARGV("--ctx-size"),
    ARGV("8192"),
    ARGV("--spec-type"),
    ARGV("draft-mtp"),
    ARGV("--spec-draft-n-max"),
    ARGV("4"),
    ARGV("--port"),
    ARGV("8082"),
    "2026-10-16T01:00:05.000000Z [8082] I srv    load_model: loading model '/models/gemma/gemma.gguf'",
    "2026-10-16T01:00:10.000000Z [8082] I slot print_timing: id 0 | task 3 | prompt eval time = 1239.27 ms / 8297 tokens (0.15 ms per token, 6695.10 tokens per second)",
    "2026-10-16T01:00:11.000000Z [8082] I slot print_timing: id 0 | task 3 |        eval time =  777.37 ms /  111 tokens (7.07 ms per token,  141.50 tokens per second)",
    "2026-10-16T01:00:11.000000Z [8082] I slot print_timing: id 0 | task 3 | draft acceptance = 0.62963 ( 51 accepted / 81 generated), mean len = 2.13",
    "2026-10-16T01:00:12.000000Z [8082] I slot print_timing: id 1 | task 4 |        eval time =  500.00 ms /  250 tokens (2.00 ms per token,  500.00 tokens per second)",
])


class TestParseLog:
    def test_one_sample_per_task_with_binding(self):
        samples, configs = t.parse_log(LOG)
        assert len(samples) == 2
        s = next(s for s in samples if s.task == 3)
        assert (s.prompt_tokens, s.prompt_tps) == (8297, 6695.10)
        assert (s.gen_tokens, s.gen_tps) == (111, 141.50)
        assert (s.draft_acc, s.draft_len) == (0.62963, 2.13)
        assert s.alias == "voice-cpu"
        assert s.model_path == "/models/gemma/gemma.gguf"
        # spec-type carries its knobs so acceptance rates never average across a knob change
        assert s.spec_type == "draft-mtp (max=4)"
        assert s.ts == pytest.approx(
            datetime(2026, 10, 16, 1, 0, 11, tzinfo=timezone.utc).timestamp())

    def test_configs_record_argv_without_volatile(self):
        _, configs = t.parse_log(LOG)
        by_port = {c.instance: c for c in configs}
        assert by_port["8082"].alias == "voice-cpu"
        assert by_port["8082"].model_path == "/models/gemma/gemma.gguf"
        assert by_port["8082"].argv["--ctx-size"] == "8192"
        assert "--port" not in by_port["8082"].argv

    def test_garbage_degrades_to_nothing(self):
        samples, configs = t.parse_log("total nonsense\n\n[99] unrelated line")
        assert samples == []
        assert configs == []


class TestStatsProperties:
    def st(self, **kw):
        defaults = dict(n=10, gen_p50=40.0, gen_p25=30.0, gen_p75=50.0,
                        draft_len_p50=None, spec_types=[], last_ts=0.0)
        return t.Stats(**{**defaults, **kw})

    def test_draft_n_max_from_spec_string(self):
        assert self.st(spec_types=["draft-mtp (n-max=8)"]).draft_n_max == 8
        assert self.st(spec_types=["draft-mtp"]).draft_n_max == 0

    def test_draft_utilisation_normalised_by_cap(self):
        # mean len counts the guaranteed token, so the ceiling is n_max + 1
        s = self.st(draft_len_p50=3.0, spec_types=["draft-mtp (n-max=4)"])
        assert s.draft_utilisation == pytest.approx((3.0 - 1) / 4)

    def test_spread_pct(self):
        assert self.st().spread_pct == 50

    def test_age_str_empty_without_samples(self):
        assert self.st().age_str == ""
