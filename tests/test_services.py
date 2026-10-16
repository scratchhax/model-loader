from __future__ import annotations

import pytest

from app import services as s


def rates(**over):
    d = dict(cur_gen_tps=0.0, cur_gen_tokens=0, cur_pp_tps=0.0, cur_pp_pct=0,
             cur_pp_done=0, cur_pp_total=0, cur_prompt_tokens=0,
             last_gen_tps=0.0, last_gen_tokens=0, last_pp_tps=0.0, newest="",
             open_task=None, open_slot=None, expert_hit_pct=None)
    return {**d, **over}


def slotdict(i=0, processing=True, ctx_used=1000, ctx_total=8192,
             cached=0, processed=0, task=5, decoded=0):
    return {"id": i, "is_processing": processing, "n_prompt_tokens": ctx_used,
            "n_ctx": ctx_total, "n_prompt_tokens_cache": cached,
            "n_prompt_tokens_processed": processed, "id_task": task,
            "next_token": [{"n_decoded": decoded}] if decoded else []}


@pytest.fixture
def clean_slot_state():
    for store in (s._speed_prev, s._gen_ema):
        for k in [k for k in store if k.startswith("c#")]:
            del store[k]
    yield
    for store in (s._speed_prev, s._gen_ema):
        for k in [k for k in store if k.startswith("c#")]:
            del store[k]


class TestRatesFromLog:
    @pytest.fixture
    def log(self, monkeypatch):
        def _set(text, ok=True):
            monkeypatch.setattr(
                s, "container_logs",
                lambda name, tail, current_run_only: (ok, text))
        return _set

    def test_open_task_and_bound_rates(self, log):
        log("I slot launch_slot_: id  0 | task 100 | processing task\n"
            "I slot print_timing: id 0 | task 100 | prompt processing, n_tokens = 7105, "
            "progress = 0.50, t = 4.70 s / 1511.80 tokens per second\n"
            "I slot print_timing: id 0 | task 100 | n_gen = 806, tg =  26.02 t/s, "
            "tg_3s =  25.99 t/s")
        r = s._rates_from_log("c")
        assert (r["open_slot"], r["open_task"]) == (0, 100)
        assert r["cur_gen_tps"] == 25.99  # tg_3s preferred over tg
        assert r["cur_gen_tokens"] == 806
        assert r["newest"] == "gen"
        assert r["last_gen_tps"] == 25.99

    def test_released_task_has_no_open_task(self, log):
        log("I slot launch_slot_: id  0 | task 100 | processing task\n"
            "I slot      release: id  0 | task 100 | stop processing: n_tokens = 834\n"
            "I slot print_timing: id 0 | task 100 | n_gen = 806, tg =  26.02 t/s")
        r = s._rates_from_log("c")
        assert r["open_task"] is None
        assert r["cur_gen_tps"] == 0.0
        assert r["last_gen_tps"] == 26.02  # last_* still reported, dimmed

    def test_newest_task_wins_across_slots(self, log):
        log("I slot launch_slot_: id  0 | task 5 | processing task\n"
            "I slot launch_slot_: id  1 | task 7 | processing task")
        r = s._rates_from_log("c")
        assert (r["open_slot"], r["open_task"]) == (1, 7)

    def test_prefill_progress_bound_to_open_task(self, log):
        log("I slot launch_slot_: id  0 | task 9 | processing task\n"
            "I slot print_timing: id 0 | task 9 | prompt processing, n_tokens = 4000, "
            "progress = 0.25, t = 2.00 s / 2000.00 tokens per second")
        r = s._rates_from_log("c")
        assert r["cur_pp_tps"] == 2000.0 and r["cur_pp_pct"] == 25
        assert r["cur_pp_done"] == 4000
        assert r["newest"] == "prefill"

    def test_proxy_spam_lines_are_ignored(self, log):
        log("I srv  proxy_request: GET /slots | task 999 | n_gen = 1, tg = 999.9 t/s\n"
            "I slot launch_slot_: id  0 | task 1 | processing task")
        r = s._rates_from_log("c")
        assert r["last_gen_tps"] == 0.0

    def test_log_failure_returns_zeros(self, log):
        log("", ok=False)
        r = s._rates_from_log("c")
        assert r["open_task"] is None and r["cur_gen_tps"] == 0.0


class TestSlotSpeed:
    def test_idle_clears_smoothers(self, clean_slot_state):
        s._speed_prev["c#0"] = (1.0, 10, 5, True)
        s._gen_ema["c#0"] = 30.0
        out = s._slot_speed("c", slotdict(processing=False), rates(), now=100.0)
        assert out.state == "idle"
        assert "c#0" not in s._speed_prev and "c#0" not in s._gen_ema

    def test_first_frame_borrows_the_log_rate(self, clean_slot_state):
        out = s._slot_speed("c", slotdict(decoded=10),
                            rates(open_task=5, cur_gen_tps=42.0), now=100.0)
        assert out.state == "generating"
        assert out.gen_tps == 42.0

    def test_second_frame_uses_the_token_delta(self, clean_slot_state):
        s._slot_speed("c", slotdict(decoded=10), rates(open_task=5), now=100.0)
        out = s._slot_speed("c", slotdict(decoded=30), rates(open_task=5), now=102.0)
        assert out.gen_tps == pytest.approx(10.0)  # 20 tokens / 2 s, first EMA sample

    def test_task_change_does_not_borrow_previous_smoothing(self, clean_slot_state):
        s._slot_speed("c", slotdict(decoded=10), rates(open_task=5), now=100.0)
        out = s._slot_speed("c", slotdict(decoded=30, task=6),
                            rates(open_task=None), now=102.0)
        assert out.gen_tps == 0.0

    def test_prefill_counts_cache_as_read(self, clean_slot_state):
        out = s._slot_speed("c", slotdict(decoded=0, ctx_used=1000, cached=800,
                                          processed=100), rates(), now=100.0)
        assert out.state == "prefill"
        assert (out.prefill_done, out.prefill_total, out.prefill_pct) == (900, 1000, 90)


class TestSoloPath:
    def test_strata_with_live_newest_is_outside(self):
        assert s._outside_slots("strata", {"newest": "generating"}) is True

    def test_llama_never_outside(self):
        assert s._outside_slots("llama", {"newest": "generating"}) is False

    def test_strata_idle_is_not_outside(self):
        assert s._outside_slots("strata", {"newest": ""}) is False


class TestPruneSlotState:
    def test_only_live_slots_survive(self, clean_slot_state):
        s._speed_prev.update({"c#0": (1, 1, 1, True), "c#1": (1, 1, 1, True),
                              "other#0": (1, 1, 1, True)})
        s._gen_ema.update({"c#0": 1.0, "c#1": 2.0, "other#0": 3.0})
        s._prune_slot_state("c", {0})
        assert set(k for k in s._speed_prev if k.startswith("c#")) == {"c#0"}
        assert set(k for k in s._gen_ema if k.startswith("c#")) == {"c#0"}
        assert "other#0" in s._speed_prev  # other containers untouched


class TestStripLayout:
    def test_density_thresholds(self):
        assert s.slot_density(1) == "roomy" and s.slot_density(4) == "roomy"
        assert s.slot_density(5) == "tight" and s.slot_density(8) == "tight"
        assert s.slot_density(9) == "dense"

    def test_column_thresholds(self):
        assert s.slot_columns(4) == 1
        assert s.slot_columns(8) == 2 and s.slot_columns(12) == 2
        assert s.slot_columns(16) == 3


class TestContainerAttrs:
    def test_ports_prefer_bound_internal(self):
        attrs = {"NetworkSettings": {"Ports": {
            "8081/tcp": None, "8080/tcp": [{"HostPort": "8082"}]}}}
        result, internal = s._extract_ports(attrs)
        assert internal == 8080
        assert result == ["8082->8080/tcp"]

    def test_unbound_only_falls_back_to_any(self):
        attrs = {"NetworkSettings": {"Ports": {"8080/tcp": None}}}
        result, internal = s._extract_ports(attrs)
        assert internal == 8080 and result == []

    def test_api_key_from_cmd_and_entrypoint(self):
        assert s._api_key_from_attrs(
            {"Config": {"Cmd": ["llama-server", "--api-key", "sekret"]}}) == "sekret"
        assert s._api_key_from_attrs(
            {"Config": {"Entrypoint": ["llama-server", "--api-key=abc"]}}) == "abc"
        assert s._api_key_from_attrs({"Config": {"Cmd": ["llama-server"]}}) == ""

    def test_sleep_idle_forms(self):
        def cmd(*toks):
            return {"Config": {"Cmd": list(toks)}}
        assert s._sleep_idle_seconds(cmd("--sleep-idle-seconds", "30")) == 30
        assert s._sleep_idle_seconds(cmd("--sleep-idle-seconds=-1")) == 0
        assert s._sleep_idle_seconds(cmd("--sleep-idle-seconds", "-1")) == 0
        assert s._sleep_idle_seconds(cmd("--host", "0.0.0.0")) == 0


class TestNaming:
    def test_split_quant(self):
        assert s.split_quant("Qwen3.8-27B-UD-Q4_K_M") == ("Qwen3.8-27B", "UD-Q4_K_M")
        assert s.split_quant("Model-F16") == ("Model", "F16")
        assert s.split_quant("NoQuantHere") == ("NoQuantHere", "")

    def test_variants_fold_onto_present_base(self):
        fold = s.canonical_families({"Qwen3.8-Flash-Next", "Qwen3.8-Flash-Next-GSQ-RCO",
                                     "Qwen3.8-Flash-Next-Uncensored"})
        assert all(v == "Qwen3.8-Flash-Next" for v in fold.values())

    def test_without_base_variants_separate(self):
        fold = s.canonical_families({"gemma-4-12b-it", "gemma-4-E4B-it-qat"})
        assert fold == {"gemma-4-12b-it": "gemma-4-12b-it",
                        "gemma-4-E4B-it-qat": "gemma-4-E4B-it-qat"}

    def test_shortest_prefix_wins(self):
        fold = s.canonical_families({"a", "a-b", "a-b-c"})
        assert fold["a-b-c"] == "a"


class TestSizeTier:
    def test_unknown_card(self):
        assert s.size_tier(1024 ** 3, 0) == ("unknown", "Size unknown", "")

    def test_thresholds(self):
        GiB = 1024 ** 3
        assert s.size_tier(40 * GiB, 24)[:2] == ("multi", "Needs more than one GPU")
        assert s.size_tier(15 * GiB, 24)[:2] == ("single", "Fits one GPU")
        assert s.size_tier(5 * GiB, 24)[:2] == ("share", "Shares a GPU")


class TestReadIniAliases:
    def test_sections_and_missing_file(self, tmp_path):
        p = tmp_path / "models.ini"
        p.write_text("[alpha]\nngl = 999\n\n[beta]\n")
        assert s.read_ini_aliases(p) == ["alpha", "beta"]
        assert s.read_ini_aliases(tmp_path / "gone.ini") == []
