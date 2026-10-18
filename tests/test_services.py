from __future__ import annotations

import pytest

from app import services as s


def rates(**over):
    d = dict(cur_gen_tps=0.0, cur_gen_tokens=0, cur_pp_tps=0.0, cur_pp_pct=0,
             cur_pp_done=0, cur_pp_total=0, cur_prompt_tokens=0,
             last_gen_tps=0.0, last_gen_tokens=0, last_pp_tps=0.0, newest="",
             open_task=None, open_slot=None, expert_hit_pct=None,
             conv_cache=None)
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
        # _rates_from_log lives in the speed submodule and resolves container_logs there,
        # so the patch target is the submodule, not the package facade.
        from app.services import speed

        def _set(text, ok=True):
            monkeypatch.setattr(
                speed, "container_logs",
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


class TestSoloContextInUse:
    """The headline context readout must follow the solo path, not just the slots.

    A solo conversation lives in no slot, so summing only the slots left the hero's
    "context in use" flat at zero for the whole of a solo generation.
    """
    def _speed(self, monkeypatch, r_over, raw_slots, engine="strata", seed=None):
        import asyncio
        import time as _time
        from app.services import speed as sp
        for store in (sp._speed_cache, sp._rates_cache, sp._last_slots,
                      sp._last_solo, sp._settled_task,
                      sp._tps_history, sp._ctx_history):
            store.clear()
        r = rates(**{"open_task": "t1", "newest": "gen", "cur_gen_tps": 40.0,
                     "cur_gen_tokens": 50, **r_over})
        sp._rates_cache["c"] = (_time.time(), r)
        monkeypatch.setattr(sp, "engine_for", lambda name: engine)

        async def fake_slots(name, port, model):
            return raw_slots
        monkeypatch.setattr(sp, "_slot_states", fake_slots)
        if seed is not None:
            seed(sp)
        return sp, asyncio.run(sp.inference_speed("c", 8080, "m1"))

    def test_solo_generation_counts_toward_pool_context(self, monkeypatch):
        raw = [{"id": i, "n_ctx": 4096, "is_processing": False,
                "n_prompt_tokens": 0} for i in range(4)]
        sp, out = self._speed(monkeypatch, {"cur_prompt_tokens": 1000}, raw)
        assert out.state == "generating"
        assert out.cache_known is False  # Strata cannot answer the cache question live
        assert out.solo is not None and out.solo.ctx_used == 1050
        assert out.ctx_used == 1050
        # Strata slots share one pool: the denominator is the pool, not per-slot x count.
        assert out.ctx_total == 4096
        assert sp._ctx_history["c"][-1][1] == 1050.0

    def test_solo_generation_excludes_idle_slot_context(self, monkeypatch):
        # Live shape measured on the box: two idle slots parked with 139,632 and
        # 30,415 tokens while a solo request of 83,278+148 generates. The headline
        # must follow the solo conversation, not the parked KV.
        raw = [{"id": 0, "n_ctx": 262144, "is_processing": False,
                "n_prompt_tokens": 139632},
               {"id": 1, "n_ctx": 262144, "is_processing": False,
                "n_prompt_tokens": 0},
               {"id": 2, "n_ctx": 262144, "is_processing": False,
                "n_prompt_tokens": 30415},
               {"id": 3, "n_ctx": 262144, "is_processing": False,
                "n_prompt_tokens": 0}]
        sp, out = self._speed(monkeypatch, {"cur_prompt_tokens": 83278,
                                            "cur_gen_tokens": 148}, raw)
        assert out.state == "generating"
        assert out.solo is not None and out.solo.ctx_used == 83426
        assert out.ctx_used == 83426

    def test_batch_generation_excludes_parked_slot_context(self, monkeypatch):
        # Same rule while slots generate: the headline is the working conversations.
        # Slot 0 generating 83,278 while slot 2 parks 30,415 of someone else's chat.
        raw = [{"id": 0, "n_ctx": 262144, "is_processing": True,
                "n_prompt_tokens": 83278},
               {"id": 1, "n_ctx": 262144, "is_processing": False,
                "n_prompt_tokens": 0},
               {"id": 2, "n_ctx": 262144, "is_processing": False,
                "n_prompt_tokens": 30415},
               {"id": 3, "n_ctx": 262144, "is_processing": False,
                "n_prompt_tokens": 0}]
        sp, out = self._speed(monkeypatch, {}, raw)
        assert out.state == "generating"
        assert out.ctx_used == 83278
        assert out.prompt_tokens == 83278  # the working slot's prompt, not the parked

    def test_idle_headline_still_shows_resident_slot_sum(self, monkeypatch):
        # Nothing working: the remembered figures are the honest "how full the
        # window has been" answer, parked conversations and all.
        raw = [{"id": 0, "n_ctx": 262144, "is_processing": False,
                "n_prompt_tokens": 139632},
               {"id": 1, "n_ctx": 262144, "is_processing": False,
                "n_prompt_tokens": 0},
               {"id": 2, "n_ctx": 262144, "is_processing": False,
                "n_prompt_tokens": 30415},
               {"id": 3, "n_ctx": 262144, "is_processing": False,
                "n_prompt_tokens": 0}]
        sp, out = self._speed(monkeypatch, {"open_task": None, "newest": ""}, raw)
        assert out.state == "idle"
        assert out.ctx_used == 139632 + 30415

    def test_idle_solo_row_is_history_not_pool_context(self, monkeypatch):
        import time as _time
        raw = [{"id": i, "n_ctx": 4096, "is_processing": False,
                "n_prompt_tokens": 0} for i in range(4)]

        def seed(sp):
            sp._last_solo["c/m1"] = (_time.time(), 40.0, 1000, 50)
        sp, out = self._speed(
            monkeypatch,
            {"open_task": None, "newest": "", "last_gen_tps": 40.0,
             "last_gen_tokens": 50}, raw, seed=seed)
        assert out.state == "idle"
        assert out.solo is not None and out.solo.ctx_used == 1050
        assert out.ctx_used == 0  # the dimmed row's 1050 is history, not resident KV

    def test_llama_pool_context_unchanged(self, monkeypatch):
        import asyncio
        import time as _time
        from app.services import speed as sp
        for store in (sp._speed_cache, sp._rates_cache, sp._last_slots,
                      sp._last_solo, sp._settled_task,
                      sp._tps_history, sp._ctx_history):
            store.clear()
        sp._rates_cache["c"] = (_time.time(), rates())
        monkeypatch.setattr(sp, "engine_for", lambda name: "llama")

        async def fake_props(name, port, model):
            return {"n_ctx": 8192, "total_slots": 2}
        monkeypatch.setattr(sp, "_props_state", fake_props)
        out = asyncio.run(sp.inference_speed("c", 8080, "m1"))
        assert out.solo is None
        assert out.cache_known is True  # llama reports n_prompt_tokens_cache
        # llama's n_ctx IS a per-slot share: the pool total stays share x count.
        assert out.ctx_used == 0 and out.ctx_total == 16384


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


class TestConversationCache:
    """Strata's host-RAM conversation cache must reach the hero tile intact.

    /metrics reports it as a section; the rates reader normalizes it and
    inference_speed passes it through untouched. Absent or disabled means None,
    which means no tile - never a zero that reads as an empty cache.
    """
    CC = {"parked": 2, "slots": 8, "bytes": 8194164960, "parks": 339,
          "restores": 99, "evictions": 100, "requests": 500,
          "requests_reused": 477}

    def _metrics(self, cc):
        return {"live": {"state": "idle"}, "requests": [],
                "totals": {"requests": 500}, "conversation_cache": cc}

    def test_enabled_section_is_normalized(self, monkeypatch):
        from app.services import speed as sp
        cc = {"enabled": True, **self.CC}
        monkeypatch.setattr(sp, "strata_metrics", lambda n: self._metrics(cc))
        assert sp._rates_from_log_strata("strata")["conv_cache"] == self.CC

    def test_absent_or_disabled_is_none(self, monkeypatch):
        from app.services import speed as sp
        monkeypatch.setattr(sp, "strata_metrics",
                            lambda n: self._metrics({"enabled": False}))
        assert sp._rates_from_log_strata("strata")["conv_cache"] is None
        monkeypatch.setattr(sp, "strata_metrics",
                            lambda n: {"live": {}, "requests": [], "totals": {}})
        assert sp._rates_from_log_strata("strata")["conv_cache"] is None

    def test_flows_through_to_the_speed_object(self, monkeypatch):
        helper = TestSoloContextInUse()
        raw = [{"id": i, "n_ctx": 262144, "is_processing": False,
                "n_prompt_tokens": 0} for i in range(4)]
        sp, out = helper._speed(monkeypatch, {"conv_cache": dict(TestConversationCache.CC)},
                                raw)
        assert out.conv_cache == TestConversationCache.CC

    def test_llama_speed_carries_none(self, monkeypatch):
        helper = TestSoloContextInUse()
        raw = [{"id": 0, "n_ctx": 8192, "is_processing": False,
                "n_prompt_tokens": 0}]
        sp, out = helper._speed(monkeypatch, {}, raw, engine="llama")
        assert out.conv_cache is None


class TestFitChips:
    """vram_fit_chips verdicts, especially the strata streaming case: an MoE over VRAM
    is not oom on strata — the box ran a 78.1 GiB IQ3_S Flash-Next at ~120 tok/s while
    the size-only chip said otherwise.
    """
    VRAM = 63.7
    RAM = 120.0
    MOE = {"model": {"expert_count": 128, "expert_used_count": 8}}
    DENSE = {"model": {"expert_count": None}}

    def _stub(self, monkeypatch, ram=None, arena=0.0):
        from app import hw
        from app.services import stratactl as sc
        monkeypatch.setattr(sc, "_fit_backends",
                            lambda: {"strata": self.VRAM, "llama-9700x2": self.VRAM})
        monkeypatch.setattr(sc, "_cpu_backends", lambda: [])
        monkeypatch.setattr(sc, "engine_for",
                            lambda n: "strata" if n == "strata" else "llama")
        monkeypatch.setattr(hw, "usable_ram_gb", lambda: self.RAM if ram is None else ram)
        monkeypatch.setattr(hw, "strata_arena_gb", lambda: arena)
        return sc

    def _chip(self, chips, name):
        return next(c for c in chips if c["name"] == name)

    def test_moe_over_vram_streams_on_strata_and_stays_oom_on_llama(self, monkeypatch):
        sc = self._stub(monkeypatch)
        chips = sc.vram_fit_chips(int(78.1 * 1024 ** 3), self.MOE)
        assert self._chip(chips, "strata")["verdict"] == "streams"
        assert self._chip(chips, "llama-9700x2")["verdict"] == "oom"

    def test_streams_ratio_uses_the_combined_pool(self, monkeypatch):
        sc = self._stub(monkeypatch)
        c = self._chip(sc.vram_fit_chips(int(78.1 * 1024 ** 3), self.MOE), "strata")
        assert c["ratio_pct"] == round(78.1 / (self.VRAM + self.RAM) * 100)
        assert c["host_ram_gb"] == self.RAM

    def test_moe_that_fits_the_cards_is_still_fits(self, monkeypatch):
        sc = self._stub(monkeypatch)
        chips = sc.vram_fit_chips(int(30 * 1024 ** 3), self.MOE)
        assert self._chip(chips, "strata")["verdict"] == "fits"

    def test_dense_over_vram_is_still_oom_on_strata(self, monkeypatch):
        # Nothing to stream: dense weights have nowhere to go but the cards.
        sc = self._stub(monkeypatch)
        chips = sc.vram_fit_chips(int(78.1 * 1024 ** 3), self.DENSE)
        assert self._chip(chips, "strata")["verdict"] == "oom"

    def test_no_header_falls_back_to_size_only(self, monkeypatch):
        sc = self._stub(monkeypatch)
        chips = sc.vram_fit_chips(int(78.1 * 1024 ** 3))
        assert self._chip(chips, "strata")["verdict"] == "oom"

    def test_over_the_combined_pool_is_still_impossible(self, monkeypatch):
        sc = self._stub(monkeypatch)
        chips = sc.vram_fit_chips(int(200 * 1024 ** 3), self.MOE)
        assert chips[0]["verdict"] == "impossible"

    def test_strata_arena_counts_for_strata_only(self, monkeypatch):
        # Strata up with a big arena makes usable_ram_gb tiny - but switching strata to
        # another model reclaims its own arena, so an MoE may count on it. llama may not.
        sc = self._stub(monkeypatch, ram=6.4, arena=87.2)
        chips = sc.vram_fit_chips(int(78.1 * 1024 ** 3), self.MOE)
        c = self._chip(chips, "strata")
        assert c["verdict"] == "streams"
        assert c["host_ram_gb"] == 93.6
        assert c["ratio_pct"] == round(78.1 / (self.VRAM + 93.6) * 100)
        assert self._chip(chips, "llama-9700x2")["verdict"] == "oom"

    def test_dense_cannot_reclaim_the_arena(self, monkeypatch):
        # Same numbers, dense model: no streaming anywhere, and the machine-level
        # ceiling stays VRAM + usable (arena excluded).
        sc = self._stub(monkeypatch, ram=6.4, arena=87.2)
        chips = sc.vram_fit_chips(int(78.1 * 1024 ** 3), self.DENSE)
        assert chips[0]["verdict"] == "impossible"
