"""The engine registry is what the old per-module constants are now read from (review items 9, 10)."""
from __future__ import annotations

from app import engine_registry as reg
from app import router_api as ra
from app import updates
from app.engine_colors import ENGINE_COLORS
from app.services import engines, speed


def test_order_and_default_unchanged():
    assert reg.NAMES == ("strata", "llama", "maya", "deepseek") == ra.ENGINES


def test_derived_constants_match_what_they_were():
    assert ra.SINGLE_MODEL == {"strata", "maya", "deepseek"}
    assert speed._METRICS_ENGINES == {"strata", "maya", "deepseek"}
    assert engines.READY_TIMEOUT_S["maya"] == 900.0
    assert engines.READY_TIMEOUT_S.get("llama", engines._READY_TIMEOUT_DEFAULT_S) == 420.0
    assert ENGINE_COLORS["strata"] == ("indigo", "99,102,241")
    assert [e["engine"] for e in updates.ENGINES] == ["strata", "maya", "llama"]
    assert updates.ENGINES[0]["runbook"] == "~/ai-lab/strata/UPGRADING.md"


def test_serving_is_one_rule():
    assert reg.serving("strata", 200, {"loaded": True})
    assert not reg.serving("strata", 200, {"loaded": False})          # parked or starting
    assert reg.serving("maya", 200, {"status": "ok"})
    assert not reg.serving("maya", 200, {"status": "reloading"})
    assert reg.serving("llama", 200, None)
    assert not reg.serving("llama", 503, None)                         # "Loading model"
    assert not reg.serving("strata", 200, "not json")


def test_get_is_forgiving():
    assert reg.get(" Strata ").label == "Strata"
    assert reg.get(None) is None and reg.get("nope") is None


class TestDeepSeek:
    """DeepSeek-V4.1-Flash runs on Strata 0.2.7's server in its own container: the Strata wire
    protocol, none of the Qwen install's machinery (2026-10-29)."""

    def test_it_speaks_strata_and_maya_does_not(self):
        assert reg.STRATA_PROTOCOL == ("strata", "deepseek")
        assert reg.speaks_strata("deepseek") and reg.speaks_strata(" Strata ")
        assert not reg.speaks_strata("maya") and not reg.speaks_strata("llama")
        assert not reg.speaks_strata(None)

    def test_it_is_serving_on_strata_s_rule(self):
        assert reg.serving("deepseek", 200, {"loaded": True})
        assert not reg.serving("deepseek", 200, {"loaded": False})

    def test_a_cold_load_gets_the_long_timeout(self):
        # ~150 GB of experts into the cards and RAM: Strata's 420 s would flag a healthy start.
        assert engines.READY_TIMEOUT_S["deepseek"] == 900.0

    def test_it_has_no_update_card_of_its_own(self):
        # The Strata card already tracks upstream's tags; a second card for the same repo
        # would say the same thing twice.
        assert "deepseek" not in [e["engine"] for e in updates.ENGINES]

    def test_it_parks_like_strata(self):
        assert reg.get("deepseek").parks and reg.get("deepseek").strata_metrics

    def test_the_lone_request_rule_applies(self):
        assert speed._outside_slots("deepseek", {"newest": {"id": 1}})
        assert not speed._outside_slots("maya", {"newest": {"id": 1}})
