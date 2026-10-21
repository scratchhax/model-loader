"""The engine registry is what the old per-module constants are now read from (review items 9, 10)."""
from __future__ import annotations

from app import engine_registry as reg
from app import router_api as ra
from app import updates
from app.engine_colors import ENGINE_COLORS
from app.services import engines, speed


def test_order_and_default_unchanged():
    assert reg.NAMES == ("strata", "llama", "maya") == ra.ENGINES


def test_derived_constants_match_what_they_were():
    assert ra.SINGLE_MODEL == {"strata", "maya"}
    assert speed._METRICS_ENGINES == {"strata", "maya"}
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
