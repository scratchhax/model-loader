"""Two chips that told the truth about the past as if it were the present.

* llama-voice wore a red "load failed" for days: llama.cpp keeps a model's failed flag until
  that model is tried again, and a gemma-4-31B load misrouted there once was never retried.
* The engine row called a PARKED Strata "active" and disabled its button, because it read
  "container running" as "holding the cards".
"""
from __future__ import annotations

import json
import time

import pytest

from app import db
from app.routers import overview
from app.services import backends
from app.services.backends import LlamaBackend
from tests.test_routes_smoke import web  # noqa: F401 - the fixture: real app, empty box


@pytest.fixture
def fdb(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "f.db")
    db.init()
    monkeypatch.setattr(backends, "_failed_seen", None)
    yield


class TestFailureAge:
    def test_first_seen_then_ages(self, fdb, monkeypatch):
        t = [1000.0]
        monkeypatch.setattr(backends.time, "time", lambda: t[0])
        assert backends._note_failure("llama-voice", "gemma (exit 1)") == 0.0
        t[0] += 600
        assert backends._note_failure("llama-voice", "gemma (exit 1)") == 600.0

    def test_survives_a_restart(self, fdb, monkeypatch):
        # An in-memory clock would restart at every deploy and call a week-old failure new.
        t = [1000.0]
        monkeypatch.setattr(backends.time, "time", lambda: t[0])
        backends._note_failure("llama-voice", "gemma (exit 1)")
        monkeypatch.setattr(backends, "_failed_seen", None)        # a new process
        t[0] += 3 * 86400
        assert backends._note_failure("llama-voice", "gemma (exit 1)") == 3 * 86400

    def test_cleared_and_a_new_failure_starts_fresh(self, fdb, monkeypatch):
        t = [1000.0]
        monkeypatch.setattr(backends.time, "time", lambda: t[0])
        backends._note_failure("b", "m1 (exit 1)")
        t[0] += 500
        assert backends._note_failure("b", None) == 0.0
        assert json.loads(db.get_setting(backends._FAILED_KEY)) == {}
        assert backends._note_failure("b", "m2 (exit 1)") == 0.0

    def test_backends_are_independent(self, fdb):
        backends._note_failure("a", "x")
        backends._note_failure("b", "y")
        backends._note_failure("a", None)
        assert list(json.loads(db.get_setting(backends._FAILED_KEY))) == ["b\ty"]


class TestCurrentOrHistory:
    def _b(self, **kw):
        return LlamaBackend(name="llama-voice", found=True, status="running", **kw)

    def test_empty_backend_is_current_however_old(self):
        # At --models-max 1 a failed load leaves the router empty: that is the alarm.
        b = self._b(load_failed="gemma (exit 1)", load_failed_age=9 * 86400)
        assert b.load_failed_current

    def test_serving_something_else_makes_an_old_failure_history(self):
        b = self._b(load_failed="gemma (exit 1)", load_failed_age=3 * 86400,
                    loaded_model="wheatley-voice")
        assert not b.load_failed_current

    def test_fresh_failure_is_current_even_while_serving(self):
        b = self._b(load_failed="gemma (exit 1)", load_failed_age=60, loaded_model="wheatley-voice")
        assert b.load_failed_current

    def test_no_failure(self):
        assert not self._b(loaded_model="x").load_failed_current


class TestParkedEngine:
    def _strip(self, web, monkeypatch, parked: bool):
        from app import hw
        cards = [hw.GpuCard(index=i, name="AMD Radeon Graphics (gfx1201)", util_pct=0.0,
                            vram_used_gb=29.0, vram_total_gb=31.86, temp_c=40.0, power_w=20.0)
                 for i in (0, 1)]
        # The engine row sits inside the "on the cards" table, so one GPU process is needed.
        tenants = [{"label": "maya", "comm": "maya", "pid": 1, "foreign": False, "total_gb": 57.8,
                    "per_card": [28.9, 28.9], "working": False, "cu_card": None, "cu_pct": 0,
                    "cu_peak": 0, "cu_total": 0}]
        base = {"cards": cards, "gpu_history": {}, "card_breakdowns": [], "tenants": tenants,
                "ejectable": [], "share": {}, "engine_switch": {},
                "engines": [{"engine": "maya", "names": ["maya"], "running": True},
                            {"engine": "strata", "names": ["strata"], "running": True},
                            {"engine": "llama", "names": ["llama-9700x2"], "running": False}]}
        monkeypatch.setattr(overview, "_gpu_strip_context", lambda: dict(base))

        async def hero():
            return [LlamaBackend(name="strata", found=True, status="running",
                                 unloaded_model="qwen" if parked else None)]
        monkeypatch.setattr(overview, "_hero_backends_cached", hero)
        monkeypatch.setattr(overview.services, "engine_for", lambda n: n)
        return web.get("/gpu-strip").text

    def test_parked_strata_is_offered_not_active(self, web, monkeypatch):
        html = self._strip(web, monkeypatch, parked=True)
        assert "parked · take over" in html
        assert 'hx-post="/engines/strata/activate"' in html
        assert html.count(">active</span>") == 1                       # maya only

    def test_loaded_strata_still_reads_active(self, web, monkeypatch):
        html = self._strip(web, monkeypatch, parked=False)
        assert "parked · take over" not in html and 'hx-post="/engines/strata/activate"' not in html
        assert html.count(">active</span>") == 2                       # maya and strata
