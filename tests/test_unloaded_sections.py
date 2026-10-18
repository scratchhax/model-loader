"""Tests for surfacing ini sections no running llama backend has loaded.

llama-server reads --models-preset only at startup, so a section added to models.ini after
a backend started is real but unserved until a restart. The Router dropdown lists those
"(not loaded)" instead of hiding them, and the Containers card says which running backend
predates the last ini edit.
"""
from __future__ import annotations

import time

from app import router_api
from app.routers.containers import ini_stale_since


class TestUnloadedSections:
    def test_missing_sections_surface_sorted(self, monkeypatch):
        monkeypatch.setattr(router_api.ini, "section_names",
                            lambda: ["b-model", "a-model", "served"])
        assert router_api.unloaded_sections(["served"]) == ["a-model", "b-model"]

    def test_live_id_not_in_ini_is_ignored(self, monkeypatch):
        monkeypatch.setattr(router_api.ini, "section_names", lambda: ["x"])
        assert router_api.unloaded_sections(["x", "ghost"]) == []

    def test_nothing_served_lists_every_section(self, monkeypatch):
        monkeypatch.setattr(router_api.ini, "section_names", lambda: ["b", "a"])
        assert router_api.unloaded_sections([]) == ["a", "b"]

    def test_empty_ini_lists_nothing(self, monkeypatch):
        monkeypatch.setattr(router_api.ini, "section_names", lambda: [])
        assert router_api.unloaded_sections(["whatever"]) == []


class TestIniStaleSince:
    def test_started_before_ini_edit_is_stale(self):
        assert ini_stale_since("2026-10-14T20:23:03.327353374Z", time.time()) is True

    def test_started_after_ini_edit_is_fresh(self):
        assert ini_stale_since("2026-10-18T20:00:00.000000000Z", 1e9) is False

    def test_no_nanoseconds_parses(self):
        assert ini_stale_since("2026-10-14T20:23:03Z", time.time()) is True

    def test_unknowable_inputs_answer_false(self):
        now = time.time()
        assert ini_stale_since("", now) is False
        assert ini_stale_since("0001-01-01T00:00:00Z", now) is False
        assert ini_stale_since("not-a-date", now) is False
        assert ini_stale_since("2026-10-14T20:23:03Z", 0.0) is False
