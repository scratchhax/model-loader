"""Blocking work in async routes runs in a thread, not on the event loop (review item 22).

On the loop, one slow docker call stalled every other request the UI process serves, the
Overview's half-second poll included. Each probe records whether it ran with an event loop
running in its own thread: a worker thread has none.
"""
from __future__ import annotations

import asyncio

from app import engine_controls
from app import strata_models
from tests.test_routes_smoke import web  # noqa: F401 - the fixture: real app, empty box


def _where(seen):
    try:
        asyncio.get_running_loop()
        seen.append("loop")
    except RuntimeError:
        seen.append("thread")


def test_router_page_blocking_work_is_off_the_loop(web, monkeypatch):
    seen: list[str] = []
    real = engine_controls.strata_choices
    monkeypatch.setattr(engine_controls, "strata_choices", lambda: (_where(seen), real())[1])
    assert web.get("/router").status_code == 200
    assert seen == ["thread"]


def test_strata_page_catalog_read_is_off_the_loop(web, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(strata_models, "catalog", lambda *a, **k: (_where(seen), [])[1])
    assert web.get("/strata/models").status_code == 200
    assert seen and set(seen) == {"thread"}
