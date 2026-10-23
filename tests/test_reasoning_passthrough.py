"""The router passes a model's earlier reasoning through; only its own narration is removed.

It used to delete `reasoning_content` from every assistant turn on every engine (a workaround for
one llama template). GLM's template replays that reasoning by default (`clear_thinking` false), so
Maya lost its own thinking in every agent loop and saw each earlier turn as thinking-off. Whether a
model sees its earlier reasoning is now the engine's own option: llama's `reasoning-preserve`,
Strata's and Maya's templates plus `chat_template_kwargs`.
"""
from __future__ import annotations

import pytest

from app import narrate
from app import router_api as ra


@pytest.fixture(autouse=True)
def _said(monkeypatch):
    monkeypatch.setattr(narrate, "_SAID", {})
    narrate.note("Switching to maya: strata is parking (12 GB still held).")


def _body(reasoning):
    return {"messages": [{"role": "user", "content": "hi"},
                         {"role": "assistant", "content": "answer", "reasoning_content": reasoning},
                         {"role": "user", "content": "next"}]}


def test_the_models_reasoning_is_left_alone():
    body = _body("First I check the file.\nThen I call the tool.")
    assert ra.strip_narration_blocks(body) == 0
    assert body["messages"][1]["reasoning_content"] == "First I check the file.\nThen I call the tool."


def test_the_routers_own_lines_come_out_of_it():
    body = _body("Switching to maya: strata is parking (31 GB still held).\nThe user wants a list.")
    assert ra.strip_narration_blocks(body) == 1
    assert body["messages"][1]["reasoning_content"] == "The user wants a list."


def test_reasoning_that_was_only_narration_is_removed():
    body = _body("Switching to maya: strata is parking (4 GB still held).")
    ra.strip_narration_blocks(body)
    assert "reasoning_content" not in body["messages"][1]


def test_the_strip_helpers_are_gone():
    assert not hasattr(ra, "drop_replayed_reasoning") and not hasattr(ra, "wants_reasoning_replay")
