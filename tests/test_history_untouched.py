"""strip_narration_blocks changes only messages that carry the router's own lines.

It used to flatten EVERY assistant turn with list content into one string, narration or not. That
changed the prompt the engine rendered (its prefix cache missed from that turn on), and on an
Anthropic-shaped turn it threw away the tool_use blocks and turned thinking into answer text.
"""
from __future__ import annotations

import copy

import pytest

from app import narrate
from app import router_api as ra

SWITCH = "Switching to maya: strata is parking (12 GB still held)."


@pytest.fixture(autouse=True)
def _said(monkeypatch):
    monkeypatch.setattr(narrate, "_SAID", {})
    narrate.note(SWITCH)


def _assistant(content, **extra):
    return {"messages": [{"role": "user", "content": "hi"},
                         {"role": "assistant", "content": content, **extra},
                         {"role": "user", "content": "next"}]}


def _unchanged(body):
    before = copy.deepcopy(body)
    assert ra.strip_narration_blocks(body) == 0
    assert body == before


def test_an_openai_text_part_list_is_left_as_it_arrived():
    _unchanged(_assistant([{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]))


def test_an_anthropic_turn_keeps_its_thinking_and_its_tool_call():
    _unchanged(_assistant([
        {"type": "thinking", "thinking": "I should call the tool.", "signature": "sig"},
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Oslo"}}]))


def test_a_reasoning_block_with_no_narration_in_it_is_left_alone():
    _unchanged(_assistant([{"type": "reasoning", "text": "The user wants a list."},
                           {"type": "text", "text": "Here it is."}]))


def test_a_plain_string_turn_with_reasoning_is_left_alone():
    _unchanged(_assistant("answer", reasoning_content="First I check the file."))


def test_narration_comes_out_and_the_thinking_beside_it_becomes_text():
    # The strict-template case this function exists for: Open WebUI replays the router's status
    # inside a reasoning block, and Devstral's template rejects any non-text chunk.
    body = _assistant([{"type": "reasoning", "text": SWITCH.replace("12", "31") + "\nI'll check the file."},
                       {"type": "text", "text": "Done."}])
    assert ra.strip_narration_blocks(body) == 1
    assert body["messages"][1]["content"] == "I'll check the file.\nDone."


def test_a_tool_block_survives_in_a_message_that_carried_narration():
    tool = {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Oslo"}}
    body = _assistant([{"type": "reasoning", "text": SWITCH}, {"type": "text", "text": "Checking."}, tool])
    assert ra.strip_narration_blocks(body) == 1
    assert body["messages"][1]["content"] == [{"type": "text", "text": "Checking."}, tool]


def test_a_turn_that_was_only_narration_keeps_a_single_space():
    body = _assistant([{"type": "reasoning", "text": SWITCH}])
    assert ra.strip_narration_blocks(body) == 1
    assert body["messages"][1]["content"] == " "


def test_nothing_is_touched_before_the_router_has_said_anything(monkeypatch):
    monkeypatch.setattr(narrate, "_SAID", {})
    _unchanged(_assistant([{"type": "reasoning", "text": SWITCH}]))
