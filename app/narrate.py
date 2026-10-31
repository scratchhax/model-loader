"""Status narration for a streamed request that has to wait before a backend exists.

An engine swap on this box is not quick: a soft park gives the VRAM back in ~0.3 s, but the
driver takes a moment to actually release it, a llama router then loads a model in 10-40 s, and
a cold Strata start reads ~55 GB of experts before its port opens, measured at 90 s. A client
that asked for an engine which is not currently holding the cards therefore faces a wait that
no amount of tuning removes.

Done invisibly, that wait is indistinguishable from a hang. The client shows an empty bubble and
a spinner, the user presses stop, and the switch they triggered completes into an abandoned
request. So the wait is narrated instead: the router opens the stream immediately and describes
what the switch is doing, in the switch's own words, until a backend exists to relay.

Three consequences worth knowing before reading the code.

1. Narrating COMMITS to HTTP 200. Once the first chunk is out there is no status code left to
   set, so a failed switch has to arrive as text inside the stream. That is better where the
   user is looking and worse for anything reading the status column, which is what `result`
   below is for: the caller records what really happened rather than the 200 it already sent.

2. The lines are not invented. Every one comes from the switch's own phase and message, so
   "31.4 GB still held" is a real reading rather than a bar pretending to know a proportion.
   Inventing one would be the only dishonest thing a progress display can do, and this avoids
   having one by never computing one.

3. OpenAI shape only. The Anthropic stream is a different event grammar (`message_start`,
   `content_block_delta`), and a half-right imitation of it would break clients more
   interestingly than a plain refusal does. Anthropic-shaped callers keep the old 503.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import AsyncIterator, Awaitable, Callable

# How often the progress source is asked. The switch updates its own message about this often
# (the drain loop every 1 s, the readiness poll every 2 s), so a faster poll would mostly
# re-read the same string.
POLL_S = 1.0

# If the line has not changed for this long, emit an SSE comment. Comments are part of the
# protocol and carry no content, so this keeps the socket warm through a quiet phase without
# putting anything on screen. Open WebUI's AIOHTTP_CLIENT_TIMEOUT is unset here, which means its
# built-in default applies and we do not get to know what it is; not relying on it is cheaper
# than finding out at second 90 of a cold start.
KEEPALIVE_AFTER_S = 5.0

# How often a phase that is still running repeats itself with a fresher number. The switch
# polls every 2 s; echoing that is a wall of text, and saying nothing for two minutes looks
# stalled. This is the gap between those two failures.
PROGRESS_EVERY_S = 15.0

# Digits masked out, so "up 42s" and "up 44s" are recognised as the same line.
_DIGITS = re.compile(r"[0-9]+")

# Which delta field the status goes in.
#   "reasoning" -> reasoning_content, which renders as a collapsible thinking block. This stack
#                  already emits that field (Strata returns reasoning_content), so clients here
#                  understand it, and the status stays out of the answer body.
#   "content"   -> plain content. Always renders, and is saved as part of the assistant message,
#                  so it comes back as prompt on the next turn.
CHANNEL_REASONING = "reasoning"
CHANNEL_CONTENT = "content"

_FIELD = {CHANNEL_REASONING: "reasoning_content", CHANNEL_CONTENT: "content"}

# ------------------------------------------------------- what this router has already said
#
# Narration is the only thing this module puts into a transcript, and once it is in one it comes
# back forever: Open WebUI stores the assistant turn and replays it on every later request,
# including requests routed to a model whose chat template rejects non-text chunks in assistant
# content. So the router records the SHAPE of every line it emits - the same digit-masked form the
# dedup below uses - and the request path recognises its own words arriving back and removes
# exactly those.
#
# Shape matching rather than templates, because the loading phase passes through whatever the
# switch happens to be saying and no fixed pattern would catch it. Shape matching needs no
# vocabulary: it is exact by construction, since the line being recognised is the line that was
# sent. Digits are masked so "up 42s" replayed as "up 42s" matches, and a re-phrased retry of the
# same phase matches too.
#
# In memory here; the caller persists it. This module is imported by tests that should not be
# writing to disk.

_SAID: dict[str, None] = {}
_SAID_MAX = 400


def shape_of(text: str) -> str:
    """The identity of a line: whitespace-trimmed, digits masked."""
    return _DIGITS.sub("#", text.strip()) if text else ""


def note(text: str) -> None:
    """Record one emitted line so its shape is recognised on the way back in."""
    shape = shape_of(text)
    if not shape:
        return
    _SAID[shape] = None
    while len(_SAID) > _SAID_MAX:
        _SAID.pop(next(iter(_SAID)))      # oldest first; this dict is insertion-ordered


def said() -> frozenset[str]:
    return frozenset(_SAID)


def remember(shapes) -> None:
    """Seed from a previous run. Shapes the router has already said in an earlier process are
    still in people's transcripts, so forgetting them would re-break those conversations."""
    for shape in shapes:
        if shape:
            _SAID[shape] = None


def _sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


def _frame(cid: str, created: int, model: str, delta: dict, finish: str | None = None) -> bytes:
    return _sse({
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })


def opening(cid: str, created: int, model: str) -> bytes:
    """The role chunk, sent once before any text.

    The relayed stream will send its own role chunk afterwards. That duplication is harmless -
    deltas accumulate, and `{"role": "assistant"}` twice says the same thing twice - and it is
    why narration sends no terminating chunk on the success path: the upstream stream owns the
    end of the response, including its `[DONE]`.
    """
    return _frame(cid, created, model, {"role": "assistant"})


def line(cid: str, created: int, model: str, text: str, channel: str) -> bytes:
    return _frame(cid, created, model, {_FIELD[channel]: text})


def closing(cid: str, created: int, model: str) -> bytes:
    """End the stream ourselves. Only on the failure path, where no upstream stream follows."""
    return _frame(cid, created, model, {}, finish="stop") + b"data: [DONE]\n\n"


def keepalive() -> bytes:
    """An SSE comment: protocol-legal, ignored by clients, invisible on screen."""
    return b": waiting\n\n"


def phase_line(state: dict) -> str:
    """One human line for a switch state dict, or "" when there is nothing to say.

    `state` is services.engine_switch_state(): {engine, phase, msg, stopped, unloaded, started}.
    The wording is deliberately concrete about what is holding things up, because the two slow
    phases have different causes and different fixes - "draining" means the driver has not
    released the VRAM yet, "loading" means it has and the model is being read - and a single
    "please wait" would hide which one you are in.
    """
    phase = str(state.get("phase") or "")
    msg = str(state.get("msg") or "").strip()
    engine = str(state.get("engine") or "the engine")

    if phase == "unloading":
        who = ", ".join(state.get("unloaded") or ()) or "the current engine"
        return f"Parking {who} to free the cards for {engine}, once any answer in progress finishes."
    if phase == "stopping":
        giving = list(state.get("unloaded") or ()) or list(state.get("stopped") or ())
        who = ", ".join(giving) if giving else "the current engine"
        verb = "Parking" if state.get("unloaded") else "Stopping"
        return f"{verb} {who} to free the cards for {engine}."
    if phase == "draining":
        return f"Waiting for the driver to release the VRAM: {msg}." if msg else \
               "Waiting for the driver to release the VRAM."
    if phase == "starting":
        return f"Starting {engine}."
    if phase == "loading":
        return (msg[:1].upper() + msg[1:] + ".") if msg else f"Loading {engine}."
    if phase == "done":
        started = ", ".join(state.get("started") or ()) or engine
        return f"Ready, routing to {started}."
    if phase == "slow":
        return f"Still coming up: {msg}" if msg else "Still coming up."
    if phase == "failed":
        return f"The switch to {engine} failed: {msg}" if msg else \
               f"The switch to {engine} failed."
    return ""


async def narrate_wait(
    progress: Callable[[], Awaitable[tuple[str, str]]],
    result: dict,
    *,
    cid: str,
    model: str,
    created: int | None = None,
    channel: str = CHANNEL_REASONING,
    poll_s: float = POLL_S,
    keepalive_after_s: float = KEEPALIVE_AFTER_S,
    progress_every_s: float = PROGRESS_EVERY_S,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AsyncIterator[bytes]:
    """Stream status until `progress()` reports the wait is over.

    `progress()` is awaited repeatedly and returns (state, text):

        ("waiting", line)   narrate `line` when it differs from the one already shown
        ("ready",   note)   stop; the caller relays the real stream next
        ("failed",  why)    say `why` in the stream and terminate it

    `result` is filled in with {"state", "text", "waited_s", "lines"}, because an async generator
    cannot hand a value back to whoever iterates it and the caller needs the real outcome: it has
    already sent 200 and must still record a failure as a failure.

    `clock` and `sleep` are injected so the loop is testable without real time passing. A test
    that had to sleep through a 90 s cold start would not be run.
    """
    created = created if created is not None else int(clock())
    t0 = clock()
    shown_shape = ""
    any_shown = False
    last_line = t0
    last_emit = t0
    count = 0

    yield opening(cid, created, model)

    while True:
        state, text = await progress()
        if state != "waiting":
            result.update(state=state, text=text, waited_s=clock() - t0, lines=count)
            if state == "failed":
                # In-band, because the status code is long gone. Phrased as the end of the
                # narration rather than as a protocol error: the client is mid-stream, and a
                # bare {"error": ...} frame there is not something every client reads.
                note(text)
                yield line(cid, created, model, f"\n{text}\n", channel)
                yield closing(cid, created, model)
            return

        # Dedup on the line's SHAPE, not its text. The switch builds its message with the
        # elapsed seconds in it ("Strata up 42s, model not loaded yet"), so the text changes
        # every single poll and exact-match dedup never fires: a 141 s Strata load produced
        # seventy near-identical lines in a real chat, which is worse than silence because it
        # buries the two lines that actually said something. Comparing with the digits masked
        # means a phase speaks once, and then only often enough to show it is still moving.
        shape = _DIGITS.sub("#", text) if text else ""
        new_phase = bool(text) and shape != shown_shape
        stale = bool(text) and (clock() - last_line) >= progress_every_s

        if new_phase or stale:
            note(text)
            yield line(cid, created, model, ("\n" if any_shown else "") + text, channel)
            shown_shape, any_shown = shape, True
            last_line = last_emit = clock()
            count += 1
        elif (clock() - last_emit) >= keepalive_after_s:
            yield keepalive()
            last_emit = clock()

        await sleep(poll_s)
