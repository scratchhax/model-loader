"""Strata: the one backend here that is not llama.cpp.

Strata (https://github.com/Niko1221/Strata) serves exactly one model family,
Qwen3.8-Flash-Next, out of a prepared expert pack. The experts live in host RAM and the hot
ones are streamed onto a single card, so none of this app's llama.cpp machinery describes it:
not models.ini, not autoconfig's weights/KV/compute budget, not the argv vocabulary, not
--models-max eviction. `services.engine_for()` is how the rest of the app asks, and
`_backend_list()` drops it so autoconfig never tries to size it.

What IS ours is the one setting that decides whether it can co-exist with everything else on
this box: which card it owns. Upstream takes that as `--gpu N`, which its container entry
point reads from a GPU environment variable, so the pin is a line in an env file that this
module owns the way `ini` owns models.ini - single writer, never hand-edited.

Why an env file and not the container's environment directly: compose bakes `env_file` into a
container at CREATE time, so changing it needs a recreate and a plain restart silently keeps
the old card. The compose service therefore ALSO re-sources the same file from its entry point
on every start, which is what makes a restart enough. Both halves read this one file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# The engine name a container declares in its `ai-lab.engine` label.
ENGINE = "strata"

# Where the pin file is mounted inside model-loader. The strata container sees the same
# directory read-only at /strata-pin; this app is the only writer.
ENV_PATH = Path(os.environ.get("STRATA_ENV_PATH") or "/strata/strata.env")

# A pin line. Strata takes ONE card as --gpu N (GPU=) and several as --gpus N,M (GPUS=), and
# the two are separate switches upstream, so the file carries whichever applies and not both.
# Anchored so a GPU= inside a comment is not mistaken for the setting - the file is mostly
# comments explaining itself.
_PIN_RE = re.compile(r"^(?:export\s+)?GPU\s*=\s*(\d+)\s*$")
_PINS_RE = re.compile(r"^(?:export\s+)?GPUS\s*=\s*(\d+(?:\s*,\s*\d+)*)\s*$")


def read_pin(path: Path | None = None) -> str:
    """The card selection the file names: "1", or "1,0" for a split, or "" when it says nothing.

    A STRING, not an int, because "both cards" is a real answer here and a second card is not
    an index. "" is also a real answer, not an error: with neither set, Strata's setup picks
    the card with the most VRAM by itself, and the UI has to be able to say so rather than
    inventing a zero.

    GPUS wins when both are somehow present, matching the entry point, which passes --gpus
    after --gpu and lets the later flag stand.
    """
    p = path or ENV_PATH
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return ""
    single = ""
    for line in text.splitlines():
        line = line.strip()
        m = _PINS_RE.match(line)
        if m:
            return ",".join(part.strip() for part in m.group(1).split(","))
        m = _PIN_RE.match(line)
        if m:
            single = m.group(1)
    return single


def pin_in_container(attrs: dict | None) -> str:
    """The selection the RUNNING container was created with, from its own environment.

    Read separately from the file on purpose. The file is what the next start will use; this
    is what the process is actually on. They differ exactly between a pin change and the
    restart that applies it, and that gap is the one thing a user needs told about.
    """
    try:
        env = ((attrs or {}).get("Config") or {}).get("Env") or []
    except AttributeError:
        return ""
    single = ""
    for entry in env:
        if not isinstance(entry, str):
            continue
        key, _, val = entry.partition("=")
        key, val = key.strip(), val.strip()
        if key == "GPUS" and val:
            return ",".join(part.strip() for part in val.split(","))
        if key == "GPU" and val.isdigit():
            single = val
    return single


def set_pin(selection: str, path: Path | None = None) -> tuple[bool, str]:
    """Rewrite the pin file to name `selection` ("1" or "1,0"). Returns (ok, message).

    Rewrites the existing line in place and leaves every comment untouched, because those
    comments are the file's record of why it is not HIP_VISIBLE_DEVICES - exactly the thing
    someone reading it later needs. An append-only writer would leave two GPU= lines and `sh`
    sourcing the file would take the last; a truncating writer would throw the explanation away.

    Exactly one of GPU / GPUS survives. Leaving a stale GPU= beside a new GPUS= would mean the
    file disagrees with itself, and which one won would depend on the order the entry point
    happens to pass them.
    """
    cards = [c.strip() for c in str(selection).split(",") if c.strip()]
    if not cards or not all(c.isdigit() for c in cards):
        return False, f"not a card selection: {selection!r}"
    if len(set(cards)) != len(cards):
        return False, "the same card twice"
    p = path or ENV_PATH
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False, f"{p} is missing — is ./strata mounted into this container?"
    except OSError as e:
        return False, f"cannot read {p}: {e}"

    want = f"GPU={cards[0]}" if len(cards) == 1 else "GPUS=" + ",".join(cards)
    out: list[str] = []
    placed = False
    for line in text.splitlines():
        if _PIN_RE.match(line.strip()) or _PINS_RE.match(line.strip()):
            if not placed:          # the first pin line becomes the new one; any other is dropped
                out.append(want)
                placed = True
            continue
        out.append(line)
    if not placed:
        out.append(want)

    try:
        p.write_text("\n".join(out) + "\n", encoding="utf-8")
    except OSError as e:
        return False, f"cannot write {p}: {e}"
    if len(cards) == 1:
        return True, f"pinned to card {cards[0]}"
    return True, (f"split across cards {', '.join(cards)} — card {cards[0]} is the main one. "
                  "Upstream measured a split LOSING on this pair (4K prompts 1,776 -> 1,244 "
                  "tok/s, decode ~60 -> ~51) because one 32 GB R9700 already holds every "
                  "expert; worth measuring here, not assuming.")


def card_count(exec_candidates=()) -> int:
    """How many GPUs there are to choose between, as Strata numbers them.

    Strata enumerates AMD cards from /sys/class/kfd/kfd/topology/nodes in node order, skipping
    CPU nodes - the same order HIP uses, and the same order gpu_procs already derives for its
    per-process VRAM attribution. So the count is shared rather than probed again.

    0 means "we could not tell", which the caller must treat as unknown: the topology nodes are
    EPERM to a container without /dev/kfd, and this app has none, so the figure comes from
    exec'ing into a backend that does. Offering a guessed single card would quietly make the
    second one unselectable.
    """
    from . import gpu_procs
    try:
        return len(gpu_procs.card_index_by_gpu_id(exec_candidates))
    except Exception:  # noqa: BLE001 - an unreadable topology is "unknown", never an error page
        return 0
