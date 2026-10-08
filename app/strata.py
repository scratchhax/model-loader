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

import json
import os
import re
import shutil
import tempfile
import time
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
# Any pin line, set or cleared - the cleared kind (GPUS= with nothing after it) matters:
# see set_pin.
_PIN_ANY_RE = re.compile(r"^(?:export\s+)?GPUS?\s*=")


def _atomic_replace(path: Path, data: str) -> None:
    """tmp + fsync + os.replace, preserving the target's mode.

    Both files this module writes are READ BY OTHER PROCESSES at start: the strata
    entry point sources strata.env, and the server loads its run config. A truncated
    write (plain write_text) leaves them a half file; mkstemp alone would fix the
    truncation but land on mode 0600, so the original mode is carried over.
    """
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        mode = 0o644
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


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


def set_pin(selection: str, path: Path | None = None) -> tuple[bool, str]:
    """Rewrite the pin file to name `selection` ("1" or "1,0"). Returns (ok, message).

    Rewrites the existing line in place and leaves every comment untouched, because those
    comments are the file's record of why it is not HIP_VISIBLE_DEVICES - exactly the thing
    someone reading it later needs. An append-only writer would leave two GPU= lines and `sh`
    sourcing the file would take the last; a truncating writer would throw the explanation away.

    Both switches are always named: the chosen one set, the other explicitly CLEARED. The
    entry point sources this file and then exports GPU and GPUS as they stand - a compose
    service that bakes `GPUS=1,0` into the container environment keeps that value through a
    restart unless the file overwrites it, and the entry point passes --gpus after --gpu, so
    the stale bake wins. (Measured on this box: a pin to one card left the engine across
    both until the file said GPUS= as well.) An empty assignment is what sourcing needs to
    see to clear the variable; deleting the line does nothing.
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

    single = len(cards) == 1
    want = f"GPU={cards[0]}" if single else "GPUS=" + ",".join(cards)
    clear = "GPUS=" if single else "GPU="
    out: list[str] = []
    placed = False
    for line in text.splitlines():
        if _PIN_ANY_RE.match(line.strip()):
            if not placed:          # the first pin line becomes the new one; any other is dropped
                out.append(want)
                placed = True
            continue
        out.append(line)
    if not placed:
        out.append(want)
    out.append(clear)   # always: sourcing must beat whatever compose baked in

    try:
        _atomic_replace(p, "\n".join(out) + "\n")
    except OSError as e:
        return False, f"cannot write {p}: {e}"
    if single:
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


# ---------------------------------------------------------------------------
# The run config: the second file this module owns.
#
# setup.py records each installed model as $STRATA_DATA/config/strata-<tag>.json and
# serve/server.py reads its server-level keys (idle_unload_s, min_free_vram_mib,
# before_load, vram_elastic) from that same file. Upstream explicitly blesses foreign
# keys in it: SETUP_KEYS names what setup writes itself, and carry_over() (#629) keeps
# everything else across a re-setup. So the coexistence knobs live in the real config,
# and this app is a second writer of exactly the keys it names - never of setup's.
#
# The entry point has no CONFIG= hook (that was the plan's assumption; the stock
# docker-entrypoint.sh always uses $STRATA_DATA/config/strata-<tag>.json), which turns
# out to be simpler: no env line, no shadow file, one config, two honest writers.
# ---------------------------------------------------------------------------

BACKUPS_TO_KEEP = 5


def _env_of(attrs: dict | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for entry in ((attrs or {}).get("Config") or {}).get("Env") or []:
        if isinstance(entry, str) and "=" in entry:
            k, _, v = entry.partition("=")
            env[k.strip()] = v.strip()
    return env


def config_tag(attrs: dict | None) -> str:
    """The config name tag the entry point derives: FAMILY=unsloth, MODEL=UD-IQ4_XS ->
    "unsloth-ud-iq4_xs" (qwen has an empty family tag, like the entry point's case)."""
    env = _env_of(attrs)
    model = (env.get("MODEL") or "").lower()
    if not model:
        return ""
    family = (env.get("FAMILY") or "qwen").lower()
    return model if family == "qwen" else f"{family}-{model}"


def run_config_path(attrs: dict | None, own_attrs: dict | None = None) -> Path | None:
    """The run config file of this Strata container, as THIS app can reach it. None if unknown.

    Two hops, both from docker inspect: the strata container's mount for $STRATA_DATA gives
    the HOST path of its /data, and this app's own mounts translate that host path into the
    path visible here (/home/jason/ai-lab/strata is /strata inside model-loader and /data's
    parent inside strata - the same directory, seen twice). With no matching mount - running
    on the host, say - the host path is used as-is, which is right there too.
    """
    env = _env_of(attrs)
    tag = config_tag(attrs)
    if not tag:
        return None
    data = env.get("STRATA_DATA") or "/data"
    host = ""
    for m in (attrs or {}).get("Mounts") or []:
        if (m or {}).get("Destination") == data:
            host = m.get("Source") or ""
            break
    if not host:
        return None
    path = host
    best, mapped = "", host
    for m in (own_attrs or {}).get("Mounts") or []:
        s = (m or {}).get("Source") or ""
        d = (m or {}).get("Destination") or ""
        if s and d and (host == s or host.startswith(s.rstrip("/") + "/")) and len(s) > len(best):
            best, mapped = s, d.rstrip("/") + host[len(s):]
    return Path(mapped) / "config" / f"strata-{tag}.json"


def _backup(path: Path) -> None:
    """One rolling backup per write, the ini.py discipline: timestamped, pruned, best-effort."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(path.name + f".bak-{ts}")
    n = 1
    while backup.exists():
        backup = path.with_name(path.name + f".bak-{ts}-{n}")
        n += 1
    try:
        shutil.copy(path, backup)
    except OSError:
        pass
    try:
        backups = sorted(path.parent.glob(path.name + ".bak-*"), reverse=True)
        for old in backups[BACKUPS_TO_KEEP:]:
            try:
                old.unlink()
            except OSError:
                continue
    except OSError:
        pass


def merge_run_config(path: Path, keys: dict) -> tuple[bool, str]:
    """Ensure `keys` in the run config, touching nothing else. Returns (ok, message).

    A value of None removes the key. The file is re-read first and written whole only if
    something actually differs, so a restart with no change leaves no backup and no mtime
    bump. Atomic tmp + os.replace, indent=1 - byte-compatible with what setup.py's own
    write_config would produce, so the file never ping-pongs in style between the writers.
    """
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return False, f"{path.name} is not there yet — the model's setup writes it first"
    except (OSError, ValueError) as e:
        return False, f"cannot read {path.name}: {e}"
    if not isinstance(cfg, dict):
        return False, f"{path.name} is not a config object"

    def _merge(target: dict) -> list[str]:
        changed: list[str] = []
        for k, v in keys.items():
            if v is None:
                if k in target:
                    del target[k]
                    changed.append(k)
            elif target.get(k) != v:
                target[k] = v
                changed.append(k)
        return changed

    changed = _merge(cfg)
    if not changed:
        return True, "no change"

    # setup.py inside the Strata container is the other writer of this file, and no
    # advisory lock spans two processes that do not both agree to take it. So the
    # race gets narrowed instead of closed: re-apply the merge to the freshest copy
    # microseconds before publishing, so anything setup.py wrote in between survives.
    try:
        fresh = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(fresh, dict):
            cfg = fresh
            changed = _merge(cfg)
            if not changed:
                return True, "no change"
    except (OSError, ValueError):
        pass   # publish the merge computed above rather than fail the call

    _backup(path)
    try:
        _atomic_replace(path, json.dumps(cfg, indent=1))
    except OSError as e:
        return False, f"cannot write {path.name}: {e}"
    return True, "applied " + ", ".join(sorted(changed))


def config_note(path: Path | None, keys: dict) -> str:
    """One line for the card when the run config does not yet say what the settings say.

    "" whenever there is nothing to say: no keys configured, or they are all in place.
    The note is deliberately about the DELTA, not the file - the config is mostly setup's,
    and a note about keys this app does not own would be noise.
    """
    if not keys:
        return ""
    if path is None:
        return "coexistence keys set, but this container's run config path is unknown"
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return "coexistence keys saved, but the run config does not exist yet — it applies when the model is set up"
    except (OSError, ValueError) as e:
        return f"coexistence keys not applied: {e}"
    if not isinstance(cfg, dict):
        return f"coexistence keys not applied: {path.name} is not a config object"
    missing = sorted(k for k, v in keys.items() if v is not None and cfg.get(k) != v)
    if missing:
        return ("coexistence keys saved but not in the run config yet: "
                + ", ".join(missing) + " — they are applied at app start; restart the app to re-apply")
    return ""
