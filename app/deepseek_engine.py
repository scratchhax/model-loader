"""DeepSeek's engine version: pinned to one installed package, or following upstream.

Strata 0.2's setup checks upstream's signed feed on EVERY start of the DeepSeek container and swaps
in a newer strata-dsv41 if one is published - independently of the image tag, which compose pins.
On 2026-10-30 that turned a routine engine switch into a 15-minute one: 0.2.8 shipped engine
0.1.54 the night before, the next start fetched it unasked, and a new engine version counts as
untuned, so the tuner ran again.

Upstream's own off switch is STRATA_DSV41_EXE: with it set, setup and launch use that build and
never run the upgrade ("the person chose the engine"). An upgrade is also what deletes the older
package folders, so a pinned version stays on disk. This module owns ~/ai-lab/strata/deepseek.env,
which holds that one variable; the DeepSeek container's wrapper (strata/entrypoint-ds.sh on
wheatley) sources it on every start, so a pin or unpin applies at the next start with no recreate.
Empty means follow upstream: the next start takes the newest engine and re-tunes once for it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .fileio import atomic_replace
from .strata import STRATA_ROOT

PIN_PATH = Path(os.environ.get("DEEPSEEK_ENV_PATH") or STRATA_ROOT / "deepseek.env")
# Where setup installs the packages, seen from here (/strata is ~/ai-lab/strata) and from inside
# the DeepSeek container (its /data is ~/ai-lab/strata/data-ds).
ENGINE_DIR = Path(os.environ.get("DEEPSEEK_ENGINE_DIR") or STRATA_ROOT / "data-ds" / "install" / "engine-dsv41")
CONTAINER_ENGINE_DIR = "/data/install/engine-dsv41"
EXE = "strata-dsv41"
KEY = "STRATA_DSV41_EXE"
CONTAINER = os.environ.get("DEEPSEEK_CONTAINER", "strata-ds")
WRAPPER = "entrypoint-ds.sh"

# A package folder: "<version>-<system>-<backend>-<variant>", e.g. 0.1.54-linux-x86_64-hip-rx9070.
_FOLDER = re.compile(r"^(\d+(?:\.\d+)+)-(.+)$")

_HEADER = """\
# DeepSeek's engine pin. OWNED BY MODEL LOADER (Overview -> Engine updates -> DeepSeek engine):
# change it there, not by hand. Sourced by strata/entrypoint-ds.sh on every start of strata-ds.
#
# Set: setup and launch use this strata-dsv41 build and never fetch a newer one.
# Empty: each start takes upstream's newest DeepSeek engine, and a new version re-tunes once (~13 min).
"""


def _vkey(version: str) -> tuple:
    return tuple(int(x) for x in version.split("."))


def installed() -> list[dict]:
    """Installed engine packages, newest first: [{folder, version, flavor}]. [] when unreadable."""
    try:
        entries = list(ENGINE_DIR.iterdir())
    except OSError:
        return []
    out = []
    for p in entries:
        m = _FOLDER.match(p.name)
        if m and p.is_dir() and (p / EXE).exists():
            out.append({"folder": p.name, "version": m.group(1), "flavor": m.group(2)})
    return sorted(out, key=lambda d: _vkey(d["version"]), reverse=True)


def read_pin() -> str:
    """The pinned package's folder name, or "" when following upstream (or no file yet)."""
    try:
        text = PIN_PATH.read_text()
    except OSError:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(KEY + "="):
            value = line.split("=", 1)[1].strip().strip("'\"")
            parts = Path(value).parts
            return parts[-2] if len(parts) >= 2 and parts[-1] == EXE else ""
    return ""


def _write(folder: str) -> None:
    value = f"{CONTAINER_ENGINE_DIR}/{folder}/{EXE}" if folder else ""
    atomic_replace(PIN_PATH, f"{_HEADER}{KEY}={value}\n")


def set_pin(folder: str) -> tuple[bool, str]:
    """Pin to an installed package (by folder name), or follow upstream with ""."""
    if not folder:
        _write("")
        return True, "DeepSeek follows upstream: its next start takes the newest engine (and re-tunes once if it is new)"
    found = next((d for d in installed() if d["folder"] == folder), None)
    if found is None:
        return False, f"{folder} is not installed, so it cannot be pinned"
    _write(folder)
    return True, f"DeepSeek pinned to engine {found['version']}: from its next start no newer engine is fetched"


def wired(docker) -> bool | None:
    """Does the DeepSeek container read the pin file? True when its entrypoint is our wrapper;
    None when it cannot be told (no docker, no container). A pin it does not read is not a pin."""
    if docker is None:
        return None
    try:
        attrs = docker.containers.get(CONTAINER).attrs or {}
    except Exception:  # noqa: BLE001 - no container is a normal state off wheatley
        return None
    entry = (attrs.get("Config") or {}).get("Entrypoint") or []
    return any(WRAPPER in str(part) for part in entry)


def status(docker=None, latest: str = "") -> dict:
    """What the Engine updates card shows: installed packages, the pin, upstream's newest."""
    inst = installed()
    pin = read_pin()
    pinned = next((d for d in inst if d["folder"] == pin), None)
    newest = inst[0] if inst else None
    return {
        "installed": inst,
        "pinned": pinned,
        "pin_missing": bool(pin) and pinned is None,
        "newest": newest,
        "latest": latest,
        "behind": bool(latest and newest and _vkey(latest) > _vkey(newest["version"])),
        "wired": wired(docker),
    }
