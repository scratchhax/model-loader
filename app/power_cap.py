"""The GPU board-power cap: what the owner asked for, and whether the cards actually hold it.

This app cannot set the cap, and should not: /sys is read-only in its container. It writes the
wanted value to /data/power-cap.env (~/ai-lab/model_loader_data on the host), and a host systemd
path unit, amd-power-cap.path, re-runs amd-power-cap.service, which writes power1_cap on each
amdgpu card and reads it back. The service reads the same file at boot, so the number set here is
also the one every boot applies. Everything privileged stays in that root-owned script, which
takes the value only as a plain whole number.

What this module adds is the check. The cards' caps are readable through the read-only sysfs, so
the Overview can say when a card does not hold the cap that was asked for. From 2026-10-22 the
service failed at every boot (ROCm 10.1's amd-smi changed its argument order) and both cards sat
at 300 W, past what the UPS can carry, with nothing on screen to say so.
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .fileio import atomic_replace

ENV_NAME = "power-cap.env"
KEY = "AMD_POWER_CAP_W"
DEFAULT_W = 210           # what amd-power-cap.service applies when the file is missing
SETTLE_S = 20.0           # the path unit and the script take a few seconds; no alarm inside this
SYSFS = "/sys/bus/pci/drivers/amdgpu"


@dataclass(frozen=True)
class Card:
    pci: str
    cap_w: int
    min_w: int
    max_w: int


def _watts(path: str) -> int:
    try:
        with open(path, encoding="ascii") as f:
            return round(int(f.read().strip()) / 1_000_000)
    except (OSError, ValueError):
        return 0


def cards(root: str | None = None) -> list[Card]:
    """Each amdgpu card that reports a power cap, in PCI order."""
    out = []
    for cap in sorted(glob.glob(os.path.join(root or SYSFS, "*", "hwmon", "hwmon*", "power1_cap"))):
        hw = os.path.dirname(cap)
        out.append(Card(pci=os.path.basename(os.path.dirname(os.path.dirname(hw))),
                        cap_w=_watts(cap), min_w=_watts(os.path.join(hw, "power1_cap_min")),
                        max_w=_watts(os.path.join(hw, "power1_cap_max"))))
    return out


def env_path() -> Path:
    return settings.data_dir / ENV_NAME


def desired() -> int:
    """The cap the service applies: the file's value, or the service's own default without one.

    Read the way the script reads it, so the two cannot disagree about what was asked for: a value
    that is not a plain whole number is the default there too.
    """
    try:
        text = env_path().read_text(encoding="utf-8")
    except OSError:
        return DEFAULT_W
    for line in text.splitlines():
        name, _, value = line.strip().partition("=")
        if name.strip() == KEY:
            value = value.strip()
            return int(value) if value.isdigit() and 2 <= len(value) <= 4 else DEFAULT_W
    return DEFAULT_W


def allowed(cs: list[Card]) -> tuple[int, int]:
    """The range every card accepts. (0, 0) when no card says."""
    lows = [c.min_w for c in cs if c.min_w]
    highs = [c.max_w for c in cs if c.max_w]
    return (max(lows) if lows else 0, min(highs) if highs else 0)


def set_desired(watts: int) -> None:
    atomic_replace(env_path(), f"{KEY}={int(watts)}\n")


def _expected(want: int, c: Card) -> int:
    """What the script will leave on this card: the request, clamped to the card's own range."""
    if c.min_w and want < c.min_w:
        return c.min_w
    if c.max_w and want > c.max_w:
        return c.max_w
    return want


def status(now: float | None = None, root: str | None = None) -> dict:
    """{want, cards, range, settling, off}: `off` lists the cards not holding the cap."""
    cs = cards(root)
    want = desired()
    try:
        age = (now or time.time()) - env_path().stat().st_mtime
    except OSError:
        age = SETTLE_S + 1
    off = [c for c in cs if c.cap_w != _expected(want, c)]
    settling = bool(off) and age < SETTLE_S
    return {"want": want, "cards": cs, "range": allowed(cs), "settling": settling,
            "off": [] if settling else off}
