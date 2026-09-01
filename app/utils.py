from __future__ import annotations

import re


def human_bytes(n: float) -> str:
    step = 1024.0
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < step:
            return f"{int(n)} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= step
    return f"{n:.1f} EB"


_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})(?=\.[^.]+$)")


def shard_key(filename: str) -> tuple[str, int | None, int | None]:
    """Return (base_name_without_shard_suffix, part_index, part_total) or (name, None, None)."""
    m = _SHARD_RE.search(filename)
    if not m:
        return filename, None, None
    return _SHARD_RE.sub("", filename), int(m.group(1)), int(m.group(2))
