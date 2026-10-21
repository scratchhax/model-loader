"""Writing files that another process reads: atomically, and with a rolling backup.

strata.env (sourced by Strata's entry point at start), Strata's run configs and Maya's configs are
all read by an engine while this app may be writing them. They used to share Strata's private
helpers, with Maya reaching into strata.py for them; they live here now.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path


def atomic_replace(path: Path, data: str) -> None:
    """tmp + fsync + os.replace, preserving the target's mode.

    A truncated write (plain write_text) leaves a reader half a file; mkstemp alone would fix the
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


def backup(path: Path, keep: int = 5) -> None:
    """One rolling backup per write, the ini.py discipline: timestamped, pruned, best-effort."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = path.with_name(path.name + f".bak-{ts}")
    n = 1
    while dest.exists():
        dest = path.with_name(path.name + f".bak-{ts}-{n}")
        n += 1
    try:
        shutil.copy(path, dest)
    except OSError:
        pass
    try:
        for old in sorted(path.parent.glob(path.name + ".bak-*"), reverse=True)[keep:]:
            try:
                old.unlink()
            except OSError:
                continue
    except OSError:
        pass
