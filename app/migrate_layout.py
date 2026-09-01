"""One-shot migration: move each flat GGUF into /models/<stem>/ and update models.ini.

Idempotent: safe to re-run. Files already in subdirs are left alone.
Multi-referenced mmproj files are COPIED into each referencing section's dir, then the
top-level original is removed.
"""
from __future__ import annotations

import io
import shutil
from pathlib import Path

from . import ini
from .config import settings


def _log(msg: str) -> None:
    print(f"[migrate] {msg}", flush=True)


def run() -> dict:
    root = settings.models_dir
    if not root.is_dir():
        return {"ok": False, "error": f"{root} not a directory"}

    moved: list[str] = []
    mmproj_placed: list[str] = []
    ini_updates: list[str] = []
    skipped: list[str] = []

    # ---- pass 1: move flat main GGUFs (non-mmproj) into their own subdirs
    for f in sorted(root.iterdir()):
        if not (f.is_file() and f.suffix.lower() == ".gguf"):
            continue
        if "mmproj" in f.name.lower():
            continue  # handled in pass 3
        stem = f.stem
        target_dir = root / stem
        target = target_dir / f.name
        if target.exists() and target.resolve() == f.resolve():
            skipped.append(f"{f.name} (already in place)")
            continue
        try:
            target_dir.mkdir(exist_ok=True)
            shutil.move(str(f), str(target))
            moved.append(f"{f.name} -> {stem}/{f.name}")
        except OSError as e:
            skipped.append(f"{f.name}: move failed ({e})")

    # ---- pass 2: rewrite ini so every section has explicit `model = <stem>/<file>.gguf`
    cp = ini.read_ini()
    for sec in cp.sections():
        items = dict(cp.items(sec))
        if items.get("model"):
            continue  # user already set it explicitly
        candidate = root / sec / f"{sec}.gguf"
        if candidate.is_file():
            abs_path = f"/models/{sec}/{sec}.gguf"
            cp.set(sec, "model", abs_path)
            ini_updates.append(f"[{sec}] model = {abs_path}")

    # ---- pass 3: handle mmproj files still at top-level
    top_level_mmproj = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".gguf" and "mmproj" in p.name.lower()]
    for mp in top_level_mmproj:
        # Which sections reference this file by basename?
        refs: list[str] = []
        for sec in cp.sections():
            items = dict(cp.items(sec))
            mv = (items.get("mmproj") or "").strip()
            if not mv:
                continue
            # match by basename (users often set "/models/mmproj-model-f16.gguf")
            if Path(mv).name == mp.name:
                refs.append(sec)

        if not refs:
            # Orphan mmproj — move into its own stem-dir so it stops floating around
            stem = mp.stem
            target = root / stem / mp.name
            try:
                (root / stem).mkdir(exist_ok=True)
                if not target.exists():
                    shutil.move(str(mp), str(target))
                    mmproj_placed.append(f"{mp.name} (orphan) -> {stem}/{mp.name}")
            except OSError as e:
                skipped.append(f"{mp.name} (orphan): {e}")
            continue

        # Copy into each referencing section's dir, then delete original
        for sec in refs:
            sec_dir = root / sec
            sec_dir.mkdir(exist_ok=True)
            target = sec_dir / mp.name
            try:
                if not target.exists():
                    shutil.copy2(str(mp), str(target))
                mmproj_abs = f"/models/{sec}/{mp.name}"
                cp.set(sec, "mmproj", mmproj_abs)
                ini_updates.append(f"[{sec}] mmproj = {mmproj_abs}")
                mmproj_placed.append(f"{mp.name} -> {sec}/{mp.name}")
            except OSError as e:
                skipped.append(f"{mp.name} -> {sec}: {e}")
        # Remove the original at top-level after all copies
        try:
            mp.unlink()
        except OSError as e:
            skipped.append(f"unlink {mp.name}: {e}")

    # ---- pass 4: promote any relative model/mmproj paths back to absolute (llama-server needs abs)
    for sec in cp.sections():
        items = dict(cp.items(sec))
        for key in ("model", "mmproj"):
            v = (items.get(key) or "").strip()
            if v and not v.startswith("/"):
                cp.set(sec, key, f"/models/{v}")
                ini_updates.append(f"[{sec}] {key}: {v} → /models/{v}")

    # ---- write updated ini via ini._atomic_write (handles backup)
    if ini_updates:
        ini._atomic_write(cp)

    return {
        "ok": True,
        "moved_ggufs": moved,
        "mmproj_placed": mmproj_placed,
        "ini_updates": ini_updates,
        "skipped": skipped,
    }


if __name__ == "__main__":
    import json
    result = run()
    print(json.dumps(result, indent=2))
