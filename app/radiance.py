"""Radiance's model choice: which .rad container the radiance service starts, and its own flags.

Radiance serves one model per process and reads it at startup, like Strata. The radiance service's
entry point (~/ai-lab/radiance/entrypoint.sh on wheatley) reads two kinds of file from the folder
the .rad containers live in, which is also this app's models folder (`rad/` under it):

    rad/radiance.env      RADIANCE_MODEL=<file>.rad   the one this app writes
    rad/<stem>.args       that model's own flags, one or more per line, # comments

The flags every model shares (port, key, slots, context, headroom) stay on the service's command
line in docker-compose.yaml; the .args file carries what differs between models (Flash-Next's
expert tiering and its MTP depth; the 27B needs none), and its --served-model-name.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import settings
from .fileio import atomic_replace

CONTAINER = "radiance"
SELECTION = "radiance.env"
_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.rad$")


def rad_dir() -> Path:
    return Path(settings.models_dir) / "rad"


def read_args(stem: str) -> list[str]:
    """The flags in rad/<stem>.args, comments dropped; [] when there is no file."""
    try:
        text = (rad_dir() / f"{stem}.args").read_text(encoding="utf-8")
    except OSError:
        return []
    return [tok for line in text.splitlines() for tok in line.split("#", 1)[0].split()]


def served_name(stem: str, args: list[str] | None = None) -> str:
    """The id the model is served as: its --served-model-name, else Radiance's default (the stem)."""
    args = read_args(stem) if args is None else args
    if "--served-model-name" in args and args.index("--served-model-name") + 1 < len(args):
        return args[args.index("--served-model-name") + 1]
    return stem


def installed() -> list[dict]:
    """Every .rad container in rad/, by name: file, stem, size, its flags and its served name."""
    out = []
    for p in sorted(rad_dir().glob("*.rad")):
        if not _FILE_RE.match(p.name):
            continue
        stem = p.name[:-4]
        args = read_args(stem)
        try:
            gb = round(p.stat().st_size / 1e9, 1)
        except OSError:
            gb = 0.0
        out.append({"file": p.name, "stem": stem, "gb": gb, "args": args,
                    "has_args": (rad_dir() / f"{stem}.args").exists(),
                    "served": served_name(stem, args)})
    return out


def selected() -> str:
    """The .rad file the next start loads, from rad/radiance.env; "" when none is chosen."""
    try:
        for line in (rad_dir() / SELECTION).read_text(encoding="utf-8").splitlines():
            if line.startswith("RADIANCE_MODEL="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def select(file: str) -> tuple[bool, str]:
    """Choose the .rad the radiance service starts next. Only a file that is in rad/."""
    names = {m["file"]: m for m in installed()}
    if file not in names:
        return False, f"{file!r} is not a .rad container in {rad_dir()}"
    atomic_replace(rad_dir() / SELECTION,
                   "# Written by Model Loader (Router tab): the model the radiance service starts.\n"
                   f"RADIANCE_MODEL={file}\n")
    m = names[file]
    note = "" if m["has_args"] else " (no .args file: the shared flags only)"
    return True, f"next start: {m['served']}{note}"


def restart_if_idle(docker) -> str:
    """Apply a new choice now when Radiance holds the cards and has nothing in flight: restart its
    container in the background (~1 min warm; it reads the new model at start). Otherwise the
    choice waits for the next start. Returns what happened, for the form note."""
    import threading

    import httpx
    from .services import auth_headers
    try:
        c = docker.containers.get(CONTAINER)
    except Exception:  # noqa: BLE001 - no container, nothing to restart
        return "it loads at Radiance's next start"
    if c.status != "running":
        return "it loads at Radiance's next start"
    try:
        h = httpx.get(f"http://{CONTAINER}:8080/health", timeout=3.0, headers=auth_headers(CONTAINER)).json()
    except (httpx.HTTPError, ValueError):
        h = {}
    if int(h.get("in_flight") or 0) > 0:
        return "Radiance is answering a request, so it loads at the next start (or restart it when it is done)"
    threading.Thread(target=lambda: c.restart(timeout=60), daemon=True, name="radiance-restart").start()
    return "restarting Radiance to load it (about a minute warm)"


def choices() -> list[tuple[str, str]]:
    """[(file, label)] for the Router tab's Radiance row."""
    return [(m["file"], f"{m['served']}  ({m['gb']:.0f} GB)" + ("" if m["has_args"] else "  — no .args file"))
            for m in installed()]


# ---------------------------------------------------------------- what the serving model is

_QUANT_RE = re.compile(r"(?<![a-z0-9])(bf16|fp8|mxfp4|m2xfp4|nvfp4|w4a8|w4a16|w5|w8a8|i8g|iq\d+r|paro-int\d|paro)(?![a-z0-9])")
_facts_cache: dict[str, tuple[float, dict]] = {}
_FACTS_TTL_S = 10.0


def quant_of(file: str) -> str:
    """The weight formats a .rad file's name declares, joined: qwen3.8-next-flash-fp8-iq4r-moe.rad
    -> "fp8+iq4r" (an fp8/int8 trunk with 4-bit IQR experts), qwen3.8-27b-fp8.rad -> "fp8"."""
    seen: list[str] = []
    for m in _QUANT_RE.finditer(file.lower().removesuffix(".rad")):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return "+".join(seen)


def _prom_sum(text: str, name: str) -> float | None:
    vals = []
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            try:
                vals.append(float(line.rsplit(" ", 1)[1]))
            except (ValueError, IndexError):
                pass
    return sum(vals) if vals else None


def server_facts(container: str, port: int) -> dict:
    """What the Overview's hero shows for a serving Radiance, read from the server itself (cached
    10 s): {file, quant, size_bytes, ctx, drafter, depth, accepted, drafted}. The model has no
    models.ini section and no GGUF header, which is where these come from on llama. {} on failure.

    `accepted`/`drafted` are Radiance's counters since it started (radiance:draft_*_total), so the
    acceptance is a running figure for this run, not llama's per-request median.
    """
    import time

    import httpx
    from .services import auth_headers
    now = time.time()
    hit = _facts_cache.get(container)
    if hit and now - hit[0] < _FACTS_TTL_S:
        return hit[1]
    out: dict = {}
    base, hdr = f"http://{container}:{port}", auth_headers(container)
    try:
        with httpx.Client(timeout=httpx.Timeout(3.0)) as c:
            info = c.get(base + "/server_info", headers=hdr).json()
            models = c.get(base + "/v1/models", headers=hdr).json().get("data") or [{}]
            metrics = c.get(base + "/metrics", headers=hdr).text
    except (httpx.HTTPError, ValueError, AttributeError):
        _facts_cache[container] = (now, out)
        return out
    spec = ((info.get("engine") or {}).get("speculation") or {})
    root = str(models[0].get("root") or "")
    file = root.rsplit("/", 1)[-1]
    out = {"file": file, "quant": quant_of(file),
           "ctx": int(models[0].get("max_model_len") or (info.get("server") or {}).get("max_model_len") or 0),
           "drafter": str(spec.get("drafter") or "") if spec.get("enabled") else "",
           "depth": int(spec.get("depth") or 0),
           "accepted": _prom_sum(metrics, "radiance:draft_accepted_total"),
           "drafted": _prom_sum(metrics, "radiance:draft_tokens_total")}
    try:
        out["size_bytes"] = (rad_dir() / file).stat().st_size if file else 0
    except OSError:
        out["size_bytes"] = 0
    _facts_cache[container] = (now, out)
    return out


def hero_mtp(facts: dict) -> dict | None:
    """The Overview's speculation readout for Radiance, in the shape overview._hero_mtp returns.
    None when it does not speculate."""
    if not facts.get("drafter"):
        return None
    acc, drafted = facts.get("accepted"), facts.get("drafted")
    pct = round(100.0 * acc / drafted, 1) if acc is not None and drafted else None
    label = "MTP" if facts["drafter"] == "mtp" else facts["drafter"].upper()
    basis = (f"{pct:.0f}% of the {int(drafted):,} tokens the draft proposed since Radiance started were "
             f"accepted by the target model (its own counters; depth {facts.get('depth') or '?'})."
             if pct is not None else "no drafted tokens yet: the figure appears after the first generation.")
    return {"spec_type": facts["drafter"], "builtin": False, "label": label, "acc_pct": pct,
            "mean_len": 0.0, "n": int(drafted or 0), "age": "", "n_max": int(facts.get("depth") or 0),
            "basis": basis, "cost": None}
