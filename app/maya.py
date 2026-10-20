"""Maya (Project Maya: GLM-5.3-Flash / DeepSeek-V4.1-Flash), the third engine.

Like Strata it serves one model per container, chosen by the config Maya's own setup wrote, so
none of the llama.cpp machinery applies. This module owns two things about it:

  * which models are INSTALLED - the set a model dropdown has to offer whether or not the
    container is running. The engines take turns, so Maya is usually stopped exactly when you
    are looking at it;
  * a small, named set of settings in the config that will run next, the way strata_models owns
    Strata's tunables: single writer of exactly those keys, never of setup's.

The configs live on Maya's volume (~/ai-lab/maya/data/conf, /data/conf inside it; beside each one
the dashboard's <config>.shared-settings.json, which is not a model). Model Loader mounts that
directory read-write at CONF_DIR. Reads fall back to docker's archive API (the call behind
`docker cp`, which works on a stopped container) when the mount is missing, so the dropdown still
fills on a deployment without it; writes need the mount.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import threading
import time
from pathlib import Path

from docker.errors import DockerException, NotFound

ENGINE = "maya"
CONTAINER = os.environ.get("MAYA_CONTAINER", "maya")
# The config directory as Maya sees it, and as this app sees it through its mount.
ENGINE_CONF_DIR = "/data/conf"
CONF_DIR = Path(os.environ.get("MAYA_CONF_DIR") or "/maya/conf")

_TTL_S = 30.0
_cache: dict[str, tuple[float, list[dict]]] = {}
_lock = threading.Lock()


def _is_model_config(name: str) -> bool:
    return name.startswith("maya-") and name.endswith(".json") \
        and not name.endswith(".shared-settings.json")


# ------------------------------------------------------------------------------- reading

def _configs_from_mount(conf_dir: Path) -> list[dict] | None:
    """Every model config in the mounted dir, or None when there is no such mount."""
    if not conf_dir.is_dir():
        return None
    out: list[dict] = []
    for p in sorted(conf_dir.iterdir()):
        if not (p.is_file() and _is_model_config(p.name)):
            continue
        try:
            cfg = json.loads(p.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(cfg, dict):
            cfg["_file"] = p.name
            out.append(cfg)
    return out


def _configs_from_archive(client, container: str) -> list[dict]:
    """Every model config in the container's conf dir via docker cp. [] when unreadable."""
    try:
        stream, _ = client.containers.get(container).get_archive(ENGINE_CONF_DIR)
        raw = b"".join(stream)
    except (NotFound, DockerException, OSError, AttributeError):
        return []
    out: list[dict] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            for m in tar.getmembers():
                name = m.name.rsplit("/", 1)[-1]
                if not (m.isfile() and _is_model_config(name)):
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                try:
                    cfg = json.loads(f.read().decode("utf-8-sig"))
                except (ValueError, UnicodeDecodeError):
                    continue
                if isinstance(cfg, dict):
                    cfg["_file"] = name
                    out.append(cfg)
    except tarfile.TarError:
        return []
    return out


def _configs(client, container: str, conf_dir: Path | None = None) -> list[dict]:
    got = _configs_from_mount(conf_dir or CONF_DIR)
    return got if got is not None else _configs_from_archive(client, container)


def installed(client, container: str = CONTAINER, conf_dir: Path | None = None) -> list[dict]:
    """[{model, quant, file, context}] for each installed Maya model, cached for _TTL_S.

    `model` is the id Maya serves it as (what /v1/models answers, and what `maya` resolves to),
    `quant` the download it was set up from. Sorted by model then quant so the order is stable.
    """
    key = f"{container}:{conf_dir or CONF_DIR}"
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < _TTL_S:
            return hit[1]
    rows = []
    for cfg in _configs(client, container, conf_dir):
        inst = cfg.get("installer") or {}
        model = str(cfg.get("model_name") or "").strip()
        if not model:
            continue
        rows.append({"model": model, "quant": str(inst.get("quant") or inst.get("ds4_quant") or ""),
                     "file": cfg["_file"], "context": _arg(cfg.get("args") or [], "--max-context")})
    rows.sort(key=lambda r: (r["model"], r["quant"]))
    with _lock:
        _cache[key] = (now, rows)
    return rows


def choices(client, container: str = CONTAINER) -> list[tuple[str, str]]:
    """[(model id, label)] for the Router tab's dropdown: the installed models."""
    return [(r["model"], f"{r['model']}  ({r['quant']})" if r["quant"] else r["model"])
            for r in installed(client, container)]


def run_config_name(client, container: str = CONTAINER) -> str:
    """The file name of the config the container starts with, from its own command line.

    `--config /data/conf/<name>` is in the compose service's command, so that - not a guess
    from what is installed - is the config the next start reads. "" when it cannot be told.
    """
    try:
        cmd = (client.containers.get(container).attrs.get("Config") or {}).get("Cmd") or []
    except (NotFound, DockerException, AttributeError):
        return ""
    if "--config" in cmd[:-1]:
        return str(cmd[cmd.index("--config") + 1]).rsplit("/", 1)[-1]
    return ""


def health(client, container: str = CONTAINER) -> dict | None:
    """Maya's /health as a dict when it is serving, else None."""
    import httpx
    try:
        c = client.containers.get(container)
        if c.status != "running":
            return None
        with httpx.Client(timeout=httpx.Timeout(3.0)) as h:
            r = h.get(f"http://{container}:8080/health")
        return r.json() if r.status_code == 200 else None
    except (NotFound, DockerException, httpx.HTTPError, ValueError):
        return None


# ------------------------------------------------------------------------------- settings
#
# Only knobs that cannot stop Maya starting. Each sizes itself against what is free at start:
# the expert pool takes what VRAM is left, the RAM tier what RAM is left, so a bigger value
# costs speed rather than a start. The rest of Maya's ~60 settings are for other hardware
# (NVLink, Windows APUs, Volta, EXL3), debugging, or "changes the answers" approximations, and
# are deliberately not offered.
#
# kind "arg": a flag in the config's args list. kind "env": keys in its env block. A choice of
# "" means "not set" - the key is removed and the engine's own default applies.

TUNABLES = {
    "context": {
        "label": "Context window",
        "about": ("Tokens per conversation. Past the 32K positions each layer keeps in VRAM, the "
                  "attention cache streams from RAM at about 12 KB a token (1.6 GB at 128K), so "
                  "a longer window costs RAM, not VRAM, and does not shrink the expert pool."),
        "kind": "arg", "key": "--max-context",
        "choices": [("8192", "8K"), ("32768", "32K"), ("65536", "64K"), ("131072", "128K")],
        "metric": "max_context",
    },
    "parallel": {
        "label": "Concurrent conversations",
        "about": ("How many conversations decode together. One keeps a single chat at full "
                  "speed; more serves an agent that fans out, each extra conversation holding its "
                  "own state."),
        "kind": "arg", "key": "--parallel",
        "choices": [("", "1 (default)"), ("2", "2"), ("4", "4")],
        "metric": None,
    },
    "slots": {
        "label": "Parked conversations",
        "about": ("Conversations kept aside on the SSD, so switching back to one does not re-read "
                  "its prompt. Maya's default is 4; set it above the number of conversations a "
                  "client keeps going."),
        "kind": "env", "key": ("MAYA_GLM_SLOTS",),
        "choices": [("", "4 (default)"), ("0", "off"), ("8", "8"), ("16", "16")],
        "metric": None,
    },
    "cpu-lane": {
        "label": "CPU threads per card",
        "about": ("The CPU computes the experts that are in neither card's VRAM, one thread pool "
                  "per card, half the cores each by default. Card 1 sits on a slower PCIe link "
                  "here (an expert crosses it in 1.07 ms against card 0's 0.19 ms), so it leans "
                  "on the CPU harder - Maya's docs name exactly this case for setting the pools "
                  "per card."),
        "kind": "env", "key": ("MAYA_GLM_CPU_LANE0", "MAYA_GLM_CPU_LANE1"),
        "choices": [("", "7 + 7 (default)"), ("5,9", "5 + 9"), ("4,10", "4 + 10")],
        "metric": None,
    },
}


def _arg(args: list, key: str) -> str:
    return str(args[args.index(key) + 1]) if key in args[:-1] else ""


def config_values(cfg: dict) -> dict[str, str]:
    """{tunable: value as the choice list spells it} read from one config."""
    args, env = cfg.get("args") or [], cfg.get("env") or {}
    out: dict[str, str] = {}
    for name, spec in TUNABLES.items():
        if spec["kind"] == "arg":
            out[name] = _arg(args, spec["key"])
        else:
            vals = [str(env.get(k, "")) for k in spec["key"]]
            out[name] = ",".join(vals) if any(vals) else ""
    return out


def tuning_state(cfg: dict, metrics: dict | None = None) -> dict[str, str]:
    """{tunable: current value}, preferring what the ENGINE reports over what the config says.

    The two differ exactly between a change and the restart that applies it, and showing the
    config there would claim a setting is live when it is not. A value outside the offered
    choices is kept as it is rather than snapped to one, so a hand edit stays visible.
    """
    out = config_values(cfg)
    eng = (metrics or {}).get("engine") or {}
    for name, spec in TUNABLES.items():
        key = spec.get("metric")
        if key and eng.get(key) is not None:
            out[name] = str(eng[key])
    return out


def read_config(name: str, conf_dir: Path | None = None) -> dict | None:
    try:
        cfg = json.loads(((conf_dir or CONF_DIR) / name).read_text(encoding="utf-8-sig"))
        return cfg if isinstance(cfg, dict) else None
    except (OSError, ValueError):
        return None


def set_tunable(name: str, value: str, config: str, conf_dir: Path | None = None) -> tuple[bool, str]:
    """Write one setting into `config` for the next start. (ok, message).

    Re-reads the file, changes only the named keys, and replaces it atomically with a rolling
    backup - Strata's run-config discipline (strata._atomic_replace / _backup), because this
    file is read by another program at start and a half-written one is a container that will
    not come up.
    """
    from . import strata as _st   # the atomic writer and backup rotation, shared
    if name not in TUNABLES:
        return False, f"not a Maya setting: {name}"
    spec = TUNABLES[name]
    if value not in {v for v, _ in spec["choices"]}:
        return False, f"not an allowed value for {spec['label'].lower()}: {value!r}"
    if not config or "/" in config or not _is_model_config(config):
        return False, f"not a Maya config: {config!r}"
    path = (conf_dir or CONF_DIR) / config
    if not path.parent.is_dir():
        return False, (f"{path.parent} is not mounted into Model Loader, so Maya's config "
                       "cannot be written from here")
    cfg = read_config(config, conf_dir)
    if cfg is None:
        return False, f"cannot read {config}"

    if spec["kind"] == "arg":
        args = list(cfg.get("args") or [])
        key = spec["key"]
        if not value:                        # back to Maya's default: the flag goes
            if key in args:
                i = args.index(key)
                del args[i:i + 2]
        elif key in args[:-1]:
            args[args.index(key) + 1] = value
        elif key in args:                    # a flag with its value missing: complete it
            args.append(value)
        else:
            args += [key, value]
        cfg["args"] = args
    else:
        env = dict(cfg.get("env") or {})
        parts = value.split(",") if value else []
        for i, k in enumerate(spec["key"]):
            if parts:
                env[k] = parts[i]
            else:
                env.pop(k, None)
        cfg["env"] = env

    _st._backup(path)
    try:
        _st._atomic_replace(path, json.dumps(cfg, indent=1))
    except OSError as e:
        return False, f"cannot write {config}: {e}"
    with _lock:
        _cache.clear()
    label = dict(spec["choices"]).get(value, value)
    return True, f"{spec['label']} set to {label}; applies on Maya's next start"
