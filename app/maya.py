"""Maya (Project Maya: GLM-5.3-Flash / DeepSeek-V4.1-Flash), the third engine.

Like Strata it serves one model per container, chosen by the config Maya's own setup wrote, so
none of the llama.cpp machinery applies. What this module owns is the one question the Router
tab asks of it: which models are INSTALLED. That is the set a model dropdown has to offer,
whether or not the container is running - the engines take turns, so Maya is usually stopped
exactly when you are looking at it.

The configs live on Maya's own volume (/data/conf/maya-*.json, beside them the dashboard's
<config>.shared-settings.json, which is not a model). Model Loader has no mount of that volume,
and adding one would mean recreating this container for a read; docker can copy files out of a
container whether it runs or not (the API behind `docker cp`), so they are read from there.
"""

from __future__ import annotations

import io
import json
import tarfile
import threading
import time

from docker.errors import DockerException, NotFound

ENGINE = "maya"
CONF_DIR = "/data/conf"

_TTL_S = 30.0
_cache: dict[str, tuple[float, list[dict]]] = {}
_lock = threading.Lock()


def _read_configs(client, container: str) -> list[dict]:
    """Every model config in the container's conf dir, as parsed JSON. [] when unreadable."""
    try:
        stream, _ = client.containers.get(container).get_archive(CONF_DIR)
        raw = b"".join(stream)
    except (NotFound, DockerException, OSError):
        return []
    out: list[dict] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            for m in tar.getmembers():
                name = m.name.rsplit("/", 1)[-1]
                if not (m.isfile() and name.startswith("maya-") and name.endswith(".json")) \
                        or name.endswith(".shared-settings.json"):
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


def installed(client, container: str = ENGINE) -> list[dict]:
    """[{model, quant, file}] for each installed Maya model, cached for _TTL_S.

    `model` is the id Maya serves it as (what /v1/models answers, and what `maya` resolves to),
    `quant` the download it was set up from. Sorted by model then quant so the order is stable.
    """
    now = time.time()
    with _lock:
        hit = _cache.get(container)
        if hit and now - hit[0] < _TTL_S:
            return hit[1]
    rows = []
    for cfg in _read_configs(client, container):
        inst = cfg.get("installer") or {}
        model = str(cfg.get("model_name") or "").strip()
        if not model:
            continue
        rows.append({"model": model, "quant": str(inst.get("quant") or inst.get("ds4_quant") or ""),
                     "file": cfg["_file"]})
    rows.sort(key=lambda r: (r["model"], r["quant"]))
    with _lock:
        _cache[container] = (now, rows)
    return rows


def choices(client, container: str = ENGINE) -> list[tuple[str, str]]:
    """[(model id, label)] for the Router tab's dropdown: the installed models."""
    return [(r["model"], f"{r['model']}  ({r['quant']})" if r["quant"] else r["model"])
            for r in installed(client, container)]
