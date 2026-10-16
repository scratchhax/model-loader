from __future__ import annotations

import asyncio
import configparser
import re
import shutil
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import docker
import httpx
from docker.errors import APIError, DockerException, NotFound

from .. import diagnose
from .. import ini
from .. import strata as strata_engine
from ..config import settings
from ..utils import docker_client, human_bytes, shard_key

from .dockerbase import _docker_client
from .backends import discover_llama_containers
from .. import hw


def _gpu_device_container(attrs: dict) -> bool:
    """Whether docker was told to give this container a GPU.

    Read from the device list rather than from what is currently on the cards, because a
    stopped container holds no VRAM and appears in no kernel table - and a stopped container
    is exactly what the switch has to reason about.

    /dev/kfd is the AMD compute device. DeviceRequests covers nvidia-container-toolkit, which
    is how this box was wired before the AMD swap and is cheap to keep honouring.
    """
    hc = (attrs or {}).get("HostConfig") or {}
    paths = {str(d.get("PathOnHost", "")) for d in (hc.get("Devices") or [])}
    if "/dev/kfd" in paths:
        return True
    return bool(hc.get("DeviceRequests"))


def gpu_holders() -> list[dict]:
    """Every container that was given a GPU: [{name, engine, backend, running, status}].

    `engine` is "" for a container that is not a backend at all (chatterbox, comfyui), which
    the switch still has to stop - they hold VRAM just the same.
    """
    client = _docker_client()
    if client is None:
        return []
    backends = {d["name"]: str(d.get("engine") or "llama") for d in discover_llama_containers()}
    out: list[dict] = []
    try:
        containers = client.containers.list(all=True)
    except DockerException:
        return []
    for c in containers:
        if c.name == "model-loader":
            continue
        try:
            attrs = c.attrs or {}
        except DockerException:
            continue
        if not _gpu_device_container(attrs):
            continue
        out.append({"name": c.name, "engine": backends.get(c.name, ""),
                    "backend": c.name in backends,
                    "running": c.status == "running", "status": c.status})
    out.sort(key=lambda r: r["name"])
    return out


def gpu_engines() -> list[dict]:
    """The engines that have at least one GPU backend: [{engine, names, running}].

    `running` means every one of that engine's GPU backends is up, so the switch can render a
    single active/inactive state per engine rather than per container. With one backend each -
    which is what this box has - the distinction never comes up, but a second GPU router would
    otherwise show as "active" while half of it was down.
    """
    by: dict[str, list[dict]] = {}
    for row in gpu_holders():
        if row["backend"] and row["engine"]:
            by.setdefault(row["engine"], []).append(row)
    return [{"engine": e, "names": [r["name"] for r in rows],
             "running": all(r["running"] for r in rows)}
            for e, rows in sorted(by.items())]
