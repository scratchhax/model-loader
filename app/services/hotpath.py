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


# ---------- hot-path HTTP ----------

_HTTP: httpx.AsyncClient | None = None
_HTTP_LOOP: "asyncio.AbstractEventLoop | None" = None


def _http() -> httpx.AsyncClient:
    """Shared AsyncClient for the hot poll paths (/v1/models, /slots, /props).

    A fresh client per request churned a TCP connect+teardown per backend every
    500 ms-2 s; one pooled client keeps the connections alive. It is bound to the
    loop that created it, so it is rebuilt if the running loop ever changes (the
    test suites spin a fresh loop per run). Per-call timeouts are passed at the
    request sites, which differ by an order of magnitude.
    """
    global _HTTP, _HTTP_LOOP
    loop = asyncio.get_running_loop()
    if _HTTP is None or _HTTP.is_closed or _HTTP_LOOP is not loop:
        _HTTP = httpx.AsyncClient(timeout=httpx.Timeout(2.0))
        _HTTP_LOOP = loop
    return _HTTP
