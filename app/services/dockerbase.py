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
import socket

from ..utils import docker_client, human_bytes, shard_key


# ---------- docker / containers ----------

# One shared client for the whole app (see utils.docker_client). The alias keeps the
# ~25 call sites below spelled the way they always have.
_docker_client = docker_client


def strata_own_attrs(client) -> dict:
    """This app's own inspect data. Its mounts are the second half of the path translation:
    a host path only means something here once it has been mapped through what this
    container mounts where. Outside Docker there is nothing to translate."""
    try:
        return client.containers.get(socket.gethostname()).attrs or {}
    except Exception:  # noqa: BLE001 - a name docker does not know just means "no mapping"
        return {}
