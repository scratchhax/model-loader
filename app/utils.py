from __future__ import annotations

import re
import shlex
import threading
import time

import docker
from docker.errors import DockerException


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


_DOCKER_CLIENT: "docker.DockerClient | None" = None
_DOCKER_CLIENT_OK = 0.0
_DOCKER_LOCK = threading.Lock()


def docker_client() -> "docker.DockerClient | None":
    """One shared docker client for the whole app, not one per call.

    Every module used to call `docker.from_env()` per call and none ever closed it:
    with the 500 ms-2 s polls that leaked a requests.Session and a urllib3 pool
    continuously. The SDK client is safe to share across threads (the urllib3 pool
    is thread-safe and the session is read-only after construction), and it has to
    survive a dockerd restart, so the cached client is pinged at most once per 5 s
    and rebuilt when the ping fails - the self-heal the per-call version got free.

    Returns None when docker is unreachable, like the per-call version did.
    """
    global _DOCKER_CLIENT, _DOCKER_CLIENT_OK
    with _DOCKER_LOCK:
        now = time.time()
        if _DOCKER_CLIENT is not None and now - _DOCKER_CLIENT_OK < 5.0:
            return _DOCKER_CLIENT
        if _DOCKER_CLIENT is not None:
            try:
                _DOCKER_CLIENT.ping()
                _DOCKER_CLIENT_OK = now
                return _DOCKER_CLIENT
            except DockerException:
                try:
                    _DOCKER_CLIENT.close()
                except Exception:  # noqa: BLE001 - best effort; rebuild regardless
                    pass
        try:
            # from_env() itself round-trips /version, so a fresh client needs no ping.
            _DOCKER_CLIENT = docker.from_env()
            _DOCKER_CLIENT_OK = time.time()
        except DockerException:
            _DOCKER_CLIENT = None
        return _DOCKER_CLIENT


def timed_shell(cmd: str, seconds: int = 5) -> list[str]:
    """Wrap a shell command for docker exec_run so it cannot hang forever.

    exec_run has no timeout parameter: a wedged nvidia-smi/rocm-smi on a sick
    GPU blocks the read indefinitely, freezing whatever thread asked - the 2 s
    sampler, or worse an event-loop handler. coreutils `timeout` inside the
    container caps it (exit 124, which every caller already treats as "no
    data"); on an image without `timeout` the exec dies at 127 instead.
    Degraded telemetry beats a frozen sampler.
    """
    return ["sh", "-c", f"timeout {seconds} sh -c {shlex.quote(cmd)}"]
