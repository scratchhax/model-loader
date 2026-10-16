from __future__ import annotations

from datetime import datetime, timezone

from docker.errors import DockerException, NotFound

from .. import diagnose

from .dockerbase import _docker_client


def _started_epoch(container) -> float | None:
    """When the container's current run began, as fractional epoch seconds. None if unknown.

    Docker reports StartedAt with nanoseconds ("2026-09-13T18:50:08.749054149Z"). Keeping the
    fraction matters: flooring to the second can pull in the previous run's last lines, which is
    exactly what reading "this run only" exists to exclude.
    """
    raw = str(((getattr(container, "attrs", None) or {}).get("State") or {}).get("StartedAt") or "")
    head, _, frac = raw.rstrip("Z").partition(".")
    try:
        base = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None
    digits = "".join(ch for ch in frac if ch.isdigit())
    return base + (float("0." + digits) if digits else 0.0)


def container_logs(name: str, tail: int = 200, current_run_only: bool = False) -> tuple[bool, str]:
    """A container's recent log output.

    `current_run_only` limits the read to the present run. `docker logs` keeps history across
    restarts, so a plain tail reaches back into previous runs - fine for a human scrolling the log
    panel, wrong for anything that treats what it reads as the container's CURRENT state.
    """
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    try:
        c = client.containers.get(name)
    except NotFound:
        return False, "container not found"
    except DockerException as e:
        return False, f"docker error: {e}"
    try:
        kw = {}
        if current_run_only:
            since = _started_epoch(c)
            if since is not None:
                kw["since"] = since
        raw = c.logs(tail=tail, stdout=True, stderr=True, timestamps=False, **kw)
    except DockerException as e:
        return False, f"log fetch failed: {e}"
    return True, raw.decode("utf-8", errors="replace")


def diagnose_container(name: str) -> tuple[bool, list[dict], str]:
    """Recognised failure signatures from a backend's recent log, each with a suggested fix.

    Reads a wider tail than the log panel's 200 lines: a failed load is often preceded by a lot
    of chatter, and the decisive line can sit well above the most recent noise. Returns
    (ok, findings, err); findings is empty-and-ok when the log simply holds no known signature.

    Reads the CURRENT run only. `docker logs` keeps history across restarts, and a 1200-line tail
    easily reaches back past one: after a preset was fixed and the router restarted, Diagnose
    kept reporting the old section's missing-projector error from hours before the restart.
    Bounding by the run is the right cut rather than treating the router's "listening" line as a
    success marker, because the router validates presets during boot and only THEN listens - so
    a startup preset error sits just before that line, and would be hidden the moment it happened.
    """
    ok, tail = container_logs(name, tail=1200, current_run_only=True)
    if not ok:
        return False, [], tail
    findings = [{"error": f.error, "hint": f.hint, "model": f.model} for f in diagnose.diagnose(tail)]
    return True, findings, ""
