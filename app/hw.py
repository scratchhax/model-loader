from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import docker
from docker.errors import DockerException, NotFound

from .config import settings


@dataclass
class GpuStats:
    vendor: str
    name: str
    util_pct: float
    vram_used_gb: float
    vram_total_gb: float
    temp_c: float
    power_w: float


@dataclass
class ContainerRuntimeStats:
    cpu_pct: float
    mem_used_gb: float
    mem_limit_gb: float


@dataclass
class BackendStats:
    ok: bool
    error: str | None = None
    gpu: GpuStats | None = None
    container: ContainerRuntimeStats | None = None


_CACHE: dict[str, BackendStats] = {}
_SAMPLE_INTERVAL_S = 2.0
_VENDOR_CACHE: dict[str, str] = {}
_LOCK = threading.Lock()
_sampler_started = False


def _client() -> docker.DockerClient | None:
    try:
        return docker.from_env()
    except DockerException:
        return None


def _detect_vendor(container) -> str:
    with _LOCK:
        cached = _VENDOR_CACHE.get(container.name)
        if cached:
            return cached
    image = ""
    try:
        image = ((container.image.tags or [container.image.short_id]) or [""])[0].lower()
    except DockerException:
        pass
    if "rocm" in image:
        vendor = "amd"
    elif "cuda" in image or "nvidia" in image:
        vendor = "nvidia"
    else:
        vendor = "unknown"
    with _LOCK:
        _VENDOR_CACHE[container.name] = vendor
    return vendor


def _read_nvidia(container) -> GpuStats | None:
    cmd = (
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw,name --format=csv,noheader,nounits"
    )
    try:
        r = container.exec_run(cmd, demux=False)
    except DockerException:
        return None
    if r.exit_code != 0:
        return None
    try:
        line = r.output.decode(errors="replace").strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        util, mem_used, mem_total, temp, power = parts[0], parts[1], parts[2], parts[3], parts[4]
        gpu_name = parts[5] if len(parts) > 5 else "GPU"
        return GpuStats(
            vendor="nvidia",
            name=gpu_name,
            util_pct=float(util),
            vram_used_gb=round(float(mem_used) / 1024.0, 1),
            vram_total_gb=round(float(mem_total) / 1024.0, 1),
            temp_c=float(temp),
            power_w=float(power),
        )
    except (ValueError, IndexError):
        return None


def _read_amd(container, container_name: str) -> GpuStats | None:
    cmd = "rocm-smi --showuse --showmemuse --showtemp --showpower --json"
    try:
        r = container.exec_run(cmd, demux=False)
    except DockerException:
        return None
    if r.exit_code != 0:
        return None
    try:
        data = json.loads(r.output.decode(errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not data:
        return None
    try:
        card_key, card = next(iter(data.items()))
    except StopIteration:
        return None
    total_gb = float(settings.gpu_vram_map.get(container_name, 0)) or 0.0
    try:
        vram_pct = float(card.get("GPU Memory Allocated (VRAM%)", 0) or 0)
        util = float(card.get("GPU use (%)", 0) or 0)
        temp = float(card.get("Temperature (Sensor edge) (C)", 0) or 0)
        power = float(card.get("Average Graphics Package Power (W)", 0) or 0)
    except ValueError:
        return None
    return GpuStats(
        vendor="amd",
        name=card_key,
        util_pct=util,
        vram_used_gb=round(total_gb * vram_pct / 100.0, 1),
        vram_total_gb=round(total_gb, 1),
        temp_c=temp,
        power_w=power,
    )


def _read_container_runtime(container) -> ContainerRuntimeStats | None:
    try:
        s = container.stats(stream=False)
    except DockerException:
        return None
    try:
        cpu = s.get("cpu_stats") or {}
        pre = s.get("precpu_stats") or {}
        cpu_delta = (cpu.get("cpu_usage") or {}).get("total_usage", 0) - (pre.get("cpu_usage") or {}).get("total_usage", 0)
        sys_delta = cpu.get("system_cpu_usage", 0) - pre.get("system_cpu_usage", 0)
        percpu = (cpu.get("cpu_usage") or {}).get("percpu_usage") or []
        online = cpu.get("online_cpus") or len(percpu) or 1
        cpu_pct = (cpu_delta / sys_delta) * online * 100.0 if sys_delta > 0 else 0.0

        mem = s.get("memory_stats") or {}
        stats_sub = mem.get("stats") or {}
        cache_bytes = int(stats_sub.get("cache") or stats_sub.get("inactive_file") or 0)
        mem_used = max(0, int(mem.get("usage", 0)) - cache_bytes)
        mem_limit = int(mem.get("limit") or 0)
        return ContainerRuntimeStats(
            cpu_pct=round(cpu_pct, 1),
            mem_used_gb=round(mem_used / (1024 ** 3), 2),
            mem_limit_gb=round(mem_limit / (1024 ** 3), 1) if mem_limit else 0.0,
        )
    except (KeyError, TypeError, ZeroDivisionError, ValueError):
        return None


def _collect(name: str) -> BackendStats:
    client = _client()
    if client is None:
        return BackendStats(ok=False, error="docker unreachable")
    try:
        c = client.containers.get(name)
    except NotFound:
        return BackendStats(ok=False, error="container not found")
    except DockerException as e:
        return BackendStats(ok=False, error=f"{type(e).__name__}: {e}")

    if c.status != "running":
        return BackendStats(ok=False, error=f"container is {c.status}")

    vendor = _detect_vendor(c)
    gpu = _read_nvidia(c) if vendor == "nvidia" else _read_amd(c, name) if vendor == "amd" else None
    cont = _read_container_runtime(c)
    return BackendStats(ok=True, gpu=gpu, container=cont)


def vram_gb_for(name: str) -> float:
    """Return total VRAM in GiB for a llama container.
    Prefers settings.gpu_vram_map, falls back to what the sampler has already probed."""
    m = settings.gpu_vram_map
    if name in m:
        return float(m[name])
    stats = stats_for(name)
    if stats.ok and stats.gpu and stats.gpu.vram_total_gb > 0:
        return float(stats.gpu.vram_total_gb)
    return 0.0


def stats_for(name: str) -> BackendStats:
    """Non-blocking. Returns cached stats populated by the background sampler."""
    with _LOCK:
        cached = _CACHE.get(name)
    if cached is not None:
        return cached
    return BackendStats(ok=False, error="warming up…")


def _sample_loop() -> None:
    # Local import to avoid circulars at module load
    from . import services
    while True:
        try:
            names = services._effective_container_names()
        except Exception:  # noqa: BLE001
            names = []
        for name in names:
            try:
                s = _collect(name)
            except Exception as e:  # noqa: BLE001
                s = BackendStats(ok=False, error=f"sampler error: {e}")
            with _LOCK:
                _CACHE[name] = s
        time.sleep(_SAMPLE_INTERVAL_S)


def start_sampler() -> None:
    global _sampler_started
    if _sampler_started:
        return
    _sampler_started = True
    t = threading.Thread(target=_sample_loop, daemon=True, name="hw-sampler")
    t.start()
