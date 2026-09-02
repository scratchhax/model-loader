from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace

import docker
from docker.errors import DockerException, NotFound

from .config import settings


@dataclass
class GpuCard:
    """One physical GPU. Kept alongside the aggregate because the aggregate hides the
    thing that actually matters on a layer-split multi-GPU box: pooled VRAM can look
    roomy while ONE card is full. Non-split allocations (mmproj, compute buffers,
    cuBLAS workspace) land on the main GPU, so OOMs are routinely device-specific."""
    index: int
    name: str
    util_pct: float
    vram_used_gb: float
    vram_total_gb: float
    temp_c: float
    power_w: float

    @property
    def vram_free_gb(self) -> float:
        return round(max(0.0, self.vram_total_gb - self.vram_used_gb), 2)

    @property
    def vram_pct(self) -> float:
        return round(100.0 * self.vram_used_gb / self.vram_total_gb, 1) if self.vram_total_gb else 0.0


@dataclass
class GpuStats:
    vendor: str
    name: str
    util_pct: float
    vram_used_gb: float
    vram_total_gb: float
    temp_c: float
    power_w: float
    gpu_count: int = 1
    cards: list[GpuCard] = field(default_factory=list)  # per-device detail


@dataclass
class HistoryPoint:
    ts: float
    gpu_util: float
    vram_used_gb: float
    cpu_pct: float
    mem_used_gb: float
    per_gpu_util: list[float] = field(default_factory=list)
    per_gpu_vram_used_gb: list[float] = field(default_factory=list)


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
# Rolling in-memory history for sparklines. 450 samples at the 2s interval = 15 minutes.
# Deliberately not persisted: this is a live view, not telemetry, and a bounded deque
# per backend costs a few KB and can never grow without limit.
_HISTORY_MAXLEN = 450
_HISTORY: dict[str, deque] = {}
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
    # Vendor decides which probe to run, and it is read off the image tag because that is
    # the only thing available without exec'ing into a container that may not be running.
    if "rocm" in image or "hip" in image:
        vendor = "amd"
    elif "cuda" in image or "nvidia" in image:
        vendor = "nvidia"
    elif "vulkan" in image:
        # Vulkan is a rendering API, not a vendor: the card underneath may be AMD, NVIDIA or
        # Intel, and the image ships neither rocm-smi nor nvidia-smi as a rule. Treated as its
        # own case so it can try both probes and then degrade to a declared VRAM figure,
        # rather than falling into "unknown" and reporting no GPU at all.
        vendor = "vulkan"
    else:
        vendor = "unknown"
    with _LOCK:
        _VENDOR_CACHE[container.name] = vendor
    return vendor


def _read_nvidia(container) -> GpuStats | None:
    """Read NVIDIA GPU stats. When a container sees N GPUs, aggregate:
       - util: average across cards
       - VRAM used / total: sum
       - temperature: max (hottest card sets the throttle)
       - power: sum
       - name: shown as "N × <name>" if all same, else "<name>+<name>+..."
    """
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
        lines = [ln for ln in r.output.decode(errors="replace").strip().splitlines() if ln.strip()]
        if not lines:
            return None
        utils, mems_used, mems_total, temps, powers, names = [], [], [], [], [], []
        cards: list[GpuCard] = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            utils.append(float(parts[0]))
            mems_used.append(float(parts[1]))
            mems_total.append(float(parts[2]))
            temps.append(float(parts[3]))
            powers.append(float(parts[4]))
            names.append(parts[5] if len(parts) > 5 else "GPU")
            cards.append(GpuCard(
                index=len(cards), name=names[-1],
                util_pct=utils[-1],
                vram_used_gb=round(mems_used[-1] / 1024.0, 2),
                vram_total_gb=round(mems_total[-1] / 1024.0, 2),
                temp_c=temps[-1], power_w=powers[-1],
            ))
        if not utils:
            return None
        # name: "N × X" when homogeneous, else joined
        if len(set(names)) == 1:
            display_name = f"{len(names)} × {names[0]}" if len(names) > 1 else names[0]
        else:
            display_name = " + ".join(names)
        return GpuStats(
            vendor="nvidia",
            name=display_name,
            util_pct=round(sum(utils) / len(utils), 1),
            vram_used_gb=round(sum(mems_used) / 1024.0, 1),
            vram_total_gb=round(sum(mems_total) / 1024.0, 1),
            temp_c=max(temps),
            power_w=round(sum(powers), 1),
            gpu_count=len(utils),
            cards=cards,
        )
    except (ValueError, IndexError):
        return None


def _read_vulkan(container, container_name: str) -> GpuStats | None:
    """Best-effort stats for a Vulkan backend.

    Vulkan says nothing about the vendor underneath, and the llama.cpp Vulkan image carries
    no vendor SMI tool, so there is usually nothing to query. Order of attempts:

      1. rocm-smi, then nvidia-smi — occasionally present on images built on a vendor base,
         and if either answers we get real utilisation for free.
      2. The GPU_VRAM declaration. No utilisation, temperature or power, but it makes
         vram_total_gb non-zero, which is the number Autoconfig actually needs. Without it
         Autoconfig sees a 0 GB budget and reports "doesn't fit at any context" for every
         model, which reads as a broken app rather than a missing setting.

    Returns None only when we have neither a probe nor a declared size — the caller then
    surfaces the "declare GPU_VRAM" message instead of silently showing nothing.
    """
    probed = _read_amd(container, container_name) or _read_nvidia(container)
    if probed is not None:
        return replace(probed, vendor="vulkan")

    declared = float(settings.gpu_vram_map.get(container_name, 0) or 0)
    if declared <= 0:
        return None
    return GpuStats(
        vendor="vulkan",
        name="Vulkan device (no SMI tool in image — VRAM from GPU_VRAM)",
        util_pct=0.0,
        vram_used_gb=0.0,
        vram_total_gb=round(declared, 1),
        temp_c=0.0,
        power_w=0.0,
    )


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
    if vendor == "nvidia":
        gpu = _read_nvidia(c)
    elif vendor == "amd":
        gpu = _read_amd(c, name)
    elif vendor == "vulkan":
        gpu = _read_vulkan(c, name)
    else:
        gpu = None
    cont = _read_container_runtime(c)
    return BackendStats(ok=True, gpu=gpu, container=cont)


def gpu_count_for(name: str) -> int:
    """Return the number of GPUs visible inside the named container. Defaults to 1 if unknown."""
    stats = stats_for(name)
    if stats.ok and stats.gpu and stats.gpu.gpu_count > 0:
        return stats.gpu.gpu_count
    return 1


def card_vram_gb_for(name: str) -> list[float]:
    """Per-card total VRAM (GiB) for a llama container, in device order.

    Pooled VRAM is the wrong number for a layer-split box: llama.cpp must place each layer
    on ONE card, so a config can fit the pool comfortably and still OOM a single device.
    Returns [] when the sampler has not probed this container yet, in which case callers
    should fall back to dividing the pooled figure evenly.
    """
    stats = stats_for(name)
    if stats.ok and stats.gpu and stats.gpu.cards:
        return [c.vram_total_gb for c in stats.gpu.cards]
    return []


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


def _record_history(name: str, s: BackendStats) -> None:
    """Append one sample. Called from the sampler only; caller holds no lock."""
    if not s.ok:
        return
    g, c = s.gpu, s.container
    pt = HistoryPoint(
        ts=time.time(),
        gpu_util=g.util_pct if g else 0.0,
        vram_used_gb=g.vram_used_gb if g else 0.0,
        cpu_pct=c.cpu_pct if c else 0.0,
        mem_used_gb=c.mem_used_gb if c else 0.0,
        per_gpu_util=[card.util_pct for card in (g.cards if g else [])],
        per_gpu_vram_used_gb=[card.vram_used_gb for card in (g.cards if g else [])],
    )
    with _LOCK:
        buf = _HISTORY.get(name)
        if buf is None:
            buf = _HISTORY[name] = deque(maxlen=_HISTORY_MAXLEN)
        buf.append(pt)


def history_for(name: str) -> list[HistoryPoint]:
    with _LOCK:
        return list(_HISTORY.get(name) or ())


def sparkline(values: list[float], max_hint: float | None = None) -> str:
    """SVG polyline points in a 100x20 viewBox. Matches the download-row sparkline style.

    max_hint pins the vertical scale (e.g. 100 for a percentage, or total VRAM) so the
    line means the same thing frame to frame. Without it a flat-but-busy metric would
    rescale every tick and look like wild swings.
    """
    if len(values) < 2:
        return ""
    top = max_hint if max_hint else (max(values) or 1.0)
    if top <= 0:
        top = 1.0
    n = len(values)
    return " ".join(
        f"{i * (100 / (n - 1)):.1f},{20 - min(1.0, v / top) * 18:.1f}"
        for i, v in enumerate(values)
    )


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
        discovery_ok = True
        try:
            names = services._effective_container_names()
        except Exception:  # noqa: BLE001
            names, discovery_ok = [], False
        for name in names:
            try:
                s = _collect(name)
            except Exception as e:  # noqa: BLE001
                s = BackendStats(ok=False, error=f"sampler error: {e}")
            with _LOCK:
                _CACHE[name] = s
            try:
                _record_history(name, s)
            except Exception:  # noqa: BLE001 — history is cosmetic, never break sampling
                pass
        # Drop history for backends that have genuinely gone away, so a removed container's
        # buffer doesn't sit around for the life of the process.
        #
        # Only prune when discovery actually succeeded AND returned something. A transient
        # docker-socket failure yields an empty list that is indistinguishable from "all
        # containers removed" — pruning on that would wipe every backend's cache and its
        # entire 15-minute history because the socket blipped for one 2s tick.
        if discovery_ok and names:
            with _LOCK:
                for gone in [k for k in _HISTORY if k not in names]:
                    _HISTORY.pop(gone, None)
                    _CACHE.pop(gone, None)
        time.sleep(_SAMPLE_INTERVAL_S)


def start_sampler() -> None:
    global _sampler_started
    if _sampler_started:
        return
    _sampler_started = True
    t = threading.Thread(target=_sample_loop, daemon=True, name="hw-sampler")
    t.start()


def host_ram_gb() -> float:
    """Total host RAM in GiB, or 0.0 if it cannot be read.

    Read from /proc/meminfo, which inside a container reports the HOST's memory unless
    something like lxcfs is masking it. Needed as a ceiling: CPU-offloaded layers live in
    system RAM, so a model larger than VRAM + RAM cannot run at any offload setting, and
    offering it a context estimate is worse than saying nothing.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    return 0.0
