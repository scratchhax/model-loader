from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
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
_SAMPLE_INTERVAL_S = 1.0   # rocm-smi/nvidia-smi through docker exec costs ~90 ms, so 1 s is
                           # a ~9% duty cycle - cheap enough for a live readout, and the
                           # dashboard cannot show anything fresher than this.
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


def _amd_num(card: dict, include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> float:
    """First numeric value in a rocm-smi card dict whose key matches, or 0.0.

    Keys are matched by substring rather than spelled out because rocm-smi renames them between
    releases: power has been "Average Graphics Package Power (W)" and "Current Socket Graphics
    Package Power (W)", and VRAM totals have moved between --showmemuse and --showmeminfo. A
    literal lookup silently returns zero on the wrong version, which then reads as "this backend
    has no VRAM" and stops autoconfig dead.
    """
    for key, val in card.items():
        k = key.lower()
        if all(n in k for n in include) and not any(x in k for x in exclude):
            try:
                return float(str(val).strip())
            except (TypeError, ValueError):
                continue
    return 0.0


def _amd_name(card_key: str, card: dict) -> str:
    """A useful label for an AMD card, with its ISA when the marketing name is vague.

    rocm-smi reports "AMD Radeon Graphics" for anything its PCI-ID table predates - on this rig
    that is a pair of Radeon AI PRO R9700s, which the container's older ROCm userspace does not
    recognise even though the host's newer amd-smi does. The gfx target is the more useful fact
    anyway: gfx1201 is what decides whether a ROCm build has kernels for the card at all.
    """
    name = ""
    for key in ("Card Series", "Card Model", "Device Name", "Card SKU", "Market Name"):
        v = str(card.get(key) or "").strip()
        if v and v.lower() not in ("n/a", "unknown"):
            name = v
            break
    gfx = str(card.get("GFX Version") or "").strip()
    if gfx and gfx.lower() not in ("n/a", "unknown"):
        return f"{name} ({gfx})" if name else gfx
    return name or card_key


def _read_amd(container, container_name: str) -> GpuStats | None:
    """Per-card AMD stats, aggregated the same way as NVIDIA.

    Reads EVERY card, not just the first. The earlier version took `next(iter(data.items()))`
    and sourced VRAM only from the GPU_VRAM override, so on a two-card ROCm box the sampler
    reported one device and no per-card capacities - which quietly disables the per-card fit
    check in autoconfig. That check is the thing standing between a config that clears the
    pooled budget and one that OOMs a single device, so on AMD the most dangerous class of
    mis-sizing was invisible.

    VRAM comes from the hardware when rocm-smi reports it, and falls back to the GPU_VRAM
    override (split across the cards) only when it does not.
    """
    # --showmeminfo gives byte-accurate totals; the rest are cheap. Older rocm-smi builds reject
    # unknown flags outright, so fall back to the minimal set it has always understood.
    cmds = (
        "rocm-smi --showid --showproductname --showuse --showmemuse "
        "--showmeminfo vram --showtemp --showpower --json",
        "rocm-smi --showuse --showmemuse --showtemp --showpower --json",
    )
    data: dict | None = None
    for cmd in cmds:
        try:
            r = container.exec_run(cmd, demux=False)
        except DockerException:
            return None
        if r.exit_code != 0:
            continue
        try:
            parsed = json.loads(r.output.decode(errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(parsed, dict) and parsed:
            data = parsed
            break
    if not data:
        return None

    # rocm-smi keys cards as "card0", "card1", ...; sort numerically so device order is stable.
    def _idx(key: str) -> int:
        digits = "".join(ch for ch in key if ch.isdigit())
        return int(digits) if digits else 0

    items = sorted(((k, v) for k, v in data.items() if isinstance(v, dict)), key=lambda kv: _idx(kv[0]))
    if not items:
        return None

    env_total = float(settings.gpu_vram_map.get(container_name, 0) or 0.0)
    per_card_env = (env_total / len(items)) if env_total else 0.0

    cards: list[GpuCard] = []
    for i, (card_key, card) in enumerate(items):
        total_b = _amd_num(card, ("vram", "total"), exclude=("used", "%"))
        used_b = _amd_num(card, ("vram", "used"))
        total_gb = round(total_b / (1024 ** 3), 2) if total_b else round(per_card_env, 2)
        used_gb = round(used_b / (1024 ** 3), 2) if used_b else 0.0
        if not used_gb:
            # No byte figure: derive from the percentage this build does report.
            pct = _amd_num(card, ("vram", "%"))
            used_gb = round(total_gb * pct / 100.0, 2) if (pct and total_gb) else 0.0
        util = _amd_num(card, ("gpu use",)) or _amd_num(card, ("gfx", "activity"))
        temp = _amd_num(card, ("temperature", "edge")) or _amd_num(card, ("temperature", "junction"))
        power = _amd_num(card, ("power",), exclude=("cap", "limit", "max"))
        cards.append(GpuCard(index=i, name=_amd_name(card_key, card), util_pct=util,
                             vram_used_gb=used_gb, vram_total_gb=total_gb,
                             temp_c=temp, power_w=power))

    names = [c.name for c in cards]
    display = (f"{len(names)} × {names[0]}" if len(set(names)) == 1 and len(names) > 1
               else (names[0] if len(names) == 1 else " + ".join(names)))
    return GpuStats(
        vendor="amd",
        name=display,
        util_pct=round(sum(c.util_pct for c in cards) / len(cards), 1),
        vram_used_gb=round(sum(c.vram_used_gb for c in cards), 1),
        vram_total_gb=round(sum(c.vram_total_gb for c in cards), 1),
        temp_c=max(c.temp_c for c in cards),
        power_w=round(sum(c.power_w for c in cards), 1),
        gpu_count=len(cards),
        cards=cards,
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


# ---------------------------------------------------------------- host sensors + power roll-up

# Rolling history for the two things the dashboard draws. 240 samples at the panel's 1s poll is
# four minutes - long enough to see a job start and the fans catch up.
_POWER_HISTORY_MAX = 240
_POWER_MIN_GAP_S = 0.5
_power_history: list[tuple[float, float, float, str]] = []   # (ts, wall_w, temp_c, temp_label)


def power_history() -> tuple[list[float], list[float]]:
    """(wall watts, hottest temperature) series, oldest first."""
    return [w for _t, w, _c, _l in _power_history], [c for _t, _w, c, _l in _power_history]


_HOST_SENSORS_PATH = Path("/data/host_sensors.json")
_HOST_SENSORS_MAX_AGE_S = 15.0     # a stale file is worse than no file


def host_sensors() -> dict:
    """CPU package power and host temperatures, published by host-sensors.service.

    Empty when the file is missing or stale: RAPL is root-only and invisible inside a container,
    so if that unit is not running this genuinely cannot be known, and guessing would undermine
    the one number the page exists to provide.
    """
    try:
        raw = json.loads(_HOST_SENSORS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    ts = float(raw.get("ts") or 0)
    if ts and (time.time() - ts) > _HOST_SENSORS_MAX_AGE_S:
        return {"stale": True, "age_s": round(time.time() - ts, 1)}
    return raw


def power_rollup(baseline_w: float, psu_efficiency: float) -> dict:
    """What the machine is drawing, split into measured parts and a stated allowance.

    GPU figures come from the same sampler the dashboard already uses, so the cards' numbers
    here and on the GPU strip cannot disagree.
    """
    from . import services
    gpu_w = 0.0
    gpu_cards: list[dict] = []
    gpu_temp_max = 0.0
    for name in services._effective_container_names():
        st = stats_for(name)
        if not (st.ok and st.gpu and st.gpu.cards):
            continue
        for c in st.gpu.cards:
            gpu_w += c.power_w
            gpu_temp_max = max(gpu_temp_max, c.temp_c)
            gpu_cards.append({"index": c.index, "watts": round(c.power_w, 1),
                              "temp_c": round(c.temp_c, 1)})
        break

    host = host_sensors()
    cpu_w = host.get("cpu_package_w")
    # Hottest device, named: "78 C" is not actionable unless you know whether it is a GPU
    # junction, the CPU package or an NVMe controller baking under the cards.
    hottest_c, hottest_label = 0.0, ""
    for label, value in (("GPU", gpu_temp_max), ("CPU", host.get("cpu_temp_c")),
                         ("NVMe", host.get("nvme_temp_max_c"))):
        try:
            v = float(value or 0)
        except (TypeError, ValueError):
            continue
        if v > hottest_c:
            hottest_c, hottest_label = v, label
    measured_w = gpu_w + (cpu_w or 0.0)
    dc_w = measured_w + max(0.0, baseline_w)
    eff = psu_efficiency if 0.5 <= psu_efficiency <= 1.0 else 0.9
    wall_w = dc_w / eff
    now = time.time()
    if not _power_history or (now - _power_history[-1][0]) >= _POWER_MIN_GAP_S:
        _power_history.append((now, round(wall_w, 1), round(hottest_c, 1), hottest_label))
        if len(_power_history) > _POWER_HISTORY_MAX:
            del _power_history[:len(_power_history) - _POWER_HISTORY_MAX]

    return {
        "hottest_c": round(hottest_c, 1) if hottest_c else None,
        "hottest_label": hottest_label,
        "gpu_w": round(gpu_w, 1),
        "gpu_cards": gpu_cards,
        "gpu_temp_max_c": round(gpu_temp_max, 1) if gpu_temp_max else None,
        "cpu_w": round(cpu_w, 1) if cpu_w is not None else None,
        "cpu_temp_c": host.get("cpu_temp_c"),
        "nvme_temp_max_c": host.get("nvme_temp_max_c"),
        "baseline_w": round(max(0.0, baseline_w), 1),
        "measured_w": round(measured_w, 1),
        "wall_w": round(wall_w, 1),
        "psu_efficiency": eff,
        "host_ok": bool(host) and not host.get("stale"),
        "host_stale": bool(host.get("stale")),
    }


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


def usable_ram_gb() -> float:
    """Host RAM minus the reserve the OS and everything else need.

    This, not MemTotal, is what a model may actually occupy. Sizing against total RAM plans
    for memory that is already spoken for by the page cache, the other containers on the box
    and the kernel itself — the result loads, then swaps or gets OOM-killed.

    Reserve defaults to 32 GB and is set with HOST_RAM_RESERVE_GB.
    """
    total = host_ram_gb()
    if total <= 0:
        return 0.0
    return round(max(0.0, total - float(settings.host_ram_reserve_gb)), 1)
