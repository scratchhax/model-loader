"""Every process holding VRAM, attributed per card and to the container that owns it.

The rest of the app only knows about llama backends, because they are the only thing it
starts. The cards do not care. Anything with /dev/kfd can hold VRAM - a TTS server, an image
generator, a stray benchmark - and until this module existed such a tenant reached the
overview only as an unexplained bulge in the "other" residual.

Worse, a SMEARED one. The per-card meter apportions what it cannot identify by each card's
measured usage, so a tenant sitting entirely on one card was drawn across both. Measured
2026-10-04 with chatterbox-tts holding 6.28 GB on card 0 and nothing at all on card 1, the
strip reported "other 3.42 GB" on card 0 and "other 2.58 GB" on card 1 - 2.58 GB of a
container's footprint attributed to a card it had not allocated a byte on.

The kernel publishes the answer, per process AND per card:

    /sys/class/kfd/kfd/proc/<host pid>/vram_<gpu id>     bytes

Three things make that usable here rather than merely present:

  - /sys is already mounted in this container, so there is no rocm-smi call and no docker
    exec in the path. It is a handful of small reads on a 1 s poll.
  - the gpu id resolves to the card index the rest of the app uses, via each topology node's
    `location_id` - which is the PCI bus, and PCI bus order is exactly what rocm-smi numbers
    card0/card1 by. Verified on this box: node 1 / gpu_id 23334 / location 0x300 is rocm-smi's
    card0 at 0000:03:00.0, node 2 / gpu_id 52525 / location 0x800 is card1 at 0000:08:00.0.
  - the pids are the HOST's, and this container cannot read /proc for them. But it holds the
    docker socket, and `docker top` reports host pids. That is the whole of the mapping.

One wrinkle, found by measurement rather than reasoning: the per-process `vram_*` files are
world-readable from any container, but the TOPOLOGY nodes beside them return EPERM unless the
reader holds /dev/kfd. Tested on this box - a plain container, `--cap-add SYS_ADMIN`,
`apparmor=unconfined` and `seccomp=unconfined` all fail; `--privileged` works, and so does any
ordinary container that was given `devices: [/dev/kfd, /dev/dri]`. So the map is read locally
when it can be and otherwise through one `docker exec` into a GPU container the app already
talks to. Topology is fixed for the life of the kernel, so that costs one exec per process,
not one per poll - and the alternative, handing this container the GPU just to read two
integers, buys nothing the socket it already holds does not.

AMD only, deliberately rather than by oversight. This is a KFD interface; the NVIDIA
equivalent is `nvidia-smi --query-compute-apps=pid,used_gpu_memory,gpu_uuid`, which is a
different shape (one row per process per device, keyed by uuid, and it reports nothing at all
for a process in a container it cannot see into). Nothing here has been tested against it, so
rather than ship a plausible-looking untested path this probe returns nothing on a box with no
KFD, and every caller is written to degrade to the behaviour it had before.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from docker.errors import DockerException

from .utils import docker_client, timed_shell

_KFD_PROC = Path("/sys/class/kfd/kfd/proc")
_KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")

# A process map changes only when something starts or stops, and asking docker costs one API
# call per running container. 30 s is for the slow drift (a container restarting under the
# same name); a pid this app has never seen refreshes immediately regardless, so a model that
# respawns is attributed on the very next poll rather than up to 30 s later.
_OWNER_TTL_S = 30.0
# Cooldown on re-probing an unresolved topology (see card_index_by_gpu_id).
_TOPO_RETRY_S = 60.0

_LOCK = threading.Lock()
_GPU_ID_MAP: dict[str, int] = {}
_TOPO_TRIED = 0.0
_OWNERS: tuple[float, dict[int, tuple[str, str]]] = (0.0, {})
_OWNER_SCANNING = False


@dataclass
class Tenant:
    """One process holding (or merely registered for) VRAM."""
    pid: int
    comm: str                 # process name, from docker top; "" when the owner is unknown
    container: str            # owning container, or "" for a host process
    backend: str              # the llama backend name when it is one of ours, else ""
    per_card_gb: dict[int, float] = field(default_factory=dict)
    # Peak compute units this process held per card inside the sampler's window, and the CUs a
    # card has in total. Both 0/empty on a box where the stats node cannot be read, which is
    # why every consumer must treat "no CU data" as unknown rather than as idle.
    cu_by_card: dict[int, int] = field(default_factory=dict)
    cu_total: int = 0
    # Milliseconds this process's memory has spent evicted, summed over cards. Climbs only when
    # something else took the VRAM, so a rising value is contention, not activity.
    evicted_ms: int = 0

    @property
    def total_gb(self) -> float:
        return sum(self.per_card_gb.values())

    @property
    def cu_peak(self) -> int:
        """Busiest single card, in CUs. 0 when it has done no measurable work."""
        return max(self.cu_by_card.values(), default=0)

    @property
    def working(self) -> bool:
        """Did this process actually run anything in the last few seconds?

        Deliberately NOT 'holds VRAM'. Chatterbox sits on ~6 GB around the clock and is doing
        nothing almost all of it; a panel that calls that 'in use' is the reason this field
        exists.
        """
        return self.cu_peak > 0

    @property
    def cu_pct(self) -> int:
        """Busiest card's occupancy as a percentage of one card's CUs. 0 when unknown."""
        if not self.cu_total:
            return 0
        return max(0, min(100, round(100.0 * self.cu_peak / self.cu_total)))

    @property
    def foreign(self) -> bool:
        """True when this is NOT a backend the app manages - the whole point of the module."""
        return not self.backend

    @property
    def label(self) -> str:
        return self.container or self.comm or f"pid {self.pid}"


# ---------------------------------------------------------------------------------------------
# Compute occupancy: the one per-process signal that says who is WORKING, not merely resident.
#
# /sys/class/kfd/kfd/proc/<pid>/stats_<gpuid>/cu_occupancy is compute units in use by that
# process on that card, right now. Measured on this box by driving chatterbox while watching
# both it and the llama servers:
#
#     card utilisation          card0 71%   card1 27%
#     pid 293730 python3        cu_occupancy max 34      <- chatterbox, correctly attributed
#     pid 2662870 llama-server  cu_occupancy max  0
#     pid 2664529 llama-server  cu_occupancy max  0
#
# It knows nothing about what the process is, which is the whole point: ComfyUI, a stray
# PyTorch script and a second llama-server all light up identically with no adapter.
#
# The catch is that it is INSTANTANEOUS. Over that 1.6 s TTS burst only 11 of ~50 samples read
# non-zero, so a 1 s poll would show a busy tenant as idle most of the time. Hence a background
# sampler at 20 Hz and a decaying max over _CU_WINDOW_S - the UI says "busiest in the last few
# seconds" rather than pretending this is a continuous reading.
_CU_HZ = 20.0
_CU_WINDOW_S = 3.0
_CU_PEAKS: dict[tuple[int, int], tuple[float, int]] = {}   # (pid, card) -> (when, cu)
_CU_LOCK = threading.Lock()
_CU_THREAD: threading.Thread | None = None


# simd_count / simd_per_cu for one GPU node - 128 / 2 = 64 on an R9700. Measured once and kept
# for the life of the process: it is a property of the silicon, and a card swap is a restart.
_CU_SH = (
    r"for n in /sys/class/kfd/kfd/topology/nodes/*/; do "
    r'g=$(cat "$n/gpu_id" 2>/dev/null); [ -n "$g" ] && [ "$g" != 0 ] || continue; '
    r"""s=$(awk '$1=="simd_count"{print $2; exit}' "$n/properties" 2>/dev/null); """
    r"""p=$(awk '$1=="simd_per_cu"{print $2; exit}' "$n/properties" 2>/dev/null); """
    r'[ "${s:-0}" -gt 0 ] && [ "${p:-0}" -gt 0 ] && { echo "$((s / p))"; break; }; done'
)
_CU_TOTAL: int | None = None


def _cu_per_card_local() -> int:
    try:
        for node in sorted(_KFD_NODES.glob("*/properties")):
            text = node.read_text()
            simd, per_cu = _node_prop(text, "simd_count"), _node_prop(text, "simd_per_cu")
            if simd and per_cu:
                return simd // per_cu
    except OSError:
        pass
    return 0


def cu_per_card(exec_candidates=()) -> int:
    """Compute units on one card, for the denominator. 0 when it cannot be determined.

    Local read first, then through a container that holds /dev/kfd - the same two-step
    card_index_by_gpu_id needs, and for the same reason: the topology nodes are EPERM to a
    container without the device, so this app reads 0 for every property from inside its own
    container and has to borrow a backend's view. Verified: readable on the host, all zeros
    from inside model-loader.
    """
    global _CU_TOTAL
    if _CU_TOTAL is not None:
        return _CU_TOTAL
    got = _cu_per_card_local()
    if not got and exec_candidates:
        try:
            client = docker_client()
            if client is None:
                return 0
            for name in exec_candidates:
                try:
                    c = client.containers.get(name)
                    if c.status != "running":
                        continue
                    code, out = c.exec_run(timed_shell(_CU_SH), demux=False)
                except Exception:  # noqa: BLE001 - a container without a shell is not the one
                    continue
                if code == 0:
                    text = out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out)
                    digits = "".join(ch for ch in text if ch.isdigit())
                    if digits:
                        got = int(digits)
                        break
        except (DockerException, OSError):
            got = 0
    # Only a real answer is cached. A 0 means no backend was up to ask yet, and that changes.
    if got:
        _CU_TOTAL = got
    return got


def _sample_cu(gpu_ids: dict[str, int]) -> None:
    """One pass over every KFD process, recording any non-zero occupancy with its timestamp."""
    now = time.time()
    try:
        procs = list(_KFD_PROC.iterdir())
    except OSError:
        return
    hits: list[tuple[tuple[int, int], tuple[float, int]]] = []
    for proc in procs:
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        for stats in proc.glob("stats_*"):
            idx = gpu_ids.get(stats.name[len("stats_"):])
            if idx is None:
                continue
            try:
                cu = int((stats / "cu_occupancy").read_text().strip() or 0)
            except (OSError, ValueError):
                continue
            if cu > 0:
                hits.append(((pid, idx), (now, cu)))
    if not hits:
        return
    with _CU_LOCK:
        for key, val in hits:
            prev = _CU_PEAKS.get(key)
            # Keep the larger reading while the window is still open; otherwise start fresh.
            if prev and (now - prev[0]) < _CU_WINDOW_S and prev[1] >= val[1]:
                continue
            _CU_PEAKS[key] = val


def _cu_loop(gpu_ids: dict[str, int]) -> None:
    period = 1.0 / _CU_HZ
    while True:
        _sample_cu(gpu_ids)
        with _CU_LOCK:
            cutoff = time.time() - _CU_WINDOW_S
            for k in [k for k, (when, _) in _CU_PEAKS.items() if when < cutoff]:
                del _CU_PEAKS[k]
        time.sleep(period)


def _ensure_cu_sampler(gpu_ids: dict[str, int]) -> None:
    """Start the sampler once, lazily. A daemon thread, so it never holds up a shutdown."""
    global _CU_THREAD
    if not gpu_ids:
        return
    with _CU_LOCK:
        if _CU_THREAD is not None and _CU_THREAD.is_alive():
            return
        _CU_THREAD = threading.Thread(target=_cu_loop, args=(dict(gpu_ids),),
                                      name="kfd-cu-sampler", daemon=True)
        _CU_THREAD.start()


def _cu_for(pid: int) -> dict[int, int]:
    """{card: peak CUs} this pid has used inside the window. Empty when it has done nothing."""
    cutoff = time.time() - _CU_WINDOW_S
    with _CU_LOCK:
        return {card: cu for (p, card), (when, cu) in _CU_PEAKS.items()
                if p == pid and when >= cutoff}


def _node_prop(text: str, key: str) -> int:
    for line in text.splitlines():
        k, _, v = line.partition(" ")
        if k == key:
            try:
                return int(v.strip() or 0)
            except ValueError:
                return 0
    return 0


# One shell line, run either nowhere (we read /sys ourselves) or inside a GPU container. It
# prints "<gpu id> <location id>" per node and stays silent for the CPU nodes, so the parser
# below is the same whichever side produced it.
_TOPO_SH = (
    r"for n in /sys/class/kfd/kfd/topology/nodes/*/; do "
    r'g=$(cat "$n/gpu_id" 2>/dev/null); [ -n "$g" ] && [ "$g" != 0 ] || continue; '
    r"""l=$(awk '$1=="location_id"{print $2; exit}' "$n/properties" 2>/dev/null); """
    r'echo "$g ${l:-0}"; done'
)


def _parse_topo(text: str) -> dict[str, int]:
    """Lines of "<gpu id> <location id>" -> {gpu id: card index}, ordered by PCI bus."""
    found: list[tuple[int, str]] = []
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        try:
            found.append((int(parts[1]), parts[0]))
        except ValueError:
            continue
    found.sort()
    return {gid: i for i, (_loc, gid) in enumerate(found)}


def _topo_local() -> dict[str, int]:
    """Read the topology directly. Works only when this container holds /dev/kfd."""
    lines = []
    try:
        for node in _KFD_NODES.iterdir():
            try:
                gid = (node / "gpu_id").read_text().strip()
                props = (node / "properties").read_text()
            except OSError:
                continue          # EPERM on every node is the normal case here
            if gid and gid != "0":
                lines.append("%s %d" % (gid, _node_prop(props, "location_id")))
    except OSError:
        return {}
    return _parse_topo("\n".join(lines))


def _topo_via_exec(candidates) -> dict[str, int]:
    """Read it through a container that does hold /dev/kfd. First one that answers wins."""
    client = docker_client()
    if client is None:
        return {}
    for name in candidates:
        try:
            c = client.containers.get(name)
            if c.status != "running":
                continue
            code, out = c.exec_run(timed_shell(_TOPO_SH), demux=False)
        except Exception:  # noqa: BLE001 - a container without a shell is just not the one
            continue
        if code != 0:
            continue
        got = _parse_topo(out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out))
        if got:
            return got
    return {}


def card_index_by_gpu_id(exec_candidates=()) -> dict[str, int]:
    """{kfd gpu id: card index}, matching rocm-smi's card numbering.

    Cached for the life of the process once it resolves - topology is fixed for the life of
    the kernel. While it has NOT resolved it is retried on a cooldown rather than every poll,
    because the usual reason is that no GPU container is up yet, and that fixes itself.
    """
    global _GPU_ID_MAP, _TOPO_TRIED
    with _LOCK:
        if _GPU_ID_MAP:
            return _GPU_ID_MAP
        if time.time() - _TOPO_TRIED < _TOPO_RETRY_S:
            return {}
        _TOPO_TRIED = time.time()
    out = _topo_local() or _topo_via_exec(exec_candidates)
    with _LOCK:
        if out:
            _GPU_ID_MAP = out
    return out


def _scan_owners() -> dict[int, tuple[str, str]] | None:
    """{host pid: (container name, process name)}, or None when docker is unreachable."""
    client = docker_client()
    if client is None:
        return None
    out: dict[int, tuple[str, str]] = {}
    try:
        containers = client.containers.list()
    except (DockerException, OSError):
        return None
    for c in containers:
        try:
            top = c.top(ps_args="-eo pid,comm")
        except Exception:  # noqa: BLE001 - a container that exited mid-list is not an error
            continue
        for row in (top or {}).get("Processes") or []:
            if len(row) < 2:
                continue
            try:
                out[int(row[0])] = (c.name, row[1].strip())
            except (TypeError, ValueError):
                continue
    return out


def _refresh_owners() -> None:
    global _OWNERS, _OWNER_SCANNING
    try:
        got = _scan_owners()
        with _LOCK:
            # A failed scan still stamps the clock, so an unreachable socket costs one attempt
            # per TTL rather than one per poll, and it keeps the names it already had.
            _OWNERS = (time.time(), _OWNERS[1] if got is None else got)
    finally:
        with _LOCK:
            _OWNER_SCANNING = False


def _owner_map(want: set[int]) -> dict[int, tuple[str, str]]:
    """{host pid: (container name, process name)} for every process in a running container.

    Rebuilt when the cache is stale OR when `want` holds a pid it does not know, so a process
    that has just appeared is named rather than waiting out the TTL. A pid that is genuinely
    not in any container - something started on the host - stays absent and cannot force more
    than one rebuild per TTL, because the staleness check runs first.

    The rebuild is one docker API call PER RUNNING CONTAINER: measured 88-181 ms across the 13
    on this box, which is a long time to hold the event loop on a route that polls at 1 s. So
    it runs on a thread and the caller keeps the previous map, except for the very first one,
    where there is nothing to keep and an unnamed first render would be worse than a pause.
    The cost of being a refresh behind is that a brand-new process is anonymous for one poll.
    """
    global _OWNER_SCANNING
    with _LOCK:
        ts, cached = _OWNERS
        if (time.time() - ts) < _OWNER_TTL_S and not (want - set(cached)):
            return cached
        if not ts:
            _OWNER_SCANNING = True      # first call: block, so nothing renders anonymous
            blocking = True
        elif _OWNER_SCANNING:
            return cached               # a rebuild is already in flight
        else:
            _OWNER_SCANNING = True
            blocking = False
    if blocking:
        _refresh_owners()
        with _LOCK:
            return _OWNERS[1]
    threading.Thread(target=_refresh_owners, name="gpu-owner-scan", daemon=True).start()
    return cached


def tenants(backend_names=()) -> list[Tenant]:
    """Every GPU process, largest first. Empty on a box with no KFD (so, on NVIDIA).

    `backend_names` is what the app manages; it both tags each tenant and supplies the
    candidates for the one-off topology exec, since a llama backend is by definition a
    container holding /dev/kfd.

    Processes holding zero bytes are INCLUDED. A llama child that has gone to sleep under
    `--sleep-idle-seconds` is exactly that: registered with the driver, holding nothing. It is
    a fact worth seeing on the overview, and dropping it here would mean the caller could not
    tell "asleep" from "gone".
    """
    known = set(backend_names or ())
    gpu_ids = card_index_by_gpu_id(sorted(known))
    if not gpu_ids:
        return []

    raw: list[tuple[int, dict[int, float]]] = []
    try:
        procs = list(_KFD_PROC.iterdir())
    except OSError:
        return []
    for proc in procs:
        if not proc.name.isdigit():
            continue
        per_card: dict[int, float] = {}
        for f in proc.glob("vram_*"):
            idx = gpu_ids.get(f.name[len("vram_"):])
            if idx is None:
                continue
            try:
                per_card[idx] = int(f.read_text().strip() or 0) / (1024 ** 3)
            except (OSError, ValueError):
                continue
        if per_card:
            raw.append((int(proc.name), per_card))

    # Start the occupancy sampler on the first real scan rather than at import: it needs the
    # gpu-id map, and on a box with no KFD there is nothing to sample and no thread to leak.
    _ensure_cu_sampler(gpu_ids)
    cu_total = cu_per_card(sorted(known))

    owners = _owner_map({pid for pid, _ in raw})
    out = [
        Tenant(pid=pid, comm=comm, container=cont,
               backend=cont if cont in known else "", per_card_gb=per_card,
               cu_by_card=_cu_for(pid), cu_total=cu_total,
               evicted_ms=_evicted_ms(pid))
        for pid, per_card in raw
        for cont, comm in (owners.get(pid, ("", "")),)
    ]
    out.sort(key=lambda t: (-t.total_gb, t.pid))
    return out


def _evicted_ms(pid: int) -> int:
    """Total ms this process's memory has been evicted, over all cards. 0 when unreadable."""
    total = 0
    try:
        for f in (_KFD_PROC / str(pid)).glob("stats_*/evicted_ms"):
            try:
                total += int(f.read_text().strip() or 0)
            except (OSError, ValueError):
                continue
    except OSError:
        return 0
    return total


def foreign_by_card(ts: list[Tenant], floor_gb: float = 0.01) -> dict[int, list[tuple[str, float]]]:
    """{card index: [(label, GB)]} for tenants the app does not manage.

    Only these need naming in a meter: a backend's own allocation is what the meter already
    itemises into weights, KV and compute, so listing it again as a tenant would double-count
    it. The floor drops a registered-but-empty process, which is a row in the tenant table and
    not a segment in a bar.
    """
    by_card: dict[int, list[tuple[str, float]]] = {}
    for t in ts:
        if not t.foreign:
            continue
        for idx, gb in sorted(t.per_card_gb.items()):
            if gb >= floor_gb:
                by_card.setdefault(idx, []).append((t.label, gb))
    return by_card


# Last VRAM each foreign tenant was seen holding, so a STOPPED one can still say what it will
# want back. It vanishes from the kernel's tables the moment it stops, which is exactly when
# that number becomes interesting.
_LAST_SEEN: dict[str, float] = {}


def note_seen(ts: list[Tenant]) -> None:
    for t in ts:
        if t.foreign and t.container and t.total_gb > 0.005:
            _LAST_SEEN[t.container] = t.total_gb


def gpu_capable_containers(backend_names=()) -> list[dict]:
    """Containers that were given a GPU but are NOT llama backends, running or stopped.

    Discovered from the device list docker was told to pass through, not from what is currently
    on the cards, because the whole point is to describe something that has been ejected - a
    stopped container holds no VRAM and appears in no kernel table.

    /dev/kfd is the AMD compute device; a container with it can run compute, and one without it
    cannot, which makes it a better test than the image name or a label someone has to remember
    to set.
    """
    known = set(backend_names or ())
    out: list[dict] = []
    client = docker_client()
    if client is None:
        return []
    try:
        containers = client.containers.list(all=True)
    except (DockerException, OSError):
        return []
    for c in containers:
        if c.name in known:
            continue
        try:
            devs = (c.attrs.get("HostConfig") or {}).get("Devices") or []
            paths = {str(d.get("PathOnHost", "")) for d in devs}
        except Exception:  # noqa: BLE001 - a container mid-removal is not an error
            continue
        if "/dev/kfd" not in paths:
            continue
        out.append({"name": c.name, "running": c.status == "running",
                    "status": c.status, "last_gb": _LAST_SEEN.get(c.name, 0.0)})
    out.sort(key=lambda r: r["name"])
    return out


def set_container_running(name: str, start: bool) -> tuple[bool, str]:
    """Start or stop one container. Returns (ok, message)."""
    try:
        client = docker_client()
        if client is None:
            return False, f"{name}: docker unreachable"
        c = client.containers.get(name)
        if start:
            if c.status == "running":
                return True, f"{name} is already running"
            c.start()
            return True, f"{name} starting - it reloads its weights, give it a moment"
        if c.status != "running":
            return True, f"{name} is already stopped"
        # No explicit timeout: an explicit one is sent as ?t= and OVERRIDES the container's
        # own StopTimeout, so a hardcoded value here silently ignores whatever the service
        # declared in compose. Each service knows its own shutdown cost; this does not.
        #
        # It matters more than it looks. comfyui's PID 1 is the image's bash entrypoint, which
        # never execs, and bash defers a signal until its foreground child finishes - so python
        # is never told to stop and the full grace period always elapses before SIGKILL. With 30
        # hardcoded here, ejecting it took a measured 37 s for a container that had nothing to
        # flush. Its compose entry sets stop_grace_period: 5s, which is only honoured if we stay
        # out of the way.
        c.stop()
        return True, f"{name} stopped and its VRAM released"
    except (DockerException, OSError) as e:
        return False, f"{name}: {e}"

