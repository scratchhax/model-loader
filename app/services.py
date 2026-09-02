from __future__ import annotations

import asyncio
import configparser
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import docker
import httpx
from docker.errors import APIError, DockerException, NotFound

from . import ini
from .config import settings
from .utils import human_bytes, shard_key


# ---------- models directory ----------

@dataclass
class GgufEntry:
    display_name: str        # shown to user (base name, without shard suffix if grouped)
    parts: list[Path]        # one entry, or many shards sorted by index
    total_bytes: int
    mtime: float             # newest part's mtime
    is_sharded: bool
    subdir: str = ""         # empty for flat files, subdir name for shard groups in a subdir
    is_companion: bool = False   # True for mmproj / other files referenced from a main model's section
    aliases: list[str] = field(default_factory=list)  # ini section names that reference this
    # If this is a main model with a companion mmproj in the same subdir, these fields
    # describe the companion. Standalone companion entries are hidden from listings.
    companion_name: str = ""
    companion_bytes: int = 0
    companion_parts: list[Path] = field(default_factory=list)

    @property
    def human_size(self) -> str:
        return human_bytes(self.total_bytes)

    @property
    def modified(self) -> str:
        return datetime.fromtimestamp(self.mtime, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

    @property
    def stem(self) -> str:
        # e.g. "gemma-4-12b-it-Q4_K_M.gguf" -> "gemma-4-12b-it-Q4_K_M"
        return self.display_name[:-5] if self.display_name.lower().endswith(".gguf") else self.display_name

    @property
    def model_id(self) -> str:
        """The id llama-server serves this file under.

        That is its ini section name when it has one — which is NOT necessarily the filename
        stem, since a section can be renamed to give the model a short API id — and otherwise
        the stem, which is what llama-server falls back to.
        """
        if self.stem in self.aliases:
            return self.stem      # a section named after the file: the natural id
        return self.aliases[0] if self.aliases else self.stem

    @property
    def route_key(self) -> str:
        """URL-safe key for /model/<key> routes; for subdir'd entries this includes the subdir."""
        return f"{self.subdir}/{self.display_name}" if self.subdir else self.display_name

    @property
    def first_shard_rel(self) -> str:
        """Relative path (from models_dir) to the first shard/file. Used in ini `model = ...`."""
        first = self.parts[0].name
        return f"{self.subdir}/{first}" if self.subdir else first


@dataclass
class DiskInfo:
    total: int
    free: int
    used_pct: float

    @property
    def total_h(self) -> str: return human_bytes(self.total)
    @property
    def free_h(self) -> str: return human_bytes(self.free)


@dataclass
class ModelsDirSnapshot:
    path: Path
    exists: bool
    error: str | None = None
    disk: DiskInfo | None = None
    ggufs: list[GgufEntry] = field(default_factory=list)
    ini_aliases: list[str] = field(default_factory=list)
    ini_present: bool = False


def read_ini_aliases(ini_path: Path) -> list[str]:
    if not ini_path.exists():
        return []
    try:
        cp = configparser.ConfigParser(strict=False)
        cp.read(ini_path, encoding="utf-8")
        return cp.sections()
    except (OSError, configparser.Error):
        return []


def snapshot_models_dir() -> ModelsDirSnapshot:
    path = settings.models_dir
    snap = ModelsDirSnapshot(path=path, exists=path.exists())
    if not path.exists():
        snap.error = "directory not found (is the volume mounted?)"
        return snap

    try:
        u = shutil.disk_usage(path)
        snap.disk = DiskInfo(total=u.total, free=u.free, used_pct=round(100 * (u.total - u.free) / u.total, 1))
    except OSError as e:
        snap.error = f"disk usage failed: {e}"

    aliases = read_ini_aliases(settings.models_ini_path)
    snap.ini_aliases = aliases
    # file -> section, so a section renamed to a short API id still owns its GGUF
    by_file = ini.sections_by_file()
    snap.ini_present = settings.models_ini_path.exists()

    # collect gguf files, grouping shards. Also descend one level into subdirs
    # (multi-shard downloads land in /models/<base>/ to keep the top level tidy).
    # Key = (subdir, shard_base). subdir="" for flat.
    groups: dict[tuple[str, str], list[Path]] = {}
    try:
        for p in path.iterdir():
            if p.is_file() and p.suffix.lower() == ".gguf":
                base, _, _ = shard_key(p.name)
                groups.setdefault(("", base), []).append(p)
            elif p.is_dir() and not p.name.startswith("."):
                try:
                    subparts = list(p.iterdir())
                except OSError:
                    continue
                for sp in subparts:
                    if sp.is_file() and sp.suffix.lower() == ".gguf":
                        base, _, _ = shard_key(sp.name)
                        groups.setdefault((p.name, base), []).append(sp)
    except OSError as e:
        snap.error = f"listing failed: {e}"
        return snap

    entries: list[GgufEntry] = []
    for (subdir, base), parts in groups.items():
        parts.sort(key=lambda pp: pp.name)
        total = sum(pp.stat().st_size for pp in parts)
        mtime = max(pp.stat().st_mtime for pp in parts)
        stem_no_ext = base[:-5] if base.lower().endswith(".gguf") else base
        rel = f"{subdir}/{parts[0].name}" if subdir else parts[0].name
        matched = by_file.get(rel) or [a for a in aliases if a == stem_no_ext]
        is_comp = "mmproj" in base.lower()
        entries.append(GgufEntry(
            display_name=base,
            parts=parts,
            total_bytes=total,
            mtime=mtime,
            is_sharded=len(parts) > 1,
            subdir=subdir,
            is_companion=is_comp,
            aliases=matched,
        ))

    # Fold companion mmproj entries into their same-subdir main model.
    # A companion is a standalone file that only makes sense paired with its main model:
    # you can't run it alone, can't configure it, can't do anything with it. So we hide
    # it from the list and expose it as a badge on the main model instead. Un-paired
    # companions (mmproj in a subdir with no main model) stay visible for cleanup.
    main_by_subdir: dict[str, GgufEntry] = {
        e.subdir: e for e in entries if not e.is_companion and e.subdir
    }
    surviving: list[GgufEntry] = []
    for e in entries:
        if e.is_companion and e.subdir and e.subdir in main_by_subdir:
            main = main_by_subdir[e.subdir]
            # ACCUMULATE. A subdir can hold several projectors (mmproj-BF16 + mmproj-F32 is
            # a common upload pattern). Assigning here instead of appending meant the second
            # one overwrote the first, so deleting the model stranded a projector on disk and
            # left the subdir non-empty -- which then silently defeated the rmdir below.
            main.companion_name = (f"{main.companion_name}, {e.display_name}"
                                   if main.companion_name else e.display_name)
            main.companion_bytes += e.total_bytes
            main.companion_parts.extend(e.parts)
            continue  # drop the standalone companion row
        surviving.append(e)
    # sort: (remaining) companions after main models, alpha within group
    surviving.sort(key=lambda e: (e.subdir.lower(), e.is_companion, e.display_name.lower()))
    snap.ggufs = surviving
    return snap


def delete_gguf(display_name: str, subdir: str = "") -> tuple[bool, str, int]:
    """Delete a GGUF (and all shards). Returns (ok, message, bytes_freed).
    If subdir is empty, matches only flat files with that display_name; otherwise the entry in that subdir."""
    snap = snapshot_models_dir()
    if snap.error and not snap.ggufs:
        return False, snap.error, 0
    # match by display_name AND subdir (allows same base filename to exist both flat and in a subdir)
    match = next((g for g in snap.ggufs if g.display_name == display_name and g.subdir == subdir), None)
    if match is None and subdir == "":
        # fall back: match any subdir if only display_name given (bulk-delete callers pass just display_name)
        match = next((g for g in snap.ggufs if g.display_name == display_name), None)
    if match is None:
        return False, f"not found: {display_name}", 0
    freed = 0
    removed: list[str] = []

    # Whole-directory delete when this model is the only model in its subdir.
    #
    # Downloads land one model per directory, and everything beside the weights there is
    # support material for THAT model: projectors (often more than one), chat_template.jinja,
    # tokenizer.model. Deleting file-by-file means anything not explicitly enumerated is
    # stranded -- and a single leftover keeps the directory non-empty, so the tidy-up rmdir
    # below silently does nothing and the orphans persist invisibly. Measured on a real
    # deletion: 3.3 GB of projectors left behind.
    #
    # Guarded on there being no OTHER main GGUF present, so a directory someone has put two
    # models into degrades to per-file deletion rather than taking the neighbour with it.
    if match.subdir:
        subpath = settings.models_dir / match.subdir
        own = {p.resolve() for p in list(match.parts) + list(match.companion_parts)}
        try:
            others = [
                q for q in subpath.iterdir()
                if q.is_file() and q.suffix.lower() == ".gguf"
                and not ini._is_companion(q.name) and q.resolve() not in own
            ]
        except OSError:
            others = []
        if not others and subpath.is_dir():
            try:
                for q in sorted(subpath.rglob("*")):
                    if q.is_file():
                        freed += q.stat().st_size
                        removed.append(q.name)
                shutil.rmtree(subpath)
                return True, f"deleted {len(removed)} file(s), removed {match.subdir}/", freed
            except OSError as e:
                return False, f"failed to remove {match.subdir}/: {e}", freed

    # Fallback: shared directory, or a flat file with no directory of its own.
    paths_to_remove: list[Path] = list(match.parts) + list(match.companion_parts)
    for p in paths_to_remove:
        try:
            size = p.stat().st_size
            p.unlink()
            freed += size
            removed.append(p.name)
        except OSError as e:
            return False, f"failed to delete {p.name}: {e}. removed so far: {removed}", freed
    # if this entry lived in a subdir, remove the dir when empty
    if match.subdir:
        subpath = settings.models_dir / match.subdir
        try:
            if subpath.is_dir() and not any(subpath.iterdir()):
                subpath.rmdir()
        except OSError:
            pass
    return True, f"deleted {len(removed)} file(s)", freed


# ---------- docker / containers ----------

def _docker_client() -> docker.DockerClient | None:
    try:
        return docker.from_env()
    except DockerException:
        return None


@dataclass
class ContainerInfo:
    name: str
    found: bool
    status: str | None = None
    image: str | None = None
    id: str | None = None


# ---------- llama backends: richer view + restart + probe ----------

# error-of-last-restart per container, shown in the card until it clears
_restart_errors: dict[str, str] = {}
_restart_errors_lock = threading.Lock()


@dataclass
class LlamaBackend:
    name: str
    found: bool
    status: str = ""           # running | exited | restarting | not_found | error
    image: str = ""
    short_id: str = ""
    started_at: str = ""
    uptime: str = ""
    host_ports: list[str] = field(default_factory=list)
    internal_port: int | None = None
    loaded_model: str | None = None
    probe_error: str | None = None
    last_restart_error: str | None = None


def _parse_started_at(iso: str) -> tuple[str, str]:
    # docker returns e.g. "2025-08-11T00:51:00.123456789Z"
    if not iso or iso.startswith("0001"):
        return "", ""
    try:
        # trim nanoseconds to microseconds
        core, _, frac = iso.partition(".")
        if frac:
            frac = frac.rstrip("Z")[:6]
            iso_norm = f"{core}.{frac}+00:00"
        else:
            iso_norm = core.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_norm)
    except ValueError:
        return iso, ""
    local = dt.astimezone().strftime("%Y-%m-%d %H:%M")
    delta = datetime.now(timezone.utc) - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        up = f"{secs}s"
    elif secs < 3600:
        up = f"{secs // 60}m"
    elif secs < 86400:
        up = f"{secs // 3600}h {(secs % 3600) // 60}m"
    else:
        up = f"{secs // 86400}d {(secs % 86400) // 3600}h"
    return local, up


def _extract_ports(attrs: dict) -> tuple[list[str], int | None]:
    """Return (['8082->8080/tcp'], 8080)."""
    ports_map = (attrs.get("NetworkSettings") or {}).get("Ports") or {}
    result: list[str] = []
    internal: int | None = None
    for cont_port, bindings in ports_map.items():
        # cont_port like "8080/tcp"
        try:
            internal = int(cont_port.split("/")[0])
        except ValueError:
            pass
        if not bindings:
            continue
        for b in bindings:
            hp = b.get("HostPort")
            if hp:
                result.append(f"{hp}->{cont_port}")
    return result, internal


async def _probe_loaded_model(container_name: str, internal_port: int | None) -> tuple[str | None, str | None]:
    if internal_port is None:
        return None, None
    url = f"http://{container_name}:{internal_port}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0, read=3.0)) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return None, f"HTTP {r.status_code}"
            data = r.json()
            items = data.get("data") or []
            if not items:
                return None, "no models configured"
            loaded_ids = [
                str(it.get("id") or "")
                for it in items
                if (it.get("status") or {}).get("value") == "loaded"
            ]
            if loaded_ids:
                return ", ".join(i for i in loaded_ids if i), None
            return None, f"{len(items)} configured, none loaded"
    except (httpx.HTTPError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"


def discover_llama_containers() -> list[dict]:
    """Return metadata for every container whose image looks like llama.cpp:server-*.
    Includes stopped containers so the user can see and start them.
    """
    client = _docker_client()
    if client is None:
        return []
    out: list[dict] = []
    try:
        containers = client.containers.list(all=True)
    except DockerException:
        return []
    for c in containers:
        try:
            tags = c.image.tags or [c.image.short_id]
            img = (tags[0] if tags else "").lower()
        except DockerException:
            img = ""
        if "ghcr.io/ggml-org/llama.cpp" not in img and "llama.cpp" not in img:
            continue
        if c.name == "model-loader":
            continue
        vendor = "rocm" if "rocm" in img else "cuda" if "cuda" in img else ("cpu" if "server" in img else "unknown")
        out.append({"name": c.name, "image": img, "vendor": vendor})
    return out


def _effective_container_names() -> list[str]:
    """Union of LLAMA_CONTAINERS env (retained even if not currently running) and any
    live-discovered llama.cpp containers on the docker socket. Order: env names first, then
    newly-discovered ones. Duplicates removed while preserving order."""
    whitelist = settings.llama_container_names
    discovered = [c["name"] for c in discover_llama_containers()]
    seen: set[str] = set()
    result: list[str] = []
    for n in list(whitelist) + discovered:
        if n and n not in seen:
            seen.add(n)
            result.append(n)
    return result


async def snapshot_llama_backends() -> list[LlamaBackend]:
    client = _docker_client()
    effective = _effective_container_names()
    if client is None:
        return [LlamaBackend(name=n, found=False, status="docker unreachable") for n in effective]

    out: list[LlamaBackend] = []
    probe_targets: list[tuple[int, str, int | None]] = []  # (idx, name, internal_port)

    for i, name in enumerate(effective):
        b = LlamaBackend(name=name, found=False, status="not_found")
        with _restart_errors_lock:
            b.last_restart_error = _restart_errors.get(name)
        try:
            c = client.containers.get(name)
            b.found = True
            b.status = c.status
            b.image = (c.image.tags or [c.image.short_id])[0]
            b.short_id = c.short_id
            attrs = c.attrs or {}
            state = (attrs.get("State") or {})
            started, up = _parse_started_at(state.get("StartedAt", ""))
            b.started_at = started
            b.uptime = up
            b.host_ports, b.internal_port = _extract_ports(attrs)
        except NotFound:
            pass
        except DockerException as e:
            b.status = f"error: {e}"
        out.append(b)
        if b.found and b.status == "running":
            probe_targets.append((i, name, b.internal_port))

    if probe_targets:
        results = await asyncio.gather(
            *[_probe_loaded_model(n, p) for _, n, p in probe_targets],
            return_exceptions=False,
        )
        for (i, _n, _p), (loaded, err) in zip(probe_targets, results):
            out[i].loaded_model = loaded
            out[i].probe_error = err
    return out


def restart_llama_backend(name: str) -> tuple[bool, str]:
    """Fire-and-forget restart. Errors captured in _restart_errors."""
    if name not in _effective_container_names():
        return False, "container not in configured list"
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    try:
        c = client.containers.get(name)
    except NotFound:
        return False, "container not found"
    except DockerException as e:
        return False, f"docker error: {e}"

    with _restart_errors_lock:
        _restart_errors.pop(name, None)

    def _do() -> None:
        try:
            c.restart(timeout=15)
        except (DockerException, APIError) as e:
            with _restart_errors_lock:
                _restart_errors[name] = f"{type(e).__name__}: {e}"

    threading.Thread(target=_do, daemon=True, name=f"restart-{name}").start()
    return True, "restart queued"


def _fit_backends() -> dict[str, float]:
    """{backend_name: total VRAM GiB} for everything we can actually plan against.

    Prefers the live probe over settings.gpu_vram_map. The static map is an optional
    override and is empty by default — relying on it alone silently disabled the fit
    chips entirely once the hardcoded example values were removed.
    """
    from . import hw
    out: dict[str, float] = {}
    for name in _effective_container_names():
        vram = float(settings.gpu_vram_map.get(name, 0) or 0)
        if vram <= 0:
            st = hw.stats_for(name)
            if st.ok and st.gpu and st.gpu.vram_total_gb > 0:
                vram = float(st.gpu.vram_total_gb)
        if vram > 0:
            out[name] = vram
    return out


# Above this, a CPU-resident DENSE model is technically loadable and practically unusable.
# Calibrated on measured tok/s on a dual-channel DDR5 box: a 6.6 GB dense model managed
# 4.4 tok/s, implying roughly 29 GB/s of effective read bandwidth, so 30 GB of dense weights
# lands near 1 tok/s.
#
# KNOWN LIMITATION: this is size-based, and size is the wrong axis for MoE. A mixture-of-
# experts model reads only its active experts per token, so a 30 GB MoE can be several times
# faster than a 30 GB dense one — measured here, a 15.8 GB MoE beat a 6.6 GB dense by 2x.
# Telling them apart needs expert_count/expert_used_count from the GGUF header, which the
# chips do not have (they receive a file size and nothing else). The tooltip says so rather
# than pretending the number applies to both.
_CPU_CRAWL_GB = 30.0


def _cpu_backends() -> list[str]:
    """Names of discovered llama containers with no GPU — they run on the CPU.

    _fit_backends() keys on VRAM and so cannot see these at all. They are still real places
    to run a model: the constraint is system RAM rather than VRAM, and ctx-size is the only
    fit lever, since ngl, n-cpu-moe and tensor-split all presuppose a GPU.
    """
    names = set(_effective_container_names())
    out: list[str] = []
    for d in discover_llama_containers():
        n = d.get("name") or ""
        if n in names and (d.get("vendor") or "").lower() in ("cpu", "", "unknown"):
            from . import hw
            st = hw.stats_for(n)
            if not (st.ok and st.gpu and st.gpu.vram_total_gb > 0):
                out.append(n)
    return out


def vram_fit_chips(size_bytes: int) -> list[dict]:
    """Per-backend fit verdicts: {'name', 'vram_gb', 'verdict', 'ratio_pct'}.

    verdict is one of {fits, tight, oom, impossible}.

    `oom` and `impossible` are genuinely different answers and must not look alike. `oom`
    means "not on the GPU alone" — offload moves layers into system RAM and it runs, slower.
    `impossible` means the model exceeds VRAM **plus** RAM, so there is nowhere for those
    layers to go and no setting rescues it. Rendering both as a red cross against a backend
    name tells the user the GPU is the constraint, when for `impossible` the machine is.

    An impossible model returns a SINGLE machine-level chip rather than one per backend,
    because naming backends implies picking a different one would help.
    """
    from . import hw
    gb = size_bytes / (1024 ** 3)

    # usable, not total: the OS reserve is already spoken for (see hw.usable_ram_gb).
    ram = hw.usable_ram_gb()
    pooled = max((v for v in _fit_backends().values() if v > 0), default=0.0)
    if ram > 0 and pooled > 0 and gb > (pooled + ram):
        return [{
            "name": "this machine",
            "vram_gb": round(pooled, 1),
            "host_ram_gb": round(ram, 1),
            "verdict": "impossible",
            "ratio_pct": round(gb / (pooled + ram) * 100),
            "needs_gb": round(gb),
            "ceiling_gb": round(pooled + ram),
        }]

    out: list[dict] = []
    for name, vram in _fit_backends().items():
        if vram <= 0:
            continue
        ratio = gb / vram
        if ratio < 0.65:
            verdict = "fits"
        elif ratio < 0.9:
            verdict = "tight"
        else:
            verdict = "oom"
        out.append({
            "name": name,
            "vram_gb": vram,
            "verdict": verdict,
            "ratio_pct": round(ratio * 100),
        })

    # CPU backends are sized against usable RAM. "Fits" and "usable" diverge sharply here:
    # generation is RAM-bandwidth-bound, so a large dense model can occupy memory perfectly
    # well and still produce well under a token per second. A plain green tick would be
    # true and misleading, so anything past a threshold gets its own 'slow' verdict.
    for name in _cpu_backends():
        if ram <= 0:
            continue
        ratio = gb / ram
        if ratio >= 1.0:
            verdict = "oom"
        elif gb >= _CPU_CRAWL_GB:
            verdict = "slow"
        elif ratio < 0.75:
            verdict = "fits"
        else:
            verdict = "tight"
        out.append({
            "name": name,
            "vram_gb": ram,
            "verdict": verdict,
            "ratio_pct": round(ratio * 100),
            "is_cpu": True,
        })
    return out


async def test_prompt(container_name: str, prompt: str, max_tokens: int = 256) -> dict:
    """Send a short chat completion to a container's llama-server. Returns dict with reply, tokens, elapsed_s, err."""
    client = _docker_client()
    internal_port: int | None = None
    if client is not None:
        try:
            c = client.containers.get(container_name)
            _, internal_port = _extract_ports(c.attrs or {})
        except (NotFound, DockerException):
            pass
    if internal_port is None:
        return {"ok": False, "err": "container not reachable"}

    # discover a loaded model id first
    loaded, err = await _probe_loaded_model(container_name, internal_port)
    if not loaded:
        return {"ok": False, "err": err or "no model loaded on this backend"}
    model_id = loaded.split(",")[0].strip()

    url = f"http://{container_name}:{internal_port}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    import time as _time
    t0 = _time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=180.0)) as ac:
            r = await ac.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}", "model": model_id}
    elapsed = _time.time() - t0

    reply = ""
    choices = data.get("choices") or []
    if choices:
        msg = (choices[0] or {}).get("message") or {}
        reply = msg.get("content") or ""
    usage = data.get("usage") or {}
    completion_toks = int(usage.get("completion_tokens") or 0)
    tps = round(completion_toks / elapsed, 1) if elapsed > 0 and completion_toks else None
    return {
        "ok": True,
        "model": model_id,
        "reply": reply,
        "completion_tokens": completion_toks,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "elapsed_s": round(elapsed, 2),
        "tokens_per_s": tps,
    }


_OWUI_DB = "/app/backend/data/webui.db"


def _has_webui_db(c) -> bool:
    """Definitive test: does this container actually hold OpenWebUI's database?"""
    try:
        r = c.exec_run(["test", "-f", _OWUI_DB])
        return r.exit_code == 0
    except (DockerException, APIError):
        return False


def _find_open_webui(client):
    """Locate the OpenWebUI container.

    Matching on `"open-webui" in <full image ref>` is too loose: an unrelated image from the
    same GitHub org — e.g. ghcr.io/open-webui/open-terminal — contains that string in its ORG
    path and would match first, after which every DB call fails with
    "unable to open database file" because that container has no webui.db.

    So: compare against the image's repository basename (not the org), prefer an exact
    container-name match, and confirm the database is actually present before accepting.
    """
    def _repo_basename(c) -> str:
        try:
            ref = ((c.image.tags or [""]) + [""])[0].lower()
        except DockerException:
            return ""
        ref = ref.split("@", 1)[0]                 # strip digest
        path = ref.rsplit(":", 1)[0]              # strip tag
        return path.rsplit("/", 1)[-1]            # image name only, no org/registry

    try:
        containers = list(client.containers.list(all=True))
    except DockerException:
        return None

    exact_name = [c for c in containers if (c.name or "").lower() in ("open-webui", "openwebui")]
    by_image = [c for c in containers if _repo_basename(c) in ("open-webui", "openwebui")]

    # Running instances first, then anything else; verify the DB before committing.
    for group in (exact_name, by_image):
        for c in sorted(group, key=lambda x: 0 if x.status == "running" else 1):
            if _has_webui_db(c):
                return c
    # Nothing verified — fall back to the best name/image guess so callers can still report
    # a sensible container name in their error message.
    return (exact_name or by_image or [None])[0]


def _openwebui_persisted_urls(ow) -> list[str]:
    """Read the actual URLs OpenWebUI is using — from its sqlite config, not the env var.
    Env is only a first-startup seed; persistent config in webui.db is the source of truth."""
    script = (
        "import sqlite3, json;"
        "c = sqlite3.connect('/app/backend/data/webui.db');"
        "r = c.execute(\"SELECT value FROM config WHERE key='openai.api_base_urls'\").fetchone();"
        "print(r[0] if r else '[]')"
    )
    try:
        r = ow.exec_run(["python3", "-c", script])
    except DockerException:
        return []
    if r.exit_code != 0:
        return []
    try:
        import json as _json
        return _json.loads(r.output.decode(errors="replace").strip() or "[]")
    except (ValueError, UnicodeDecodeError):
        return []


def _compose_service_of(client, container_name: str) -> str:
    """Return the compose service name of a container (from labels), or empty."""
    try:
        c = client.containers.get(container_name)
        return c.labels.get("com.docker.compose.service", "") or ""
    except (DockerException, NotFound):
        return ""


def openwebui_state() -> dict:
    """Report the URLs OpenWebUI currently has PERSISTED and which discovered llama backends are missing."""
    client = _docker_client()
    if client is None:
        return {"found": False, "reason": "docker unreachable", "current_urls": [], "missing_backends": []}
    ow = _find_open_webui(client)
    if ow is None:
        return {"found": False, "reason": "open-webui container not found", "current_urls": [], "missing_backends": []}

    current_urls = _openwebui_persisted_urls(ow)

    # A llama backend counts as "represented" if either its container-name URL or
    # its compose service-name URL is already in the list.
    discovered = discover_llama_containers()
    missing: list[dict] = []
    for d in discovered:
        candidate_urls = [f"http://{d['name']}:8080/v1"]
        svc = _compose_service_of(client, d["name"])
        if svc:
            candidate_urls.append(f"http://{svc}:8080/v1")
        if not any(u in current_urls for u in candidate_urls):
            missing.append({"name": d["name"], "vendor": d.get("vendor", ""), "url": candidate_urls[0]})

    # Detect stale entries — URLs pointing at compose-local hosts that no longer exist
    valid_hosts: set[str] = set()
    try:
        for c in client.containers.list(all=True):
            if c.name:
                valid_hosts.add(c.name)
            svc = c.labels.get("com.docker.compose.service") if c.labels else None
            if svc:
                valid_hosts.add(svc)
    except DockerException:
        pass
    from urllib.parse import urlparse
    stale: list[str] = []
    for u in current_urls:
        host = urlparse(u).hostname or ""
        looks_local = host and "." not in host and ":" not in host
        if looks_local and valid_hosts and host not in valid_hosts:
            stale.append(u)

    # Container facts + per-connection settings. Model Loader writes to this container's
    # database, so the page that does the writing should also show what it is writing to.
    image = ""
    short_id = ""
    ports: list[str] = []
    try:
        image = ((ow.image.tags or [ow.image.short_id]) or [""])[0]
        short_id = ow.short_id
        # Docker lists IPv4 and IPv6 bindings separately for the same mapping; dedupe so the
        # card doesn't show "3000→8080" twice.
        seen_ports: set[str] = set()
        for cport, binds in ((ow.attrs or {}).get("NetworkSettings", {}).get("Ports") or {}).items():
            for b in (binds or []):
                hp = b.get("HostPort")
                if not hp:
                    continue
                label = f"{hp}→{cport.split('/')[0]}"
                if label not in seen_ports:
                    seen_ports.add(label)
                    ports.append(label)
    except (DockerException, AttributeError, KeyError):
        pass

    # Pair each persisted URL with its connection config (prefix, enabled, model filter).
    cfgs = _openwebui_api_configs(ow)
    # Every llama backend serves the same models.ini, so a whitelist id that is not a section
    # name is one no backend can resolve. OpenWebUI still LISTS such an id — the whitelist is
    # what it renders — so the model appears in the picker and then fails with "model not
    # found" on first use. That is invisible from both ends unless something names it here.
    known_ids = set(ini.section_names())
    conns: list[dict] = []
    for i, u in enumerate(current_urls):
        c = cfgs.get(str(i)) or {}
        host = urlparse(u).hostname or ""
        wanted = c.get("model_ids") or []
        conns.append({
            "url": u,
            "host": host,
            "prefix_id": c.get("prefix_id") or "",
            "enabled": c.get("enable", True),
            "model_ids": wanted,
            "unknown_ids": [m for m in wanted if m not in known_ids],
            "live_ids": [m for m in wanted if m in known_ids],
            "stale": u in stale,
        })

    return {
        "found": True,
        "container_name": ow.name,
        "status": ow.status,
        "image": image,
        "short_id": short_id,
        "ports": ports,
        "current_urls": current_urls,
        "connections": conns,
        "companions": _openwebui_companions(client, ow, valid_hosts),
        "missing_backends": missing,
        "stale_urls": stale,
    }


def _openwebui_companions(client, ow, valid_hosts: set[str]) -> list[dict]:
    """Other services OpenWebUI depends on, and the containers behind them.

    Right now that means the terminal server (ghcr.io/open-webui/open-terminal), which
    OpenWebUI records under `terminal_server.connections`. It matters here for two reasons:
    it's part of a working OpenWebUI stack, and its image lives under the same GitHub org,
    which is exactly what previously made backend discovery grab the wrong container.
    """
    import json as _json
    from urllib.parse import urlparse as _urlparse

    script = (
        "import sqlite3, json;"
        "c=sqlite3.connect('/app/backend/data/webui.db');"
        "r=c.execute(\"SELECT value FROM config WHERE key='terminal_server.connections'\").fetchone();"
        "print(r[0] if r else '[]')"
    )
    try:
        res = ow.exec_run(["python3", "-c", script])
        raw = (res.output or b"").decode(errors="replace").strip() if res.exit_code == 0 else "[]"
        entries = _json.loads(raw or "[]")
    except (DockerException, APIError, ValueError):
        entries = []

    # Index running containers by name so each configured URL can be tied to a real container.
    by_name: dict[str, object] = {}
    try:
        for c in client.containers.list(all=True):
            if c.name:
                by_name[c.name] = c
    except DockerException:
        pass

    out: list[dict] = []
    for e in entries if isinstance(entries, list) else []:
        url = str(e.get("url") or "")
        host = _urlparse(url).hostname or ""
        c = by_name.get(host)
        image = ""
        status = "not found"
        if c is not None:
            status = getattr(c, "status", "") or ""
            try:
                image = ((c.image.tags or [c.image.short_id]) or [""])[0]
            except (DockerException, AttributeError):
                pass
        looks_local = bool(host) and "." not in host and ":" not in host
        out.append({
            "kind": "terminal server",
            "name": e.get("name") or host,
            "url": url,
            "host": host,
            "enabled": bool(e.get("enabled", True)),
            "container": host if c is not None else "",
            "image": image,
            "status": status,
            # Same stale rule as the llama backends: a compose-local host that no longer exists.
            "stale": looks_local and bool(valid_hosts) and host not in valid_hosts,
        })
    return out


def _openwebui_api_configs(ow) -> dict:
    """Read openai.api_configs out of webui.db. Best-effort: an empty dict just means the
    card shows URLs without their prefixes, never an error."""
    import json as _json
    script = (
        "import sqlite3, json;"
        "c=sqlite3.connect('/app/backend/data/webui.db');"
        "r=c.execute(\"SELECT value FROM config WHERE key='openai.api_configs'\").fetchone();"
        "print(r[0] if r else '{}')"
    )
    try:
        res = ow.exec_run(["python3", "-c", script])
        if res.exit_code != 0:
            return {}
        return _json.loads((res.output or b"").decode(errors="replace").strip() or "{}")
    except (DockerException, APIError, ValueError):
        return {}


def prune_openwebui_unknown_ids() -> list[str]:
    """Drop whitelist ids that no models.ini section provides, on every connection.

    Called after a rename, where the old id becomes unservable the moment the section changes
    name. Only ever REMOVES ids that cannot resolve, so it can't hide a working model; an
    empty whitelist means "offer everything", so a connection is never emptied to nothing —
    that would silently widen it instead of narrowing it.
    """
    st = openwebui_state()
    if not st.get("found"):
        return []
    cleaned: list[str] = []
    for c in st.get("connections") or []:
        dead = c.get("unknown_ids") or []
        keep = c.get("live_ids") or []
        if not dead or not keep:
            continue  # nothing dead, or pruning would empty the list into "offer everything"
        ok, _ = set_openwebui_model_filter(c["url"], keep)
        if ok:
            cleaned.extend(dead)
    return cleaned


def openwebui_capability_plan() -> dict[str, bool]:
    """{prefixed OpenWebUI model id -> supports vision}.

    A models.ini section is multimodal iff it declares `mmproj`. OpenWebUI stores capability
    per model id, and its ids carry the connection's prefix, so the same section served by two
    connections needs an entry for each.

    Only models a connection actually offers are included: an entry for a model the connection
    filters out would be a record for something the user can never select.
    """
    plan: dict[str, bool] = {}
    try:
        st = openwebui_state()
        if not st.get("found"):
            return {}
        # Vision means the projector encodes IMAGES, not merely that a projector exists --
        # llama.cpp uses the same --mmproj slot for audio encoders too.
        vision_by_section = {}
        for name in ini.section_names():
            mm = (ini.get_section(name) or {}).get("mmproj", "").strip()
            vision_by_section[name] = bool(mm) and "vision" in projector_modalities(mm)
        for c in st.get("connections") or []:
            if c.get("stale"):
                continue
            offered = c.get("model_ids") or list(vision_by_section)  # empty filter = all
            prefix = c.get("prefix_id") or ""
            for name in offered:
                if name not in vision_by_section:
                    continue  # stale whitelist id; prune handles those
                plan[f"{prefix}.{name}" if prefix else name] = vision_by_section[name]
    except (OSError, KeyError, AttributeError):
        return {}
    return plan


def sync_openwebui_capabilities() -> tuple[bool, str]:
    """Push per-model `capabilities.vision` into OpenWebUI, derived from models.ini.

    Without this OpenWebUI offers the image-upload control on every model, because it has no
    capability record for them. Picking a text-only model then fails at inference with
    "image input is not supported" -- the model id and the fact that it has a projector live
    in two different systems, and nothing warns you at selection time.

    Existing meta is MERGED, not replaced: capabilities a user set by hand in OpenWebUI's own
    model editor (web_search, code_interpreter, ...) survive. Only `vision` is authoritative
    here, because it is the only one derivable from the ini.
    """
    import json as _json
    plan = openwebui_capability_plan()
    if not plan:
        return False, "nothing to sync (no OpenWebUI connections, or no models.ini sections)"
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    ow = _find_open_webui(client)
    if ow is None:
        return False, "open-webui not found"

    script = """
import sqlite3, json, os, time
c = sqlite3.connect('/app/backend/data/webui.db')
now = int(time.time())
plan = json.loads(os.environ['PLAN'])

row = c.execute('SELECT id FROM user WHERE role=? ORDER BY created_at LIMIT 1', ('admin',)).fetchone()
if row is None:
    row = c.execute('SELECT id FROM user ORDER BY created_at LIMIT 1').fetchone()
if row is None:
    print('NOUSER'); raise SystemExit(0)
uid = row[0]

changed = 0
for mid, vision in plan.items():
    cur = c.execute('SELECT meta FROM model WHERE id=?', (mid,)).fetchone()
    if cur is None:
        meta = {'capabilities': {'vision': bool(vision)}}
        name = mid.split('.', 1)[1] if '.' in mid else mid
        c.execute(
            'INSERT INTO model(id, user_id, base_model_id, name, params, meta, updated_at, created_at, is_active)'
            ' VALUES(?,?,?,?,?,?,?,?,1)',
            (mid, uid, None, name, '{}', json.dumps(meta), now, now))
        changed += 1
    else:
        try: meta = json.loads(cur[0]) if cur[0] else {}
        except Exception: meta = {}
        if not isinstance(meta, dict): meta = {}
        caps = meta.get('capabilities')
        if not isinstance(caps, dict): caps = {}
        if caps.get('vision') == bool(vision):
            continue
        caps['vision'] = bool(vision)
        meta['capabilities'] = caps
        c.execute('UPDATE model SET meta=?, updated_at=? WHERE id=?', (json.dumps(meta), now, mid))
        changed += 1
c.commit()
print('OK %d' % changed)
"""
    try:
        r = ow.exec_run(["python3", "-c", script], environment={"PLAN": _json.dumps(plan)})
    except (DockerException, APIError) as e:
        return False, f"exec into open-webui failed: {e}"
    out = (r.output or b"").decode(errors="replace").strip()
    if r.exit_code != 0 or not out.startswith("OK"):
        return False, f"capability sync failed: {out[:200]}"
    n = out.split()[1] if len(out.split()) > 1 else "0"
    vis = sum(1 for v in plan.values() if v)
    # No restart. Unlike openai.api_configs (PersistentConfig, read once at boot), the `model`
    # table is ordinary application data that OpenWebUI queries per request, so capability
    # changes are picked up on the next page load.
    if n == "0":
        return True, f"already in sync ({vis} of {len(plan)} vision-capable)"
    return True, f"{n} model(s) updated ({vis} of {len(plan)} vision-capable)"


def align_openwebui_capabilities() -> tuple[bool, str]:
    """Make OpenWebUI's per-model capabilities match models.ini, authoritatively.

    Differs from sync_openwebui_capabilities() in two ways:
      * REPLACES the capabilities block on backend records instead of merging into it, so
        OpenWebUI's permissive defaults (vision on for everything) are overwritten rather
        than preserved.
      * DELETES backend records for model ids nothing serves any more -- the residue of
        deleted or renamed models, which otherwise keep claiming capabilities forever.

    Two categories are never touched:
      * Records with a base_model_id: those are workspace models the USER built in
        OpenWebUI (a custom name, system prompt and params on top of a base). Deleting one
        destroys real work that does not exist anywhere else.
      * Any record another model names as its base_model_id, even if the backend no longer
        serves it -- removing it would orphan the workspace model sitting on top.
    """
    import json as _json
    plan = openwebui_capability_plan()
    if not plan:
        return False, "nothing to align (no OpenWebUI connections, or no models.ini sections)"
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    ow = _find_open_webui(client)
    if ow is None:
        return False, "open-webui not found"

    script = """
import sqlite3, json, os, time
c = sqlite3.connect('/app/backend/data/webui.db')
now = int(time.time())
plan = json.loads(os.environ['PLAN'])

row = c.execute('SELECT id FROM user WHERE role=? ORDER BY created_at LIMIT 1', ('admin',)).fetchone()
if row is None:
    row = c.execute('SELECT id FROM user ORDER BY created_at LIMIT 1').fetchone()
if row is None:
    print('NOUSER'); raise SystemExit(0)
uid = row[0]

# ids that are somebody's base -- protected from deletion
bases = {r[0] for r in c.execute('SELECT DISTINCT base_model_id FROM model WHERE base_model_id IS NOT NULL')}

updated = 0
for mid, vision in plan.items():
    cur = c.execute('SELECT meta, base_model_id FROM model WHERE id=?', (mid,)).fetchone()
    if cur is not None and cur[1]:
        continue                      # user workspace model; not ours to rewrite
    if cur is None:
        meta = {}
    else:
        try: meta = json.loads(cur[0]) if cur[0] else {}
        except Exception: meta = {}
        if not isinstance(meta, dict): meta = {}
    meta['capabilities'] = {'vision': bool(vision)}   # replace, not merge
    if cur is None:
        name = mid.split('.', 1)[1] if '.' in mid else mid
        c.execute('INSERT INTO model(id, user_id, base_model_id, name, params, meta, updated_at, created_at, is_active)'
                  ' VALUES(?,?,?,?,?,?,?,?,1)', (mid, uid, None, name, '{}', json.dumps(meta), now, now))
    else:
        c.execute('UPDATE model SET meta=?, updated_at=? WHERE id=?', (json.dumps(meta), now, mid))
    updated += 1

removed, kept = [], []
for mid, base in c.execute('SELECT id, base_model_id FROM model').fetchall():
    if base or mid in plan:
        continue                      # workspace model, or still served
    if mid in bases:
        kept.append(mid); continue    # another model is built on it
    c.execute('DELETE FROM model WHERE id=?', (mid,))
    removed.append(mid)
c.commit()
print('OK ' + json.dumps({'updated': updated, 'removed': removed, 'kept': kept}))
"""
    try:
        r = ow.exec_run(["python3", "-c", script], environment={"PLAN": _json.dumps(plan)})
    except (DockerException, APIError) as e:
        return False, f"exec into open-webui failed: {e}"
    out = (r.output or b"").decode(errors="replace").strip()
    if r.exit_code != 0 or not out.startswith("OK"):
        return False, f"align failed: {out[:200]}"
    try:
        res = _json.loads(out[3:])
    except ValueError:
        return True, out[:160]
    vis = sum(1 for v in plan.values() if v)
    msg = f"aligned {res['updated']} model(s) — {vis} vision-capable of {len(plan)}"
    if res["removed"]:
        msg += f"; removed {len(res['removed'])} stale record(s): " + ", ".join(res["removed"][:4])
    if res["kept"]:
        msg += f"; kept {len(res['kept'])} stale record(s) still used as a base model"
    return True, msg


def set_openwebui_model_filter(url: str, model_ids: list[str]) -> tuple[bool, str]:
    """Restrict which models one OpenWebUI connection exposes.

    OpenWebUI's `openai.api_configs[<index>].model_ids` is a whitelist: empty means "offer
    everything this endpoint reports", non-empty means "offer only these". Model ids are the
    RAW ids from the backend's /v1/models — the connection's prefix_id is applied by
    OpenWebUI afterwards for display, so it must not appear here.

    Useful when several backends share one models.ini but shouldn't all serve every model —
    e.g. a CPU backend that should only offer the small models it can actually run.
    """
    import json as _json
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    ow = _find_open_webui(client)
    if ow is None:
        return False, "open-webui not found"

    script = """
import sqlite3, json, os, time
c = sqlite3.connect('/app/backend/data/webui.db')
now = int(time.time())

def _get(k, default):
    r = c.execute('SELECT value FROM config WHERE key=?', (k,)).fetchone()
    if not r:
        return default
    try: return json.loads(r[0])
    except Exception: return default

def _set(k, v):
    val = json.dumps(v)
    if c.execute('SELECT 1 FROM config WHERE key=?', (k,)).fetchone():
        c.execute('UPDATE config SET value=?, updated_at=? WHERE key=?', (val, now, k))
    else:
        c.execute('INSERT INTO config(key, value, updated_at) VALUES(?, ?, ?)', (k, val, now))

target = os.environ['TARGET_URL']
ids = json.loads(os.environ['MODEL_IDS'])
urls = _get('openai.api_base_urls', [])
cfgs = _get('openai.api_configs', {})
if target not in urls:
    print('NOTFOUND'); raise SystemExit(0)
idx = str(urls.index(target))
cfg = cfgs.get(idx) or {'enable': True, 'tags': [], 'prefix_id': '', 'model_ids': [],
                        'connection_type': 'external', 'auth_type': 'bearer', 'passthrough_params': []}
cfg['model_ids'] = ids
cfgs[idx] = cfg
_set('openai.api_configs', cfgs)
c.commit()
print('OK ' + str(len(ids)))
"""
    try:
        r = ow.exec_run(
            ["python3", "-c", script],
            environment={"TARGET_URL": url, "MODEL_IDS": _json.dumps(model_ids)},
        )
    except (DockerException, APIError) as e:
        return False, f"exec into open-webui failed: {e}"

    out = (r.output or b"").decode(errors="replace").strip()
    if r.exit_code != 0:
        return False, f"DB update failed (exit {r.exit_code}): {out[:200]}"
    if out.startswith("NOTFOUND"):
        return False, f"connection not found in OpenWebUI: {url}"

    # PersistentConfig only re-reads on boot, same as the endpoint sync.
    try:
        ow.restart(timeout=30)
    except (DockerException, APIError) as e:
        return False, f"filter saved but restart failed: {e}"
    n = len(model_ids)
    return True, (f"{url}: now offering {n} selected model(s)" if n
                  else f"{url}: now offering all models")


def toggle_openwebui_model(url: str, model_id: str, show: bool,
                           all_model_ids: list[str]) -> tuple[bool, str]:
    """Show/hide ONE model on ONE OpenWebUI connection.

    The wrinkle: model_ids == [] means "offer everything", not "offer nothing". So hiding a
    model on a connection that is currently unfiltered cannot just remove an entry — there
    are no entries. It has to materialise the full model list minus that one, which converts
    the connection from implicit-all to an explicit whitelist.

    That has a lasting consequence worth surfacing to the caller: once explicit, models
    downloaded later will NOT appear on that connection until they're added. Returns a
    message saying so, so the UI can warn rather than silently changing future behaviour.
    """
    cur = openwebui_state()
    if not cur.get("found"):
        return False, cur.get("reason") or "open-webui not found"
    conn = next((c for c in cur.get("connections", []) if c["url"] == url), None)
    if conn is None:
        return False, f"connection not found: {url}"

    existing = list(conn.get("model_ids") or [])
    was_implicit_all = not existing
    if show:
        if was_implicit_all:
            return True, f"{model_id} already visible on {conn['host']} (offering all models)"
        if model_id in existing:
            return True, f"{model_id} already visible on {conn['host']}"
        new_ids = existing + [model_id]
    else:
        if was_implicit_all:
            # Convert implicit-all into an explicit list so one model can be excluded.
            new_ids = [m for m in all_model_ids if m != model_id]
        else:
            new_ids = [m for m in existing if m != model_id]
            if not new_ids:
                # An empty list would silently mean "all models" — the opposite of hiding
                # the last one. Refuse rather than do the reverse of what was asked.
                return False, (f"cannot hide the last model on {conn['host']}: an empty list "
                               "means 'offer everything' in OpenWebUI. Disable the connection "
                               "instead if you want it to serve nothing.")

    ok, msg = set_openwebui_model_filter(url, new_ids)
    if ok and was_implicit_all and not show:
        msg += (f" — {conn['host']} now uses an explicit list, so models added later "
                "will not appear there until you enable them")
    return ok, msg


def sync_openwebui_endpoints() -> tuple[bool, str]:
    """Reconcile OpenWebUI's PERSISTED config (webui.db → openai.api_base_urls) with reality:
      1. Prune URLs whose target container/service no longer exists on the docker socket
      2. Add discovered llama backends that aren't in the list yet
    Preserves existing per-connection prefixes/tags for endpoints that stay.
    Then restarts open-webui so PersistentConfig re-hydrates from the updated DB.
    """
    import json as _json
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    ow = _find_open_webui(client)
    if ow is None:
        return False, "open-webui not found"

    # Build the set of hostnames that are valid targets (container names + compose service names)
    valid_hosts: set[str] = set()
    try:
        for c in client.containers.list(all=True):
            if c.name:
                valid_hosts.add(c.name)
            svc = c.labels.get("com.docker.compose.service") if c.labels else None
            if svc:
                valid_hosts.add(svc)
    except DockerException:
        pass

    state = openwebui_state()
    missing = state.get("missing_backends") or []

    additions = [{"name": m["name"]} for m in missing]

    script = r"""
import sqlite3, json, os, time
from urllib.parse import urlparse
now = int(time.time())
c = sqlite3.connect('/app/backend/data/webui.db')

def _get(k, default):
    r = c.execute('SELECT value FROM config WHERE key=?', (k,)).fetchone()
    if not r: return default
    try: return json.loads(r[0])
    except Exception: return default

def _set(k, v):
    val = json.dumps(v)
    if c.execute('SELECT 1 FROM config WHERE key=?', (k,)).fetchone():
        c.execute('UPDATE config SET value=?, updated_at=? WHERE key=?', (val, now, k))
    else:
        c.execute('INSERT INTO config(key, value, updated_at) VALUES(?, ?, ?)', (k, val, now))

urls = _get('openai.api_base_urls', [])
keys = _get('openai.api_keys', [])
cfgs = _get('openai.api_configs', {})
valid_hosts = set(json.loads(os.environ.get('VALID_HOSTS', '[]')))

# ---- 1) Prune stale entries whose host isn't a live container / service ----
kept_urls, kept_keys, kept_cfgs = [], [], {}
removed = 0
for old_idx, u in enumerate(urls):
    host = urlparse(u).hostname or ''
    # Keep non-llama hosts (e.g. openrouter, real openai) — only prune if it LOOKS like a compose-local
    # llama host (unqualified, no dots, was in the DB but no matching container exists any more).
    looks_local = host and '.' not in host and ':' not in host
    if looks_local and valid_hosts and host not in valid_hosts:
        removed += 1
        continue
    new_idx = str(len(kept_urls))
    kept_urls.append(u)
    kept_keys.append(keys[old_idx] if old_idx < len(keys) else 'dummy')
    if str(old_idx) in cfgs:
        kept_cfgs[new_idx] = cfgs[str(old_idx)]

# ---- 2) Add missing entries ----
added = 0
for a in json.loads(os.environ.get('ADDITIONS', '[]')):
    url = 'http://' + a['name'] + ':8080/v1'
    if url in kept_urls:
        continue
    kept_urls.append(url)
    kept_keys.append('dummy')
    idx = str(len(kept_urls) - 1)
    pref = a['name'].replace('llama-', '').replace('_', '-').upper()[:12]
    kept_cfgs[idx] = {
        'enable': True,
        'tags': [],
        'prefix_id': pref,
        'model_ids': [],
        'connection_type': 'external',
        'auth_type': 'bearer',
        'passthrough_params': [],
    }
    added += 1

_set('openai.api_base_urls', kept_urls)
_set('openai.api_keys', kept_keys)
_set('openai.api_configs', kept_cfgs)
_set('openai.enable', True)

c.commit()
print('added=' + str(added) + ' removed=' + str(removed))
"""

    try:
        r = ow.exec_run(
            ["python3", "-c", script],
            environment={
                "ADDITIONS": _json.dumps(additions),
                "VALID_HOSTS": _json.dumps(sorted(valid_hosts)),
            },
        )
    except (DockerException, APIError) as e:
        return False, f"exec into open-webui failed: {e}"

    out = r.output.decode(errors="replace").strip() if r.output else ""
    if r.exit_code != 0:
        return False, f"DB update failed (exit {r.exit_code}): {out[:300]}"

    # Restart so PersistentConfig re-hydrates from the updated DB
    try:
        ow.restart(timeout=30)
    except (DockerException, APIError) as e:
        return False, f"DB updated but restart failed: {e}. Restart open-webui manually."

    if additions or "removed=0" not in out:
        return True, f"reconciled OpenWebUI ({out}) and restarted"
    return True, f"already in sync — no changes needed ({out})"


def container_logs(name: str, tail: int = 200) -> tuple[bool, str]:
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
        raw = c.logs(tail=tail, stdout=True, stderr=True, timestamps=False)
    except DockerException as e:
        return False, f"log fetch failed: {e}"
    return True, raw.decode("utf-8", errors="replace")


def openwebui_capability_state() -> dict[str, "bool | None"]:
    """{prefixed model id -> vision as OpenWebUI currently has it}. None = no record.

    Read side of the capability sync, so the UI can show whether OpenWebUI agrees with
    models.ini for a given model rather than offering a blind "align" that gives no
    indication whether anything was actually wrong. Skips workspace models (those with a
    base_model_id): their capabilities belong to the user, not to us.
    """
    import json as _json
    client = _docker_client()
    if client is None:
        return {}
    ow = _find_open_webui(client)
    if ow is None:
        return {}
    script = """
import sqlite3, json
c = sqlite3.connect('/app/backend/data/webui.db')
out = {}
for i, b, m in c.execute('SELECT id, base_model_id, meta FROM model'):
    if b:
        continue
    try:
        d = json.loads(m) if m else {}
    except Exception:
        d = {}
    out[i] = (d.get('capabilities') or {}).get('vision')
print(json.dumps(out))
"""
    try:
        r = ow.exec_run(["python3", "-c", script])
        if r.exit_code != 0:
            return {}
        return _json.loads((r.output or b"{}").decode(errors="replace").strip() or "{}")
    except (DockerException, APIError, ValueError):
        return {}


def align_openwebui_capability_for(section: str) -> tuple[bool, str]:
    """Align capabilities for ONE models.ini section, across every connection serving it.

    Same authoritative replace as the bulk align, scoped to a single model. Stale-record
    cleanup is deliberately NOT done here: a stale record has no section to hang a per-model
    control off, so removing those belongs to the sweep instead.
    """
    import json as _json
    full = openwebui_capability_plan()
    # endswith("." + section), not rsplit: section names contain dots of their own
    # (Qwen3.8-27B-Q4_K_M), so splitting on the last dot matches the wrong thing.
    plan = {mid: v for mid, v in full.items()
            if mid == section or mid.endswith("." + section)}
    if not plan:
        return False, f"{section}: not offered by any OpenWebUI connection"
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    ow = _find_open_webui(client)
    if ow is None:
        return False, "open-webui not found"

    script = """
import sqlite3, json, os, time
c = sqlite3.connect('/app/backend/data/webui.db')
now = int(time.time())
plan = json.loads(os.environ['PLAN'])
row = c.execute('SELECT id FROM user WHERE role=? ORDER BY created_at LIMIT 1', ('admin',)).fetchone()
if row is None:
    row = c.execute('SELECT id FROM user ORDER BY created_at LIMIT 1').fetchone()
if row is None:
    print('NOUSER'); raise SystemExit(0)
uid = row[0]
n = 0
for mid, vision in plan.items():
    cur = c.execute('SELECT meta, base_model_id FROM model WHERE id=?', (mid,)).fetchone()
    if cur is not None and cur[1]:
        continue
    meta = {}
    if cur is not None:
        try:
            meta = json.loads(cur[0]) if cur[0] else {}
        except Exception:
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
    meta['capabilities'] = {'vision': bool(vision)}
    if cur is None:
        name = mid.split('.', 1)[1] if '.' in mid else mid
        c.execute('INSERT INTO model(id, user_id, base_model_id, name, params, meta, updated_at, created_at, is_active)'
                  ' VALUES(?,?,?,?,?,?,?,?,1)', (mid, uid, None, name, '{}', json.dumps(meta), now, now))
    else:
        c.execute('UPDATE model SET meta=?, updated_at=? WHERE id=?', (json.dumps(meta), now, mid))
    n += 1
c.commit()
print('OK %d' % n)
"""
    try:
        r = ow.exec_run(["python3", "-c", script], environment={"PLAN": _json.dumps(plan)})
    except (DockerException, APIError) as e:
        return False, f"exec into open-webui failed: {e}"
    out = (r.output or b"").decode(errors="replace").strip()
    if r.exit_code != 0 or not out.startswith("OK"):
        return False, f"align failed: {out[:200]}"
    vis = any(plan.values())
    return True, f"{section}: vision = {str(vis).lower()} on {', '.join(sorted(plan))}"


_PROJ_MODALITY_CACHE: dict[tuple, frozenset] = {}


def projector_modalities(mmproj_rel_or_abs: str) -> frozenset:
    """What a projector actually encodes: {'vision'}, {'audio'}, or both. Empty if unreadable.

    llama.cpp uses the same `--mmproj` slot for BOTH image and audio encoders (Qwen2-Audio,
    Ultravox and friends ship an audio projector). So the presence of an mmproj says the model
    is multimodal, not that it takes pictures. Deriving OpenWebUI's `vision` flag from the
    mere existence of a projector would offer image upload on an audio-only model -- the same
    class of wrong as offering it on a text-only one.

    The projector declares itself: a vision encoder carries clip.vision.* keys, an audio
    encoder clip.audio.*. Cached on (path, size, mtime) since this reads a file header.
    """
    from pathlib import Path as _Path
    raw_path = (mmproj_rel_or_abs or "").strip()
    if not raw_path:
        return frozenset()
    rel = raw_path.replace("/models/", "", 1).lstrip("/")
    p = settings.models_dir / rel
    try:
        st = p.stat()
    except OSError:
        return frozenset()
    key = (str(p), st.st_size, int(st.st_mtime))
    hit = _PROJ_MODALITY_CACHE.get(key)
    if hit is not None:
        return hit
    mods: set[str] = set()
    try:
        from . import gguf_meta
        kv = gguf_meta.read_raw(p)
        if isinstance(kv, dict) and "kv" in kv and isinstance(kv["kv"], dict):
            kv = kv["kv"]
        if isinstance(kv, dict):
            for k, v in kv.items():
                lk = str(k).lower()
                if lk.startswith("clip.vision.") or lk == "clip.has_vision_encoder" and v:
                    mods.add("vision")
                elif lk.startswith("clip.audio.") or lk == "clip.has_audio_encoder" and v:
                    mods.add("audio")
    except Exception:  # noqa: BLE001 -- a projector we cannot parse must not break the page
        return frozenset()
    out = frozenset(mods)
    if len(_PROJ_MODALITY_CACHE) > 32:
        _PROJ_MODALITY_CACHE.clear()
    _PROJ_MODALITY_CACHE[key] = out
    return out


def sections_with_speculative() -> dict[str, str]:
    """{section name -> draft head filename} for sections with speculative decoding wired up.

    Requires BOTH a draft model and a spec-type that is not "none": naming a head without
    selecting a type does nothing, and a type without a head cannot run.
    """
    out: dict[str, str] = {}
    try:
        for name in ini.section_names():
            sec = ini.get_section(name) or {}
            model = (sec.get("spec-draft-model") or "").strip()
            stype = (sec.get("spec-type") or "").strip().lower()
            if model and stype and stype != "none":
                out[name] = model.rsplit("/", 1)[-1]
    except (OSError, KeyError, AttributeError):
        return {}
    return out


def section_modalities() -> dict[str, list[str]]:
    """{section name -> sorted modalities its projector encodes}. Empty list = text-only.

    Separate from openwebui_capability_plan(), which reduces this to a single vision bool
    because that is all OpenWebUI models. Audio has no OpenWebUI capability flag at all, so
    an audio modality can be shown here but cannot be pushed anywhere — it is Model Loader's
    own display, not a setting.
    """
    out: dict[str, list[str]] = {}
    try:
        for name in ini.section_names():
            mm = (ini.get_section(name) or {}).get("mmproj", "").strip()
            out[name] = sorted(projector_modalities(mm)) if mm else []
    except (OSError, KeyError, AttributeError):
        return {}
    return out


# Vendors that mean "this backend has a GPU". Read from the image tag by
# discover_llama_containers, so it needs no live probe — which matters because this decision
# is made right after a download completes, possibly before the stats sampler has warmed up.
_GPU_VENDORS = {"cuda", "rocm", "vulkan", "nvidia", "amd"}


def assign_new_model_to_gpu(section: str) -> tuple[bool, str]:
    """Offer a newly-created section on every GPU-backed OpenWebUI connection.

    A connection with a non-empty model_ids list is an explicit whitelist, so a model added
    later is offered by nothing until someone ticks it — a new download lands in models.ini,
    works perfectly, and is invisible in the chat UI with no indication why. Defaulting it
    onto the GPU backends matches what anyone downloading a model actually wants.

    Deliberately narrow:
      * GPU connections only. A CPU backend should not silently inherit a 27B.
      * Connections already offering everything (empty whitelist) are skipped — they serve it
        already, and writing an explicit list would convert them to a whitelist, quietly
        changing behaviour for every FUTURE model.
      * Callers must only invoke this on CREATION. Re-running it on an ordinary save would
        undo a deliberate removal.
    """
    try:
        gpu_hosts = {d.get("name") for d in discover_llama_containers()
                     if (d.get("vendor") or "").lower() in _GPU_VENDORS}
    except DockerException:
        return False, "docker unreachable"
    if not gpu_hosts:
        return False, "no GPU-backed llama container discovered"

    st = openwebui_state()
    if not st.get("found"):
        return False, st.get("reason") or "open-webui not found"

    touched: list[str] = []
    for c in st.get("connections") or []:
        if c.get("stale") or c["host"] not in gpu_hosts:
            continue
        ids = list(c.get("model_ids") or [])
        if not ids:
            continue                      # already offers everything
        if section in ids:
            continue                      # nothing to do
        ok, _ = set_openwebui_model_filter(c["url"], ids + [section])
        if ok:
            touched.append(c.get("prefix_id") or c["host"])
    if not touched:
        return True, ""
    return True, f"offered on {', '.join(touched)}"
