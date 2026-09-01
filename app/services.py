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
    aliases: list[str] = field(default_factory=list)  # ini section names that reference this

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
        matched = [a for a in aliases if a == stem_no_ext]
        entries.append(GgufEntry(
            display_name=base,
            parts=parts,
            total_bytes=total,
            mtime=mtime,
            is_sharded=len(parts) > 1,
            subdir=subdir,
            aliases=matched,
        ))
    entries.sort(key=lambda e: e.display_name.lower())
    snap.ggufs = entries
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
    for p in match.parts:
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


@dataclass
class ContainersSnapshot:
    socket_ok: bool
    error: str | None
    containers: list[ContainerInfo]


def snapshot_containers() -> ContainersSnapshot:
    client = _docker_client()
    if client is None:
        return ContainersSnapshot(socket_ok=False, error="cannot reach docker socket", containers=[])
    try:
        client.ping()
    except DockerException as e:
        return ContainersSnapshot(socket_ok=False, error=f"docker ping failed: {e}", containers=[])

    out: list[ContainerInfo] = []
    for name in settings.llama_container_names:
        try:
            c = client.containers.get(name)
            image = (c.image.tags or [c.image.short_id])[0]
            out.append(ContainerInfo(name=name, found=True, status=c.status, image=image, id=c.short_id))
        except NotFound:
            out.append(ContainerInfo(name=name, found=False))
        except DockerException as e:
            out.append(ContainerInfo(name=name, found=False, status=f"error: {e}"))
    return ContainersSnapshot(socket_ok=True, error=None, containers=out)


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


def vram_fit_chips(size_bytes: int) -> list[dict]:
    """For each configured GPU, return {'name', 'vram_gb', 'verdict' in {fits,tight,oom}, 'ratio_pct'}."""
    gb = size_bytes / (1024 ** 3)
    out: list[dict] = []
    for name, vram in settings.gpu_vram_map.items():
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


def _find_open_webui(client):
    try:
        for c in client.containers.list(all=True):
            try:
                img = ((c.image.tags or [""]) + [""])[0].lower()
            except DockerException:
                img = ""
            if "open-webui" in img or "openwebui" in img:
                return c
    except DockerException:
        pass
    return None


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

    return {
        "found": True,
        "container_name": ow.name,
        "status": ow.status,
        "current_urls": current_urls,
        "missing_backends": missing,
    }


def sync_openwebui_endpoints() -> tuple[bool, str]:
    """Add missing llama backends to OpenWebUI's PERSISTED config (webui.db → openai.api_base_urls).
    Then restart open-webui so it reloads its PersistentConfig from the updated DB.
    Preserves the user's existing connections, keys, and per-connection prefixes/tags.
    """
    import json as _json
    client = _docker_client()
    if client is None:
        return False, "docker unreachable"
    ow = _find_open_webui(client)
    if ow is None:
        return False, "open-webui not found"

    state = openwebui_state()
    missing = state.get("missing_backends") or []
    if not missing:
        return True, "already in sync — no missing backends"

    # additions is a JSON-safe list; passed as env var to the exec'd python script for clean escaping
    additions = [{"name": m["name"]} for m in missing]

    script = r"""
import sqlite3, json, os, time
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

added = 0
for a in json.loads(os.environ.get('ADDITIONS', '[]')):
    url = 'http://' + a['name'] + ':8080/v1'
    if url in urls:
        continue
    urls.append(url)
    keys.append('dummy')
    idx = str(len(urls) - 1)
    # short readable prefix from container name (strip 'llama-', uppercase, cap 12 chars)
    pref = a['name'].replace('llama-', '').replace('_', '-').upper()[:12]
    cfgs[idx] = {
        'enable': True,
        'tags': [],
        'prefix_id': pref,
        'model_ids': [],
        'connection_type': 'external',
        'auth_type': 'bearer',
        'passthrough_params': [],
    }
    added += 1

_set('openai.api_base_urls', urls)
_set('openai.api_keys', keys)
_set('openai.api_configs', cfgs)
# Ensure the OpenAI provider is enabled
_set('openai.enable', True)

c.commit()
print('added=' + str(added))
"""

    try:
        r = ow.exec_run(
            ["python3", "-c", script],
            environment={"ADDITIONS": _json.dumps(additions)},
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

    return True, f"added {len(missing)} backend(s) to OpenWebUI DB and restarted ({out})"


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
