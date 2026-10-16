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

from .backends import discover_llama_containers
from .dockerbase import _docker_client
from .modalities import projector_modalities
from .stratactl import strata_model_ids


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
        # Single-model servers are not chat endpoints. Orpheus TTS is a llama.cpp container
        # serving SNAC audio codes on 5006; offering to add it to Open WebUI's model list
        # would register a URL that is wrong (:8080) for a service that is not a chat model.
        if not d.get("router"):
            continue
        candidate_urls = [f"http://{d['name']}:8080/v1"]
        svc = _compose_service_of(client, d["name"])
        if svc:
            candidate_urls.append(f"http://{svc}:8080/v1")
        if not any(u in current_urls for u in candidate_urls):
            missing.append({"name": d["name"], "vendor": d.get("vendor", ""), "url": candidate_urls[0]})

    # A non-llama engine is not a router and never appears in discover_llama_containers, so
    # it joins the missing list only when its container says so: ai-lab.openwebui=1 is the
    # opt-in. Strata serves one real chat model out of a 60 GB arena; whether that belongs
    # in the chat picker is a decision, not a discovery.
    for c in client.containers.list(all=True):
        labels = c.labels or {}
        if (labels.get("ai-lab.engine") or "").strip().lower() != strata_engine.ENGINE:
            continue
        if (labels.get("ai-lab.openwebui") or "").strip().lower() not in ("1", "true", "yes", "on"):
            continue
        port = next((e.split("=", 1)[1] for e in
                     ((c.attrs.get("Config") or {}).get("Env") or [])
                     if e.startswith("PORT=")), "8080") or "8080"
        candidate_urls = [f"http://{c.name}:{port}/v1"]
        svc = _compose_service_of(client, c.name)
        if svc:
            candidate_urls.append(f"http://{svc}:{port}/v1")
        if not any(u in current_urls for u in candidate_urls):
            missing.append({"name": c.name, "vendor": strata_engine.ENGINE, "url": candidate_urls[0]})

    # If Open WebUI is pointed at THIS app's router, the per-backend connections are absent on
    # purpose and none of them is missing.
    #
    # Without this the page is permanently wrong, and not merely noisily: every llama router on
    # the box reads as missing forever, the Reconcile banner never clears, and pressing it adds
    # those connections back - undoing the single-endpoint setup the router exists to provide.
    # Observed with exactly one connection configured, http://model-loader:8086/v1, and
    # llama-9700x2, llama-cpu and llama-voice all listed as missing.
    #
    # The check is for this container's own name or compose service on the router port, so it
    # recognises the router only when Open WebUI is actually talking to US. A different host on
    # 8086 is somebody else's proxy and says nothing about whether these backends are reachable.
    router_urls: list[str] = []
    try:
        own = client.containers.get(socket.gethostname())
        rport = int(settings.router_port)
        names = [own.name]
        svc = (own.labels or {}).get("com.docker.compose.service")
        if svc and svc != own.name:          # compose usually makes these identical
            names.append(svc)
        router_urls = [f"http://{n}:{rport}/v1" for n in names if n]
    except (DockerException, NotFound, ValueError, TypeError):
        router_urls = []

    routed_via = next((u for u in router_urls if u in current_urls), "")
    if routed_via:
        missing = []

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
    # A Strata connection's whitelist ids are its own /v1/models, not ini sections - without
    # them every Strata model id reads as unknown here while working fine in the picker.
    for c in client.containers.list(all=True):
        if ((c.labels or {}).get("ai-lab.engine") or "").strip().lower() == strata_engine.ENGINE \
                and c.status == "running":
            known_ids.update(strata_model_ids(c.name))
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
        # Set when Open WebUI reaches these backends through our router instead of directly.
        "routed_via": routed_via,
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
      * OVERWRITES `capabilities.vision` on backend records instead of deferring to what is
        already there, so OpenWebUI's permissive default (vision on for everything) loses.
        Only that one key is touched -- other capabilities are left exactly as found.
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
    # Authoritative about `vision` ONLY. Replacing the whole capabilities block also
    # destroyed image_generation, web_search, code_interpreter and citations -- flags this
    # app knows nothing about and has no business clearing. Aligning vision used to switch
    # off image generation for every model it touched.
    caps = meta.get('capabilities')
    if not isinstance(caps, dict):
        caps = {}
    caps['vision'] = bool(vision)
    meta['capabilities'] = caps
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
