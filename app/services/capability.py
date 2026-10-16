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
from .modalities import projector_modalities, section_modalities
from .openwebui import (_find_open_webui, align_openwebui_capabilities,
                        openwebui_capability_plan, openwebui_state,
                        set_openwebui_model_filter)


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
    # Same rule as align_openwebui_capabilities(): own `vision`, leave the rest alone.
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
