from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from . import autoconfig
from . import telemetry, db, gguf_meta, hf, hw, ini, services
from .config import settings
from .downloader import manager
from .utils import human_bytes, shard_key

app = FastAPI(title="Model Loader")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.globals["hue"] = lambda s: sum(ord(c) for c in (s or "")) % 360


@app.on_event("startup")
def _startup() -> None:
    db.init()
    hw.start_sampler()


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/palette.json")
def palette() -> dict:
    """Everything Cmd+K can jump to or trigger."""
    snap = services.snapshot_models_dir()
    items: list[dict] = []
    for g in snap.ggufs:
        items.append({
            "kind": "model",
            "title": g.display_name,
            "hint": g.human_size,
            "url": f"/model/{g.display_name}",
            "icon": "layers-3",
        })
    for s in ini.list_sections():
        items.append({
            "kind": "section",
            "title": s.name,
            "hint": f"{len(s.items)} opts",
            "url": f"/config/section/{s.name}/edit",
            "icon": "file-cog",
        })
    for name in services._effective_container_names():
        items.append({
            "kind": "container",
            "title": name,
            "hint": "view details",
            "url": "/containers",
            "icon": "server",
        })
        items.append({
            "kind": "action",
            "title": f"Restart {name}",
            "hint": "container",
            "action": "post",
            "url": f"/containers/{name}/restart",
            "icon": "rotate-cw",
        })
    for p in db.list_prompts():
        items.append({
            "kind": "prompt",
            "title": p["name"],
            "hint": "saved prompt",
            "url": "/prompts",
            "icon": "message-square-quote",
        })
    # global actions
    items += [
        {"kind": "action", "title": "Check for HF updates", "hint": "compare local vs HF", "action": "post", "url": "/models/check-updates", "icon": "refresh-cw"},
        {"kind": "action", "title": "Clear finished downloads", "hint": "history clear", "action": "post", "url": "/downloads/clear", "icon": "check"},
        {"kind": "page", "title": "Search Hugging Face", "url": "/search", "icon": "search"},
        {"kind": "page", "title": "Downloads", "url": "/downloads", "icon": "download"},
        {"kind": "page", "title": "models.ini editor", "url": "/config", "icon": "file-cog"},
        {"kind": "page", "title": "Containers", "url": "/containers", "icon": "server"},
        {"kind": "page", "title": "Prompts", "url": "/prompts", "icon": "message-square-quote"},
        {"kind": "page", "title": "Settings", "url": "/settings", "icon": "settings"},
        {"kind": "page", "title": "Overview / dashboard", "url": "/", "icon": "gauge"},
    ]
    return {"items": items}


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    snap = services.snapshot_models_dir()
    backends = await services.snapshot_llama_backends()
    sections = ini.list_sections()
    # recent downloads: successful ones from history, deduped by filename
    rows = db.recent_downloads(20)
    seen: set[str] = set()
    recent: list[dict] = []
    owners: set[str] = set()
    for r in rows:
        if r["status"] != "done":
            continue
        if r["filename"] in seen:
            continue
        seen.add(r["filename"])
        owner = (r["repo_id"].split("/", 1)[0] if "/" in r["repo_id"] else "") if r["repo_id"] else ""
        if owner:
            owners.add(owner)
        recent.append({
            "filename": r["filename"],
            "size_h": human_bytes(int(r["total_bytes"] or 0)),
            "owner": owner,
        })
        if len(recent) >= 5:
            break
    avatars = await hf.owner_avatars(list(owners)) if owners else {}
    for r in recent:
        r["avatar_url"] = avatars.get(r["owner"], "")
    active = sum(1 for j in manager.snapshot() if j.status in ("queued", "downloading"))
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "snap": snap,
        "backends": backends,
        "stats": _stats_by_name(),
        "ini_sections": sections,
        "recent_downloads": recent,
        "active_downloads": active,
    })


# ---------- Models directory ----------

async def _loaded_map() -> dict[str, list[str]]:
    """stem -> [backend_name, ...] for currently-loaded models."""
    backends = await services.snapshot_llama_backends()
    m: dict[str, list[str]] = {}
    for b in backends:
        if not b.loaded_model:
            continue
        for mid in [s.strip() for s in b.loaded_model.split(",")]:
            if mid:
                m.setdefault(mid, []).append(b.name)
    return m


def _update_status_map(snap) -> dict[str, dict]:
    """filename -> {'status': 'up-to-date'|'stale'|'unknown', 'remote': iso, 'checked_at': ts, 'delta_days': int|None}"""
    from datetime import datetime, timezone
    checks = db.all_update_checks()
    out: dict[str, dict] = {}
    for g in snap.ggufs:
        row = checks.get(g.display_name)
        if not row:
            continue
        remote = row.get("hf_last_modified") or ""
        if not remote:
            out[g.display_name] = {"status": "unknown", "remote": "", "checked_at": row["checked_at"], "delta_days": None}
            continue
        try:
            core, _, frac = remote.partition(".")
            if frac:
                frac = frac.rstrip("Z")[:6]
                remote_dt = datetime.fromisoformat(f"{core}.{frac}+00:00")
            else:
                remote_dt = datetime.fromisoformat(core.replace("Z", "+00:00"))
        except ValueError:
            out[g.display_name] = {"status": "unknown", "remote": remote, "checked_at": row["checked_at"], "delta_days": None}
            continue
        local_dt = datetime.fromtimestamp(g.mtime, tz=timezone.utc)
        delta_days = (remote_dt - local_dt).days
        status = "stale" if remote_dt > local_dt else "up-to-date"
        out[g.display_name] = {"status": status, "remote": remote_dt.strftime("%Y-%m-%d"), "checked_at": row["checked_at"], "delta_days": delta_days}
    return out


async def _models_avatar_map(snap) -> tuple[dict[str, str], dict[str, str]]:
    """(filename -> owner, owner -> avatar_url) for the files currently in the models dir."""
    owner_by_file = db.owner_by_filename()
    file_to_owner: dict[str, str] = {}
    for g in snap.ggufs:
        for p in g.parts:
            # look up by plain basename and by subdir-qualified name (for shards in subdirs)
            keys = [p.name]
            if g.subdir:
                keys.append(f"{g.subdir}/{p.name}")
            for k in keys:
                if k in owner_by_file:
                    file_to_owner[g.display_name] = owner_by_file[k]
                    break
            if g.display_name in file_to_owner:
                break
    avatars = await hf.owner_avatars(list(set(file_to_owner.values()))) if file_to_owner else {}
    return file_to_owner, avatars


def _owui_visibility() -> dict:
    """Per-connection model visibility for the Models page.

    Returns {"conns": [{host, url, prefix_id, explicit}], "visible": {model_id: {url: bool}}}.
    Only populated when there is more than one connection — with a single backend every
    model is on it and a toggle would be pure noise.
    """
    st = services.openwebui_state()
    conns = [c for c in (st.get("connections") or []) if not c.get("stale")]
    if not st.get("found"):
        return {"conns": [], "visible": {}, "caps": {}}

    # Capabilities are computed regardless of connection count. Unlike the visibility
    # toggles -- which are noise with a single backend, since everything is on it -- a wrong
    # vision flag is just as broken with one connection as with five.
    want = services.openwebui_capability_plan()
    have = services.openwebui_capability_state()
    mods = services.section_modalities()
    spec = services.sections_with_speculative()
    caps: dict[str, dict] = {}
    # Match the id back to its section by NAME, never by splitting on dots: OpenWebUI ids are
    # "<prefix>.<section>" but section names contain dots too (Qwen3.8-27B-Q4_K_M), so
    # rsplit(".", 1) yields "8-27B-Q4_K_M" and silently matches nothing.
    _sections = set(ini.section_names())
    for mid, should in want.items():
        section = next((n for n in _sections if mid == n or mid.endswith("." + n)), None)
        if section is None:
            continue
        row = caps.setdefault(section, {"should": should, "current": [], "mismatch": False,
                                        "modalities": mods.get(section, []),
                                        "speculative": spec.get(section, "")})
        cur = have.get(mid)
        row["current"].append((mid, cur))
        if cur != should:
            row["mismatch"] = True

    if len(conns) < 2:
        return {"conns": [], "visible": {}, "caps": caps}
    ids = sorted(ini.section_names())
    visible: dict[str, dict] = {}
    for mid in ids:
        row: dict[str, bool] = {}
        for c in conns:
            filt = c.get("model_ids") or []
            row[c["url"]] = (mid in filt) if filt else True  # empty filter = offers everything
        visible[mid] = row
    return {
        "conns": [{"host": c["host"], "url": c["url"], "prefix_id": c.get("prefix_id") or "",
                   "explicit": bool(c.get("model_ids"))} for c in conns],
        "visible": visible,
        "caps": caps,
    }


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request) -> HTMLResponse:
    snap = services.snapshot_models_dir()
    file_to_owner, avatars = await _models_avatar_map(snap)
    return templates.TemplateResponse("models.html", {
        "request": request, "snap": snap, "flash": None,
        "loaded_map": await _loaded_map(),
        "file_to_owner": file_to_owner, "avatars": avatars,
        "update_status": _update_status_map(snap),
        "owui": _owui_visibility(),
    })


@app.get("/model/{filename:path}", response_class=HTMLResponse)
async def model_local_detail(request: Request, filename: str) -> HTMLResponse:
    from datetime import datetime, timezone
    from fastapi import HTTPException

    if ".." in filename or filename.startswith("/") or filename.count("/") > 1:
        raise HTTPException(status_code=400, detail="bad filename")

    # Resolve: filename may be "flat.gguf" OR "subdir/flat.gguf" OR "subdir" (routes to first shard).
    snap = services.snapshot_models_dir()
    entry = None
    if "/" in filename:
        sub, base = filename.split("/", 1)
        entry = next((g for g in snap.ggufs if g.subdir == sub and g.display_name == base), None)
    else:
        entry = next((g for g in snap.ggufs if g.display_name == filename and g.subdir == ""), None)
        if entry is None:
            # try as a subdir name pointing at the group inside
            entry = next((g for g in snap.ggufs if g.subdir == filename), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="not found")

    path = entry.parts[0]  # first shard is where metadata lives
    st = path.stat()
    raw = gguf_meta.read_raw(path)
    summary = gguf_meta.summarize(raw)
    stem = entry.stem
    filename = entry.route_key

    owner_by_file = db.owner_by_filename()
    # try subdir-qualified name first, then plain basename (for flat downloads)
    owner = owner_by_file.get(filename, "") or owner_by_file.get(entry.parts[0].name, "")
    avatars = await hf.owner_avatars([owner]) if owner else {}
    avatar_url = avatars.get(owner, "") if owner else ""

    # The section that owns this file may be named something else entirely (renamed to give
    # the model a short API id), so resolve file -> section rather than assuming stem == id.
    owning = ini.sections_by_file().get(entry.first_shard_rel) or []
    model_id = stem if stem in owning else (owning[0] if owning else stem)

    lm = await _loaded_map()
    loaded_on = lm.get(model_id, [])

    in_ini = model_id in ini.section_names()
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

    return templates.TemplateResponse("model_local.html", {
        "request": request,
        "filename": filename,
        "path": str(path),
        "stem": stem,
        "model_id": model_id,
        "size_h": human_bytes(st.st_size),
        "mtime": mtime,
        "raw": raw,
        "summary": summary,
        "owner": owner,
        "avatar_url": avatar_url,
        "loaded_on": loaded_on,
        "in_ini": in_ini,
    })


@app.post("/models/delete", response_class=HTMLResponse)
async def models_delete(request: Request, name: str = Form(...)) -> HTMLResponse:
    ok, msg, freed = services.delete_gguf(name)
    # Deleting the weights does not remove the id from OpenWebUI's whitelist, and OpenWebUI
    # RENDERS that whitelist rather than intersecting it with what the backend serves -- so a
    # deleted model keeps appearing in the picker and fails with "model not found" on use.
    # Prune here; it is a no-op (and costs no restart) when nothing is actually stale.
    if ok:
        try:
            pruned = services.prune_openwebui_unknown_ids()
            if pruned:
                msg = f"{msg}; removed {len(pruned)} stale id(s) from OpenWebUI"
            services.sync_openwebui_capabilities()
        except Exception:  # noqa: BLE001 -- deletion must succeed even if OpenWebUI is down
            pass
    snap = services.snapshot_models_dir()
    file_to_owner, avatars = await _models_avatar_map(snap)
    flash = {"ok": ok, "msg": msg, "freed_h": human_bytes(freed) if freed else None}
    return templates.TemplateResponse("_models_list.html", {
        "request": request, "snap": snap, "flash": flash,
        "loaded_map": await _loaded_map(),
        "file_to_owner": file_to_owner, "avatars": avatars,
        "update_status": _update_status_map(snap),
    })


@app.post("/models/delete-bulk", response_class=HTMLResponse)
async def models_delete_bulk(request: Request) -> HTMLResponse:
    form = await request.form()
    names = form.getlist("names") if hasattr(form, "getlist") else form.get("names") or []
    if isinstance(names, str):
        names = [names]
    total_freed = 0
    ok_count = 0
    errors: list[str] = []
    for n in names:
        ok, msg, freed = services.delete_gguf(str(n))
        if ok:
            ok_count += 1
            total_freed += freed
        else:
            errors.append(f"{n}: {msg}")
    # One prune after the whole batch, not per file -- each whitelist write restarts
    # open-webui, so doing it inside the loop would restart it once per deleted model.
    if ok_count:
        try:
            services.prune_openwebui_unknown_ids()
        except Exception:  # noqa: BLE001
            pass
    snap = services.snapshot_models_dir()
    file_to_owner, avatars = await _models_avatar_map(snap)
    if errors:
        flash = {"ok": False, "msg": f"deleted {ok_count}, {len(errors)} failed: " + "; ".join(errors[:3])}
    else:
        flash = {"ok": True, "msg": f"deleted {ok_count} file(s)", "freed_h": human_bytes(total_freed) if total_freed else None}
    return templates.TemplateResponse("_models_list.html", {
        "request": request, "snap": snap, "flash": flash,
        "loaded_map": await _loaded_map(),
        "file_to_owner": file_to_owner, "avatars": avatars,
        "update_status": _update_status_map(snap),
    })


@app.post("/models/check-updates", response_class=HTMLResponse)
async def models_check_updates(request: Request) -> HTMLResponse:
    records = db.download_records()
    if records:
        await hf.check_updates_for(records)
    snap = services.snapshot_models_dir()
    file_to_owner, avatars = await _models_avatar_map(snap)
    return templates.TemplateResponse("_models_list.html", {
        "request": request, "snap": snap,
        "flash": {"ok": True, "msg": f"checked {len(records)} record(s) against HF"},
        "loaded_map": await _loaded_map(),
        "file_to_owner": file_to_owner, "avatars": avatars,
        "update_status": _update_status_map(snap),
    })


# ---------- Settings ----------

@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("settings.html", {"request": request, "token": db.get_setting("hf_token", ""), "flash": None})


@app.post("/settings", response_class=HTMLResponse)
async def settings_save(request: Request, hf_token: str = Form(""), action: str = Form("save")) -> HTMLResponse:
    hf_token = hf_token.strip()
    db.set_setting("hf_token", hf_token)
    flash = {"ok": True, "msg": "Saved."}
    if action == "test":
        try:
            ok, msg = await hf.validate_token(hf_token)
            flash = {"ok": ok, "msg": f"Saved. {msg}"}
        except Exception as e:  # noqa: BLE001
            flash = {"ok": False, "msg": f"Saved, but test failed: {e}"}
    return templates.TemplateResponse("settings.html", {"request": request, "token": hf_token, "flash": flash})


# ---------- Search ----------

@app.get("/search", response_class=HTMLResponse)
def search_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("search.html", {"request": request, "query": ""})


@app.get("/search/results", response_class=HTMLResponse)
async def search_results(request: Request, q: str = "", sort: str = "downloads",
                         limit: int = 30) -> HTMLResponse:
    error: str | None = None
    results = []
    avatars: dict[str, str] = {}
    browse_mode = not q.strip()  # empty query = "show me trending" browse view
    try:
        # Empty query is now valid — HF returns the top-N gguf models under `sort`
        results = await hf.search_models(q.strip(), sort=sort, limit=limit)
        owners = [(m.id.split("/", 1)[0] if "/" in m.id else m.id) for m in results]
        avatars = await hf.owner_avatars(owners)
    except httpx.HTTPStatusError as e:
        error = f"HF returned HTTP {e.response.status_code}"
    except httpx.HTTPError as e:
        error = f"network error: {e}"
    # Cross-check against local downloads so we can flag repos the user already has.
    # Must be checked against the FILESYSTEM, not just the history log — see the helper.
    downloaded = _downloaded_and_still_present()
    return templates.TemplateResponse("_search_results.html", {
        "request": request, "results": results, "error": error, "avatars": avatars,
        "browse_mode": browse_mode, "browse_sort": sort,
        "downloaded_by_repo": downloaded,
    })


def _files_present_on_disk() -> set[str]:
    """Every GGUF currently in the models directory, keyed the way download_history stores it.

    History rows use two shapes: bare "model.gguf" for downloads that predate the per-model
    subdirectory layout, and "stem/model.gguf" for everything since. Both are indexed so a
    caller can match either.
    """
    names: set[str] = set()
    try:
        snap = services.snapshot_models_dir()
    except OSError:
        return names
    for g in snap.ggufs:
        for part in list(g.parts) + list(g.companion_parts):
            names.add(part.name)
            if g.subdir:
                names.add(f"{g.subdir}/{part.name}")
    return names


def _downloaded_and_still_present() -> dict[str, list[str]]:
    """{repo_id: [filename, ...]} for repos whose files are ACTUALLY on disk right now.

    download_history is a log, not an inventory: a row stays `done` forever, so a model that
    was downloaded and later deleted keeps being reported as owned. Search then tells you that
    you already have something you do not, which is exactly backwards from the point of the
    flag — it exists to stop you re-downloading, and a false positive stops you downloading
    at all. Intersect the log with the filesystem.
    """
    present = _files_present_on_disk()
    out: dict[str, list[str]] = {}
    for repo, files in db.downloaded_files_by_repo().items():
        kept = [f for f in files if f in present or f.rsplit("/", 1)[-1] in present]
        if kept:
            out[repo] = kept
    return out


@app.get("/search/repo/{repo_id:path}", response_class=HTMLResponse)
async def search_repo(request: Request, repo_id: str) -> HTMLResponse:
    error: str | None = None
    gated: str = ""
    groups: list[dict] = []
    try:
        detail = await hf.repo_detail(repo_id)
        # group by shard_base
        by_base: dict[str, list[hf.HfFile]] = {}
        for f in detail.files:
            by_base.setdefault(f.shard_base, []).append(f)
        for base, files in by_base.items():
            files.sort(key=lambda x: (x.shard_index or 0, x.path))
            shard_total = files[0].shard_total if files[0].shard_index else None
            total_size = sum(x.size for x in files)
            groups.append({
                "shard_base": base,
                "shard_total": shard_total,
                "total_size_h": human_bytes(total_size),
                "total_size": total_size,
                "quant": next((x.quant for x in files if x.quant), None),
                "fit_chips": services.vram_fit_chips(total_size),
                "files": [{"path": x.path, "size": x.size, "size_h": human_bytes(x.size), "quant": x.quant} for x in files],
            })
        groups.sort(key=lambda g: (0 if g["files"][0]["path"].lower().endswith(".gguf") else 1, g["shard_base"].lower()))

        # Real context estimates need the model's layer/head counts, which only exist in the
        # GGUF header — so range-fetch ONE header for the repo. Those fields are properties of
        # the model and identical across its quants, so a single fetch (~1 MB) covers every
        # group; only file size varies, and we already know that per group.
        probe = next(
            (g for g in groups
             if g["files"][0]["path"].lower().endswith(".gguf")
             and "mmproj" not in g["files"][0]["path"].lower()),
            None,
        )
        # A repo's projector ships alongside its quants; it will be loaded with the model,
        # so its VRAM has to come off the budget or every multimodal estimate reads high.
        # Pick the smallest, matching what _find_mmproj does locally.
        mmproj_sizes = [
            g["total_size"] for g in groups
            if "mmproj" in g["files"][0]["path"].lower()
        ]
        mmproj_gb = (min(mmproj_sizes) / (1024 ** 3)) if mmproj_sizes else 0.0

        if probe:
            summary = await hf.gguf_header(repo_id, probe["files"][0]["path"])
            # A gated repo lists its files publicly but refuses the weights, so estimates
            # would silently vanish with no explanation. Surface the reason instead.
            gated = hf.gated_reason(repo_id) if not summary else ""
            if summary:
                for g in groups:
                    first = g["files"][0]["path"].lower()
                    if not first.endswith(".gguf") or "mmproj" in first:
                        continue
                    g["estimates"] = _preset_estimates(summary, g["total_size"], mmproj_gb)
                    g["native_ctx"] = (summary.get("model") or {}).get("context_length") or 0
    except httpx.HTTPStatusError as e:
        error = f"HF returned HTTP {e.response.status_code} for {repo_id}"
    except httpx.HTTPError as e:
        error = f"network error: {e}"
    return templates.TemplateResponse("_repo_files.html", {"request": request, "repo_id": repo_id, "groups": groups, "error": error, "gated": gated})


# ---------- Downloads ----------

_QUEUED_CHIP = (
    '<a href="/downloads" title="{title}" '
    'class="shrink-0 inline-flex items-center gap-1 rounded-md bg-emerald-100 dark:bg-emerald-950 '
    'text-emerald-800 dark:text-emerald-300 px-3 py-1.5 text-xs font-medium">'
    '✓ {label}</a>'
)


def _preset_estimates(summary: dict, size_bytes: int, mmproj_gb: float = 0.0) -> list[dict]:
    """Fast / Balanced / Long-ctx context estimates for a model we have NOT downloaded.

    Runs the very same autoconfig fit math used on local models, so a search-page estimate
    and the eventual Config recommendation agree instead of being two different guesses.
    Returns [] when there is nothing meaningful to show (no GPU backend, or the GGUF header
    lacks the fields needed to size a KV cache).
    """
    backends = []
    for name, vram in services._fit_backends().items():
        backends.append({
            "name": name, "vendor": "cuda", "vram_gb": vram,
            "gpu_count": hw.gpu_count_for(name), "card_vram_gb": hw.card_vram_gb_for(name),
            "host_ram_gb": hw.host_ram_gb(),
            "baseline": {},
        })
    if not backends or not summary:
        return []
    out: list[dict] = []
    for key, label in (("fast", "Fast"), ("balanced", "Balanced"), ("long-ctx", "Long ctx")):
        try:
            rec = autoconfig.analyze(
                summary=summary, file_size=size_bytes, backends=backends,
                preset=key, models_dir=None, section_name="",
                mmproj_gb_override=(mmproj_gb or None),
            )
        except Exception:  # noqa: BLE001 — an estimate must never break the search page
            return []
        if rec.error or not rec.recommended_ctx:
            continue
        chosen = next((p for p in rec.presets if p.key == rec.active_preset), None)
        out.append({
            "key": key,
            "label": label,
            "ctx": rec.recommended_ctx,
            "ctx_h": autoconfig.format_ctx(rec.recommended_ctx),
            "gpu_layers": chosen.gpu_layers if chosen else 0,
            "total_layers": chosen.total_layers if chosen else 0,
            "speed_pct": round((chosen.speed_score if chosen else 1.0) * 100),
            "offload": bool(chosen and chosen.offload_kind),
        })
        # Collapse duplicates: a model that fits fully at native has one real answer, and
        # three identical chips would imply choices that don't exist.
        if len(out) > 1 and out[-1]["ctx"] == out[0]["ctx"] and not out[-1]["offload"]:
            out.pop()
    return out


def _model_stem(filename: str) -> str:
    """Basename minus .gguf extension. Used as the subdir name for one-dir-per-model layout."""
    b = Path(filename).name
    return b[:-5] if b.lower().endswith(".gguf") else b


def _dest_for_main(main_filename: str) -> tuple[str, str]:
    """(subdir, dest_filename) for a MAIN model file. Subdir = filename stem."""
    stem = _model_stem(main_filename)
    return stem, f"{stem}/{Path(main_filename).name}"


def _dest_for_companion(main_stem: str, companion_filename: str) -> str:
    """Place a companion file (mmproj, tokenizer.model, chat_template) inside the main model's subdir."""
    return f"{main_stem}/{Path(companion_filename).name}"


@app.post("/download", response_class=HTMLResponse)
async def download_single(repo_id: str = Form(...), path: str = Form(...), size: int = Form(0)) -> HTMLResponse:
    base = Path(path).name
    if "mmproj" in base.lower():
        # Standalone mmproj download: put in a subdir named after its own stem
        stem = _model_stem(base)
        filename = f"{stem}/{base}"
        manager.enqueue(repo_id=repo_id, hf_path=path, filename=filename, total_bytes=size)
        return HTMLResponse(_QUEUED_CHIP.format(label="Queued", title=f"Queued {filename} — see Downloads"))

    # Main GGUF: subdir = stem. Companions from the same repo are queued alongside it.
    main_stem, filename = _dest_for_main(base)
    manager.enqueue(repo_id=repo_id, hf_path=path, filename=filename, total_bytes=size)
    extras = 0
    try:
        detail = await hf.repo_detail(repo_id)
        for f in detail.files:
            fp_name = Path(f.path).name
            if "mmproj" in fp_name.lower():
                manager.enqueue(
                    repo_id=repo_id,
                    hf_path=f.path,
                    filename=_dest_for_companion(main_stem, fp_name),
                    total_bytes=f.size,
                )
                extras += 1
                if extras >= 4:
                    break  # cap: some repos have many mmproj variants

        # A speculative-decoding head, when the repo ships one for this model. Only the
        # SMALLEST is taken: repos commonly publish BF16/F16/Q8_0/Q4_0 of the same head, they
        # are 60-170 MB, they sit in VRAM all session, and draft quality below Q8 barely moves
        # the acceptance rate. Queueing all four would waste bandwidth and disk to no purpose.
        #
        # These used to be skipped entirely after community draft models segfaulted
        # llama-server. That was a generic draft paired with an unrelated model; a head shipped
        # in the model's own repo is trained against it, and llama.cpp drives it through
        # draft-mtp rather than draft-simple. Autoconfig still only PROPOSES enabling it.
        heads = [f for f in detail.files
                 if f.path.lower().endswith(".gguf")
                 and "mmproj" not in Path(f.path).name.lower()
                 and autoconfig._looks_like_draft(Path(f.path).name)]
        if heads:
            head = min(heads, key=lambda f: f.size or 0)
            manager.enqueue(
                repo_id=repo_id,
                hf_path=head.path,
                filename=_dest_for_companion(main_stem, Path(head.path).name),
                total_bytes=head.size,
            )
            extras += 1
    except httpx.HTTPError:
        pass  # non-fatal — companions can be fetched manually later

    label = "Queued" if not extras else f"Queued (+ {extras} companion)"
    return HTMLResponse(_QUEUED_CHIP.format(label=label, title=f"Queued {filename}" + (f" and {extras} companion file(s)" if extras else "")))


@app.post("/download/multi", response_class=HTMLResponse)
async def download_multi(repo_id: str = Form(...), shard_base: str = Form(...)) -> HTMLResponse:
    try:
        detail = await hf.repo_detail(repo_id)
    except httpx.HTTPError as e:
        return HTMLResponse(
            f'<span class="shrink-0 inline-flex items-center rounded-md bg-red-100 dark:bg-red-950 '
            f'text-red-800 dark:text-red-300 px-3 py-1.5 text-xs font-medium" title="{e}">HF error</span>'
        )
    # All shards + companion mmproj go into the same subdir named after the shard base.
    subdir = Path(shard_base).stem  # strip .gguf
    count = 0
    mmproj_count = 0
    for f in detail.files:
        if f.shard_base == shard_base:
            base = Path(f.path).name
            manager.enqueue(repo_id=repo_id, hf_path=f.path, filename=f"{subdir}/{base}", total_bytes=f.size)
            count += 1
    # Auto-queue any mmproj files in the same repo into the same subdir
    for f in detail.files:
        fp_name = Path(f.path).name
        if "mmproj" in fp_name.lower():
            manager.enqueue(repo_id=repo_id, hf_path=f.path, filename=_dest_for_companion(subdir, fp_name), total_bytes=f.size)
            mmproj_count += 1
            if mmproj_count >= 4:
                break
    label = f"Queued {count} shards"
    if mmproj_count:
        label += f" + {mmproj_count} mmproj"
    return HTMLResponse(_QUEUED_CHIP.format(label=label, title=f"Queued {count} shards + {mmproj_count} companion files from {repo_id} into subdir '{subdir}/'"))


@app.post("/download/url", response_class=HTMLResponse)
async def download_url(url: str = Form(...), filename: str = Form("")) -> HTMLResponse:
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return HTMLResponse('<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">URL must start with http:// or https://</div>')
    if not filename.strip():
        from urllib.parse import urlparse, unquote
        p = urlparse(url).path
        filename = unquote(p.rsplit("/", 1)[-1]) if p else ""
    filename = filename.strip()
    if not filename:
        return HTMLResponse('<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">could not derive filename — set one explicitly</div>')
    if "/" in filename or ".." in filename:
        return HTMLResponse('<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">filename must be a plain basename</div>')

    total = 0
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            r = await client.head(url)
            if r.status_code < 400:
                total = int(r.headers.get("content-length") or 0)
    except httpx.HTTPError:
        pass  # size will be filled in during download

    # URL imports also get their own subdir (per-filename-stem, so main + optional mmproj coexist if named right)
    stem = _model_stem(filename)
    final_filename = f"{stem}/{filename}"
    manager.enqueue_url(url=url, filename=final_filename, total_bytes=total)
    return HTMLResponse(
        f'<div class="rounded-md bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 px-3 py-2 text-sm">'
        f'Queued <span class="font-mono">{filename}</span>. See progress below.</div>'
    )


@app.get("/downloads", response_class=HTMLResponse)
def downloads_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("downloads.html", {"request": request, "jobs": manager.snapshot()})


@app.get("/downloads/rows", response_class=HTMLResponse)
def downloads_rows(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("_download_rows.html", {"request": request, "jobs": manager.snapshot()})


@app.post("/downloads/{job_id}/cancel")
def downloads_cancel(job_id: str) -> Response:
    manager.cancel(job_id)
    return Response(status_code=204)


@app.post("/downloads/clear", response_class=HTMLResponse)
def downloads_clear(request: Request) -> HTMLResponse:
    manager.clear_finished()
    return templates.TemplateResponse("_download_rows.html", {"request": request, "jobs": manager.snapshot()})


# ---------- models.ini ----------

def _config_context(request: Request, flash: dict | None = None) -> dict:
    return {
        "request": request,
        "sections": ini.list_sections(),
        "unregistered": ini.unregistered_gguf_stems(),
        "raw": ini.raw_text(),
        "backups": ini.list_backups(),
        "ini_path": str(settings.models_ini_path),
        "llama_backends": services._effective_container_names(),
        "flash": flash,
    }


@app.get("/config", response_class=HTMLResponse)
def config_page(request: Request, saved: str = "", deleted: str = "", err: str = "", edit: str = "") -> HTMLResponse:
    flash = None
    if saved:
        flash = {"ok": True, "msg": f"Saved section [{saved}]."}
    elif deleted:
        flash = {"ok": True, "msg": f"Deleted section [{deleted}]."}
    elif err:
        flash = {"ok": False, "msg": err}
    ctx = _config_context(request, flash)
    # ?edit=<section> makes the page open straight into that section's form, which is how
    # direct links from elsewhere (e.g. a model's "edit models.ini") arrive here.
    ctx["edit_section"] = edit if (edit and edit in ini.section_names()) else ""
    return templates.TemplateResponse("config.html", ctx)


@app.get("/config/section/new", response_class=HTMLResponse)
def config_section_new(request: Request, name: str = "") -> HTMLResponse:
    warn = None
    if not name:
        warn = "Pick a filename first."
    elif not ini.valid_section_name(name):
        warn = f"Invalid section name: {name!r}"
    elif name in ini.section_names():
        warn = f"Section [{name}] already exists — use Edit instead."

    values: dict[str, str] = {}
    hints: list[str] = []
    if not warn:
        vals2, hints2 = _gguf_hints_for(name)
        values, hints = vals2, hints2

    return templates.TemplateResponse("_section_form.html", {
        "request": request,
        "mode": "new",
        "section_name": name,
        "values": values,
        "extras_text": "",
        "form_tiers": ini.FORM_TIERS,
        "warning": warn,
        "smart_hints": hints,
        "auto_filled": bool(values),
    })


def _resolve_section_gguf(name: str) -> tuple[Path | None, str, str | None]:
    """(gguf_path, model_rel, rel) for a section. rel is models-dir-relative, model_rel is
    the container-absolute /models/... path llama-server wants, or "" for flat layouts.

    Resolution order matters. An explicit `model =` wins over deriving the file from the
    section name, because (a) a section can be renamed to give the model a short API id, and
    (b) a subdir can hold several quants, where re-deriving would silently pick whichever
    sorts first rather than the one this section is actually configured for.
    """
    rel = ini.section_file_rel(name)
    explicit = ((ini.get_section(name) or {}).get("model") or "").strip()
    if explicit and rel and (settings.models_dir / rel).is_file():
        return settings.models_dir / rel, (explicit if "/" in rel else ""), rel

    if rel is not None and "/" in rel:
        subdir_name = rel.split("/", 1)[0]
        subdir_path = settings.models_dir / subdir_name
        if subdir_path.is_dir():
            # Prefer non-mmproj as the main GGUF (mmproj is the companion multimodal projector)
            gguf_files = sorted([q for q in subdir_path.iterdir() if q.is_file() and q.suffix.lower() == ".gguf"])
            main_files = [q for q in gguf_files if "mmproj" not in q.name.lower()] or gguf_files
            if main_files:
                # first shard has the metadata
                return main_files[0], f"/models/{subdir_name}/{main_files[0].name}", rel
        return None, "", rel
    if rel is not None:
        return settings.models_dir / rel, "", rel
    return settings.models_dir / f"{name}.gguf", "", rel


def _gguf_hints_for(name: str) -> tuple[dict[str, str], list[str]]:
    gguf_path, model_rel, rel = _resolve_section_gguf(name)

    if gguf_path is None or not gguf_path.is_file():
        return {}, [f"No GGUF found for `{name}` — cannot suggest defaults from metadata."]
    try:
        summary = gguf_meta.summarize(gguf_meta.read_raw(gguf_path))
    except (gguf_meta.GgufMetaError, OSError) as e:
        return {}, [f"Could not read GGUF: {e}"]
    values, hints = ini.suggest_defaults(summary)
    if model_rel:
        values["model"] = model_rel
        # Auto-detect a companion mmproj (multimodal projector — vision, audio, etc.)
        # in the same subdir so multimodal models load out of the box.
        subdir_path = Path(model_rel).parent  # e.g. /models/<stem>
        try:
            for p in Path(str(subdir_path).replace("/models", str(settings.models_dir), 1)).iterdir():
                if p.is_file() and p.suffix.lower() == ".gguf" and "mmproj" in p.name.lower():
                    values["mmproj"] = f"/models/{Path(model_rel).parts[-2]}/{p.name}"
                    hints.insert(0, f"Companion mmproj found → `mmproj = {values['mmproj']}` pre-filled.")
                    break
        except OSError:
            pass
        # Distinguish real sharding (multi-part files) from "single file in a subdir"
        from .utils import shard_key as _sk
        _, part_idx, part_total = _sk(Path(model_rel).name)
        if part_idx is not None and part_total and part_total > 1:
            hints.insert(0, f"Sharded model ({part_total} parts) → `model = {model_rel}` pre-filled to the first shard; llama-server auto-loads the rest.")
        else:
            hints.insert(0, f"Model lives in a subdir → `model = {model_rel}` pre-filled with the absolute path.")
    return values, hints


@app.get("/config/section/{name}/edit", response_class=HTMLResponse)
def config_section_edit(request: Request, name: str, reset: int = 0) -> HTMLResponse:
    # This endpoint returns a PARTIAL, designed to be swapped into /config by HTMX. Reaching
    # it by ordinary navigation (the "edit models.ini" link on a model page, a bookmark, a
    # refresh) renders the fragment with no base.html — so no stylesheet, no nav, just raw
    # form controls on white. Bounce those requests to the real page and let it open the
    # section instead.
    if request.headers.get("HX-Request", "").lower() != "true":
        from urllib.parse import quote
        return Response(status_code=303, headers={"Location": f"/config?edit={quote(name)}"})
    vals = ini.get_section(name)
    if vals is None and not reset:
        return HTMLResponse(f"unknown section: {name}", status_code=404)
    if reset:
        form_vals, hints = _gguf_hints_for(name)
        extras_text = ""
        hints = ["Reset to GGUF-derived defaults. Nothing saved yet — click Save to apply."] + hints
    else:
        form_vals, extras_text = ini.split_section_for_form(vals or {})
        hints = []
    return templates.TemplateResponse("_section_form.html", {
        "request": request,
        "mode": "edit",
        "section_name": name,
        "values": form_vals,
        "extras_text": extras_text,
        "form_tiers": ini.FORM_TIERS,
        "warning": None,
        "smart_hints": hints,
        "auto_filled": reset == 1,
    })


@app.post("/config/section/{name}")
async def config_section_save(request: Request, name: str) -> Response:
    if not ini.valid_section_name(name):
        return Response(status_code=200, headers={"HX-Redirect": f"/config?err=invalid+section+name+{name}"})
    form = await request.form()
    values: dict[str, str] = {}
    for f in ini.ALL_FIELDS:
        raw = form.get(f"fld_{f.key}", "")
        if f.kind == "bool":
            values[f.key] = "true" if raw else ""
        else:
            values[f.key] = str(raw).strip()
    extras = str(form.get("extras", ""))
    # Checked BEFORE the write: a section that did not exist a moment ago is a new model,
    # and only a new model gets a default backend. Re-running this on an ordinary save would
    # undo a deliberate removal from a connection.
    is_new_section = name not in ini.section_names()
    try:
        ini.upsert_section(name, values, extras)
    except Exception as e:  # noqa: BLE001
        return Response(status_code=200, headers={"HX-Redirect": f"/config?err=save+failed:+{e}"})
    # Whether a model can accept an image is decided here (by the presence of `mmproj`) but
    # enforced in OpenWebUI, which otherwise shows the image-upload control on everything and
    # only fails at inference with "image input is not supported". Push it across on every
    # save, so adding or removing a projector updates the UI that people actually click.
    # No restart: the `model` table is ordinary app data, not PersistentConfig.
    try:
        services.sync_openwebui_capabilities()
    except Exception:  # noqa: BLE001 -- saving the section must not depend on OpenWebUI
        pass
    # A brand-new model is offered on the GPU backends by default. Without this it lands in
    # models.ini, works, and is invisible in OpenWebUI because every connection carries an
    # explicit whitelist that predates it.
    note = ""
    if is_new_section:
        try:
            ok, msg = services.assign_new_model_to_gpu(name)
            if ok and msg:
                note = f"&note={msg}"
        except Exception:  # noqa: BLE001
            pass
    return Response(status_code=200, headers={"HX-Redirect": f"/config?saved={name}{note}"})


def _container_baseline(name: str) -> list[str]:
    """Return the Config.Cmd list for a running container, or [] on failure."""
    try:
        client = services._docker_client()
        if client is None:
            return []
        c = client.containers.get(name)
        return list((c.attrs or {}).get("Config", {}).get("Cmd") or [])
    except Exception:  # noqa: BLE001
        return []


@app.get("/config/section/{name}/autoconfig", response_class=HTMLResponse)
async def config_autoconfig(request: Request, name: str, preset: str = "",
                            sessions: int = 1, spec: str = "") -> HTMLResponse:
    import json as _json
    sessions = max(1, min(int(sessions or 1), 8))

    gguf_path, model_rel, rel = _resolve_section_gguf(name)

    if gguf_path is None or not gguf_path.is_file():
        return templates.TemplateResponse("_autoconfig_panel.html", {
            "request": request,
            "rec": autoconfig.Recommendation(plans=[], recommended_backend="", recommended_ctx=0,
                                             values={}, values_minimal={}, baseline_redundant={},
                                             quirks=[], unavailable=[], current_diff=[],
                                             error=f"No GGUF found for '{name}'."),
        })

    try:
        summary = gguf_meta.summarize(gguf_meta.read_raw(gguf_path))
    except (gguf_meta.GgufMetaError, OSError) as e:
        return templates.TemplateResponse("_autoconfig_panel.html", {
            "request": request,
            "rec": autoconfig.Recommendation(plans=[], recommended_backend="", recommended_ctx=0,
                                             values={}, values_minimal={}, baseline_redundant={},
                                             quirks=[], unavailable=[], current_diff=[],
                                             error=f"Failed to read GGUF: {e}"),
        })

    file_size = gguf_path.stat().st_size
    # If sharded, sum all shard sizes for accurate model_gb. But exclude companions
    # (mmproj, draft heads) that happen to sit in the same subdir — those aren't
    # part of the main model's parameter footprint and are budgeted separately.
    if rel and "/" in rel and gguf_path is not None:
        try:
            file_size = sum(
                p.stat().st_size for p in gguf_path.parent.iterdir()
                if p.is_file() and p.suffix.lower() == ".gguf"
                and not ini._is_companion(p.name)
            )
        except OSError:
            pass

    # Assemble backend info from whatever we effectively see (env whitelist OR auto-discovery)
    backend_list = []
    for bn in services._effective_container_names():
        vram = hw.vram_gb_for(bn)
        if vram <= 0:
            continue  # CPU backends have no VRAM budget; autoconfig can't do KV math on them yet
        gpu_count = hw.gpu_count_for(bn)
        # detect vendor via hw sampler cache (or docker inspect image)
        vendor = "unknown"
        try:
            client = services._docker_client()
            if client is not None:
                img = ((client.containers.get(bn).image.tags or [""]) or [""])[0].lower()
                if "rocm" in img:
                    vendor = "rocm"
                elif "cuda" in img:
                    vendor = "cuda"
        except Exception:  # noqa: BLE001
            pass
        cmd = _container_baseline(bn)
        base = autoconfig.parse_baseline(cmd) if cmd else {}
        backend_list.append({"name": bn, "vendor": vendor, "vram_gb": float(vram),
                             "gpu_count": gpu_count, "card_vram_gb": hw.card_vram_gb_for(bn),
                             "host_ram_gb": hw.host_ram_gb(),
                             "baseline": base})

    # Existing section (for diff)
    current_section = ini.get_section(name)

    # Detect subdir (if the resolved rel had one)
    model_subdir = ""
    if rel and "/" in rel:
        model_subdir = rel.split("/", 1)[0]

    # Fold in whatever llama-server has logged since the last look. Rate-limited internally and
    # wrapped so a log-format change or a docker hiccup costs the panel its measurements rather
    # than costing the user the page.
    try:
        telemetry.ingest(services._effective_container_names())
        tel = telemetry.stats_for(model_path=model_rel, alias=name)
    except Exception:  # noqa: BLE001
        tel = telemetry.Stats()

    rec = autoconfig.analyze(
        summary=summary,
        file_size=file_size,
        backends=backend_list,
        model_rel=model_rel,
        current_section=current_section,
        preset=preset,
        n_sessions=sessions,
        models_dir=settings.models_dir,
        section_name=name,
        model_subdir=model_subdir,
        spec_profile=spec,
    )

    # Prepare per-plan row map for the template + which ctx columns to show
    plan_row_map = {p.name: {r.ctx: r for r in p.rows} for p in rec.plans}
    all_ctx = sorted({r.ctx for p in rec.plans for r in p.rows})
    # pick a compact set: some small, some near the max/native
    interesting = set()
    for target in (8192, 32768, 65536, 131072, 196608, 262144):
        if target in all_ctx:
            interesting.add(target)
    for p in rec.plans:
        if p.max_ctx:
            interesting.add(p.max_ctx)
    native = summary.get("model", {}).get("context_length")
    if isinstance(native, int) and native in all_ctx:
        interesting.add(native)
    ctx_columns = sorted(interesting)[:6]  # cap to 6 columns for width

    return templates.TemplateResponse("_autoconfig_panel.html", {
        "request": request,
        "rec": rec,
        "section_name": name,
        "model_display": model_rel or (rel or f"{name}.gguf"),
        "file_size": file_size,
        "arch": summary.get("arch"),
        "params": (summary.get("general") or {}).get("params"),
        "ctx_columns": ctx_columns,
        "plan_row_map": plan_row_map,
        "format_ctx": autoconfig.format_ctx,
        "tel": tel,
        "values_json": _json.dumps(rec.values),
        "values_minimal_json": _json.dumps(rec.values_minimal),
        # Full offload frontier for the custom slider. Every entry is an achievable
        # config, so the slider snaps to real points rather than interpolating.
        "frontier_json": _json.dumps([
            {
                "ctx": p.ctx,
                "total_ctx": p.ctx * sessions,
                "ngl": p.ngl,
                "n_cpu_moe": p.n_cpu_moe,
                "kind": p.offload_kind,
                "gpu_layers": p.gpu_layers,
                "total_layers": p.total_layers,
                "gpu_gb": p.gpu_gb,
                "kv_gb": p.kv_gb,
                "speed": p.speed_score,
            }
            for p in rec.frontier
        ]),
    })


@app.post("/config/section/{name}/rename")
async def config_section_rename(request: Request, name: str) -> Response:
    """Rename a preset. The section name IS the model id llama-server serves, so this is how
    you give a model a short, human name — `model =` keeps pointing at the same file."""
    form = await request.form()
    new = (form.get("new_name") or "").strip()
    if not new:
        return Response(status_code=200, headers={"HX-Redirect": "/config?err=new+name+required"})
    if new == name:
        return Response(status_code=200, headers={"HX-Redirect": f"/config?edit={new}"})
    if not ini.valid_section_name(new):
        return Response(status_code=200,
                        headers={"HX-Redirect": f"/config?err=invalid+name:+{new}"})
    if new in ini.section_names():
        return Response(status_code=200,
                        headers={"HX-Redirect": f"/config?err=section+already+exists:+{new}"})
    if not ini.rename_section(name, new):
        return Response(status_code=200, headers={"HX-Redirect": f"/config?err=rename+failed"})

    # Reconcile OpenWebUI's per-connection whitelists in ONE pass (each write restarts
    # open-webui, so doing this as two passes would cost two restarts).
    #
    # Two things have to happen together. Carry the old id across, or the model silently
    # vanishes from the picker. And drop ids no section provides any more: OpenWebUI RENDERS
    # its whitelist rather than intersecting it with what the backend reports, so a dead id
    # stays visible in the model picker and fails with "model not found" only when someone
    # tries to use it — the one failure mode neither end shows you.
    try:
        known = set(ini.section_names())
        for c in (services.openwebui_state().get("connections") or []):
            ids = c.get("model_ids") or []
            if not ids:
                continue  # empty whitelist = "offer everything"; nothing to reconcile
            fixed = [new if m == name else m for m in ids]
            fixed = [m for m in fixed if m in known]
            # Never write an empty list: that would flip the connection from a filter to
            # "offer every model", quietly widening what this backend exposes.
            if fixed and fixed != ids:
                services.set_openwebui_model_filter(c["url"], fixed)
    except Exception:  # noqa: BLE001 — renaming must succeed even if OpenWebUI is unreachable
        pass

    return Response(status_code=200, headers={"HX-Redirect": f"/config?saved={new}&edit={new}"})


@app.post("/config/section/{name}/delete")
def config_section_delete(name: str) -> Response:
    ok = ini.delete_section(name)
    if not ok:
        return Response(status_code=200, headers={"HX-Redirect": f"/config?err=no+such+section:+{name}"})
    return Response(status_code=200, headers={"HX-Redirect": f"/config?deleted={name}"})


# ---------- Containers ----------

def _stats_by_name() -> dict:
    return {name: hw.stats_for(name) for name in services._effective_container_names()}


def _perf_by_name() -> dict:
    """Sparkline-ready view of the rolling history the hw sampler already keeps.

    Percentages are pinned to a 0-100 scale and VRAM to the card total, so the lines stay
    comparable between refreshes instead of auto-rescaling to whatever the current window
    happens to contain.
    """
    out: dict[str, dict] = {}
    for name in services._effective_container_names():
        pts = hw.history_for(name)
        if not pts:
            continue
        st = hw.stats_for(name)
        vram_total = st.gpu.vram_total_gb if (st.ok and st.gpu) else 0.0
        span_s = (pts[-1].ts - pts[0].ts) if len(pts) > 1 else 0.0
        mins = int(span_s // 60)
        cards = (st.gpu.cards if (st.ok and st.gpu) else []) or []
        frees = [c.vram_free_gb for c in cards]
        out[name] = {
            "points": len(pts),
            "has_gpu": bool(st.ok and st.gpu),
            "window_label": (f"last {mins}m" if mins >= 1 else f"last {int(span_s)}s"),
            "gpu_util": hw.sparkline([p.gpu_util for p in pts], 100.0),
            "vram": hw.sparkline([p.vram_used_gb for p in pts], vram_total or None),
            "cpu": hw.sparkline([p.cpu_pct for p in pts], None),
            "mem": hw.sparkline([p.mem_used_gb for p in pts], None),
            "cur_gpu_util": pts[-1].gpu_util,
            "cur_vram": pts[-1].vram_used_gb,
            "vram_total": vram_total,
            "cur_cpu": pts[-1].cpu_pct,
            "cur_mem": pts[-1].mem_used_gb,
            # spread between the most and least free card — the number that explains
            # "OOM on one GPU while the other looks fine"
            "imbalance_gb": round(max(frees) - min(frees), 2) if len(frees) > 1 else 0.0,
        }
    return out


@app.get("/containers", response_class=HTMLResponse)
async def containers_page(request: Request) -> HTMLResponse:
    backends = await services.snapshot_llama_backends()
    return templates.TemplateResponse("containers.html", {
        "request": request, "backends": backends, "stats": _stats_by_name(), "perf": _perf_by_name(),
        "prompts": db.list_prompts(),
        "openwebui": services.openwebui_state(),
        # Raw model ids a backend reports — what OpenWebUI's model_ids whitelist matches on.
        # Every llama backend serves the same models.ini, so the section names are the list.
        "all_model_ids": sorted(ini.section_names()),
    })


@app.post("/containers/sync-openwebui", response_class=HTMLResponse)
def containers_sync_openwebui() -> HTMLResponse:
    ok, msg = services.sync_openwebui_endpoints()
    if ok:
        return HTMLResponse(
            f'<div class="rounded-md bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 px-3 py-2 text-sm">✓ {msg}</div>'
        )
    return HTMLResponse(
        f'<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">Sync failed: {msg}</div>'
    )


@app.post("/containers/openwebui-align-capabilities", response_class=HTMLResponse)
def containers_openwebui_align() -> HTMLResponse:
    """Make OpenWebUI's per-model capabilities match models.ini, and drop stale records."""
    ok, msg = services.align_openwebui_capabilities()
    cls = ("bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300" if ok
           else "bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300")
    return HTMLResponse(f'<div class="rounded-md {cls} px-3 py-2 text-sm">{"✓" if ok else "Align failed:"} {msg}</div>')


@app.post("/containers/openwebui-filter", response_class=HTMLResponse)
async def containers_openwebui_filter(request: Request) -> HTMLResponse:
    """Set which models one OpenWebUI connection offers. No checked boxes = offer all."""
    form = await request.form()
    url = (form.get("url") or "").strip()
    if not url:
        return HTMLResponse(
            '<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">No connection specified.</div>'
        )
    model_ids = [str(v) for v in form.getlist("model_ids") if str(v).strip()]
    ok, msg = services.set_openwebui_model_filter(url, model_ids)
    cls = ("bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300" if ok
           else "bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300")
    mark = "✓" if ok else "Filter failed:"
    return HTMLResponse(f'<div class="rounded-md {cls} px-3 py-2 text-sm">{mark} {msg}</div>')


@app.post("/models/align-capability", response_class=HTMLResponse)
def models_align_capability(section: str = Form(...)) -> HTMLResponse:
    """Align OpenWebUI capabilities for a single models.ini section."""
    ok, msg = services.align_openwebui_capability_for(section)
    cls = ("bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300" if ok
           else "bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300")
    return HTMLResponse(f'<div class="rounded-md {cls} px-3 py-2 text-sm">{"✓" if ok else "Failed:"} {msg}</div>')


@app.post("/models/openwebui-visibility", response_class=HTMLResponse)
async def models_openwebui_visibility(request: Request) -> HTMLResponse:
    """Toggle one model's visibility on one OpenWebUI connection, from the Models page."""
    form = await request.form()
    url = (form.get("url") or "").strip()
    model_id = (form.get("model_id") or "").strip()
    show = str(form.get("show") or "").lower() in ("1", "true", "on", "yes")
    if not url or not model_id:
        return HTMLResponse(
            '<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">Missing model or connection.</div>'
        )
    ok, msg = services.toggle_openwebui_model(url, model_id, show, sorted(ini.section_names()))
    cls = ("bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300" if ok
           else "bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300")
    return HTMLResponse(f'<div class="rounded-md {cls} px-3 py-2 text-sm">{"✓" if ok else "Failed:"} {msg}</div>')


@app.get("/containers/{name}/dashboard", response_class=HTMLResponse)
async def containers_dashboard(request: Request, name: str) -> HTMLResponse:
    backends = await services.snapshot_llama_backends()
    b = next((x for x in backends if x.name == name), None)
    if b is None:
        return HTMLResponse(f"<span class='text-xs text-slate-500'>unknown backend: {name}</span>")
    return templates.TemplateResponse("_container_dashboard.html", {
        "request": request, "b": b, "stats": _stats_by_name(), "perf": _perf_by_name(),
    })


@app.post("/containers/{name}/restart", response_class=HTMLResponse)
def containers_restart(name: str) -> HTMLResponse:
    services.restart_llama_backend(name)
    # small chip: replaces the button when hx-swap=outerHTML; ignored when hx-swap=none
    return HTMLResponse(
        f'<span class="inline-flex items-center rounded-md bg-amber-100 dark:bg-amber-950 '
        f'text-amber-800 dark:text-amber-300 px-2.5 py-1 text-xs font-mono">restarting {name}…</span>'
    )


@app.post("/containers/{name}/test", response_class=HTMLResponse)
async def containers_test(name: str, prompt: str = Form(...), max_tokens: int = Form(256)) -> HTMLResponse:
    result = await services.test_prompt(name, prompt, max_tokens=max_tokens)
    if not result.get("ok"):
        return HTMLResponse(
            f'<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">'
            f'{result.get("err", "test failed")}</div>'
        )
    reply = (result.get("reply") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    tps = result.get("tokens_per_s")
    tps_bit = f" · {tps} tok/s" if tps else ""
    meta = (
        f'{result["model"]} · {result["completion_tokens"]} tok in {result["elapsed_s"]}s{tps_bit}'
        f' · prompt {result["prompt_tokens"]} tok'
    )
    return HTMLResponse(
        f'<div class="rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/40 p-3">'
        f'<pre class="text-xs whitespace-pre-wrap font-mono">{reply}</pre>'
        f'<div class="mt-2 text-[10px] text-slate-500 dark:text-slate-400 font-mono">{meta}</div>'
        f'</div>'
    )


@app.get("/containers/{name}/logs", response_class=HTMLResponse)
def containers_logs(name: str, q: str = "", level: str = "") -> HTMLResponse:
    ok, out = services.container_logs(name)
    if not ok:
        return HTMLResponse(f"<span class='text-red-400'>{out}</span>")

    if q or level:
        needle = q.lower()
        keep_error = level == "error"
        keep_warn = level in ("warn", "warning")
        filtered: list[str] = []
        for line in out.splitlines():
            low = line.lower()
            if needle and needle not in low:
                continue
            if keep_error and ("err" not in low and "error" not in low and "fatal" not in low):
                continue
            if keep_warn and ("err" not in low and "warn" not in low and "wrn" not in low and "error" not in low and "fatal" not in low):
                continue
            filtered.append(line)
        out = "\n".join(filtered)
        if not out:
            out = f"(no lines match q={q!r} level={level!r})"

    safe = out.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return HTMLResponse(safe or "(no output)")


# ---------- Prompt library ----------

@app.get("/prompts", response_class=HTMLResponse)
def prompts_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("prompts.html", {
        "request": request, "prompts": db.list_prompts(), "flash": None,
    })


@app.post("/prompts", response_class=HTMLResponse)
def prompts_add(request: Request, name: str = Form(...), body: str = Form(...)) -> HTMLResponse:
    name = name.strip()
    body = body.strip()
    if not name or not body:
        return templates.TemplateResponse("prompts.html", {
            "request": request, "prompts": db.list_prompts(),
            "flash": {"ok": False, "msg": "name and body are required"},
        })
    db.add_prompt(name, body)
    return templates.TemplateResponse("prompts.html", {
        "request": request, "prompts": db.list_prompts(),
        "flash": {"ok": True, "msg": f"saved '{name}'"},
    })


@app.post("/prompts/{pid}/delete", response_class=HTMLResponse)
def prompts_delete(request: Request, pid: int) -> HTMLResponse:
    ok = db.delete_prompt(pid)
    return templates.TemplateResponse("prompts.html", {
        "request": request, "prompts": db.list_prompts(),
        "flash": {"ok": ok, "msg": "deleted" if ok else "not found"},
    })


@app.get("/prompts/{pid}/body", response_class=HTMLResponse)
def prompts_body(pid: int) -> HTMLResponse:
    p = db.get_prompt(pid)
    if not p:
        return HTMLResponse("", status_code=404)
    return HTMLResponse(p["body"])
