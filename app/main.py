from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from . import autoconfig, db, gguf_meta, hf, hw, ini, services
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


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request) -> HTMLResponse:
    snap = services.snapshot_models_dir()
    file_to_owner, avatars = await _models_avatar_map(snap)
    return templates.TemplateResponse("models.html", {
        "request": request, "snap": snap, "flash": None,
        "loaded_map": await _loaded_map(),
        "file_to_owner": file_to_owner, "avatars": avatars,
        "update_status": _update_status_map(snap),
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

    lm = await _loaded_map()
    loaded_on = lm.get(stem, [])

    in_ini = stem in ini.section_names()
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

    return templates.TemplateResponse("model_local.html", {
        "request": request,
        "filename": filename,
        "path": str(path),
        "stem": stem,
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
async def search_results(request: Request, q: str = "", sort: str = "downloads") -> HTMLResponse:
    error: str | None = None
    results = []
    avatars: dict[str, str] = {}
    if q.strip():
        try:
            results = await hf.search_models(q.strip(), sort=sort)
            owners = [(m.id.split("/", 1)[0] if "/" in m.id else m.id) for m in results]
            avatars = await hf.owner_avatars(owners)
        except httpx.HTTPStatusError as e:
            error = f"HF returned HTTP {e.response.status_code}"
        except httpx.HTTPError as e:
            error = f"network error: {e}"
    return templates.TemplateResponse("_search_results.html", {
        "request": request, "results": results, "error": error, "avatars": avatars,
    })


@app.get("/search/repo/{repo_id:path}", response_class=HTMLResponse)
async def search_repo(request: Request, repo_id: str) -> HTMLResponse:
    error: str | None = None
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
    except httpx.HTTPStatusError as e:
        error = f"HF returned HTTP {e.response.status_code} for {repo_id}"
    except httpx.HTTPError as e:
        error = f"network error: {e}"
    return templates.TemplateResponse("_repo_files.html", {"request": request, "repo_id": repo_id, "groups": groups, "error": error})


# ---------- Downloads ----------

_QUEUED_CHIP = (
    '<a href="/downloads" title="{title}" '
    'class="shrink-0 inline-flex items-center gap-1 rounded-md bg-emerald-100 dark:bg-emerald-950 '
    'text-emerald-800 dark:text-emerald-300 px-3 py-1.5 text-xs font-medium">'
    '✓ {label}</a>'
)


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

    # Main GGUF: subdir = stem. Also auto-queue any mmproj found in the same HF repo.
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
    except httpx.HTTPError:
        pass  # non-fatal — user can grab mmproj manually later

    label = "Queued" if not extras else f"Queued (+ {extras} companion)"
    return HTMLResponse(_QUEUED_CHIP.format(label=label, title=f"Queued {filename}" + (f" and {extras} mmproj file(s)" if extras else "")))


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
def config_page(request: Request, saved: str = "", deleted: str = "", err: str = "") -> HTMLResponse:
    flash = None
    if saved:
        flash = {"ok": True, "msg": f"Saved section [{saved}]."}
    elif deleted:
        flash = {"ok": True, "msg": f"Deleted section [{deleted}]."}
    elif err:
        flash = {"ok": False, "msg": err}
    return templates.TemplateResponse("config.html", _config_context(request, flash))


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


def _gguf_hints_for(name: str) -> tuple[dict[str, str], list[str]]:
    # Resolve the section name to an actual file. Supports flat and subdir'd layouts.
    stems_map = ini._stems_present()
    rel = stems_map.get(name)
    gguf_path: Path | None = None
    model_rel = ""
    if rel is not None and "/" in rel:
        subdir_name = rel.split("/", 1)[0]
        subdir_path = settings.models_dir / subdir_name
        if subdir_path.is_dir():
            gguf_files = sorted([p for p in subdir_path.iterdir() if p.is_file() and p.suffix.lower() == ".gguf"])
            if gguf_files:
                gguf_path = gguf_files[0]  # first shard has the metadata
                model_rel = f"{subdir_name}/{gguf_files[0].name}"
    elif rel is not None:
        gguf_path = settings.models_dir / rel
    else:
        gguf_path = settings.models_dir / f"{name}.gguf"

    if gguf_path is None or not gguf_path.is_file():
        return {}, [f"No GGUF found for `{name}` — cannot suggest defaults from metadata."]
    try:
        summary = gguf_meta.summarize(gguf_meta.read_raw(gguf_path))
    except (gguf_meta.GgufMetaError, OSError) as e:
        return {}, [f"Could not read GGUF: {e}"]
    values, hints = ini.suggest_defaults(summary)
    if model_rel:
        values["model"] = model_rel
        hints.insert(0, f"Sharded model detected → `model = {model_rel}` pre-filled to point at the first shard. llama-server auto-loads the rest.")
    return values, hints


@app.get("/config/section/{name}/edit", response_class=HTMLResponse)
def config_section_edit(request: Request, name: str, reset: int = 0) -> HTMLResponse:
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
    try:
        ini.upsert_section(name, values, extras)
    except Exception as e:  # noqa: BLE001
        return Response(status_code=200, headers={"HX-Redirect": f"/config?err=save+failed:+{e}"})
    return Response(status_code=200, headers={"HX-Redirect": f"/config?saved={name}"})


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
async def config_autoconfig(request: Request, name: str, preset: str = "balanced") -> HTMLResponse:
    import json as _json

    # Resolve name → actual GGUF file (flat or subdir'd)
    stems_map = ini._stems_present()
    rel = stems_map.get(name)
    gguf_path: Path | None = None
    model_rel = ""
    if rel is not None and "/" in rel:
        subdir_name = rel.split("/", 1)[0]
        subdir_path = settings.models_dir / subdir_name
        if subdir_path.is_dir():
            # prefer non-mmproj gguf as the "main" file
            gguf_files = sorted([p for p in subdir_path.iterdir() if p.is_file() and p.suffix.lower() == ".gguf"])
            main_files = [p for p in gguf_files if "mmproj" not in p.name.lower()] or gguf_files
            if main_files:
                gguf_path = main_files[0]
                model_rel = f"/models/{subdir_name}/{main_files[0].name}"
    elif rel is not None:
        gguf_path = settings.models_dir / rel
    else:
        gguf_path = settings.models_dir / f"{name}.gguf"

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
    # If sharded, sum all shard sizes for accurate model_gb
    if rel and "/" in rel and gguf_path is not None:
        try:
            file_size = sum(p.stat().st_size for p in gguf_path.parent.iterdir()
                            if p.is_file() and p.suffix.lower() == ".gguf")
        except OSError:
            pass

    # Assemble backend info from whatever we effectively see (env whitelist OR auto-discovery)
    backend_list = []
    for bn in services._effective_container_names():
        vram = hw.vram_gb_for(bn)
        if vram <= 0:
            continue  # CPU backends have no VRAM budget; autoconfig can't do KV math on them yet
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
        backend_list.append({"name": bn, "vendor": vendor, "vram_gb": float(vram), "baseline": base})

    # Existing section (for diff)
    current_section = ini.get_section(name)

    # Detect subdir (if the resolved rel had one)
    model_subdir = ""
    if rel and "/" in rel:
        model_subdir = rel.split("/", 1)[0]

    rec = autoconfig.analyze(
        summary=summary,
        file_size=file_size,
        backends=backend_list,
        model_rel=model_rel,
        current_section=current_section,
        preset=preset,
        models_dir=settings.models_dir,
        section_name=name,
        model_subdir=model_subdir,
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
        "values_json": _json.dumps(rec.values),
        "values_minimal_json": _json.dumps(rec.values_minimal),
    })


@app.post("/config/section/{name}/delete")
def config_section_delete(name: str) -> Response:
    ok = ini.delete_section(name)
    if not ok:
        return Response(status_code=200, headers={"HX-Redirect": f"/config?err=no+such+section:+{name}"})
    return Response(status_code=200, headers={"HX-Redirect": f"/config?deleted={name}"})


# ---------- Containers ----------

def _stats_by_name() -> dict:
    return {name: hw.stats_for(name) for name in services._effective_container_names()}


@app.get("/containers", response_class=HTMLResponse)
async def containers_page(request: Request) -> HTMLResponse:
    backends = await services.snapshot_llama_backends()
    return templates.TemplateResponse("containers.html", {
        "request": request, "backends": backends, "stats": _stats_by_name(),
        "prompts": db.list_prompts(),
        "openwebui": services.openwebui_state(),
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


@app.get("/containers/{name}/dashboard", response_class=HTMLResponse)
async def containers_dashboard(request: Request, name: str) -> HTMLResponse:
    backends = await services.snapshot_llama_backends()
    b = next((x for x in backends if x.name == name), None)
    if b is None:
        return HTMLResponse(f"<span class='text-xs text-slate-500'>unknown backend: {name}</span>")
    return templates.TemplateResponse("_container_dashboard.html", {
        "request": request, "b": b, "stats": _stats_by_name(),
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
