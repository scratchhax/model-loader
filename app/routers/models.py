from __future__ import annotations

import asyncio

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from .. import autoconfig, db, gguf_meta, hf, ini, services
from ..config import settings
from ..utils import human_bytes
from ..web import templates

router = APIRouter()


def _section_caps(sections: list) -> dict[str, dict]:
    """{section name: capability facts} for the chips on the section cards.

    Everything is derived locally — shape from the GGUF header, modalities from the section's
    own projector, speculation from the saved profile plus the heads that exist — so the ini
    page says what the CONFIGured model can do even when OpenWebUI (whose records drive the
    models list) has never seen it. Header reads are the cached kind; this is cheap per render.
    """
    caps: dict[str, dict] = {}
    for s in sections:
        if not s.matched_file:
            continue
        vals = dict(s.items)
        shape = services.model_shape(settings.models_dir / s.matched_file)
        mmproj = (vals.get("mmproj") or "").strip()
        mods = services.projector_modalities(mmproj) if mmproj else frozenset()
        subdir = s.matched_file.split("/", 1)[0] if "/" in s.matched_file else ""
        # A draft head counts when it is named by the section, sits beside these weights, or
        # is baked into them (MTP layers) - the same three ways autoconfig resolves one.
        head_rel = (vals.get("spec-draft-model") or "").strip()
        head_ext = bool(head_rel) or bool(autoconfig._find_mtp(settings.models_dir, s.name, subdir))
        head_avail = head_ext or shape.internal_mtp
        prof_key = autoconfig.match_spec_profile(vals)
        prof = autoconfig.SPEC_PROFILE_BY_KEY.get(prof_key)
        if prof_key == "custom":
            drafts = bool((vals.get("spec-type") or "").strip())
        else:
            drafts = bool(prof and prof.spec_type and prof.needs_head)
        spec_on = head_avail and drafts
        mtp_mode = shape.internal_mtp or bool(prof and "mtp" in (prof.spec_type or ""))
        caps[s.name] = {
            "shape": shape if shape.known else None,
            "mods": mods,
            "proj_unknown": bool(mmproj) and not mods,
            "spec": ("mtp" if mtp_mode else "head") if spec_on else "",
            "head_idle": head_avail and not spec_on,
        }
    return caps


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
    """display_name -> what the last Hugging Face check says about this model.

    {'status': 'up-to-date'|'stale'|'unknown', 'reason': 'size'|'date'|'',
     'remote': 'YYYY-MM-DD', 'checked_at': ts, 'delta_days': int|None,
     'repo_id': str, 'parts': [(local rel path, repo path)]}

    Two joins had to be got right and neither was. The check records one row per DOWNLOADED
    file - a local relative path, per shard - while this map is keyed by the entry the list
    renders, whose display_name is the shard BASE and carries no subdirectory. So
    `checks.get(g.display_name)` matched nothing for any sharded model and nothing for any
    model stored in a subdirectory, which since the layout change is all of them: every model
    on the box scored "unknown" and no staleness was ever actually evaluated. The other join,
    local name against repo path, is fixed in hf._match_repo_path.

    The verdict itself is now size first. A commit date moves whenever anything in the repo's
    metadata is touched, so "the remote commit is newer than my file" is true of almost every
    model almost always and says nothing about the weights. A differing byte count is not
    ambiguous: the GGUF was rebuilt. Only when the size is unknown - an old row, or a repo the
    tree API would not describe - does this fall back to the date, and it says which test it
    used so the UI can be honest about how much the answer is worth.
    """
    from datetime import datetime, timezone
    checks = db.all_update_checks()

    def _parse(iso: str):
        try:
            core, _, frac = iso.partition(".")
            if frac:
                return datetime.fromisoformat(f"{core}.{frac.rstrip('Z')[:6]}+00:00")
            return datetime.fromisoformat(core.replace("Z", "+00:00"))
        except ValueError:
            return None

    out: dict[str, dict] = {}
    for g in snap.ggufs:
        # Every local path this entry covers, in the same form the download recorded it.
        # Companions included: an mmproj is folded into its model's row, so a projector that
        # was rebuilt has nowhere else to be reported and no other way to be refreshed.
        covered = list(g.parts) + list(g.companion_parts)
        rels = [f"{g.subdir}/{p.name}" if g.subdir else p.name for p in covered]
        rows = [(rel, checks[rel]) for rel in rels if rel in checks]
        if not rows:
            continue
        repo_id = rows[0][1].get("repo_id") or ""
        parts = [(rel, r.get("hf_path") or "") for rel, r in rows]
        checked_at = max(r.get("checked_at") or 0 for _rel, r in rows)

        # Size is per shard and decisive on its own: one shard of a different length means
        # the model was rebuilt, whatever the rest of them say.
        size_known = False
        size_differs = False
        by_name = {p.name: p for p in covered}
        for rel, r in rows:
            remote_size = r.get("hf_size")
            if remote_size is None:
                continue
            p = by_name.get(rel.rsplit("/", 1)[-1])
            if p is None:
                continue
            try:
                local_size = p.stat().st_size
            except OSError:
                continue
            size_known = True
            if int(remote_size) != local_size:
                size_differs = True

        remote_dts = [d for d in (_parse(r.get("hf_last_modified") or "") for _rel, r in rows) if d]
        newest = max(remote_dts) if remote_dts else None
        local_dt = datetime.fromtimestamp(g.mtime, tz=timezone.utc)
        delta_days = (newest - local_dt).days if newest else None
        remote_s = newest.strftime("%Y-%m-%d") if newest else ""

        if size_known:
            status, reason = ("stale", "size") if size_differs else ("up-to-date", "size")
        elif newest:
            status, reason = ("stale" if newest > local_dt else "up-to-date"), "date"
        else:
            status, reason = "unknown", ""

        out[g.display_name] = {
            "status": status, "reason": reason, "remote": remote_s,
            "checked_at": checked_at, "delta_days": delta_days,
            "repo_id": repo_id, "parts": parts,
        }
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


async def _models_list_ctx(request: Request, snap, flash=None) -> dict:
    """Everything _models_list.html needs, built in ONE place.

    The partial is rendered from four routes - the page, two deletes and the update check -
    and the three POST routes each assembled their own context by hand. They were missing
    `owui`, `badges` and `shapes`, so deleting one file silently stripped the shape chips, the
    per-connection availability toggles and every rating off every OTHER row until the next
    full reload. A partial swapped into a page has to come back with the same shape it had.
    """
    file_to_owner, avatars = await _models_avatar_map(snap)
    loaded_map = await _loaded_map()
    update_status = _update_status_map(snap)
    owui = await asyncio.to_thread(_owui_visibility)
    shapes = _shapes_for_files(snap)
    # What a section can DO is derived from its own config - GGUF header, its mmproj, its spec
    # profile - by the same helper the models.ini cards use. OpenWebUI's record is a copy of
    # that and can lag, or never have seen a section at all.
    try:
        section_caps = _section_caps(ini.list_sections())
    except Exception:  # noqa: BLE001 - an unreadable ini must not take the models page down
        section_caps = {}
    return {
        "request": request, "snap": snap, "flash": flash,
        "file_to_owner": file_to_owner, "avatars": avatars,
        "owui": owui,
        "rows": _model_rows(snap, shapes, loaded_map, update_status, owui,
                            db.badges_by_alias(), section_caps),
    }


@router.get("/models", response_class=HTMLResponse)
async def models_page(request: Request) -> HTMLResponse:
    snap = services.snapshot_models_dir()
    return templates.TemplateResponse(
        "models.html", await _models_list_ctx(request, snap))


def _model_rows(snap, shapes: dict, loaded_map: dict, update_status: dict,
                owui: dict, badges_by_alias: dict, section_caps: dict | None = None) -> list[dict]:
    """One row per SERVED MODEL, not per file, plus the facets the list groups and filters on.

    A file and a model are not the same thing. gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf backs two
    models.ini sections - its own and `voice-cpu` - with different settings over the same
    weights, and `voice-qwen` is the only name Qwen3.5-4B is served under at all. Keyed by
    file, those meta-models were a comma-separated footnote on somebody else's row: no state
    of their own, no ratings of their own, nowhere to click through to the section that
    defines them. They are the thing you actually call from OpenWebUI, so they get a row.

    What belongs to the FILE stays on the file's row and appears once: the byte count, the
    checkbox, delete. A second section over the same weights frees nothing when you remove it
    and must not double the group totals, so it carries `primary: False`, no size of its own
    and no delete button - its sort key is still the file's size, so the two sort together
    rather than the alias sinking to the bottom of a size sort.

    What belongs to the SECTION is per row and is why the split is worth making: whether it is
    resident right now, whether speculation is configured for it, which backends offer it, and
    what you have rated it. All four differ between two sections over one file.

    Family and tier come from the file either way - same weights, same placement - so an alias
    sorts into the same group as the file it is built on rather than off under its own name.
    """
    card_gb = services.largest_card_gb()
    caps = (owui or {}).get("caps") or {}
    split = {g.display_name: services.split_quant(g.stem) for g in snap.ggufs}
    # Needs the whole set at once: whether a variant has a base to fold onto is a fact about
    # what else is on disk, not about its own filename.
    canon = services.canonical_families({fam for fam, _q in split.values()})
    rows: list[dict] = []
    for g in snap.ggufs:
        family, quant = split[g.display_name]
        family = canon.get(family, family)
        tier, tier_label, tier_short = services.size_tier(g.total_bytes, card_gb)
        shape = (shapes or {}).get(g.display_name)
        us = (update_status or {}).get(g.display_name)

        # The section named after the file leads, so the row that owns the byte count is the
        # one whose name matches it; a file served only under aliases has no such row and the
        # first alias owns it instead.
        ids = sorted(g.aliases or [], key=lambda a: (a != g.stem, a.lower()))
        if not ids:
            ids = [g.stem]

        for n, served_id in enumerate(ids):
            cap = caps.get(served_id) or {}
            # The section's own config wins over OpenWebUI's copy of it, and is the only source
            # that exists for a section OpenWebUI has never been told about. The mismatch flag
            # still comes from OpenWebUI - disagreement between the two is the thing it reports.
            sect = (section_caps or {}).get(served_id) or {}
            mods = sect.get("mods") if sect else (cap.get("modalities") or [])
            mods = list(mods or [])
            spec = sect.get("spec") if sect else ("head" if cap.get("speculative") else "")
            proj_unknown = sect.get("proj_unknown") if sect else bool(g.companion_name and not mods)
            loaded = (loaded_map or {}).get(served_id) or []
            configured = bool(g.aliases)

            if g.is_companion:
                status, status_label = "orphan", "Orphan companion"
            elif loaded:
                status, status_label = "loaded", "Loaded now"
            elif configured:
                status, status_label = "configured", "In models.ini"
            else:
                status, status_label = "unconfigured", "Not configured"

            flags = []
            if shape and shape.is_moe:
                flags.append("moe")
            elif shape:
                flags.append("dense")
            if "vision" in mods:
                flags.append("vision")
            if "audio" in mods:
                flags.append("audio")
            if mods or g.companion_name:
                flags.append("multimodal")
            if spec:
                flags.append("draft")
            if (us or {}).get("status") == "stale":
                flags.append("update")
            if served_id != g.stem:
                flags.append("alias")
            flags.append(status)

            rows.append({
                "g": g,
                "id": served_id,
                # True when this name is not the filename's: a model that exists only in
                # models.ini, which is what gets the headline instead of the filename.
                "is_alias": served_id != g.stem,
                "primary": n == 0,
                "configured": configured,
                "siblings": [i for i in ids if i != served_id],
                "shape": shape,
                "mods": mods,
                "spec": spec,
                "proj_unknown": proj_unknown,
                "head_idle": bool(sect.get("head_idle")),
                "cap": cap,
                "loaded": loaded,
                # NOT named "update": Jinja resolves r.update on a dict to the dict's own
                # update METHOD, which is truthy, so `{% if r.update %}` fired on every row
                # and painted an amber "update" chip across the whole list while the tooltip
                # it filled in read "dated  ( days newer)". Anything colliding with a dict
                # method - update, items, keys, values, get, copy, pop - must not be a key here.
                "stale": us if (us or {}).get("status") == "stale" else None,
                "badges": (badges_by_alias or {}).get(served_id) or [],
                "family": family, "quant": quant,
                "tier": tier, "tier_label": tier_label, "tier_short": tier_short,
                "status": status, "status_label": status_label,
                "flags": " ".join(flags),
                "card_gb": card_gb,
                # Sort by the file's size on every row so an alias sorts beside its file;
                # count only the owning row, so group totals stay the bytes on disk.
                "sort_bytes": g.total_bytes,
                "count_bytes": g.total_bytes if n == 0 else 0,
                "search": " ".join([g.display_name, served_id, family, quant,
                                    (shape.arch if shape else ""), " ".join(g.aliases or [])]).lower(),
            })
    return rows


def _shapes_for_files(snap) -> dict:
    """{display_name: ModelShape} for the models list.

    Companions are skipped: an mmproj declares itself as `clip`, which would render a
    "dense" chip on a projector that has no layers to offload in the first place.
    """
    out: dict = {}
    for g in snap.ggufs:
        if g.is_companion or not g.parts:
            continue
        shape = services.model_shape(g.parts[0])
        if shape.known:
            out[g.display_name] = shape
    return out



@router.get("/model/{filename:path}", response_class=HTMLResponse)
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
        "badges": db.badges_for(model_id),
    })


@router.post("/models/delete", response_class=HTMLResponse)
async def models_delete(request: Request, name: str = Form(...)) -> HTMLResponse:
    def _do() -> tuple[bool, str, int]:
        ok, msg, freed = services.delete_gguf(name)
        # Prune here catches ids orphaned EARLIER -- it cannot catch the model just deleted.
        # delete_gguf removes files, not the models.ini section, so at this point the section is
        # still there and the id still resolves; `unknown_ids` means "no section provides this".
        # What makes an id dead is deleting its section, and that path prunes for itself.
        # A no-op (and no restart) when nothing is actually stale.
        if ok:
            try:
                pruned = services.prune_openwebui_unknown_ids()
                if pruned:
                    msg = f"{msg}; removed {len(pruned)} stale id(s) from OpenWebUI"
                services.sync_openwebui_capabilities()
            except Exception:  # noqa: BLE001 -- deletion must succeed even if OpenWebUI is down
                pass
        return ok, msg, freed
    # The OpenWebUI reconcile behind these calls can restart the container (up to 30 s),
    # so it runs in a thread: on the loop it would freeze every polling page meanwhile.
    ok, msg, freed = await asyncio.to_thread(_do)
    snap = await asyncio.to_thread(services.snapshot_models_dir)
    flash = {"ok": ok, "msg": msg, "freed_h": human_bytes(freed) if freed else None}
    return templates.TemplateResponse(
        "_models_list.html", await _models_list_ctx(request, snap, flash))


@router.post("/models/delete-bulk", response_class=HTMLResponse)
async def models_delete_bulk(request: Request) -> HTMLResponse:
    form = await request.form()
    names = form.getlist("names") if hasattr(form, "getlist") else form.get("names") or []
    if isinstance(names, str):
        names = [names]

    def _do() -> tuple[int, int, list[str]]:
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
        return ok_count, total_freed, errors
    ok_count, total_freed, errors = await asyncio.to_thread(_do)
    snap = await asyncio.to_thread(services.snapshot_models_dir)
    if errors:
        flash = {"ok": False, "msg": f"deleted {ok_count}, {len(errors)} failed: " + "; ".join(errors[:3])}
    else:
        flash = {"ok": True, "msg": f"deleted {ok_count} file(s)", "freed_h": human_bytes(total_freed) if total_freed else None}
    return templates.TemplateResponse(
        "_models_list.html", await _models_list_ctx(request, snap, flash))


@router.post("/models/check-updates", response_class=HTMLResponse)
async def models_check_updates(request: Request) -> HTMLResponse:
    records = db.download_records()
    if records:
        await hf.check_updates_for(records)
    snap = services.snapshot_models_dir()
    flash = {"ok": True, "msg": f"checked {len(records)} record(s) against HF"}
    return templates.TemplateResponse(
        "_models_list.html", await _models_list_ctx(request, snap, flash))
