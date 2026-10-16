from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from .. import autoconfig, gguf_meta, ini, services, telemetry
from ..config import settings
from ..utils import shard_key
from ..fit import _fit_card_vram, _fit_host_ram
from ..web import templates
from .models import _section_caps

router = APIRouter()


def _config_context(request: Request, flash: dict | None = None) -> dict:
    sections = ini.list_sections()
    # Client snippets need the address the browser used to reach us (not the docker-internal
    # host the prober uses), so snippets work from any LAN machine. Compute endpoints once.
    browser_host = request.url.hostname or ""
    endpoints = services.browser_endpoints(browser_host)
    for s in sections:
        s.client_snippets = [
            {"name": ep["name"], "base_url": ep["base_url"], "api_key": ep["api_key"],
             **ini.to_client_config(s.name, ep["base_url"], ep["api_key"])}
            for ep in endpoints
        ]
    return {
        "request": request,
        "sections": sections,
        "caps": _section_caps(sections),
        "unregistered": ini.unregistered_gguf_stems(),
        "raw": ini.raw_text(),
        "backups": ini.list_backups(),
        "ini_path": str(settings.models_ini_path),
        "llama_backends": services._effective_container_names(),
        "flash": flash,
    }


@router.get("/config", response_class=HTMLResponse)
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


@router.get("/config/section/new", response_class=HTMLResponse)
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


def _pick_main_gguf(files: list[Path], section: str) -> Path:
    """Which of a subdir's GGUFs is THIS section's model. Companions are already filtered out.

    Not the first alphabetically. A folder routinely holds several quants of one model, and the
    section name is what says which: `[...-Q8_0]` wants the Q8_0 file, not whichever sorts first.
    Shards are grouped before choosing so the answer is always part 1, the shard carrying the
    metadata and the only one llama-server should be handed.
    """
    groups: dict[str, list[tuple[int, Path]]] = {}
    for q in files:
        base, idx, _total = shard_key(q.name)
        stem = base[:-5] if base.lower().endswith(".gguf") else base
        groups.setdefault(stem, []).append((idx or 1, q))

    def first_shard(stem: str) -> Path:
        return min(groups[stem], key=lambda t: t[0])[1]

    low = section.lower()
    for stem in groups:
        if stem.lower() == low:
            return first_shard(stem)
    for stem in groups:
        if low.startswith(stem.lower()) or stem.lower().startswith(low):
            return first_shard(stem)
    # The section name matches nothing here, so fall back to the largest model present rather
    # than to sort order. Size is a far better guess at "the main one" than the alphabet.
    biggest = max(groups, key=lambda s: sum(
        autoconfig.file_size_or_none(q) or 0 for _i, q in groups[s]))
    return first_shard(biggest)


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
            gguf_files = sorted([q for q in subdir_path.iterdir() if q.is_file() and q.suffix.lower() == ".gguf"])
            # Every companion, not just mmproj. This used to test `"mmproj" not in name` and
            # then take main_files[0] - the alphabetically first survivor - which on a real
            # folder picked ...-MAX-MTP-IQ2_M.gguf over ...-MAX-Q8_0.gguf because M sorts before
            # Q. The section named ...-Q8_0 was therefore configured to run an 11.3 GiB IQ2_M
            # quant while the 27.7 GiB Q8_0 beside it was never loaded. ini._is_companion
            # already knew about draft heads; this was a third copy of the narrower mmproj-only
            # test, and the one missed when the other two were unified.
            main_files = [q for q in gguf_files
                          if not ini._is_companion(q.name, autoconfig.file_size_or_none(q))
                          ] or gguf_files
            if main_files:
                pick = _pick_main_gguf(main_files, name)
                return pick, f"/models/{subdir_name}/{pick.name}", rel
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
        _, part_idx, part_total = shard_key(Path(model_rel).name)
        if part_idx is not None and part_total and part_total > 1:
            hints.insert(0, f"Sharded model ({part_total} parts) → `model = {model_rel}` pre-filled to the first shard; llama-server auto-loads the rest.")
        else:
            hints.insert(0, f"Model lives in a subdir → `model = {model_rel}` pre-filled with the absolute path.")
    return values, hints


@router.get("/config/section/{name}/edit", response_class=HTMLResponse)
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


@router.post("/config/section/{name}")
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
        await asyncio.to_thread(services.sync_openwebui_capabilities)
    except Exception:  # noqa: BLE001 -- saving the section must not depend on OpenWebUI
        pass
    # A brand-new model is offered on the GPU backends by default. Without this it lands in
    # models.ini, works, and is invisible in OpenWebUI because every connection carries an
    # explicit whitelist that predates it.
    note = ""
    if is_new_section:
        try:
            ok, msg = await asyncio.to_thread(services.assign_new_model_to_gpu, name)
            if ok and msg:
                note = f"&note={msg}"
        except Exception:  # noqa: BLE001
            pass
    return Response(status_code=200, headers={"HX-Redirect": f"/config?saved={name}{note}"})



@router.get("/config/section/{name}/autoconfig", response_class=HTMLResponse)
async def config_autoconfig(request: Request, name: str, preset: str = "",
                            sessions: int = 0, spec: str = "", vision: str = "") -> HTMLResponse:
    import json as _json
    # 0 means the UI did not specify, so fall back to what is SAVED rather than to 1. Defaulting
    # to 1 made every fresh open of the panel propose resetting a `parallel = 3` section to one
    # slot, and Fill+Save would then quarter its per-conversation context without anyone
    # choosing that - the same silent-downgrade trap _preset_for_saved_ctx exists to prevent for
    # ctx-size. The panel's own buttons always pass sessions explicitly, so they still
    # round-trip whatever is on screen.
    if not sessions:
        try:
            sessions = int(str((ini.get_section(name) or {}).get("parallel") or "1").strip())
        except (TypeError, ValueError):
            sessions = 1
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
    # If sharded, sum THIS model's own shards. Matching on the shard base rather than on
    # "every non-companion GGUF in the folder" matters as soon as a folder holds two quants of
    # one model: the turbo folder has an 11.3 GiB IQ2_M beside a 27.7 GiB Q8_0, and the old
    # rule reported 28.3 GB of weights whichever of them was selected - for the IQ2_M that is
    # nearly triple. An unsharded file is simply its own size, so the sum only runs when this
    # file actually declares itself part of a set.
    if gguf_path is not None:
        _base, _idx, _parts = shard_key(gguf_path.name)
        if _parts:
            try:
                file_size = sum(
                    q.stat().st_size for q in gguf_path.parent.iterdir()
                    if q.is_file() and q.suffix.lower() == ".gguf"
                    and shard_key(q.name)[0] == _base
                )
            except OSError:
                pass

    backend_list = _backend_list()

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
        await asyncio.to_thread(telemetry.ingest,
                                await asyncio.to_thread(services._effective_container_names))
        tel = telemetry.stats_for(model_path=model_rel, alias=name)
        cfgh = telemetry.config_history(model_path=model_rel, alias=name)
    except Exception:  # noqa: BLE001
        tel, cfgh = telemetry.Stats(), []

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
        vision=vision,
    )

    # Vision costs context, and the panel should say how much rather than leave it to be found
    # out. When a projector exists, size the other state as well with everything else held
    # equal - same preset, sessions and spec profile - so both figures come from the same fit
    # maths instead of one real number and one guess.
    vision_cost = None
    if rec.has_projector and not rec.error:
        def _vision_state(r):
            p = next((x for x in r.presets if x.key == r.active_preset), None)
            return {"ctx": r.recommended_ctx,
                    "gpu_layers": p.gpu_layers if p else 0,
                    "total_layers": p.total_layers if p else 0}
        try:
            alt = autoconfig.analyze(
                summary=summary, file_size=file_size, backends=backend_list,
                model_rel=model_rel, current_section=current_section,
                preset=rec.active_preset, n_sessions=sessions,
                models_dir=settings.models_dir, section_name=name,
                model_subdir=model_subdir, spec_profile=spec,
                vision=("off" if rec.vision else "on"),
            )
            vision_cost = {("on" if rec.vision else "off"): _vision_state(rec),
                           ("off" if rec.vision else "on"): _vision_state(alt)}
        except Exception:  # noqa: BLE001 - the comparison is a nicety; it must not cost the panel
            vision_cost = None

    # Prepare per-plan row map for the template + which ctx columns to show
    plan_row_map = {p.name: {r.ctx: r for r in p.rows} for p in rec.plans}

    # Where the recommended configuration's VRAM actually goes, split into the segments the
    # panel draws. Built here rather than in the template so the bar shows the same figures the
    # fit maths used instead of a re-derivation that could drift from them. Every byte of the
    # card is accounted for, free space included, so the bar sums to the whole card - a
    # breakdown that silently omits a cost is how the compute buffer stayed invisible for so
    # long. Ordered weights -> context -> overhead -> compute: that order keeps the two hues
    # the palette validator flags as a weak pair for protanopia from touching.
    vram_breakdown = None
    _is_moe_model = isinstance((summary.get("model") or {}).get("expert_count"), int)         and ((summary.get("model") or {}).get("expert_count") or 0) > 1
    _rec_plan = next((p for p in rec.plans if p.name == rec.recommended_backend), None)
    _rec_row = (plan_row_map.get(rec.recommended_backend) or {}).get(rec.recommended_ctx)
    if _rec_plan and _rec_row and _rec_plan.vram_gb > 0 and _rec_row.fits:
        _total = float(_rec_plan.vram_gb)
        _parts = [
            # Say the offloaded share outright. On a model larger than the card this segment is
            # only the resident part - Flash-Next shows 51 GB here against an 84 GB file - and a
            # reader comparing it to the file size has no way to tell that is correct.
            ("model", "model weights", float(_rec_row.model_gb),
             ("GPU-resident weights: %d%% of the model, the other %d%% streams from host RAM"
              % (_rec_row.gpu_pct, 100 - _rec_row.gpu_pct)) if _rec_row.gpu_pct < 100
             else "the whole model, resident on the GPU"),
            ("ctx", "context (KV)", float(_rec_row.kv_gb),
             "KV cache for %s tokens" % f"{_rec_row.total_ctx:,}"),
            ("overhead", "overhead", float(_rec_row.reserve_gb) + float(_rec_row.aux_gb),
             "driver and runtime context, plus any vision projector and draft head"),
            ("compute", "compute buffers", float(_rec_row.compute_gb),
             "per-card graph scratch, which grows with context"),
        ]
        _used = sum(v for _k, _l, v, _n in _parts)
        _free = max(0.0, _total - _used)
        _segments = [{"key": k, "label": lbl, "gb": v, "pct": 100.0 * v / _total, "note": note}
                     for k, lbl, v, note in _parts if v > 0.005]
        if _free > 0.005:
            _segments.append({"key": "free", "label": "free", "gb": _free,
                              "pct": 100.0 * _free / _total,
                              "note": "unallocated on the card"})
        # Where the model's WEIGHTS end up, which the card bar above cannot show: anything
        # offloaded is not on the card at all, so it has no place in a bar scaled to the card.
        # Its own bar, its own denominator (the model), stated as such - two scales in one bar
        # would be the chart equivalent of a second y-axis.
        #
        # This is the number that answers "have I left VRAM". For a MoE it is specifically the
        # experts that move: attention stays resident, so counting layers would understate it,
        # which is why gpu_pct is measured in bytes.
        _weights_total_gb = file_size / (1024 ** 3)
        _weights_gpu_gb = float(_rec_row.model_gb)
        _weights_host_gb = max(0.0, _weights_total_gb - _weights_gpu_gb)
        weights_split = None
        if _weights_total_gb > 0 and _weights_host_gb > 0.05:
            weights_split = {
                "total_gb": _weights_total_gb,
                "gpu_gb": _weights_gpu_gb,
                "host_gb": _weights_host_gb,
                "gpu_pct": 100.0 * _weights_gpu_gb / _weights_total_gb,
                "host_pct": 100.0 * _weights_host_gb / _weights_total_gb,
                "is_moe": bool(_is_moe_model),
                "offload_kind": _rec_row.offload_kind,
                "n_cpu_moe": _rec_row.n_cpu_moe,
            }
        vram_breakdown = {
            "total_gb": _total, "used_gb": _used, "free_gb": _free,
            "backend": rec.recommended_backend, "ctx": rec.recommended_ctx,
            "total_ctx": _rec_row.total_ctx, "segments": _segments,
            "weights_split": weights_split,
            "subtitle": "%s at %s%s" % (
                rec.recommended_backend, autoconfig.format_ctx(rec.recommended_ctx),
                " \u00d7 %d" % rec.n_sessions if rec.n_sessions > 1 else ""),
        }
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
        "vram_breakdown": vram_breakdown,
        "format_ctx": autoconfig.format_ctx,
        "vision_cost": vision_cost,
        # Carried on every link that re-renders this panel, so picking a preset or a session
        # count does not silently flip vision back to the saved state.
        "vision_q": ("on" if rec.vision else "off") if rec.has_projector else "",
        "tel": tel,
        "cfgh": cfgh,
        "values_json": _json.dumps(rec.values),
        "values_minimal_json": _json.dumps(rec.values_minimal),
        "displaced_json": _json.dumps(rec.displaced),
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


@router.post("/config/section/{name}/rename")
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
        return Response(status_code=200, headers={"HX-Redirect": "/config?err=rename+failed"})

    # Reconcile OpenWebUI's per-connection whitelists in ONE pass (each write restarts
    # open-webui, so doing this as two passes would cost two restarts).
    #
    # Two things have to happen together. Carry the old id across, or the model silently
    # vanishes from the picker. And drop ids no section provides any more: OpenWebUI RENDERS
    # its whitelist rather than intersecting it with what the backend reports, so a dead id
    # stays visible in the model picker and fails with "model not found" only when someone
    # tries to use it — the one failure mode neither end shows you.
    def _reconcile() -> None:
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
    try:
        # Each whitelist write restarts open-webui (up to 30 s) - off the event loop.
        await asyncio.to_thread(_reconcile)
    except Exception:  # noqa: BLE001 — renaming must succeed even if OpenWebUI is unreachable
        pass

    return Response(status_code=200, headers={"HX-Redirect": f"/config?saved={new}&edit={new}"})


@router.post("/config/section/{name}/delete")
def config_section_delete(name: str) -> Response:
    ok = ini.delete_section(name)
    if not ok:
        return Response(status_code=200, headers={"HX-Redirect": f"/config?err=no+such+section:+{name}"})

    # Removing the section is the moment the id stops being servable -- llama-server resolves
    # model ids out of models.ini, so from here OpenWebUI is offering something no backend can
    # answer. And it RENDERS its whitelist rather than intersecting it with what the backend
    # reports, so the model sits in the picker looking fine and fails with "model not found"
    # only when somebody selects it.
    #
    # The rename path already reconciles for exactly this reason; delete did not, which is how
    # a deleted model kept appearing. Deleting the GGUF does not help either: that leaves the
    # section in place, so the id is still "known" at that point and nothing is pruned.
    #
    # Two writes, one restart. The whitelist lives in OpenWebUI's config table, which it reads
    # only at boot, so changing it costs a restart. The stale per-model record lives in its
    # `model` table, which is read per request, so clearing that one is free.
    try:
        services.prune_openwebui_unknown_ids()
        services.align_openwebui_capabilities()
    except Exception:  # noqa: BLE001 -- deleting a section must succeed even if OpenWebUI is down
        pass
    return Response(status_code=200, headers={"HX-Redirect": f"/config?deleted={name}"})
