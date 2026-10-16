from __future__ import annotations

import html
from pathlib import Path

import httpx
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, Response

from .. import autoconfig, hf
from ..downloader import manager
from ..web import templates, _blocked_chip

router = APIRouter()


_QUEUED_CHIP = (
    '<a href="/downloads" title="{title}" '
    'class="shrink-0 inline-flex items-center gap-1 rounded-md bg-emerald-100 dark:bg-emerald-950 '
    'text-emerald-800 dark:text-emerald-300 px-3 py-1.5 text-xs font-medium">'
    '✓ {label}</a>'
)


def _fit_host_ram() -> float:
    """RAM the fit math may count on: the host total minus a running Strata's pinned arena.

    The static HOST_RAM_RESERVE_GB covers the OS and its habits; the arena is a measured
    tenant on top of that, and 35-55 GB of pinned experts is well past what a 32 GB reserve
    was ever meant to absorb. Unchanged when no Strata runs, so a llama-only box sees
    exactly what it always saw.
    """
    return round(max(0.0, hw.host_ram_gb() - hw.strata_arena_gb()), 1)


def _fit_card_vram(bn: str) -> list[float]:
    """Per-card VRAM for the fit table, with the cards a running Strata holds zeroed out.

    The table is a promise about placement, and a card whose VRAM is spoken for by another
    engine's expert cache is not available for llama layers no matter what the free figure
    says - "free" on that card is Strata's next expert, not llama's next layer. Zeroing the
    cap rather than shrinking it keeps the arithmetic honest downstream: _split_feasible
    fills cards in order and a zero-cap card simply takes no layers, which is exactly the
    placement llama's own fitter will also arrive at once it sees the card is full.
    """
    caps = hw.card_vram_gb_for(bn)
    held = services.strata_held_cards()
    if not held or not caps:
        return caps
    return [0.0 if i in held else c for i, c in enumerate(caps)]


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
            "gpu_count": hw.gpu_count_for(name), "card_vram_gb": _fit_card_vram(name),
                    "host_ram_gb": _fit_host_ram(),
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


def _blocked_chip(reason: str) -> HTMLResponse:
    """What a refused download button turns into. The reason rides in the tooltip."""
    return HTMLResponse(
        '<span class="shrink-0 inline-flex items-center rounded-md bg-red-100 dark:bg-red-950 '
        'text-red-800 dark:text-red-300 px-3 py-1.5 text-xs font-medium" '
        f'title="{html.escape(reason)}">Blocked: llama.cpp can&rsquo;t load this</span>')


async def _foreign_repo(repo_id: str, main_path: str) -> str:
    """Why a download from this repo is refused, or "". Checked server-side as well as hidden
    in the page, so a stale page or a hand-made request can't queue a file llama.cpp won't load.
    An unreadable header (network, gated) is not a verdict and never blocks."""
    return autoconfig.foreign_gguf_reason(await hf.gguf_header(repo_id, main_path))


@router.post("/download", response_class=HTMLResponse)
async def download_single(repo_id: str = Form(...), path: str = Form(...), size: int = Form(0)) -> HTMLResponse:
    base = Path(path).name
    if base.lower().endswith(".gguf") and "mmproj" not in base.lower():
        reason = await _foreign_repo(repo_id, path)
        if reason:
            return _blocked_chip(reason)
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


@router.post("/download/multi", response_class=HTMLResponse)
async def download_multi(repo_id: str = Form(...), shard_base: str = Form(...)) -> HTMLResponse:
    try:
        detail = await hf.repo_detail(repo_id)
    except httpx.HTTPError as e:
        return HTMLResponse(
            f'<span class="shrink-0 inline-flex items-center rounded-md bg-red-100 dark:bg-red-950 '
            f'text-red-800 dark:text-red-300 px-3 py-1.5 text-xs font-medium" title="{e}">HF error</span>'
        )
    shards = sorted((f for f in detail.files if f.shard_base == shard_base),
                    key=lambda f: (f.shard_index or 0, f.path))
    if shards:
        reason = await _foreign_repo(repo_id, shards[0].path)
        if reason:
            return _blocked_chip(reason)
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


@router.post("/download/url", response_class=HTMLResponse)
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

    if filename.lower().endswith(".gguf") and "mmproj" not in filename.lower():
        reason = autoconfig.foreign_gguf_reason(await hf.url_gguf_header(url))
        if reason:
            return HTMLResponse(
                '<div class="rounded-md bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 px-3 py-2 text-sm">'
                f'Not queued. {html.escape(reason)}</div>')

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


@router.get("/downloads", response_class=HTMLResponse)
def downloads_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("downloads.html", {"request": request, "jobs": manager.snapshot()})


@router.get("/downloads/rows", response_class=HTMLResponse)
def downloads_rows(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("_download_rows.html", {"request": request, "jobs": manager.snapshot()})


@router.post("/downloads/{job_id}/cancel")
def downloads_cancel(job_id: str) -> Response:
    manager.cancel(job_id)
    return Response(status_code=204)


@router.post("/downloads/clear", response_class=HTMLResponse)
def downloads_clear(request: Request) -> HTMLResponse:
    manager.clear_finished()
    return templates.TemplateResponse("_download_rows.html", {"request": request, "jobs": manager.snapshot()})
