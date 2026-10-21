from __future__ import annotations

import html
from pathlib import Path

import httpx
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, Response

from .. import autoconfig, db, downloader, hf
from ..downloader import manager
from ..web import templates, _blocked_chip, _form_note, _toast, _QUEUED_CHIP

router = APIRouter()




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
        # the acceptance rate.
        #
        # A head from the model's own repo is trained against it, and llama.cpp drives it
        # through draft-mtp; a generic community draft paired with an unrelated model is what
        # segfaults llama-server. Autoconfig still only PROPOSES enabling it.
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
        return _blocked_chip(str(e), "HF error")
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
        return _toast(False, "URL must start with http:// or https://", fail="")
    if not filename.strip():
        from urllib.parse import urlparse, unquote
        p = urlparse(url).path
        filename = unquote(p.rsplit("/", 1)[-1]) if p else ""
    filename = filename.strip()
    if not filename:
        return _toast(False, "could not derive a filename; set one explicitly", fail="")
    if "/" in filename or ".." in filename:
        return _toast(False, "filename must be a plain basename", fail="")

    if filename.lower().endswith(".gguf") and "mmproj" not in filename.lower():
        reason = autoconfig.foreign_gguf_reason(await hf.url_gguf_header(url))
        if reason:
            return _toast(False, f"Not queued. {reason}", fail="")

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
        f'Queued <span class="font-mono">{html.escape(filename)}</span>. See progress below.</div>'
    )


@router.get("/downloads", response_class=HTMLResponse)
def downloads_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("downloads.html", {
        "request": request, "jobs": manager.snapshot(),
        "streams": downloader.streams_per_file(), "files": downloader.files_at_once(),
        "max_streams": downloader.PARALLEL_CHUNKS, "max_files": downloader.MAX_FILES_AT_ONCE})


@router.post("/downloads/limits", response_class=HTMLResponse)
def downloads_limits(streams: int = Form(1), files: int = Form(1)) -> HTMLResponse:
    """How hard downloads may pull: range streams per file, and files at once. Both default to
    one; see downloader.STREAMS_KEY for why."""
    s = max(1, min(streams, downloader.PARALLEL_CHUNKS))
    f = max(1, min(files, downloader.MAX_FILES_AT_ONCE))
    db.set_setting(downloader.STREAMS_KEY, str(s))
    db.set_setting(downloader.FILES_KEY, str(f))
    return _form_note(True, f"{s} stream{'s' if s > 1 else ''} per file, {f} file{'s' if f > 1 else ''} "
                            f"at once; up to {s * f} connections",
                      label=f"saved: up to {s * f} connection{'s' if s * f > 1 else ''}")


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
