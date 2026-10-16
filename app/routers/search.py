from __future__ import annotations

import httpx

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from .. import autoconfig, db, hf, services, strata_models
from ..fit import _preset_estimates
from ..utils import human_bytes
from ..web import templates
from .strata import _strata_results_response

router = APIRouter()


@router.get("/search", response_class=HTMLResponse)
def search_page(request: Request, engine: str = "") -> HTMLResponse:
    return templates.TemplateResponse("search.html",
                                      {"request": request, "query": "", "engine": engine})


@router.get("/search/results", response_class=HTMLResponse)
async def search_results(request: Request, q: str = "", sort: str = "downloads",
                         limit: int = 30, engine: str = "") -> HTMLResponse:
    # One search, two engines. Strata needs a file list per result to answer exactly, which is
    # why it is a chosen filter rather than something every search pays for: the llama path is
    # one Hub request, the Strata path is one per result.
    if engine == "strata":
        return await _strata_results_response(request, q)

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
    # A HINT, never a verdict. Deciding properly needs the repo's file list, and doing that for
    # every result is exactly the cost this page refuses to pay by default - so the name is used
    # to say "worth a look" and the Strata filter remains the authority. Worded as a question in
    # the UI for the same reason.
    maybe_strata = {
        m.id for m in results
        if strata_models.ARCH_MARKER in m.id.lower() and strata_models.GSQ_MARKER in m.id.lower()
    }
    return templates.TemplateResponse("_search_results.html", {
        "request": request, "results": results, "error": error, "avatars": avatars,
        "browse_mode": browse_mode, "browse_sort": sort,
        "downloaded_by_repo": downloaded, "maybe_strata": maybe_strata, "engine": engine,
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


@router.get("/search/repo/{repo_id:path}", response_class=HTMLResponse)
async def search_repo(request: Request, repo_id: str) -> HTMLResponse:
    error: str | None = None
    gated: str = ""
    foreign: str = ""
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
            # Warn BEFORE the download, not after: the header already tells us llama.cpp
            # can't load this, and without this the estimates just silently didn't appear.
            foreign = autoconfig.foreign_gguf_reason(summary)
            if summary and not foreign:
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
    return templates.TemplateResponse("_repo_files.html", {"request": request, "repo_id": repo_id, "groups": groups, "error": error, "gated": gated, "foreign": foreign})
