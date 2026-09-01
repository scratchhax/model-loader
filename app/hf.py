from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from . import db
from .utils import shard_key

AVATAR_TTL_S = 7 * 86_400

HF_API = "https://huggingface.co/api"
HF_RESOLVE = "https://huggingface.co"

_QUANT_RE = re.compile(r"\b(I?Q\d+(?:_[A-Z0-9]+)*|F16|F32|BF16|FP8|FP4)\b", re.IGNORECASE)


def infer_quant(filename: str) -> str | None:
    m = _QUANT_RE.search(filename)
    return m.group(0).upper() if m else None


def get_token() -> str:
    return db.get_setting("hf_token", "")


def _auth_headers() -> dict[str, str]:
    tok = get_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


@dataclass
class HfModel:
    id: str                 # "owner/name"
    downloads: int
    likes: int
    last_modified: str
    pipeline_tag: str | None
    tags: list[str]
    gguf_count: int | None  # None if unknown from search response


async def search_models(query: str, limit: int = 30, sort: str = "downloads") -> list[HfModel]:
    """Search HF hub for models tagged gguf. sort: downloads|likes|lastModified|trendingScore"""
    params = {
        "search": query,
        "filter": "gguf",
        "limit": str(limit),
        "sort": sort,
        "direction": "-1",
        "full": "true",
    }
    async with httpx.AsyncClient(timeout=20.0, headers=_auth_headers()) as client:
        r = await client.get(f"{HF_API}/models", params=params)
        r.raise_for_status()
        data = r.json()

    out: list[HfModel] = []
    for item in data:
        siblings = item.get("siblings") or []
        gguf_count = sum(1 for s in siblings if str(s.get("rfilename", "")).lower().endswith(".gguf")) if siblings else None
        out.append(HfModel(
            id=item.get("id") or item.get("modelId") or "",
            downloads=int(item.get("downloads") or 0),
            likes=int(item.get("likes") or 0),
            last_modified=str(item.get("lastModified") or ""),
            pipeline_tag=item.get("pipeline_tag"),
            tags=list(item.get("tags") or []),
            gguf_count=gguf_count,
        ))
    return out


@dataclass
class HfFile:
    path: str
    size: int
    quant: str | None
    shard_base: str          # base name without -NNNNN-of-NNNNN suffix
    shard_index: int | None
    shard_total: int | None


@dataclass
class HfRepoDetail:
    id: str
    files: list[HfFile]
    readme_snippet: str | None


async def repo_detail(repo_id: str, revision: str = "main") -> HfRepoDetail:
    async with httpx.AsyncClient(timeout=30.0, headers=_auth_headers(), follow_redirects=True) as client:
        tree = await client.get(f"{HF_API}/models/{repo_id}/tree/{revision}", params={"recursive": "true", "expand": "true"})
        tree.raise_for_status()
        entries: list[dict[str, Any]] = tree.json()

    files: list[HfFile] = []
    for e in entries:
        if e.get("type") != "file":
            continue
        path = e.get("path", "")
        if not path.lower().endswith(".gguf") and not _is_support_file(path):
            continue
        # HF tree entries: size is under "size" for direct files; LFS files have "lfs": {"size": ...}
        size = int((e.get("lfs") or {}).get("size") or e.get("size") or 0)
        base, idx, tot = shard_key(path)
        files.append(HfFile(
            path=path,
            size=size,
            quant=infer_quant(path),
            shard_base=base,
            shard_index=idx,
            shard_total=tot,
        ))
    files.sort(key=lambda f: (not f.path.lower().endswith(".gguf"), f.path.lower()))
    return HfRepoDetail(id=repo_id, files=files, readme_snippet=None)


def _is_support_file(path: str) -> bool:
    low = path.lower()
    # things llama.cpp sometimes needs alongside a GGUF
    return low.endswith((".mmproj", "mmproj.gguf", "chat_template.jinja", "tokenizer.model"))


async def _fetch_owner_avatar(client: httpx.AsyncClient, owner: str) -> str | None:
    for endpoint in (f"{HF_API}/organizations/{owner}/overview", f"{HF_API}/users/{owner}/overview"):
        try:
            r = await client.get(endpoint)
        except httpx.HTTPError:
            continue
        if r.status_code != 200:
            continue
        try:
            data = r.json()
        except ValueError:
            continue
        url = data.get("avatarUrl") or ""
        if not url:
            continue
        if url.startswith("//"):
            url = "https:" + url
        elif url.startswith("/"):
            url = "https://huggingface.co" + url
        return url
    return None


async def owner_avatars(owners: list[str]) -> dict[str, str]:
    """Return {owner: url} using sqlite cache with TTL; misses are fetched in parallel and cached."""
    now = time.time()
    unique = sorted({o for o in owners if o})
    result: dict[str, str] = {}
    to_fetch: list[str] = []

    for owner in unique:
        cached = db.get_avatar(owner)
        if cached and (now - cached["fetched_at"]) < AVATAR_TTL_S:
            if cached["url"]:
                result[owner] = cached["url"]
            continue
        to_fetch.append(owner)

    if to_fetch:
        headers = _auth_headers()
        async with httpx.AsyncClient(timeout=8.0, headers=headers, follow_redirects=True) as client:
            urls = await asyncio.gather(
                *[_fetch_owner_avatar(client, o) for o in to_fetch],
                return_exceptions=False,
            )
        for owner, url in zip(to_fetch, urls):
            db.set_avatar(owner, url or "", now)
            if url:
                result[owner] = url
    return result


async def repo_file_mtimes(repo_id: str, revision: str = "main") -> dict[str, str]:
    """{path -> ISO8601 lastCommit.date} for every file in a repo tree."""
    async with httpx.AsyncClient(timeout=15.0, headers=_auth_headers(), follow_redirects=True) as client:
        r = await client.get(
            f"{HF_API}/models/{repo_id}/tree/{revision}",
            params={"recursive": "true", "expand": "true"},
        )
    if r.status_code != 200:
        return {}
    try:
        entries: list[dict[str, Any]] = r.json()
    except ValueError:
        return {}
    out: dict[str, str] = {}
    for e in entries:
        if e.get("type") != "file":
            continue
        path = e.get("path", "")
        last = (e.get("lastCommit") or {}).get("date") or ""
        if path and last:
            out[path] = last
    return out


async def check_updates_for(records: list[dict]) -> int:
    """Refresh update_check rows for the given (filename, repo_id) records. Returns count checked."""
    by_repo: dict[str, list[str]] = {}
    for r in records:
        by_repo.setdefault(r["repo_id"], []).append(r["filename"])
    now = time.time()
    total = 0
    for repo_id, files in by_repo.items():
        if "/" not in repo_id:  # url-imports have hostname as repo_id — skip
            for f in files:
                db.set_update_check(f, repo_id, None, now)
            continue
        try:
            mtimes = await repo_file_mtimes(repo_id)
        except httpx.HTTPError:
            mtimes = {}
        for f in files:
            db.set_update_check(f, repo_id, mtimes.get(f), now)
            total += 1
    return total


async def validate_token(token: str) -> tuple[bool, str]:
    """Ping the whoami endpoint. Empty token is 'valid' (means anonymous)."""
    if not token:
        return True, "anonymous (no token)"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{HF_API}/whoami-v2", headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            data = r.json()
            name = data.get("name") or data.get("fullname") or "unknown"
            return True, f"authenticated as {name}"
        return False, f"HTTP {r.status_code}: {r.text[:120]}"
