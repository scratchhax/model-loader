from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from urllib.parse import urlparse

from . import db, hf
from .config import settings
from .utils import human_bytes


SPEED_SAMPLE_INTERVAL_S = 0.5
SPEED_SAMPLE_MAX = 120  # ~60s of history
PARALLEL_CHUNKS = 8       # parallel range requests per file
PARALLEL_MIN_SIZE = 32 * 1024 * 1024  # below this, use single stream (32 MB)


@dataclass
class ChunkState:
    idx: int
    start: int          # byte offset in file (inclusive)
    end: int            # byte offset in file (inclusive)
    downloaded: int = 0
    _samples: list[tuple[float, int]] = field(default_factory=list, repr=False)

    @property
    def size(self) -> int:
        return self.end - self.start + 1

    @property
    def pct(self) -> float:
        s = self.size
        return min(100.0, self.downloaded / s * 100.0) if s else 0.0

    @property
    def status(self) -> str:
        if self.downloaded >= self.size:
            return "done"
        if self.downloaded == 0:
            return "pending"
        return "downloading"

    @property
    def speed_bps(self) -> float:
        now = time.time()
        window = [(t, b) for (t, b) in self._samples if now - t <= 5.0]
        if len(window) < 2:
            return 0.0
        (t0, b0), (t1, b1) = window[0], window[-1]
        dt = t1 - t0
        return (b1 - b0) / dt if dt > 0 else 0.0

    @property
    def speed_h(self) -> str:
        s = self.speed_bps
        return f"{human_bytes(int(s))}/s" if s > 0 else "—"


@dataclass
class DownloadJob:
    id: str
    repo_id: str          # "owner/repo" for HF or hostname for URL imports
    filename: str         # basename we save as
    url: str              # full URL to fetch
    dest_path: Path
    temp_path: Path
    total_bytes: int
    downloaded_bytes: int = 0
    status: str = "queued"       # queued | downloading | done | error | canceled
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    chunks: list[ChunkState] = field(default_factory=list)
    parallel: bool = False
    _task: asyncio.Task | None = field(default=None, repr=False)
    _cancel: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _speed_samples: list[tuple[float, int]] = field(default_factory=list, repr=False)

    @property
    def pct(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return min(100.0, 100.0 * self.downloaded_bytes / self.total_bytes)

    @property
    def downloaded_h(self) -> str:
        return human_bytes(self.downloaded_bytes)

    @property
    def total_h(self) -> str:
        return human_bytes(self.total_bytes)

    @property
    def speed_bps(self) -> float:
        # Average over last ~5s window of samples
        now = time.time()
        window = [(t, b) for (t, b) in self._speed_samples if now - t <= 5.0]
        if len(window) < 2:
            return 0.0
        (t0, b0), (t1, b1) = window[0], window[-1]
        dt = t1 - t0
        return (b1 - b0) / dt if dt > 0 else 0.0

    @property
    def speed_h(self) -> str:
        s = self.speed_bps
        return f"{human_bytes(int(s))}/s" if s > 0 else "—"

    @property
    def eta_seconds(self) -> int | None:
        s = self.speed_bps
        if s <= 0 or self.total_bytes <= 0:
            return None
        remaining = self.total_bytes - self.downloaded_bytes
        return max(0, int(remaining / s))

    @property
    def eta_h(self) -> str:
        e = self.eta_seconds
        if e is None:
            return "—"
        if e < 60:
            return f"{e}s"
        if e < 3600:
            return f"{e // 60}m {e % 60}s"
        return f"{e // 3600}h {(e % 3600) // 60}m"

    @property
    def speed_history_mbps(self) -> list[float]:
        """Per-sample-pair MB/s, oldest first."""
        s = self._speed_samples
        if len(s) < 2:
            return []
        out: list[float] = []
        for (t0, b0), (t1, b1) in zip(s, s[1:]):
            dt = t1 - t0
            if dt > 0:
                out.append((b1 - b0) / dt / (1024 * 1024))
        return out

    @property
    def sparkline_points(self) -> str | None:
        """SVG polyline points string in a 100x20 viewBox, or None if no history."""
        hist = self.speed_history_mbps
        if len(hist) < 2:
            return None
        maxv = max(hist) or 1.0
        n = len(hist)
        return " ".join(
            f"{i * (100 / (n - 1)):.1f},{20 - (v / maxv) * 18:.1f}"
            for i, v in enumerate(hist)
        )


class DownloadManager:
    def __init__(self) -> None:
        self.jobs: dict[str, DownloadJob] = {}
        self._sema: asyncio.Semaphore | None = None

    def _semaphore(self) -> asyncio.Semaphore:
        if self._sema is None:
            self._sema = asyncio.Semaphore(settings.max_concurrent_downloads)
        return self._sema

    def snapshot(self) -> list[DownloadJob]:
        # active first, then most recent
        def sort_key(j: DownloadJob) -> tuple[int, float]:
            active_rank = {"downloading": 0, "queued": 1}.get(j.status, 2)
            return (active_rank, -(j.completed_at or j.started_at or 0.0))
        return sorted(self.jobs.values(), key=sort_key)

    def _make_job(self, *, repo_id: str, filename: str, url: str, total_bytes: int) -> DownloadJob:
        dest = settings.models_dir / filename
        for j in self.jobs.values():
            if j.dest_path == dest and j.status in ("queued", "downloading"):
                return j
        job = DownloadJob(
            id=uuid.uuid4().hex[:12],
            repo_id=repo_id,
            filename=filename,
            url=url,
            dest_path=dest,
            temp_path=dest.with_suffix(dest.suffix + ".download"),
            total_bytes=total_bytes,
        )
        self.jobs[job.id] = job
        job._task = asyncio.create_task(self._run(job))
        return job

    def enqueue(self, *, repo_id: str, hf_path: str, filename: str, total_bytes: int) -> DownloadJob:
        url = f"{hf.HF_RESOLVE}/{repo_id}/resolve/main/{hf_path}"
        return self._make_job(repo_id=repo_id, filename=filename, url=url, total_bytes=total_bytes)

    def enqueue_url(self, *, url: str, filename: str, total_bytes: int = 0) -> DownloadJob:
        origin = urlparse(url).netloc or "url"
        return self._make_job(repo_id=origin, filename=filename, url=url, total_bytes=total_bytes)

    def cancel(self, job_id: str) -> bool:
        j = self.jobs.get(job_id)
        if j is None or j.status not in ("queued", "downloading"):
            return False
        j._cancel.set()
        if j._task is not None:
            j._task.cancel()
        return True

    def clear_finished(self) -> int:
        removed = 0
        for jid in list(self.jobs.keys()):
            if self.jobs[jid].status in ("done", "error", "canceled"):
                del self.jobs[jid]
                removed += 1
        return removed

    async def _run(self, job: DownloadJob) -> None:
        try:
            async with self._semaphore():
                job.status = "downloading"
                job.started_at = time.time()
                await self._stream(job)
                os.replace(job.temp_path, job.dest_path)
                job.status = "done"
                job.completed_at = time.time()
        except asyncio.CancelledError:
            job.status = "canceled"
            job.completed_at = time.time()
            try:
                if job.temp_path.exists():
                    job.temp_path.unlink()
            except OSError:
                pass
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.completed_at = time.time()
        finally:
            db.record_download(
                repo_id=job.repo_id,
                filename=job.filename,
                dest_path=str(job.dest_path),
                total_bytes=job.total_bytes,
                status=job.status,
                error=job.error,
                started_at=job.started_at or time.time(),
                completed_at=job.completed_at,
            )

    def _auth_headers(self, url: str) -> dict[str, str]:
        headers: dict[str, str] = {}
        tok = hf.get_token()
        if tok and "huggingface.co" in url:
            headers["Authorization"] = f"Bearer {tok}"
        return headers

    def _note_progress(self, job: DownloadJob, delta: int) -> None:
        job.downloaded_bytes += delta
        now = time.time()
        if not job._speed_samples or (now - job._speed_samples[-1][0]) >= SPEED_SAMPLE_INTERVAL_S:
            job._speed_samples.append((now, job.downloaded_bytes))
            if len(job._speed_samples) > SPEED_SAMPLE_MAX:
                job._speed_samples = job._speed_samples[-SPEED_SAMPLE_MAX:]

    async def _stream(self, job: DownloadJob) -> None:
        job.dest_path.parent.mkdir(parents=True, exist_ok=True)
        url = job.url
        headers = self._auth_headers(url)

        # HEAD to discover total size + range support + resolved (post-redirect) URL
        total = 0
        accept_ranges = False
        resolved_url = url
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), follow_redirects=True, headers=headers) as head_client:
                h = await head_client.head(url)
                if h.status_code < 400:
                    total = int(h.headers.get("content-length") or 0)
                    accept_ranges = h.headers.get("accept-ranges", "").lower() == "bytes"
                    resolved_url = str(h.url)
        except httpx.HTTPError:
            pass  # fall through and try single-stream

        if total and total > 0:
            job.total_bytes = total

        # Choose parallel vs single-stream
        if accept_ranges and total >= PARALLEL_MIN_SIZE and PARALLEL_CHUNKS > 1:
            await self._stream_parallel(job, resolved_url, headers, total)
        else:
            await self._stream_single(job, url, headers)

    async def _stream_parallel(self, job: DownloadJob, url: str, headers: dict[str, str], total: int) -> None:
        # If a partial file exists and matches size exactly, assume it's complete (rare).
        if job.temp_path.exists() and job.temp_path.stat().st_size == total:
            job.downloaded_bytes = total
            return
        # Preallocate temp file to full size; per-worker seek+write into non-overlapping regions.
        with open(job.temp_path, "wb") as f:
            f.truncate(total)
        job.downloaded_bytes = 0
        job._speed_samples.append((time.time(), 0))
        job.parallel = True

        n = PARALLEL_CHUNKS
        chunk = total // n
        job.chunks = []
        for i in range(n):
            start = i * chunk
            end = (start + chunk - 1) if i < n - 1 else total - 1
            job.chunks.append(ChunkState(idx=i, start=start, end=end))

        async def worker(cs: ChunkState) -> None:
            worker_headers = dict(headers)
            worker_headers["Range"] = f"bytes={cs.start}-{cs.end}"
            timeout = httpx.Timeout(30.0, read=120.0)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=worker_headers) as w:
                async with w.stream("GET", url) as resp:
                    if resp.status_code not in (200, 206):
                        body = ""
                        try:
                            body = (await resp.aread()).decode(errors="replace")[:200]
                        except Exception:
                            pass
                        raise RuntimeError(f"HTTP {resp.status_code} for range {cs.start}-{cs.end}: {body}")
                    with open(job.temp_path, "r+b") as f:
                        f.seek(cs.start)
                        async for buf in resp.aiter_bytes(1024 * 1024):
                            if job._cancel.is_set():
                                raise asyncio.CancelledError
                            f.write(buf)
                            cs.downloaded += len(buf)
                            now = time.time()
                            if not cs._samples or (now - cs._samples[-1][0]) >= SPEED_SAMPLE_INTERVAL_S:
                                cs._samples.append((now, cs.downloaded))
                                if len(cs._samples) > 12:
                                    cs._samples = cs._samples[-12:]
                            self._note_progress(job, len(buf))

        tasks = [asyncio.create_task(worker(c)) for c in job.chunks]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for t in tasks:
                if not t.done():
                    t.cancel()
            # allow cancellations to settle
            for t in tasks:
                try:
                    await t
                except BaseException:
                    pass
            raise

    async def _stream_single(self, job: DownloadJob, url: str, headers: dict[str, str]) -> None:
        # Original resume-aware single-stream path (used when server doesn't support ranges, or file is small)
        start = 0
        if job.temp_path.exists():
            start = job.temp_path.stat().st_size
        if start > 0:
            headers = {**headers, "Range": f"bytes={start}-"}
        job.downloaded_bytes = start
        job._speed_samples.append((time.time(), job.downloaded_bytes))

        timeout = httpx.Timeout(30.0, read=120.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code == 416 and start > 0:
                    job.downloaded_bytes = job.total_bytes or start
                    return
                if resp.status_code not in (200, 206):
                    body = ""
                    try:
                        body = (await resp.aread()).decode(errors="replace")[:200]
                    except Exception:
                        pass
                    raise RuntimeError(f"HTTP {resp.status_code} from {url}: {body}")

                if not job.total_bytes:
                    if "content-range" in resp.headers:
                        job.total_bytes = int(resp.headers["content-range"].split("/")[-1])
                    else:
                        job.total_bytes = start + int(resp.headers.get("content-length", "0"))

                mode = "ab" if start else "wb"
                with open(job.temp_path, mode) as f:
                    async for buf in resp.aiter_bytes(1024 * 1024):
                        if job._cancel.is_set():
                            raise asyncio.CancelledError
                        f.write(buf)
                        self._note_progress(job, len(buf))


manager = DownloadManager()
