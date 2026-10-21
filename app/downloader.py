from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
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
PARALLEL_CHUNKS = 8       # the most parallel range requests per file it will open
PARALLEL_MIN_SIZE = 32 * 1024 * 1024  # below this, use single stream (32 MB)
MAX_FILES_AT_ONCE = 4

# Both default to ONE. Two files of eight ranges each was sixteen connections to the CDN, and
# that knocked this house's ISP onto its backup line (2026-10). One stream is also the only path
# that resumes: a failed parallel temp is a sparse full-size file and is thrown away.
STREAMS_KEY = "download.streams_per_file"
FILES_KEY = "download.files_at_once"
# Jobs not yet finished, so a deploy or restart picks them up again instead of dropping them.
PENDING_KEY = "download.pending"


def _clamped(key: str, hi: int) -> int:
    try:
        return max(1, min(int(db.get_setting(key, "1") or 1), hi))
    except ValueError:
        return 1


def streams_per_file() -> int:
    return _clamped(STREAMS_KEY, PARALLEL_CHUNKS)


def files_at_once() -> int:
    return _clamped(FILES_KEY, MAX_FILES_AT_ONCE)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def published_sha256(responses) -> str:
    """The file's SHA-256 from a resolve URL's answers, or "".

    huggingface.co answers with `x-linked-etag` (the sha256 of the bytes) and then redirects to a
    CDN whose answer carries only `etag`, a different 64-hex value that matches nothing on disk
    (strata_models.fetch_etags explains the cost of mixing them up). So only x-linked-etag counts,
    from whichever response in the redirect chain has it.
    """
    for r in responses:
        v = (r.headers.get("x-linked-etag") or "").strip().strip('"').lower()
        if _SHA256_RE.match(v):
            return v
    return ""


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


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
    status: str = "queued"       # queued | downloading | verifying | done | error | canceled
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    chunks: list[ChunkState] = field(default_factory=list)
    parallel: bool = False
    # The file's SHA-256 as Hugging Face publishes it ("" when the server gave none), and whether
    # the bytes on disk were checked against it.
    expected_sha256: str = ""
    verified: bool = False
    _hasher: object = field(default=None, repr=False)
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


def _models_root() -> Path:
    return settings.models_dir


def _hf_auth(url: str) -> dict[str, str]:
    """The Hugging Face token, for huggingface.co URLs only: a URL import from anywhere else must
    never be handed it."""
    tok = hf.get_token()
    return {"Authorization": f"Bearer {tok}"} if tok and "huggingface.co" in url else {}


class DownloadManager:
    """Queued, resumable, verified downloads into one directory.

    Where files land and what auth a URL gets are passed in, so the manager is not tied to the
    models directory or to Hugging Face; this app's one instance (`manager`, below) uses both.
    `filename` is always relative to `dest_root()`.
    """

    def __init__(self, dest_root=_models_root, auth=_hf_auth) -> None:
        self.jobs: dict[str, DownloadJob] = {}
        self._dest_root = dest_root
        self._auth = auth

    async def _wait_turn(self, job: DownloadJob) -> None:
        """Hold a queued job until it is the oldest one waiting and a slot is free.

        Read per check rather than fixed at start (it was a Semaphore sized once from config), so
        changing "files at once" on the Downloads page applies to the queue already there.
        """
        while True:
            active = sum(1 for j in self.jobs.values() if j.status == "downloading")
            first = next((j for j in self.jobs.values() if j.status == "queued"), None)
            if active < files_at_once() and first is job:
                return
            await asyncio.sleep(0.25)

    # ------------------------------------------------------------ surviving a restart

    def _save_pending(self) -> None:
        rows = [{"repo_id": j.repo_id, "filename": j.filename, "url": j.url,
                 "total_bytes": j.total_bytes}
                for j in self.jobs.values() if j.status in ("queued", "downloading", "verifying")]
        try:
            db.set_setting(PENDING_KEY, json.dumps(rows))
        except Exception:  # noqa: BLE001 - losing the restart list must not fail a download
            pass

    def resume_pending(self) -> int:
        """Queue again whatever was unfinished when the app last stopped. Called once at startup,
        from the UI process only. A single-stream temp resumes from where it stopped."""
        try:
            rows = json.loads(db.get_setting(PENDING_KEY, "[]") or "[]")
        except ValueError:
            rows = []
        for r in rows:
            try:
                self._make_job(repo_id=r["repo_id"], filename=r["filename"], url=r["url"],
                               total_bytes=int(r.get("total_bytes") or 0))
            except (KeyError, TypeError, ValueError):
                continue
        return len(rows)

    def snapshot(self) -> list[DownloadJob]:
        # active first, then most recent
        def sort_key(j: DownloadJob) -> tuple[int, float]:
            active_rank = {"downloading": 0, "queued": 1}.get(j.status, 2)
            return (active_rank, -(j.completed_at or j.started_at or 0.0))
        return sorted(self.jobs.values(), key=sort_key)

    def _make_job(self, *, repo_id: str, filename: str, url: str, total_bytes: int) -> DownloadJob:
        dest = self._dest_root() / filename
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
        self._save_pending()
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
        shutting_down = False
        try:
            await self._wait_turn(job)
            job.status = "downloading"
            job.started_at = time.time()
            await self._stream(job)
            await self._verify(job)
            os.replace(job.temp_path, job.dest_path)
            job.status = "done"
            job.completed_at = time.time()
        except asyncio.CancelledError:
            if not job._cancel.is_set():
                # Shutdown, not the cancel button: a deploy or restart stops the event loop,
                # which cancels every task. Treating that as a cancel deleted the partial file
                # and dropped the job from the restart list, so "survives a restart" never
                # did. The temp stays (with its ranges file, for a parallel one), the job keeps
                # its status, and the next start re-queues and resumes it.
                shutting_down = True
                raise
            job.status = "canceled"
            job.completed_at = time.time()
            for p in (job.temp_path, _ranges_path(job)):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.completed_at = time.time()
            if job.parallel and not _ranges_path(job).exists():
                # A parallel temp is a PREALLOCATED full-size sparse file. Without its ranges
                # file there is no telling the holes from the data, and keeping one is how a
                # later attempt once promoted a hole-ridden file as a finished model. With the
                # ranges file, it is kept: downloading the same file again resumes it.
                try:
                    job.temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        finally:
            self._save_pending()
            if not shutting_down:      # a stop is not an outcome worth a history row
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

    async def _verify(self, job: DownloadJob) -> None:
        """Check the finished temp against the published SHA-256 before it becomes the model.

        A single stream hashes as it writes, so only a parallel download (whose ranges land out of
        order) reads the file back. A mismatch deletes the temp: those bytes are not the file, and
        a resume from them would only build on the damage.
        """
        if not job.expected_sha256:
            return
        job.status = "verifying"   # off the network, so the next queued file may start
        if job._hasher is not None and not job.parallel:
            got = job._hasher.hexdigest()
        else:
            got = await asyncio.to_thread(sha256_file, job.temp_path)
        if got != job.expected_sha256:
            job.temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch: got {got[:12]}..., Hugging Face publishes "
                               f"{job.expected_sha256[:12]}...; the download was discarded")
        job.verified = True

    def _auth_headers(self, url: str) -> dict[str, str]:
        return dict(self._auth(url))

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
                    job.expected_sha256 = published_sha256([*h.history, h])
        except httpx.HTTPError:
            pass  # fall through and try single-stream

        if total and total > 0:
            job.total_bytes = total

        # Choose parallel vs single-stream. A parallel download that stopped resumes as one,
        # whatever the setting is now: its temp is full-size with holes, which the
        # append-only single stream would take for a finished file.
        streams = streams_per_file()
        ranges = _load_ranges(job, total) if accept_ranges else None
        if accept_ranges and total >= PARALLEL_MIN_SIZE and (ranges or streams > 1):
            await self._stream_parallel(job, resolved_url, headers, total, streams, ranges)
        else:
            _discard_ranges(job)
            await self._stream_single(job, url, headers)

    async def _stream_parallel(self, job: DownloadJob, url: str, headers: dict[str, str], total: int,
                               n: int | None = None, ranges: list | None = None) -> None:
        # Deliberately no "temp size == total means complete" shortcut: the temp is
        # preallocated to full size, so that check was true of every failed run's sparse
        # leftover, and the retry promoted a hole-ridden file as a finished model.
        # Completion is proven by received bytes, never by file size. What each range has
        # received is kept in the ranges file beside the temp, which is what lets a stopped
        # parallel download resume instead of starting over.
        job.parallel = True
        if ranges:
            job.chunks = [ChunkState(idx=i, start=s, end=e, downloaded=d)
                          for i, (s, e, d) in enumerate(ranges)]
        else:
            # Preallocate to full size; each worker seeks and writes its own region.
            with open(job.temp_path, "wb") as f:
                f.truncate(total)
            n = n or PARALLEL_CHUNKS
            chunk = total // n
            job.chunks = []
            for i in range(n):
                start = i * chunk
                end = (start + chunk - 1) if i < n - 1 else total - 1
                job.chunks.append(ChunkState(idx=i, start=start, end=end))
        job.downloaded_bytes = sum(c.downloaded for c in job.chunks)
        job._speed_samples.append((time.time(), job.downloaded_bytes))
        _save_ranges(job, total)

        async def worker(cs: ChunkState) -> None:
            if cs.downloaded >= cs.size:
                return                    # finished before the stop
            worker_headers = dict(headers)
            worker_headers["Range"] = f"bytes={cs.start + cs.downloaded}-{cs.end}"
            timeout = httpx.Timeout(30.0, read=120.0)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=worker_headers) as w:
                async with w.stream("GET", url) as resp:
                    # 206 or nothing: a 200 means the server ignored the Range header,
                    # and eight workers each writing the WHOLE body at their own offset
                    # is not a download, it is a confetti cannon pointed at the temp.
                    if resp.status_code != 206:
                        body = ""
                        try:
                            body = (await resp.aread()).decode(errors="replace")[:200]
                        except Exception:
                            pass
                        raise RuntimeError(f"HTTP {resp.status_code} for range {cs.start}-{cs.end}: {body}")
                    # Unbuffered, so a byte counted in the ranges file has reached the OS: a
                    # stopped process loses nothing it claimed (a power cut still can, which is
                    # what the SHA-256 check is for).
                    with open(job.temp_path, "r+b", buffering=0) as f:
                        f.seek(cs.start + cs.downloaded)
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
                    # A 206 that ends early is a truncated chunk, and the stream closing
                    # cleanly is not the same statement as "every byte arrived".
                    if cs.downloaded != cs.size:
                        raise RuntimeError(
                            f"range {cs.start}-{cs.end}: got {cs.downloaded} of {cs.size} bytes")

        async def keep_ranges() -> None:
            while True:
                await asyncio.sleep(2.0)
                _save_ranges(job, total)

        saver = asyncio.create_task(keep_ranges())
        tasks = [asyncio.create_task(worker(c)) for c in job.chunks]
        try:
            await asyncio.gather(*tasks)
            saver.cancel()
            _ranges_path(job).unlink(missing_ok=True)
        except BaseException:
            saver.cancel()
            for t in tasks:
                if not t.done():
                    t.cancel()
            # allow cancellations to settle
            for t in tasks:
                try:
                    await t
                except BaseException:
                    pass
            _save_ranges(job, total)
            raise

    async def _stream_single(self, job: DownloadJob, url: str, headers: dict[str, str]) -> None:
        # Original resume-aware single-stream path (used when server doesn't support ranges, or file is small).
        # Two rounds at most: round two only happens when a 416 says the resumable temp is
        # oversized, in which case it is not resumable and goes in the bin.
        for attempt in (0, 1):
            start = 0
            if job.temp_path.exists():
                start = job.temp_path.stat().st_size
            # Hash as it writes, so verifying a single stream costs no second read. A resumed
            # temp's existing bytes are hashed first (once, in a thread).
            job._hasher = hashlib.sha256()
            if start and job.expected_sha256:
                await asyncio.to_thread(_hash_prefix, job._hasher, job.temp_path, start)
            if start and job.total_bytes and start == job.total_bytes:
                # Already whole (stopped between the last byte and the rename). Asking for bytes
                # past the end gets a 416, which read as "oversized" and threw the file away.
                job.downloaded_bytes = start
                return
            req_headers = {**headers, "Range": f"bytes={start}-"} if start > 0 else dict(headers)
            job.downloaded_bytes = start
            if attempt == 0:
                job._speed_samples.append((time.time(), job.downloaded_bytes))

            timeout = httpx.Timeout(30.0, read=120.0)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=req_headers) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code == 416 and start > 0:
                        # 416 means the temp is BIGGER than the remote file - stale or
                        # corrupt, not finished. It used to be taken as "complete" and
                        # promoted, oversized, over the destination.
                        if attempt == 0:
                            job.temp_path.unlink(missing_ok=True)
                            job.downloaded_bytes = 0
                            continue
                        raise RuntimeError(f"HTTP 416 from {url} after restarting the download")
                    if resp.status_code not in (200, 206):
                        body = ""
                        try:
                            body = (await resp.aread()).decode(errors="replace")[:200]
                        except Exception:
                            pass
                        raise RuntimeError(f"HTTP {resp.status_code} from {url}: {body}")

                    # We asked to resume and the server ignored the Range header: this
                    # body is the WHOLE file, and appending it onto the partial would
                    # corrupt the model. Restart the temp from zero instead.
                    if start > 0 and resp.status_code == 200:
                        start = 0
                        job.downloaded_bytes = 0
                        job._hasher = hashlib.sha256()

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
                            job._hasher.update(buf)
                            self._note_progress(job, len(buf))

                    # A stream that ends without error has not said it sent everything:
                    # a server closing a chunked body early just... ends. Count it.
                    if job.total_bytes and job.downloaded_bytes != job.total_bytes:
                        raise RuntimeError(
                            f"stream ended at {job.downloaded_bytes} of {job.total_bytes} bytes")
                    return


def _ranges_path(job: DownloadJob) -> Path:
    return job.temp_path.with_name(job.temp_path.name + ".ranges")


def _save_ranges(job: DownloadJob, total: int) -> None:
    """What each range of a parallel download has received, beside its temp. Atomic."""
    p = _ranges_path(job)
    data = {"url": job.url, "total": total,
            "ranges": [[c.start, c.end, c.downloaded] for c in job.chunks]}
    try:
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass


def _load_ranges(job: DownloadJob, total: int) -> list | None:
    """The ranges file of a stopped parallel download of THIS file, or None. It must name the
    same URL and size as now and sit beside a full-size temp, or it describes something else."""
    try:
        data = json.loads(_ranges_path(job).read_text(encoding="utf-8"))
        if (data.get("url") != job.url or int(data.get("total") or 0) != total or not total
                or job.temp_path.stat().st_size != total):
            return None
        ranges = [[int(s), int(e), int(d)] for s, e, d in data["ranges"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not ranges or ranges[0][0] != 0 or ranges[-1][1] != total - 1:
        return None
    if any(not (0 <= d <= e - s + 1) for s, e, d in ranges):
        return None
    return ranges


def _discard_ranges(job: DownloadJob) -> None:
    """A ranges file the single stream cannot use: its temp is sparse, not appendable, so both go."""
    p = _ranges_path(job)
    if p.exists():
        for f in (p, job.temp_path):
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass


def _hash_prefix(hasher, path: Path, n: int, chunk: int = 8 * 1024 * 1024) -> None:
    with open(path, "rb") as f:
        left = n
        while left > 0:
            buf = f.read(min(chunk, left))
            if not buf:
                break
            hasher.update(buf)
            left -= len(buf)


manager = DownloadManager()
