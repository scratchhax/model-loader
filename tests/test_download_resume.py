"""Downloads survive a stop, and a multi-stream one resumes from its ranges file (review item 17).

A deploy stops the event loop, which cancels every task, and the download task took that for the
cancel button: it deleted the partial file and dropped the job from the restart list, so #103's
"survives a restart" did not. And a multi-stream download could not resume at all, because its
temp is preallocated at full size and nothing said which parts were real.
"""
from __future__ import annotations

import asyncio
import functools
import json

import httpx
import pytest

from app import db, downloader
from app.config import settings
from app.downloader import DownloadJob, DownloadManager, manager

BODY = bytes(range(256)) * 400          # 102,400 bytes


@pytest.fixture
def kv(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "d.db")
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    db.init()
    return tmp_path


# ------------------------------------------------------------------ a stop is not a cancel

def _slow_stream(monkeypatch):
    async def stream(self, job):
        job.temp_path.parent.mkdir(parents=True, exist_ok=True)
        job.temp_path.write_bytes(b"partial")
        await asyncio.sleep(3600)
    monkeypatch.setattr(DownloadManager, "_stream", stream)


def test_a_shutdown_keeps_the_temp_and_the_restart_entry(kv, monkeypatch):
    _slow_stream(monkeypatch)

    async def scenario():
        m = DownloadManager()
        j = m.enqueue_url(url="https://e.com/a.gguf", filename="a.gguf")
        await asyncio.sleep(0.4)
        j._task.cancel()                       # what stopping the event loop does
        with pytest.raises(asyncio.CancelledError):
            await j._task
        return j
    j = asyncio.run(scenario())
    assert j.status == "downloading" and j.temp_path.read_bytes() == b"partial"
    assert [r["filename"] for r in json.loads(db.get_setting(downloader.PENDING_KEY))] == ["a.gguf"]


def test_the_cancel_button_still_deletes(kv, monkeypatch):
    _slow_stream(monkeypatch)

    async def scenario():
        m = DownloadManager()
        j = m.enqueue_url(url="https://e.com/a.gguf", filename="a.gguf")
        await asyncio.sleep(0.4)
        m.cancel(j.id)
        try:
            await j._task
        except asyncio.CancelledError:
            pass
        return j
    j = asyncio.run(scenario())
    assert j.status == "canceled" and not j.temp_path.exists()
    assert json.loads(db.get_setting(downloader.PENDING_KEY)) == []


# ------------------------------------------------------------------ multi-stream resume

def _job(tmp_path):
    return DownloadJob(id="j1", repo_id="o/r", filename="m.gguf", url="http://srv/m.gguf",
                       dest_path=tmp_path / "m.gguf", temp_path=tmp_path / "m.gguf.download",
                       total_bytes=len(BODY))


def _serve(monkeypatch, seen, short_once=None):
    """Range server over BODY. `short_once`: a range start whose first answer stops 1,000 bytes in."""
    state = {"cut": short_once}

    def handler(req):
        a, b = req.headers["range"].split("=")[1].split("-")
        a, b = int(a), int(b)
        seen.append((a, b))
        data = BODY[a:b + 1]
        if state["cut"] == a:
            state["cut"] = None
            data = data[:1000]
        return httpx.Response(206, content=data,
                              headers={"content-range": f"bytes {a}-{b}/{len(BODY)}"})
    monkeypatch.setattr(httpx, "AsyncClient",
                        functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler)))


def test_a_stopped_parallel_download_fetches_only_what_is_missing(tmp_path, monkeypatch):
    job = _job(tmp_path)
    seen: list = []
    chunk = len(BODY) // 4
    _serve(monkeypatch, seen, short_once=2 * chunk)          # range 3 of 4 is cut short
    with pytest.raises(RuntimeError, match="got 1000 of"):
        asyncio.run(manager._stream_parallel(job, job.url, {}, len(BODY), 4))
    ranges = downloader._load_ranges(job, len(BODY))
    assert ranges and ranges[2][2] == 1000 and job.temp_path.stat().st_size == len(BODY)

    job2, seen2 = _job(tmp_path), []
    _serve(monkeypatch, seen2)
    asyncio.run(manager._stream_parallel(job2, job2.url, {}, len(BODY), 4, ranges))
    assert job2.temp_path.read_bytes() == BODY
    done = {r[0] for r in ranges if r[2] == r[1] - r[0] + 1}
    assert all(a not in done for a, _ in seen2)                # finished ranges not asked again
    assert (2 * chunk + 1000, 3 * chunk - 1) in seen2          # the cut one picks up where it stopped
    assert not downloader._ranges_path(job2).exists()          # gone once complete


def test_a_ranges_file_for_something_else_is_ignored(tmp_path):
    job = _job(tmp_path)
    job.temp_path.write_bytes(b"\0" * len(BODY))
    p = downloader._ranges_path(job)
    p.write_text(json.dumps({"url": "http://other/x", "total": len(BODY),
                             "ranges": [[0, len(BODY) - 1, 5]]}))
    assert downloader._load_ranges(job, len(BODY)) is None
    p.write_text(json.dumps({"url": job.url, "total": len(BODY), "ranges": [[0, len(BODY) - 1, 5]]}))
    assert downloader._load_ranges(job, len(BODY)) == [[0, len(BODY) - 1, 5]]
    assert downloader._load_ranges(job, len(BODY) + 1) is None   # the file changed size


def test_the_single_stream_never_appends_to_a_sparse_temp(tmp_path):
    job = _job(tmp_path)
    job.temp_path.write_bytes(b"\0" * len(BODY))
    downloader._ranges_path(job).write_text("{}")
    downloader._discard_ranges(job)
    assert not job.temp_path.exists() and not downloader._ranges_path(job).exists()
