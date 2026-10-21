from __future__ import annotations

import asyncio
import time

import pytest

from app import db, hf
from app.config import settings
from app.downloader import ChunkState, DownloadJob, DownloadManager

MiB = 1024 * 1024


class TestChunkState:
    def test_size_pct_and_status_transitions(self):
        c = ChunkState(idx=0, start=100, end=199)
        assert c.size == 100
        assert c.status == "pending"
        c.downloaded = 50
        assert c.pct == 50.0 and c.status == "downloading"
        c.downloaded = 150
        assert c.pct == 100.0 and c.status == "done"

    def test_speed_over_5s_window(self):
        c = ChunkState(idx=0, start=0, end=999)
        now = time.time()
        c._samples = [(now - 4, 0), (now - 2, 2000), (now, 4000)]
        assert c.speed_bps == pytest.approx(1000.0)

    def test_speed_needs_two_samples_in_window(self):
        c = ChunkState(idx=0, start=0, end=999)
        now = time.time()
        c._samples = [(now - 10, 0), (now - 9, 1000)]  # both outside the window
        assert c.speed_bps == 0.0
        assert c.speed_h == "—"


def job(**over):
    d = dict(id="j1", repo_id="o/r", filename="m.gguf", url="http://x/m.gguf",
             dest_path=None, temp_path=None, total_bytes=1000)
    return DownloadJob(**{**d, **over})


class TestDownloadJob:
    def test_pct_clamps_and_zero_total(self):
        j = job(downloaded_bytes=250)
        assert j.pct == 25.0
        j.downloaded_bytes = 2000
        assert j.pct == 100.0
        assert job(total_bytes=0).pct == 0.0

    def test_eta_formats(self):
        now = time.time()
        j = job()
        j._speed_samples = [(now - 2, 0), (now, 500)]  # 250 B/s, 1000 left
        assert j.eta_seconds == 4 and j.eta_h == "4s"
        j.downloaded_bytes = 880
        j._speed_samples = [(now - 4, 876), (now, 880)]  # 1 B/s, 120 left
        assert j.eta_h == "2m 0s"
        j.total_bytes = 7261
        j.downloaded_bytes = 1
        j._speed_samples = [(now - 4, 0), (now, 8)]  # 2 B/s, 7260 left
        assert j.eta_h == "1h 0m"

    def test_eta_none_without_speed(self):
        assert job().eta_seconds is None and job().eta_h == "—"

    def test_speed_history_and_sparkline(self):
        j = job()
        j._speed_samples = [(0.0, 0), (1.0, MiB), (2.0, 3 * MiB)]
        assert j.speed_history_mbps == [1.0, 2.0]
        assert j.sparkline_points == "0.0,11.0 100.0,2.0"
        j._speed_samples = [(0.0, 0)]
        assert j.sparkline_points is None


@pytest.fixture
def mgr(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "d.db")   # the restart list is written there
    db.init()

    async def noop(self, j):
        pass

    monkeypatch.setattr(DownloadManager, "_run", noop)
    return DownloadManager()


class TestDownloadManager:
    def test_enqueue_builds_hf_url_and_dedups_active(self, mgr):
        async def scenario():
            j1 = mgr.enqueue(repo_id="o/r", hf_path="dir/f.gguf",
                             filename="f.gguf", total_bytes=10)
            j2 = mgr.enqueue(repo_id="o/r", hf_path="dir/f.gguf",
                             filename="f.gguf", total_bytes=10)
            assert j1 is j2  # same dest, still queued -> no second job
            assert j1.url == f"{hf.HF_RESOLVE}/o/r/resolve/main/dir/f.gguf"
            assert j1.dest_path == mgr.jobs[j1.id].dest_path
            assert j1.temp_path.suffix == ".download"
            j1._task.cancel()
        asyncio.run(scenario())

    def test_enqueue_url_uses_host_as_repo(self, mgr):
        async def scenario():
            j = mgr.enqueue_url(url="https://example.com/a/b.gguf", filename="b.gguf")
            assert j.repo_id == "example.com"
            j._task.cancel()
        asyncio.run(scenario())

    def test_cancel_unknown_and_finished(self, mgr):
        assert mgr.cancel("nope") is False
        j = job(status="done")
        mgr.jobs["j1"] = j
        assert mgr.cancel("j1") is False

    def test_snapshot_orders_active_first_then_recent(self, mgr):
        old_done = job(id="a", status="done", completed_at=100.0)
        new_done = job(id="b", status="done", completed_at=200.0)
        active = job(id="c", status="downloading", started_at=50.0)
        queued = job(id="d", status="queued")
        mgr.jobs.update({j.id: j for j in (old_done, new_done, active, queued)})
        assert [j.id for j in mgr.snapshot()] == ["c", "d", "b", "a"]

    def test_clear_finished_removes_only_terminal(self, mgr):
        mgr.jobs["a"] = job(id="a", status="error")
        mgr.jobs["b"] = job(id="b", status="canceled")
        mgr.jobs["c"] = job(id="c", status="downloading")
        assert mgr.clear_finished() == 2
        assert list(mgr.jobs) == ["c"]
