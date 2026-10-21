"""Review batch 2: downloads pull one stream by default and survive a restart, the router reuses one
backend discovery briefly and probes in parallel, and the Router tab's Strata row shows what starts."""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from app import db, downloader
from app import engine_controls
from app import router_api as ra
from app import strata_models
from app.config import settings
from app.downloader import DownloadJob, DownloadManager
from tests.test_routes_smoke import web  # noqa: F401 - the fixture: real app, empty box


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def kv(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "d.db")
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    db.init()
    return tmp_path


# ------------------------------------------------------------------ item 16: one stream unless told

class TestLimits:
    def test_default_is_one_and_one(self, kv):
        assert downloader.streams_per_file() == 1 and downloader.files_at_once() == 1

    def test_clamped(self, kv):
        db.set_setting(downloader.STREAMS_KEY, "99")
        db.set_setting(downloader.FILES_KEY, "0")
        assert downloader.streams_per_file() == downloader.PARALLEL_CHUNKS
        assert downloader.files_at_once() == 1
        db.set_setting(downloader.STREAMS_KEY, "lots")
        assert downloader.streams_per_file() == 1

    def test_page_sets_them(self, web):
        r = web.post("/downloads/limits", data={"streams": "2", "files": "1"})
        assert r.status_code == 200 and "up to 2 connections" in r.text
        assert downloader.streams_per_file() == 2
        page = web.get("/downloads").text
        assert 'hx-post="/downloads/limits"' in page and '<option value="2" selected' in page


class TestQueueGate:
    def test_second_file_waits_for_the_first(self, kv, monkeypatch):
        order: list[str] = []

        async def stream(self, job):
            order.append(f"start {job.filename}")
            await asyncio.sleep(0.3)
            job.temp_path.write_bytes(b"x")
            order.append(f"end {job.filename}")
        monkeypatch.setattr(DownloadManager, "_stream", stream)

        async def scenario():
            m = DownloadManager()
            a = m.enqueue_url(url="https://e.com/a.gguf", filename="a.gguf")
            b = m.enqueue_url(url="https://e.com/b.gguf", filename="b.gguf")
            await asyncio.gather(a._task, b._task)
            return a, b
        a, b = run(scenario())
        assert order == ["start a.gguf", "end a.gguf", "start b.gguf", "end b.gguf"]
        assert a.status == b.status == "done"


# ------------------------------------------------------------------ item 17: survives a restart

class TestPending:
    def test_unfinished_jobs_are_recorded_and_finished_ones_dropped(self, kv, monkeypatch):
        async def never(self, job):
            await asyncio.sleep(3600)
        monkeypatch.setattr(DownloadManager, "_stream", never)

        async def scenario():
            m = DownloadManager()
            j = m.enqueue(repo_id="o/r", hf_path="f.gguf", filename="d/f.gguf", total_bytes=7)
            await asyncio.sleep(0.05)
            saved = json.loads(db.get_setting(downloader.PENDING_KEY))
            m.cancel(j.id)
            try:
                await j._task
            except asyncio.CancelledError:
                pass
            return saved
        saved = run(scenario())
        assert saved == [{"repo_id": "o/r", "filename": "d/f.gguf",
                          "url": "https://huggingface.co/o/r/resolve/main/f.gguf", "total_bytes": 7}]
        assert json.loads(db.get_setting(downloader.PENDING_KEY)) == []

    def test_resume_requeues(self, kv, monkeypatch):
        db.set_setting(downloader.PENDING_KEY, json.dumps([
            {"repo_id": "o/r", "filename": "d/f.gguf", "url": "https://x/f.gguf", "total_bytes": 7},
            {"broken": True}]))

        async def noop(self, j):
            pass
        monkeypatch.setattr(DownloadManager, "_run", noop)

        async def scenario():
            m = DownloadManager()
            m.resume_pending()
            return list(m.jobs.values())
        jobs = run(scenario())
        assert [(j.filename, j.url, j.total_bytes) for j in jobs] == [("d/f.gguf", "https://x/f.gguf", 7)]
        assert isinstance(jobs[0], DownloadJob)


# ------------------------------------------------------------------ item 20: discovery reused and parallel

def _backend(name):
    return {"name": name, "engine": "llama", "running": True, "cpu_only": False,
            "url": f"http://{name}:8080", "models": [], "headers": {}, "started_at": ""}


class TestDiscovery:
    @pytest.fixture(autouse=True)
    def _two(self, monkeypatch):
        self.listings = 0

        def backends(include_stopped=False):
            self.listings += 1
            return [_backend("llama-a"), _backend("llama-b")]

        async def models_of(client, b):
            await asyncio.sleep(0.2)
            return [b["name"] + "-m"]

        async def ready(client, b):
            await asyncio.sleep(0.2)
            return True
        monkeypatch.setattr(ra, "_backends", backends)
        monkeypatch.setattr(ra, "_models_of", models_of)
        monkeypatch.setattr(ra, "_ready", ready)

    def test_probes_run_at_once(self):
        t = time.time()
        bs, by_model = run(ra._live_map())
        assert time.time() - t < 0.7          # 2 backends x 2 probes x 0.2 s, backends in parallel
        assert set(by_model) == {"llama-a-m", "llama-b-m"}

    def test_reused_within_the_ttl_and_cleared_by_a_switch(self, monkeypatch):
        monkeypatch.setattr(ra, "_LIVE_TTL_S", 60.0)
        run(ra._live_map())
        run(ra._live_map())
        assert self.listings == 1
        ra.invalidate_live_map()
        run(ra._live_map())
        assert self.listings == 2


# ------------------------------------------------------------------ Router tab: Strata row is what starts next

def test_strata_row_shows_the_next_start_not_a_stale_default(web, monkeypatch):
    db.set_setting("router.default.strata", "unsloth-ud-iq4_xs")
    monkeypatch.setattr(strata_models, "read_selection",
                        lambda path: {"FAMILY": "swift", "MODEL": "IQ3_S"})
    monkeypatch.setattr(engine_controls, "strata_choices",
                        lambda: [("unsloth-ud-iq4_xs", "U"), ("swift-iq3_s", "S")])
    r = web.get("/router")
    assert '<option value="swift-iq3_s" selected' in r.text
    assert '<option value="unsloth-ud-iq4_xs" selected' not in r.text
