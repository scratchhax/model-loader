"""Characterization tests for the chunked download paths in app/downloader.py.

httpx.MockTransport drives the real _stream_single/_stream_parallel code against
tmp files: resume, 416-oversized-temp restart, Range-ignored restart, truncated
streams, and the parallel 206-or-nothing rule.
"""
import asyncio
import functools

import httpx
import pytest

from app.downloader import DownloadJob, manager


def _job(tmp_path, total=0):
    return DownloadJob(id="j1", repo_id="owner/repo", filename="m.gguf",
                       url="http://srv/m.gguf", dest_path=tmp_path / "m.gguf",
                       temp_path=tmp_path / "m.gguf.part", total_bytes=total)


@pytest.fixture
def client_with(monkeypatch):
    """Patch httpx.AsyncClient so every client the downloader builds uses a
    MockTransport driven by the handler the test passes in."""
    def install(handler):
        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(httpx, "AsyncClient",
                            functools.partial(httpx.AsyncClient, transport=transport))
        return transport
    return install


class TestStreamSingle:
    def test_fresh_200(self, tmp_path, client_with):
        job = _job(tmp_path)
        client_with(lambda req: httpx.Response(200, content=b"A" * 1000))
        asyncio.run(manager._stream_single(job, job.url, {}))
        assert job.temp_path.read_bytes() == b"A" * 1000
        assert (job.total_bytes, job.downloaded_bytes) == (1000, 1000)

    def test_resume_206_appends(self, tmp_path, client_with):
        job = _job(tmp_path)
        job.temp_path.write_bytes(b"A" * 400)
        seen = []

        def handler(req):
            seen.append(req.headers.get("range"))
            return httpx.Response(206, content=b"B" * 600,
                                  headers={"content-range": "bytes 400-999/1000"})
        client_with(handler)
        asyncio.run(manager._stream_single(job, job.url, {}))
        assert seen == ["bytes=400-"]
        assert job.temp_path.read_bytes() == b"A" * 400 + b"B" * 600
        assert (job.total_bytes, job.downloaded_bytes) == (1000, 1000)

    def test_416_oversized_temp_restarts_from_zero(self, tmp_path, client_with):
        job = _job(tmp_path)
        job.temp_path.write_bytes(b"S" * 1200)
        seen = []

        def handler(req):
            rng = req.headers.get("range")
            seen.append(rng)
            if rng is not None:
                return httpx.Response(416)
            return httpx.Response(200, content=b"C" * 1000)
        client_with(handler)
        asyncio.run(manager._stream_single(job, job.url, {}))
        assert seen == ["bytes=1200-", None]
        assert job.temp_path.read_bytes() == b"C" * 1000

    def test_range_ignored_200_restarts_temp(self, tmp_path, client_with):
        job = _job(tmp_path)
        job.temp_path.write_bytes(b"A" * 400)
        client_with(lambda req: httpx.Response(200, content=b"D" * 1000))
        asyncio.run(manager._stream_single(job, job.url, {}))
        # Appending the full body onto the 400-byte partial would give 1400 bytes.
        assert job.temp_path.read_bytes() == b"D" * 1000

    def test_truncated_stream_raises(self, tmp_path, client_with):
        job = _job(tmp_path)
        client_with(lambda req: httpx.Response(
            200, content=b"E" * 700, headers={"content-length": "1000"}))
        with pytest.raises(RuntimeError, match="stream ended at 700 of 1000"):
            asyncio.run(manager._stream_single(job, job.url, {}))

    def test_error_status_raises_with_body(self, tmp_path, client_with):
        job = _job(tmp_path)
        client_with(lambda req: httpx.Response(403, content=b"nope"))
        with pytest.raises(RuntimeError, match="HTTP 403.*nope"):
            asyncio.run(manager._stream_single(job, job.url, {}))


def _range_handler(total=4000, length_of=None, status=206):
    def handler(req):
        start, end = (int(v) for v in req.headers["range"].removeprefix("bytes=").split("-"))
        want = end - start + 1
        body = bytes((start + i) % 251 for i in range(want if length_of is None
                     else length_of(start, want)))
        return httpx.Response(status, content=body)
    return handler


class TestStreamParallel:
    def test_eight_ranges_reassemble(self, tmp_path, client_with):
        total = 4000
        job = _job(tmp_path, total=total)
        client_with(_range_handler(total))
        asyncio.run(manager._stream_parallel(job, job.url, {}, total))
        data = job.temp_path.read_bytes()
        assert len(data) == total
        assert job.parallel is True and len(job.chunks) == 8
        assert job.downloaded_bytes == total
        for off in (0, 500, 1500, 3999):
            assert data[off] == off % 251

    def test_200_instead_of_206_refused(self, tmp_path, client_with):
        total = 4000
        job = _job(tmp_path, total=total)
        client_with(_range_handler(total, status=200))
        with pytest.raises(RuntimeError, match="HTTP 200 for range"):
            asyncio.run(manager._stream_parallel(job, job.url, {}, total))

    def test_short_chunk_raises(self, tmp_path, client_with):
        total = 4000
        job = _job(tmp_path, total=total)
        client_with(_range_handler(total, length_of=lambda s, w: w - 1))
        with pytest.raises(RuntimeError, match="got 499 of 500"):
            asyncio.run(manager._stream_parallel(job, job.url, {}, total))
