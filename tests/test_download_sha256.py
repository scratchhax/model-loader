"""Downloads are checked against the SHA-256 Hugging Face publishes (review item 18)."""
from __future__ import annotations

import asyncio
import functools
import hashlib

import httpx
import pytest

from app import downloader
from app.downloader import DownloadJob, manager

BODY = b"model bytes " * 1000
SHA = hashlib.sha256(BODY).hexdigest()


def _job(tmp_path, total=0, sha=""):
    return DownloadJob(id="j1", repo_id="o/r", filename="m.gguf", url="http://srv/m.gguf",
                       dest_path=tmp_path / "m.gguf", temp_path=tmp_path / "m.gguf.download",
                       total_bytes=total, expected_sha256=sha)


@pytest.fixture
def client_with(monkeypatch):
    def install(handler):
        monkeypatch.setattr(httpx, "AsyncClient",
                            functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler)))
    return install


def _range_handler(req):
    rng = req.headers.get("range")
    if rng:
        start = int(rng.split("=")[1].split("-")[0])
        return httpx.Response(206, content=BODY[start:],
                              headers={"content-range": f"bytes {start}-{len(BODY) - 1}/{len(BODY)}"})
    return httpx.Response(200, content=BODY)


def test_only_x_linked_etag_counts():
    hub = httpx.Response(302, headers={"x-linked-etag": f'"{SHA}"', "etag": '"' + "a" * 64 + '"'})
    cdn = httpx.Response(200, headers={"etag": '"' + "b" * 64 + '"'})
    assert downloader.published_sha256([hub, cdn]) == SHA
    assert downloader.published_sha256([cdn]) == ""
    assert downloader.published_sha256([httpx.Response(200, headers={"x-linked-etag": "abc"})]) == ""


def test_single_stream_hashes_as_it_writes_and_verifies(tmp_path, client_with):
    job = _job(tmp_path, sha=SHA)
    client_with(_range_handler)
    asyncio.run(manager._stream_single(job, job.url, {}))
    assert job._hasher.hexdigest() == SHA
    asyncio.run(manager._verify(job))
    assert job.verified and job.temp_path.exists()


def test_a_resumed_temp_hashes_its_existing_bytes(tmp_path, client_with):
    job = _job(tmp_path, total=len(BODY), sha=SHA)
    job.temp_path.write_bytes(BODY[:5000])
    client_with(_range_handler)
    asyncio.run(manager._stream_single(job, job.url, {}))
    assert job.temp_path.read_bytes() == BODY
    asyncio.run(manager._verify(job))
    assert job.verified


def test_a_whole_temp_is_not_refetched(tmp_path, client_with):
    job = _job(tmp_path, total=len(BODY), sha=SHA)
    job.temp_path.write_bytes(BODY)

    def no_request(req):
        raise AssertionError("asked the server for bytes past the end")
    client_with(no_request)
    asyncio.run(manager._stream_single(job, job.url, {}))
    asyncio.run(manager._verify(job))
    assert job.verified and job.downloaded_bytes == len(BODY)


def test_mismatch_discards_the_temp(tmp_path, client_with):
    job = _job(tmp_path, sha="0" * 64)
    client_with(_range_handler)
    asyncio.run(manager._stream_single(job, job.url, {}))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        asyncio.run(manager._verify(job))
    assert not job.temp_path.exists() and not job.verified


def test_parallel_reads_the_file_back(tmp_path):
    job = _job(tmp_path, sha=SHA)
    job.temp_path.write_bytes(BODY)
    job.parallel = True
    asyncio.run(manager._verify(job))
    assert job.verified


def test_no_published_hash_is_not_checked(tmp_path):
    job = _job(tmp_path)
    job.temp_path.write_bytes(b"anything")
    asyncio.run(manager._verify(job))
    assert not job.verified and job.temp_path.exists()
