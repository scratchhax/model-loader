from __future__ import annotations

import shlex
import time

import pytest
from docker.errors import DockerException

from app import utils


class TestHumanBytes:
    @pytest.mark.parametrize("n, want", [
        (0, "0 B"),
        (512, "512 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1024 ** 2, "1.0 MB"),
        (1024 ** 3, "1.0 GB"),
        (15.3 * 1024 ** 3, "15.3 GB"),
        (1024 ** 4, "1.0 TB"),
        (1024 ** 5, "1.0 PB"),
        (1024 ** 6, "1.0 EB"),
    ])
    def test_sizes(self, n, want):
        assert utils.human_bytes(n) == want

    def test_negative_uses_abs(self):
        # The loop compares abs(n) but divides the signed value, so a negative
        # input walks the units the same way and keeps its sign.
        assert utils.human_bytes(-1536) == "-1.5 KB"


class TestShardKey:
    def test_shard_split(self):
        assert utils.shard_key("Model-Q4-00001-of-00003.gguf") == ("Model-Q4.gguf", 1, 3)

    def test_not_a_shard(self):
        assert utils.shard_key("Model-Q4_K_M.gguf") == ("Model-Q4_K_M.gguf", None, None)

    def test_shard_must_end_the_name(self):
        # The lookahead pins the pattern to just before the final extension, so a
        # mid-name match is not a shard marker.
        assert utils.shard_key("a-00001-of-00002-extra.gguf") == (
            "a-00001-of-00002-extra.gguf", None, None)

    def test_other_extension(self):
        assert utils.shard_key("a-00002-of-00002.bin") == ("a.bin", 2, 2)


class TestTimedShell:
    def test_shape(self):
        assert utils.timed_shell("nvidia-smi", 5) == [
            "sh", "-c", "timeout 5 sh -c nvidia-smi"]

    def test_quotes_dangerous_cmd(self):
        cmd = 'rocm-smi --showmeminfo vram --json | grep "a b"'
        out = utils.timed_shell(cmd, 3)
        assert out[2] == f"timeout 3 sh -c {shlex.quote(cmd)}"


class FakeClient:
    def __init__(self, ping_ok=True):
        self.ping_ok = ping_ok
        self.closed = False
        self.pings = 0

    def ping(self):
        self.pings += 1
        if not self.ping_ok:
            raise DockerException("dead dockerd")

    def close(self):
        self.closed = True


@pytest.fixture
def fresh_docker(monkeypatch):
    monkeypatch.setattr(utils, "_DOCKER_CLIENT", None)
    monkeypatch.setattr(utils, "_DOCKER_CLIENT_OK", 0.0)
    yield monkeypatch


class TestDockerClient:
    def test_cached_within_ttl(self, fresh_docker):
        calls = []
        client = FakeClient()

        def fake_from_env():
            calls.append(1)
            return client

        fresh_docker.setattr(utils.docker, "from_env", fake_from_env)
        assert utils.docker_client() is client
        assert utils.docker_client() is client
        assert len(calls) == 1
        assert client.pings == 0  # inside the 5 s window not even a ping

    def test_rebuilt_after_failed_ping(self, fresh_docker):
        old = FakeClient(ping_ok=False)
        new = FakeClient()
        fresh_docker.setattr(utils, "_DOCKER_CLIENT", old)
        fresh_docker.setattr(utils, "_DOCKER_CLIENT_OK", time.time() - 10)
        made = []

        def fake_from_env():
            made.append(new)
            return new

        fresh_docker.setattr(utils.docker, "from_env", fake_from_env)
        assert utils.docker_client() is new
        assert old.closed

    def test_none_when_unreachable(self, fresh_docker):
        def boom():
            raise DockerException("no socket")

        fresh_docker.setattr(utils.docker, "from_env", boom)
        assert utils.docker_client() is None
