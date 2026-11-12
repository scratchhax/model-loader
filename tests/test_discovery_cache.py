"""Container discovery is reused briefly and reads image names from the container's own config
(review item 21). One GPU-strip build used to make 211 docker API calls and took 2-3 s to answer
a 1 s poll: five discoveries, each listing and inspecting every container, plus an image
inspect per container for its tag."""
from __future__ import annotations

import threading

from app.services import backends


class FakeImage:
    def __init__(self):
        self.inspected = 0

    @property
    def tags(self):
        self.inspected += 1
        return ["from-the-image:tag"]

    short_id = "sha256:abc"


class FakeContainer:
    def __init__(self, name, image, labels=None):
        self.name = name
        self.image = FakeImage()
        self.attrs = {"Config": {"Image": image, "Labels": labels or {}, "Cmd": [], "Env": []}}
        self.labels = labels or {}


class FakeClient:
    def __init__(self, cs):
        self.calls = 0
        outer = self

        class Containers:
            def list(self, all=False):
                outer.calls += 1
                return cs
        self.containers = Containers()


def _with(monkeypatch, cs):
    """The fake docker, visible to this test's thread only. The cache is process-wide, and any
    earlier test that started the app leaves its hw-sampler thread calling discovery every
    second. Seeing the fake, that thread wrote its own result into the cache (an empty list,
    when it had listed the real socket a moment earlier) and the test read it back. Every other
    thread sees no docker, and with no client discovery never caches."""
    cli = FakeClient(cs)
    me = threading.current_thread()
    monkeypatch.setattr(backends, "_docker_client",
                        lambda: cli if threading.current_thread() is me else None)
    return cli


def test_image_name_comes_from_the_config(monkeypatch):
    c = FakeContainer("llama-cpu", "ghcr.io/ggml-org/llama.cpp:server")
    _with(monkeypatch, [c])
    got = backends.discover_llama_containers()
    assert got[0]["image"] == "ghcr.io/ggml-org/llama.cpp:server" and c.image.inspected == 0


def test_a_bare_image_id_falls_back_to_the_tags(monkeypatch):
    c = FakeContainer("x", "sha256:deadbeef")
    assert backends._image_name(c) == "from-the-image:tag" and c.image.inspected == 1


def test_reused_within_the_ttl_and_copies_are_handed_out(monkeypatch):
    # The fake first, then the TTL, then an empty cache: in that order no other thread can
    # have cached anything this test will read.
    cli = _with(monkeypatch, [FakeContainer("llama-cpu", "ghcr.io/ggml-org/llama.cpp:server")])
    monkeypatch.setattr(backends, "_DISCOVER_TTL_S", 60.0)
    backends.invalidate_discovery()
    a = backends.discover_llama_containers()
    a[0]["name"] = "changed by a caller"
    b = backends.discover_llama_containers()
    assert cli.calls == 1 and b[0]["name"] == "llama-cpu"
    backends.invalidate_discovery()
    backends.discover_llama_containers()
    assert cli.calls == 2
