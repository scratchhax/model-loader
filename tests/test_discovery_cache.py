"""Container discovery is reused briefly and reads image names from the container's own config
(review item 21). One GPU-strip build used to make 211 docker API calls and took 2-3 s to answer
a 1 s poll: five discoveries, each listing and inspecting every container, plus an image
inspect per container for its tag."""
from __future__ import annotations

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
    cli = FakeClient(cs)
    monkeypatch.setattr(backends, "_docker_client", lambda: cli)
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
    monkeypatch.setattr(backends, "_DISCOVER_TTL_S", 60.0)
    cli = _with(monkeypatch, [FakeContainer("llama-cpu", "ghcr.io/ggml-org/llama.cpp:server")])
    a = backends.discover_llama_containers()
    a[0]["name"] = "changed by a caller"
    b = backends.discover_llama_containers()
    assert cli.calls == 1 and b[0]["name"] == "llama-cpu"
    backends.invalidate_discovery()
    backends.discover_llama_containers()
    assert cli.calls == 2
