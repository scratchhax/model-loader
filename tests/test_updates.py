"""The engine update check: what runs here against what upstream has, without touching either.

GitHub and ghcr are faked with httpx.MockTransport and docker with small stand-ins, so this never
reaches the network.
"""
from __future__ import annotations

import json
import time

import httpx
import pytest

from app import db, updates
from tests.test_routes_smoke import web  # noqa: F401 - the fixture: real app, empty box


@pytest.fixture
def udb(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "u.db")
    db.init()
    yield


class Img:
    def __init__(self, tags, digests=()):
        self.tags = list(tags)
        self.attrs = {"RepoDigests": list(digests)}


class Ctr:
    def __init__(self, name, image, status="exited"):
        self.name, self.status = name, status
        self.attrs = {"Config": {"Image": image}, "NetworkSettings": {"Ports": {}}}


class Docker:
    def __init__(self, images, containers=()):
        outer = self

        class Images:
            def get(self, ref):
                if ref not in outer._images:
                    raise KeyError(ref)
                return outer._images[ref]

        class Containers:
            def list(self, all=False):
                return list(outer._containers)
        self._images, self._containers = images, list(containers)
        self.images, self.containers = Images(), Containers()


def _http(routes):
    def h(req):
        for (method, prefix), resp in routes.items():
            if req.method == method and str(req.url).startswith(prefix):
                return resp(req) if callable(resp) else resp
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(h))


GH = "https://api.github.com/repos/"


class TestVersions:
    @pytest.mark.parametrize("latest,current,want", [
        ("1.1.18", "1.1.16", True), ("1.1.16", "1.1.16", False), ("0.1.44.1", "0.1.44", True),
        ("0.1.9", "0.1.10", False), ("", "1.0", None), ("b1", "x", None)])
    def test_newer(self, latest, current, want):
        assert updates.newer(latest, current) is want

    def test_latest_tag_is_the_highest_version_not_the_first(self):
        tags = [{"name": "v1.1.9"}, {"name": "v1.1.18"}, {"name": "nightly"}, {"name": "v1.1.17"}]
        c = _http({("GET", GH + "mw00/project-maya/tags"): httpx.Response(200, json=tags)})
        assert updates.latest_tag(c, "mw00/project-maya") == "1.1.18"

    def test_local_version_from_the_tag_sharing_the_live_image(self):
        d = Docker({"ai-lab/maya:rocm7": Img(["ai-lab/maya:rocm7", "ai-lab/maya:rocm7-v1.1.18"])})
        assert updates.local_version(d, "ai-lab/maya", "rocm7") == "1.1.18"
        assert updates.local_version(Docker({}), "ai-lab/maya", "rocm7") == ""


class TestCheck:
    def _docker(self):
        return Docker(
            {"ai-lab/strata:rocm7": Img(["ai-lab/strata:rocm7", "ai-lab/strata:rocm7-v0.1.44.1"]),
             "ai-lab/maya:rocm7": Img(["ai-lab/maya:rocm7", "ai-lab/maya:rocm7-v1.1.18"]),
             "ghcr.io/ggml-org/llama.cpp:server": Img([], ["ghcr.io/ggml-org/llama.cpp@sha256:old"]),
             "ghcr.io/ggml-org/llama.cpp:server-rocm": Img([], ["ghcr.io/ggml-org/llama.cpp@sha256:same"])},
            [Ctr("llama-cpu", "ghcr.io/ggml-org/llama.cpp:server"),
             Ctr("llama-voice", "ghcr.io/ggml-org/llama.cpp:server"),
             Ctr("llama-9700x2", "ghcr.io/ggml-org/llama.cpp:server-rocm"),
             Ctr("open-webui", "ghcr.io/open-webui/open-webui:v0.12.3")])

    def _http(self):
        def manifest(req):
            tag = str(req.url).rsplit("/", 1)[-1]
            return httpx.Response(200, headers={
                "docker-content-digest": "sha256:new" if tag == "server" else "sha256:same"})
        return _http({
            ("GET", GH + "Niko1221/Strata/tags"): httpx.Response(200, json=[{"name": "v0.1.45"}, {"name": "v0.1.44.1"}]),
            ("GET", GH + "mw00/project-maya/tags"): httpx.Response(200, json=[{"name": "v1.1.18"}]),
            ("GET", GH + "ggml-org/llama.cpp/releases/latest"): httpx.Response(200, json={
                "tag_name": "b11250", "published_at": "2026-10-20T00:00:00Z", "html_url": "https://x/r"}),
            ("GET", "https://ghcr.io/token"): httpx.Response(200, json={"token": "t"}),
            ("HEAD", "https://ghcr.io/v2/"): manifest,
        })

    def test_each_engine(self, monkeypatch):
        monkeypatch.setattr(updates, "_running_report", lambda engine, container: "")
        st = updates.check(self._docker(), self._http())
        rows = {r["engine"]: r for r in st["engines"]}
        assert (rows["strata"]["current"], rows["strata"]["latest"], rows["strata"]["status"]) \
            == ("0.1.44.1", "0.1.45", "update")
        assert "compare/v0.1.44.1...v0.1.45" in rows["strata"]["url"]
        assert (rows["maya"]["current"], rows["maya"]["status"]) == ("1.1.18", "current")
        llama = rows["llama"]
        assert llama["status"] == "update" and llama["latest"] == "b11250"
        detail = {d["tag"]: d for d in llama["detail"]}
        assert detail["server"]["newer"] and not detail["server-rocm"]["newer"]
        assert detail["server"]["containers"] == ["llama-cpu", "llama-voice"]
        assert "open-webui" not in json.dumps(llama["detail"])

    def test_the_running_engine_report_wins(self, monkeypatch):
        # A container started before a retag still runs the old build; say what is loaded.
        monkeypatch.setattr(updates, "_running_report",
                            lambda engine, container: "1.1.16" if engine == "maya" else "")
        rows = {r["engine"]: r for r in updates.check(self._docker(), self._http())["engines"]}
        assert (rows["maya"]["current"], rows["maya"]["status"]) == ("1.1.16", "update")

    def test_network_failure_is_a_row_error_not_a_crash(self, monkeypatch):
        monkeypatch.setattr(updates, "_running_report", lambda engine, container: "")

        def boom(req):
            raise httpx.ConnectError("offline")
        st = updates.check(self._docker(), httpx.Client(transport=httpx.MockTransport(boom)))
        assert all(r["error"] and r["status"] == "unknown" for r in st["engines"])

    def test_builds_behind(self, monkeypatch):
        monkeypatch.setattr(updates, "_running_report", lambda engine, container: "")
        d = self._docker()
        d._containers[0].status = "running"
        monkeypatch.setattr(updates, "_llama_build", lambda c: "b11096-c550d2f60")
        rows = {r["engine"]: r for r in updates.check(d, self._http())["engines"]}
        assert rows["llama"]["current"] == "b11096" and rows["llama"]["behind"] == 154


class TestCaching:
    def test_fresh_cache_is_not_rechecked(self, udb, monkeypatch):
        db.set_setting(updates._KEY, json.dumps({"checked_at": time.time(), "engines": []}))
        monkeypatch.setattr(updates, "check", lambda docker: pytest.fail("must not re-check"))
        assert updates.refresh(None) is False

    def test_stale_cache_checks_in_the_background(self, udb, monkeypatch):
        db.set_setting(updates._KEY, json.dumps({"checked_at": 0, "engines": []}))
        monkeypatch.setattr(updates, "check", lambda docker: {"checked_at": 42.0, "engines": [{"x": 1}]})
        assert updates.refresh(None) is True
        for _ in range(100):
            if not updates.in_progress():
                break
            time.sleep(0.02)
        assert updates.cached() == {"checked_at": 42.0, "engines": [{"x": 1}]}


class TestCard:
    def test_card_renders_cached_rows(self, web, monkeypatch):
        monkeypatch.setattr(updates, "refresh", lambda docker, force=False: False)
        db.set_setting(updates._KEY, json.dumps({"checked_at": time.time(), "engines": [
            {"engine": "maya", "label": "Maya", "current": "1.1.16", "latest": "1.1.18",
             "status": "update", "url": "https://github.com/mw00/project-maya/compare/v1.1.16...v1.1.18",
             "runbook": "~/ai-lab/maya/UPGRADING.md", "error": "", "detail": []}]}))
        r = web.get("/updates")
        assert r.status_code == 200
        assert "update available" in r.text and "1.1.16 → 1.1.18" in r.text
        assert "~/ai-lab/maya/UPGRADING.md" in r.text and "checked just now" in r.text

    def test_check_now_forces(self, web, monkeypatch):
        seen = []
        monkeypatch.setattr(updates, "refresh", lambda docker, force=False: seen.append(force))
        assert web.post("/updates/check").status_code == 200 and seen == [True]

    def test_overview_loads_it_lazily(self, web):
        assert 'hx-get="/updates"' in web.get("/").text
