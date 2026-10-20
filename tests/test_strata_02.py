"""Strata 0.2: an API key on every endpoint but /health, a new data layout, settings in the pin file.

Upstream's 0.2 image runs `strata launch --access lan`, which requires a key, keeps its run
configs in /data/install and its packs in /data/Strata-data/packs, and rewrites the run config's
args on every start - so the tunables move to the pin file (PARALLEL, STRATA_SET), which
entrypoint-0.2.sh reads on every start.
"""
from __future__ import annotations

import asyncio
import functools
import json

import httpx
import pytest

from app import router_api as ra
from app import strata as strata_engine
from app import strata_models, updates
from app.services import backends


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ the key

class TestApiKey:
    def test_from_the_container_environment(self):
        attrs = {"Config": {"Env": ["PATH=/bin", "API_KEY=k" * 1 + "ey123", "PORT=8080"]}}
        assert backends.api_key_of(attrs) == "key123"
        assert backends.api_key_of({"Config": {"Env": ["STRATA_API_KEY=s"]}}) == "s"
        assert backends.api_key_of({"Config": {"Env": ["PATH=/bin"]}}) == ""
        assert backends.api_key_of(None) == ""

    def test_headers_only_for_a_backend_with_a_key(self, monkeypatch):
        monkeypatch.setattr(backends, "_key_map", lambda: {"strata": "abc", "llama-cpu": ""})
        assert backends.auth_headers("strata") == {"Authorization": "Bearer abc"}
        assert backends.auth_headers("llama-cpu") == {}
        assert backends.auth_headers("maya") == {}

    def test_router_carries_the_key_on_the_backend(self, monkeypatch):
        class C:
            name, status = "strata", "running"
            attrs = {"Config": {"Image": "ghcr.io/niko1221/strata:0.2.1-rocm-gfx120x",
                                "Labels": {"ai-lab.engine": "strata"}, "Env": ["API_KEY=sekret"]},
                     "NetworkSettings": {"Ports": {"8080/tcp": [{"HostPort": "8085"}]}}}

        class Cli:
            class containers:
                @staticmethod
                def list(all=False):
                    return [C()]
        monkeypatch.setattr(ra, "docker_client", lambda: Cli())
        assert ra._backends()[0]["headers"] == {"Authorization": "Bearer sekret"}

    def test_router_sends_it_upstream(self, monkeypatch, tmp_path):
        from app import db
        monkeypatch.setattr(db, "_DB_PATH", tmp_path / "r.db")
        db.init()
        ra._init_db()
        seen = []

        def h(req):
            seen.append(req.headers.get("authorization"))
            return httpx.Response(200, json={"choices": [{"message": {"content": "391"}}], "usage": {}})
        monkeypatch.setattr(httpx, "AsyncClient",
                            functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(h)))
        b = {"name": "strata", "engine": "strata", "url": "http://strata:8080", "models": ["q"],
             "ready": True, "running": True, "headers": {"Authorization": "Bearer sekret"}}

        async def resolve(asked):
            return b, "q"
        monkeypatch.setattr(ra, "resolve", resolve)

        class Req:
            headers = {}
            client = None

            async def json(self):
                return {"model": "strata", "messages": [{"role": "user", "content": "x"}]}
        r = run(ra._forward(Req(), "/v1/chat/completions", "openai"))
        assert r.status_code == 200 and seen == ["Bearer sekret"]


# ------------------------------------------------------------------ the layout

class TestLayout:
    def test_detected_from_what_is_on_disk(self, tmp_path):
        old, new = tmp_path / "data", tmp_path / "data2"
        (old / "config").mkdir(parents=True)
        (new / "install").mkdir(parents=True)
        (new / "Strata-data" / "packs").mkdir(parents=True)
        assert strata_engine.layout(old) == "0.1" and strata_engine.layout(new) == "0.2"
        assert strata_engine.configs_dir(old) == old / "config"
        assert strata_engine.configs_dir(new) == new / "install"
        assert strata_engine.packs_dir(new) == new / "Strata-data" / "packs"

    def test_data_dir_maps_the_engine_mount_through_ours(self):
        attrs = {"Mounts": [{"Destination": "/data", "Source": "/home/jason/ai-lab/strata/data2"}]}
        own = {"Mounts": [{"Destination": "/strata", "Source": "/home/jason/ai-lab/strata"},
                          {"Destination": "/models", "Source": "/home/jason/ai-lab/models"}]}
        assert str(strata_engine.data_dir(attrs, own)) == "/strata/data2"

    def test_installed_reads_the_02_layout(self, tmp_path):
        root = tmp_path / "data2"
        (root / "install").mkdir(parents=True)
        pack = root / "Strata-data" / "packs" / "unsloth-ud-iq4_xs"
        pack.mkdir(parents=True)
        (pack / "index.txt").write_text("x" * 10)
        (root / "install" / "strata-unsloth-ud-iq4_xs.json").write_text(json.dumps(
            {"args": ["--pack", "/data/Strata-data/packs/unsloth-ud-iq4_xs"]}))
        got = strata_models.installed(str(root))
        assert list(got) == ["unsloth-ud-iq4_xs"]
        assert got["unsloth-ud-iq4_xs"]["bytes"] == 10 and got["unsloth-ud-iq4_xs"]["missing"] == []

    def test_config_tag_falls_back_to_the_pin_file(self, tmp_path, monkeypatch):
        env = tmp_path / "strata.env"
        env.write_text("GPUS=0,1\nFAMILY=unsloth\nMODEL=UD-IQ4_XS\n")
        monkeypatch.setattr(strata_engine, "ENV_PATH", env)
        assert strata_engine.config_tag({"Config": {"Env": ["API_KEY=x"]}}) == "unsloth-ud-iq4_xs"


# ------------------------------------------------------------------ the settings

PIN = """# the GPU pin
GPUS=0,1
GPU=
FAMILY=unsloth
MODEL=UD-IQ4_XS
GGUF_DIR=
# carried over from 0.1.45
PARALLEL=4
STRATA_SET="prefill=auto:32768 conversation-cache-mib=8192 conversation-cache-slots=8 ple-io=direct"
"""


class TestPinFileSettings:
    @pytest.fixture
    def pin(self, tmp_path, monkeypatch):
        env = tmp_path / "strata.env"
        env.write_text(PIN)
        root = tmp_path / "data2"
        (root / "install").mkdir(parents=True)
        monkeypatch.setattr(strata_engine, "ENV_PATH", env)
        return env, str(root)

    def test_read(self, pin):
        env, _ = pin
        got = strata_models.env_settings(str(env))
        assert got["batch"] == "4" and got["conversation-cache-mib"] == "8192"
        assert got["prefill"] == "auto:32768" and got["ple-io"] == "direct"

    def test_set_tunable_writes_the_pin_file_on_02(self, pin):
        env, root = pin
        ok, msg = strata_models.set_tunable("unsloth-ud-iq4_xs", "conversation-cache-slots", "16",
                                            data_dir=root)
        assert ok and "next restart" in msg
        text = env.read_text()
        assert 'STRATA_SET="prefill=auto:32768 conversation-cache-mib=8192 ple-io=direct ' \
               'conversation-cache-slots=16"' in text
        assert text.count("STRATA_SET=") == 1 and "# the GPU pin" in text and "GPUS=0,1" in text

    def test_batch_is_parallel(self, pin):
        env, root = pin
        assert strata_models.set_tunable("unsloth-ud-iq4_xs", "batch", "2", data_dir=root)[0]
        text = env.read_text()
        assert "PARALLEL=2" in text and "PARALLEL=4" not in text

    def test_engine_report_still_wins(self, pin):
        _, root = pin
        st = strata_models.tuning_state("unsloth-ud-iq4_xs", {"engine": {"batch_slots": 1}}, root)
        assert st["batch"] == "1" and st["conversation-cache-mib"] == "8192"

    def test_01_layout_still_writes_the_run_config(self, tmp_path, monkeypatch):
        from app import db
        monkeypatch.setattr(db, "_DB_PATH", tmp_path / "t.db")    # the 0.1 path records it in the db
        db.init()
        root = tmp_path / "data"
        (root / "config").mkdir(parents=True)
        cfg = root / "config" / "strata-x.json"
        cfg.write_text(json.dumps({"args": ["--batch", "4"]}))
        monkeypatch.setattr(strata_models, "_drop_staged", lambda tag, container: None)
        assert strata_models.set_tunable("x", "batch", "2", data_dir=str(root))[0]
        assert json.loads(cfg.read_text())["args"] == ["--batch", "2"]


# ------------------------------------------------------------------ the update check

class TestImageVersion:
    def test_running_image_parses_upstream_tags(self):
        class C:
            def __init__(self, ref):
                self.attrs = {"Config": {"Image": ref}}

        class D:
            def __init__(self, ref):
                outer = self
                self._c = C(ref)

                class Cs:
                    def get(self, name):
                        return outer._c
                self.containers = Cs()
        assert updates.running_image(D("ghcr.io/niko1221/strata:0.2.1-rocm-gfx120x"), "strata") == \
            ("ghcr.io/niko1221/strata", "0.2.1", "rocm-gfx120x")
        assert updates.running_image(D("ai-lab/strata:rocm7"), "strata") == ("", "", "")

    def test_latest_image_for_the_same_flavor(self):
        tags = ["0.2.1-rocm-gfx120x", "0.2.2-cuda13", "latest-rocm-gfx120x", "0.2.0-rocm-gfx120x",
                "0.2.10-rocm-gfx120x"]

        def h(req):
            if req.url.host == "ghcr.io" and req.url.path == "/token":
                return httpx.Response(200, json={"token": "t"})
            return httpx.Response(200, json={"tags": tags})
        c = httpx.Client(transport=httpx.MockTransport(h))
        assert updates.latest_image_version(c, "ghcr.io/niko1221/strata", "rocm-gfx120x") == "0.2.10"
