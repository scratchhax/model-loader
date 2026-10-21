"""Review batch 1: settings forms keep their control, honest failure text, Maya's row is display-only,
strata.env selection writes are atomic, no silent fall back to 0.1's data dir, auto-switch waits for a
start it did not make, and a CPU-only llama backend is not routed the GPU presets."""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from app import db, ini
from app import router_api as ra
from app import fileio
from app import strata as strata_engine
from app import strata_models
from app.web import _blocked_chip, _form_note
from tests.test_routes_smoke import web  # noqa: F401 - the fixture: real app, empty box


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ item 1: forms keep their control

class TestSettingsFormsStay:
    def test_note_says_saved_or_why_not(self):
        assert "saved" in _form_note(True, "x").body.decode()
        bad = _form_note(False, "cannot write <f>").body.decode()
        assert "Failed: cannot write &lt;f&gt;" in bad

    def test_failure_chip_is_not_always_llama(self):
        assert "llama.cpp" in _blocked_chip("r").body.decode()
        assert "llama.cpp" not in _blocked_chip("r", "Can&rsquo;t install").body.decode()

    def test_router_forms_target_their_note(self, web):
        r = web.get("/router")
        assert r.status_code == 200
        assert 'hx-post="/router/default" hx-target="find .form-note"' in r.text
        assert r.text.count('class="form-note"') >= 3

    def test_default_post_answers_a_note_not_a_chip(self, web):
        r = web.post("/router/default", data={"engine": "llama", "model": "m"})
        assert r.status_code == 200 and "saved" in r.text and "<form" not in r.text
        assert ra.default_for("llama") == "m"

    def test_recording_flips_in_place(self, web):
        r = web.post("/router/recording", data={"on": "0"})
        assert "<form" in r.text and "recording off" in r.text and "turn on" in r.text
        assert db.get_setting("router.record", "1") == "0"
        r = web.post("/router/recording", data={"on": "1"})
        assert "recording on" in r.text and "turn off" in r.text


# ------------------------------------------------------------------ item 3: Maya's default is not set here

def test_maya_default_is_refused_and_not_stored(web):
    r = web.post("/router/default", data={"engine": "maya", "model": "glm-5.3-flash"})
    assert "Failed" in r.text and "compose" in r.text
    assert ra.default_for("maya") == ""


# ------------------------------------------------------------------ item 4: atomic selection write

def test_selection_write_is_atomic_and_keeps_the_mode(tmp_path, monkeypatch):
    env = tmp_path / "strata.env"
    env.write_text("# why\nGPU=1\nFAMILY=qwen\nMODEL=IQ3_S\n", encoding="utf-8")
    os.chmod(env, 0o640)
    used = []
    real = fileio.atomic_replace
    monkeypatch.setattr(fileio, "atomic_replace", lambda p, d: (used.append(p), real(p, d)))
    ok, _ = strata_models.set_selection(str(env), "swift", "IQ3_S", "")
    assert ok and used == [env]
    text = env.read_text(encoding="utf-8")
    assert "# why" in text and "GPU=1" in text and "FAMILY=swift" in text and "FAMILY=qwen" not in text
    if os.name != "nt":
        assert env.stat().st_mode & 0o777 == 0o640


# ------------------------------------------------------------------ item 6: no silent 0.1 fallback

class TestDataDirGuess:
    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch):
        monkeypatch.setattr(strata_engine, "_last_known_dir", None)
        monkeypatch.setattr(strata_engine, "_live_dir", None)

    def test_the_one_02_layout_dir_wins(self, tmp_path):
        (tmp_path / "data" / "config").mkdir(parents=True)          # 0.1
        (tmp_path / "data2" / "install").mkdir(parents=True)        # 0.2
        assert strata_engine._guess_data_dir(tmp_path) == tmp_path / "data2"

    def test_ambiguous_falls_back_to_data(self, tmp_path):
        (tmp_path / "a" / "install").mkdir(parents=True)
        (tmp_path / "b" / "install").mkdir(parents=True)
        assert strata_engine._guess_data_dir(tmp_path) == tmp_path / "data"

    def test_last_known_beats_a_guess(self, tmp_path, monkeypatch):
        monkeypatch.setattr(strata_engine, "_last_known_dir", tmp_path / "seen")
        assert strata_engine._guess_data_dir(tmp_path) == tmp_path / "seen"


# ------------------------------------------------------------------ item 2: a start Model Loader did not make

def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(epoch)) + ".123456789Z"


class TestStartingBackends:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        monkeypatch.setattr(ra, "_seen_ready", {})

    def test_started_epoch_parses_dockers_format(self):
        assert abs(ra._started_epoch(_iso(1_800_000_000)) - 1_800_000_000) < 1

    def test_fresh_and_not_serving_is_starting(self):
        now = time.time()
        b = {"name": "strata", "engine": "strata", "ready": False, "started_at": _iso(now - 30)}
        assert ra.starting_backends([b], now) == [b]

    def test_parked_is_not_starting(self):
        now = time.time()
        b = {"name": "strata", "engine": "strata", "ready": False, "started_at": _iso(now - 30)}
        ra._seen_ready["strata"] = b["started_at"]        # it served since this start
        assert ra.starting_backends([b], now) == []

    def test_old_start_is_not_starting(self):
        now = time.time()
        b = {"name": "strata", "engine": "strata", "ready": False, "started_at": _iso(now - 3600)}
        assert ra.starting_backends([b], now) == []

    def test_autoswitch_waits_for_it(self, monkeypatch, tmp_path):
        monkeypatch.setattr(db, "_DB_PATH", tmp_path / "r.db")
        db.init()
        db.set_setting("router.autoswitch.maya", "1")
        now = time.time()
        s = {"name": "strata", "engine": "strata", "ready": False, "running": True,
             "started_at": _iso(now - 20), "models": []}

        async def lm():
            return [s], {}

        async def idle():
            return {}
        monkeypatch.setattr(ra, "_live_map", lm)
        monkeypatch.setattr(ra, "switch_state", idle)
        monkeypatch.setattr(ra, "_last_switch_at", lambda: 0.0)
        ok, why = run(ra.may_autoswitch("maya"))
        assert not ok and "strata is starting up outside Model Loader" in why


# ------------------------------------------------------------------ item 5: CPU llama is not sent GPU presets

INI = """[voice-cpu]
ngl = 0
[big-gpu]
ngl = 999
split-mode = tensor
[placed]
main-gpu = 1
"""


def test_gpu_sections_from_the_ini():
    cp = ini._new_parser()
    cp.read_string(INI)
    assert ini.gpu_sections(cp) == {"big-gpu", "placed"}


def test_cpu_only_is_the_image_and_devices():
    assert ra._cpu_only({"Config": {"Image": "ghcr.io/ggml-org/llama.cpp:server"}, "HostConfig": {}})
    assert not ra._cpu_only({"Config": {"Image": "ghcr.io/ggml-org/llama.cpp:server-rocm"}})
    assert not ra._cpu_only({"Config": {"Image": "x:server"},
                             "HostConfig": {"Devices": [{"PathOnHost": "/dev/kfd"}]}})


class TestCpuBackendRouting:
    @pytest.fixture(autouse=True)
    def _box(self, monkeypatch, tmp_path):
        monkeypatch.setattr(db, "_DB_PATH", tmp_path / "r.db")
        db.init()
        cp = ini._new_parser()
        cp.read_string(INI)
        monkeypatch.setattr(ini, "read_ini", lambda: cp)
        monkeypatch.setattr(ra, "_seen_ready", {})
        monkeypatch.setattr(ra, "_backends", lambda include_stopped=False: [
            {"name": "llama-cpu", "engine": "llama", "running": True, "cpu_only": True,
             "url": "http://llama-cpu:8080", "models": [], "headers": {}, "started_at": ""}])

        async def models_of(client, b):
            return ["voice-cpu", "big-gpu", "placed"]

        async def ready(client, b):
            return True
        monkeypatch.setattr(ra, "_models_of", models_of)
        monkeypatch.setattr(ra, "_ready", ready)

    def test_listed_but_not_routed(self):
        bs, by_model = run(ra._live_map())
        assert set(by_model) == {"voice-cpu"}
        ids = [d["id"] for d in run(ra.list_models())["data"]]
        assert "big-gpu" in ids                       # still in the dropdown

    def test_asking_for_it_says_why(self):
        with pytest.raises(ra.Unresolved, match="set up for the GPUs.*CPU only"):
            run(ra.resolve("big-gpu"))
        b, m = run(ra.resolve("voice-cpu"))
        assert b["name"] == "llama-cpu" and m == "voice-cpu"

    def test_a_gpu_default_is_not_sent_to_cpu(self):
        db.set_setting("router.default.llama", "big-gpu")
        with pytest.raises(ra.Unresolved, match="CPU only"):
            run(ra.resolve("llama"))
