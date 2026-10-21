"""The Maya tab: installed models, and the small set of settings it writes into Maya's config.

The config is read by Maya at start, so the writer is held to Strata's run-config rules: change
only the named keys, replace atomically, keep a backup, refuse anything not on the list.
"""
from __future__ import annotations

import json

import pytest

from app import maya as maya_engine
from tests.test_routes_smoke import web  # noqa: F401 - the fixture: real app, empty box

CFG = "maya-maya-s-v2-iq2_xxs-hip.json"


@pytest.fixture
def conf(tmp_path, monkeypatch):
    d = tmp_path / "conf"
    d.mkdir()
    (d / CFG).write_text(json.dumps({
        "exe": "/opt/maya/build-hip/maya",
        "args": ["--glm-pack", "/data/models/x/pack", "--max-context", "32768",
                 "--expert-cache", "auto"],
        "model_name": "glm-5.3-flash",
        "env": {"MAYA_GLM_RESERVE_MB": "3072"},
        "installer": {"quant": "Maya-S-v2-IQ2_XXS"},
    }, indent=1))
    (d / (CFG + ".shared-settings.json")).write_text('{"model_name": "not-a-model"}')
    monkeypatch.setattr(maya_engine, "CONF_DIR", d)
    monkeypatch.setattr(maya_engine, "_cache", {})
    return d


def _cfg(conf):
    return json.loads((conf / CFG).read_text())


class TestReading:
    def test_mount_is_preferred_and_settings_files_skipped(self, conf):
        rows = maya_engine.installed(client=None)       # no docker needed with the mount
        assert rows == [{"model": "glm-5.3-flash", "quant": "Maya-S-v2-IQ2_XXS",
                         "file": CFG, "context": "32768"}]

    def test_values_from_config(self, conf):
        vals = maya_engine.config_values(_cfg(conf))
        assert vals == {"context": "32768", "parallel": "", "slots": "", "cpu-lane": ""}

    def test_engine_report_wins(self, conf):
        # Between a change and the restart, the engine still runs the old value; say that.
        st = maya_engine.tuning_state(_cfg(conf), {"engine": {"max_context": 65536}})
        assert st["context"] == "65536"


class TestWriting:
    def test_arg_replaced_in_place(self, conf):
        ok, msg = maya_engine.set_tunable("context", "131072", CFG)
        assert ok and "next start" in msg
        args = _cfg(conf)["args"]
        assert args.count("--max-context") == 1 and args[args.index("--max-context") + 1] == "131072"
        assert args[:2] == ["--glm-pack", "/data/models/x/pack"]     # untouched

    def test_arg_added_when_absent(self, conf):
        assert maya_engine.set_tunable("parallel", "4", CFG)[0]
        assert _cfg(conf)["args"][-2:] == ["--parallel", "4"]

    def test_arg_default_removes_the_flag(self, conf):
        maya_engine.set_tunable("parallel", "4", CFG)
        assert maya_engine.set_tunable("parallel", "", CFG)[0]
        assert "--parallel" not in _cfg(conf)["args"]
        assert _cfg(conf)["args"][-2:] == ["--expert-cache", "auto"]

    def test_absent_flag_reads_as_the_default_choice(self, web, conf, monkeypatch):
        # A config with no --parallel is Maya's default, not a hand edit.
        monkeypatch.setattr(maya_engine, "run_config_name", lambda client, container="maya": CFG)
        r = web.get("/maya")
        assert "(set by hand)" not in r.text and '<option value="" selected>1 (default)' in r.text

    def test_env_pair_written_and_cleared(self, conf):
        assert maya_engine.set_tunable("cpu-lane", "4,10", CFG)[0]
        env = _cfg(conf)["env"]
        assert (env["MAYA_GLM_CPU_LANE0"], env["MAYA_GLM_CPU_LANE1"]) == ("4", "10")
        assert env["MAYA_GLM_RESERVE_MB"] == "3072"                  # setup's keys survive
        assert maya_engine.set_tunable("cpu-lane", "", CFG)[0]       # back to the default
        env = _cfg(conf)["env"]
        assert "MAYA_GLM_CPU_LANE0" not in env and "MAYA_GLM_CPU_LANE1" not in env

    def test_backup_kept(self, conf):
        maya_engine.set_tunable("slots", "8", CFG)
        backups = list(conf.glob(CFG + ".bak-*"))
        assert len(backups) == 1 and '"MAYA_GLM_SLOTS"' not in backups[0].read_text()

    @pytest.mark.parametrize("name,value,why", [
        ("context", "999999", "not an allowed value"),
        ("nope", "1", "not a Maya setting"),
    ])
    def test_refusals(self, conf, name, value, why):
        before = (conf / CFG).read_text()
        ok, msg = maya_engine.set_tunable(name, value, CFG)
        assert not ok and why in msg and (conf / CFG).read_text() == before

    @pytest.mark.parametrize("config", ["../evil.json", "other.json", CFG + ".shared-settings.json"])
    def test_only_model_configs_in_the_dir(self, conf, config):
        ok, msg = maya_engine.set_tunable("slots", "8", config)
        assert not ok and "not a Maya config" in msg

    def test_no_mount_is_explained(self, tmp_path, monkeypatch):
        monkeypatch.setattr(maya_engine, "CONF_DIR", tmp_path / "missing")
        ok, msg = maya_engine.set_tunable("slots", "8", CFG)
        assert not ok and "not mounted" in msg


class TestRunConfig:
    def test_from_the_container_command(self):
        class C:
            attrs = {"Config": {"Cmd": ["/opt/maya/.venv/bin/python", "/opt/maya/maya.py",
                                        "--config", f"/data/conf/{CFG}", "--port", "8080"]}}

        class Cli:
            class containers:
                @staticmethod
                def get(name):
                    return C()
        assert maya_engine.run_config_name(Cli()) == CFG


class TestPage:
    def test_renders_on_an_empty_box(self, web):
        r = web.get("/maya")
        assert r.status_code == 200 and "no maya container" in r.text

    def test_renders_settings_for_the_run_config(self, web, conf, monkeypatch):
        monkeypatch.setattr(maya_engine, "run_config_name", lambda client, container="maya": CFG)
        r = web.get("/maya")
        assert r.status_code == 200
        for spec in maya_engine.TUNABLES.values():
            assert spec["label"] in r.text
        assert "runs next" in r.text and "glm-5.3-flash" in r.text
        assert '<option value="32768" selected' in r.text

    def test_post_writes_and_answers(self, web, conf):
        r = web.post("/maya/tuning", data={"name": "slots", "value": "16", "config": CFG})
        assert r.status_code == 200 and "saved" in r.text and "Failed" not in r.text
        assert _cfg(conf)["env"]["MAYA_GLM_SLOTS"] == "16"

    def test_nav_has_maya(self, web):
        assert 'href="/maya"' in web.get("/").text


def test_dashboard_port_comes_from_the_published_binding():
    from app.routers.maya import host_port
    running = {"NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8087"}]}}}
    stopped = {"NetworkSettings": {"Ports": {}}, "HostConfig": {"PortBindings": {"8080/tcp": [{"HostPort": "9001"}]}}}
    assert host_port(running) == "8087" and host_port(stopped) == "9001"
    assert host_port({}) == "" and host_port(None) == ""
