"""Characterization tests for the OpenWebUI write paths in app/services/openwebui.py.

The reconciler's read side is pinned elsewhere; these pin the WRITES: the exec
scripts that go into open-webui's webui.db, the restart policy (PersistentConfig
needs one, the model table does not), and the failure messages the UI shows.
Docker is a fake container recording exec_run/restart calls.
"""
import json
from types import SimpleNamespace

import pytest
from docker.errors import DockerException

from app.services import openwebui as ow


class FakeOW:
    def __init__(self, results=()):
        self._results = list(results)
        self.execs = []
        self.restarts = []

    def exec_run(self, cmd, environment=None):
        self.execs.append((cmd, environment))
        if self._results:
            return self._results.pop(0)
        return SimpleNamespace(exit_code=0, output=b"OK 0")

    def restart(self, timeout=30):
        self.restarts.append(timeout)


def _exec(exit_code=0, output=b"OK 0"):
    return SimpleNamespace(exit_code=exit_code, output=output)


@pytest.fixture
def wired(monkeypatch):
    """Wire the module onto a FakeOW; returns the fake for assertions."""
    state = {"client": object(), "ow": FakeOW()}
    monkeypatch.setattr(ow, "_docker_client", lambda: state["client"])
    monkeypatch.setattr(ow, "_find_open_webui", lambda client: state["ow"])
    return state


PLAN = {"a.m1": True, "a.m2": False, "m3": True}


class TestSyncCapabilities:
    def test_empty_plan_short_circuits(self, wired, monkeypatch):
        monkeypatch.setattr(ow, "openwebui_capability_plan", lambda: {})
        ok, msg = ow.sync_openwebui_capabilities()
        assert ok is False and msg.startswith("nothing to sync")
        assert wired["ow"].execs == []

    def test_docker_unreachable(self, wired, monkeypatch):
        monkeypatch.setattr(ow, "openwebui_capability_plan", lambda: PLAN)
        wired["client"] = None
        assert ow.sync_openwebui_capabilities() == (False, "docker unreachable")

    def test_webui_not_found(self, wired, monkeypatch):
        monkeypatch.setattr(ow, "openwebui_capability_plan", lambda: PLAN)
        wired["ow"] = None
        assert ow.sync_openwebui_capabilities() == (False, "open-webui not found")

    def test_reports_changes_and_no_restart(self, wired, monkeypatch):
        monkeypatch.setattr(ow, "openwebui_capability_plan", lambda: PLAN)
        wired["ow"] = FakeOW([_exec(0, b"OK 2")])
        ok, msg = ow.sync_openwebui_capabilities()
        assert ok is True
        assert msg == "2 model(s) updated (2 of 3 vision-capable)"
        cmd, env = wired["ow"].execs[0]
        assert cmd[0] == "python3" and "INSERT INTO model" in cmd[2]
        assert json.loads(env["PLAN"]) == PLAN
        # The model table is read per request: capability sync must NOT restart.
        assert wired["ow"].restarts == []

    def test_already_in_sync(self, wired, monkeypatch):
        monkeypatch.setattr(ow, "openwebui_capability_plan", lambda: PLAN)
        wired["ow"] = FakeOW([_exec(0, b"OK 0")])
        ok, msg = ow.sync_openwebui_capabilities()
        assert ok is True
        assert msg == "already in sync (2 of 3 vision-capable)"

    def test_exec_failure_reported(self, wired, monkeypatch):
        monkeypatch.setattr(ow, "openwebui_capability_plan", lambda: PLAN)
        wired["ow"] = FakeOW([_exec(1, b"NOUSER")])
        ok, msg = ow.sync_openwebui_capabilities()
        assert ok is False and msg == "capability sync failed: NOUSER"

    def test_exec_exception_reported(self, wired, monkeypatch):
        monkeypatch.setattr(ow, "openwebui_capability_plan", lambda: PLAN)

        class Boom(FakeOW):
            def exec_run(self, cmd, environment=None):
                raise DockerException("daemon down")
        wired["ow"] = Boom()
        ok, msg = ow.sync_openwebui_capabilities()
        assert ok is False and "exec into open-webui failed" in msg


class TestAlignCapabilities:
    def test_empty_plan_short_circuits(self, wired, monkeypatch):
        monkeypatch.setattr(ow, "openwebui_capability_plan", lambda: {})
        ok, msg = ow.align_openwebui_capabilities()
        assert ok is False and msg.startswith("nothing to align")

    def test_parses_counts_and_lists_stale(self, wired, monkeypatch):
        monkeypatch.setattr(ow, "openwebui_capability_plan", lambda: PLAN)
        out = b'OK {"updated": 3, "removed": ["old-model"], "kept": ["base-model"]}'
        wired["ow"] = FakeOW([_exec(0, out)])
        ok, msg = ow.align_openwebui_capabilities()
        assert ok is True
        assert "aligned 3 model(s)" in msg
        assert "removed 1 stale record(s): old-model" in msg
        assert "kept 1 stale record(s) still used as a base model" in msg
        cmd, env = wired["ow"].execs[0]
        assert "DELETE FROM model" in cmd[2]
        assert json.loads(env["PLAN"]) == PLAN

    def test_non_json_output_passes_through(self, wired, monkeypatch):
        monkeypatch.setattr(ow, "openwebui_capability_plan", lambda: PLAN)
        wired["ow"] = FakeOW([_exec(0, b"OK weird")])
        ok, msg = ow.align_openwebui_capabilities()
        assert ok is True and msg == "OK weird"


class TestSetModelFilter:
    def test_saves_and_restarts(self, wired):
        wired["ow"] = FakeOW([_exec(0, b"OK 2")])
        ok, msg = ow.set_openwebui_model_filter("http://u:8080/v1", ["m1", "m2"])
        assert ok is True
        assert msg == "http://u:8080/v1: now offering 2 selected model(s)"
        cmd, env = wired["ow"].execs[0]
        assert env["TARGET_URL"] == "http://u:8080/v1"
        assert json.loads(env["MODEL_IDS"]) == ["m1", "m2"]
        # api_configs is a PersistentConfig: the filter only lands after a boot.
        assert wired["ow"].restarts == [30]

    def test_empty_ids_means_all(self, wired):
        wired["ow"] = FakeOW([_exec(0, b"OK 0")])
        ok, msg = ow.set_openwebui_model_filter("http://u:8080/v1", [])
        assert ok is True and msg == "http://u:8080/v1: now offering all models"

    def test_unknown_connection_no_restart(self, wired):
        wired["ow"] = FakeOW([_exec(0, b"NOTFOUND")])
        ok, msg = ow.set_openwebui_model_filter("http://nope/v1", ["m"])
        assert ok is False and msg == "connection not found in OpenWebUI: http://nope/v1"
        assert wired["ow"].restarts == []

    def test_nonzero_exit(self, wired):
        wired["ow"] = FakeOW([_exec(1, b"traceback")])
        ok, msg = ow.set_openwebui_model_filter("http://u:8080/v1", ["m"])
        assert ok is False and msg.startswith("DB update failed (exit 1)")

    def test_restart_failure_reported(self, wired):
        class Boom(FakeOW):
            def restart(self, timeout=30):
                raise DockerException("no such container")
        wired["ow"] = Boom([_exec(0, b"OK 1")])
        ok, msg = ow.set_openwebui_model_filter("http://u:8080/v1", ["m"])
        assert ok is False and msg.startswith("filter saved but restart failed")


class TestPruneUnknownIds:
    def test_no_webui_returns_empty(self, wired, monkeypatch):
        monkeypatch.setattr(ow, "openwebui_state", lambda: {"found": False})
        assert ow.prune_openwebui_unknown_ids() == []

    def test_prunes_only_dead_ids_with_survivors(self, wired, monkeypatch):
        calls = []

        def fake_set(url, ids):
            calls.append((url, ids))
            return True, ""
        monkeypatch.setattr(ow, "set_openwebui_model_filter", fake_set)
        monkeypatch.setattr(ow, "openwebui_state", lambda: {"found": True, "connections": [
            {"url": "http://a/v1", "unknown_ids": ["gone"], "live_ids": ["m1", "m2"]},
            {"url": "http://b/v1", "unknown_ids": ["gone2"], "live_ids": []},
            {"url": "http://c/v1", "unknown_ids": [], "live_ids": ["m1"]},
        ]})
        assert ow.prune_openwebui_unknown_ids() == ["gone"]
        # b: pruning would empty the whitelist into "offer everything" - skipped.
        assert calls == [("http://a/v1", ["m1", "m2"])]

    def test_failed_write_not_reported_cleaned(self, wired, monkeypatch):
        monkeypatch.setattr(ow, "set_openwebui_model_filter", lambda url, ids: (False, "nope"))
        monkeypatch.setattr(ow, "openwebui_state", lambda: {"found": True, "connections": [
            {"url": "http://a/v1", "unknown_ids": ["gone"], "live_ids": ["m1"]},
        ]})
        assert ow.prune_openwebui_unknown_ids() == []
