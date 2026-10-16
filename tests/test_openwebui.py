from __future__ import annotations

from types import SimpleNamespace as NS

from app import services as s

OWUI_TAG = "ghcr.io/open-webui/open-webui:latest"
TERMINAL_TAG = "ghcr.io/open-webui/open-terminal:latest"


class FakeC:
    def __init__(self, name, tags, status="running", db_ok=True, output=b""):
        self.name = name
        self.status = status
        self.image = NS(tags=tags)
        self._db_ok = db_ok
        self._output = output

    def exec_run(self, cmd):
        if cmd[0] == "test":
            return NS(exit_code=0 if self._db_ok else 1, output=b"")
        return NS(exit_code=0 if self._db_ok else 1, output=self._output)


class FakeClient:
    def __init__(self, containers):
        self.containers = NS(list=lambda all=False: containers)


class TestFindOpenWebui:
    def test_exact_container_name_wins(self):
        right = FakeC("open-webui", [OWUI_TAG])
        other = FakeC("webui2", [OWUI_TAG])
        assert s._find_open_webui(FakeClient([other, right])) is right

    def test_org_sibling_never_matches(self):
        terminal = FakeC("open-terminal", [TERMINAL_TAG])
        owui = FakeC("owui", [OWUI_TAG])
        assert s._find_open_webui(FakeClient([terminal, owui])) is owui

    def test_db_verification_beats_name_match(self):
        wrong = FakeC("open-webui", [OWUI_TAG], db_ok=False)
        right = FakeC("webui2", [OWUI_TAG], db_ok=True)
        assert s._find_open_webui(FakeClient([wrong, right])) is right

    def test_unverified_fallback_guess(self):
        wrong = FakeC("open-webui", [OWUI_TAG], db_ok=False)
        assert s._find_open_webui(FakeClient([wrong])) is wrong

    def test_running_preferred_over_stopped(self):
        stopped = FakeC("open-webui", [OWUI_TAG], status="exited")
        running = FakeC("openwebui", [OWUI_TAG], status="running")
        assert s._find_open_webui(FakeClient([stopped, running])) is running

    def test_no_containers(self):
        assert s._find_open_webui(FakeClient([])) is None


class TestPersistedUrls:
    def test_json_output_parsed(self):
        ow = FakeC("ow", [OWUI_TAG], output=b'["http://a:8080", "http://b:8080"]')
        assert s._openwebui_persisted_urls(ow) == ["http://a:8080", "http://b:8080"]

    def test_exec_failure_and_garbage_degrade_to_empty(self):
        assert s._openwebui_persisted_urls(
            FakeC("ow", [OWUI_TAG], db_ok=False)) == []
        assert s._openwebui_persisted_urls(
            FakeC("ow", [OWUI_TAG], output=b"not json")) == []
