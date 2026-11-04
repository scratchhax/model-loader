"""Open WebUI connections to another box are not "stale" or "missing" when they answer.

The Containers page judged every connection by the local docker socket: a compose-style host
with no container here was stale, and a terminal server with no container here was missing. On
wheatley that called glados's router (http://glados:8086/v1) stale, which Reconcile would then
delete, and the GlaDOS terminal server (http://192.168.121.3:8000) missing, while both answered.
Now anything with no container here is asked about from inside Open WebUI before it is judged.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

from docker.errors import NotFound

from app.services import openwebui as ow

TERMINALS = [{"name": "wheatley", "url": "http://open-terminal:8000", "enabled": False},
             {"name": "GlaDOS", "url": "http://192.168.121.3:8000", "enabled": True}]


class C:
    def __init__(self, name, answers=(), terminals=TERMINALS):
        self.name, self.status, self.labels, self.attrs = name, "running", {}, {}
        self.image = NS(tags=[f"img/{name}:latest"], short_id="abc")
        self.short_id = "abc"
        self.answers = set(answers)       # URLs that respond when probed from in here
        self.terminals = terminals
        self.probed: list[list[str]] = []
        self.execs: list[tuple] = []

    def exec_run(self, cmd, environment=None):
        self.execs.append((cmd, environment))
        script = cmd[2] if len(cmd) > 2 else ""
        if environment and "URLS" in environment and "urlopen" in script:
            urls = json.loads(environment["URLS"])
            self.probed.append(urls)
            return NS(exit_code=0, output=json.dumps([u for u in urls if u in self.answers]).encode())
        if "terminal_server.connections" in script:
            return NS(exit_code=0, output=json.dumps(self.terminals).encode())
        return NS(exit_code=0, output=b"added=0 removed=0")

    def restart(self, timeout=30):
        pass


class Client:
    def __init__(self, containers):
        def get(name):
            raise NotFound("no")
        self.containers = NS(list=lambda all=False: containers, get=get)


def wire(monkeypatch, webui, urls, others=("open-terminal", "model-loader")):
    client = Client([webui] + [C(n) for n in others])
    monkeypatch.setattr(ow, "_docker_client", lambda: client)
    monkeypatch.setattr(ow, "_find_open_webui", lambda c: webui)
    monkeypatch.setattr(ow, "_openwebui_persisted_urls", lambda c: list(urls))
    monkeypatch.setattr(ow, "discover_llama_containers", lambda: [])
    monkeypatch.setattr(ow, "_openwebui_api_configs", lambda c: {})
    monkeypatch.setattr(ow.ini, "section_names", lambda: [])
    return client


def test_another_boxs_router_that_answers_is_not_stale(monkeypatch):
    webui = C("open-webui", answers={"http://glados:8086/v1/models"})
    wire(monkeypatch, webui, ["http://model-loader:8086/v1", "http://glados:8086/v1", "http://gone:8080/v1"])
    st = ow.openwebui_state()
    assert st["stale_urls"] == ["http://gone:8080/v1"]
    # only the ones with no container here are asked about, and a local one never is
    assert sorted(webui.probed[0]) == ["http://glados:8086/v1/models", "http://gone:8080/v1/models"]


def test_a_failed_probe_never_calls_anything_stale(monkeypatch):
    webui = C("open-webui")
    webui.exec_run_orig = webui.exec_run

    def broken(cmd, environment=None):
        if environment and "URLS" in environment:
            return NS(exit_code=1, output=b"boom")
        return webui.exec_run_orig(cmd, environment)
    webui.exec_run = broken
    wire(monkeypatch, webui, ["http://glados:8086/v1"])
    assert ow.openwebui_state()["stale_urls"] == []


def test_remote_terminal_server_is_reachable_not_missing(monkeypatch):
    webui = C("open-webui", answers={"http://192.168.121.3:8000/health"})
    wire(monkeypatch, webui, [])
    comps = {c["name"]: c for c in ow.openwebui_state()["companions"]}
    assert comps["GlaDOS"]["status"] == "reachable" and not comps["GlaDOS"]["stale"]
    # the local one is still judged by its container, and its disabled flag is Open WebUI's own
    assert comps["wheatley"]["status"] == "running" and comps["wheatley"]["enabled"] is False


def test_remote_terminal_server_that_does_not_answer_says_so(monkeypatch):
    webui = C("open-webui")
    wire(monkeypatch, webui, [])
    comps = {c["name"]: c for c in ow.openwebui_state()["companions"]}
    assert comps["GlaDOS"]["status"] == "unreachable"


def test_reconcile_keeps_a_reachable_remote_host(monkeypatch):
    webui = C("open-webui", answers={"http://glados:8086/v1/models"})
    wire(monkeypatch, webui, ["http://glados:8086/v1", "http://gone:8080/v1"])
    ok, _ = ow.sync_openwebui_endpoints()
    assert ok
    cmd, env = next((c, e) for c, e in webui.execs if e and "VALID_HOSTS" in e)
    valid = set(json.loads(env["VALID_HOSTS"]))
    assert "glados" in valid and "gone" not in valid

