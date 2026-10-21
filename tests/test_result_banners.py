"""Every result banner comes from web._banner, which escapes what it is given (review item 25).

Routes used to build the banner by hand in three colours, and several put an error message, a
container name or a model's reply into the page unescaped.
"""
from __future__ import annotations

from app import services
from app.web import _banner, _toast
from tests.test_routes_smoke import web  # noqa: F401 - the fixture: real app, empty box


def test_banner_escapes_and_toast_prefixes():
    body = _banner("warn", "<b>x</b>").body.decode()
    assert "&lt;b&gt;x&lt;/b&gt;" in body and "bg-amber-50" in body
    assert "&check; done" in _toast(True, "done").body.decode()
    assert "Sync failed: no" in _toast(False, "no", fail="Sync failed:").body.decode()
    assert ">no</div>" in _toast(False, "no", fail="").body.decode()


def test_a_route_error_is_escaped(web, monkeypatch):
    monkeypatch.setattr(services, "sync_openwebui_endpoints", lambda: (False, "<script>x</script>"))
    r = web.post("/containers/sync-openwebui")
    assert r.status_code == 200
    assert "<script>" not in r.text and "Sync failed: &lt;script&gt;" in r.text


def test_logs_error_is_escaped(web, monkeypatch):
    monkeypatch.setattr(services, "container_logs", lambda name: (False, "<img src=x>"))
    r = web.get("/containers/x/logs")
    assert "<img" not in r.text and "&lt;img" in r.text
