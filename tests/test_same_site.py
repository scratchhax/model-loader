"""A mutating request from another site is refused (review item 23).

The app holds the docker socket, so a cross-site form post from any page open on the LAN could stop
engines or rewrite models.ini. Origin (or Referer) naming another host is refused; no header at all
is a script or the router process and is let through.
"""
from __future__ import annotations

from app.main import cross_site
from tests.test_routes_smoke import web  # noqa: F401 - the fixture: real app, empty box


def test_same_host_is_fine():
    assert cross_site({"origin": "http://wheatley:8090", "host": "wheatley:8090"}) == ""
    assert cross_site({"referer": "http://wheatley:8090/router", "host": "wheatley:8090"}) == ""
    assert cross_site({"origin": "https://ml.example", "host": "model-loader:8090",
                       "x-forwarded-host": "ml.example"}) == ""


def test_other_hosts_and_null_are_refused():
    assert "evil.example" in cross_site({"origin": "https://evil.example", "host": "wheatley:8090"})
    assert cross_site({"referer": "http://evil.example/x", "host": "wheatley:8090"})
    assert cross_site({"origin": "null", "host": "wheatley:8090"})
    # a different port is a different site
    assert cross_site({"origin": "http://wheatley:3000", "host": "wheatley:8090"})


def test_no_header_is_a_script():
    assert cross_site({"host": "wheatley:8090"}) == ""


def test_the_app_refuses_a_cross_site_post(web):
    r = web.post("/router/recording", data={"on": "0"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403 and "evil.example" in r.text
    r = web.post("/router/recording", data={"on": "0"}, headers={"Origin": "http://testserver"})
    assert r.status_code == 200


def test_reads_are_not_checked(web):
    assert web.get("/router", headers={"Origin": "https://evil.example"}).status_code == 200
