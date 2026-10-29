"""DeepSeek's container gets Strata's Load / Unload, and the Qwen install's controls stay its own."""
from __future__ import annotations

from app import services
from app.routers import containers
from tests.test_routes_smoke import web  # noqa: F401  (the fixture)


def test_load_and_unload_accept_a_deepseek_backend(monkeypatch, web):  # noqa: F811
    calls = []
    monkeypatch.setattr(services, "engine_for", lambda name: "deepseek")
    monkeypatch.setattr(services, "strata_load", lambda name: (calls.append(("load", name)), (True, "ok"))[1])
    monkeypatch.setattr(services, "strata_unload", lambda name: (calls.append(("unload", name)), (True, "ok"))[1])
    for verb in ("load", "unload"):
        r = web.post(f"/containers/strata-ds/strata-{verb}")
        assert r.status_code == 200 and "not a Strata backend" not in r.text
    assert calls == [("load", "strata-ds"), ("unload", "strata-ds")]


def test_load_still_refuses_a_llama_backend(monkeypatch, web):  # noqa: F811
    monkeypatch.setattr(services, "engine_for", lambda name: "llama")
    monkeypatch.setattr(services, "strata_load", lambda name: (_ for _ in ()).throw(AssertionError("called")))
    assert "not a Strata backend" in web.post("/containers/llama-9700x2/strata-load").text


def test_the_qwen_pin_stays_strata_only(monkeypatch, web):  # noqa: F811
    # The card pin writes strata.env, which only the Qwen install reads.
    monkeypatch.setattr(services, "engine_for", lambda name: "deepseek")
    monkeypatch.setattr(containers.strata_engine, "set_pin",
                        lambda card: (_ for _ in ()).throw(AssertionError("pin written")))
    assert "not a Strata backend" in web.post("/containers/strata-ds/strata-pin", data={"card": "0"}).text
