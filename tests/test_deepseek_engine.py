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


class _Now:
    """threading.Thread stand-in that runs its target at start(), so a watcher is testable."""
    def __init__(self, target, **_k):
        self.target = target

    def start(self):
        self.target()


def test_a_slow_start_is_closed_out_when_it_serves(monkeypatch):
    # DeepSeek re-tuned for 15 min after an engine update (2026-10-30), served, and the banner
    # still said "still loading, slower than usual".
    from app.services import engines
    monkeypatch.setattr(engines.threading, "Thread", _Now)
    monkeypatch.setattr(engines, "_wait_ready", lambda name, timeout=None, quiet=False: (True, ""))
    monkeypatch.setattr(engines, "_switch_state", {"engine": "deepseek", "phase": "slow", "msg": "did not answer", "t0": 123.0})
    engines._watch_slow_start("strata-ds", 123.0)
    assert engines._switch_state["phase"] == "done" and engines._switch_state["t1"] >= 123.0


def test_the_watcher_leaves_a_newer_switch_alone(monkeypatch):
    from app.services import engines
    monkeypatch.setattr(engines.threading, "Thread", _Now)
    monkeypatch.setattr(engines, "_wait_ready", lambda *a, **k: (True, ""))
    monkeypatch.setattr(engines, "_switch_state", {"engine": "strata", "phase": "loading", "t0": 999.0})
    engines._watch_slow_start("strata-ds", 123.0)              # an older switch's watcher
    assert engines._switch_state["phase"] == "loading" and "t1" not in engines._switch_state


def test_the_first_paint_does_not_wait_on_the_engines(monkeypatch, web):  # noqa: F811
    from app.routers import overview

    async def boom(*a, **k):
        raise AssertionError("the first paint probed the engines")
    monkeypatch.setattr(overview, "_hero_context", boom)
    monkeypatch.setattr(overview, "_llm_activity", boom)
    r = web.get("/")
    assert r.status_code == 200 and "reading the cards" in r.text and "reading the engine" in r.text
