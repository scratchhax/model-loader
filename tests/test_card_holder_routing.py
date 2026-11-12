"""A backend is an answer only while its engine holds the cards.

2026-11-12: llama-9700x2 was restarted from its container card while Strata held all 64 GB.
Both answered as ready, the router forwarded a CyberStrike request to llama because llama was
"up", and llama died allocating its first 0.9 GB. The auto-switch never ran, because it only ran
when resolve() failed, and a request for a real model id could not trigger it at all.
"""
from __future__ import annotations

import asyncio

import pytest

from app import db, hw, ini
from app import router_api as ra
from app import services
from app.routers import containers


def run(coro):
    return asyncio.run(coro)


def _b(name, engine, models, ready=True, cpu_only=False):
    return {"name": name, "engine": engine, "running": True, "url": f"http://{name}:8080",
            "models": models, "ready": ready, "cpu_only": cpu_only, "headers": {}}


INI = """[cyber]
ngl = 999
[voice-cpu]
ngl = 0
"""


@pytest.fixture(autouse=True)
def box(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "r.db")
    db.init()
    ra._init_db()
    cp = ini._new_parser()
    cp.read_string(INI)
    monkeypatch.setattr(ini, "read_ini", lambda: cp)
    # Unpinned Strata on a two-card box unless a test says otherwise.
    monkeypatch.setattr(services, "strata_held_cards", lambda: ())
    monkeypatch.setattr(hw, "sysfs_card_vram_gb", lambda: [32.0, 32.0])


def live(monkeypatch, *bs):
    by_model = {}
    for b in bs:
        if b.get("ready"):
            for m in b["models"]:
                by_model.setdefault(m, b)

    async def f():
        return list(bs), by_model
    monkeypatch.setattr(ra, "_live_map", f)


class TestResolve:
    def test_up_beside_the_holder_is_not_an_answer(self, monkeypatch):
        live(monkeypatch, _b("strata", "strata", ["q"]), _b("llama-9700x2", "llama", ["cyber"]))
        with pytest.raises(ra.Unresolved, match="strata holds the cards") as e:
            run(ra.resolve("cyber"))
        assert e.value.engine == "llama"

    def test_the_alias_too(self, monkeypatch):
        db.set_setting("router.default.llama", "cyber")
        live(monkeypatch, _b("strata", "strata", ["q"]), _b("llama-9700x2", "llama", ["cyber"]))
        with pytest.raises(ra.Unresolved, match="holds the cards") as e:
            run(ra.resolve("llama"))
        assert e.value.engine == "llama"

    def test_a_parked_strata_holds_nothing(self, monkeypatch):
        live(monkeypatch, _b("strata", "strata", [], ready=False), _b("llama-9700x2", "llama", ["cyber"]))
        b, m = run(ra.resolve("cyber"))
        assert b["name"] == "llama-9700x2" and m == "cyber"

    def test_a_cpu_router_holds_no_card(self, monkeypatch):
        live(monkeypatch, _b("strata", "strata", ["q"]),
             _b("llama-cpu", "llama", ["voice-cpu"], cpu_only=True))
        assert run(ra.resolve("voice-cpu"))[0]["name"] == "llama-cpu"
        assert run(ra.resolve("strata"))[0]["name"] == "strata"

    def test_a_pinned_strata_share_is_not_a_collision(self, monkeypatch):
        monkeypatch.setattr(services, "strata_held_cards", lambda: (1,))
        live(monkeypatch, _b("strata", "strata", ["q"]), _b("llama-9700x2", "llama", ["cyber"]))
        assert run(ra.resolve("cyber"))[0]["name"] == "llama-9700x2"
        assert run(ra.resolve("strata"))[0]["name"] == "strata"

    def test_strata_on_every_card_is_no_share(self, monkeypatch):
        monkeypatch.setattr(services, "strata_held_cards", lambda: (0, 1))
        live(monkeypatch, _b("strata", "strata", ["q"]), _b("llama-9700x2", "llama", ["cyber"]))
        with pytest.raises(ra.Unresolved, match="holds the cards"):
            run(ra.resolve("cyber"))

    def test_a_stopped_llama_model_names_its_engine(self, monkeypatch):
        live(monkeypatch, _b("strata", "strata", ["q"]))
        with pytest.raises(ra.Unresolved, match="no running backend serves") as e:
            run(ra.resolve("cyber"))
        assert e.value.engine == "llama"
        with pytest.raises(ra.Unresolved) as e:
            run(ra.resolve("not-a-section"))
        assert e.value.engine == ""


class TestForwardSwitches:
    def test_a_model_id_hands_the_cards_to_its_engine(self, monkeypatch):
        live(monkeypatch, _b("strata", "strata", ["q"]), _b("llama-9700x2", "llama", ["cyber"]))
        asked, switched = [], []

        async def may(target):
            asked.append(target)
            return True, ""

        async def req(target):
            switched.append(target)
            return True, ""

        async def no_swap(e):
            return False
        monkeypatch.setattr(ra, "may_autoswitch", may)
        monkeypatch.setattr(ra, "request_switch", req)
        monkeypatch.setattr(ra, "swap_in_flight_for", no_swap)
        monkeypatch.setattr(ra, "narrated_response", lambda *a, **k: "narrating")

        class Req:
            headers = {}
            client = None

            async def json(self):
                return {"model": "cyber", "stream": True, "messages": [{"role": "user", "content": "x"}]}
        assert run(ra._forward(Req(), "/v1/chat/completions", "openai")) == "narrating"
        assert asked == ["llama"] and switched == ["llama"]


class TestRestartNeverStarts:
    """Restart applies changes to what is running. It must never start a stopped engine (that
    would take the cards), nor switch engines: a models.ini edit saved for later is picked up
    at the router's next start."""

    def test_a_stopped_backend_is_left_alone(self, monkeypatch):
        from app import fit
        monkeypatch.setattr(fit, "_container_status", lambda n: "exited")
        monkeypatch.setattr(services, "restart_llama_backend",
                            lambda n: pytest.fail("a stopped engine must not be started"))
        monkeypatch.setattr(services, "activate_engine", lambda *a, **k: pytest.fail("no switch"))
        r = containers.containers_restart("llama-9700x2")
        assert b"is stopped; it picks up changes when it next starts" in r.body

    def test_a_running_one_still_restarts(self, monkeypatch):
        from app import fit
        monkeypatch.setattr(fit, "_container_status", lambda n: "running")
        done = []
        monkeypatch.setattr(services, "restart_llama_backend", lambda n: done.append(n))
        containers.containers_restart("llama-9700x2")
        assert done == ["llama-9700x2"]

    def test_apply_now_offers_only_running_routers(self, monkeypatch):
        from app import fit
        from app.routers import config
        monkeypatch.setattr(services, "_effective_container_names",
                            lambda: ["llama-9700x2", "llama-cpu", "strata"])
        monkeypatch.setattr(services, "engine_for", lambda n: "strata" if n == "strata" else "llama")
        monkeypatch.setattr(fit, "_container_status",
                            lambda n: "exited" if n == "llama-9700x2" else "running")
        assert config._llama_restart_split() == {"llama_backends": ["llama-cpu"],
                                                 "llama_stopped": ["llama-9700x2"]}
