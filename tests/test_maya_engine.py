"""Maya as the third engine, and the Strata park that reported a timeout as a failure.

Maya (Project Maya: GLM-5.3-Flash / DeepSeek-V4.1-Flash) is discovered like any engine, by its
`ai-lab.engine` label. What it needs that a llama router does not: readiness from /health's
status, the single-model answer to its alias, and - because it cannot park the way Strata does -
an alias that stays listed while its container is stopped.
"""
from __future__ import annotations

import asyncio
import functools

import httpx
import pytest

from app import db
from app import router_api as ra
from app.services import engines, stratactl


@pytest.fixture
def rdb(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "router.db")
    db.init()
    ra._init_db()
    ra._init_rollup()
    yield tmp_path / "router.db"


def run(coro):
    return asyncio.run(coro)


def _patch_async(monkeypatch, handler):
    monkeypatch.setattr(httpx, "AsyncClient",
                        functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler)))


def _patch_sync(monkeypatch, handler):
    monkeypatch.setattr(httpx, "Client",
                        functools.partial(httpx.Client, transport=httpx.MockTransport(handler)))


MAYA = {"engine": "maya", "url": "http://maya:8080", "models": ["glm-5.3-flash"]}


# ------------------------------------------------------------------ router readiness

class TestMayaReady:
    def test_ok_is_ready(self, monkeypatch, rdb):
        _patch_async(monkeypatch, lambda req: httpx.Response(200, json={"status": "ok"}))
        assert run(ra._ready(httpx.AsyncClient(), MAYA)) is True

    @pytest.mark.parametrize("status", ["reloading", "updating", ""])
    def test_up_but_not_taking_requests(self, monkeypatch, rdb, status):
        _patch_async(monkeypatch, lambda req: httpx.Response(200, json={"status": status}))
        assert run(ra._ready(httpx.AsyncClient(), MAYA)) is False

    def test_closed_port_while_loading(self, monkeypatch, rdb):
        def h(req):
            raise httpx.ConnectError("port not open until the model is in")
        _patch_async(monkeypatch, h)
        assert run(ra._ready(httpx.AsyncClient(), MAYA)) is False

    def test_non_200_and_non_object(self, monkeypatch, rdb):
        _patch_async(monkeypatch, lambda req: httpx.Response(503, json={"status": "ok"}))
        assert run(ra._ready(httpx.AsyncClient(), MAYA)) is False
        _patch_async(monkeypatch, lambda req: httpx.Response(200, json=["ok"]))
        assert run(ra._ready(httpx.AsyncClient(), MAYA)) is False

    def test_strata_rule_unchanged(self, monkeypatch, rdb):
        # Maya's "status": "ok" must not leak into Strata, whose /health also says ok while
        # parked - it is loaded:true that means serving there.
        _patch_async(monkeypatch, lambda req: httpx.Response(200, json={"status": "ok",
                                                                        "loaded": False}))
        b = {"engine": "strata", "url": "http://strata:8080", "models": ["q"]}
        assert run(ra._ready(httpx.AsyncClient(), b)) is False


# ------------------------------------------------------------------ resolve

def _lm(bs, by_model):
    async def f():
        return bs, by_model
    return f


def _b(name, engine, models, ready=True, running=True):
    return {"name": name, "engine": engine, "running": running,
            "url": f"http://{name}:8080", "models": models, "ready": ready}


class TestMayaResolve:
    def test_alias_gets_whatever_is_loaded(self, monkeypatch, rdb):
        m = _b("maya", "maya", ["glm-5.3-flash"])
        monkeypatch.setattr(ra, "_live_map", _lm([m], {"glm-5.3-flash": m}))
        b, model = run(ra.resolve("maya"))
        assert b is m and model == "glm-5.3-flash"

    def test_no_default_needed(self, monkeypatch, rdb):
        # The llama path demands a default when a backend offers several models; a single-model
        # engine never asks, whatever the router has recorded.
        m = _b("maya", "maya", ["glm-5.3-flash"])
        monkeypatch.setattr(ra, "_live_map", _lm([m], {}))
        assert run(ra.resolve("maya"))[1] == "glm-5.3-flash"

    def test_held_by_another_engine(self, monkeypatch, rdb):
        s = _b("strata", "strata", ["q"])
        monkeypatch.setattr(ra, "_live_map", _lm([s], {"q": s}))
        with pytest.raises(ra.Unresolved, match="not the engine holding the cards"):
            run(ra.resolve("maya"))

    def test_still_loading(self, monkeypatch, rdb):
        m = _b("maya", "maya", [], ready=False)
        monkeypatch.setattr(ra, "_live_map", _lm([m], {}))
        with pytest.raises(ra.Unresolved, match="starting up"):
            run(ra.resolve("maya"))

    def test_default_engine_order_unchanged(self):
        # ENGINES[0] answers a request that names no model, and /props asks it first.
        assert ra.ENGINES[:2] == ("strata", "llama") and "maya" in ra.ENGINES
        assert ra.SINGLE_MODEL == {"strata", "maya"}


# ------------------------------------------------------------------ the dropdown

class TestAliasesListedWhileStopped:
    def _setup(self, monkeypatch, stopped):
        live = [_b("strata", "strata", ["q"])]
        monkeypatch.setattr(ra, "_live_map", _lm(live, {"q": live[0]}))
        monkeypatch.setattr(ra, "_backends", lambda include_stopped=False:
                            live + (stopped if include_stopped else []))

    def _ids(self):
        return [d["id"] for d in run(ra.list_models())["data"]]

    def test_stopped_engine_keeps_its_alias(self, monkeypatch, rdb):
        # The engine you would switch TO is the stopped one; hiding it made the dropdown useless
        # as a switch. Maya cannot park, so this is the only way it is ever listed while Strata
        # holds the cards.
        self._setup(monkeypatch, [_b("maya", "maya", [], ready=False, running=False)])
        ids = self._ids()
        assert ids[:2] == ["strata", "maya"] and "q" in ids

    def test_no_container_no_alias(self, monkeypatch, rdb):
        self._setup(monkeypatch, [])
        assert self._ids() == ["strata", "q"]


class TestPropsTriesEveryEngine:
    def test_falls_through_to_maya(self, monkeypatch, rdb):
        m = _b("maya", "maya", ["glm-5.3-flash"])

        async def resolve(e):
            if e == "maya":
                return m, "glm-5.3-flash"
            raise ra.Unresolved(f"{e} not here")
        monkeypatch.setattr(ra, "resolve", resolve)
        _patch_async(monkeypatch, lambda req: httpx.Response(200, json={"n_ctx": 32768}))
        r = run(ra.props())
        assert r.status_code == 200 and b"32768" in r.body

    def test_nobody_answers(self, monkeypatch, rdb):
        async def resolve(e):
            raise ra.Unresolved(f"{e} not here")
        monkeypatch.setattr(ra, "resolve", resolve)
        r = run(ra.props())
        assert r.status_code == 503 and b"maya not here" in r.body


# ------------------------------------------------------------------ the switch's wait

class _Cli:
    class containers:
        @staticmethod
        def get(name):
            class C:
                attrs = {"NetworkSettings": {"Ports": {"8080/tcp": None}}}
            return C()


class TestWaitReady:
    def _setup(self, monkeypatch, engine):
        monkeypatch.setattr(engines, "_docker_client", lambda: _Cli())
        monkeypatch.setattr(engines, "engine_for", lambda name: engine)
        monkeypatch.setattr(engines.time, "sleep", lambda s: None)
        # _wait_ready writes progress into the module's switch state; a fresh dict keeps that
        # out of every later test that expects "no switch has run".
        monkeypatch.setattr(engines, "_switch_state", {})

    def test_maya_ok_is_ready(self, monkeypatch):
        self._setup(monkeypatch, "maya")
        seen = []

        def h(req):
            seen.append(req.url.path)
            return httpx.Response(200, json={"status": "ok"})
        _patch_sync(monkeypatch, h)
        assert engines._wait_ready("maya") == (True, "")
        assert seen == ["/health"]

    def test_maya_reloading_waits_then_gives_up(self, monkeypatch):
        self._setup(monkeypatch, "maya")
        _patch_sync(monkeypatch, lambda req: httpx.Response(200, json={"status": "reloading"}))
        ok, note = engines._wait_ready("maya", timeout=0.05)
        assert ok is False and "/health did not answer" in note
        assert engines._switch_state.get("msg", "").endswith("loading the model")

    def test_maya_gets_the_long_timeout(self):
        assert engines.READY_TIMEOUT_S["maya"] >= 600
        # The router's narration must outlast every engine's wait, or it abandons a start
        # that is still going.
        assert ra._NARRATE_MAX_S > max(engines.READY_TIMEOUT_S.values())


# ------------------------------------------------------------------ the Strata park

class TestStrataUnloadTimeout:
    def _control(self, monkeypatch, result):
        calls = []

        def ctl(name, path, timeout):
            calls.append((path, timeout))
            return result
        monkeypatch.setattr(stratactl, "_strata_control", ctl)
        monkeypatch.setattr(stratactl.time, "sleep", lambda s: None)
        return calls

    def test_slow_answer_but_parked_is_success(self, monkeypatch):
        # The 2026-10-20 case: /unload ran past its timeout, the model was unloaded anyway.
        self._control(monkeypatch, (False, "http://strata:8080/unload: timed out after 60s"))
        monkeypatch.setattr(stratactl, "strata_loaded", lambda name: False)
        ok, msg = stratactl.strata_unload("strata")
        assert ok is True and "parked" in msg

    def test_timeout_and_still_loaded_is_a_failure(self, monkeypatch):
        self._control(monkeypatch, (False, "http://strata:8080/unload: timed out after 60s"))
        monkeypatch.setattr(stratactl, "strata_loaded", lambda name: True)
        monkeypatch.setattr(stratactl, "_UNLOAD_CONFIRM_S", 0.0)
        ok, msg = stratactl.strata_unload("strata")
        assert ok is False and "still reports the model loaded" in msg

    def test_parks_during_the_confirm_window(self, monkeypatch):
        self._control(monkeypatch, (False, "http://strata:8080/unload: timed out after 60s"))
        answers = iter([True, None, False])
        monkeypatch.setattr(stratactl, "strata_loaded", lambda name: next(answers))
        assert stratactl.strata_unload("strata")[0] is True

    def test_a_real_refusal_is_not_second_guessed(self, monkeypatch):
        # 409 means a request is in flight and the model is NOT unloaded; /health must not be
        # consulted to turn that into a success.
        msg409 = "a request is running or queued - Strata will not unload mid-request"
        self._control(monkeypatch, (False, msg409))
        monkeypatch.setattr(stratactl, "strata_loaded",
                            lambda name: pytest.fail("must not poll /health on a refusal"))
        assert stratactl.strata_unload("strata") == (False, msg409)

    def test_timeout_is_longer_than_it_was(self, monkeypatch):
        calls = self._control(monkeypatch, (True, "unloaded"))
        assert stratactl.strata_unload("strata") == (True, "unloaded")
        assert calls == [("/unload", stratactl._UNLOAD_TIMEOUT_S)]
        assert stratactl._UNLOAD_TIMEOUT_S > 10.0

    def test_control_names_a_timeout(self, monkeypatch):
        class Cli:
            class containers:
                @staticmethod
                def get(name):
                    class C:
                        attrs = {"NetworkSettings": {"Ports": {"8080/tcp": [{"HostPort": "8085"}]}}}
                    return C()
        monkeypatch.setattr(stratactl, "_docker_client", lambda: Cli())

        def h(req):
            raise httpx.ReadTimeout("slow", request=req)
        _patch_sync(monkeypatch, h)
        ok, msg = stratactl._strata_control("strata", "/unload", timeout=1.0)
        assert ok is False and stratactl._TIMED_OUT in msg


# ------------------------------------------------------------------ the overview's meters

class TestMayaSpeed:
    """Maya publishes Strata's /metrics, so it rides Strata's reader; its /slots is smaller."""

    def _r(self, newest="generating", prompt=17, gen=73, tps=33.0):
        return {"newest": newest, "cur_gen_tps": tps, "cur_gen_tokens": gen,
                "cur_prompt_tokens": prompt, "cur_pp_pct": 0, "cur_pp_done": 0, "cur_pp_total": 0}

    def test_engine_is_read_through_metrics(self):
        from app.services import speed
        assert "maya" in speed._METRICS_ENGINES and "strata" in speed._METRICS_ENGINES

    def test_busy_slot_takes_its_size_from_metrics(self):
        # Measured shape, Maya 1.1.16, mid-generation.
        from app.services import speed
        raw = [{"id": 0, "n_ctx": 32768, "is_processing": True}]
        sl = speed._slots_strata(raw, self._r())[0]
        assert (sl.state, sl.gen_tps, sl.decoded) == ("generating", 33.0, 73)
        assert (sl.ctx_used, sl.ctx_total, sl.prompt_tokens) == (90, 32768, 17)

    def test_reading_counts_the_prompt_only(self):
        from app.services import speed
        raw = [{"id": 0, "n_ctx": 32768, "is_processing": True}]
        sl = speed._slots_strata(raw, self._r(newest="prefill", prompt=4000, gen=0))[0]
        assert (sl.state, sl.ctx_used) == ("prefill", 4000)

    def test_idle_slot_claims_nothing(self):
        from app.services import speed
        raw = [{"id": 0, "n_ctx": 32768, "is_processing": False}]
        sl = speed._slots_strata(raw, self._r(newest=""))[0]
        assert (sl.state, sl.ctx_used, sl.prompt_tokens) == ("idle", 0, 0)

    def test_strata_shape_unchanged(self):
        # Strata sends n_prompt_tokens; that stays the figure, even when it disagrees with live.
        from app.services import speed
        raw = [{"id": 0, "n_ctx": 262144, "is_processing": True, "n_prompt_tokens": 33137}]
        sl = speed._slots_strata(raw, self._r(prompt=5, gen=999))[0]
        assert (sl.ctx_used, sl.prompt_tokens) == (33137, 33137)


# ------------------------------------------------------------------ the Router tab

from tests.test_routes_smoke import web  # noqa: E402,F401 - the fixture: real app, empty box


class TestRouterPageWithMaya:
    def test_renders_with_maya_running(self, web, monkeypatch):
        # The first live run with Maya up 500'd this page: its model list reached the template
        # as bare strings where (value, label) pairs were expected.
        m = _b("maya", "maya", ["glm-5.3-flash"])
        monkeypatch.setattr(ra, "_live_map", _lm([m], {"glm-5.3-flash": m}))
        r = web.get("/router")
        assert r.status_code == 200
        assert 'value="maya"' in r.text
        # No default picker for a one-model engine: it would be a control that does nothing.
        assert "always means that model" in r.text
        defaults = r.text.split('hx-post="/router/default"')[1:]
        assert defaults and not any('value="maya"' in f.split("</form>")[0] for f in defaults)


class TestAutoswitchToggle:
    """The control was an amber chip that read as a status badge, and clicking it replaced it
    with a 'Saved' chip - nothing left to click. It is a labelled switch now, and stays one."""

    def test_page_shows_a_switch_per_engine(self, web):
        r = web.get("/router")
        assert r.status_code == 200
        assert r.text.count('role="switch"') == len(ra.ENGINES)
        assert "Auto-switch" in r.text

    def test_post_returns_the_switch_in_its_new_state(self, web):
        r = web.post("/router/autoswitch", data={"engine": "maya", "on": "1"})
        assert r.status_code == 200
        assert 'role="switch"' in r.text and 'aria-checked="true"' in r.text
        assert 'name="on" value=""' in r.text          # the next click turns it off
        assert ra.autoswitch_enabled("maya")

        r = web.post("/router/autoswitch", data={"engine": "maya", "on": ""})
        assert 'aria-checked="false"' in r.text and 'name="on" value="1"' in r.text
        assert not ra.autoswitch_enabled("maya")

    def test_unknown_engine_refused(self, web):
        r = web.post("/router/autoswitch", data={"engine": "nope", "on": "1"})
        assert 'role="switch"' not in r.text
