"""Characterization tests for the router's brain: app/router_api.py.

The routing decisions - conversation keys, backend ranking, alias resolution,
pins and defaults, the retention rollups - are pure logic over sqlite and stubbed
docker/httpx, and they are the parts that decide where every inference request on
the box actually goes.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import time

import httpx
import pytest

from app import db
from app import router_api as ra


@pytest.fixture
def rdb(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "router.db")
    db.init()
    ra._init_db()
    ra._init_rollup()
    yield tmp_path / "router.db"


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ conversation_key

class TestConversationKey:
    def test_explicit_user_is_believed(self):
        key = ra.conversation_key({"user": "alice", "messages": [{"role": "user", "content": "x"}]})
        assert key == "u:" + hashlib.sha256(b"alice").hexdigest()[:16]

    def test_user_beats_messages_and_is_stripped(self):
        a = ra.conversation_key({"user": " bob ", "messages": [{"role": "user", "content": "one"}]})
        b = ra.conversation_key({"user": "bob", "messages": [{"role": "user", "content": "two"}]})
        assert a == b

    def test_prefix_of_growing_history_keys_the_same(self):
        first = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        t1 = ra.conversation_key({"messages": first})
        t2 = ra.conversation_key({"messages": first + [{"role": "assistant", "content": "A"},
                                                       {"role": "user", "content": "next"}]})
        assert t1 == t2 and t1.startswith("p:")

    def test_different_openings_are_different_conversations(self):
        a = ra.conversation_key({"messages": [{"role": "user", "content": "one"}]})
        b = ra.conversation_key({"messages": [{"role": "user", "content": "two"}]})
        assert a != b

    def test_prompt_fallback_when_no_messages(self):
        key = ra.conversation_key({"prompt": "tell me a joke"})
        assert key.startswith("p:")

    def test_empty_body_gets_no_key_not_a_shared_one(self):
        # A shared "hash of nothing" merged unrelated failed probes into one thread.
        assert ra.conversation_key({}) == ""
        assert ra.conversation_key({"messages": []}) == ""


# ------------------------------------------------------------------ ranking

class TestRank:
    def test_cpu_backend_sorts_last(self):
        assert ra._rank({"name": "llama"}) < ra._rank({"name": "llama-cpu"})

    def test_purpose_named_sorts_after_general(self):
        assert ra._rank({"name": "llama"}) < ra._rank({"name": "llama-voice"})
        assert ra._rank({"name": "llama"}) < ra._rank({"name": "llama-asr"})

    def test_full_order(self):
        names = ["llama-cpu", "llama-voice", "llama", "strata"]
        assert sorted(names, key=lambda n: ra._rank({"name": n})) == \
            ["llama", "strata", "llama-voice", "llama-cpu"]


# ------------------------------------------------------------------ discovery

class FakeContainer:
    def __init__(self, name, image, labels=None, status="running", port="8085"):
        self.name, self.status = name, status
        ports = {"8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": port}]} if port else {}
        self.attrs = {
            "Config": {"Image": image, "Labels": labels or {}},
            "NetworkSettings": {"Ports": ports},
        }


class FakeClient:
    def __init__(self, containers, raises=False):
        self._c, self._raises = containers, raises
        outer = self

        class _Containers:
            def list(self, all=False):
                if outer._raises:
                    raise RuntimeError("daemon down")
                return [c for c in outer._c if all or c.status == "running"]

        self.containers = _Containers()


def _docker(monkeypatch, containers=None, raises=False, none=False):
    if none:
        monkeypatch.setattr(ra, "docker_client", lambda: None)
    else:
        cli = FakeClient(containers or [], raises=raises)
        monkeypatch.setattr(ra, "docker_client", lambda: cli)


class TestBackends:
    def test_label_names_the_engine(self, monkeypatch, rdb):
        _docker(monkeypatch, [FakeContainer("strata", "strata:latest",
                                            labels={"ai-lab.engine": "strata"})])
        bs = ra._backends()
        assert bs == [{"name": "strata", "engine": "strata", "running": True,
                       "cpu_only": False, "url": "http://strata:8080", "models": [], "headers": {},
                       "started_at": ""}]

    def test_llamacpp_image_fallback(self, monkeypatch, rdb):
        _docker(monkeypatch, [FakeContainer("rtr", "ghcr.io/ggml-org/llama.cpp:server")])
        assert ra._backends()[0]["engine"] == "llama"

    def test_name_fallback(self, monkeypatch, rdb):
        _docker(monkeypatch, [FakeContainer("llama-voice", "custom:img")])
        assert ra._backends()[0]["engine"] == "llama"

    def test_unrelated_container_ignored(self, monkeypatch, rdb):
        _docker(monkeypatch, [FakeContainer("open-webui", "ghcr.io/open-webui/open-webui")])
        assert ra._backends() == []

    def test_running_without_port_is_unusable(self, monkeypatch, rdb):
        _docker(monkeypatch, [FakeContainer("odd", "llama.cpp", port=None)])
        assert ra._backends() == []

    def test_stopped_only_for_include_stopped(self, monkeypatch, rdb):
        _docker(monkeypatch, [FakeContainer("llama-off", "llama.cpp", status="exited")])
        assert ra._backends() == []
        assert ra._backends(include_stopped=True)[0]["name"] == "llama-off"

    def test_no_docker_or_hiccup_is_nothing_found(self, monkeypatch, rdb):
        _docker(monkeypatch, none=True)
        assert ra._backends() == []
        _docker(monkeypatch, raises=True)
        assert ra._backends() == []


def _transport(handler):
    return httpx.MockTransport(handler)


def _patch_client(monkeypatch, handler):
    monkeypatch.setattr(httpx, "AsyncClient",
                        functools.partial(httpx.AsyncClient, transport=_transport(handler)))


class TestReadyAndModels:
    def test_strata_health_decides(self, monkeypatch, rdb):
        def h(req):
            return httpx.Response(200, json={"loaded": req.url.path == "/health"})
        _patch_client(monkeypatch, h)
        b = {"engine": "strata", "url": "http://s:1", "models": ["m"]}
        assert run(ra._ready(httpx.AsyncClient(), b)) is True

    def test_strata_up_but_serving_nothing_is_not_ready(self, monkeypatch, rdb):
        _patch_client(monkeypatch, lambda req: httpx.Response(200, json={"loaded": False}))
        b = {"engine": "strata", "url": "http://s:1", "models": ["m"]}
        assert run(ra._ready(httpx.AsyncClient(), b)) is False

    def test_strata_health_error_is_not_ready(self, monkeypatch, rdb):
        def h(req):
            raise httpx.ConnectError("closed port")
        _patch_client(monkeypatch, h)
        b = {"engine": "strata", "url": "http://s:1", "models": ["m"]}
        assert run(ra._ready(httpx.AsyncClient(), b)) is False

    def test_llama_readiness_is_having_models(self):
        assert run(ra._ready(None, {"engine": "llama", "models": ["m"]})) is True
        assert run(ra._ready(None, {"engine": "llama", "models": []})) is False

    def test_models_of_lists_ids(self, monkeypatch, rdb):
        _patch_client(monkeypatch, lambda req: httpx.Response(
            200, json={"data": [{"id": "a"}, {"id": ""}, {"nope": 1}]}))
        assert run(ra._models_of(httpx.AsyncClient(), {"url": "http://x:1"})) == ["a"]

    def test_models_of_swallows_errors(self, monkeypatch, rdb):
        def h(req):
            raise httpx.ConnectError("nope")
        _patch_client(monkeypatch, h)
        assert run(ra._models_of(httpx.AsyncClient(), {"url": "http://x:1"})) == []


class TestLiveMap:
    def _handler(self, req):
        if req.url.host == "llama-cpu":
            return httpx.Response(200, json={"data": [{"id": "shared"}, {"id": "cpu-only"}]})
        if req.url.host == "llama-voice":
            return httpx.Response(200, json={"data": [{"id": "shared"}]})
        if req.url.path == "/health":
            return httpx.Response(200, json={"loaded": True})
        return httpx.Response(200, json={"data": [{"id": "strata-model"}]})

    def test_ranked_and_first_ready_wins(self, monkeypatch, rdb):
        monkeypatch.setattr(ra, "_backends", lambda include_stopped=False: [
            {"name": "llama-cpu", "engine": "llama", "running": True,
             "url": "http://llama-cpu:8080", "models": []},
            {"name": "llama-voice", "engine": "llama", "running": True,
             "url": "http://llama-voice:8080", "models": []},
            {"name": "strata", "engine": "strata", "running": True,
             "url": "http://strata:8080", "models": []},
        ])
        _patch_client(monkeypatch, self._handler)
        bs, by_model = run(ra._live_map())
        assert [b["name"] for b in bs] == ["strata", "llama-voice", "llama-cpu"]
        assert by_model["shared"]["name"] == "llama-voice"   # ranked before cpu
        assert by_model["strata-model"]["name"] == "strata"
        assert all(b["ready"] for b in bs)


# ------------------------------------------------------------------ resolve

def _lm(bs, by_model):
    async def f():
        return bs, by_model
    return f


def _b(name, engine, models, ready=True):
    return {"name": name, "engine": engine, "running": True,
            "url": f"http://{name}:1", "models": models, "ready": ready}


class TestResolve:
    def test_no_backends(self, monkeypatch, rdb):
        monkeypatch.setattr(ra, "_live_map", _lm([], {}))
        with pytest.raises(ra.Unresolved, match="no inference backend is running"):
            run(ra.resolve("strata"))

    def test_strata_alias_gets_whatever_it_loaded(self, monkeypatch, rdb):
        s = _b("strata", "strata", ["model-x"])
        monkeypatch.setattr(ra, "_live_map", _lm([s], {}))
        b, m = run(ra.resolve("strata"))
        assert b is s and m == "model-x"

    def test_starting_up_gets_its_own_answer(self, monkeypatch, rdb):
        s = _b("strata", "strata", [], ready=False)
        monkeypatch.setattr(ra, "_live_map", _lm([s], {}))
        with pytest.raises(ra.Unresolved, match="starting up.*not ready"):
            run(ra.resolve("strata"))

    def test_other_engine_holds_the_cards(self, monkeypatch, rdb):
        l = _b("llama", "llama", ["m"])
        with pytest.raises(ra.Unresolved, match="not the engine holding the cards"):
            monkeypatch.setattr(ra, "_live_map", _lm([l], {}))
            run(ra.resolve("strata"))

    def test_pin_to_missing_backend_is_an_error_not_a_fallback(self, monkeypatch, rdb):
        db.set_setting("router.backend.strata", "strata-b")
        a = _b("strata-a", "strata", ["m"])
        monkeypatch.setattr(ra, "_live_map", _lm([a], {}))
        with pytest.raises(ra.Unresolved, match="pinned to strata-b, which is not running"):
            run(ra.resolve("strata"))

    def test_pin_is_honoured(self, monkeypatch, rdb):
        db.set_setting("router.backend.llama", "llama-voice")
        v = _b("llama-voice", "llama", ["m"])
        c = _b("llama-cpu", "llama", ["m"])
        monkeypatch.setattr(ra, "_live_map", _lm([v, c], {"m": v}))
        b, m = run(ra.resolve("llama"))
        assert b["name"] == "llama-voice"

    def test_llama_default_honoured_when_offered(self, monkeypatch, rdb):
        db.set_setting("router.default.llama", "big")
        l = _b("llama", "llama", ["big", "small"])
        monkeypatch.setattr(ra, "_live_map", _lm([l], {"big": l}))
        b, m = run(ra.resolve("llama"))
        assert (b["name"], m) == ("llama", "big")

    def test_llama_default_elsewhere_moves_the_request(self, monkeypatch, rdb):
        db.set_setting("router.default.llama", "voice-only")
        l = _b("llama", "llama", ["big"])
        v = _b("llama-voice", "llama", ["voice-only"])
        monkeypatch.setattr(ra, "_live_map", _lm([l, v], {"voice-only": v}))
        b, m = run(ra.resolve("llama"))
        assert (b["name"], m) == ("llama-voice", "voice-only")

    def test_llama_default_offered_by_nobody(self, monkeypatch, rdb):
        db.set_setting("router.default.llama", "gone")
        l = _b("llama", "llama", ["big"])
        monkeypatch.setattr(ra, "_live_map", _lm([l], {}))
        with pytest.raises(ra.Unresolved, match="default llama model 'gone' is not offered"):
            run(ra.resolve("llama"))

    def test_no_default_falls_back_to_last_used(self, monkeypatch, rdb):
        ra._record(ts=time.time(), engine="llama", model="was-using", status=200)
        l = _b("llama", "llama", ["was-using", "other"])
        monkeypatch.setattr(ra, "_live_map", _lm([l], {"was-using": l}))
        b, m = run(ra.resolve("llama"))
        assert m == "was-using"

    def test_no_default_no_history_single_model_answers(self, monkeypatch, rdb):
        l = _b("llama", "llama", ["only"])
        monkeypatch.setattr(ra, "_live_map", _lm([l], {"only": l}))
        b, m = run(ra.resolve("llama"))
        assert m == "only"

    def test_no_default_no_history_many_models_asks(self, monkeypatch, rdb):
        l = _b("llama", "llama", ["a", "b"])
        monkeypatch.setattr(ra, "_live_map", _lm([l], {}))
        with pytest.raises(ra.Unresolved, match="no default model is set for llama"):
            run(ra.resolve("llama"))

    def test_real_id_goes_to_preferred_backend(self, monkeypatch, rdb):
        v = _b("llama-voice", "llama", ["voice"])
        c = _b("llama-cpu", "llama", ["voice"])
        monkeypatch.setattr(ra, "_live_map", _lm([v, c], {"voice": v}))
        b, m = run(ra.resolve("voice"))
        assert b["name"] == "llama-voice" and m == "voice"

    def test_unknown_id(self, monkeypatch, rdb):
        l = _b("llama", "llama", ["m"])
        monkeypatch.setattr(ra, "_live_map", _lm([l], {"m": l}))
        with pytest.raises(ra.Unresolved, match="no running backend serves 'nope'"):
            run(ra.resolve("nope"))


# ------------------------------------------------------------------ accounting

class TestAccounting:
    def test_record_and_last_used_only_counts_200s(self, rdb):
        now = time.time()
        ra._record(ts=now - 10, engine="llama", model="failed", status=502)
        ra._record(ts=now - 5, engine="llama", model="good", status=200)
        ra._record(ts=now - 1, engine="llama", model="newer-fail", status=500)
        assert ra.last_model_for("llama") == "good"

    def test_kill_switch_stops_bookkeeping(self, rdb):
        db.set_setting("router.record", "0")
        ra._record(ts=time.time(), engine="llama", model="ignored", status=200)
        assert ra.last_model_for("llama") == ""

    def test_defaults_and_pins_are_settings(self, rdb):
        assert ra.default_for("llama") == ""
        db.set_setting("router.default.llama", "m")
        db.set_setting("router.backend.strata", "s")
        assert ra.default_for("llama") == "m"
        assert ra.pinned_backend("strata") == "s"

    def test_client_of(self):
        class Req:
            def __init__(self, headers, host):
                self.headers, self.client = headers, type("C", (), {"host": host})()
        a = ra._client_of(Req({"authorization": "Bearer sk-secret"}, "9.9.9.9"))
        assert a == "key:" + hashlib.sha256(b"sk-secret").hexdigest()[:12]
        assert ra._client_of(Req({}, "9.9.9.9")) == "9.9.9.9"
        assert ra._client_of(Req({"authorization": "Bearer "}, "")) == "unknown"


class TestUsageFromSse:
    def test_openai_usage_line(self):
        acc = {}
        line = "data: " + json.dumps({"usage": {
            "prompt_tokens": 100, "completion_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 90}}})
        ra._usage_from_sse(line, "openai", acc)
        assert acc == {"prompt_tokens": 100, "completion_tokens": 7, "cached_tokens": 90}

    def test_openai_partial_lines_keep_earlier_values(self):
        acc = {}
        ra._usage_from_sse('data: {"usage": {"prompt_tokens": 10}}', "openai", acc)
        ra._usage_from_sse('data: {"usage": {"completion_tokens": 3}}', "openai", acc)
        assert acc["prompt_tokens"] == 10 and acc["completion_tokens"] == 3

    def test_anthropic_shapes(self):
        acc = {}
        ra._usage_from_sse('data: {"message": {"usage": {"input_tokens": 5, "output_tokens": 2}}}',
                           "anthropic", acc)
        assert acc == {"prompt_tokens": 5, "completion_tokens": 2}
        acc2 = {}
        ra._usage_from_sse('data: {"usage": {"input_tokens": 9}}', "anthropic", acc2)
        assert acc2["prompt_tokens"] == 9

    def test_noise_is_ignored(self):
        acc = {}
        for line in ("", ": keepalive", "data:", "data: [DONE]", "data: {broken"):
            ra._usage_from_sse(line, "openai", acc)
        assert acc == {}


class TestRetention:
    def test_compact_folds_raw_into_hourly_once(self, rdb):
        now = time.time()
        old = now - (ra.RETAIN_RAW_DAYS + 1) * 86400
        ra._record(ts=old, engine="llama", model="m", status=200, prompt_tokens=10,
                   completion_tokens=2, total_ms=100)
        ra._record(ts=old + 5, engine="llama", model="m", status=500, total_ms=50)
        ra._record(ts=now, engine="llama", model="m", status=200, prompt_tokens=1)
        out = ra.compact(now)
        assert out["hourly"] == 1 and out["raw_deleted"] == 2
        # Idempotent: a second pass over the same window must not double the totals.
        out2 = ra.compact(now)
        assert out2["hourly"] == 0 and out2["raw_deleted"] == 0
        t = ra.totals(now - 400 * 86400)
        assert t["requests"] == 3 and t["failures"] == 1
        assert t["prompt_tokens"] == 11

    def test_totals_span_raw_and_rollup(self, rdb):
        now = time.time()
        ra._record(ts=now - (ra.RETAIN_RAW_DAYS + 1) * 86400, engine="strata",
                   model="s", status=200, prompt_tokens=100)
        ra._record(ts=now, engine="strata", model="s", status=200, prompt_tokens=1)
        ra.compact(now)
        t = ra.totals(now - 200 * 86400)
        assert t["requests"] == 2 and t["prompt_tokens"] == 101

    def test_expiry_removes_ancient_rollups(self, rdb):
        now = time.time()
        ra._record(ts=now - (ra.RETAIN_TOTAL_DAYS + 1) * 86400, engine="llama",
                   model="m", status=200)
        out = ra.compact(now)
        assert out["raw_deleted"] == 1
        assert ra.totals(0)["requests"] == 0
