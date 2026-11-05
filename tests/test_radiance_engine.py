"""Radiance (codeberg.org/StillDeadcode/radiance) as the fifth engine (2026-11-05).

Its own server: /health says "ok" once the model is in, /stats lists every running request with
the slot it holds, finished requests are only in its log. It pins ~90 GB of RAM, so a take-over by
it stops a Strata instead of parking it.
"""
from __future__ import annotations

import asyncio

from app import engine_registry as reg
from app import router_api as ra
from app.services import backends, engines, speed
from tests.test_routes_smoke import web  # noqa: F401  (the fixture)


def test_it_is_registered_with_its_own_protocol():
    spec = reg.get("radiance")
    assert spec.label == "Radiance" and spec.single_model and spec.needs_ram
    assert "radiance" in ra.SINGLE_MODEL and "radiance" not in speed._METRICS_ENGINES
    assert reg.speaks_radiance("radiance") and not reg.speaks_radiance("strata")
    assert not reg.speaks_strata("radiance")
    assert not spec.parks and spec.updates["kind"] == "dockerhub"


def test_serving_needs_health_ok():
    assert reg.serving("radiance", 200, {"status": "ok", "model": "m", "in_flight": 0})
    assert not reg.serving("radiance", 200, {"status": "loading"})
    assert not reg.serving("radiance", 503, None)
    # Maya's rule is unchanged by moving it onto ready_status.
    assert reg.serving("maya", 200, {"status": "ok"}) and not reg.serving("maya", 200, {"status": "updating"})


def test_its_key_is_read_from_its_own_variable():
    attrs = {"Config": {"Env": ["RADIANCE_API_KEY=k123", "PATH=/bin"]}}
    assert backends.api_key_of(attrs) == "k123"


def test_a_soft_take_over_by_radiance_stops_strata(monkeypatch):
    monkeypatch.setattr(engines, "gpu_engines", lambda: [{"engine": "radiance", "names": ["radiance"]},
                                                         {"engine": "strata", "names": ["strata"]}])
    monkeypatch.setattr(engines, "gpu_holders", lambda: [
        {"name": "strata", "engine": "strata", "running": True, "backend": True},
        {"name": "radiance", "engine": "radiance", "running": False, "backend": True}])
    monkeypatch.setattr(engines, "_vram_by_container", lambda: {"strata": 60.0})
    monkeypatch.setattr(engines, "_switch_state", {})      # the module's own; never leak it

    class _Hold:
        def __init__(self, **_k):
            pass

        def start(self):
            pass
    monkeypatch.setattr(engines.threading, "Thread", _Hold)
    ok, msg = engines.activate_engine("radiance", soft=True)
    engines._switch_lock.release()
    assert ok and "stopping strata" in msg and "unloading" not in msg
    assert engines._switch_state["stopped"] == ["strata"] and engines._switch_state["unloaded"] == []


def _stats(*reqs):
    return {"requests": list(reqs), "prefill_tps": 2500.0, "decode_tps": 190.0}


def _run(monkeypatch, stats, log=""):
    async def fake_stats(name, port):
        return stats

    async def fake_props(name, port, model):
        return {"n_ctx": 262144, "total_slots": 4}
    monkeypatch.setattr(speed, "_radiance_stats", fake_stats)
    monkeypatch.setattr(speed, "_props_state", fake_props)
    monkeypatch.setattr(speed, "container_logs", lambda name, tail=0, current_run_only=True: (True, log))
    monkeypatch.setattr(speed, "engine_for", lambda name: "radiance")
    for cache in (speed._speed_cache, speed._rates_cache, speed._last_slots, speed._speed_prev, speed._gen_ema):
        cache.clear()
    return asyncio.run(speed.inference_speed("radiance", 8080, "qwen3.8-flash-next"))


def test_two_decoding_slots_and_two_idle(monkeypatch):
    out = _run(monkeypatch, _stats(
        {"id": 2, "state": "decode", "prompt": 65, "computed": 180, "cached": 0, "output": 116,
         "ctx": 181, "slot": 0, "tps": 108.1},
        {"id": 3, "state": "decode", "prompt": 65, "computed": 171, "cached": 0, "output": 107,
         "ctx": 172, "slot": 1, "tps": 99.7}))
    assert out.state == "generating" and out.live
    assert [s.state for s in out.slots] == ["generating", "generating", "idle", "idle"]
    assert round(out.gen_tps, 1) == 207.8          # the machine's rate: both slots
    assert out.slots[0].ctx_total == 262144 and out.slots[0].ctx_used == 181
    assert out.ctx_total == 4 * 262144 and out.cache_known


def test_a_prefill_shows_its_progress(monkeypatch):
    out = _run(monkeypatch, _stats(
        {"id": 1, "state": "prefill", "prompt": 33069, "computed": 8192, "cached": 4096, "output": 0,
         "ctx": 33069, "slot": 2, "tps": 0.0}))
    sl = out.slots[2]
    assert out.state == "prefill" and sl.state == "prefill"
    assert (sl.prefill_done, sl.prefill_total, sl.prefill_pct) == (12288, 33069, 37)
    assert out.prefill_tps == 2500.0 and out.prefill_total == 33069 and out.ctx_cached == 4096


def test_idle_shows_the_last_request_from_the_log(monkeypatch):
    log = ("I req chatcmpl-1 done from 172.18.0.1:36364 finish=length prompt=33069 (cache 1069) gen=500 "
           "reason=500 ttft=16.00s 64.0 tok/s 23.58s\n")
    out = _run(monkeypatch, _stats(), log)
    assert out.state == "idle" and not out.live and len(out.slots) == 4
    assert out.gen_tps == 64.0 and out.gen_tokens == 500
    assert out.prefill_tps == 2000.0               # (33069 - 1069) read / 16 s


def _rad(tmp_path, monkeypatch):
    from app import radiance
    from app.config import settings
    monkeypatch.setattr(settings, "models_dir", tmp_path)
    d = tmp_path / "rad"
    d.mkdir()
    (d / "qwen3.8-next-flash-fp8-iq4r-moe.rad").write_bytes(b"x" * 10)
    (d / "qwen3.8-next-flash-fp8-iq4r-moe.args").write_text(
        "# Flash-Next: expert tiering and MTP\n--served-model-name qwen3.8-flash-next-iq4r\n"
        "--tp-wire wht6 --placement expert_tiered   # trailing comment\n--host-pool-mib 12288\n")
    (d / "qwen3.8-27b-fp8.rad").write_bytes(b"y" * 10)
    return radiance, d


def test_the_picker_lists_every_rad_with_its_served_name(tmp_path, monkeypatch):
    radiance, d = _rad(tmp_path, monkeypatch)
    (d / "notes.txt").write_text("not a model")
    assert radiance.read_args("qwen3.8-next-flash-fp8-iq4r-moe") == [
        "--served-model-name", "qwen3.8-flash-next-iq4r", "--tp-wire", "wht6", "--placement",
        "expert_tiered", "--host-pool-mib", "12288"]
    ch = dict(radiance.choices())
    assert set(ch) == {"qwen3.8-next-flash-fp8-iq4r-moe.rad", "qwen3.8-27b-fp8.rad"}
    assert ch["qwen3.8-next-flash-fp8-iq4r-moe.rad"].startswith("qwen3.8-flash-next-iq4r")
    assert "no .args file" in ch["qwen3.8-27b-fp8.rad"]     # served as its stem, shared flags only


def test_select_writes_what_the_entry_point_reads(tmp_path, monkeypatch):
    radiance, d = _rad(tmp_path, monkeypatch)
    ok, msg = radiance.select("qwen3.8-27b-fp8.rad")
    assert ok and "qwen3.8-27b-fp8" in msg and radiance.selected() == "qwen3.8-27b-fp8.rad"
    assert "RADIANCE_MODEL=qwen3.8-27b-fp8.rad\n" in (d / "radiance.env").read_text()
    ok, msg = radiance.select("../etc/passwd.rad")
    assert not ok and radiance.selected() == "qwen3.8-27b-fp8.rad"


def test_the_router_row_sets_radiance_s_next_model(tmp_path, monkeypatch, web):  # noqa: F811
    radiance, d = _rad(tmp_path, monkeypatch)
    monkeypatch.setattr(radiance, "restart_if_idle", lambda docker: "it loads at Radiance's next start")
    r = web.post("/router/default", data={"engine": "radiance", "model": "qwen3.8-next-flash-fp8-iq4r-moe.rad"})
    assert "next start: qwen3.8-flash-next-iq4r" in r.text and "next start" in r.text
    assert radiance.selected() == "qwen3.8-next-flash-fp8-iq4r-moe.rad"
    assert "is not a .rad container" in web.post("/router/default", data={"engine": "radiance", "model": "nope.rad"}).text


def test_a_running_idle_radiance_restarts_and_a_busy_one_waits(monkeypatch):
    from app import radiance
    import httpx

    class C:
        status = "running"
        restarted = False

        def restart(self, timeout=0):
            C.restarted = True

    class D:
        class containers:
            @staticmethod
            def get(name):
                return C()
    monkeypatch.setattr(__import__("threading"), "Thread",
                        lambda target, **k: type("T", (), {"start": lambda self: target()})())
    monkeypatch.setattr(httpx, "get", lambda *a, **k: type("R", (), {"json": lambda self: {"in_flight": 1}})())
    assert "answering a request" in radiance.restart_if_idle(D()) and not C.restarted
    monkeypatch.setattr(httpx, "get", lambda *a, **k: type("R", (), {"json": lambda self: {"in_flight": 0}})())
    assert "restarting Radiance" in radiance.restart_if_idle(D()) and C.restarted


def test_an_image_name_that_says_nothing_is_known_by_its_devices():
    # stilldeadcode/radiance:1.17.0 names no GPU stack; /dev/kfd does. As "unknown" the Overview
    # ranked it below the CPU voice model and led with that while Radiance held both cards.
    amd = {"HostConfig": {"Devices": [{"PathOnHost": "/dev/kfd"}, {"PathOnHost": "/dev/dri"}]}}
    nv = {"HostConfig": {"Devices": [], "DeviceRequests": [{"Driver": "nvidia"}]}}
    assert backends._vendor_from_devices(amd) == "rocm"
    assert backends._vendor_from_devices(nv) == "cuda"
    assert backends._vendor_from_devices({"HostConfig": {}}) == "" and backends._vendor_from_devices(None) == ""


class _Ctr:
    def __init__(self, image):
        self.attrs = {"Config": {"Image": image}}


class _Docker:
    def __init__(self, **by_name):
        outer = by_name

        class Containers:
            def get(self, name):
                return outer[name]
        self.containers = Containers()


def _hub(tags):
    import httpx

    def h(req):
        if str(req.url).startswith("https://hub.docker.com/v2/repositories/stilldeadcode/radiance/tags"):
            return httpx.Response(200, json={"results": [{"name": t} for t in tags]})
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(h))


def test_the_newest_hub_tag_is_the_highest_version():
    from app import updates
    c = _hub(["latest", "1.9.0", "1.17.0", "1.16.0", "sha-19d4e8f"])
    assert updates.latest_hub_tag(c, "stilldeadcode/radiance") == "1.17.0"


def test_a_stopped_radiance_is_known_by_its_image_tag():
    from app import updates
    d = _Docker(radiance=_Ctr("stilldeadcode/radiance:1.16.0"), other=_Ctr("stilldeadcode/radiance:latest"))
    assert updates.hub_image_version(d, "radiance") == "1.16.0"
    assert updates.hub_image_version(d, "other") == "" and updates.hub_image_version(d, "gone") == ""


def test_the_check_row_says_update_and_links_the_codeberg_compare(monkeypatch):
    from app import deepseek_engine, updates
    monkeypatch.setattr(updates, "ENGINES", [e for e in updates.ENGINES if e["engine"] == "radiance"])
    monkeypatch.setattr(updates, "_running_report", lambda engine, container: "")
    monkeypatch.setattr(deepseek_engine, "installed", lambda: [])
    out = updates.check(_Docker(radiance=_Ctr("stilldeadcode/radiance:1.16.0")), _hub(["1.16.0", "1.17.0"]))
    (row,) = out["engines"]
    assert (row["engine"], row["current"], row["latest"], row["status"]) == ("radiance", "1.16.0", "1.17.0", "update")
    assert row["url"] == "https://codeberg.org/StillDeadcode/radiance/compare/v1.16.0...v1.17.0"
    assert "image tag" in row["runbook"]


def test_a_serving_radiance_reports_its_own_version(monkeypatch):
    from app import deepseek_engine, updates
    monkeypatch.setattr(updates, "ENGINES", [e for e in updates.ENGINES if e["engine"] == "radiance"])
    monkeypatch.setattr(updates, "_running_report", lambda engine, container: "1.17.0")
    monkeypatch.setattr(deepseek_engine, "installed", lambda: [])
    (row,) = updates.check(_Docker(radiance=_Ctr("stilldeadcode/radiance:latest")), _hub(["1.17.0"]))["engines"]
    assert row["current"] == "1.17.0" and row["status"] == "current"
