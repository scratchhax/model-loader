"""The read-only JSON API: serialisation rules, and every endpoint answering on an empty box."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app import hw
from app.routers import api, overview
from app.services.speed import InferenceSpeed, SlotSpeed

from test_routes_smoke import web  # noqa: F401 - the shared empty-box fixture


@dataclass
class _Thing:
    used: float
    total: float
    _secret: str = "x"
    tags: set = field(default_factory=set)

    @property
    def free(self) -> float:
        return self.total - self.used

    @property
    def broken(self) -> float:
        raise RuntimeError("never asked for")


def test_dataclass_gets_fields_and_properties_but_not_private_ones():
    out = api.jsonable(_Thing(used=3.0, total=10.0, tags={"b", "a"}))
    assert out == {"used": 3.0, "total": 10.0, "tags": ["a", "b"], "free": 7.0}


def test_nan_and_infinity_become_null():
    assert api.jsonable({"a": math.nan, "b": math.inf, "c": 1.5}) == {"a": None, "b": None, "c": 1.5}


def test_drawn_values_are_dropped_by_name_and_suffix():
    ctx = {"tps_spark": "M0,0 L1,1", "power_curve": {"line": "M"}, "spark": {}, "ring": [],
           "request": object(), "slot_density": "roomy", "tps_peak": 41.5}
    assert api.jsonable(ctx) == {"tps_peak": 41.5}


def test_paths_rows_and_tuples():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT 1 AS id, 'x' AS name").fetchone()
    assert api.jsonable({"p": Path("/models/a.gguf"), "r": row, "t": (1, 2)}) == {
        "p": str(Path("/models/a.gguf")), "r": {"id": 1, "name": "x"}, "t": [1, 2]}


def test_int_keys_become_strings():
    assert api.jsonable({0: "card0", 1: "card1"}) == {"0": "card0", "1": "card1"}


def test_speed_keeps_slot_properties():
    sp = InferenceSpeed(model="m", slots=[SlotSpeed(index=0, ctx_used=50, ctx_total=200)])
    out = api.jsonable(sp)
    assert out["slots"][0]["ctx_pct"] == 25
    assert out["model"] == "m"


def test_history_point_field_named_like_a_sparkline_survives():
    """gpu_util is a sparkline on the Containers page and a real FIELD on HistoryPoint; only the
    former may be dropped."""
    pt = hw.HistoryPoint(ts=1.0, gpu_util=55.0, vram_used_gb=3.0, cpu_pct=1.0, mem_used_gb=2.0)
    assert api.jsonable(pt)["gpu_util"] == 55.0


def test_status_is_the_hero_context_minus_drawing(web, monkeypatch):  # noqa: F811
    async def hero():
        return {"hero_model": "m", "tps_spark": "M0,0", "tps_curve": {"line": "M"},
                "tps_peak": 30.0, "speed": InferenceSpeed(model="m", gen_tps=30.0, live=True)}
    monkeypatch.setattr(overview, "_hero_context", hero)
    body = web.get("/api/v1/status").json()
    assert body["hero"]["hero_model"] == "m"
    assert body["hero"]["speed"]["gen_tps"] == 30.0
    assert "tps_spark" not in body["hero"] and "tps_curve" not in body["hero"]


def test_downloads_never_carry_the_url_or_paths(web, monkeypatch):  # noqa: F811
    from app.downloader import DownloadJob, manager
    job = DownloadJob(id="j1", repo_id="o/r", filename="m.gguf",
                      url="https://example.invalid/m.gguf?sig=secret",
                      dest_path=Path("/models/m.gguf"), temp_path=Path("/models/.m.part"),
                      total_bytes=100, downloaded_bytes=25, status="downloading")
    monkeypatch.setattr(manager, "snapshot", lambda: [job])
    body = web.get("/api/v1/downloads").json()
    (j,) = body["jobs"]
    assert j["pct"] == 25.0 and j["status"] == "downloading"
    assert "url" not in j and "dest_path" not in j and "temp_path" not in j
    assert "secret" not in str(body)


def test_containers_perf_has_numbers_not_sparklines(web, monkeypatch):  # noqa: F811
    from app.routers import containers
    monkeypatch.setattr(containers, "_perf_by_name", lambda: {"llama": {
        "gpu_util": "M0,0", "vram": "M0,0", "cpu": "M0,0", "mem": "M0,0",
        "cur_gpu_util": 40.0, "imbalance_gb": 1.5}})
    perf = web.get("/api/v1/containers").json()["perf"]["llama"]
    assert perf == {"cur_gpu_util": 40.0, "imbalance_gb": 1.5}


def test_series_carries_each_slots_history(web, monkeypatch):  # noqa: F811
    from app import services
    monkeypatch.setattr(services, "_effective_container_names", lambda: ["llama"])
    monkeypatch.setattr(services, "slot_series", lambda n: {
        -1: {"tps": [(10.0, 40.0)], "ctx": [(10.0, 1050.0)]},
        0: {"tps": [(10.0, 0.0)], "ctx": [(10.0, 512.0)]}})
    slots = web.get("/api/v1/series").json()["backends"]["llama"]["slots"]
    assert slots == {"solo": {"tps": [[10.0, 40.0]], "ctx": [[10.0, 1050.0]]},
                     "0": {"tps": [[10.0, 0.0]], "ctx": [[10.0, 512.0]]}}


@pytest.mark.parametrize("path", [
    "/api/v1", "/api/v1/status", "/api/v1/hero", "/api/v1/cards", "/api/v1/strip",
    "/api/v1/box", "/api/v1/series", "/api/v1/router", "/api/v1/router?range=all",
    "/api/v1/containers", "/api/v1/downloads", "/api/v1/models", "/api/v1/benchmarks",
    "/api/v1/benchmarks/1", "/api/v1/updates", "/api/v1/maya", "/api/v1/engines/switch",
])
def test_every_endpoint_answers_json_on_an_empty_box(web, path):  # noqa: F811
    r = web.get(path)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/json")
    assert isinstance(r.json(), dict)


def test_index_lists_every_get_route(web):  # noqa: F811
    listed = {p.split("?")[0] for p in web.get("/api/v1").json()["endpoints"]}
    routes = {r.path for r in api.router.routes} - {"/api/v1"}
    assert routes == listed
