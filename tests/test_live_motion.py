"""The overview's continuous motion and heat colouring.

Charts are drawn in the browser (_live.html) from /api/v1/series and must survive the panels'
HTMX swaps; rings and bars move linearly over their panel's poll interval; levels against a
ceiling are coloured on the heat ramp, tenants keep their kind colours.
"""

from __future__ import annotations

import json
import re

from app import hw
from app.routers import overview
from app.web import HEAT_STOPS, heat, templates

from test_gpu_activity import LLM, _wire
from test_routes_smoke import web  # noqa: F401 - the shared empty-box fixture


class TestHeat:
    def test_the_ends_and_the_stops(self):
        assert heat(0) == "rgb(71,85,105)"          # slate: idle
        assert heat(0.35) == "rgb(99,102,241)"      # indigo
        assert heat(0.65) == "rgb(192,132,252)"     # violet
        assert heat(1) == "rgb(251,146,60)"         # orange: at the ceiling

    def test_it_clamps_and_tolerates_nothing(self):
        assert heat(-3) == heat(0) and heat(7) == heat(1)
        assert heat(None) == heat(0) and heat("x") == heat(0)

    def test_between_two_stops_it_blends(self):
        # Halfway from indigo (99,102,241) to violet (192,132,252).
        r, g, b = map(int, re.findall(r"\d+", heat(0.5)))
        assert (r, g) == (146, 117) and b in (246, 247)

    def test_the_browser_gets_the_same_stops(self):
        assert templates.env.globals["heat_stops"] == [[f, list(c)] for f, c in HEAT_STOPS]


def _cards_html(monkeypatch):
    _wire(monkeypatch)
    ctx = overview._cards_context(LLM)
    return templates.env.get_template("_cards.html").render(**ctx, parked_engines=set())


class TestCards:
    def test_the_compute_chart_is_live_and_survives_the_swap(self, monkeypatch):
        html = _cards_html(monkeypatch)
        m = re.search(r'<div id="gpu-live-0" hx-preserve="true"[^>]*data-live-chart=\'([^\']+)\'', html)
        assert m, "card 0's chart must be a preserved live chart"
        spec = json.loads(m.group(1))
        assert spec == {"src": "card", "key": "0", "window": 900, "ctx": "gpu-live-ctx-0"}

    def test_tenant_kinds_ride_on_the_swapped_wrapper(self, monkeypatch):
        html = _cards_html(monkeypatch)
        m = re.search(r'id="gpu-live-ctx-1" data-kinds="([^"]+)"', html)
        kinds = json.loads(m.group(1).replace("&#34;", '"'))
        assert kinds == {"comfy": "image"}

    def test_the_compute_arc_is_heat_and_moves_for_the_whole_second(self, monkeypatch):
        html = _cards_html(monkeypatch)
        arc = html.split('id="gpu-util-0"')[1].split("/>")[0]
        assert f"stroke: {heat(0.37)}" in arc               # 37% util on the wired card
        assert "duration-1000 ease-linear" in arc and "emerald" not in arc
        assert 'id="gpu-util-num-0" data-count>37<' in html

    def test_vram_segments_move_linearly_too(self, monkeypatch):
        html = _cards_html(monkeypatch)
        seg = html.split('id="gpu-ring-0-llama-gpu0"')[1].split(">")[0]
        assert "duration-1000 ease-linear" in seg


class TestBoxBar:
    def test_power_and_heat_are_live_against_their_ceilings(self, web):  # noqa: F811
        html = web.get("/boxbar").text
        for key in ("power", "temp"):
            assert re.search(rf'id="bar-{key}-live" hx-preserve="true"', html)
            spec = json.loads(re.search(rf'id="bar-{key}-live"[^>]*data-live-chart=\'([^\']+)\'', html).group(1))
            assert spec == {"src": key, "ctx": f"bar-{key}"}
            cell = html.split(f'id="bar-{key}"')[1].split(">")[0]
            assert re.search(r'data-top="\d+(\.\d+)?"', cell)

    def test_rings_are_heat_until_a_threshold_is_crossed(self, monkeypatch, web):  # noqa: F811
        monkeypatch.setattr(hw, "host_disks", lambda _p: [
            {"label": "models", "paths": ["/models"], "free": 1011.0, "total": 2760.0},
            {"label": "docker", "paths": ["/etc/hosts"], "free": 0.0, "total": 5630.0},
        ])
        html = web.get("/boxbar").text
        ok = html.split('id="bar-disk-ring-0"')[1].split("/>")[0]
        full = html.split('id="bar-disk-ring-1"')[1].split("/>")[0]
        assert "style=\"stroke: rgb(" in ok and "text-red" not in ok
        assert "text-red-500" in full and "style=" not in full     # critical wins over heat

    def test_numbers_count(self, web):  # noqa: F811
        html = web.get("/boxbar").text
        for nid in ("bar-power-num", "bar-load-num", "bar-ram-num"):
            assert f'id="{nid}" data-count' in html


def test_the_overview_carries_the_motion_script(web):  # noqa: F811
    html = web.get("/").text
    assert 'const SERIES_URL = "/api/v1/series";' in html
    stops = json.loads(re.search(r"const HEAT = (\[.*?\]);", html).group(1))
    assert stops == [[f, list(c)] for f, c in HEAT_STOPS]


def test_the_hero_charts_follow_the_backend_it_leads_with(web, monkeypatch):  # noqa: F811
    real = overview._hero_context

    async def hero():
        ctx = await real()          # the empty box's standby context, every key present
        ctx.update(hero_model="m", hero_backend="llama-gpu0", hero_engine="llama")
        return ctx
    monkeypatch.setattr(overview, "_hero_context", hero)
    html = web.get("/hero").text
    assert 'id="hero-live-ctx" data-backend="llama-gpu0"' in html
    for metric in ("tps", "ctx"):
        spec = json.loads(re.search(rf'id="hero-live-{metric}" hx-preserve="true"[^>]*data-live-chart=\'([^\']+)\'', html).group(1))
        assert spec == {"src": "backend", "metric": metric, "window": 120, "ctx": "hero-live-ctx"}


def test_settle_runs_on_the_poll_not_a_quick_ease():
    """The rings and bars must not finish early and pause: their transitions take the panel's
    whole poll. Ease-out at 700 ms inside a 1 s poll was the "move, pause, move" complaint."""
    for name in ("_cards.html", "_box_bar.html", "_viz.html", "_hero.html"):
        src = open(f"app/templates/{name}", encoding="utf-8").read()
        assert "transition-all duration-700 ease-out" not in src, name
