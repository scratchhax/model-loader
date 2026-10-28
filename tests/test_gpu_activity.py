"""The card panels: tenants per card, the ComfyUI adapter, and the per-card compute timeline.

Pinned against glados's real shape: a llama backend alone on GPU0, ComfyUI alone on GPU1, both
seen only through host-sensors' nvidia-smi table.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment, FileSystemLoader

from app import gpu_activity as ga
from app import gpu_procs as gp
from app import hw
from app.routers import overview

GLADOS = {
    "gpus": [
        {"index": 0, "bus": "00000000:01:00.0", "name": "NVIDIA GeForce RTX 5070",
         "mem_used_mib": 11716, "mem_total_mib": 12227, "util_pct": 37, "power_w": 120,
         "power_cap_w": 175, "temp_c": 57, "fan_pct": 30},
        {"index": 1, "bus": "00000000:08:00.0", "name": "NVIDIA GeForce RTX 5070",
         "mem_used_mib": 7828, "mem_total_mib": 12227, "util_pct": 0, "power_w": 1,
         "power_cap_w": 175, "temp_c": 52, "fan_pct": 0},
    ],
    "procs": [
        {"pid": 435428, "gpu": 0, "mem_mib": 11704, "sm_pct": 21},
        {"pid": 2886019, "gpu": 1, "mem_mib": 7796, "sm_pct": 0},
    ],
}
OWNERS = {435428: ("llama-gpu0", "llama-server"), 2886019: ("comfy", "python")}

TEMPLATES = Path(__file__).resolve().parents[1] / "app" / "templates"

ACTIVE = {"jobs": [
    {"id": "b", "status": "pending", "create_time": 3000},
    {"id": "a", "status": "in_progress", "create_time": 1000, "execution_start_time": 1500},
    {"id": "c", "status": "pending", "create_time": 4000},
]}
DONE = {"jobs": [
    {"id": "z", "status": "completed", "execution_start_time": 10_000, "execution_end_time": 22_000,
     "preview_output": {"filename": "q_00001_.png", "subfolder": "glados",
                        "metadata": {"width": 1024, "height": 1440}}},
    {"id": "y", "status": "completed", "execution_start_time": 0, "execution_end_time": 10_000},
    {"id": "x", "status": "failed", "execution_start_time": 0, "execution_end_time": 1_000},
    {"id": "w", "status": "completed", "execution_start_time": 0, "execution_end_time": 60_000},
]}


class TestKinds:
    def test_backend_is_llm_whatever_its_name(self):
        assert ga.kind_of("comfy", True) == "llm"

    def test_names(self):
        assert ga.kind_of("comfy", False) == "image"
        assert ga.kind_of("chatterbox-tts", False) == "audio"
        assert ga.kind_of("qwen-asr", False) == "audio"
        assert ga.kind_of("pid 42", False) == "other"


class TestComfy:
    def test_parse(self):
        st = ga.parse_comfy(ACTIVE, DONE)
        assert st.busy and st.running.id == "a" and st.pending == 2
        assert [j.id for j in st.recent] == ["z", "y", "x", "w"]
        assert st.last.filename == "q_00001_.png" and (st.last.width, st.last.height) == (1024, 1440)
        assert st.failed_recent == 1

    def test_typical_is_the_median_of_successes(self):
        # 12, 10 and 60 s succeeded; the 1 s failure is not a render time.
        assert ga.parse_comfy(ACTIVE, DONE).typical_s == 12.0

    def test_progress_is_an_estimate_that_never_finishes(self, monkeypatch):
        st = ga.parse_comfy({"jobs": [{"id": "a", "status": "in_progress",
                                       "execution_start_time": 1_000_000}]}, DONE)
        monkeypatch.setattr(ga.time, "time", lambda: 1_000.0 + 6.0)
        assert st.progress_pct == 50
        monkeypatch.setattr(ga.time, "time", lambda: 1_000.0 + 600.0)
        assert st.progress_pct == 95

    def test_running_job_start_is_inferred_from_the_queue(self):
        # A running job carries create_time only (measured on ComfyUI 0.42). Queued at 5 s,
        # behind work that last ended at 60 s: it started at 60 s.
        st = ga.parse_comfy({"jobs": [{"id": "r", "status": "in_progress", "create_time": 5_000}]}, DONE)
        assert st.running.start_ms == 60_000
        # Queued after the worker went idle: it started when it was queued.
        st = ga.parse_comfy({"jobs": [{"id": "r", "status": "in_progress", "create_time": 90_000}]}, DONE)
        assert st.running.start_ms == 90_000

    def test_no_history_no_estimate(self):
        st = ga.parse_comfy(ACTIVE, {"jobs": []})
        assert st.typical_s == 0.0 and st.progress_pct is None

    def test_idle(self):
        st = ga.parse_comfy({"jobs": []}, DONE)
        assert not st.busy and st.pending == 0 and st.elapsed_s == 0.0

    def test_env_url_wins(self, monkeypatch):
        monkeypatch.setenv("COMFY_URL", "http://example:8188/")
        assert ga._discover_comfy_url("comfy") == "http://example:8188"

    def test_unreachable_is_a_status_not_an_exception(self, monkeypatch):
        monkeypatch.setattr(ga, "comfy_url", lambda c: "http://127.0.0.1:9")
        monkeypatch.setattr(ga, "_get_json", lambda u: (_ for _ in ()).throw(OSError("refused")))
        ga._comfy_cache.clear()
        st = ga.comfy_status("comfy")
        assert not st.ok and st.error == "OSError"


class TestSparklines:
    def _samples(self, shares, util=0.0):
        return [ga.CardSample(ts=float(i), util=util, vram_gb=0, share=s) for i, s in enumerate(shares)]

    def test_too_short(self):
        assert ga.sparklines(self._samples([{}] * 3))["lines"] == []

    def test_idle_tenant_draws_no_line_but_the_card_does(self):
        out = ga.sparklines(self._samples([{}] * 8, util=40.0))
        assert out["lines"] == [] and out["span_s"] == 7
        assert out["card"]["now"] == 40

    def test_one_line_per_tenant_busiest_first(self):
        ga._kinds_seen.update({"llama-gpu0": "llm", "comfy": "image"})
        shares = [{"llama-gpu0": 30}] * 4 + [{"comfy": 90}] * 4
        out = ga.sparklines(self._samples(shares), buckets=4)
        assert [ln["label"] for ln in out["lines"]] == ["comfy", "llama-gpu0"]
        assert [ln["kind"] for ln in out["lines"]] == ["image", "llm"]
        comfy, llama = out["lines"]
        assert comfy["peak"] == 90 and comfy["now"] == 90 and llama["now"] == 0

    def test_two_samples_per_column(self):
        # 8 samples asked into 60 columns would leave most columns empty (a comb).
        out = ga.sparklines(self._samples([{"x": 50}] * 8), buckets=60)
        assert out["lines"][0]["line"].count("C") == 3     # 4 columns -> 3 curve segments

    def test_curve_never_leaves_the_box(self):
        out = ga.sparklines(self._samples([{"x": 0}, {"x": 100}] * 6), buckets=12)
        nums = [float(v) for seg in out["lines"][0]["line"].replace("M", " ").replace("C", " ").split()
                for v in seg.split(",")[1:]]
        assert min(nums) >= ga._PAD - 0.01 and max(nums) <= ga._H - ga._PAD + 0.01


def _wire(monkeypatch, comfy=None):
    backend_card = hw.GpuCard(index=0, name="NVIDIA GeForce RTX 5070", util_pct=37,
                              vram_used_gb=11.44, vram_total_gb=11.94, temp_c=57,
                              power_w=120, bus="00000000:01:00.0")
    st = SimpleNamespace(ok=True, gpu=SimpleNamespace(cards=[backend_card]))
    monkeypatch.setattr(overview.services, "_effective_container_names", lambda: ["llama-gpu0"])
    monkeypatch.setattr(overview.hw, "stats_for", lambda n: st)
    monkeypatch.setattr(overview.hw, "history_for", lambda n: [])
    monkeypatch.setattr(overview.hw, "host_sensors", lambda: {"nvidia": GLADOS})
    monkeypatch.setattr(overview, "_solo_backend", lambda: "")
    monkeypatch.setattr(gp, "card_index_by_gpu_id", lambda *_a, **_k: {})
    monkeypatch.setattr(gp, "host_sensors", lambda: {"nvidia": GLADOS})
    monkeypatch.setattr(gp, "_owner_map", lambda pids: {p: OWNERS[p] for p in pids if p in OWNERS})
    monkeypatch.setattr(gp, "gpu_capable_containers", lambda *_a, **_k: [
        {"name": "comfy", "running": True, "last_gb": 7.6},
        {"name": "quinn-train", "running": False, "last_gb": 9.0}])
    monkeypatch.setattr(overview.services, "engine_for", lambda n: "llama")
    monkeypatch.setattr(overview.services, "gpu_engines", lambda: [], raising=False)
    monkeypatch.setattr(ga, "comfy_status", lambda c: comfy or ga.parse_comfy({"jobs": []}, DONE))
    monkeypatch.setattr(ga, "timeline", lambda i: [])


LLM = {"llama-gpu0": {"model": "gemma-4-12b-it-IQ4_NL", "loading": False, "asleep": False,
                      "state": "generating", "tps": 64.2, "prefill_pct": 0, "slots_busy": 1}}


class TestCardsContext:
    def test_one_panel_per_host_card_each_with_its_tenant(self, monkeypatch):
        _wire(monkeypatch)
        ctx = overview._cards_context(LLM)
        p0, p1 = ctx["panels"]
        assert [r["label"] for r in p0["rows"]] == ["llama-gpu0"]
        assert [r["label"] for r in p1["rows"]] == ["comfy"]
        llama, comfy = p0["rows"][0], p1["rows"][0]
        assert llama["kind"] == "llm" and llama["llm"]["tps"] == 64.2 and llama["working"]
        assert comfy["kind"] == "image" and comfy["comfy"].last.id == "z"
        assert not comfy["working"]                       # resident, idle, not rendering
        assert comfy["eject"]["name"] == "comfy"
        assert [e["name"] for e in ctx["stopped"]] == ["quinn-train"]

    def test_a_render_counts_as_working_between_samples(self, monkeypatch):
        _wire(monkeypatch, comfy=ga.parse_comfy(ACTIVE, DONE))
        p1 = overview._cards_context({})["panels"][1]
        assert p1["working"][0]["label"] == "comfy"

    def test_segments_are_measured_and_name_the_remainder(self, monkeypatch):
        _wire(monkeypatch)
        p0 = overview._cards_context(LLM)["panels"][0]
        labels = [s["label"] for s in p0["segs"]]
        assert labels[0] == "llama-gpu0"
        # 11.44 GB used on the card, 11.43 attributed: too little left over to draw.
        assert "other" not in labels
        # The ring: 11.43 of 11.94 GB, starting at the top, a 2px gap before the free track.
        ring = p0["ring"][0]
        assert ring["offset"] == 0 and abs(ring["dash"] - (overview._RING_C * 11.43 / 11.94 - 2)) < 0.5

    def test_renders(self, monkeypatch):
        _wire(monkeypatch, comfy=ga.parse_comfy(ACTIVE, DONE))
        ctx = overview._cards_context(LLM)
        e = Environment(loader=FileSystemLoader(str(TEMPLATES)))
        html = e.get_template("_cards.html").render(**ctx, parked_engines=set())
        assert "GPU 0" in html and "GPU 1" in html
        assert "64 tok/s" in html and "rendering" in html and "2 queued" in html
        assert "<img" not in html                         # no thumbnails: they blinked on every poll
        assert "quinn-train" in html and "restore" in html

    def test_renders_with_nothing(self):
        e = Environment(loader=FileSystemLoader(str(TEMPLATES)))
        assert "No GPU readings yet" in e.get_template("_cards.html").render(panels=[], now=time.time())
