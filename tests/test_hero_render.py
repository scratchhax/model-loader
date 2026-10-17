from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path(__file__).resolve().parents[1] / "app" / "templates"


@pytest.fixture(scope="module")
def env():
    e = Environment(loader=FileSystemLoader(str(TEMPLATES)))
    e.globals["hue"] = lambda s: 0
    e.globals["strata_only"] = lambda b: False
    e.globals["badge_categories"] = []
    e.globals["badge_labels"] = {}
    e.filters["ts"] = lambda v: "2026-10-16 12:00"
    return e


def slot(i, state, tps=0.0, ctx_used=0, ctx_total=262144):
    pct = round(100.0 * ctx_used / ctx_total) if ctx_total else 0
    return NS(index=i, state=state, gen_tps=tps, decoded=0, ctx_used=ctx_used,
              ctx_total=ctx_total, ctx_pct=pct, ctx_cached=0, prefill_pct=0,
              prefill_done=0, prefill_total=0, prompt_tokens=0, task_id=None)


def speed(slots, solo=None, solo_ts=0.0):
    live = [s for s in slots if s.state != "idle"]
    if solo is not None and solo.state != "idle":
        live = live + [solo]
    state = "generating" if any(s.state == "generating" for s in live) else (
        "prefill" if live else "idle")
    return NS(model="m", state=state, gen_tps=0.0, gen_tokens=0, prefill_tps=0.0,
              prefill_pct=0, prompt_tokens=0, prefill_done=0, prefill_total=0,
              ctx_used=sum(s.ctx_used for s in slots), ctx_total=262144 * len(slots),
              ctx_cached=0, live=False, slots=slots, solo=solo, solo_last_ts=solo_ts,
              expert_hit_pct=None)


BASE = dict(hero_model="m", hero_backend="strata", hero_shape=None, hero_quant="",
            hero_size_h="", hero_ctx_cfg="", hero_vram_used=10.0, hero_vram_total=48.0,
            hero_loading=False, hero_loading_s=0.0, hero_loading_gpu=False,
            hero_asleep=False, hero_foreign=[], hero_weights_split=None,
            hero_others=[], hero_mtp=None, tps_spark="", tps_peak=0.0, tps_samples=0,
            pp_spark="", pp_peak=0.0, pp_samples=0, cx_spark="", cx_peak=0.0,
            cx_samples=0, slot_density="roomy", slot_columns=1)

IDLE4 = [slot(i, "idle", ctx_used=u) for i, u in enumerate([1432, 0, 10380, 0])]


def render(env, sp):
    return env.get_template("_hero.html").render(speed=sp, **BASE)


def test_solo_row_idle_shows_last_pass_dimmed(env):
    solo = slot(-1, "idle", tps=65.3, ctx_used=101830)
    solo.decoded = 1591
    out = render(env, speed(IDLE4, solo, 1792174695.0))
    assert "The solo path, idle. Last solo pass: 65.3 tok/s, 1,591 tokens" in out
    assert "bg-slate-700" in out and "2026-10-16 12:00" in out
    assert "border-slate-800" in out  # ctx tile still slate at fill 0


def test_solo_row_live_at_high_fill_turns_tile_rose(env):
    solo = slot(-1, "generating", tps=80.0, ctx_used=249037)
    out = render(env, speed(IDLE4, solo))
    assert "The solo path: this request runs outside" in out
    assert "border-rose-500/40" in out and "text-rose-300" in out
    assert "currently 95%" in out


def test_slot_generating_at_mid_fill_turns_tile_amber(env):
    solo = slot(-1, "idle", tps=65.3, ctx_used=101830)
    solo.decoded = 1591
    hot = slot(2, "generating", tps=30.0, ctx_used=209715)
    out = render(env, speed([IDLE4[0], IDLE4[1], hot, IDLE4[3]], solo, 1792174695.0))
    assert "border-amber-500/40" in out and "currently 80%" in out


def test_llama_lone_active_slot_gets_chip_not_solo_row(env):
    hot = slot(2, "generating", tps=30.0, ctx_used=209715)
    out = render(env, speed([IDLE4[0], IDLE4[1], hot, IDLE4[3]]))
    assert "The only slot generating" in out and "The solo path" not in out


def test_strata_never_soloed_shows_placeholder(env):
    out = render(env, speed(IDLE4, slot(-1, "idle")))
    assert "no solo pass since the app started" in out


def test_live_generation_at_low_fill_is_emerald(env):
    solo = slot(-1, "generating", tps=80.0, ctx_used=78643)
    out = render(env, speed(IDLE4, solo))
    assert "border-emerald-500/40" in out and "text-emerald-300" in out
    assert "currently 30%" in out


def test_token_k_figures_are_decimal_not_1024(env):
    # Slot rows print raw token counts; the k readouts must read as the same
    # number. 111,780 tokens is 111.8k in decimal - the /1024 bytes convention
    # made the readout sit ~2% below the count it describes.
    solo = slot(-1, "idle", tps=65.3, ctx_used=111780)
    solo.decoded = 0
    sp = speed([slot(0, "idle", ctx_used=111780)], solo=solo)
    out = render(env, sp)
    assert "111,780" in out          # raw count in the row/tooltip
    assert ">111.8<" in out          # readout, decimal k
    assert "111.8k of 262k window" in out  # context bar, decimal k
