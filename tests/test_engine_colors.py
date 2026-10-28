"""Each engine's colour, and the Now Serving card wearing the active engine's shade."""
from __future__ import annotations

from types import SimpleNamespace

from app.engine_colors import ENGINE_COLORS, engine_tint
from app.web import templates

STATE_COLOURS = {"emerald", "sky", "amber", "red", "slate"}     # the card already means these


def test_every_engine_has_its_own_colour_and_none_is_a_state_colour():
    names = [n for n, _ in ENGINE_COLORS.values()]
    assert len(set(names)) == len(names) and not (set(names) & STATE_COLOURS)
    assert {"strata", "llama", "maya"} <= set(ENGINE_COLORS)


def test_lookup():
    assert engine_tint("Strata") == {"name": "indigo", "rgb": "99,102,241"}
    assert engine_tint("nope") is None and engine_tint(None) is None


def _render(**kw):
    speed = SimpleNamespace(live=False, state="idle", gen_tps=0.0, prefill_tps=0.0, slots=[], solo=None,
                            ctx_used=0, ctx_total=0, ctx_cached=0, prompt_tokens=0, gen_tokens=0,
                            prefill_pct=0, prefill_done=0, prefill_total=0, expert_hit_pct=None,
                            cache_known=False, conv_cache=None, ctx_slot_total=0, ctx_slot_used=0,
                            ctx_slot_pct=0, solo_last_ts=0)
    ctx = dict(request=None, speed=speed, hero_loading=False, hero_loading_s=0.0, hero_others=[],
               hero_shape=None, hero_quant="", hero_size_h="", hero_ctx_cfg="", hero_vram_used=None,
               hero_vram_total=0.0, tps_spark="", tps_peak=0.0, tps_samples=0, pp_spark="", pp_peak=0.0,
               pp_samples=0, cx_spark="", cx_peak=0.0, cx_samples=0, hero_mtp=None, hero_foreign=None,
               hero_asleep=False)
    ctx.update(kw)
    return templates.get_template("_hero.html").render(**ctx)


def test_backend_name_wears_the_active_engines_shade():
    # The card itself no longer washes in the engine's colour: it glows the way the GPU panels
    # above it do. The engine's shade stays on the backend name, where it still says who holds
    # the cards.
    html = _render(hero_model="glm-5.3-flash", hero_backend="maya", hero_engine="maya")
    assert "rgba(217,70,239, 0.85)" in html and "rgba(217,70,239, 0.16)" not in html


def test_no_model_no_tint():
    html = _render(hero_model="", hero_backend="", hero_engine="strata")
    assert "rgba(99,102,241" not in html
