"""Each inference engine's colour - the one place it is decided.

The Now Serving card takes on a light shade of the colour of the engine holding the cards, so which
engine is live reads from across the room before any text does. Chosen away from the colours the
card already spends on STATE (emerald = generating, sky = reading a prompt, amber = loading, red =
failure): an engine tint that matched one of those would make "Strata is live" and "a prompt is
being read" the same picture.

The RGB triples are Tailwind's -500 shades, so a class-based use elsewhere (border-indigo-500/40)
matches exactly. The card applies them as inline rgba, not as classes, because the Tailwind CDN
only generates a class it has seen in the DOM and a colour that first appears mid-swap would render
as nothing.
"""

from __future__ import annotations

from . import engine_registry

ENGINE_COLORS: dict[str, tuple[str, str]] = {n: e.color for n, e in engine_registry.ENGINES.items()}


def engine_tint(engine: str | None) -> dict | None:
    """{"name": tailwind colour, "rgb": "r,g,b"} for an engine, None for an unknown one."""
    hit = ENGINE_COLORS.get((engine or "").strip().lower())
    return {"name": hit[0], "rgb": hit[1]} if hit else None
