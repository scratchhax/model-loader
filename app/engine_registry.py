"""Every inference engine this app drives, declared once.

router_api.ENGINES and SINGLE_MODEL, services.engines.READY_TIMEOUT_S, speed._METRICS_ENGINES,
engine_colors.ENGINE_COLORS and updates.ENGINES are all read from here, and so is the one rule
they share, "is it serving" (`serving()` below). A fourth engine is one more entry.

Order matters: ENGINES[0] is what a request naming no model gets, and /v1/models lists the engine
aliases in this order.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Engine:
    name: str
    label: str
    # Tailwind colour name and its -500 RGB, for the Now Serving tint (see engine_colors).
    color: tuple[str, str]
    # Serves exactly one model and ignores the name it is asked for, so the loaded model IS the
    # answer to the engine alias. llama's router picks by name instead.
    single_model: bool
    # Where "is it serving" is asked, and what a 200 has to say (see serving()).
    ready_path: str
    # Seconds a start gets before it is reported slow.
    ready_timeout_s: float
    # Publishes Strata's /metrics document (live / requests / totals); the rest are read from logs.
    strata_metrics: bool = False
    # Can give its VRAM back without stopping (POST /unload) and take it again (POST /load).
    parks: bool = False
    # Where its upstream releases are and how to read them (updates.py), and its upgrade runbook.
    updates: dict = field(default_factory=dict)
    runbook: str = ""
    # The server it speaks. "strata" for every engine that IS Strata's server, whatever model is
    # behind it: /health's loaded flag, /unload and /load, /slots, the solo path outside the
    # slots. The "strata" engine name itself stays the Qwen install (tuning, pin file, pack
    # catalog, update check); this is only the wire protocol.
    protocol: str = ""
    # What /health's "status" must say for the engine to be serving, for engines whose /health
    # answers before the model is in or says something else while it is not taking requests.
    ready_status: str = ""
    # Pins most of the box's RAM while it runs, so an engine it displaces is STOPPED, never
    # parked: a parked Strata keeps tens of GB resident, and the start would not fit beside it.
    needs_ram: bool = False


ENGINES: dict[str, Engine] = {e.name: e for e in (
    Engine(
        name="strata", label="Strata", color=("indigo", "99,102,241"),
        single_model=True, ready_path="/health", ready_timeout_s=420.0,
        strata_metrics=True, parks=True, protocol="strata",
        updates={"repo": "Niko1221/Strata", "kind": "tags", "image": "ai-lab/strata",
                 "live_tag": "rocm7", "container": "strata"},
        runbook="~/ai-lab/strata/UPGRADING.md"),
    Engine(
        name="llama", label="llama.cpp", color=("orange", "249,115,22"),
        single_model=False, ready_path="/v1/models", ready_timeout_s=420.0,
        updates={"repo": "ggml-org/llama.cpp", "kind": "release", "image": "ghcr.io/ggml-org/llama.cpp"},
        runbook="docker compose pull <service>, then recreate it"),
    # A start from a cold page cache takes ~5 min (Maya warms 43 + 46 GB of expert tiers before
    # its port opens), so 420 s would call a healthy start slow. Maya is a fork of Strata's
    # server and kept its /metrics document field for field.
    Engine(
        name="maya", label="Maya", color=("fuchsia", "217,70,239"),
        single_model=True, ready_path="/health", ready_timeout_s=900.0,
        strata_metrics=True, ready_status="ok",
        updates={"repo": "mw00/project-maya", "kind": "tags", "image": "ai-lab/maya",
                 "live_tag": "rocm7", "container": "maya"},
        runbook="~/ai-lab/maya/UPGRADING.md"),
    # DeepSeek-V4.1-Flash on Strata 0.2.7's own DeepSeek engine (strata-dsv41), in a container
    # of its own. Not the "strata" engine with another model: 0.2.3 and later hand an undeclared
    # tool call back as text, which breaks Open WebUI's tool search, so the Qwen install stays on
    # 0.2.1 and this one runs the newer image beside it. Exclusive like the others; a load reads
    # ~150 GB of experts into the cards and RAM, so it gets Maya's long timeout (upstream measured
    # ~1 min on four R9700s; the router's narration, _NARRATE_MAX_S, has to outlast it).
    Engine(
        name="deepseek", label="DeepSeek", color=("teal", "20,184,166"),
        single_model=True, ready_path="/health", ready_timeout_s=900.0,
        strata_metrics=True, parks=True, protocol="strata",
        runbook="~/ai-lab/strata/UPGRADING.md"),
    # Radiance (codeberg.org/StillDeadcode/radiance): its own C++/HIP server, one .rad model per
    # container. Its port opens once the model is in and /health then says "ok"; a start is ~1 min
    # warm and a few from a cold page cache (it reads a 122 GB file). It serves its own JSON
    # (/stats: every running request with its slot, state and progress), not Strata's /metrics,
    # and it has no park. It pins ~90 GB of RAM (an n-gram table, a host expert pool, a prefix
    # tier), hence needs_ram.
    Engine(
        name="radiance", label="Radiance", color=("rose", "244,63,94"),
        single_model=True, ready_path="/health", ready_timeout_s=600.0,
        protocol="radiance", ready_status="ok", needs_ram=True,
        runbook="docker-compose.yaml's radiance service; trial scripts in ~/ai-lab/radiance"),
)}

NAMES: tuple[str, ...] = tuple(ENGINES)
DEFAULT_READY_TIMEOUT_S = 420.0
# Every engine that is Strata's server underneath (see Engine.protocol).
STRATA_PROTOCOL: tuple[str, ...] = tuple(n for n, e in ENGINES.items() if e.protocol == "strata")


def get(name: str | None) -> Engine | None:
    return ENGINES.get((name or "").strip().lower())


def speaks_strata(engine: str | None) -> bool:
    """Is this engine Strata's server (whatever model it runs)?"""
    return (engine or "").strip().lower() in STRATA_PROTOCOL


def serving(engine: str, status_code: int, body) -> bool:
    """Does this answer from `ready_path` mean the engine can take a request now?

    A 200 means different things per engine. llama's /v1/models only answers once its router is
    up (its /health says 503 "Loading model" meanwhile). Strata's /v1/models answers 200 with the
    model listed "unloaded" while it starts or sits parked, so it is asked /health, whose
    {"loaded": true} settles it. Maya's /health says "ok" only once the model is in; "updating"
    and "reloading" are up but not taking requests.
    """
    if status_code != 200:
        return False
    if speaks_strata(engine):
        return isinstance(body, dict) and bool(body.get("loaded"))
    spec = get(engine)
    if spec and spec.ready_status:
        return isinstance(body, dict) and str(body.get("status") or "") == spec.ready_status
    return True


def speaks_radiance(engine: str | None) -> bool:
    """Is this engine Radiance's server (its /stats document, its log lines)?"""
    spec = get(engine)
    return bool(spec and spec.protocol == "radiance")
