"""The routing controls each engine gets: pin, what runs next, auto-switch.

Built here, not in main.py, because three pages show them: the Router tab (every engine) and each
engine's own settings page (Strata, Maya, models.ini for llama), all through _engine_row.html.
Blocking (docker, Maya's configs, install configs on disk), so callers run it in a thread.
"""

from __future__ import annotations

from . import maya as maya_engine
from . import router_api, services, strata_models


def strata_choices() -> list[tuple[str, str]]:
    """[(install tag, label)] for the Strata default — the INSTALLED packs, not the loaded one.

    Strata serves exactly one model at a time, so asking it what it offers returns one id, and a
    dropdown built from that can only ever say "the thing already running". But a Strata default
    is a statement about the NEXT start, so the set to choose from is what is installed. Those
    come from the install configs on the volume, which is also why this still lists them while
    Strata is stopped.
    """
    import json as _json
    from .routers import strata as strata_router
    out: list[tuple[str, str]] = []
    prov = strata_models.provenance()
    for tag, d in strata_models.installed(strata_router.STRATA_DATA).items():
        name = tag
        try:
            name = _json.loads(open(d["config"], encoding="utf-8").read()).get("model_name") or tag
        except (OSError, ValueError):
            pass
        owner = (prov.get(tag, {}).get("repo") or "").split("/")[0]
        real = strata_models.install_label(strata_models.describe_install(tag, prov))
        label = f"{real}  ({owner}, served as {name})" if real else (f"{name}  ({owner})" if owner else name)
        if d.get("missing"):
            label += "  — source files missing"
        out.append((tag, label))
    return out


def rows_context(bs: list[dict]) -> dict:
    """What _engine_row.html needs, for every engine, given the router's live backends."""
    from .routers import strata as strata_router
    per_engine: dict[str, list] = {}
    for b in bs:
        if b["engine"] == "strata":
            continue                       # built from installed packs below, not from /v1/models
        per_engine.setdefault(b["engine"], [])
        for m in b["models"]:
            if m not in per_engine[b["engine"]]:
                per_engine[b["engine"]].append(m)
    # (value, label) pairs, which is what the template unpacks. Every engine but Strata offers
    # its model ids as both; Strata's are built from installed packs below.
    per_engine.setdefault("llama", [])     # always present: the unloaded sections join it below
    for e in list(per_engine):
        per_engine[e] = [(m, m) for m in per_engine[e]]
    # ini sections no running llama backend serves (added after its last start) are offered
    # too, labelled - a dropdown that silently omits them reads as the app losing the model.
    unloaded = set(router_api.unloaded_sections([m for m, _ in per_engine.get("llama", [])]))
    per_engine["llama"] += [(n, n) for n in sorted(unloaded)]
    per_engine["strata"] = strata_choices()
    # Maya's list is its INSTALLED models, read from its config volume, so it is there while the
    # container is stopped - which, the engines taking turns, is most of the time you look.
    # Falls back to whatever a running Maya advertises if the configs cannot be read.
    maya_choices = maya_engine.choices(services._docker_client())
    if maya_choices:
        per_engine["maya"] = maya_choices
    defaults = {e: router_api.default_for(e) for e in router_api.ENGINES}
    # Maya's row shows what it WILL serve: the model of the config its compose command names.
    # router.default.maya was never read by anything, so a choice there was a no-op.
    maya_run_config = maya_engine.run_config_name(services._docker_client())
    maya_runs = next((r["model"] for r in maya_engine.installed(services._docker_client())
                      if r["file"] == maya_run_config), "")
    if maya_runs:
        defaults["maya"] = maya_runs
    elif len(per_engine.get("maya", [])) == 1:
        defaults["maya"] = per_engine["maya"][0][0]
    # Strata's row, likewise: the install strata.env names (the Strata tab can change it), not a
    # stored router.default.strata.
    strata_next = strata_router._strata_selected_tag(strata_models.read_selection(strata_router.STRATA_ENV_FILE))
    if strata_next:
        defaults["strata"] = strata_next
    return {
        "engines": router_api.ENGINES,
        "per_engine": per_engine,
        "unloaded": unloaded,
        "defaults": defaults,
        "maya_run_config": maya_run_config,
        "pins": {e: router_api.pinned_backend(e) for e in router_api.ENGINES},
        "autoswitch": {e: router_api.autoswitch_enabled(e) for e in router_api.ENGINES},
        "autoswitch_dwell": int(router_api.dwell_s()),
        # Stopped ones included ON PURPOSE: the backend worth pinning is usually the one that
        # is not running, because the engines take turns.
        "backends_by_engine": {
            e: [(b["name"], b.get("running", True))
                for b in sorted(router_api._backends(include_stopped=True), key=router_api._rank)
                if b["engine"] == e]
            for e in router_api.ENGINES},
        "live_engines": sorted({b["engine"] for b in bs}),
    }


async def rows_context_async() -> dict:
    import asyncio
    bs, _ = await router_api._live_map()
    return await asyncio.to_thread(rows_context, bs)
