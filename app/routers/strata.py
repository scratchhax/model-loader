from __future__ import annotations

import asyncio
import os

import httpx
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from .. import hf, services, strata_models
from ..config import settings
from ..downloader import manager
from ..utils import human_bytes
from ..web import templates, _blocked_chip, _QUEUED_CHIP

router = APIRouter()


# ===================================================================== Strata models
#
# The Strata half of Search HF. It is a SEPARATE page rather than a filter on the llama search
# because the two engines disagree about what a model is: llama takes any GGUF file, Strata
# takes a complete shard set of one architecture in one quantization format and prepares an
# expert pack from it. Mixing them into one result list would mean most rows carry a "not for
# this engine" badge, which is noise, not information.

STRATA_CONTAINER = os.environ.get("STRATA_CONTAINER", "strata")
STRATA_ENV_FILE = os.environ.get("STRATA_ENV_PATH", "/strata/strata.env")
# None: wherever the running deployment's /data is (strata.live_data_dir), so the 0.2 volume
# (./strata/data2) is found without configuration. STRATA_DATA_DIR pins it.
STRATA_DATA = os.environ.get("STRATA_DATA_DIR") or None


def _strata_state() -> dict:
    """Catalog, what is installed, and which pairs still have a free pack name."""
    tiers = strata_models.catalog(STRATA_CONTAINER)
    inst = strata_models.installed(STRATA_DATA)
    sel = strata_models.read_selection(STRATA_ENV_FILE)
    prov = strata_models.provenance()
    for tag, d in inst.items():
        d["about"] = strata_models.describe_install(tag, prov)
    return {
        "tiers": tiers,
        "installed": inst,
        "free": {t.install_tag for t in strata_models.free_slots(set(inst), tiers)},
        "selection": sel,
        "engine_up": bool(tiers),
        # {install_tag: repo} so a search row can say whether the slot it wants is held by THIS
        # repo (already installed) or by a different one (genuinely taken).
        "slot_repo": {t: (v.get("repo") or "") for t, v in prov.items()},
        # Which repos anything was installed FROM, independent of the catalog. Deriving this by
        # matching installed tags against catalog tiers looked equivalent and is not: the
        # catalog comes from the RUNNING engine, so rolling the engine back can drop a tier
        # that an installed pack still uses (0.1.40.2 has no swift/IQ3_S; 0.1.43 added it).
        # The pack still runs, so "you have this" must not stop being true because the engine
        # changed underneath it.
        "installed_repos": {(v.get("repo") or "") for v in prov.values() if v.get("repo")},
        # Tuning applies to the model that will RUN next, which is the one the selection names -
        # not whatever happens to be loaded, because a change made now lands on the next start.
        "tuning_tag": _strata_selected_tag(sel),
        "tunables": strata_models.TUNABLES,
    }


def _strata_selected_tag(sel: dict) -> str:
    """The install tag the pin file's FAMILY/MODEL names — the same tag the entry point builds."""
    fam, mod = (sel.get("FAMILY") or "").strip(), (sel.get("MODEL") or "").strip()
    if not mod:
        return ""
    prefix = "" if fam in ("", "qwen") else f"{fam}-"
    return f"{prefix}{mod.lower()}"


@router.post("/strata/tuning", response_class=HTMLResponse)
def strata_tuning(install_tag: str = Form(...), flag: str = Form(...),
                  value: str = Form(...)) -> HTMLResponse:
    ok, msg = strata_models.set_tunable(install_tag, flag, value, STRATA_CONTAINER, STRATA_DATA)
    if not ok:
        return _blocked_chip(msg)
    return HTMLResponse(_QUEUED_CHIP.format(label="Saved", title=msg))


@router.get("/strata/models", response_class=HTMLResponse)
async def strata_models_page(request: Request) -> HTMLResponse:
    st = _strata_state()
    owners = [d["about"]["owner"] for d in st["installed"].values() if d["about"]["owner"]]
    try:
        avatars = await hf.owner_avatars(owners) if owners else {}
    except httpx.HTTPError:
        avatars = {}      # an icon is decoration; never fail the page for one
    metrics = services.strata_metrics(STRATA_CONTAINER)
    st["tuning"] = (strata_models.tuning_state(st["tuning_tag"], metrics, STRATA_DATA)
                    if st["tuning_tag"] else {})
    st["engine_running"] = metrics is not None
    return templates.TemplateResponse("strata_models.html",
                                      {"request": request, "query": "", "avatars": avatars, **st})


@router.get("/strata/search/results", response_class=HTMLResponse)
async def strata_search_results(request: Request, q: str = "") -> HTMLResponse:
    """Kept so an existing link or bookmark still works; the page itself now searches through
    /search?engine=strata, so there is one search box in the app rather than two."""
    return await _strata_results_response(request, q)


async def _strata_results_response(request: Request, q: str = "") -> HTMLResponse:
    """HF search, with every repo classified against the engine's catalog.

    Repos that match nothing are kept and shown as refused WITH THE REASON. Hiding them would
    be worse: the common case is a Qwen3.8-Flash-Next repo in a quantization Strata cannot
    read, and a user who searched for it deserves to be told that rather than shown an empty
    page that looks like a broken search.
    """
    st = _strata_state()
    error: str | None = None
    rows: list[dict] = []
    avatars: dict[str, str] = {}
    if not st["engine_up"]:
        error = ("The Strata container is not running, so its model catalog cannot be read. "
                 "Start Strata from the overview and search again.")
    else:
        try:
            found = await hf.search_models(q.strip() or "Qwen3.8-Flash-Next", limit=25)

            # Classifying a repo needs its FILE LIST, which is a second request per result, and
            # done one at a time that is the whole latency of this page: 25 repos x ~0.5 s was
            # ~13 s of mostly-idle waiting. They are independent, so they go together.
            #
            # Bounded, not unbounded: a bare gather would open 25 simultaneous connections to
            # the Hub per keystroke-driven search, which is how an app earns a 429 and then
            # shows nothing at all. Eight keeps it one round of waiting without being rude.
            sem = asyncio.Semaphore(8)

            async def _detail(m):
                async with sem:
                    try:
                        return m, await hf.repo_detail(m.id)
                    except httpx.HTTPError:
                        return m, None

            # Order is the Hub's relevance order, and gather preserves it. Sorting by
            # installability instead was tempting and wrong: it would bury the repo someone
            # searched for by name under whatever happens to be installable.
            for m, detail in await asyncio.gather(*(_detail(m) for m in found)):
                if detail is None:
                    continue
                files = [{"path": f.path, "size": f.size} for f in detail.files]
                matches = strata_models.classify_repo(files, st["tiers"])
                rows.append({"repo": m, "matches": matches,
                             "reason": "" if matches else _strata_refusal(files)})
            # Owner avatars, the same call and the same cache the llama search uses, so a repo
            # looks like itself on both pages instead of like two different things.
            avatars = await hf.owner_avatars([r["repo"].id.split("/", 1)[0] for r in rows])
        except httpx.HTTPStatusError as e:
            error = f"HF returned HTTP {e.response.status_code}"
        except httpx.HTTPError as e:
            error = f"network error: {e}"
    return templates.TemplateResponse("_strata_results.html", {
        "request": request, "rows": rows, "error": error, "avatars": avatars,
        "installed": st["installed"], "free": st["free"], "slot_repo": st["slot_repo"],
        "installed_repos": st["installed_repos"],
    })


def _strata_refusal(files: list[dict]) -> str:
    """Why a repo matched nothing — named precisely, because "unsupported" teaches nothing."""
    names = [f["path"].rsplit("/", 1)[-1].lower() for f in files if f["path"].lower().endswith(".gguf")]
    if not names:
        return "no GGUF files"
    if not any(strata_models.ARCH_MARKER in n for n in names):
        return "not Qwen3.8-Flash-Next — Strata serves only that architecture"
    if not any(strata_models.GSQ_MARKER in n for n in names):
        return ("Qwen3.8-Flash-Next, but not GSQ-RCO quantized — Strata's pack builder reads "
                "GSQ-RCO files (and Unsloth's own UD-IQ4_XS / UD-Q4_K_XL, which it downloads itself)")
    return "GSQ-RCO, but no complete shard set for a quant tier this engine lists"


@router.post("/strata/install", response_class=HTMLResponse)
async def strata_install(repo_id: str = Form(...), family: str = Form(...),
                         model: str = Form(...)) -> HTMLResponse:
    """Plan the install, place what we already have, and queue only what is missing."""
    st = _strata_state()
    tier = next((t for t in st["tiers"] if t.family == family and t.model == model), None)
    if tier is None:
        return _blocked_chip("that family/tier is not in the engine's catalog")
    if tier.install_tag in st["installed"]:
        return _blocked_chip(f"{tier.install_tag} is already installed — remove it first")

    try:
        detail = await hf.repo_detail(repo_id)
    except httpx.HTTPError as e:
        return _blocked_chip(f"cannot read {repo_id}: {e}")
    files = [{"path": f.path, "size": f.size} for f in detail.files]
    match = next((m for m in strata_models.classify_repo(files, [tier])), None)
    if match is None:
        return _blocked_chip("this repo no longer matches that tier")

    # Content hashes first: without them every reuse check degrades to "same size", which is
    # not good enough to act on (see plan_install).
    paths = [f["src"] for f in match.files] + ([match.mmproj["src"]] if match.mmproj else [])
    etags = await strata_models.fetch_etags(repo_id, paths)
    for f in match.files:
        f["sha"] = etags.get(f["src"], "")
    if match.mmproj:
        match.mmproj["sha"] = etags.get(match.mmproj["src"], "")

    subdir = f"strata-{repo_id.split('/')[-1]}-{model}"
    plan = await asyncio.to_thread(strata_models.plan_install, match,
                                   str(settings.models_dir), subdir)
    await asyncio.to_thread(strata_models.materialize_reuse, plan)

    for f in plan["fetch"]:
        manager.enqueue(repo_id=repo_id, hf_path=f["src"],
                        filename=f"{subdir}/{f['dst']}", total_bytes=f["size"])

    # Recorded at QUEUE time, not on completion: the slot is spoken for from here on, and a
    # download that is interrupted still leaves files whose origin someone will need to know.
    strata_models.record_provenance(tier.install_tag, repo_id, family, model,
                                   tier.canonical_repo)

    saved = plan["saved_bytes"]
    note = f", reusing {human_bytes(saved)} already on disk" if saved else ""
    return HTMLResponse(_QUEUED_CHIP.format(
        label=f"Queued {human_bytes(plan['fetch_bytes'])}",
        title=f"{len(plan['fetch'])} file(s) queued into {subdir}{note}"))


@router.post("/strata/select", response_class=HTMLResponse)
def strata_select(family: str = Form(...), model: str = Form(...),
                  gguf_dir: str = Form("")) -> HTMLResponse:
    """Point the next Strata start at an installed pair. Takes effect on restart, not now."""
    ok, msg = strata_models.set_selection(STRATA_ENV_FILE, family, model, gguf_dir)
    if not ok:
        return _blocked_chip(msg)
    return HTMLResponse(_QUEUED_CHIP.format(label="Saved", title=msg + " — restart Strata to apply"))
