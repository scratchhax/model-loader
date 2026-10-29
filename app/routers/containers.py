from __future__ import annotations

import asyncio
import html
import os
from datetime import datetime

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, Response

from .. import db, engine_registry, gpu_procs, hw, ini, services, vram_live
from ..config import settings
from .. import strata as strata_engine
from ..downloader import manager
from ..fit import _gpu_tenants, _solo_backend
from ..web import templates, _banner, _toast
from .models import _update_status_map

router = APIRouter()


def _stats_by_name() -> dict:
    return {name: hw.stats_for(name) for name in services._effective_container_names()}


def _perf_by_name() -> dict:
    """Sparkline-ready view of the rolling history the hw sampler already keeps.

    Percentages are pinned to a 0-100 scale and VRAM to the card total, so the lines stay
    comparable between refreshes instead of auto-rescaling to whatever the current window
    happens to contain.
    """
    out: dict[str, dict] = {}
    for name in services._effective_container_names():
        pts = hw.history_for(name)
        if not pts:
            continue
        st = hw.stats_for(name)
        vram_total = st.gpu.vram_total_gb if (st.ok and st.gpu) else 0.0
        span_s = (pts[-1].ts - pts[0].ts) if len(pts) > 1 else 0.0
        mins = int(span_s // 60)
        cards = (st.gpu.cards if (st.ok and st.gpu) else []) or []
        frees = [c.vram_free_gb for c in cards]
        out[name] = {
            "points": len(pts),
            "has_gpu": bool(st.ok and st.gpu),
            "window_label": (f"last {mins}m" if mins >= 1 else f"last {int(span_s)}s"),
            "gpu_util": hw.sparkline([p.gpu_util for p in pts], 100.0),
            "vram": hw.sparkline([p.vram_used_gb for p in pts], vram_total or None),
            "cpu": hw.sparkline([p.cpu_pct for p in pts], None),
            "mem": hw.sparkline([p.mem_used_gb for p in pts], None),
            "cur_gpu_util": pts[-1].gpu_util,
            "cur_vram": pts[-1].vram_used_gb,
            "vram_total": vram_total,
            "cur_cpu": pts[-1].cpu_pct,
            "cur_mem": pts[-1].mem_used_gb,
            # spread between the most and least free card — the number that explains
            # "OOM on one GPU while the other looks fine"
            "imbalance_gb": round(max(frees) - min(frees), 2) if len(frees) > 1 else 0.0,
        }
    return out


@router.get("/containers", response_class=HTMLResponse)
async def containers_page(request: Request) -> HTMLResponse:
    backends = await services.snapshot_llama_backends()
    try:
        ini_mtime = os.path.getmtime(settings.models_ini_path)
    except OSError:
        ini_mtime = 0.0
    return templates.TemplateResponse("containers.html", {
        "request": request, "backends": backends, "stats": _stats_by_name(), "perf": _perf_by_name(),
        "prompts": db.list_prompts(),
        "openwebui": await asyncio.to_thread(services.openwebui_state),
        # Raw model ids a backend reports — what OpenWebUI's model_ids whitelist matches on.
        # Every llama backend serves the same models.ini, so the section names are the list.
        "all_model_ids": sorted(ini.section_names()),
        "ini_stale": {b.name for b in backends
                      if b.status == "running" and getattr(b, "engine", "llama") == "llama"
                      and ini_stale_since(b.started_at, ini_mtime)},
        "strata_pin": _strata_pin_state(backends),
    })


def ini_stale_since(started_at: str, ini_mtime: float) -> bool:
    """True when a backend started before models.ini was last written.

    llama-server reads --models-preset only at startup, so such a backend cannot be serving
    sections added since - the Router page names them as "(not loaded)". Unparseable or
    missing start times answer False: no note beats a wrong note.
    """
    if not started_at or started_at.startswith("0001") or ini_mtime <= 0:
        return False
    try:
        core, _, frac = started_at.partition(".")
        if frac:
            dt = datetime.fromisoformat(f"{core}.{frac.rstrip('Z')[:6]}+00:00")
        else:
            dt = datetime.fromisoformat(core.replace("Z", "+00:00"))
    except ValueError:
        return False
    return dt.timestamp() < ini_mtime


def _strata_pin_state(backends) -> dict | None:
    """Which card Strata is pinned to, which one it is actually running on, and the choices.

    None when there is no Strata backend at all, which is the normal case on a box without
    one and is what keeps the card out of the template.

    `file` and `live` are read from different places deliberately: the file is what the next
    start will use, `live` is what the running container was created with. They are equal
    except between a pin change and the restart that applies it, and saying so is the whole
    point of showing both.
    """
    b = next((x for x in backends if getattr(x, "engine", "llama") == strata_engine.ENGINE), None)
    if b is None:
        return None
    attrs = {}
    own = {}
    try:
        client = services._docker_client()
        if client is not None:
            attrs = client.containers.get(b.name).attrs or {}
            own = services.strata_own_attrs(client)
    except Exception:  # noqa: BLE001 - a missing container just means no live pin to compare
        attrs = {}
    # Exec candidates for the topology read: this app has no /dev/kfd, so the card count comes
    # from a container that does. Any GPU-bearing backend will answer, Strata included.
    cands = [x.name for x in backends if x.status == "running" and x.vendor in ("rocm", "cuda")]
    n = strata_engine.card_count(tuple(cands))
    # The choices, as the values the form posts. Single cards first, then the split, which is
    # offered rather than recommended: a split only pays when no single card holds the experts
    # the model needs (compare the engine's ranked pairs with its expert cache slots).
    choices = [{"value": str(i), "label": f"card {i}"} for i in range(n)]
    if n > 1:
        split = ",".join(str(i) for i in range(n - 1, -1, -1))   # highest first = main card
        choices.append({"value": split, "label": f"both ({split} split)"})
    ts_idle = db.get_setting("strata_idle_unload_s", "")
    arena = hw.strata_arena_gb()
    avail = hw.host_ram_available_gb()
    return {
        "backend": b.name,
        "file": strata_engine.read_pin(),
        # The file IS the live pin while running: the entry point sources it on every start,
        # so any restart applies it, while docker inspect keeps reporting the create-time
        # environment until a recreate.
        "live": strata_engine.read_pin() if b.status == "running" else "",
        "choices": choices,
        "running": b.status == "running",
        "config_note": strata_engine.config_note(
            strata_engine.run_config_path(attrs, own), services.strata_desired_keys()),
        "timeshare": {
            "idle_s": ts_idle,
            "min_free": db.get_setting("strata_min_free_vram_mib", ""),
            "yield_on": db.get_setting("strata_yield_on_load", "") == "1",
        },
        # The reclaim-stall zone: with the arena pinned, RAM the kernel can still take back
        # without touching GPU queues is what MemAvailable says it has. Under a few GB,
        # the next allocation of anyone else on the box is reclaimed FROM THE ENGINE, and
        # ROCm's reaction to that is to suspend KFD queues - the `verify: timed out` stall.
        "reclaim": {"arena": arena, "available": avail,
                    "warn": arena > 0 and 0 < avail < 8.0},
    }


@router.post("/containers/{name}/strata-pin", response_class=HTMLResponse)
def containers_strata_pin(name: str, card: str = Form(...)) -> HTMLResponse:
    """Pin Strata to one card, and restart it so the pin takes effect.

    The restart is part of the action rather than a second button because the file alone
    changes nothing: the engine reads it at start. Leaving the user to notice that is how a
    setting looks applied while the process is still on the old card.
    """
    if services.engine_for(name) != strata_engine.ENGINE:
        return _toast(False, "not a Strata backend", fail="")
    ok, msg = strata_engine.set_pin(card)
    if not ok:
        return _toast(False, msg, fail="")
    services.restart_llama_backend(name)
    return _banner("warn", f"{msg}; restarting {name}. It reloads ~55 GB into RAM, so give it "
                           f"a minute or two before the port answers.")


def _strata_config_toast(ok: bool, msg: str) -> HTMLResponse:
    return _banner("ok" if ok else "fail", msg)


@router.post("/containers/{name}/strata-timeshare", response_class=HTMLResponse)
def containers_strata_timeshare(name: str, idle_s: str = Form(""), min_free_mib: str = Form(""),
                                yield_on: str = Form("")) -> HTMLResponse:
    """Save the time-share settings and write them into the run config now.

    The write happens here, the effect happens at Strata's next start - server.py reads
    these keys when it boots. Nothing restarts Strata from this route, on purpose: the
    button that would apply the settings immediately is also the button that would drop
    whatever model is answering on that box right now, and that is the pin route's
    deliberate, warned-about job, not a side effect of Save.
    """
    if services.engine_for(name) != strata_engine.ENGINE:
        return _strata_config_toast(False, "not a Strata backend")
    vals = {}
    for key, raw in (("strata_idle_unload_s", idle_s.strip()),
                     ("strata_min_free_vram_mib", min_free_mib.strip())):
        if raw and (not raw.isdigit() or int(raw) <= 0):
            return _strata_config_toast(False, f"{key.split('strata_')[-1]}: a positive whole number, or empty for off")
        vals[key] = raw if raw and int(raw) > 0 else ""
    for key, val in vals.items():
        db.set_setting(key, val)
    db.set_setting("strata_yield_on_load", "1" if yield_on == "1" else "")

    keys = services.strata_desired_keys()
    try:
        client = services._docker_client()
        attrs = client.containers.get(name).attrs or {}
        path = strata_engine.run_config_path(attrs, services.strata_own_attrs(client))
    except Exception as e:  # noqa: BLE001 - saved in the db either way; the note on the card covers the rest
        return _strata_config_toast(True, "saved — but the run config could not be reached "
                                          f"({e}); it will be applied at app start")
    if path is None:
        return _strata_config_toast(True, "saved — run config path unknown; it will be applied at app start")
    ok, msg = strata_engine.merge_run_config(path, keys)
    if not ok:
        return _strata_config_toast(True, f"saved — but not in the run config yet: {msg}")
    return _strata_config_toast(True, f"saved — run config: {msg}. Takes effect on {name}'s next start.")


@router.post("/internal/yield")
def internal_yield() -> Response:
    """Strata's before_load endpoint: give up the cards, because a reload is coming.

    No auth, like everything here (LAN-only tool, see the README) - and the worst a stray
    POST does is stop the llama backends, which is a pressable button elsewhere. The hook
    only exists at all while the Strata card has yield switched on, and a failed hook is a
    warning in Strata's log, not a refused load: a box with nothing to yield degrades to
    "load anyway".
    """
    return Response(services.strata_yield() + "\n", media_type="text/plain")


@router.post("/containers/{name}/strata-load", response_class=HTMLResponse)
def containers_strata_load(name: str) -> HTMLResponse:
    """Load Strata's parked model without touching the container.

    The whole point of the park/load pair: a container restart re-reads ~55 GB of experts
    and costs 90 s, while /load comes back from the OS page cache in seconds. Fire-and-
    forget - the endpoint blocks until the model answers - so the card's own poll is what
    shows the flip, and a refusal (usually "the GPU is in use by another program") surfaces
    on the card via services.strata_error().
    """
    if not engine_registry.speaks_strata(services.engine_for(name)):
        return _toast(False, "not a Strata backend")
    ok, msg = services.strata_load(name)
    return _toast(ok, msg)


@router.post("/containers/{name}/strata-unload", response_class=HTMLResponse)
def containers_strata_unload(name: str) -> HTMLResponse:
    """Park Strata: VRAM and pinned RAM back now, container and page cache left warm.

    The soft counterpart of Stop. It refuses (409) while a request is in flight rather than
    pulling the model out from under it, which is the right answer and worth seeing.
    """
    if not engine_registry.speaks_strata(services.engine_for(name)):
        return _toast(False, "not a Strata backend")
    ok, msg = services.strata_unload(name)
    return _toast(ok, msg)


@router.post("/containers/sync-openwebui", response_class=HTMLResponse)
def containers_sync_openwebui() -> HTMLResponse:
    ok, msg = services.sync_openwebui_endpoints()
    return _toast(ok, msg, fail="Sync failed:")


@router.post("/containers/openwebui-align-capabilities", response_class=HTMLResponse)
def containers_openwebui_align() -> HTMLResponse:
    """Make OpenWebUI's per-model capabilities match models.ini, and drop stale records."""
    ok, msg = services.align_openwebui_capabilities()
    return _toast(ok, msg, fail="Align failed:")


@router.post("/containers/openwebui-filter", response_class=HTMLResponse)
async def containers_openwebui_filter(request: Request) -> HTMLResponse:
    """Set which models one OpenWebUI connection offers. No checked boxes = offer all."""
    form = await request.form()
    url = (form.get("url") or "").strip()
    if not url:
        return _toast(False, "No connection specified.", fail="")
    model_ids = [str(v) for v in form.getlist("model_ids") if str(v).strip()]
    # Writes webui.db through docker exec and restarts open-webui (up to 30 s) - off the loop.
    ok, msg = await asyncio.to_thread(services.set_openwebui_model_filter, url, model_ids)
    return _toast(ok, msg, fail="Filter failed:")



@router.post("/models/redownload", response_class=HTMLResponse)
async def models_redownload(name: str = Form(...)) -> HTMLResponse:
    """Queue a fresh copy of every part of one model from the repo it came from (the stale chip).

    It re-downloads the parts the update check actually resolved - local path to repo path, one
    row per shard - rather than guessing a repo layout, and leaves the existing files in place
    until each download finishes: the downloader writes to a temp file and os.replace()s it, so
    a model that is loaded right now keeps serving from the old inode until it is reloaded.
    """
    snap = services.snapshot_models_dir()
    g = next((e for e in snap.ggufs if e.display_name == name), None)
    if g is None:
        return _toast(False, f"{name} is not in the models directory")
    info = _update_status_map(snap).get(name) or {}
    repo_id = info.get("repo_id") or ""
    parts = [(rel, rp) for rel, rp in (info.get("parts") or []) if rp]
    if not repo_id or "/" not in repo_id:
        return _toast(False, f"no Hugging Face repo recorded for {name} — re-download it from Search HF")
    if not parts:
        return _toast(False, f"could not work out which files in {repo_id} {name} came from; "
                             "run Check for updates first")
    by_name = {p.name: p for p in list(g.parts) + list(g.companion_parts)}
    for rel, repo_path in parts:
        p = by_name.get(rel.rsplit("/", 1)[-1])
        try:
            size = p.stat().st_size if p is not None else 0
        except OSError:
            size = 0
        manager.enqueue(repo_id=repo_id, hf_path=repo_path, filename=rel, total_bytes=size)
    n = len(parts)
    return _toast(True, f"queued {n} file{'s' if n != 1 else ''} from {repo_id} — see Downloads")


@router.post("/models/align-capability", response_class=HTMLResponse)
def models_align_capability(section: str = Form(...)) -> HTMLResponse:
    """Align OpenWebUI capabilities for a single models.ini section."""
    ok, msg = services.align_openwebui_capability_for(section)
    return _toast(ok, msg)


@router.post("/models/openwebui-visibility", response_class=HTMLResponse)
async def models_openwebui_visibility(request: Request) -> HTMLResponse:
    """Toggle one model's visibility on one OpenWebUI connection, from the Models page."""
    form = await request.form()
    url = (form.get("url") or "").strip()
    model_id = (form.get("model_id") or "").strip()
    show = str(form.get("show") or "").lower() in ("1", "true", "on", "yes")
    if not url or not model_id:
        return _toast(False, "Missing model or connection.", fail="")
    # Same as the filter route: webui.db write + open-webui restart behind it.
    ok, msg = await asyncio.to_thread(
        services.toggle_openwebui_model, url, model_id, show, sorted(ini.section_names()))
    return _toast(ok, msg)


@router.get("/containers/{name}/dashboard", response_class=HTMLResponse)
async def containers_dashboard(request: Request, name: str) -> HTMLResponse:
    backends = await services.snapshot_llama_backends()
    b = next((x for x in backends if x.name == name), None)
    if b is None:
        return HTMLResponse(f"<span class='text-xs text-slate-500'>unknown backend: {html.escape(name)}</span>")
    # Throughput rides along with the rest of the 2 s poll rather than having its own timer:
    # one refresh, one consistent picture, and no second interval to reason about.
    speed = await services.inference_speed(name, b.internal_port, b.loaded_model)
    # Live "where the VRAM goes" meter, from what this backend actually loaded against
    # measured usage. Only when this backend owns the cards alone (see _solo_backend);
    # None otherwise, and the card draws no meter. A backend with no GPU gets the RAM
    # variant instead: the whole is its own process footprint, not a shared device, so
    # there is no sole-ownership question to fail.
    st = _stats_by_name().get(name)
    vram_breakdown = None
    vb_heading = None
    if st and st.ok:
        if st.gpu and _solo_backend() == name:
            tenants = _gpu_tenants()
            vram_breakdown = vram_live.breakdown(
                name, st.gpu,
                frozenset(i.strip() for i in (b.loaded_model or "").split(",") if i.strip()),
                foreign=[(t.label, t.total_gb) for t in tenants if t.foreign and t.total_gb > 0.005])
        elif not st.gpu and st.container:
            vram_breakdown = vram_live.ram_breakdown(name, st.container)
            if vram_breakdown:
                vb_heading = "Where the RAM goes"
    return templates.TemplateResponse("_container_dashboard.html", {
        "request": request, "b": b, "stats": _stats_by_name(), "perf": _perf_by_name(),
        "speed": speed, "vram_breakdown": vram_breakdown, "vb_heading": vb_heading,
        # A failed /load or /unload (usually the 503 "the GPU is in use by another program")
        # has nowhere else to surface: the load itself is fire-and-forget, so the card that
        # asked is the one that has to say why it is still parked. Clears itself once the
        # model is visibly loaded again.
        "strata_error": services.strata_error(name, loaded=bool(b.loaded_model)),
    })


@router.post("/gpu-tenants/{name}/{action}", response_class=HTMLResponse)
def gpu_tenant_power(name: str, action: str) -> HTMLResponse:
    """Eject (stop) or restore (start) a GPU-capable container that is not a llama backend.

    Stopping rather than calling the service's own unload endpoint, deliberately: every such
    service has a different API, or none, and docker is the one control that works for all of
    them. It also frees MORE - chatterbox's own unload leaves 1.34 GB of HIP context behind
    where stopping the container returns the whole 5.88 GB. The cost is a slower restore,
    measured ~10 s against ~5, which is a trade worth making for something ejected precisely
    because it is not being used.

    Refuses to touch a llama backend: those are the router's, and the Containers page owns them.
    """
    if name in set(services._effective_container_names()):
        return _toast(False, f"{name} is a llama backend - use the Containers page")
    if action not in ("eject", "restore"):
        return _toast(False, f"unknown action {action}")
    ok, msg = gpu_procs.set_container_running(name, start=(action == "restore"))
    return _toast(ok, msg)


@router.get("/engines/switch")
def engines_switch_state() -> dict:
    """What the running (or last) engine switch is doing. {} when none has run.

    JSON rather than a rendered panel, and a GET so the read-only guard is indifferent to it,
    because the caller is not a browser: the router process polls this to narrate a swap into a
    streaming chat response while the client waits.

    It has to be an HTTP read and not an import. `services._switch_state` is a module-level dict
    and `_switch_lock` a threading.Lock, so each process gets its OWN copy - and this container
    runs two, `app.main` on 8090 and `app.router_api` on 8086. The router importing
    activate_engine() directly would mean two schedulers, each holding the only lock it can see,
    racing over the same two cards: the exact failure the read-only guard was written to stop,
    reintroduced inside one container. So the UI process owns the switch and this is how anyone
    else finds out about it.
    """
    return services.engine_switch_state()


@router.post("/engines/{engine}/activate", response_class=HTMLResponse)
def engines_activate(engine: str, soft: str = Form(""), share: str = Form("")) -> HTMLResponse:
    """Hand the cards to one engine: release everything else holding a GPU, then start it.

    The engines here cannot share the hardware, and a llama router at --models-max 1 has no
    unload endpoint on this build, so stopping the container is the only thing that makes it
    give the cards back.

    `soft` (the checkbox on the Engine row) asks backends that support it to unload instead:
    Strata gives the same VRAM and RAM back in ~0.3 s and keeps its container and settings. It
    does NOT make the return trip quick (about as long as a cold start). Hard remains the
    default because a parked server can still be woken by a stray request, and "who owns the
    cards" deserves an answer that cannot change its mind.

    `share` (the Share button, offered only when a running engine demonstrably holds a
    strict subset of the cards) starts this engine and displaces nothing - the cards are
    divided, not handed over.
    """
    ok, msg = services.activate_engine(engine, soft=soft == "1", share=share == "1")
    return _toast(ok, msg)


@router.post("/containers/{name}/power", response_class=HTMLResponse)
def containers_power(name: str, action: str = Form(...)) -> HTMLResponse:
    """Start or stop one backend. The lifecycle control a backend card was missing.

    Restart was the only button, which is fine until two engines take turns on the same cards.
    Kept separate from the engine switch on purpose: this one touches exactly what it is told
    to and nothing else, which is what you want when the question is "is it this container".
    """
    if action not in ("start", "stop"):
        return _toast(False, f"unknown action {action}")
    ok, msg = services.set_backend_running(name, start=(action == "start"))
    return _toast(ok, msg)


@router.post("/containers/{name}/restart", response_class=HTMLResponse)
def containers_restart(name: str) -> HTMLResponse:
    services.restart_llama_backend(name)
    # small chip: replaces the button when hx-swap=outerHTML; ignored when hx-swap=none
    return HTMLResponse(
        f'<span class="inline-flex items-center rounded-md bg-amber-100 dark:bg-amber-950 '
        f'text-amber-800 dark:text-amber-300 px-2.5 py-1 text-xs font-mono">restarting {name}…</span>'
    )


@router.post("/containers/{name}/test", response_class=HTMLResponse)
async def containers_test(name: str, prompt: str = Form(...), max_tokens: int = Form(256)) -> HTMLResponse:
    result = await services.test_prompt(name, prompt, max_tokens=max_tokens)
    if not result.get("ok"):
        return _toast(False, str(result.get("err", "test failed")), fail="")
    reply = (result.get("reply") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    tps = result.get("tokens_per_s")
    tps_bit = f" · {tps} tok/s" if tps else ""
    meta = (
        f'{result["model"]} · {result["completion_tokens"]} tok in {result["elapsed_s"]}s{tps_bit}'
        f' · prompt {result["prompt_tokens"]} tok'
    )
    return HTMLResponse(
        f'<div class="rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/40 p-3">'
        f'<pre class="text-xs whitespace-pre-wrap font-mono">{reply}</pre>'
        f'<div class="mt-2 text-[10px] text-slate-500 dark:text-slate-400 font-mono">{html.escape(meta)}</div>'
        f'</div>'
    )


@router.get("/containers/{name}/logs", response_class=HTMLResponse)
def containers_logs(name: str, q: str = "", level: str = "") -> HTMLResponse:
    ok, out = services.container_logs(name)
    if not ok:
        return HTMLResponse(f"<span class='text-red-600 dark:text-red-400'>{html.escape(out)}</span>")

    if q or level:
        needle = q.lower()
        keep_error = level == "error"
        keep_warn = level in ("warn", "warning")
        filtered: list[str] = []
        for line in out.splitlines():
            low = line.lower()
            if needle and needle not in low:
                continue
            if keep_error and ("err" not in low and "error" not in low and "fatal" not in low):
                continue
            if keep_warn and ("err" not in low and "warn" not in low and "wrn" not in low and "error" not in low and "fatal" not in low):
                continue
            filtered.append(line)
        out = "\n".join(filtered)
        if not out:
            out = f"(no lines match q={q!r} level={level!r})"

    safe = out.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return HTMLResponse(safe or "(no output)")


@router.get("/containers/{name}/diagnose", response_class=HTMLResponse)
def containers_diagnose(request: Request, name: str) -> HTMLResponse:
    """Parse a backend's recent log for a failed-load signature and propose a fix.

    Rendered as a Jinja partial for consistency with every other fragment here — Jinja
    autoescapes the log lines, so no hand-rolled escaping is needed.
    """
    ok, findings, err = services.diagnose_container(name)
    return templates.TemplateResponse("_diagnose_result.html", {
        "request": request, "ok": ok, "err": err, "findings": findings,
    })
