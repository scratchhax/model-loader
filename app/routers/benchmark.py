from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from .. import bench, db, ini, services
from ..fit import _backend_list, _predicted_vram_gb
from ..web import templates

router = APIRouter()


def _bench_ctx(request: Request, flash: str = "") -> dict:
    """Everything the benchmark page needs. Sections are offered rather than GGUF files:
    a benchmark runs against an alias the router can serve, which is what a section is."""
    return {
        "request": request,
        "sections": ini.section_names(),
        "sweep_args": {n: " ".join(bench.sweep_args_for_section(n)) for n in ini.section_names()},
        "prompts": db.list_prompts(),
        "backends": services._effective_container_names(),
        "max_tokens_default": bench.DEFAULT_MAX_TOKENS,
        "max_tokens_ceiling": bench.MAX_TOKENS_CEILING,
        "job": bench.state(),
        "runs": db.bench_runs(limit=15),
        "flash": flash,
    }


@router.get("/benchmark", response_class=HTMLResponse)
def benchmark_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("benchmark.html", _bench_ctx(request))


@router.post("/benchmark/start", response_class=HTMLResponse)
def benchmark_start(request: Request,
                    backend: str = Form(...),
                    aliases: list[str] = Form(default=[]),
                    prompt_ids: list[int] = Form(default=[]),
                    reps: int = Form(3),
                    max_tokens: int = Form(bench.DEFAULT_MAX_TOKENS)) -> HTMLResponse:
    ok, err = bench.start(backend=backend, aliases=aliases, prompt_ids=prompt_ids,
                          reps=reps, max_tokens=max_tokens)
    return templates.TemplateResponse("_bench_progress.html", {
        "request": request, "job": bench.state(), "flash": "" if ok else err,
    })


@router.post("/benchmark/sweep", response_class=HTMLResponse)
def benchmark_sweep(request: Request,
                    backend: str = Form(...),
                    aliases: list[str] = Form(default=[]),
                    n_prompt: int = Form(512),
                    n_gen: int = Form(128),
                    depths: str = Form("0,4096,16384"),
                    reps: int = Form(3)) -> HTMLResponse:
    ok, err = bench.start_sweep(backend=backend, aliases=aliases, n_prompt=n_prompt,
                                n_gen=n_gen, depths=depths, reps=reps)
    return templates.TemplateResponse("_bench_progress.html", {
        "request": request, "job": bench.state(), "flash": "" if ok else err,
    })


@router.get("/benchmark/progress", response_class=HTMLResponse)
def benchmark_progress(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("_bench_progress.html", {
        "request": request, "job": bench.state(), "flash": "",
    })


@router.post("/benchmark/cancel", response_class=HTMLResponse)
def benchmark_cancel(request: Request) -> HTMLResponse:
    bench.cancel()
    return templates.TemplateResponse("_bench_progress.html", {
        "request": request, "job": bench.state(), "flash": "",
    })


@router.get("/benchmark/banner", response_class=HTMLResponse)
def benchmark_banner(request: Request) -> HTMLResponse:
    """Site-wide "a benchmark is running" strip. Polled from every page, so it stays cheap:
    it reads in-memory job state and touches neither docker nor the database."""
    return templates.TemplateResponse("_bench_banner.html", {
        "request": request, "job": bench.state(),
    })


# ---------------------------------------------------------------- a run's detail page

def _median(xs: list[float]) -> float:
    """Median, not mean. Same reasoning as everywhere else here: one request that queued behind
    a model load reads as a fraction of the true rate and drags an average with it."""
    xs = sorted(x for x in xs if x)
    if not xs:
        return 0.0
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def _bench_charts(run_id: int, results: list, sweeps: list) -> dict:
    """Series for the charts, shaped so the template just hands them to Chart.js.

    Cold and contended rows are excluded from every aggregate: one includes a model load, the
    other ran while something else had the GPU, and neither describes steady-state speed. They
    stay in the table, where the badges explain them.
    """
    import json as _json
    clean = [r for r in results if not r["cold"] and not r["contended"] and not r["err"]]

    aliases = sorted({r["alias"] for r in clean})
    prompts = sorted({r["prompt_name"] for r in clean})

    def _pick(alias, prompt, field, digits):
        """Median for one cell, or None when there is nothing clean to report.

        None rather than 0.0: Chart.js skips a null point but draws a zero, and a zero bar on
        a speed chart reads as "this model produced nothing" when the truth is "every row for
        it was cold, contended or failed". A model that would not load looked like a model
        that ran infinitely slowly.
        """
        vals = [r[field] for r in clean
                if r["alias"] == alias and r["prompt_name"] == prompt and r[field]]
        return round(_median(vals), digits) if vals else None

    gen_series = [{"label": p, "data": [_pick(a, p, "gen_tps", 1) for a in aliases]}
                  for p in prompts]
    ttft_series = [{"label": p, "data": [_pick(a, p, "ttft_ms", 0) for a in aliases]}
                   for p in prompts]

    # Depth decay, straight from llama bench. One line per model per test kind, x = depth.
    depths = sorted({w["n_depth"] or 0 for w in sweeps})
    decay: list[dict] = []
    for a in sorted({w["alias"] for w in sweeps}):
        for kind, want_gen in (("tg", True), ("pp", False)):
            pts = []
            for d in depths:
                m = [w["avg_ts"] for w in sweeps
                     if w["alias"] == a and (w["n_depth"] or 0) == d
                     and bool(w["n_gen"]) == want_gen and w["avg_ts"]]
                pts.append(round(m[0], 1) if m else None)
            if any(p is not None for p in pts):
                decay.append({"label": f"{a} {kind}", "data": pts, "kind": kind})

    # Measured VRAM against what autoconfig predicted. The one chart that can falsify the fit
    # maths: everything else here measures speed, which the estimator never claimed to know.
    vram_labels, vram_measured, vram_predicted = [], [], []
    capacity = 0.0
    try:
        for b in _backend_list():
            capacity = max(capacity, float(b.get("vram_gb") or 0))
    except Exception:  # noqa: BLE001
        capacity = 0.0
    for a in aliases:
        peaks = []
        for r in clean:
            if r["alias"] != a:
                continue
            try:
                peaks.append(sum(_json.loads(r["peak_vram_json"] or "[]")))
            except ValueError:
                pass
        if not peaks:
            continue
        vram_labels.append(a)
        vram_measured.append(round(max(peaks), 2))
        vram_predicted.append(_predicted_vram_gb(a))

    return {
        "capacity_gb": round(capacity, 2),
        "aliases": aliases,
        "gen": gen_series,
        "ttft": ttft_series,
        "depths": depths,
        "decay": decay,
        "vram_labels": vram_labels,
        "vram_measured": vram_measured,
        "vram_predicted": vram_predicted,
    }


@router.get("/benchmark/run/{run_id}", response_class=HTMLResponse)
def benchmark_run(request: Request, run_id: int) -> HTMLResponse:
    import json as _json
    ctx = _bench_ctx(request)
    results = db.bench_results(run_id)
    sweeps = db.bench_sweeps(run_id)
    variants = db.bench_variants(run_id)
    ctx.update({
        "run": db.bench_run(run_id),
        "variants": variants,
        "results": results,
        "sweeps": sweeps,
        "charts_json": _json.dumps(_bench_charts(run_id, results, sweeps)),
        "badges": {v["alias"]: db.badges_for(v["alias"]) for v in variants},
    })
    return templates.TemplateResponse("benchmark.html", ctx)
