from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from .. import bench, db, ini, services
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
