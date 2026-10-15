"""One endpoint for everything that needs inference on this box.

THE PROBLEM THIS SOLVES. Clients bind to model IDs, and on this host the ID is not stable. The
engines are mutually exclusive, so switching from llama.cpp to Strata changes every usable ID at
once; switching which pack Strata runs changes it again
(`qwen3.8-flash-next-abliterated-iq3_s` one day, `...unsloth-ud-iq4_xs` the next). Anything that
saved a model name - an Open WebUI workspace model, an IDE config - breaks on every switch.

SO THE ALIAS IS THE ENGINE NAME. `strata` and `llama` are always valid and always mean "whatever
that engine currently has", which is the thing that does not change. The real IDs are still
advertised and still work, so naming a specific model when you mean a specific model costs you
nothing.

WHAT THIS DELIBERATELY DOES NOT DO. It never starts, stops or switches an engine. Lifecycle
belongs to the live Model Loader, which knows about VRAM drain and exclusivity; a second thing
driving the same two cards is how this setup already produced crash loops. When the engine a
request asks for is not the one holding the cards, the router says so and stops.

WHY A DEFAULT MODEL IS PER ENGINE AND MEANS TWO DIFFERENT THINGS. llama.cpp's router loads on
demand, so naming a default is enough to get a model. Strata cannot: its loaded model is whatever
the container started with, and changing it needs a restart. So a Strata default is a statement
of intent for the next start, not something this can satisfy now - and the router reports that
rather than silently bouncing an engine somebody is using.
"""

from __future__ import annotations

import hashlib
import json
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import db, services
from .config import settings
from .utils import docker_client

app = FastAPI(title="Model Router")

# The engines this router knows how to address, in the order /v1/models lists them.
ENGINES = ("strata", "llama")

# Long on purpose. A 200K-token prompt takes 60-90 s to read on this box before a single token
# comes back, and a default-length client timeout would turn the engine's slowest legitimate
# work into a router error.
_CONNECT_TIMEOUT_S = 5.0
_READ_TIMEOUT_S = 900.0


# --------------------------------------------------------------------------- accounting

def _init_db() -> None:
    """The router's own table. req_timing is shaped around llama.cpp's log lines - instance, task
    id, draft acceptance - and is filled by scraping those logs every 20 s. What a proxy sees is
    a different and better thing: the request as the CLIENT experienced it, for either engine,
    including the failures a log never records. Keeping them apart means neither has to pretend
    to be the other."""
    with db._LOCK, db._conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS router_request (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                REAL NOT NULL,
                conv              TEXT NOT NULL DEFAULT '',
                client            TEXT NOT NULL DEFAULT '',
                asked             TEXT NOT NULL DEFAULT '',
                engine            TEXT NOT NULL DEFAULT '',
                backend           TEXT NOT NULL DEFAULT '',
                model             TEXT NOT NULL DEFAULT '',
                shape             TEXT NOT NULL DEFAULT 'openai',
                stream            INTEGER NOT NULL DEFAULT 0,
                status            INTEGER NOT NULL DEFAULT 0,
                error             TEXT NOT NULL DEFAULT '',
                prompt_tokens     INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                cached_tokens     INTEGER NOT NULL DEFAULT 0,
                ttft_ms           REAL NOT NULL DEFAULT 0,
                total_ms          REAL NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS router_request_ts   ON router_request(ts DESC);
            CREATE INDEX IF NOT EXISTS router_request_conv ON router_request(conv, ts DESC);
            """
        )


def _record(**row) -> None:
    cols = ("ts", "conv", "client", "asked", "engine", "backend", "model", "shape", "stream",
            "status", "error", "prompt_tokens", "completion_tokens", "cached_tokens",
            "ttft_ms", "total_ms")
    vals = [row.get(k, 0 if k.endswith(("_tokens", "_ms", "status", "stream")) else "") for k in cols]
    try:
        with db._LOCK, db._conn() as c:
            c.execute(f"INSERT INTO router_request({','.join(cols)}) "
                      f"VALUES({','.join('?' * len(cols))})", vals)
    except Exception:  # noqa: BLE001 - accounting must never fail a request
        pass


def conversation_key(body: dict) -> str:
    """A stable id for a multi-turn conversation, which the OpenAI protocol does not provide.

    Clients send the whole history every turn and nothing identifies the thread - Open WebUI
    sends no conversation header at all. What IS stable across the turns of one conversation is
    its BEGINNING: the system prompt and first exchange are resent verbatim every time, which is
    also exactly why the engines report such high `reused` token counts. Hashing that prefix
    gives a key that follows a conversation without the client doing anything.

    A client that sets `user` is believed instead, since that is an explicit statement.
    """
    who = str(body.get("user") or "").strip()
    if who:
        return "u:" + hashlib.sha256(who.encode()).hexdigest()[:16]
    msgs = body.get("messages") or []
    head = json.dumps(msgs[:2], sort_keys=True)[:2000] if msgs else json.dumps(body.get("prompt"))[:2000]
    return "p:" + hashlib.sha256(head.encode("utf-8", "replace")).hexdigest()[:16]


# --------------------------------------------------------------------------- discovery

def _backends() -> list[dict]:
    """Running inference containers, as {name, engine, url, models}.

    Discovered the same way the UI discovers them - the `ai-lab.engine` label, falling back to
    the llama.cpp image name - so the router and the app can never disagree about what exists.
    """
    cli = docker_client()
    if cli is None:
        return []
    out: list[dict] = []
    try:
        for c in cli.containers.list():
            if c.status != "running":
                continue
            labels = (c.attrs.get("Config") or {}).get("Labels") or {}
            engine = labels.get("ai-lab.engine") or ""
            image = ((c.attrs.get("Config") or {}).get("Image") or "")
            if not engine:
                if "llama.cpp" not in image and "llama" not in c.name:
                    continue
                engine = "llama"
            _, port = services._extract_ports(c.attrs)
            if port is None:
                continue
            out.append({"name": c.name, "engine": engine,
                        "url": f"http://{c.name}:{port}", "models": []})
    except Exception:  # noqa: BLE001 - a docker hiccup is "nothing found", never a 500
        return []
    return out


async def _models_of(client: httpx.AsyncClient, b: dict) -> list[str]:
    try:
        r = await client.get(f"{b['url']}/v1/models", timeout=5.0)
        if r.status_code == 200:
            return [m.get("id", "") for m in (r.json().get("data") or []) if m.get("id")]
    except httpx.HTTPError:
        pass
    return []


async def _live_map() -> tuple[list[dict], dict[str, dict]]:
    """(backends with their models, {model_id: backend})."""
    bs = _backends()
    by_model: dict[str, dict] = {}
    async with httpx.AsyncClient() as client:
        for b in bs:
            b["models"] = await _models_of(client, b)
            for m in b["models"]:
                by_model.setdefault(m, b)
    return bs, by_model


def default_for(engine: str) -> str:
    return db.get_setting(f"router.default.{engine}", "")


# --------------------------------------------------------------------------- routing

class Unresolved(Exception):
    """No backend can serve this request, with a reason meant for a human."""


async def resolve(asked: str) -> tuple[dict, str]:
    """(backend, model to ask it for) for a requested name, or raise Unresolved.

    `asked` is either an engine alias - the stable thing - or a real model id.
    """
    bs, by_model = await _live_map()
    if not bs:
        raise Unresolved("no inference backend is running")

    if asked in ENGINES:
        live = [b for b in bs if b["engine"] == asked]
        if not live:
            other = ", ".join(sorted({b["engine"] for b in bs})) or "none"
            raise Unresolved(
                f"{asked} is not the engine holding the cards right now (running: {other}). "
                f"The engines here are exclusive; switch them in Model Loader.")
        b = live[0]
        # Strata serves exactly one model and ignores the name it is given, so whatever it has
        # loaded IS the answer. llama's router picks by name, so a default has to be chosen.
        if b["engine"] == "strata":
            return b, (b["models"][0] if b["models"] else asked)
        want = default_for("llama")
        if want and want in by_model:
            return by_model[want], want
        if want:
            raise Unresolved(f"the default llama model {want!r} is not offered by {b['name']}")
        if len(b["models"]) == 1:
            return b, b["models"][0]
        raise Unresolved("no default model is set for llama — choose one in the Router tab")

    if asked in by_model:
        return by_model[asked], asked
    raise Unresolved(f"no running backend serves {asked!r}")


# --------------------------------------------------------------------------- endpoints

@app.on_event("startup")
def _startup() -> None:
    db.init()
    _init_db()


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "backends": [b["name"] for b in _backends()]}


@app.get("/v1/models")
async def list_models() -> dict:
    """Engine aliases first, then every real id.

    Both, not either: the aliases are what a saved config should point at, and the real ids are
    what you use when you mean one specific model. Offering only aliases would make this router
    cost you the ability to say "the voice model".
    """
    bs, _ = await _live_map()
    now = int(time.time())
    data = []
    for e in ENGINES:
        if any(b["engine"] == e for b in bs):
            data.append({"id": e, "object": "model", "created": now, "owned_by": "model-router"})
    for b in bs:
        for m in b["models"]:
            data.append({"id": m, "object": "model", "created": now, "owned_by": b["engine"]})
    return {"object": "list", "data": data}


def _client_of(request: Request) -> str:
    key = (request.headers.get("authorization") or "").replace("Bearer ", "").strip()
    if key:
        return "key:" + hashlib.sha256(key.encode()).hexdigest()[:12]
    return (request.client.host if request.client else "") or "unknown"


async def _forward(request: Request, path: str, shape: str) -> JSONResponse:
    t0 = time.time()
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": {"message": "body must be JSON"}})

    asked = str(body.get("model") or "").strip()
    conv, client_id = conversation_key(body), _client_of(request)
    streaming = bool(body.get("stream"))

    if streaming:
        # Stage 1 does not relay SSE. Saying so is better than quietly answering non-streamed:
        # a client that asked to stream and got one block back looks like a hung model.
        _record(ts=t0, conv=conv, client=client_id, asked=asked, shape=shape, stream=1,
                status=501, error="streaming not implemented yet")
        return JSONResponse(status_code=501, content={"error": {
            "message": "this router does not relay streaming responses yet; retry with "
                       "\"stream\": false, or use the engine directly"}})

    try:
        backend, model = await resolve(asked or ENGINES[0])
    except Unresolved as e:
        _record(ts=t0, conv=conv, client=client_id, asked=asked, shape=shape,
                status=503, error=str(e), total_ms=(time.time() - t0) * 1000)
        return JSONResponse(status_code=503, content={"error": {"message": str(e)}})

    out = dict(body)
    out["model"] = model          # llama picks by name; Strata ignores it
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(_READ_TIMEOUT_S,
                                                           connect=_CONNECT_TIMEOUT_S)) as c:
            r = await c.post(f"{backend['url']}{path}", json=out)
        payload = r.json() if r.content else {}
    except httpx.HTTPError as e:
        _record(ts=t0, conv=conv, client=client_id, asked=asked, engine=backend["engine"],
                backend=backend["name"], model=model, shape=shape, status=502,
                error=f"{type(e).__name__}: {e}", total_ms=(time.time() - t0) * 1000)
        return JSONResponse(status_code=502, content={"error": {"message": f"{backend['name']}: {e}"}})

    usage = (payload.get("usage") or {}) if isinstance(payload, dict) else {}
    details = usage.get("prompt_tokens_details") or {}
    _record(ts=t0, conv=conv, client=client_id, asked=asked, engine=backend["engine"],
            backend=backend["name"], model=model, shape=shape, status=r.status_code,
            prompt_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            cached_tokens=int(details.get("cached_tokens") or 0),
            total_ms=(time.time() - t0) * 1000)
    return JSONResponse(status_code=r.status_code, content=payload)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _forward(request, "/v1/chat/completions", "openai")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _forward(request, "/v1/completions", "openai")


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Strata serves the Anthropic shape as well, and the IDE here has used it."""
    return await _forward(request, "/v1/messages", "anthropic")
