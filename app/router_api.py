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

import asyncio
import hashlib
import itertools
import json
import threading
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import db, narrate, services
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
        # The kill switch. Off, the router still routes and only the bookkeeping stops, so a
        # misbehaving write can be taken out of the path in one click instead of a config edit
        # and a restart while people are using it.
        if db.get_setting("router.record", "1") == "0":
            return
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
    raw = msgs[:2] if msgs else body.get("prompt")
    if not raw:
        # Nothing to key on - a malformed or empty body. Hashing "null" would give every such
        # request the SAME key, which silently merges unrelated traffic into one conversation:
        # four failed probes and eight real Strata turns appeared as a single 12-turn thread.
        # No key is honest; a shared one is not.
        return ""
    head = json.dumps(raw, sort_keys=True)[:2000]
    return "p:" + hashlib.sha256(head.encode("utf-8", "replace")).hexdigest()[:16]


# --------------------------------------------------------------------------- discovery

def _backends(include_stopped: bool = False) -> list[dict]:
    """Inference containers, as {name, engine, url, models, running}.

    Discovered the same way the UI discovers them - the `ai-lab.engine` label, falling back to
    the llama.cpp image name - so the router and the app can never disagree about what exists.

    `include_stopped` exists for the pin UI and nothing else. The engines here are exclusive, so
    the backend you most want to pin is usually the one that is NOT running: pinning llama to
    the GPU router matters precisely while Strata holds the cards, and offering only running
    containers hid that choice exactly when it was worth making. Routing still considers running
    backends only - a pin to something stopped is refused, loudly, at request time.
    """
    cli = docker_client()
    if cli is None:
        return []
    out: list[dict] = []
    try:
        for c in cli.containers.list(all=include_stopped):
            running = c.status == "running"
            if not running and not include_stopped:
                continue
            labels = (c.attrs.get("Config") or {}).get("Labels") or {}
            engine = labels.get("ai-lab.engine") or ""
            image = ((c.attrs.get("Config") or {}).get("Image") or "")
            if not engine:
                if "llama.cpp" not in image and "llama" not in c.name:
                    continue
                engine = "llama"
            _, port = services._extract_ports(c.attrs)
            if port is None and running:
                continue              # a running backend with no reachable port is unusable
            out.append({"name": c.name, "engine": engine, "running": running,
                        "url": f"http://{c.name}:{port}" if port else "", "models": []})
    except Exception:  # noqa: BLE001 - a docker hiccup is "nothing found", never a 500
        return []
    return out


async def _ready(client: httpx.AsyncClient, b: dict) -> bool:
    """Can this backend serve a request RIGHT NOW, not merely is its container up?

    The two are different and the gap is ninety seconds wide. docker reports `running` the
    moment the process starts, and the engines then take their time: Strata reads ~55 GB of
    experts before it will answer, and a llama router opens its port before any model is in.

    The question is per-engine, the same split `services._wait_ready` makes and for the same
    reason. On Strata it is /health, because /v1/models answers 200 with the model listed as
    "unloaded" - a 200 there can mean "up and serving nothing", which is exactly the state a
    cold start and a parked server both pass through. /health carries the one bool that settles
    it. On llama it is whether /v1/models answered at all, since that router only opens the port
    once it is ready to take requests and picks the model per request anyway.

    Without this a request arriving during a Strata start was routed INTO the closed port and
    came back 502 "All connection attempts failed", which reads as a broken engine rather than
    one that is still coming up - and it meant no caller could ever be told to wait instead.
    """
    if b["engine"] != "strata":
        return bool(b["models"])
    try:
        r = await client.get(f"{b['url']}/health", timeout=5.0)
        return r.status_code == 200 and bool(r.json().get("loaded"))
    except (httpx.HTTPError, ValueError):
        return False


async def _models_of(client: httpx.AsyncClient, b: dict) -> list[str]:
    try:
        r = await client.get(f"{b['url']}/v1/models", timeout=5.0)
        if r.status_code == 200:
            return [m.get("id", "") for m in (r.json().get("data") or []) if m.get("id")]
    except httpx.HTTPError:
        pass
    return []


def _rank(b: dict) -> tuple:
    """Preference order when more than one backend offers the same model.

    llama-cpu and llama-voice both serve the whole of models.ini, so most ids are offered by
    several backends and "whichever docker listed first" is not an answer - it can quietly send
    a 27B model to CPU inference and look like the GPU has gone slow. A CPU backend is therefore
    always last, and a purpose-named one (voice, asr) comes after the general router, so the
    default is the thing a person would have picked.
    """
    name = b["name"].lower()
    return (("cpu" in name), any(k in name for k in ("voice", "asr", "embed")), name)


async def _live_map() -> tuple[list[dict], dict[str, dict]]:
    """(backends with their models and readiness, {model_id: the PREFERRED READY backend}).

    `ready` is carried rather than filtered out, because the two callers want different things:
    /v1/models keeps advertising an engine alias that is merely mid-start, so it does not
    disappear from a client's dropdown and come back during a swap, while resolve() refuses to
    route to it. Dropping it here would have done both at once.
    """
    bs = sorted(_backends(), key=_rank)
    by_model: dict[str, dict] = {}
    async with httpx.AsyncClient() as client:
        for b in bs:
            b["models"] = await _models_of(client, b)
            b["ready"] = await _ready(client, b)
            if not b["ready"]:
                continue
            for m in b["models"]:
                by_model.setdefault(m, b)      # first wins, and the list is ranked
    return bs, by_model


def default_for(engine: str) -> str:
    return db.get_setting(f"router.default.{engine}", "")




def last_model_for(engine: str) -> str:
    """The model this engine last served successfully, or "".

    Read from the router's own request log rather than kept as a setting, because the log is
    already the record of what happened and a parallel copy would be one more thing that can
    disagree with it. Only 200s count: a model that failed to load is not what you were using.
    """
    try:
        with db._conn() as c:
            row = c.execute(
                "SELECT model FROM router_request WHERE engine = ? AND status = 200 "
                "AND model <> '' ORDER BY ts DESC LIMIT 1", (engine,)).fetchone()
        return row["model"] if row else ""
    except Exception:  # noqa: BLE001 - a log read must never be the reason routing fails
        return ""
def pinned_backend(engine: str) -> str:
    """The container an engine's traffic must go to, or "" for automatic.

    Automatic ranks CPU last and purpose-named servers after the general router, which is a
    guess dressed as a rule: llama-cpu and llama-voice both serve the whole of models.ini, so
    with the GPU router stopped "llama" lands on the VOICE server - it answers, slowly, and
    nothing says why. A name is not a statement of intent, so this lets you make one.
    """
    return db.get_setting(f"router.backend.{engine}", "")


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
        live = [b for b in bs if b["engine"] == asked and b.get("ready")]
        if not live:
            # A container of this engine that is up but not SERVING yet gets its own answer.
            # Lumping it in with "not the engine holding the cards" sent people to flip a switch
            # that was already flipped, and routing to it returned 502 from a closed port, which
            # reads as a broken engine rather than one still coming up.
            coming = [b["name"] for b in bs if b["engine"] == asked and not b.get("ready")]
            if coming:
                raise Unresolved(
                    f"{asked} is starting up ({', '.join(coming)} is running but not serving "
                    f"yet). Nothing is wrong; it is not ready.")
            other = ", ".join(sorted({b["engine"] for b in bs if b.get("ready")})) or "none"
            raise Unresolved(
                f"{asked} is not the engine holding the cards right now (running: {other}). "
                f"The engines here are exclusive; switch them in Model Loader.")
        pin = pinned_backend(asked)
        if pin:
            # A pin is an instruction, so a missing pinned backend is an error rather than a
            # quiet fallback. Falling back would send traffic somewhere the user explicitly did
            # not choose and say nothing - which is the behaviour the pin exists to stop.
            match = [b for b in live if b["name"] == pin]
            if not match:
                raise Unresolved(f"{asked} is pinned to {pin}, which is not running "
                                 f"(available: {', '.join(b['name'] for b in live)})")
            live = match
        b = live[0]
        # Strata serves exactly one model and ignores the name it is given, so whatever it has
        # loaded IS the answer. llama's router picks by name, so a default has to be chosen.
        if b["engine"] == "strata":
            return b, (b["models"][0] if b["models"] else asked)
        want, src = default_for("llama"), "default"
        if not want:
            # Fall back to whatever llama last actually served. "Pick the engine and get the
            # model I was using" is the behaviour a model dropdown implies, and requiring a
            # default to be configured first turned the first use of the alias into an error
            # message instead of an answer.
            want, src = last_model_for("llama"), "last-used"
        if want and want in b["models"]:
            return b, want            # honour the pin: ask THIS backend for the default
        if want and want in by_model:
            return by_model[want], want
        if want:
            raise Unresolved(f"the {src} llama model {want!r} is not offered by {b['name']}")
        if len(b["models"]) == 1:
            return b, b["models"][0]
        raise Unresolved("no default model is set for llama and it has not served anything yet "
                         "— choose a default in the Router tab")

    if asked in by_model:
        return by_model[asked], asked
    raise Unresolved(f"no running backend serves {asked!r}")


# --------------------------------------------------------------------------- endpoints

@app.on_event("startup")
async def _startup() -> None:
    db.init()
    _init_db()
    _init_rollup()

    async def _compactor():
        # Hourly, and once at startup. Compaction is cheap when there is nothing old enough to
        # fold, so running it often costs nothing and means a box left off for a month catches
        # up on its own rather than needing a maintenance step nobody remembers.
        while True:
            try:
                await asyncio.to_thread(compact)
            except Exception:  # noqa: BLE001 - housekeeping must never take the router down
                pass
            await asyncio.sleep(3600)

    asyncio.create_task(_compactor())


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
    seen: set[str] = set()
    for b in bs:
        for m in b["models"]:
            if m in seen:
                continue              # several backends serve one models.ini; list each id once
            seen.add(m)
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
    want = asked or ENGINES[0]
    conv, client_id = conversation_key(body), _client_of(request)
    streaming = bool(body.get("stream"))

    try:
        backend, model = await resolve(want)
    except Unresolved as e:
        # Only a streamed OpenAI-shaped request can be told to wait, because that is the only
        # shape with somewhere to put the explanation. Everything else keeps the 503, which now
        # at least says why no switch happened.
        narratable = streaming and shape == "openai"
        detail = str(e)

        if narratable and await swap_in_flight_for(want):
            # A switch is ALREADY running toward this engine, whoever started it. Describe it.
            return narrated_response(body, path, shape, conv, client_id, want, t0)

        if narratable:
            ok, why = await may_autoswitch(want)
            if ok:
                started, msg = await request_switch(want)
                if started:
                    # The switch is running. narrated_response describes it from here and
                    # relays the answer into the same response when the engine is up.
                    return narrated_response(body, path, shape, conv, client_id, want, t0)
                detail = f"{detail} Tried to switch automatically but {msg}"
            else:
                detail = f"{detail} Not switching automatically: {why}"

        _record(ts=t0, conv=conv, client=client_id, asked=asked, shape=shape,
                status=503, error=detail, total_ms=(time.time() - t0) * 1000)
        return JSONResponse(status_code=503, content={"error": {"message": detail}})

    out = dict(body)
    out["model"] = model          # llama picks by name; Strata ignores it

    if streaming:
        return await _forward_stream(out, backend, model, path, shape, conv, client_id, asked, t0)

    token = _inflight_add(backend["engine"])
    try:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(_READ_TIMEOUT_S,
                                                               connect=_CONNECT_TIMEOUT_S)) as c:
                r = await c.post(f"{backend['url']}{path}", json=out)
            payload = r.json() if r.content else {}
        except httpx.HTTPError as e:
            _record(ts=t0, conv=conv, client=client_id, asked=asked, engine=backend["engine"],
                    backend=backend["name"], model=model, shape=shape, status=502,
                    error=f"{type(e).__name__}: {e}", total_ms=(time.time() - t0) * 1000)
            return JSONResponse(status_code=502,
                                content={"error": {"message": f"{backend['name']}: {e}"}})

        usage = (payload.get("usage") or {}) if isinstance(payload, dict) else {}
        details = usage.get("prompt_tokens_details") or {}
        _record(ts=t0, conv=conv, client=client_id, asked=asked, engine=backend["engine"],
                backend=backend["name"], model=model, shape=shape, status=r.status_code,
                prompt_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens")
                                      or usage.get("output_tokens") or 0),
                cached_tokens=int(details.get("cached_tokens") or 0),
                total_ms=(time.time() - t0) * 1000)
        return JSONResponse(status_code=r.status_code, content=payload)
    finally:
        # Every exit, or a failed non-streamed request leaves a phantom in flight that vetoes
        # every future automatic switch.
        _inflight_drop(token)


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


# --------------------------------------------------------------------------- streaming

def _usage_from_sse(line: str, shape: str, acc: dict) -> None:
    """Pull token counts out of a relayed SSE line, without buffering the stream.

    Counting tokens here instead would mean owning a tokenizer, and the cost of getting that
    wrong is on display elsewhere on this box: Open WebUI estimates with cl100k_base against a
    model whose vocabulary is 248,320, and reports roughly four times the real figure. The
    engine already knows the answer, so the engine is the only honest source.
    """
    if not line.startswith("data:"):
        return
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return
    try:
        d = json.loads(payload)
    except ValueError:
        return
    if shape == "anthropic":
        u = (d.get("message") or {}).get("usage") or d.get("usage") or {}
        if u.get("input_tokens"):
            acc["prompt_tokens"] = int(u["input_tokens"])
        if u.get("output_tokens"):
            acc["completion_tokens"] = int(u["output_tokens"])
        return
    u = d.get("usage") or {}
    if u:
        acc["prompt_tokens"] = int(u.get("prompt_tokens") or acc.get("prompt_tokens") or 0)
        acc["completion_tokens"] = int(u.get("completion_tokens") or acc.get("completion_tokens") or 0)
        acc["cached_tokens"] = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)


async def _relay_gen(out: dict, backend: dict, model: str, path: str, shape: str,
                     conv: str, client_id: str, asked: str, t0: float,
                     t_ready: float | None = None):
    """Relay an SSE response chunk for chunk, and record what it cost.

    A generator rather than a response, so something can be streamed AHEAD of it: a request that
    had to wait for an engine swap narrates the swap first and then relays this, in one response.

    TWO THINGS THIS HAS TO GET RIGHT, both invisible while they work:

    1. Do not buffer. The point of streaming is the first token arriving early; collecting the
       response to inspect it would hand the client one block at the end - the same thing the
       old 501 was avoiding, only slower and pretending to work.

    2. Let a client disconnect reach the engine. When the caller goes away this generator is
       closed, the httpx stream context exits, and the upstream connection drops - which is how
       Strata learns to cancel. Swallowing that would leave a 200K-token prefill burning both
       cards for a minute after somebody pressed stop.

    `t_ready` is when the BACKEND became usable, which is not when the request arrived if the
    client waited through a swap first. TTFT is measured from it, so the number keeps meaning
    "how long until this engine said something", while total_ms still runs from t0 so the wait
    stays visible as the gap between the two. Without the distinction a 90 s cold start reads as
    a 90 s time-to-first-token and the charts accuse the engine of being broken.
    """
    if shape != "anthropic":
        # Ask for the usage chunk. Without it an OpenAI-shaped stream carries no token counts at
        # all, and the accounting for every streamed request - i.e. nearly all of them - is blank.
        opts = dict(out.get("stream_options") or {})
        opts["include_usage"] = True
        out["stream_options"] = opts

    acc: dict = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    state: dict = {"ttft_ms": 0.0, "status": 0, "error": ""}
    first_from = t0 if t_ready is None else t_ready

    pending = b""
    token = _inflight_add(backend["engine"])
    # A streamed response commits to its status code with the first byte, so an engine that dies
    # half way through a generation was recorded as a 200 - which is how a model that crashed
    # twice in a row showed up in the Router tab as two healthy requests, and then got picked by
    # last_model_for() as "the model you were using". A stream is only complete when it says so,
    # and these are the two ways it says it.
    saw_end = False
    client_gone = False
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(_READ_TIMEOUT_S,
                                                           connect=_CONNECT_TIMEOUT_S)) as c:
            async with c.stream("POST", f"{backend['url']}{path}", json=out) as r:
                state["status"] = r.status_code
                async for chunk in r.aiter_bytes():
                    if not chunk:
                        continue
                    if not state["ttft_ms"]:
                        state["ttft_ms"] = (time.time() - first_from) * 1000
                    yield chunk
                    # Split a copy for inspection only; the bytes above already went out
                    # untouched, so nothing here can delay or alter what the client sees.
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        text = line.decode("utf-8", "replace")
                        _usage_from_sse(text, shape, acc)
                        stripped = text.strip()
                        if stripped == "data: [DONE]" or stripped == "event: message_stop":
                            saw_end = True
    except (GeneratorExit, asyncio.CancelledError):
        # The CLIENT went away. Two different exceptions mean the same thing here: GeneratorExit
        # if this generator was closed directly, asyncio.CancelledError if the ASGI server
        # cancelled the task, which is what actually happens on a disconnect. CancelledError is
        # a BaseException, so an earlier version catching only GeneratorExit never saw it and
        # blamed the engine instead: a client that died of its own bug was recorded as
        # "llama-9700x2 ended the stream before finishing" while the model completed normally.
        #
        # So this is recorded as whatever it had reached, not as a failure. Re-raised because
        # swallowing it would leave the upstream connection open and the engine generating into
        # nothing.
        client_gone = True
        raise
    except httpx.HTTPError as e:
        state["error"] = f"{type(e).__name__}: {e}"
        state["status"] = state["status"] or 502
        yield b"data: " + json.dumps(
            {"error": {"message": f"{backend['name']}: {e}"}}).encode() + b"\n\n"
    finally:
        _inflight_drop(token)
        if not saw_end and not client_gone and not state["error"] and state["status"] == 200:
            # The upstream hung up mid-generation. llama.cpp's router logs "http client error:
            # Failed to read connection" on its side when a model instance aborts; this is the
            # same event seen from in front of it.
            state["status"] = 502
            state["error"] = (f"{backend['name']} ended the stream before finishing "
                              f"({model} may have crashed)")
        # Runs on normal completion AND on client disconnect, so a cancelled request is
        # recorded rather than vanishing - those are often the ones worth seeing.
        _record(ts=t0, conv=conv, client=client_id, asked=asked, engine=backend["engine"],
                backend=backend["name"], model=model, shape=shape, stream=1,
                status=state["status"], error=state["error"],
                prompt_tokens=acc["prompt_tokens"],
                completion_tokens=acc["completion_tokens"],
                cached_tokens=acc["cached_tokens"],
                ttft_ms=state["ttft_ms"], total_ms=(time.time() - t0) * 1000)


async def _forward_stream(out: dict, backend: dict, model: str, path: str, shape: str,
                          conv: str, client_id: str, asked: str, t0: float,
                          t_ready: float | None = None):
    return StreamingResponse(
        _relay_gen(out, backend, model, path, shape, conv, client_id, asked, t0, t_ready),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------- narrated waiting
#
# READING ONLY. Nothing in this section starts a switch. The trigger and its policy are
# deliberately not here yet, so the worst this code can do is describe something the UI process
# was already doing on its own. Keeping the POST that moves hardware out of this piece means it
# can be deployed and watched before anything is allowed to evict an engine.

_UI_BASE = "http://127.0.0.1:8090"

# The UI process allows 420 s for a backend to answer after `docker start`, so waiting longer
# than that here would only report on a switch that has already given up.
_NARRATE_MAX_S = 480.0


def narrate_channel() -> str:
    """Which delta field status lines go in. A setting, because the right answer depends on a
    client behaviour nobody here controls: reasoning_content renders as a collapsible block and
    SHOULD be left out of the next turn's history, but whether a given client actually leaves it
    out is its business, not ours. Flip to content if it turns out not to."""
    want = db.get_setting("router.narrate.channel", narrate.CHANNEL_REASONING)
    return want if want in (narrate.CHANNEL_REASONING, narrate.CHANNEL_CONTENT) \
        else narrate.CHANNEL_REASONING


async def switch_state() -> dict:
    """What the UI process says the current switch is doing, or {} if it cannot be asked.

    Over HTTP on purpose - see the endpoint's own docstring. `services._switch_state` in THIS
    process is a different dict that nothing in this process ever writes to, so importing it
    would report an eternal calm while containers stopped.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(4.0)) as c:
            r = await c.get(f"{_UI_BASE}/engines/switch")
            return r.json() if r.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        return {}


_LIVE_PHASES = ("stopping", "draining", "starting", "loading")


async def swap_in_flight_for(asked: str) -> bool:
    """Is a switch ALREADY running, and is it bringing up the engine `asked` names?

    This is the whole of the narration trigger, and it is deliberately the half that cannot
    cause anything: it answers "is the hardware already moving", never "should it move". The
    switch is started by a person clicking the engine control in the UI, and all this does is
    turn the 503 that used to meet them a second later into a description of their own switch.

    Both conditions matter. Without "already running", this would be auto-switching with no
    policy. Without "for this engine", a request for Strata arriving during a switch to llama
    would wait out the whole llama startup only to be told Strata is unavailable, which is
    strictly worse than being told so immediately.

    Only for an engine ALIAS. A specific model id resolves through a different path with its own
    requirement that the backend be running, and a model that a stopped backend would have
    offered is not something this can promise to deliver.
    """
    if asked not in ENGINES:
        return False
    state = await switch_state()
    return (str(state.get("phase") or "") in _LIVE_PHASES
            and str(state.get("engine") or "") == asked)



# --------------------------------------------------------------------------- auto-switch policy
#
# The goal this serves, stated plainly: pick an engine in a client's model dropdown, and that
# engine comes up with its configured default (or the model it last served) and answers. Pick
# the other one, and it hands the cards over and does the same.
#
# Which means this section is the part that MOVES HARDWARE, on a box where the two engines
# cannot share it. So it is gated four ways, and every refusal carries a sentence saying which
# gate stopped it - a silent "no" here looks exactly like a hang.

_DWELL_DEFAULT_S = 60.0

# A streamed request that is abandoned without its generator closing would otherwise sit in the
# in-flight table forever and veto every future switch. Nothing legitimate runs this long, so
# anything older is swept: a leak degrades to a wrong answer for half an hour, not permanently.
_INFLIGHT_MAX_AGE_S = 1800.0

_INFLIGHT: dict[int, tuple[str, float]] = {}
_INFLIGHT_SEQ = itertools.count(1)
_INFLIGHT_LOCK = threading.Lock()


def _inflight_add(engine: str) -> int:
    token = next(_INFLIGHT_SEQ)
    with _INFLIGHT_LOCK:
        _INFLIGHT[token] = (engine, time.time())
    return token


def _inflight_drop(token: int) -> None:
    with _INFLIGHT_LOCK:
        _INFLIGHT.pop(token, None)


def inflight_for(engine: str) -> int:
    """How many requests this process is currently serving on `engine`.

    Accurate because only the router process serves inference; the UI process never forwards.
    Sweeps stale entries on the way through rather than on a timer, so there is no thread to
    get wedged.
    """
    cutoff = time.time() - _INFLIGHT_MAX_AGE_S
    with _INFLIGHT_LOCK:
        for tok in [t for t, (_, started) in _INFLIGHT.items() if started < cutoff]:
            _INFLIGHT.pop(tok, None)
        return sum(1 for eng, _ in _INFLIGHT.values() if eng == engine)


def autoswitch_enabled(engine: str) -> bool:
    return db.get_setting(f"router.autoswitch.{engine}", "") == "1"


def dwell_s() -> float:
    try:
        return max(0.0, float(db.get_setting("router.autoswitch.dwell", "")
                              or _DWELL_DEFAULT_S))
    except ValueError:
        return _DWELL_DEFAULT_S


def _last_switch_at() -> float:
    try:
        return float(db.get_setting("router.autoswitch.last_at", "") or 0.0)
    except ValueError:
        return 0.0


async def may_autoswitch(target: str) -> tuple[bool, str]:
    """Is this request allowed to hand the cards to `target`? (ok, reason it was refused).

    Checked cheapest first, and the order matters: the two that read over HTTP or scan state go
    last so an everyday "it is switched off" costs one settings read.

    There is deliberately NO idle grace here. An earlier draft refused to switch if the current
    engine had served anything in the last few minutes, which is the right rule for protecting
    somebody else's session and the wrong rule for the person who just changed their own
    dropdown twenty seconds after reading an answer - it refused the exact thing the feature
    exists to do. In-flight is the honest test: nothing is being taken from anyone if nothing
    is being served. The grace belongs back here the day a second person's work gets yanked,
    and `conversation_key` is how it would tell "mine" from "theirs".
    """
    if target not in ENGINES:
        return False, f"{target!r} is not an engine name"
    if not autoswitch_enabled(target):
        return False, (f"automatic switching to {target} is off. Turn it on in the Router tab, "
                       f"or switch engines in Model Loader.")

    since = time.time() - _last_switch_at()
    dwell = dwell_s()
    if since < dwell:
        # Without this, two clients that disagree about which engine they want flap the box,
        # each swap killing the model the previous one just loaded.
        return False, (f"a switch finished {int(since)}s ago and the engines settle for "
                       f"{int(dwell)}s. Try again shortly.")

    state = await switch_state()
    if str(state.get("phase") or "") in _LIVE_PHASES:
        return False, f"a switch to {state.get('engine') or 'another engine'} is already running"

    busy = [(e, inflight_for(e)) for e in ENGINES if e != target and inflight_for(e)]
    if busy:
        who = ", ".join(f"{e} ({n} in flight)" for e, n in busy)
        return False, (f"not taking the cards from {who} mid-request. "
                       f"Try again when it is idle.")
    return True, ""


async def request_switch(engine: str) -> tuple[bool, str]:
    """Ask the UI process to hand the cards to `engine`.

    A POST to the same endpoint the engine control uses, for the reason given on that endpoint:
    `_switch_lock` and `_switch_state` are per-process and this container runs two, so the UI
    process owns the switch and everyone else asks. Soft parks a displaced Strata instead of
    stopping it, which returns the VRAM in ~0.3 s and leaves the container and its settings up.
    That makes the RELEASE immediate, which is what lets the engine being asked for start
    sooner. It does not make the return leg quick: a parked Strata asked back took 104 s when
    measured, so a dropdown you change twice costs roughly 45 s one way and 110 s the other.
    """
    db.set_setting("router.autoswitch.last_at", str(time.time()))
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as c:
            r = await c.post(f"{_UI_BASE}/engines/{engine}/activate", data={"soft": "1"})
        if r.status_code != 200:
            return False, f"the engine control refused: HTTP {r.status_code}"
        return True, ""
    except httpx.HTTPError as e:
        return False, f"could not reach the engine control: {type(e).__name__}: {e}"

def swap_progress(asked: str, deadline: float):
    """Build the progress callable narrate_wait() drives: is `asked` routable yet?

    Readiness is "resolve() succeeds", not "the switch says done". The switch reports on
    containers; resolve answers the only question the request actually has, which is whether
    some backend will serve it. On this box those differ: a Strata that is up and parked answers
    /health with loaded:false and is not a usable target.
    """
    async def progress() -> tuple[str, str]:
        try:
            await resolve(asked)
            return "ready", ""
        except Unresolved as e:
            why = str(e)

        state = await switch_state()
        phase = str(state.get("phase") or "")

        if phase == "failed":
            return "failed", narrate.phase_line(state)

        if phase == "done":
            # The switch finished and the engine STILL does not resolve. Waiting cannot fix
            # this - it is a configuration answer, not a timing one, and the most common shape
            # of it is a llama router that started perfectly with no default model chosen. So
            # say both halves: the switch worked, the routing did not.
            return "failed", f"{narrate.phase_line(state)} The request still cannot be " \
                             f"routed: {why}"

        if not (phase in _LIVE_PHASES or phase == "slow"):
            # Nothing is switching and the engine is not routable, so there is nothing to wait
            # for. Saying so beats a spinner that never ends.
            return "failed", f"Nothing is switching and the engine is not available: {why}"

        if time.time() >= deadline:
            return "failed", f"Gave up after {int(_NARRATE_MAX_S)}s. " \
                             f"{narrate.phase_line(state) or 'The switch is still running.'}"

        return "waiting", narrate.phase_line(state)

    return progress


async def narrated_stream(out: dict, path: str, shape: str, conv: str, client_id: str,
                          asked: str, t0: float, channel: str | None = None):
    """One response: the swap narrated, then the real stream relayed into the same body.

    This sends 200 before knowing whether the swap will work, which is the trade narration
    makes. So the failure path records the TRUTH - 503 and the reason - rather than the 200
    already on the wire. Recording the wire status would take the Router tab's failure count to
    zero while things were visibly breaking in the chat.
    """
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(t0)
    result: dict = {}

    async for blob in narrate.narrate_wait(
            swap_progress(asked, t0 + _NARRATE_MAX_S), result,
            cid=cid, model=asked, created=created,
            channel=channel or narrate_channel()):
        yield blob

    if result.get("state") != "ready":
        _record(ts=t0, conv=conv, client=client_id, asked=asked, shape=shape, stream=1,
                status=503, error=str(result.get("text") or "")[:500],
                total_ms=(time.time() - t0) * 1000)
        return

    # Resolve again rather than trusting the probe: between "it resolved" and "we connect" the
    # answer can change, and this is the call whose result is actually used.
    try:
        backend, model = await resolve(asked)
    except Unresolved as e:
        _record(ts=t0, conv=conv, client=client_id, asked=asked, shape=shape, stream=1,
                status=503, error=str(e)[:500], total_ms=(time.time() - t0) * 1000)
        yield narrate.line(cid, created, model=asked, text=f"\n{e}\n",
                           channel=channel or narrate_channel())
        yield narrate.closing(cid, created, asked)
        return

    sent = dict(out)
    sent["model"] = model

    # Close the status off before the model speaks. Both the narration and most of these models'
    # own thinking land in the SAME delta field, so without an explicit terminator the two run
    # together into one sentence - observed verbatim in a real chat as
    # "...port not answering yet.The user just switched models", router status and private
    # reasoning indistinguishable. A closing line plus a blank one gives the handoff a visible
    # seam and says which backend actually took the request, which is the one fact the narration
    # could not state until now.
    if result.get("lines"):
        yield narrate.line(cid, created, asked,
                           f"\nReady, routing to {backend['name']} ({model}).\n\n",
                           channel or narrate_channel())

    async for blob in _relay_gen(sent, backend, model, path, shape, conv, client_id, asked,
                                 t0, time.time()):
        yield blob


def narrated_response(out: dict, path: str, shape: str, conv: str, client_id: str,
                      asked: str, t0: float, channel: str | None = None) -> StreamingResponse:
    return StreamingResponse(
        narrated_stream(out, path, shape, conv, client_id, asked, t0, channel),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------- prefix tolerance
#
# The same four endpoints without the /v1 prefix.
#
# Clients disagree about where the version lives: some want the base URL to END in /v1 and
# append "/models", others take a bare host and append "/v1/models". Open WebUI does the former,
# so a base URL of "http://host:8086" produced GET /models -> 404, surfaced as "Server connection
# failed" during model discovery - a message that says nothing about the cause.
#
# A router is a front door. Being strict here buys nothing and costs a confusing failure at the
# first thing a new client does, so both spellings are served and neither is special.

@app.get("/models")
async def list_models_noprefix() -> dict:
    return await list_models()


@app.post("/chat/completions")
async def chat_completions_noprefix(request: Request):
    return await _forward(request, "/v1/chat/completions", "openai")


@app.post("/completions")
async def completions_noprefix(request: Request):
    return await _forward(request, "/v1/completions", "openai")


@app.post("/messages")
async def anthropic_messages_noprefix(request: Request):
    return await _forward(request, "/v1/messages", "anthropic")


@app.get("/health")
async def health_llamacpp() -> dict:
    """llama.cpp serves /health; clients configured as llama.cpp probe it. Same answer as
    /healthz, under the name that mode expects."""
    return await healthz()


@app.get("/props")
async def props() -> JSONResponse:
    """llama.cpp's /props, relayed from whichever backend would serve a request right now.

    Open WebUI uses this in llama.cpp mode to learn a server's context size and chat template.
    It is ANSWERED FOR ONE BACKEND and cannot be anything else: this router fronts two engines
    with different context lengths, so there is no single honest answer to "what is your n_ctx"
    - only "here is the one that would serve you". The alternative, inventing a merged answer,
    would describe a server that does not exist.

    A client that wants the general picture should talk to this router as an OpenAI endpoint,
    where no such question is asked.
    """
    try:
        backend, _ = await resolve(ENGINES[0])
    except Unresolved:
        try:
            backend, _ = await resolve(ENGINES[1])
        except Unresolved as e:
            return JSONResponse(status_code=503, content={"error": {"message": str(e)}})
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{backend['url']}/props")
        return JSONResponse(status_code=r.status_code, content=r.json() if r.content else {})
    except httpx.HTTPError as e:
        return JSONResponse(status_code=502, content={"error": {"message": f"{backend['name']}: {e}"}})


# --------------------------------------------------------------------------- retention
#
# Detail is worth most when it is fresh. A request from an hour ago is something you might read
# line by line; one from six months ago is only interesting as part of a trend. So the data gets
# coarser with age rather than being kept whole or discarded whole:
#
#   0-14 days     every request, exactly as it happened
#   14-90 days    hourly totals per engine and model - the conversation key is DROPPED, being
#                 the highest-cardinality column, and nobody asks which individual chat ran two
#                 months ago
#   90-365 days   daily totals per engine and model
#   past a year   deleted
#
# Compaction is destructive, which is why each step writes the summary BEFORE deleting what it
# summarised, inside one transaction: a crash halfway leaves duplicated detail, which is
# recoverable, rather than a hole, which is not.
RETAIN_RAW_DAYS = 14
RETAIN_HOURLY_DAYS = 90
RETAIN_TOTAL_DAYS = 365
_DAY = 86400.0

_ROLLUP_UPSERT = (
    "INSERT INTO router_rollup(grain, bucket, engine, model, requests, failures, "
    " prompt_tokens, completion_tokens, cached_tokens, ttft_ms_sum, ttft_n, total_ms_sum) "
    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
    "ON CONFLICT(grain, bucket, engine, model) DO UPDATE SET "
    " requests = requests + excluded.requests, failures = failures + excluded.failures, "
    " prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
    " completion_tokens = completion_tokens + excluded.completion_tokens, "
    " cached_tokens = cached_tokens + excluded.cached_tokens, "
    " ttft_ms_sum = ttft_ms_sum + excluded.ttft_ms_sum, ttft_n = ttft_n + excluded.ttft_n, "
    " total_ms_sum = total_ms_sum + excluded.total_ms_sum"
)


def _init_rollup() -> None:
    with db._LOCK, db._conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS router_rollup (
                grain             TEXT NOT NULL,
                bucket            REAL NOT NULL,
                engine            TEXT NOT NULL DEFAULT '',
                model             TEXT NOT NULL DEFAULT '',
                requests          INTEGER NOT NULL DEFAULT 0,
                failures          INTEGER NOT NULL DEFAULT 0,
                prompt_tokens     INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                cached_tokens     INTEGER NOT NULL DEFAULT 0,
                ttft_ms_sum       REAL NOT NULL DEFAULT 0,
                ttft_n            INTEGER NOT NULL DEFAULT 0,
                total_ms_sum      REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (grain, bucket, engine, model)
            );
            CREATE INDEX IF NOT EXISTS router_rollup_bucket ON router_rollup(bucket DESC);
            """
        )


def compact(now: float | None = None) -> dict:
    """Run the retention tiers. Safe to call repeatedly; returns what it did.

    Every write is an upsert because compaction may cover the same window twice after a restart,
    and a second pass must add nothing rather than double the totals.
    """
    now = now or time.time()
    raw_before = now - RETAIN_RAW_DAYS * _DAY
    hourly_before = now - RETAIN_HOURLY_DAYS * _DAY
    expire_before = now - RETAIN_TOTAL_DAYS * _DAY
    out = {"hourly": 0, "daily": 0, "raw_deleted": 0, "rollup_deleted": 0}
    with db._LOCK, db._conn() as c:
        rows = c.execute(
            "SELECT CAST(ts / 3600 AS INTEGER) * 3600.0, engine, model, COUNT(*), "
            "       SUM(status >= 400), SUM(prompt_tokens), SUM(completion_tokens), "
            "       SUM(cached_tokens), SUM(ttft_ms), SUM(ttft_ms > 0), SUM(total_ms) "
            "FROM router_request WHERE ts < ? GROUP BY 1, engine, model",
            (raw_before,)).fetchall()
        for r in rows:
            c.execute(_ROLLUP_UPSERT, ("hour", r[0], r[1] or "", r[2] or "", r[3], r[4] or 0,
                                       r[5] or 0, r[6] or 0, r[7] or 0, r[8] or 0.0,
                                       r[9] or 0, r[10] or 0.0))
        out["hourly"] = len(rows)
        out["raw_deleted"] = c.execute("DELETE FROM router_request WHERE ts < ?",
                                       (raw_before,)).rowcount

        rows = c.execute(
            "SELECT CAST(bucket / 86400 AS INTEGER) * 86400.0, engine, model, SUM(requests), "
            "       SUM(failures), SUM(prompt_tokens), SUM(completion_tokens), "
            "       SUM(cached_tokens), SUM(ttft_ms_sum), SUM(ttft_n), SUM(total_ms_sum) "
            "FROM router_rollup WHERE grain = 'hour' AND bucket < ? GROUP BY 1, engine, model",
            (hourly_before,)).fetchall()
        for r in rows:
            c.execute(_ROLLUP_UPSERT, ("day", r[0], r[1] or "", r[2] or "", r[3], r[4] or 0,
                                       r[5] or 0, r[6] or 0, r[7] or 0, r[8] or 0.0,
                                       r[9] or 0, r[10] or 0.0))
        out["daily"] = len(rows)
        out["rollup_deleted"] = c.execute(
            "DELETE FROM router_rollup WHERE grain = 'hour' AND bucket < ?",
            (hourly_before,)).rowcount
        out["rollup_deleted"] += c.execute(
            "DELETE FROM router_rollup WHERE bucket < ?", (expire_before,)).rowcount
    return out


def totals(since: float) -> dict:
    """Request and token totals from `since` to now, ACROSS BOTH TIERS.

    A window reaching past the raw retention has to read the rollups too, or a "last 90 days"
    quietly means "last 14" - a wrong answer that looks like a right one.
    """
    out = {"requests": 0, "failures": 0, "prompt_tokens": 0, "completion_tokens": 0,
           "cached_tokens": 0, "total_ms": 0.0, "ttft_ms_sum": 0.0, "ttft_n": 0}
    keys = ("requests", "failures", "prompt_tokens", "completion_tokens", "cached_tokens",
            "total_ms", "ttft_ms_sum", "ttft_n")
    with db._conn() as c:
        for sql, params in (
            ("SELECT COUNT(*), SUM(status >= 400), SUM(prompt_tokens), SUM(completion_tokens), "
             "       SUM(cached_tokens), SUM(total_ms), SUM(ttft_ms), SUM(ttft_ms > 0) "
             "FROM router_request WHERE ts >= ?", (since,)),
            ("SELECT SUM(requests), SUM(failures), SUM(prompt_tokens), SUM(completion_tokens), "
             "       SUM(cached_tokens), SUM(total_ms_sum), SUM(ttft_ms_sum), SUM(ttft_n) "
             "FROM router_rollup WHERE bucket >= ?", (since,)),
        ):
            r = c.execute(sql, params).fetchone()
            if r:
                for k, v in zip(keys, r):
                    out[k] += v or 0
    return out


def retention_state() -> dict:
    """What is actually stored, so the page can describe its own coverage rather than claim it."""
    with db._conn() as c:
        raw_n, raw_oldest = c.execute(
            "SELECT COUNT(*), MIN(ts) FROM router_request").fetchone()
        roll = {g: (n, old) for g, n, old in c.execute(
            "SELECT grain, COUNT(*), MIN(bucket) FROM router_rollup GROUP BY grain")}
    return {"raw_rows": raw_n or 0, "raw_oldest": raw_oldest or 0,
            "hourly_rows": (roll.get("hour") or (0, 0))[0],
            "daily_rows": (roll.get("day") or (0, 0))[0],
            "oldest": min([t for t in [raw_oldest] + [v[1] for v in roll.values()] if t] or [0]),
            "raw_days": RETAIN_RAW_DAYS, "hourly_days": RETAIN_HOURLY_DAYS,
            "total_days": RETAIN_TOTAL_DAYS}


def series(since: float, now: float | None = None) -> dict:
    """Bucketed activity between `since` and now, reading raw rows AND rollups.

    The bucket is chosen from the window rather than fixed, because the same chart has to answer
    "what happened this afternoon" and "what did this year look like": hours for a week or less,
    days beyond. A fixed bucket makes one of those unreadable.

    Returns {"grain", "points": [{t, requests, failures, read, reused, output}]}. `read` is
    prompt tokens MINUS the reused ones - the part the engine actually had to process. Charting
    raw prompt totals would show a long conversation as ever-growing work when the opposite is
    true: almost all of it is cache hits, which is the single most important thing these engines
    do and the reason a 200K-token chat stays fast.
    """
    now = now or time.time()
    span = now - (since or 0.0)
    step = 3600.0 if (since and span <= 8 * _DAY) else _DAY
    grain = "hour" if step == 3600.0 else "day"
    buckets: dict[float, dict] = {}

    def slot(ts: float) -> dict:
        b = (int(ts // step)) * step
        return buckets.setdefault(b, {"t": b, "requests": 0, "failures": 0,
                                      "read": 0, "reused": 0, "output": 0})

    with db._conn() as c:
        for r in c.execute(
                "SELECT ts, status, prompt_tokens, cached_tokens, completion_tokens "
                "FROM router_request WHERE ts >= ?", (since,)):
            d = slot(r[0])
            d["requests"] += 1
            d["failures"] += 1 if (r[1] or 0) >= 400 else 0
            d["reused"] += r[3] or 0
            d["read"] += max(0, (r[2] or 0) - (r[3] or 0))
            d["output"] += r[4] or 0
        for r in c.execute(
                "SELECT bucket, requests, failures, prompt_tokens, cached_tokens, "
                "       completion_tokens FROM router_rollup WHERE bucket >= ?", (since,)):
            d = slot(r[0])
            d["requests"] += r[1] or 0
            d["failures"] += r[2] or 0
            d["reused"] += r[4] or 0
            d["read"] += max(0, (r[3] or 0) - (r[4] or 0))
            d["output"] += r[5] or 0

    if not buckets:
        return {"grain": grain, "points": []}
    # Fill the WHOLE REQUESTED WINDOW, not just the range that happens to contain data. Filling
    # only min..max drew two busy hours as two half-width bars spanning a "last 24 hours" chart -
    # the axis claiming those two hours were the day. Quiet buckets are information: they are
    # what makes a burst look like a burst.
    lo = (int((since or min(buckets)) // step)) * step
    hi = (int(now // step)) * step
    t = lo
    while t <= hi:
        slot(t)
        t += step
    return {"grain": grain, "points": [buckets[k] for k in sorted(buckets)]}
