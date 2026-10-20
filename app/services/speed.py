from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

import httpx
from docker.errors import DockerException, NotFound


from .backends import _extract_ports, _probe_loaded_model, engine_for
from .dockerbase import _docker_client
from .hotpath import _http
from .logs import container_logs
from .stratactl import strata_metrics


@dataclass
class SlotSpeed:
    """One slot's own figures. There is no fixed number of these - see slot_density()."""
    index: int = 0
    state: str = "idle"        # "generating" | "prefill" | "idle"
    gen_tps: float = 0.0       # this slot's own rate, not a share of the total
    decoded: int = 0           # tokens this slot has produced for its current task
    ctx_used: int = 0          # tokens held by this slot
    ctx_total: int = 0         # this slot's share of the KV pool (ctx-size / parallel)
    ctx_cached: int = 0        # of ctx_used, how many came from cache
    prefill_pct: int = 0       # progress through this slot's prompt, 0-100
    # The numerator and denominator that percentage came from. A bare percentage cannot tell a
    # 400-token prompt from an 80,000-token one, and those take 0.3 s and 70 s respectively on
    # this box - the difference between "it is about to answer" and "go and get a coffee".
    prefill_done: int = 0      # prompt tokens read so far, cache hits included
    prefill_total: int = 0     # the whole prompt this slot is working through
    # The prompt this slot's CURRENT task actually arrived with: cache hits plus processed.
    # Deliberately not n_prompt_tokens, which keeps growing while the model generates - it is
    # the slot's whole context, not the prompt. Dividing the (frozen) cache count by the
    # (growing) context made a perfectly cached conversation appear to shed cache as it
    # answered: measured on one task, n_prompt_tokens ran 47,626 -> 51,042 over 90 s while
    # n_prompt_tokens_processed sat still at 46,844.
    prompt_tokens: int = 0
    task_id: int | None = None

    @property
    def ctx_pct(self) -> int:
        """How full this slot is, for the strip's bar. 0 when the budget is unknown."""
        if self.ctx_total <= 0:
            return 0
        return max(0, min(100, int(round(100.0 * self.ctx_used / self.ctx_total))))


@dataclass
class InferenceSpeed:
    """How fast the loaded model is working right now, for the card's speedometer.

    The aggregate fields describe the MACHINE: gen_tps is the sum over generating slots, and
    ctx_used/ctx_total describe the whole KV pool. With parallel = 1, which is most sections,
    that is identical to the single slot's own figures. With more, reporting one slot's rate as
    though it were the machine's was simply wrong - measured on a parallel = 3 model, two slots
    generated at 38.5 and 27.4 tok/s simultaneously (a persistent 40% gap, because speculative
    decoding accepts more drafts on code than on prose), so no single slot's number described
    either the conversation you were watching or the box. Per-slot detail lives in `slots`.
    """
    model: str = ""
    state: str = "idle"        # "generating" | "prefill" | "idle" - busiest slot wins
    gen_tps: float = 0.0       # tokens/s out, summed over generating slots
    gen_tokens: int = 0        # tokens produced in the run these rates came from
    prefill_tps: float = 0.0   # tokens/s in
    prefill_pct: int = 0       # progress through the current prompt, 0-100
    prompt_tokens: int = 0     # the prompts the busy slots arrived with, cache + processed.
                               # The honest denominator for "how much came from cache".
    prefill_done: int = 0      # prompt tokens read so far, summed over prefilling slots
    prefill_total: int = 0     # the whole prompt, summed the same way. 0 when only the log
                               # knows, because the log's total is progress-derived guesswork
    ctx_used: int = 0          # tokens held across all slots
    ctx_total: int = 0         # the whole KV pool: per-slot ctx x slot count
    ctx_cached: int = 0        # of ctx_used, how many came from cache instead of being re-read
    live: bool = False         # rates describe work happening NOW, not the last run
    slots: list = field(default_factory=list)   # list[SlotSpeed], one per live slot
    # Strata's solo path: the request the engine is working on OUTSIDE any batch slot (see
    # _outside_slots). None of `slots` holds it, so the strip needs its own row or the one
    # thing generating has no bar at all. None on every other engine and whenever the work
    # IS in a slot.
    solo: "SlotSpeed | None" = None
    # When the solo figures were last live, so an idle solo row can date what it is showing.
    solo_last_ts: float = 0.0
    # Strata only: the share of routed experts served from the GPU's cache on the last request,
    # which its `done:` line reports. There is no llama.cpp equivalent - llama has no expert
    # cache - so this is None on every other engine and the tile is simply absent.
    #
    # It is the number that decides whether this engine is fast, more than tok/s is: at 99.9%
    # almost nothing crosses PCIe, and on this box PCIe measured 6.6 GB/s against a 0.55
    # pcie_frac default. A drop here is the first sign a context or cache change went too far.
    expert_hit_pct: float | None = None
    # Does this engine report KV cache occupancy? llama does (n_prompt_tokens_cache);
    # Strata does not - its /slots has no cache field and /metrics only answers AFTER a
    # request completes, with no solo/batch attribution. Callers must not read a 0 from
    # Strata as "missed the cache": it means "unknown".
    cache_known: bool = True
    # Strata only: its host-RAM conversation cache, straight from /metrics. Idle conversations'
    # KV is PARKED there (8 GiB budget, 8 slots on this box) and restored on the next turn,
    # which is why an agent session resumes in under a second instead of re-reading its whole
    # history. Evictions count conversations dropped to make room - each one costs a full
    # re-read on the next turn - so a climbing eviction count is the early warning for a
    # context or concurrency change that outgrew the budget. None on other engines, and on a
    # Strata too old to expose the endpoint.
    conv_cache: dict | None = None


def slot_density(n: int) -> str:
    """How much room each slot row gets, chosen from the slot count alone.

    The strip has to survive `parallel = 8` without pushing the hero off the page, and a fixed
    row height cannot do that: 8 comfortable rows measured 128px, far more than the 24px
    sparkline they sit beside. So rows shrink and labels abbreviate as the count grows, and
    slot_columns() wraps them - every slot stays visible at any count, nothing truncates or
    scrolls. 8 slots land at 4 rows of 16px, which is about the height of one readout.
    """
    if n <= 4:
        return "roomy"      # ~22px rows, "433 / 128,768"
    if n <= 8:
        return "tight"      # ~16px rows, "34%"
    return "dense"          # ~13px rows, "34%"


def slot_columns(n: int) -> int:
    """How many columns to wrap the slot strip into, so its HEIGHT stays roughly flat.

    Without this the strip grows linearly and a future parallel = 16 would be 208px of hero.
    Wrapping keeps it between 4 and 6 rows tall at every count a homelab will realistically
    use: 8 slots become 2 x 4, 16 become 3 x 6.
    """
    if n <= 4:
        return 1
    if n <= 12:
        return 2
    return 3


# llama-server prints these at its default log level, every ~3 s while a slot works. There is no
# metrics endpoint to ask instead: /metrics needs a flag this deployment does not pass, and
# /slots carries state but no counters or rates. So the numbers come from the log, and the
# state comes from /slots - each from the source that actually has it.
#
#   ... n_gen =    806, tg =  26.02 t/s, tg_3s =  25.99 t/s
#   ... prompt processing, n_tokens =   7105, progress = 1.00, t =   4.70 s / 1511.80 tokens per second
_RE_TG = re.compile(r"n_gen\s*=\s*(\d+),\s*tg\s*=\s*([\d.]+) t/s"
                    r"(?:,\s*tg_3s\s*=\s*([\d.]+))?")
_RE_PP = re.compile(r"prompt processing, n_tokens\s*=\s*(\d+), progress\s*=\s*([\d.]+), "
                    r"t\s*=\s*[\d.]+ s / ([\d.]+) tokens per second")
_RE_TASK = re.compile(r"task\s+(\d+)\s*\|")
# Slot occupancy, straight from the log. A task is in flight between its launch line and its
# release line, which is all that is needed to decide whether anything is running:
#   ... I slot launch_slot_: id  2 | task 11269 | processing task, is_child = 0
#   ... I slot      release: id  2 | task 11269 | stop processing: n_tokens = 83416, truncated = 0
# Tasks this process has seen launched and not yet released, per container:
#   {container: {slot_id: (task_id, first_seen)}}
# Occupancy has to outlive the log window it was read from - see the long note in
# _rates_from_log for the arithmetic that makes a single read unable to answer "is anything
# running". Bounded two ways so a release line that never arrives cannot wedge a backend into
# permanent "busy": a launch is only ever replaced by a newer task id on the same slot, and
# anything older than _OPEN_TASK_MAX_AGE_S is dropped. That ceiling matches the router's own
# in-flight sweep, and nothing legitimate generates for half an hour.
_OPEN_TASK_MAX_AGE_S = 1800.0
_open_tasks: dict[str, dict[int, tuple[int, float]]] = {}

_RE_LAUNCH = re.compile(r"launch_slot_:\s*id\s+(\d+)\s*\|\s*task\s+(\d+)")
_RE_RELEASE = re.compile(r"release:\s*id\s+(\d+)\s*\|\s*task\s+(\d+)")

# One read per container per interval, shared by every caller in that window. The card polls
# every 2 s and reading a log tail is the expensive half, so this keeps a burst of renders from
# multiplying docker calls.
# Two different clocks. The slot read is ~1 ms so it happens on every poll; the log read is
# ~11 ms and its numbers only change every few seconds anyway, so it is throttled. An earlier
# version cached the whole result for 1.8 s, which meant polling faster just redrew stale text.
_SPEED_TTL = 0.35          # floor on full recomputation, to survive a burst of requests
_RATES_TTL = 1.5           # how often the log tail is re-read
_speed_cache: dict[str, tuple[float, "InferenceSpeed | None"]] = {}
_rates_cache: dict[str, tuple[float, dict]] = {}
_GEN_EMA_ALPHA = 0.4       # smoothing for the token-delta rate; raw 500 ms deltas twitch
# (when, token count, task) of the previous sample, so a rate can be derived from how fast the
# slot's token count is moving. llama only logs a generation timing line every ~3 s, which left
# the first seconds of a reply looking like it was still reading the prompt.
# Keyed "container#slot", never by container alone, so the slot count is never baked in: a
# section can go from parallel = 1 to 8 and back and these just grow and shrink with it.
# _prune_slot_state drops keys for slots that no longer exist, so a step down leaves nothing
# behind to be read as a live slot later.
_speed_prev: dict[str, tuple[float, int, int | None, bool]] = {}
_gen_ema: dict[str, float] = {}      # smoothed token-delta rate, per slot


def _slot_key(container_name: str, index: int) -> str:
    return f"{container_name}#{index}"


def _slot_speed(container_name: str, s: dict, r: dict, now: float) -> "SlotSpeed":
    """One slot's figures, and its own EMA state. Called once per slot per sample.

    Each slot keeps a separate smoother because their rates genuinely differ: measured on a
    parallel = 3 section, two slots decoding 400 tokens each in the same batch held 38.5 and
    27.4 tok/s for seven seconds straight. Speculative decoding is why - the draft head is
    accepted far more often on code than on prose - so one shared EMA would blur a real signal.
    """
    index = int(s.get("id") or 0)
    key = _slot_key(container_name, index)
    ctx_used = int(s.get("n_prompt_tokens") or 0)
    ctx_total = int(s.get("n_ctx") or 0)
    cached_tok = int(s.get("n_prompt_tokens_cache") or 0)
    processed = int(s.get("n_prompt_tokens_processed") or 0)
    task_id = s.get("id_task")
    task_id = int(task_id) if isinstance(task_id, (int, float)) else None
    nt = s.get("next_token")
    nt0 = nt[0] if isinstance(nt, list) and nt and isinstance(nt[0], dict) else {}
    decoded = int(nt0.get("n_decoded") or 0)

    if not s.get("is_processing"):
        # A finished run must not smooth into the next one on this slot.
        _speed_prev.pop(key, None)
        _gen_ema.pop(key, None)
        return SlotSpeed(index=index, state="idle", ctx_used=ctx_used,
                         ctx_total=ctx_total, ctx_cached=cached_tok)

    # One decoded token is the switch out of prefill. See the n_prompt_tokens note below.
    state = "generating" if decoded else "prefill"
    prefill_pct = 0
    prefill_done = prefill_total = 0
    if state == "prefill" and ctx_used:
        # Cache hits count as read: they are tokens the slot no longer has to process, and
        # excluding them would make a 95%-cached prompt crawl from 0 while finishing instantly.
        prefill_done, prefill_total = cached_tok + processed, ctx_used
        prefill_pct = min(100, int(round(100.0 * prefill_done / prefill_total)))

    gen_tps = 0.0
    prev = _speed_prev.get(key)
    if state == "generating" and prev and prev[2] == task_id and prev[3]:
        dt, dn = now - prev[0], decoded - prev[1]
        if 0.25 < dt < 10.0 and dn > 0:
            raw = dn / dt
            last = _gen_ema.get(key)
            gen_tps = (raw if last is None
                       else _GEN_EMA_ALPHA * raw + (1 - _GEN_EMA_ALPHA) * last)
            _gen_ema[key] = gen_tps
    elif state == "generating" and r.get("open_task") == task_id:
        # First generating sample for this task, so there is no delta yet. The log's figure
        # describes only the newest open task, so at most one slot can borrow it.
        gen_tps = r["cur_gen_tps"]
    _speed_prev[key] = (now, decoded, task_id, state == "generating")

    return SlotSpeed(index=index, state=state, gen_tps=gen_tps, decoded=decoded,
                     ctx_used=ctx_used, ctx_total=ctx_total, ctx_cached=cached_tok,
                     prefill_pct=prefill_pct, prefill_done=prefill_done,
                     prefill_total=prefill_total, prompt_tokens=cached_tok + processed,
                     task_id=task_id)


def _prune_slot_state(container_name: str, live: set[int]) -> None:
    """Forget per-slot state for slots this backend no longer has."""
    prefix = f"{container_name}#"
    keep = {_slot_key(container_name, i) for i in live}
    for store in (_speed_prev, _gen_ema):
        for k in [k for k in store if k.startswith(prefix) and k not in keep]:
            del store[k]

# /slots is the ONLY llama endpoint this app must not poll while a model is idle. It is served as
# a task on the server's own queue, which is the same queue the idle detector watches, so polling
# it both resets `--sleep-idle-seconds` and wakes an already-sleeping model. Measured on an
# R9700 against a 20 s threshold: polling /slots every 500 ms held VRAM at 25.63 GiB for 45 s and
# it never slept, and polling it once asleep took VRAM straight back from 22.51 to 25.61 GiB.
# /health and /props both slept normally under the same polling, and /v1/models is router state
# that is never proxied to the child, so those three are safe at any rate.
#
# So occupancy comes from the log instead, and /slots is touched only while a task is genuinely
# in flight - at which point the model was never going to sleep anyway, so the poll is free.
_PROPS_TTL = 60.0                    # n_ctx and slot count only change when a model reloads
# Both of these are keyed by "container/model", not container alone: the router swaps models
# under one container, and a 128k context read from the model before the swap must not be shown
# against the one after it.
_props_cache: dict[str, tuple[float, dict]] = {}
# Last per-slot context figures seen while busy, so the strip survives going idle without a
# /slots read. One (ctx_used, ctx_total) per slot, in index order; the length IS the slot count.
_last_slots: dict[str, list[tuple[int, int]]] = {}
# The last SOLO pass, so its row stays on screen when the solo path is empty. /metrics'
# requests list carries no solo/batch marker - the engine does not say which route a finished
# request took - so remembering it here is the only way to show the last SOLO figure rather
# than the last figure. (ts, gen_tps, prompt_tokens, gen_tokens); the last live sample before
# a pass finishes is its final one, so no completion hook is needed.
_last_solo: dict[str, tuple[float, float, int, int]] = {}


def _evict_other_models(cache: dict, key: str) -> None:
    """Drop this container's entries for OTHER models. Both caches above are keyed
    container/model, so every swap used to leave a permanent entry behind; the old
    model's figures are unreachable the moment a new one loads."""
    prefix = key.split("/", 1)[0] + "/"
    for stale in [k for k in cache if k.startswith(prefix) and k != key]:
        cache.pop(stale, None)


# A task the log still calls open but /slots has reported finished. Remembering it stops a
# missing release line - a crash, or a tail that scrolled - from polling /slots forever.
_settled_task: dict[str, int] = {}

# Rolling generation-speed history for the dashboard sparkline. 240 samples at the hero's 500 ms
# poll is about two minutes, which is long enough to see a request start, run and finish.
_TPS_HISTORY_MAX = 240
_TPS_MIN_GAP_S = 0.25                # two tabs polling at once must not double-sample
_tps_history: dict[str, list[tuple[float, float]]] = {}


def tps_history(container_name: str) -> list[float]:
    """Recent generation speeds for a backend, oldest first. Empty until something has run."""
    return [v for _ts, v in _tps_history.get(container_name, [])]


def _record_tps(container_name: str, value: float) -> None:
    hist = _tps_history.setdefault(container_name, [])
    now = time.time()
    if hist and (now - hist[-1][0]) < _TPS_MIN_GAP_S:
        return
    hist.append((now, round(float(value or 0.0), 2)))
    if len(hist) > _TPS_HISTORY_MAX:
        del hist[:len(hist) - _TPS_HISTORY_MAX]


# Prefill is NOT a sustained rate the way generation is, so its history is kept separately and
# sparsely: one reading per completed request, not one per poll.
#
# Generation can be sampled continuously - a slot decoding for twenty seconds gives twenty
# honest samples of what the machine is doing right now. A prefill happens once, and the only
# trustworthy figure for it is the one the engine reports afterwards (read tokens over
# prompt_ms). Sampling that on a one-second poll would redraw the same completed request twenty
# times and turn one prompt into a plateau.
#
# So the series here is "the rate of each prefill, in order", which is a different and more
# useful shape: it shows long prompts dragging the rate down across a session, where a
# zero-filled line would mostly draw zeros.
_pp_history: dict[str, list[tuple[float, float]]] = {}




# Context in use, continuous like generation and unlike prefill: a slot holds its context
# between requests, so every poll is a true reading of what the window currently contains.
# Zeros are recorded rather than skipped for the same reason they are in _record_tps - the
# gaps are part of the shape, and a line drawn only from busy moments implies continuous work.
_ctx_history: dict[str, list[tuple[float, float]]] = {}


def ctx_history(container_name: str) -> list[float]:
    """Recent context-in-use readings for a backend, in tokens, oldest first."""
    return [v for _ts, v in _ctx_history.get(container_name, [])]


def _record_ctx(container_name: str, tokens: float) -> None:
    hist = _ctx_history.setdefault(container_name, [])
    now = time.time()
    if hist and (now - hist[-1][0]) < _TPS_MIN_GAP_S:
        return
    hist.append((now, float(tokens or 0.0)))
    if len(hist) > _TPS_HISTORY_MAX:
        del hist[:len(hist) - _TPS_HISTORY_MAX]
def prefill_history(container_name: str) -> list[float]:
    """Completed prefill rates for a backend, oldest first, one per request."""
    return [v for _ts, v in _pp_history.get(container_name, [])]


def _record_prefill_tps(container_name: str, value: float) -> None:
    """Record a COMPLETED prefill rate, once.

    `value` must be the engine's own finished figure (last_pp_tps), never the in-progress one:
    a rate measured part way through a prefill is an artefact of where the poll landed.

    Deduped on the value rather than on a request id, because no request id survives into this
    dict - `open_task` is None the moment the request finishes, which is exactly when the
    completed rate becomes available. The cost of that choice is that two consecutive requests
    whose rates round to the same figure collapse into one point. On rates like 640.9 and
    2756.2 that is rare, and a lost point in a sparkline is a far smaller lie than a plateau.
    """
    v = round(float(value or 0.0), 1)
    if v <= 0:
        return
    hist = _pp_history.setdefault(container_name, [])
    if hist and hist[-1][1] == v:
        return
    hist.append((time.time(), v))
    if len(hist) > _TPS_HISTORY_MAX:
        del hist[:len(hist) - _TPS_HISTORY_MAX]


async def _slot_states(container_name: str, internal_port: int, model_id: str) -> list[dict]:
    """Every slot the loaded model has, in index order, or [] if they cannot be read.

    All of them, not just the working one: one response already carries the lot, so reporting
    per-slot detail costs no extra HTTP. It used to return slots[0] unconditionally, which
    reported an idle slot for every request that landed anywhere else - measured at 38% of tasks
    (5,304 of 13,922) on Qwen3.8-27B-UD-Q4_K_M, each one a gap in the speedometer and a zero in
    the history. The caller no longer has to pick one at all.
    """
    url = f"http://{container_name}:{internal_port}/slots"
    try:
        r = await _http().get(url, params={"model": model_id},
                              timeout=httpx.Timeout(1.5, read=2.0))
        if r.status_code != 200:
            return []
        slots = r.json()
        if not isinstance(slots, list):
            return []
        out = [s for s in slots if isinstance(s, dict)]
        # Index order, because the strip renders in this order and llama is not required to
        # return them sorted. Slots without an id sort last rather than crashing the sort.
        out.sort(key=lambda s: s.get("id") if isinstance(s.get("id"), int) else 1 << 30)
        return out
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return []


async def _props_state(container_name: str, internal_port: int, model_id: str) -> dict:
    """Static per-load facts: the real n_ctx, the slot count, and whether the model is asleep.

    Safe to poll at any rate - see the note by _PROPS_TTL. It is also the right source for the
    context a single conversation actually gets, which is NOT what /v1/models reports. The argv
    there carries `--ctx-size`, the total KV pool; llama-server divides that pool by `parallel`
    and rounds up to a multiple of 128 to get the per-slot context, which is what `n_ctx` means
    here and what the slot objects report. On Qwen3.8-27B-UD-Q4_K_M, ctx-size 386000 with
    parallel 3 gives 128768 (386000/3 = 128666.7, rounded up), and the load log says so outright:
    "n_slots = 3, n_ctx_slot = 128768". Both numbers are real, they answer different questions,
    so the panel compares per-slot usage against the per-slot total.
    """
    key = f"{container_name}/{model_id}"
    cached = _props_cache.get(key)
    if cached and (time.time() - cached[0]) < _PROPS_TTL:
        return cached[1]
    out: dict = {}
    try:
        r = await _http().get(f"http://{container_name}:{internal_port}/props",
                              params={"model": model_id},
                              timeout=httpx.Timeout(1.5, read=2.0))
        if r.status_code == 200:
            d = r.json()
            gen = d.get("default_generation_settings") or {}
            out = {"n_ctx": int(gen.get("n_ctx") or 0),
                   "total_slots": int(d.get("total_slots") or 0),
                   "is_sleeping": bool(d.get("is_sleeping"))}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        out = {}
    _evict_other_models(_props_cache, key)
    _props_cache[key] = (time.time(), out)
    return out


# Strata's progress lines. A different engine logs a different vocabulary, and none of
# llama.cpp's patterns appear in it - which is exactly why the speedometer read as permanently
# idle for this backend: _rates_from_log found no task launches, so `open_task` stayed None and
# /slots was never even asked.
#
# What it writes to stdout, with thousands separators, one line every second or so:
#
#   [strata] reading the prompt: 32,619 of 32,624 tokens, 2 s so far
#   [strata] thinking: 328 of max 32000 tokens, 60.1 tok/s, 8 s
#   [strata] answering: 283 of max 32000 tokens, 75.3 tok/s, 4 s
#   [strata] writing a tool call: bash: 471 of max 32000 tokens, 62.5 tok/s, 10 s
#   [strata] done: 383 tokens in 6 s (71.9 tok/s) (stop, cancel=False), expert cache 99.9% hit
#
# thinking / answering / writing a tool call are all GENERATING - the phase differs, the rate
# does not - so they share one pattern. Keeping them apart would make a reasoning model look
# idle for the whole of its reasoning, which on a thinking model is most of the request.
_RE_ST_PROMPT = re.compile(r"reading the prompt:\s*([\d,]+)\s+of\s+([\d,]+)\s+tokens,\s*(\d+)\s*s")
_RE_ST_GEN = re.compile(r"(?:thinking|answering|writing a tool call[^:]*):\s*([\d,]+)\s+of\s+max\s+"
                        r"[\d,]+\s+tokens,\s*([\d.]+)\s*tok/s")
_RE_ST_DONE = re.compile(r"done:\s*([\d,]+)\s+tokens\s+in\s+\d+\s*s\s*\(([\d.]+)\s*tok/s\)")
_RE_ST_HIT = re.compile(r"expert cache\s+([\d.]+)%\s+hit")


def _int_c(txt: str) -> int:
    """An integer that may carry thousands separators. Strata writes 32,619; int() would raise."""
    try:
        return int(str(txt).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0


def _rates_from_log_strata(name: str) -> dict:
    """The same dict _rates_from_log returns, read from Strata's log instead.

    `open_task` has to be a value that CHANGES per request, because the caller uses it both as
    "is anything in flight" and as the key for its settled-task marker. Strata logs no task ids,
    so the count of completed requests in the window stands in: it is monotonic within a run and
    only advances when a request finishes, which is exactly the property the marker needs.

    In flight means the newest progress line is newer than the newest `done:`. Comparing
    positions rather than timestamps keeps this correct when several requests land in the same
    second, which they do under an agent client.
    """
    out = {"cur_gen_tps": 0.0, "cur_gen_tokens": 0, "cur_pp_tps": 0.0, "cur_pp_pct": 0,
           "cur_pp_done": 0, "cur_pp_total": 0, "cur_prompt_tokens": 0,
           "last_gen_tps": 0.0, "last_gen_tokens": 0, "last_pp_tps": 0.0, "newest": "",
           "open_task": None, "open_slot": 0, "expert_hit_pct": None, "conv_cache": None}

    # /metrics first: the same numbers as data instead of as prose. The log scrape below
    # stays as the fallback for a Strata too old to have the endpoint.
    m = strata_metrics(name)
    if m:
        live = m.get("live") or {}
        last = (m.get("requests") or [{}])[0]
        totals = m.get("totals") or {}
        state = live.get("state") or "idle"
        out["cur_gen_tps"] = float(live.get("tok_s") or 0.0)
        out["cur_gen_tokens"] = int(live.get("generated") or 0)
        # The conversation the SOLO pass is carrying, needed for its context bar during
        # GENERATION - prompt_total/prompt_read only appear while reading, and the solo
        # request lives in no slot, so nothing else reports its size once the read is done.
        out["cur_prompt_tokens"] = int(live.get("prompt_tokens") or 0)
        if state == "reading":
            out["cur_pp_tps"] = float(live.get("prefill_tok_s_mean") or 0.0)
            pt, pr = live.get("prompt_total"), live.get("prompt_read")
            if pt:
                out["cur_pp_pct"] = round(100.0 * (pr or 0) / pt)
                out["cur_pp_done"] = int(pr or 0)
            # The SIZE is known from the first sample; the PROGRESS is not. Strata reports
            # prompt_read/prompt_total only once a 16,384-token chunk has landed, so a read
            # stays null for the first several seconds of a long prompt and for the WHOLE of
            # any read smaller than one chunk - which is every turn of an agent client working
            # behind a cached prefix (measured: 833 fresh tokens of an 83,456-token prompt).
            # Publishing the total regardless lets the UI say which of those two situations
            # this is. It is the distinction the percentage existed to make: 400 tokens and
            # 166,754 tokens are 0.3 s and 57 s of waiting on this box.
            out["cur_pp_total"] = int(pt or live.get("prompt_tokens") or 0)
        out["last_gen_tps"] = float(last.get("decode_tok_s") or 0.0)
        out["last_gen_tokens"] = int(last.get("output_tokens") or 0)
        # Prompt rate over the tokens actually READ, not the whole prompt: with prefix-cache
        # reuse prompt_ms covers only the few thousand tokens that were not a cache hit, and
        # dividing the full prompt by it printed 432,000 tok/s on a warm agent session. An
        # older engine without prompt_read gets 0 - an honest zero beats a wrong 400k.
        read = last.get("prompt_read")
        if last.get("prompt_ms") and read:
            out["last_pp_tps"] = round(read / (last["prompt_ms"] / 1000.0), 1)
        hr = last.get("hit_rate")
        out["expert_hit_pct"] = round(hr * 100.0, 1) if isinstance(hr, (int, float)) else None
        cc = m.get("conversation_cache") or {}
        if cc.get("enabled"):
            # Parked conversations and the host RAM they occupy are the live state; the
            # counters are since-boot and belong in the tooltip, not the row.
            out["conv_cache"] = {
                "parked": int(cc.get("parked") or 0), "slots": int(cc.get("slots") or 0),
                "bytes": int(cc.get("bytes") or 0), "parks": int(cc.get("parks") or 0),
                "restores": int(cc.get("restores") or 0),
                "evictions": int(cc.get("evictions") or 0),
                "requests": int(cc.get("requests") or 0),
                "requests_reused": int(cc.get("requests_reused") or 0)}
        n = totals.get("requests")
        # `open_task` must be None when nothing is running and must CHANGE the moment a request
        # STARTS. totals.requests is a COMPLETED counter, so while a request was in flight it
        # still read the value the caller had already filed in _settled_task - which made
        # inference_speed take its "this task is finished" branch, skip the /slots read, and
        # report a model that was demonstrably generating as idle for the request's whole
        # duration. (Observed: /metrics live.state "generating" while the speedometer showed
        # standby.) The in-flight request's own id - completed + 1 - is stable for as long as it
        # runs and never collides with the settled marker.
        working = state not in ("", "idle")
        out["open_task"] = str((n or 0) + 1) if working else None
        # "reading" is Strata's word for prefill. Passed through unmapped it did not equal
        # "prefill" in _slots_strata, which then fell through to "generating" - so a long prompt
        # read was labelled as decode, at a decode rate it was not producing.
        out["newest"] = "prefill" if state == "reading" else ("generating" if working else "")
        return out

    ok, text = container_logs(name, tail=400, current_run_only=True)
    if not ok:
        return out
    lines = text.splitlines()

    done_n = 0
    last_done_at = -1
    newest_gen = newest_prompt = None
    newest_gen_at = newest_prompt_at = -1
    for i, line in enumerate(lines):
        if (m := _RE_ST_DONE.search(line)):
            done_n += 1
            last_done_at = i
            out["last_gen_tokens"] = _int_c(m.group(1))
            try:
                out["last_gen_tps"] = float(m.group(2))
            except ValueError:
                pass
            if (h := _RE_ST_HIT.search(line)):
                try:
                    out["expert_hit_pct"] = float(h.group(1))
                except ValueError:
                    pass
            continue
        if (m := _RE_ST_GEN.search(line)):
            newest_gen, newest_gen_at = m, i
            continue
        if (m := _RE_ST_PROMPT.search(line)):
            newest_prompt, newest_prompt_at = m, i

    # Nothing after the last completion: idle, and the last_* figures are what the card dims.
    if max(newest_gen_at, newest_prompt_at) <= last_done_at:
        return out

    out["open_task"] = done_n + 1
    if newest_gen is not None and newest_gen_at > newest_prompt_at:
        out["cur_gen_tokens"] = _int_c(newest_gen.group(1))
        try:
            out["cur_gen_tps"] = float(newest_gen.group(2))
        except ValueError:
            pass
        out["newest"] = "generating"
    elif newest_prompt is not None:
        done, total = _int_c(newest_prompt.group(1)), _int_c(newest_prompt.group(2))
        secs = _int_c(newest_prompt.group(3))
        out["cur_pp_done"] = done
        out["cur_prompt_tokens"] = total
        out["cur_pp_pct"] = min(100, round(done * 100 / total)) if total else 0
        # "N s so far" is whole seconds and starts at 0, so the first sample of a short prompt
        # would divide by zero. Reporting no rate beats reporting an infinite one.
        out["cur_pp_tps"] = (done / secs) if secs > 0 else 0.0
        out["newest"] = "prefill"
    return out




def _outside_slots(engine: str, r: dict) -> bool:
    """Is the engine working on something that lives in no slot?

    Strata runs a LONE request on its solo path - the fastest route, the one that keeps its MTP
    drafts - and only moves a conversation into a batch slot when a second request arrives. So
    with parallel > 1 every slot reports is_processing=false for the whole of a single request,
    while /metrics says live.running=1 and live.outside_slots=1 and the log reader says
    newest='generating'.

    Reading busy-ness from the slots alone was therefore wrong the moment slots were enabled:
    the settled-task guard concluded the request had finished mid-generation, dropped the slot
    read, and the panel fell to the idle branch - grey sparklines, an IDLE badge and the previous
    request's frozen numbers, for every request. Measured during a live generation:
    is_processing=[False,False,False,False] with running=1, outside_slots=1.

    The slot strip stays honest about this: none of them holds the work, and it says so. Only
    the headline state comes from the engine.
    """
    return engine == "strata" and bool(r.get("newest"))
def _slots_strata(raw: list[dict], r: dict) -> list["SlotSpeed"]:
    """Strata's /slots mapped onto SlotSpeed, filling only what it actually reports.

    Its shape is its own and much smaller than llama's:

        [{"id": 0, "n_ctx": 262144, "is_processing": false, "n_prompt_tokens": 33137}]

    There is no n_prompt_tokens_cache / n_prompt_tokens_processed, so the cache-ratio and
    per-slot prefill readouts have no input here. They are left at 0 DELIBERATELY - an absent
    measurement rendered as 0% cached is a claim, and a wrong one. The rates come from the log
    instead, which for a one-slot server describes that slot exactly.
    """
    out: list[SlotSpeed] = []
    for i, sl in enumerate(raw):
        busy = bool(sl.get("is_processing"))
        state = "idle"
        if busy:
            state = "prefill" if r.get("newest") == "prefill" else "generating"
        out.append(SlotSpeed(
            index=int(sl.get("id") or i), state=state,
            gen_tps=r["cur_gen_tps"] if state == "generating" else 0.0,
            decoded=r["cur_gen_tokens"] if state == "generating" else 0,
            ctx_used=_int_c(sl.get("n_prompt_tokens") or 0),
            ctx_total=_int_c(sl.get("n_ctx") or 0),
            prefill_pct=r["cur_pp_pct"] if state == "prefill" else 0,
            prefill_total=r.get("cur_pp_total", 0) if state == "prefill" else 0,
            prefill_done=r["cur_pp_done"] if state == "prefill" else 0,
            prompt_tokens=_int_c(sl.get("n_prompt_tokens") or 0) if state != "idle" else 0,
        ))
    return out


def _rates_from_log(name: str) -> dict:
    """Occupancy and throughput from the current run's log tail.

    Two things come out of one read. `open_task` is the newest task that has a launch line and no
    matching release, which is how this app knows whether anything is running WITHOUT asking
    /slots and waking the model. And the rates, split by WHICH request they describe: every timing
    line names its task, so tying them to the open task is the difference between "reading the
    prompt at 1,985 tok/s" and a stale number from the request before it. An earlier version
    ignored the task id and reported the previous run's rates - and its finished 100% progress
    bar - as if they were current.

    Returns rates for the live task (cur_*) and, separately, the most recent of any task (last_*),
    which is what the card shows, dimmed, while the model sits idle.
    """
    out = {"cur_gen_tps": 0.0, "cur_gen_tokens": 0, "cur_pp_tps": 0.0, "cur_pp_pct": 0,
           "cur_pp_done": 0, "cur_pp_total": 0,
           "last_gen_tps": 0.0, "last_gen_tokens": 0, "last_pp_tps": 0.0, "newest": "",
           "open_task": None, "open_slot": None}
    # A wide tail, then the router's own chatter removed. It has to be both. While a slot
    # generates it logs a timing line only every ~3 s, so a long reply buries the
    # prompt-processing lines from the start of that same request; and in router mode llama.cpp
    # logs EVERY proxied request, so the window fills with this app's own polling. Measured on
    # llama-9700x2 while Devstral was generating: 94.8% proxy lines, 63.5 of them per second,
    # arriving exactly 50 ms apart. tail=1200 therefore covers about ELEVEN SECONDS of wall
    # clock, and a 88k-token prefill was measured at 104 s and 118 s on that same model.
    #
    # That arithmetic is the whole reason this card used to say "idle" mid-generation, and the
    # reason widening the tail is not the fix. The launch line is the newest thing in the log the
    # moment a request starts, so this read always sees it - and then it scrolls out while the
    # request is still running, the pairing below finds nothing unpaired, and occupancy flips
    # back to none. A long request is therefore reported busy for ~11 s and idle for the rest of
    # its life, which is precisely backwards: the slower the model, the sooner the lie.
    #
    # So occupancy is carried across reads in _open_tasks instead. Seeing the launch once is
    # enough; the release line retires it. The tail stays narrow because it only has to be wide
    # enough to catch each line as it is written, and the cost of reading it is not free: 4.5 ms
    # at tail=1200 against 144 ms at 60000, on a path that runs every 1.5 s per backend.
    ok, text = container_logs(name, tail=1200, current_run_only=True)
    if not ok:
        return out
    lines = [ln for ln in text.splitlines() if "proxy_reques" not in ln]

    # Oldest first, pairing launches with releases per slot. Whatever is still unpaired at the
    # end is in flight - but "unpaired" now means unpaired across every read since the launch,
    # not merely unpaired within this window, for the windowing reason above. A release line with
    # no launch in the window is from before it and retires the remembered launch, which is the
    # whole point: the launch was seen on an earlier read, and this is the line that closes it.
    pending = _open_tasks.setdefault(name, {})
    now = time.time()
    for line in lines:
        m = _RE_LAUNCH.search(line)
        if m:
            slot_id, tid = int(m.group(1)), int(m.group(2))
            prev = pending.get(slot_id)
            # Only a NEW task id refreshes the clock. Re-reading the same launch line while it
            # happens to still be in the window must not keep extending its deadline, or a
            # release that never arrives (a crashed child, a container restart mid-request)
            # would leave this backend reading busy forever. Task ids rise monotonically within
            # a run, so a different id on the same slot is genuinely a new task - and that also
            # clears whatever a restart left behind, since ids restart from 1.
            if prev is None or prev[0] != tid:
                pending[slot_id] = (tid, now)
            continue
        m = _RE_RELEASE.search(line)
        if m:
            slot_id, tid = int(m.group(1)), int(m.group(2))
            if pending.get(slot_id, (None,))[0] == tid:
                del pending[slot_id]
    for slot_id in [s for s, (_, seen) in pending.items()
                    if now - seen > _OPEN_TASK_MAX_AGE_S]:
        del pending[slot_id]
    if pending:
        # Several slots can run at once; the newest task is the one the panel follows.
        slot_id = max(pending, key=lambda s: pending[s][0])
        out["open_slot"], out["open_task"] = slot_id, pending[slot_id][0]

    task_id = out["open_task"]
    for line in reversed(lines):
        m_task = _RE_TASK.search(line)
        if not m_task:
            continue
        mine = task_id is not None and int(m_task.group(1)) == task_id
        m = _RE_TG.search(line)
        if m:
            tps, toks = float(m.group(3) or m.group(2)), int(m.group(1))
            if not out["last_gen_tps"]:
                out["last_gen_tps"], out["last_gen_tokens"] = tps, toks
            if mine and not out["cur_gen_tps"]:
                out["cur_gen_tps"], out["cur_gen_tokens"] = tps, toks
                out["newest"] = out["newest"] or "gen"
            continue
        m = _RE_PP.search(line)
        if m:
            tps, pct = float(m.group(3)), int(round(float(m.group(2)) * 100))
            # n_tokens is CUMULATIVE, not the size of this chunk. Verified against a live task:
            #   n_tokens = 18665, progress = 0.23   ...   n_tokens = 57157, progress = 0.69
            # which implies the same ~82k prompt at both ends. Only the running count is taken
            # from here; the total is NOT, because progress is printed to two decimals and
            # dividing by it gives +/-6% at the start of a prompt. An exact total comes from
            # /slots or it does not come at all.
            done = int(m.group(1))
            if not out["last_pp_tps"]:
                out["last_pp_tps"] = tps
            if mine and not out["cur_pp_tps"]:
                out["cur_pp_tps"], out["cur_pp_pct"] = tps, pct
                out["cur_pp_done"] = done
                out["newest"] = out["newest"] or "prefill"
    return out


async def inference_speed(container_name: str, internal_port: int | None,
                          loaded_model: str | None) -> "InferenceSpeed | None":
    """Live throughput for a backend, or None when there is nothing to show.

    Cheap parts first: no loaded model means no speedometer, and that is the common case on an
    idle box. Both reads are skipped entirely in that state.
    """
    if not loaded_model or internal_port is None:
        return None
    cached = _speed_cache.get(container_name)
    if cached and (time.time() - cached[0]) < _SPEED_TTL:
        return cached[1]

    model_id = loaded_model.split(",")[0].strip()
    _engine = engine_for(container_name)

    # The log read comes FIRST now, because it is what decides whether /slots may be touched.
    cached_rates = _rates_cache.get(container_name)
    if cached_rates and (time.time() - cached_rates[0]) < _RATES_TTL:
        r = cached_rates[1]
    else:
        # Per ENGINE, because the log is the engine's own vocabulary. Dispatching here rather
        # than teaching one parser both means neither grows patterns it can match by accident.
        _reader = (_rates_from_log_strata if _engine == "strata" else _rates_from_log)
        r = await asyncio.to_thread(_reader, container_name)
        _rates_cache[container_name] = (time.time(), r)

    # Only ask /slots when a task is actually in flight. An idle model must be left alone or it
    # can never sleep, and asking would wake it outright.
    raw_slots: list[dict] = []
    open_task = r["open_task"]
    if open_task is None:
        # Nothing in flight. Drop any settled-task marker here rather than letting it persist:
        # task ids restart from 1 when the router reloads a child, so a marker left over from a
        # previous life could collide with a genuine new task and silence one request's readout.
        _settled_task.pop(container_name, None)
        # Strata is the exception to the hands-off rule. Its GET /slots is a read of a static
        # view: upstream keeps the idle-unload clock to the generation paths only, and there
        # is no sleep state to wake - and n_prompt_tokens, the conversation it carries on,
        # lives nowhere else. Without this read an idle Strata renders "-" for context after
        # any restart, when there is no remembered busy sample to fall back on.
        if _engine == "strata":
            raw_slots = await _slot_states(container_name, internal_port, model_id)
    elif _settled_task.get(container_name) != open_task:
        raw_slots = await _slot_states(container_name, internal_port, model_id)
        # A Strata solo request occupies no slot, so is_processing is false everywhere while
        # it generates. See _outside_slots: trusting that here marked the task settled
        # mid-generation and froze the panel at idle for the rest of it.
        if not _outside_slots(_engine, r) and not any(s.get("is_processing") for s in raw_slots):
            # /slots is authoritative. Record the disagreement so a release line that never
            # arrived does not keep this backend awake for the rest of the process's life.
            _settled_task[container_name] = open_task
            raw_slots = []

    now = time.time()
    ctx_key = f"{container_name}/{model_id}"
    slots: list[SlotSpeed] = []

    if raw_slots:
        _prune_slot_state(container_name, {int(s.get("id") or 0) for s in raw_slots})
        if _engine == "strata":
            slots = _slots_strata(raw_slots, r)
        else:
            slots = [_slot_speed(container_name, s, r, now) for s in raw_slots]
        _evict_other_models(_last_slots, ctx_key)
        _last_slots[ctx_key] = [(sl.ctx_used, sl.ctx_total) for sl in slots]
    else:
        # Idle, and /slots was deliberately not read. Rebuild the strip from the last busy
        # sample - that KV is still resident, so the figures are real - or from /props when
        # there is no history yet. /props carries total_slots, so even a freshly started process
        # renders the right NUMBER of idle slots without touching /slots.
        remembered = _last_slots.get(ctx_key)
        if remembered:
            slots = [SlotSpeed(index=i, state="idle", ctx_used=u, ctx_total=t)
                     for i, (u, t) in enumerate(remembered)]
        else:
            p = await _props_state(container_name, internal_port, model_id)
            per_slot = int(p.get("n_ctx") or 0)
            count = max(1, int(p.get("total_slots") or 1))
            slots = [SlotSpeed(index=i, state="idle", ctx_total=per_slot)
                     for i in range(count)]
        _prune_slot_state(container_name, set())

    generating = [sl for sl in slots if sl.state == "generating"]
    prefilling = [sl for sl in slots if sl.state == "prefill"]
    # Work on the solo path belongs to no slot, so the slot lists are empty while the engine is
    # busy (see _outside_slots). The strip stays accurate - none of them holds it - but the
    # headline state has to come from the engine or the panel reports idle mid-generation.
    outside = not generating and not prefilling and _outside_slots(_engine, r)
    busy = bool(generating or prefilling or outside)
    if generating:
        state = "generating"
    elif prefilling:
        state = "prefill"
    elif outside:
        state = "prefill" if r.get("newest") == "prefill" else "generating"
    else:
        state = "idle"

    # The solo path gets its own row because none of the slot rows holds its request: Strata
    # runs a lone request outside the batch slots (see _outside_slots), so without this the
    # one thing generating has no bar at all - just four idle slot rows and a headline. The
    # row is present whenever Strata has batch slots, showing the last solo pass dimmed when
    # the path is empty: the strip resizing on every solo-to-batch hand-off was more
    # distracting than a row that is idle half the time.
    solo = None
    solo_last_ts = 0.0
    if _engine == "strata" and len(slots) > 1:
        per_slot_ctx = max((sl.ctx_total for sl in slots), default=0)
        if outside:
            prompt_tok = int(r.get("cur_prompt_tokens") or 0)
            gen_tok = int(r.get("cur_gen_tokens") or 0)
            solo_last_ts = now
            _evict_other_models(_last_solo, ctx_key)
            _last_solo[ctx_key] = (now, r["cur_gen_tps"], prompt_tok, gen_tok)
            solo = SlotSpeed(
                index=-1,
                state="prefill" if r.get("newest") == "prefill" else "generating",
                gen_tps=r["cur_gen_tps"],
                decoded=gen_tok,
                # Context fill like the slot rows: what the conversation holds, over the
                # budget one conversation gets. The log fallback has no prompt size mid-
                # generation, so the bar degrades to generated-only there rather than
                # guessing.
                ctx_used=prompt_tok + gen_tok,
                ctx_total=per_slot_ctx,
                prefill_pct=r["cur_pp_pct"],
                prefill_done=r["cur_pp_done"],
                prefill_total=r["cur_pp_total"],
                prompt_tokens=prompt_tok,
            )
        else:
            ts, tps, prompt_tok, gen_tok = _last_solo.get(ctx_key, (0.0, 0.0, 0, 0))
            solo_last_ts = ts
            solo = SlotSpeed(
                index=-1, state="idle", gen_tps=tps, decoded=gen_tok,
                ctx_used=prompt_tok + gen_tok, ctx_total=per_slot_ctx,
                prompt_tokens=prompt_tok,
            )

    # The machine's rate, not one slot's. Summing is the honest aggregate: slots decode in the
    # same batch, so their rates add up to what the box is actually producing.
    gen_tps = sum(sl.gen_tps for sl in generating)
    gen_tokens = sum(sl.decoded for sl in generating)
    if state == "generating" and not gen_tps:
        # First frame of a run, before any slot has two samples to difference.
        gen_tps, gen_tokens = r["cur_gen_tps"], gen_tokens or r["cur_gen_tokens"]
    if state == "idle":
        gen_tps, gen_tokens = r["last_gen_tps"], r["last_gen_tokens"]
        prefill_tps, prefill_pct = r["last_pp_tps"], 0
    else:
        prefill_tps, prefill_pct = r["cur_pp_tps"], r["cur_pp_pct"]
        if not prefill_pct and prefilling:
            prefill_pct = max(sl.prefill_pct for sl in prefilling)

    # Summed over prefilling slots for the same reason gen_tps is summed: the figure describes
    # the machine. Both are 0 off the prefill path, since `prefilling` is empty then.
    # Busy slots only. An idle slot still reports the context it is holding, and folding that
    # into the cache ratio mixes one conversation's cache with another conversation's size.
    prompt_tokens = sum(sl.prompt_tokens for sl in slots
                        if sl.state in ("prefill", "generating"))
    if solo is not None and solo.state != "idle":
        # The solo conversation is the live one while it runs; its prompt is the
        # denominator the cache bar divides by.
        prompt_tokens = solo.prompt_tokens
    prefill_done = sum(sl.prefill_done for sl in prefilling)
    prefill_total = sum(sl.prefill_total for sl in prefilling)
    if not prefill_done and state == "prefill":
        # /slots had nothing - the log's running count, with no total to go with it.
        prefill_done = r["cur_pp_done"]

    # Pool-wide context: while conversations WORK, the headline is those conversations -
    # the same rule the solo path follows above. Slots parked with idle KV are holding,
    # not working, and folding them in made a batch request read its own size plus every
    # bystander's. With nothing live the sum of all slots is the honest figure: that KV
    # is still resident, and the tile reads "how full the window has been".
    live_slots = generating + prefilling
    ctx_used = sum(sl.ctx_used for sl in live_slots or slots)
    ctx_cached = sum(sl.ctx_cached for sl in live_slots or slots)
    per_slot_ctx = max((sl.ctx_total for sl in slots), default=0)
    if _engine == "strata":
        # Strata's slots share ONE pool: /slots reports each slot's n_ctx as the whole engine
        # context (measured live: context 262144, batch_slots 4, every slot n_ctx 262144),
        # not the per-slot share llama-server derives from --ctx-size / parallel. Multiplying
        # by the slot count inflated the context bar's denominator fourfold - "of 1048k
        # window" against a 262k window.
        ctx_total = per_slot_ctx
    else:
        ctx_total = per_slot_ctx * len(slots)
    if solo is not None and solo.state in ("prefill", "generating"):
        # The headline follows the LIVE conversation, which is what the tile's own tooltip
        # promises ("the live conversation closest to its ceiling") and what ns.fill does.
        # A solo request lives outside every slot, so while it generates its context IS
        # the figure. Summing the idle slots on top counted other conversations' parked
        # KV - slots that are holding, not working - and the number read hundreds of
        # thousands while the conversation generating was a fraction of that.
        ctx_used = solo.ctx_used
        ctx_cached = solo.ctx_cached

    # Idle samples are recorded as zero rather than skipped: the gaps between requests are part
    # of the shape, and a line drawn only from busy moments would imply continuous work.
    _record_tps(container_name, gen_tps if state == "generating" else 0.0)
    # The completed figure only, so one point per request. See _record_prefill_tps.
    _record_prefill_tps(container_name, r["last_pp_tps"])
    _record_ctx(container_name, ctx_used)

    out = InferenceSpeed(
        model=model_id, state=state, gen_tps=gen_tps, gen_tokens=gen_tokens,
        prefill_tps=prefill_tps, prefill_pct=prefill_pct,
        prefill_done=prefill_done, prefill_total=prefill_total,
        prompt_tokens=prompt_tokens,
        ctx_used=ctx_used, ctx_total=ctx_total, ctx_cached=ctx_cached,
        live=busy, slots=slots, solo=solo, solo_last_ts=solo_last_ts,
        expert_hit_pct=r.get("expert_hit_pct"),
        cache_known=_engine != "strata",
        conv_cache=r.get("conv_cache"),
    )
    _speed_cache[container_name] = (time.time(), out)
    return out


async def test_prompt(container_name: str, prompt: str, max_tokens: int = 256) -> dict:
    """Send a short chat completion to a container's llama-server. Returns dict with reply, tokens, elapsed_s, err."""
    client = _docker_client()
    internal_port: int | None = None
    if client is not None:
        try:
            c = client.containers.get(container_name)
            _, internal_port = _extract_ports(c.attrs or {})
        except (NotFound, DockerException):
            pass
    if internal_port is None:
        return {"ok": False, "err": "container not reachable"}

    # discover a loaded model id first
    loaded, err, _failed, loading, sleeping, unloaded = await _probe_loaded_model(
        container_name, internal_port)
    if not loaded:
        if loading:
            return {"ok": False, "err": f"{loading} is still loading - try again in a moment"}
        if sleeping:
            # Deliberately not woken here: a test prompt is a diagnostic, and silently paying a
            # 9 s reload to answer one would hide the very state the caller wants to know about.
            loaded = sleeping
        elif unloaded:
            # Same reasoning as sleeping: a request WOULD load it (measured 4.6 s warm), but a
            # diagnostic that silently pays a reload - or a 503 when another engine holds the
            # card - reports the wrong thing. Name the state; let the caller decide to pay.
            return {"ok": False, "err": f"{unloaded} is unloaded - the server is up but holds "
                                        "no model; press Load or send a real request"}
        return {"ok": False, "err": err or "no model loaded on this backend"}
    model_id = loaded.split(",")[0].strip()

    url = f"http://{container_name}:{internal_port}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    import time as _time
    t0 = _time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=180.0)) as ac:
            r = await ac.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}", "model": model_id}
    elapsed = _time.time() - t0

    reply = ""
    choices = data.get("choices") or []
    if choices:
        msg = (choices[0] or {}).get("message") or {}
        reply = msg.get("content") or ""
    usage = data.get("usage") or {}
    completion_toks = int(usage.get("completion_tokens") or 0)
    tps = round(completion_toks / elapsed, 1) if elapsed > 0 and completion_toks else None
    return {
        "ok": True,
        "model": model_id,
        "reply": reply,
        "completion_tokens": completion_toks,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "elapsed_s": round(elapsed, 2),
        "tokens_per_s": tps,
    }
