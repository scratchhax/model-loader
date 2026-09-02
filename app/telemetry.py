"""Passive request telemetry, scraped from llama-server's own logs.

llama-server already reports everything worth knowing about a request — prompt speed,
generation speed, and (when speculating) draft acceptance — as `slot print_timing` lines.
Nothing here measures anything: it reads numbers that were already produced by real traffic
and then thrown away. That is the whole appeal. A benchmark button would have to load a model,
warm it, and take the box offline to learn less than this does for free, and what it learned
would describe a synthetic prompt rather than the work actually being asked of the model.

The log lines look like this (with `docker logs -t` prefixing an RFC3339 timestamp):

    [60279] 0.05.690 I srv    load_model: loading model '/models/foo/foo.gguf'
    [60279] 0.05.690 I slot print_timing: id 0 | task 3 | prompt eval time = 1239.27 ms / 8297 tokens (0.15 ms per token, 6695.10 tokens per second)
    [60279] 0.05.690 I slot print_timing: id 0 | task 3 |        eval time =  777.37 ms /  111 tokens (7.07 ms per token,  141.50 tokens per second)
    [60279] 0.05.690 I slot print_timing: id 0 | task 3 | draft acceptance = 0.62963 ( 51 accepted / 81 generated), mean len = 2.13

`[PID]` identifies the server instance the router spawned, which is what ties a measurement to
a model: the router logs the argv (including `--alias`, i.e. the models.ini section name) just
before the instance announces which GGUF it is loading. Binding is done by matching the model
PATH out of the argv against the path in the load line rather than by assuming the two events
are adjacent, because the router is free to interleave them when running several instances.

Every parse here is best-effort by design. These are human-readable log strings, not an API;
upstream can reword them in any release. A parser that finds nothing must degrade to showing
no measurements, never to breaking the page that asked for them.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from . import db

# A request generating a handful of tokens is nearly all fixed overhead — prompt processing,
# sampler setup, first-token latency — so its apparent tok/s is meaningless and reliably
# extreme in both directions. Excluding them is what stops one 3-token "ok" from dragging the
# numbers around; the median below absorbs whatever outliers survive this floor.
_MIN_GEN_TOKENS = 16

# Read at most this many log lines per ingest. The container has no log rotation configured,
# so the file grows without bound; capping the read keeps ingest cost flat as it does. Samples
# already stored are unaffected — this limits only how far back a single pass can recover.
_TAIL_LINES = 20000

# Re-scraping on every panel render would parse the same 20k lines several times per page.
_INGEST_INTERVAL_S = 20.0
_LAST_INGEST_KEY = "telemetry_last_ingest"

# The router announces an instance as "... with name=<alias> on port <N>", and that <N> is
# exactly the number the instance then prefixes its own lines with, so it is the binding key.
# A second header line ("... with args:") sits between the announcement and the argv, and must
# not be mistaken for the end of the block.
_RE_SPAWN = re.compile(r"srv\s+load: spawning server instance with name=(\S+) on port (\d+)")
_RE_SPAWN_ARGS = re.compile(r"srv\s+load: spawning server instance with args:")
_RE_ARGV = re.compile(r"srv\s+load:\s{2,}(\S.*?)\s*$")
_RE_LOAD = re.compile(r"\[(\d+)\].*?srv\s+load_model: loading model '([^']+)'")
_RE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\S+?)\s")

# Recorded alongside spec-type so a change to any of them is visible as a different config.
# Draft depth and the confidence gate both change what an acceptance rate means, so averaging
# across a change to either silently mixes two populations.
_SPEC_KNOB_FLAGS = ("--spec-draft-n-max", "--spec-draft-n-min", "--spec-draft-p-min",
                    "--draft-p-min", "--draft-max", "--draft-min")

# Below this, a single instance has too little data to stand on its own and stats fall back to
# pooling across instances - flagged as such rather than presented as one configuration.
_MIN_SAMPLES_PER_CONFIG = 5

_RE_PROMPT = re.compile(
    r"\[(\d+)\].*?print_timing: id\s+(\d+) \| task (\d+) \| prompt eval time =\s*"
    r"([\d.]+) ms /\s*(\d+) tokens \(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)")
_RE_EVAL = re.compile(
    r"\[(\d+)\].*?print_timing: id\s+(\d+) \| task (\d+) \|\s+eval time =\s*"
    r"([\d.]+) ms /\s*(\d+) tokens \(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)")
_RE_DRAFT = re.compile(
    r"\[(\d+)\].*?print_timing: id\s+(\d+) \| task (\d+) \| draft acceptance = ([\d.]+) "
    r"\(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)")


@dataclass
class Sample:
    ts: float
    instance: str
    task: int
    model_path: str = ""
    alias: str = ""
    spec_type: str = ""
    prompt_tokens: int = 0
    prompt_tps: float = 0.0
    gen_tokens: int = 0
    gen_tps: float = 0.0
    draft_acc: float | None = None
    draft_len: float | None = None


@dataclass
class Stats:
    """Robust summary of a group of samples.

    The headline is the MEDIAN, not the mean. Generation rate on a shared box is nowhere near
    normally distributed: a request that queues behind another, or lands while a second model
    is being loaded, reads as a fraction of the true rate, and a single such sample drags a
    mean somewhere misleading. The median ignores it by construction. p25/p75 travel alongside
    so the display can show spread instead of implying a precision the sample size cannot
    support.
    """
    n: int = 0
    gen_p50: float = 0.0
    gen_p25: float = 0.0
    gen_p75: float = 0.0
    prompt_p50: float = 0.0
    draft_acc_p50: float | None = None
    draft_len_p50: float | None = None
    draft_n: int = 0
    last_ts: float = 0.0
    spec_types: list[str] = field(default_factory=list)
    # True when the samples come from a single server instance, i.e. one configuration. The
    # router respawns an instance on any argv change, so an instance boundary is exactly a
    # config boundary - which makes this the difference between "your current setup does this"
    # and "the last few setups averaged out to this".
    single_config: bool = True
    config_count: int = 1

    @property
    def age_str(self) -> str:
        """How stale the newest sample is. Measurements taken under a configuration the user
        has since changed are worse than no measurement, so the age is shown alongside."""
        if not self.last_ts:
            return ""
        d = max(0.0, time.time() - self.last_ts)
        if d < 90:
            return "just now"
        for cutoff, size, unit in ((3600, 60, "min"), (86400, 3600, "hour"), (None, 86400, "day")):
            if cutoff is None or d < cutoff:
                n = int(d // size)
                return f"{n} {unit}{'s' if n != 1 else ''} ago"
        return ""

    @property
    def spread_pct(self) -> int:
        """p25-p75 spread as a percentage of the median. A wide spread means the median is a
        summary of genuinely varied conditions, not a stable rate to plan around."""
        if self.gen_p50 <= 0:
            return 0
        return int(round(100.0 * (self.gen_p75 - self.gen_p25) / self.gen_p50))


def _pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile. No interpolation: with n in the tens, interpolating between
    two neighbouring samples invents precision that is not in the data."""
    if not values:
        return 0.0
    s = sorted(values)
    i = int(round(q * (len(s) - 1)))
    return s[max(0, min(i, len(s) - 1))]


def _parse_ts(line: str) -> float:
    m = _RE_TS.match(line)
    if not m:
        return 0.0
    try:
        return datetime.fromisoformat(m.group(1).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def parse_log(text: str) -> list[Sample]:
    """Turn raw `docker logs -t` output into completed samples.

    A request's three timing lines arrive separately, so samples accumulate in a dict keyed by
    (instance, task) and are emitted once the whole text has been walked. A request whose lines
    straddle the tail boundary simply comes out incomplete, and the caller drops it.
    """
    pending_alias = ""
    pending_port = ""
    pending_argv: list[str] = []
    by_pid: dict[str, tuple[str, str, str]] = {}      # port -> (path, alias, spec_type)
    samples: dict[tuple[str, int], Sample] = {}

    def _flush_spawn() -> None:
        """Close an argv block, recording the spec settings it declared."""
        nonlocal pending_alias, pending_port, pending_argv
        if pending_port:
            path, spec = "", ""
            knobs: list[str] = []
            for i, tok in enumerate(pending_argv):
                if tok in ("-m", "--model") and i + 1 < len(pending_argv):
                    path = pending_argv[i + 1]
                elif tok == "--spec-type" and i + 1 < len(pending_argv):
                    spec = pending_argv[i + 1]
                elif tok in _SPEC_KNOB_FLAGS and i + 1 < len(pending_argv):
                    knobs.append(f"{tok.split('-')[-1]}={pending_argv[i + 1]}")
            prev = by_pid.get(pending_port, ("", "", ""))
            if spec and knobs:
                spec = spec + " (" + " ".join(knobs) + ")"
            by_pid[pending_port] = (path or prev[0], pending_alias or prev[1], spec)
        pending_alias, pending_port, pending_argv = "", "", []

    for line in text.splitlines():
        ts = _parse_ts(line)

        m = _RE_SPAWN.search(line)
        if m:
            _flush_spawn()
            pending_alias, pending_port = m.group(1), m.group(2)
            # Bind immediately: even if the argv block is cut off by the tail boundary, the
            # alias is already known and every later line from this instance can be attributed.
            by_pid[pending_port] = ("", pending_alias, "")
            continue

        if pending_port:
            # The "with args:" header sits between the announcement and the argv list; treating
            # it as a non-argv line would end the block before a single argument was read.
            if _RE_SPAWN_ARGS.search(line):
                continue
            m = _RE_ARGV.search(line)
            if m:
                pending_argv.append(m.group(1))
                continue
            _flush_spawn()

        m = _RE_LOAD.search(line)
        if m:
            _flush_spawn()
            pid, path = m.group(1), m.group(2)
            prev = by_pid.get(pid, ("", "", ""))
            by_pid[pid] = (path, prev[1], prev[2])
            continue

        for rx, kind in ((_RE_PROMPT, "prompt"), (_RE_EVAL, "eval"), (_RE_DRAFT, "draft")):
            m = rx.search(line)
            if not m:
                continue
            pid, task = m.group(1), int(m.group(3))
            key = (pid, task)
            s = samples.get(key)
            if s is None:
                path, alias, spec = by_pid.get(pid, ("", "", ""))
                s = Sample(ts=ts, instance=pid, task=task,
                           model_path=path, alias=alias, spec_type=spec)
                samples[key] = s
            if ts:
                s.ts = ts
            if kind == "prompt":
                s.prompt_tokens, s.prompt_tps = int(m.group(5)), float(m.group(6))
            elif kind == "eval":
                s.gen_tokens, s.gen_tps = int(m.group(5)), float(m.group(6))
            else:
                s.draft_acc, s.draft_len = float(m.group(4)), float(m.group(7))
            break

    return list(samples.values())


def ingest(container_names: list[str], *, force: bool = False) -> int:
    """Scrape the named containers and store whatever is not already held.

    Deliberately re-reads the whole tail rather than tracking a watermark: storage is keyed on
    (backend, instance, task), so re-ingesting is a no-op and the pass is self-healing if a
    read fails or the app restarts mid-window. There is no partial-window bookkeeping to get
    wrong, which for a homelab tool is worth more than the saved parsing.
    """
    now = time.time()
    if not force:
        try:
            last = float(db.get_setting(_LAST_INGEST_KEY, "0") or 0)
        except ValueError:
            last = 0.0
        if now - last < _INGEST_INTERVAL_S:
            return 0

    from . import services  # late import: services pulls in config, which reads the environment

    total = 0
    for name in container_names:
        try:
            client = services._docker_client()
            if client is None:
                continue
            c = client.containers.get(name)
            raw = c.logs(tail=_TAIL_LINES, stdout=True, stderr=True, timestamps=True)
            text = raw.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - telemetry must never break a page render
            continue
        rows = [s for s in parse_log(text) if s.gen_tokens > 0 and s.ts > 0]
        try:
            total += db.record_timings(name, rows)
        except Exception:  # noqa: BLE001
            continue

    db.set_setting(_LAST_INGEST_KEY, str(now))
    return total


def stats_for(*, model_path: str = "", alias: str = "", limit: int = 400) -> Stats:
    """Robust throughput summary for one model, over its most recent samples.

    Bounded to `limit` so the numbers track the configuration in use now rather than averaging
    across every setting the model has ever been run under.
    """
    try:
        rows = db.recent_timings(model_path=model_path, alias=alias,
                                 min_gen_tokens=_MIN_GEN_TOKENS, limit=limit)
    except Exception:  # noqa: BLE001
        return Stats()
    if not rows:
        return Stats()

    # Prefer the newest instance on its own. A respawn means the argv changed, so mixing
    # instances mixes configurations - and no amount of outlier trimming detects that, because
    # every sample is a perfectly valid measurement of a setup that no longer applies.
    all_instances = {r["instance"] for r in rows}
    newest = rows[0]["instance"]
    current = [r for r in rows if r["instance"] == newest]
    if len(current) >= _MIN_SAMPLES_PER_CONFIG:
        rows, single = current, True
    else:
        single = len(all_instances) <= 1

    gen = [r["gen_tps"] for r in rows if r["gen_tps"]]
    prompt = [r["prompt_tps"] for r in rows if r["prompt_tps"]]
    acc = [r["draft_acc"] for r in rows if r["draft_acc"] is not None]
    dlen = [r["draft_len"] for r in rows if r["draft_len"] is not None]
    return Stats(
        n=len(rows),
        gen_p50=_pct(gen, 0.50), gen_p25=_pct(gen, 0.25), gen_p75=_pct(gen, 0.75),
        prompt_p50=_pct(prompt, 0.50),
        draft_acc_p50=_pct(acc, 0.50) if acc else None,
        draft_len_p50=_pct(dlen, 0.50) if dlen else None,
        draft_n=len(acc),
        last_ts=max((r["ts"] for r in rows), default=0.0),
        spec_types=sorted({(r["spec_type"] or "") for r in rows} - {""}),
        single_config=single,
        config_count=len({r["instance"] for r in rows}),
    )
