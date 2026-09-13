"""Turn a llama-server log tail into a plain-language diagnosis of a failed load.

When a model fails to load, llama-server *does* say why — but the reason is one line buried in
hundreds of startup lines, phrased in ggml internals. The usual next move is to scroll the log
looking for "something red". This reads the same tail and maps the recognised failure signatures
to a concrete next action, so the real error surfaces inline with a fix rather than a stack of
timestamps.

Two disciplines keep it honest against LIVE logs, not just synthetic ones:

* **Recency.** A router logs a fresh instance for every load, and a benign startup warning from
  many hours ago must not be reported forever. We only scan lines after the most recent sign of
  a successful load (or, failing that, the most recent load attempt) — so an error that a later
  successful load or served request superseded drops out.

* **Attribution, for a router.** A router runs one child llama-server per model and forwards its
  output with a `[port]` prefix, announcing the pairing when it spawns the child. So a failed load
  is read from that instance's OWN lines, and only when its most recent run exited non-zero. The
  model-agnostic recency window above cannot do this: another model serving a request after the
  failure would hide it, while the dashboard still says that model failed; and a model that hit an
  out-of-memory, retried and then loaded fine would be reported as broken.

* **Fail-closed wording.** Every rule demands explicit failure language. A pattern that merely
  mentions `mmproj`, `truncat`, etc. in normal operation must never fire — healthy multimodal
  servers print `mmproj` on every startup, and every finished request prints `truncated = 0`.

Like the telemetry parser, this is best-effort by design: these are human-readable strings, not
an API, and upstream can reword them in any release. A rule that no longer matches simply stops
firing; the caller shows a "no known signature found" fallback rather than a wrong answer. It
never claims success — it only ever proposes what to check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Union


@dataclass(frozen=True)
class Finding:
    error: str      # the offending log line, trimmed of container/pid noise
    hint: str       # the concrete thing to try next
    model: str = "" # the router instance it came from; "" for the server itself


# Leading container log decoration: an RFC3339 timestamp (docker logs -t) then a "[pid]" token
# and a llama.cpp "H H.MMM" uptime stamp. Stripped for display; matching ignores it anyway.
# Both parts are optional: Diagnose reads logs without docker timestamps, where a router child's
# line begins directly with its "[port]".
_RE_DECOR = re.compile(r"^(?:\d{4}-\d{2}-\d{2}T\S+\s+)?(?:\[\d+\]\s+)?")

# Router mode. The router announces each child, then forwards the child's output with a "[port]"
# prefix, then reports how it ended:
#     load: spawning server instance with name=MODEL on port 52493
#     [52493] ... E gguf_init_from_file: failed to open GGUF file '...'
#     operator(): instance name=MODEL exited with status 1
# A clean eviction exits with status 0.
_RE_CHILD = re.compile(r"^(?:\d{4}-\d{2}-\d{2}T\S+\s+)?\[(\d+)\]\s")
_RE_SPAWN = re.compile(r"spawning server instance with name=(\S+) on port (\d+)")
_RE_EXIT = re.compile(r"instance name=(\S+) exited with status (-?\d+)")
_RE_ERROR_LINE = re.compile(r"\sE\s")

# A load completed / the server is serving. Recency scans only lines after the LAST of these: a
# failure older than the most recent success is stale (the router recovered). Deliberately broad,
# since any of these appearing means the box was healthy after whatever preceded it.
_RE_SUCCESS = re.compile(
    r"print_timing|all slots idle|model loaded|server (?:is )?listening|http server (?:listening|started)",
    re.I,
)
# A load was attempted. Fallback boundary when nothing has succeeded yet: the failure, if any,
# must come after the most recent attempt to be the current one.
_RE_ATTEMPT = re.compile(r"loading model|spawning server instance", re.I)

# A rule's hint may be a fixed string or a callable that reads the matched line, for the one
# case where the fix depends on WHICH file failed — a missing `model =` vs a missing `mmproj =`.
Hint = Union[str, Callable[[str], str]]


def _path_hint(line: str) -> str:
    """Fix wording for the file-open rule: name the key that points at the missing file.

    The router prints the offending path (in quotes, or bare). If it's a projector file the key
    is `mmproj =`; otherwise `model =`. This keeps a single deleted-mmproj report from surfacing
    as a generic model-path problem or a projector mismatch, both of which send you chasing the
    wrong thing.
    """
    m = re.search(r"'([^']+)'", line) or re.search(r"(/[^\s'\"]+)", line)
    path = m.group(1) if m else ""
    low = path.lower()
    if "mmproj" in low:
        return ("The projector file named in `mmproj =` doesn't exist at that path (it may have "
                "been deleted or moved). Point `mmproj =` at an existing projector for this exact "
                "model, or clear it to run the model text-only.")
    if path:
        return (f"The `model =` path doesn't resolve inside the container — nothing at {path}. "
                "The models directory must be mounted at the same path in llama-server as it is in Model Loader.")
    return ("A file the server needed wasn't found. Check the section's `model =` path resolves "
            "inside the container — the models directory must be mounted at the same path in llama-server.")


# Ordered most-specific first. When several rules could match, the specific one (KV cache,
# ctx-vs-trained, projector mismatch) is reported before the generic "out of memory"/assert.
_RULES: list[tuple[re.Pattern[str], Hint]] = [
    (re.compile(
        r"failed to allocate buffer for kv cache|kv[_ ]?cache.{0,40}out of memory",
        re.I),
     "The KV cache didn't fit. Lower ctx-size, or shrink the cache by quantizing it — set cache-type-k and cache-type-v to q8_0 (or q4_0)."),

    (re.compile(
        r"exceeds the trained context|context size.{0,30}exceeds|too large for the model|n_ctx_exceeds",
        re.I),
     "ctx-size is above what the model was trained for. Lower ctx-size to the model's trained length, or enable RoPE scaling (rope-scaling = yarn) if you must reach past it."),

    # Projector MISMATCH only — a projector that loads but is incompatible. Explicitly does NOT
    # match a projector file that simply failed to open (that's the path rule below, which names
    # the mmproj key); and requires failure language so a healthy multimodal server's routine
    # `load_mmproj: using mmproj = ...` never fires.
    (re.compile(
        r"(?:mmproj|projector|vision).{0,40}(?:fail|unable|mismatch|incompatib|corrupt|invalid)"
        r"|clip:.{0,30}(?:fail|unable)"
        r"|(?:fail|unable).{0,20}(?:load|init).{0,20}(?:mmproj|projector|vision)",
        re.I),
     "The multimodal projector loaded but isn't usable with this model. Point mmproj at the projector built for the exact same model and quant, or clear it to run the model text-only."),

    # A required file could not be opened. Names `mmproj =` vs `model =` from the path so the one
    # root cause is reported once, with the correct key, framed as *missing* rather than mismatch.
    (re.compile(
        r"failed to open GGUF|gguf_init_from_file.{0,40}fail|could not open|failed to open file|No such file or directory",
        re.I),
     _path_hint),

    (re.compile(
        r"invalid magic|unsupported GGUF|GGUF version|corrupt|unexpected end|premature EOF|short read|file.{0,20}truncat",
        re.I),
     "The GGUF looks corrupt or incomplete — usually an interrupted download. Re-download it, and for a sharded model make sure every shard is present."),

    (re.compile(
        r"out of memory|CUDA out of memory|cudaMalloc|ggml_cuda.{0,40}alloc|ggml_backend_cuda.{0,60}alloc|failed to allocate \d",
        re.I),
     "Ran out of VRAM. Lower n-gpu-layers (ngl), reduce ctx-size, or let llama.cpp place it itself: clear ngl, tensor-split and n-cpu-moe and enable --fit (Autoconfig does exactly this for overflowing MoE models)."),

    (re.compile(
        r"tensor .{0,40}not found|no tensor named|missing tensor|vocab.{0,30}fail|not a valid checkpoint",
        re.I),
     "A required tensor is missing — the file is incomplete or it's the wrong file (a draft/MTP/mmproj loaded as if it were a full model). Re-download, or select the main GGUF."),

    (re.compile(
        r"Address already in use|failed to listen|EADDRINUSE|bind.{0,30}address",
        re.I),
     "The server port is already taken — another llama-server instance or container holds it. Free the port or change --port before this one can start."),

    (re.compile(
        r"GGML_ASSERT|LLAMA_ASSERT|terminate called after|std::bad_alloc|assertion .{0,40}failed",
        re.I),
     "A fatal assertion aborted the server. The specific cause is on the lines immediately above this one in the log."),
]


def _clean(line: str) -> str:
    line = _RE_DECOR.sub("", line).strip()
    if len(line) > 240:
        line = line[:237] + "..."
    return line


def _window(lines: list[str]) -> list[str]:
    """Recency gate: only the lines after the most recent sign of health (or attempt) count."""
    for i in range(len(lines) - 1, -1, -1):
        if _RE_SUCCESS.search(lines[i]):
            return lines[i + 1:]
    for i in range(len(lines) - 1, -1, -1):
        if _RE_ATTEMPT.search(lines[i]):
            return lines[i:]
    return lines


def _match(lines: list[str], model: str = "") -> list[Finding]:
    """Every rule that matches, most recent line per rule, ordered by specificity."""
    findings: list[tuple[int, Finding]] = []
    for idx, (pat, hint) in enumerate(_RULES):
        for line in reversed(lines):
            if pat.search(line):
                text = hint(line) if callable(hint) else hint
                findings.append((idx, Finding(error=_clean(line), hint=text, model=model)))
                break
    findings.sort(key=lambda t: t[0])
    return [f for _i, f in findings]


@dataclass
class _Instance:
    name: str
    port: str
    lines: list[str]
    status: int | None = None       # None = still running, or its end fell outside the tail


def _split_router_log(lines: list[str]) -> tuple[list[_Instance], list[str]]:
    """(child instances in spawn order, the router's own lines).

    A port can be reused by a later instance, so a child line belongs to whichever instance most
    recently claimed that port. Child lines whose spawn fell outside the tail belong to nobody and
    are dropped - such an instance started long ago and is not the load that just failed.
    """
    owner: dict[str, _Instance] = {}
    latest_by_name: dict[str, _Instance] = {}
    instances: list[_Instance] = []
    router: list[str] = []
    for line in lines:
        child = _RE_CHILD.match(line)
        if child:
            inst = owner.get(child.group(1))
            if inst is not None:
                inst.lines.append(line)
            continue
        router.append(line)
        spawn = _RE_SPAWN.search(line)
        if spawn:
            inst = _Instance(name=spawn.group(1), port=spawn.group(2), lines=[])
            owner[inst.port] = inst
            latest_by_name[inst.name] = inst
            instances.append(inst)
            continue
        ended = _RE_EXIT.search(line)
        if ended:
            inst = latest_by_name.get(ended.group(1))
            if inst is not None and inst.status is None:
                try:
                    inst.status = int(ended.group(2))
                except ValueError:
                    pass
    return instances, router


def diagnose(log_tail: str) -> list[Finding]:
    """Map a llama-server log tail to recognised failure signatures + fixes.

    For a router log: each model whose MOST RECENT instance exited non-zero is diagnosed from that
    instance's own lines, so a later successful load of the same model supersedes the failure and
    nothing another model does can hide it. An instance that failed with no signature we know is
    still reported, so the dashboard's "load failed" never leads to a blank. The router's own
    lines are then checked across the whole run, which is where preset errors found at boot live;
    the caller reads the current container run only, so those are current by construction.

    For a plain llama-server (no instances in the tail): the recency window, as before.

    Within a set of lines each rule reports its most recent match, ordered by specificity. An
    empty list means nothing recognised - not "healthy"; the caller must say so.
    """
    if not log_tail:
        return []
    lines = log_tail.splitlines()
    instances, router = _split_router_log(lines)
    if not instances:
        return _match(_window(lines))

    latest: dict[str, _Instance] = {}
    for inst in instances:
        latest[inst.name] = inst

    findings: list[Finding] = []
    for inst in latest.values():
        if inst.status in (None, 0):
            continue
        hits = _match(inst.lines, model=inst.name)
        if not hits:
            last_error = next((ln for ln in reversed(inst.lines) if _RE_ERROR_LINE.search(ln)), "")
            hits = [Finding(
                error=_clean(last_error) or f"instance exited with status {inst.status}",
                hint=(f"{inst.name} exited with status {inst.status} while loading, but its log holds "
                      "no failure this recognises. The lines just before its exit are the ones to read."),
                model=inst.name)]
        findings.extend(hits)
    findings.extend(_match(router))
    return findings
