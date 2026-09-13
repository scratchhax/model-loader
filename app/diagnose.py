"""Turn a llama-server log tail into a plain-language diagnosis of a failed load.

When a model fails to load, llama-server *does* say why — but the reason is one line buried in
hundreds of startup lines, phrased in ggml internals. The usual next move is to scroll the log
looking for "something red". This reads the same tail and maps the recognised failure signatures
to a concrete next action, so the real error surfaces inline with a fix rather than a stack of
timestamps.

Like the telemetry parser, this is best-effort by design: these are human-readable strings, not
an API, and upstream can reword them in any release. A rule that no longer matches simply stops
firing; the caller shows a "no known signature found" fallback rather than a wrong answer. It
never claims success — it only ever proposes what to check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    error: str      # the offending log line, trimmed of container/pid noise
    hint: str       # the concrete thing to try next


# Leading container log decoration: an RFC3339 timestamp (docker logs -t) then a "[pid]" token
# that llama.cpp prints. Stripped so the shown line reads cleanly; the match itself ignores it.
_RE_DECOR = re.compile(r"^\d{4}-\d{2}-\d{2}T\S+\s+(?:\[\d+\]\s+)?")

# Ordered most-specific first: when several rules could match the same log, the specific one
# (KV cache, ctx-vs-trained, mmproj) should win over the generic "out of memory" or assert.
# Each entry is (pattern, hint). Patterns run case-insensitively.
_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(
        r"failed to allocate buffer for kv cache|kv[_ ]?cache.{0,40}out of memory",
        re.I),
     "The KV cache didn't fit. Lower ctx-size, or shrink the cache by quantizing it — set ctk and ctv to q8_0 (or q4_0)."),

    (re.compile(
        r"exceeds the trained context|context size.{0,40}exceeds|n_ctx.{0,40}exceeds|too large for the model|requested context",
        re.I),
     "ctx-size is above what the model was trained for. Lower ctx-size to the model's trained length, or enable RoPE scaling (rope-scaling = yarn) if you need to reach past it."),

    (re.compile(
        r"mmproj|multimodal|projector.{0,40}fail|clip:.{0,30}(fail|unable)|unable to load.{0,20}vision",
        re.I),
     "The multimodal projector (mmproj) failed to load — it almost certainly doesn't match this model. Point the section's mmproj at the projector built for the exact same model and quant."),

    (re.compile(
        r"failed to open GGUF|gguf_init_from_file.{0,40}fail|could not open|No such file or directory|failed to open file",
        re.I),
     "The model path didn't resolve inside the container. Check the section's `model =` path — the models directory must be mounted at the same path in llama-server as it is in Model Loader."),

    (re.compile(
        r"invalid magic|unsupported GGUF|GGUF version|corrupt|truncat|premature EOF|short read",
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
    # ggml prints "[XY] ggml/..." prefixes too; keep the message but cap runaway lines.
    if len(line) > 240:
        line = line[:237] + "..."
    return line


def diagnose(log_tail: str) -> list[Finding]:
    """Map a llama-server log tail to recognised failure signatures + fixes.

    Scans the whole tail, but for each rule reports only the most recent matching line (a load
    retried several times logs the same error over and over; the latest is the relevant one).
    Findings come back ordered by specificity, not log position, so the most actionable line is
    first. An empty list means no known signature matched — not "healthy", just "unrecognised";
    the caller must say so rather than imply success.
    """
    if not log_tail:
        return []
    lines = log_tail.splitlines()
    findings: list[tuple[int, Finding]] = []
    for idx, (pat, hint) in enumerate(_RULES):
        for line in reversed(lines):
            if pat.search(line):
                findings.append((idx, Finding(error=_clean(line), hint=hint)))
                break
    findings.sort(key=lambda t: t[0])
    return [f for _i, f in findings]
