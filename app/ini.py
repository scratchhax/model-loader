from __future__ import annotations

import configparser
import io
import os
import re
import shlex
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .config import settings

BACKUPS_TO_KEEP = 10

# ---- field schema: drives form + serialization ----

@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: str            # int | text | bool | select
    choices: tuple[str, ...] = ()
    placeholder: str = ""
    help: str = ""


_KV_TYPES = ("", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")
_ONOFF = ("", "on", "off")

COMMON_FIELDS: tuple[Field, ...] = (
    Field("model", "Model file (relative to /models)", "text", placeholder="auto (section-name.gguf); set for sharded",
          help="Explicit model path relative to /models. Leave blank for flat files (llama-server resolves section-name.gguf automatically). Required for sharded models: point at the first shard, e.g. MyModel-Q4_K_M/MyModel-Q4_K_M-00001-of-00003.gguf."),
    Field("ctx-size", "Context size", "int", placeholder="e.g. 8192, 131072, 196608",
          help="Prompt context length in tokens. 0 = load from model metadata. Larger uses more VRAM for KV cache."),
    Field("ngl", "GPU layers", "text", placeholder="999 (all) or an integer or 'auto'",
          help="Number of layers offloaded to VRAM. 999 or 'all' forces every layer; lower values push work to CPU."),
    Field("flash-attn", "Flash attention", "select", choices=("", "on", "off", "auto"),
          help="Fused attention kernel that reduces KV cache memory and speeds prompt processing. 'auto' probes support at load."),
    Field("cache-type-k", "KV cache — K quant", "select", choices=_KV_TYPES,
          help="Data type for the K side of the KV cache. q8_0 is a strong VRAM saver with minimal quality loss. f16 = default."),
    Field("cache-type-v", "KV cache — V quant", "select", choices=_KV_TYPES,
          help="Data type for the V side of the KV cache. Keep this equal to K. Going below q8_0 saves VRAM but "
               "can silently fall off the CUDA flash-attention fast path and run attention on the CPU — measured "
               "15x slower prompt eval (1737 -> 115 tok/s) on a 27B with a 256-wide head dim. The fallback is "
               "invisible on short prompts, so if you change this, benchmark a LONG one and watch GPU utilization."),
    Field("parallel", "Parallel slots (--np)", "int", placeholder="1",
          help="Number of concurrent generation slots the server hosts. -1 = auto. Each slot needs its own KV allocation."),
    Field("jinja", "Enable --jinja templating", "bool",
          help="Use the model's built-in Jinja chat template (from tokenizer.chat_template). Needed for many modern chat models."),
    Field("chat-template", "Chat template", "text", placeholder="name or inline template",
          help="Override the chat template. Accepts a built-in name (llama3, chatml, gemma, etc.) or an inline Jinja string."),
)

RUNTIME_FIELDS: tuple[Field, ...] = (
    Field("batch-size", "Batch size (logical)", "int", placeholder="2048",
          help="Logical maximum batch size for prompt processing. Higher = faster prompt eval but more scratch memory."),
    Field("ubatch-size", "Ubatch size (physical)", "int", placeholder="512",
          help="Physical batch size the backend actually dispatches. Lower it if OOM during prompt processing."),
    Field("keep", "Tokens to keep from prompt", "int", placeholder="0",
          help="On context-shift, how many tokens from the very start of the prompt to preserve. -1 = keep all (no shifting)."),
    Field("swa-full", "Full-size SWA cache", "bool",
          help="Use full-size cache for sliding-window attention models (Gemma, Mistral SWA variants). Higher VRAM, avoids some artifacts."),
    Field("split-mode", "Multi-GPU split mode", "select", choices=("", "none", "layer", "row", "tensor"),
          help="How to split a model across multiple GPUs. none = single GPU, layer = pipeline-split (default), row = per-row parallel, tensor = experimental."),
    Field("main-gpu", "Main GPU index", "int", placeholder="0",
          help="Which GPU holds the model when split-mode=none, or which GPU holds intermediates/KV when split-mode=row."),
    Field("tensor-split", "Tensor split ratio", "text", placeholder="e.g. 3,1",
          help="Comma-separated proportions of the model to place on each GPU. '3,1' = 75% on GPU0, 25% on GPU1."),
    Field("load-mode", "Model load mode", "select", choices=("", "none", "mmap", "mlock", "mmap+mlock", "dio"),
          help="mmap = default memory-map, mlock = lock in RAM (no swap), mmap+mlock = both, dio = direct I/O when supported."),
    Field("numa", "NUMA optimizations", "select", choices=("", "distribute", "isolate", "numactl"),
          help="Enable NUMA hints for multi-socket boxes. distribute = spread threads, isolate = stay on one node, numactl = use provided map."),
    Field("device", "Device list", "text", placeholder="e.g. CUDA0,CUDA1",
          help="Comma-separated devices to use for offloading. 'none' disables offload. Use --list-devices at CLI to enumerate."),
    Field("fit", "Auto-fit VRAM", "select", choices=_ONOFF,
          help="When on (default), llama-server auto-tunes any unset args to fit the model in device memory."),
    Field("fit-target", "Fit target margin (MiB per device)", "text", placeholder="1024 or 512,1024",
          help="Reserved headroom per device for --fit. Single value broadcasts to all devices; comma list assigns per-device. Default 1024."),
    Field("fit-ctx", "Fit minimum ctx", "int", placeholder="4096",
          help="The smallest ctx-size --fit is allowed to fall back to. Prevents fit from shrinking your context below this."),
    Field("kv-offload", "KV cache offload to GPU", "select", choices=_ONOFF,
          help="Whether KV cache lives on the GPU. Default on. Disable to save VRAM at the cost of a lot of PCIe traffic."),
    Field("repack", "Weight repacking", "select", choices=_ONOFF,
          help="Repack weights for faster on-device access. Default on. Rarely worth disabling."),
    Field("op-offload", "Host operation offload", "select", choices=_ONOFF,
          help="Offload host-side tensor operations to the device. Default on."),
    Field("no-host", "Bypass host buffer (--no-host)", "bool",
          help="Skip the host staging buffer, allowing extra device buffers to be used. Advanced."),
    Field("check-tensors", "Check tensor data at load", "bool",
          help="Scan tensor data for invalid values at load. Diagnostic use; slows load significantly."),
    Field("cont-batching", "Continuous batching", "select", choices=_ONOFF,
          help="Interleave prompt+generation across slots. Default on. Off = one request at a time."),
    Field("context-shift", "Context shift", "select", choices=_ONOFF,
          help="On overflow, shift out oldest tokens (keeping --keep from the start) to make room. Default on for endless-chat use."),
    Field("cache-reuse", "Prefix cache reuse (min tokens)", "int", placeholder="1",
          help="Reuse KV cache from a previous request when the new prompt shares a prefix. Set to 1 to enable — huge TTFT win for chat continuations (only new tokens are prompt-evaluated). 0 disables. Free, safe, should almost always be on."),
    Field("kv-unified", "Unified KV cache", "select", choices=_ONOFF,
          help="Share one KV cache across all slots (saves memory) vs. per-slot caches (better cache locality)."),
    Field("cache-ram", "Cache RAM budget (MiB)", "int", placeholder="8192  ·  -1 = unlimited",
          help="Server-side RAM budget for prefix / prompt caches. -1 = unlimited, 0 = disable."),
    Field("ctx-checkpoints", "SWA / ctx checkpoints", "int", placeholder="e.g. 8",
          help="How many rolling context checkpoints to retain. More = faster context restoration on branching, more memory."),
    Field("checkpoint-min-step", "Checkpoint min step", "int", placeholder="8192",
          help="Minimum token spacing between checkpoints. Default 8192."),
    Field("warmup", "Warmup run at startup", "select", choices=_ONOFF,
          help="Run a tiny inference at startup to force lazy init. Default on. Off saves ~1s at boot."),
    Field("override-tensor", "Override tensor buffer", "text", placeholder="pattern=buffer,...",
          help="Force certain tensors onto a specific buffer/device. Advanced: e.g. 'blk\\.\\d+\\.ffn_.*=CPU'."),
    Field("override-kv", "Override GGUF metadata", "text", placeholder="KEY=TYPE:VALUE,...",
          help="Force a metadata key at load time. Types: int/float/bool/str. Example: 'tokenizer.ggml.add_bos_token=bool:false'."),
)

ROPE_FIELDS: tuple[Field, ...] = (
    Field("rope-scaling", "RoPE scaling method", "select", choices=("", "none", "linear", "yarn"),
          help="How to extend context beyond the trained length. yarn = the modern choice for most long-context models."),
    Field("rope-scale", "RoPE context scale", "text", placeholder="float",
          help="Multiplier that expands context by factor N. Example: 4.0 with a 32K-native model gives 128K context."),
    Field("rope-freq-base", "RoPE base frequency", "text", placeholder="float",
          help="RoPE base frequency (theta). Loaded from model unless overridden. Higher values push effective context out."),
    Field("rope-freq-scale", "RoPE freq scale", "text", placeholder="float",
          help="Alternate way to expand context: multiplies by 1/N. rope-scale = 4 is equivalent to rope-freq-scale = 0.25."),
    Field("yarn-orig-ctx", "YaRN original ctx", "int", placeholder="0 = model default",
          help="Model's native training context length. YaRN uses this to compute the extrapolation ratio."),
    Field("yarn-ext-factor", "YaRN extrapolation factor", "text", placeholder="float; 0 = full interpolation",
          help="Mix between interpolation and extrapolation. 0 = pure interpolation (safest quality), 1 = pure extrapolation."),
    Field("yarn-attn-factor", "YaRN attention factor", "text", placeholder="float",
          help="Scale factor for √(t) attention magnitude. Default -1 = library default."),
    Field("yarn-beta-slow", "YaRN β slow (alpha)", "text", placeholder="float",
          help="High-frequency correction dim. Default -1 = library default."),
    Field("yarn-beta-fast", "YaRN β fast (beta)", "text", placeholder="float",
          help="Low-frequency correction dim. Default -1 = library default."),
)

MOE_FIELDS: tuple[Field, ...] = (
    Field("cpu-moe", "Keep ALL MoE weights on CPU", "bool",
          help="For Mixture-of-Experts models: keep every expert on CPU. Big VRAM saver at the cost of PCIe/RAM bandwidth per token."),
    Field("n-cpu-moe", "Keep first N layers' MoE on CPU", "int", placeholder="e.g. 20",
          help="Partial variant: only push the first N layers' experts to CPU. Tune to fit exactly in VRAM."),
)

MULTIMODAL_FIELDS: tuple[Field, ...] = (
    Field("mmproj", "Multimodal projector file", "text", placeholder="/models/name-mmproj.gguf",
          help="Path to the vision/audio projector GGUF that pairs with a multimodal model. Required for image/audio input."),
    Field("mmproj-url", "Multimodal projector URL", "text", placeholder="https://…",
          help="Alternative to mmproj: download the projector at startup from a URL."),
    Field("mmproj-auto", "Auto-download mmproj", "select", choices=_ONOFF,
          help="Automatically fetch a matching mmproj when the model repo publishes one."),
    Field("mmproj-offload", "GPU-offload mmproj", "select", choices=_ONOFF,
          help="Run the projector on GPU. Default on when a GPU is present."),
    Field("image-min-tokens", "Image min tokens", "int",
          help="Minimum tokens each image is allowed to consume in the prompt context (vision models only)."),
    Field("image-max-tokens", "Image max tokens", "int",
          help="Maximum tokens each image is allowed to consume. Tuning knob for latency vs quality on vision."),
    Field("mtmd-batch-max-tokens", "Multimodal batch max tokens", "int",
          help="Max image tokens per image-encoding batch. Lower if the projector OOMs; higher for throughput."),
)

SPECULATIVE_FIELDS: tuple[Field, ...] = (
    Field("spec-type", "Speculative decoding type", "select",
          choices=("", "none", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash",
                   "draft-dspark", "ngram-simple", "ngram-map-k", "ngram-map-k4v",
                   "ngram-mod", "ngram-cache"),
          help="Strategy. draft-* runs a small model to propose tokens (draft-mtp for a head shipped with the "
               "weights, draft-simple for a separate GGUF sharing the tokenizer). ngram-* needs no model at all "
               "and replays literal repeats from the prompt. llama.cpp accepts a comma-separated list, so "
               "'draft-mtp,ngram-simple' runs both. Autoconfig sets this from the workload profile."),
    Field("spec-draft-model", "Draft model path", "text", placeholder="/models/<stem>/draft.gguf",
          help="Local path to a small 'draft' GGUF used to propose tokens the main model verifies. For Qwen3 with MTP, use the model's official MTP draft head."),
    Field("spec-draft-hf", "Draft HF repo", "text", placeholder="user/repo[:quant]",
          help="Hugging Face repo to pull the draft model from (llama-server auto-downloads at startup)."),
    Field("spec-draft-ngl", "Draft GPU layers", "int", placeholder="999",
          help="How many draft-model layers to place on GPU; 999 offloads everything. Leave this high: a draft "
               "evaluated on the CPU is slower than the GPU model it is supposed to be racing ahead of, which "
               "defeats the entire mechanism."),
    Field("spec-draft-device", "Draft device list", "text",
          help="Comma-separated devices for the draft model. Often the same as main; sometimes a spare GPU."),
    Field("spec-draft-n-max", "Draft n_max", "int", placeholder="3",
          help="How deep to guess: max draft tokens per attempt (llama.cpp default 3). Multiplies the win on "
               "predictable text and the waste on unpredictable text — 8 for code, 2 for prose. Draft models "
               "only; ngram-* strategies take their length from spec-ngram-*-size-m instead."),
    Field("spec-draft-n-min", "Draft n_min", "int", placeholder="0",
          help="Floor on draft length — always attempt at least this many (default 0)."),
    Field("spec-draft-p-min", "Draft p_min", "text", placeholder="0.00",
          help="How sure the drafter must be before it bothers (default 0.00 = always try). Raise it to decline "
               "the marginal bets when acceptance is poor; lower it when the text is predictable enough that "
               "even mediocre guesses land."),
    Field("spec-draft-p-split", "Draft p_split", "text", placeholder="0.10",
          help="Probability at which speculation branches into a draft tree rather than a single line (default 0.10)."),
    Field("spec-draft-cpu-moe", "Draft: all MoE on CPU", "bool",
          help="For MoE draft models: keep experts on CPU."),
    Field("spec-draft-n-cpu-moe", "Draft: N MoE layers on CPU", "int",
          help="Partial CPU offload for the draft's MoE layers."),
    Field("cache-type-k-draft", "Draft KV cache K", "select", choices=_KV_TYPES,
          help="KV K quant for the draft model. Can differ from the main model's."),
    Field("cache-type-v-draft", "Draft KV cache V", "select", choices=_KV_TYPES,
          help="KV V quant for the draft model."),
    Field("lookup-cache-static", "Static lookup cache (path)", "text",
          help="For ngram lookup decoding: path to a read-only n-gram store built from a corpus."),
    Field("lookup-cache-dynamic", "Dynamic lookup cache (path)", "text",
          help="Writable n-gram cache the server updates as it serves. Persists to disk between restarts."),
)

LORA_FIELDS: tuple[Field, ...] = (
    Field("lora", "LoRA adapter(s)", "text", placeholder="/path/a.gguf,/path/b.gguf",
          help="Path(s) to LoRA adapter GGUF files applied at load. Comma-separated for multiple."),
    Field("lora-scaled", "LoRA scaled adapters", "text", placeholder="a.gguf:0.5,b.gguf:1.0",
          help="LoRA adapters with per-adapter scale factors. Useful for blending styles."),
    Field("control-vector", "Control vector(s)", "text",
          help="Path(s) to control-vector files that steer generation. Comma-separated for multiple."),
    Field("control-vector-scaled", "Control vector scaled", "text", placeholder="a.gguf:0.7,...",
          help="Control vectors with per-vector scale factors."),
    Field("control-vector-layer-range", "Control vector layer range", "text", placeholder="START END",
          help="Restrict where control vectors apply. Two integers, e.g. '10 30' for layers 10 through 30 inclusive."),
)

CPU_FIELDS: tuple[Field, ...] = (
    Field("threads", "Threads (generation)", "int", placeholder="-1 = auto",
          help="CPU threads used during single-token generation. Usually = physical cores. -1 auto-picks."),
    Field("threads-batch", "Threads (batch/prompt)", "int",
          help="Threads used during batched prompt processing. Higher can help prompt eval on many-core CPUs."),
    Field("cpu-mask", "CPU affinity mask (hex)", "text",
          help="Bitmask of CPU cores to pin threads to, arbitrarily long hex. Complements cpu-range."),
    Field("cpu-range", "CPU range lo-hi", "text",
          help="Range of CPU indices for affinity, e.g. '0-15'."),
    Field("cpu-strict", "Strict CPU placement", "bool",
          help="Force threads to stay on the mask/range. Off = kernel may migrate."),
    Field("prio", "Process priority", "select", choices=("", "-1", "0", "1", "2", "3"),
          help="Scheduling priority: -1 low, 0 normal (default), 1 medium, 2 high, 3 realtime."),
    Field("poll", "Polling level (0-100)", "int", placeholder="50",
          help="How aggressively worker threads busy-wait for work. Higher = lower latency, more idle CPU."),
)

REASONING_FIELDS: tuple[Field, ...] = (
    Field("reasoning", "Enable reasoning / thinking", "select", choices=("", "on", "off", "auto"),
          help="Whether the model uses its reasoning (thinking) mode. 'auto' = detected from the chat template "
               "(the right default for most reasoning models). Force 'on' to always think, 'off' to skip thinking "
               "for lower latency."),
    Field("reasoning-format", "Reasoning output format", "select", choices=("", "none", "deepseek", "deepseek-legacy"),
          help="How thoughts appear in responses. 'deepseek' = thoughts in message.reasoning_content (OpenAI-compatible clients like OpenWebUI hide them). "
               "'deepseek-legacy' = keeps <think> tags in content too. 'none' = raw, unparsed. Default: auto."),
    Field("reasoning-budget", "Reasoning token budget", "int",
          placeholder="-1 unrestricted · 0 = skip thinking · N = hard cap",
          help="Max tokens the model can spend on reasoning per response. -1 = unlimited (default), 0 = skip thinking entirely, "
               "N>0 = hard cap. Rough map to OpenAI 'reasoning_effort': low ≈ 512, medium ≈ 2048, high ≈ 8192 or -1."),
    Field("reasoning-budget-message", "Reasoning-budget cutoff message", "text",
          placeholder="e.g. \"Final answer:\"",
          help="Message injected right before the end-of-thinking tag when the budget runs out. Nudges the model to wrap up."),
    Field("reasoning-preserve", "Preserve reasoning across turns", "select", choices=("", "on", "off"),
          help="Keep the full thinking trace in the conversation history, not just the last turn. Needed for templates with "
               "'supports_preserve_reasoning'. Uses more context per turn."),
)

MISC_FIELDS: tuple[Field, ...] = (
    Field("alias", "Alias override", "text",
          help="Server-facing model name. Defaults to the section name; only override if you need a different API-visible id."),
    Field("tags", "Tags (comma-separated)", "text",
          help="Informational tags surfaced in /v1/models. Not used for routing — cosmetic."),
    Field("embedding", "Embedding-only mode", "bool",
          help="Restrict this slot to embedding requests. Use with dedicated embedding models."),
    Field("rerank", "Rerank-only mode", "bool",
          help="Enable the /rerank endpoint. Use with dedicated cross-encoder / rerank models."),
    Field("pooling", "Embedding pooling", "select", choices=("", "none", "mean", "cls", "last", "rank"),
          help="Pooling strategy over token embeddings. Model default if unspecified."),
    Field("embd-normalize", "Embedding normalize", "int",
          help="L-norm for output embeddings. -1 = none, 0 = max-abs, 1 = taxicab, 2 = Euclidean (default)."),
    Field("chat-template-file", "Chat template file (path)", "text",
          help="Read a Jinja chat template from a file. Useful for external template management."),
    Field("chat-template-kwargs", "Chat template kwargs (JSON)", "text",
          help="Extra variables passed to the Jinja template. JSON object, e.g. '{\"enable_thinking\": true}'."),
    Field("special", "Emit special tokens", "bool",
          help="Include model-defined special tokens in output. Off by default; on for debugging tokenization."),
    Field("backend-sampling", "Backend sampling (experimental)", "bool",
          help="Move sampling into the backend for speed. Experimental — may interact oddly with some sampler combos."),
    Field("verbosity", "Log verbosity (0-5)", "int",
          help="0 generic, 1 error, 2 warn, 3 info (default), 4 trace, 5 debug. Higher = noisier logs."),
)


# Order matters — this is what the form renders. (label, fields, open_by_default)
FORM_TIERS: tuple[tuple[str, tuple[Field, ...], bool], ...] = (
    ("Common", COMMON_FIELDS, True),
    ("Runtime tuning", RUNTIME_FIELDS, False),
    ("RoPE / YaRN", ROPE_FIELDS, False),
    ("Mixture-of-Experts offload", MOE_FIELDS, False),
    ("Multimodal / vision", MULTIMODAL_FIELDS, False),
    ("Speculative decoding", SPECULATIVE_FIELDS, False),
    ("LoRA / control vectors", LORA_FIELDS, False),
    ("CPU threading", CPU_FIELDS, False),
    ("Reasoning / thinking", REASONING_FIELDS, False),
    ("Embeddings & misc", MISC_FIELDS, False),
)

ALL_FIELDS: tuple[Field, ...] = tuple(f for _, group, _ in FORM_TIERS for f in group)
ALL_KNOWN_KEYS = {f.key for f in ALL_FIELDS}


# ---- ini I/O ----

def _new_parser() -> configparser.ConfigParser:
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str  # preserve case
    return cp


def read_ini() -> configparser.ConfigParser:
    cp = _new_parser()
    if settings.models_ini_path.exists():
        cp.read(settings.models_ini_path, encoding="utf-8")
    return cp


def section_names() -> list[str]:
    return read_ini().sections()


_TRUTHY = {"true", "on", "yes", "1"}
_FALSY = {"false", "off", "no", "0"}


_FIELD_KIND = {f.key: f.kind for f in ALL_FIELDS}
# Selects whose "on" is spelled as a bare flag on the command line rather than `--key on`.
_BARE_ON_OFF: set[str] = set()

def to_cli(name: str, items: list[tuple[str, str]]) -> str:
    # If the section has an explicit `model` key, respect it; otherwise default to /models/<section>.gguf
    explicit_model = next((v for k, v in items if k == "model" and v), None)
    # `model` is documented as relative to /models, but autoconfig writes the container-absolute
    # path llama-server actually wants. Accept both rather than emitting /models//models/...
    if not explicit_model:
        model_path = f"/models/{name}.gguf"
    elif explicit_model.startswith("/"):
        model_path = explicit_model
    else:
        model_path = f"/models/{explicit_model}"
    parts: list[str] = ["--alias", shlex.quote(name), "--model", shlex.quote(model_path)]
    for k, v in items:
        if k == "model":
            continue  # already emitted
        val = str(v).strip()
        low = val.lower()
        # Whether a key is a bare flag is a property of the KEY, not of its value. Deciding it
        # from the text alone silently mangles numeric options that happen to read as boolean:
        # `parallel = 1` became a bare `--parallel` (no slot count) and `main-gpu = 0` vanished
        # entirely. Only genuinely boolean fields collapse to a flag.
        if _FIELD_KIND.get(k, "text") in ("bool", "flag"):
            if low in _FALSY:
                continue
            parts.append(f"--{k}")
            continue
        if not val:
            continue
        if low in _TRUTHY or low in _FALSY:
            # on/off-style select (flash-attn, kv-offload, ...): llama-server wants the word.
            parts.append(f"--{k}")
            if k not in _BARE_ON_OFF:
                parts.append(low)
            continue
        parts.append(f"--{k}")
        parts.append(shlex.quote(val))
    return " ".join(parts)


@dataclass
class SectionView:
    name: str
    items: list[tuple[str, str]]   # in file order
    has_file: bool
    matched_file: str | None       # filename with .gguf if found
    cli: str = ""


def _is_companion(filename: str) -> bool:
    """Return True for GGUFs that are companion artefacts, not standalone models.
    These are referenced from a main section (via `mmproj = ...`, `model-draft = ...`, etc.)
    and should not be offered as their own configurable ini sections.

    Covers:
      - mmproj files (multimodal projectors — vision, audio, etc.)
      - draft heads for speculative decoding (Qwen3 MTP, generic -draft-)
    """
    n = filename.lower()
    if "mmproj" in n:
        return True
    # Import lazily to avoid circular: autoconfig imports ini for ALL_KNOWN_KEYS.
    from .autoconfig import _looks_like_draft
    return _looks_like_draft(filename)


def _stems_present() -> dict[str, str]:
    """stem -> display_name (with subdir prefix if in subdir). Scans /models one level deep.
    Excludes companion files (mmproj etc.) — they belong to a main model's section, not their own."""
    out: dict[str, str] = {}
    root = settings.models_dir
    from .utils import shard_key as _sk
    try:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() == ".gguf":
                if _is_companion(p.name):
                    continue
                base, _, _ = _sk(p.name)
                stem = base[:-5] if base.lower().endswith(".gguf") else base
                out[stem] = base
            elif p.is_dir() and not p.name.startswith("."):
                try:
                    kids = list(p.iterdir())
                except OSError:
                    continue
                for sp in kids:
                    if sp.is_file() and sp.suffix.lower() == ".gguf":
                        if _is_companion(sp.name):
                            continue
                        base, _, _ = _sk(sp.name)
                        stem = base[:-5] if base.lower().endswith(".gguf") else base
                        out[stem] = f"{p.name}/{base}"
    except OSError:
        pass
    return out


def section_file_rel(name: str, vals: dict | None = None) -> str | None:
    """models-dir-relative path of the GGUF a section points at, or None.

    Resolves from the section's explicit `model =` value FIRST, and only falls back to
    matching the section name against file stems. That ordering matters: a section can be
    renamed to give a model a short API id, at which point its name no longer resembles the
    filename. Name-matching alone then reports "no GGUF found", which breaks Autoconfig, the
    fit chips and the per-model backend toggles for a section that is perfectly valid.
    """
    v = vals if vals is not None else (get_section(name) or {})
    mp = (v.get("model") or "").strip()
    if mp:
        rel = mp.replace("/models/", "", 1).lstrip("/")
        try:
            if (settings.models_dir / rel).is_file():
                return rel
        except OSError:
            pass
    return _stems_present().get(name)


def sections_by_file() -> dict[str, list[str]]:
    """{models-dir-relative path -> [section name, ...]} in file order.

    The reverse of section_file_rel(): given a file on disk, which sections configure it.
    A section name IS the id llama-server serves under, and it can be renamed away from the
    filename to give a model a short API id, so a file cannot be matched to its section by
    name alone. The value is a LIST because one GGUF can legitimately back several sections —
    the same weights served twice under different ids with different settings (a chat preset
    and a persona preset, say).
    """
    cp = read_ini()
    out: dict[str, list[str]] = {}
    for n in cp.sections():
        rel = section_file_rel(n, dict(cp.items(n)))
        if rel:
            out.setdefault(rel, []).append(n)
    return out


def _files_claimed_by_sections() -> set[str]:
    """Relative paths already referenced by some section, however that section is named."""
    cp = read_ini()
    out: set[str] = set()
    for n in cp.sections():
        rel = section_file_rel(n, dict(cp.items(n)))
        if rel:
            out.add(rel)
    return out


def list_sections() -> list[SectionView]:
    cp = read_ini()
    stems_map = _stems_present()
    out: list[SectionView] = []
    for name in cp.sections():
        # by explicit model= first, so renamed sections keep their file association
        matched = section_file_rel(name, dict(cp.items(name))) or stems_map.get(name)
        items = list(cp.items(name))
        out.append(SectionView(
            name=name, items=items,
            has_file=matched is not None, matched_file=matched,
            cli=to_cli(name, items),
        ))
    return out


def get_section(name: str) -> dict[str, str] | None:
    cp = read_ini()
    if name not in cp.sections():
        return None
    return dict(cp.items(name))


def unregistered_gguf_stems() -> list[str]:
    """Filenames (stem, without .gguf) that don't have a section yet. Scans one subdir level too."""
    have = set(section_names())
    # A file is "registered" if ANY section points at it — not merely when a section shares
    # its name. Without this, renaming a section makes its own model reappear as unregistered.
    claimed = _files_claimed_by_sections()
    return sorted([stem for stem, rel in _stems_present().items()
                   if stem not in have and rel not in claimed])


# ---- writing ----

_SECTION_NAME_RE = re.compile(r"^[A-Za-z0-9._\-+]+$")


def valid_section_name(name: str) -> bool:
    return bool(_SECTION_NAME_RE.match(name))


def suggest_defaults(summary: dict) -> tuple[dict[str, str], list[str]]:
    """Given a GGUF summary (from gguf_meta.summarize), return (field_values, hints).

    field_values: {ini_key: str_value} suitable for pre-filling the form.
    hints: list of short human-readable notes to show above the form.
    """
    fields: dict[str, str] = {}
    hints: list[str] = []

    model = summary.get("model") or {}
    ctx = model.get("context_length")
    if isinstance(ctx, int) and ctx > 0:
        fields["ctx-size"] = str(ctx)

    # Runtime defaults. These used to be supplied by the container's CLI baseline, but
    # compose args OVERRIDE models.ini rather than falling back to it — so the compose
    # commands were reduced to router-only plumbing and every preset must now be
    # self-sufficient. `parallel` in particular is not optional: llama-server defaults
    # to n_slots = 4, which would silently quarter each slot's share of ctx-size.
    fields["ngl"] = "999"
    fields["flash-attn"] = "on"
    fields["cache-type-k"] = "q8_0"
    fields["cache-type-v"] = "q8_0"
    fields["parallel"] = "1"
    fields["split-mode"] = "layer"

    if summary.get("chat_template"):
        fields["jinja"] = "true"

    rope_type = model.get("rope_scaling_type")
    if rope_type and str(rope_type).lower() != "none":
        fields["rope-scaling"] = str(rope_type)
        rope_factor = model.get("rope_scaling_factor")
        if isinstance(rope_factor, (int, float)) and rope_factor > 0:
            fields["rope-scale"] = str(rope_factor)
        rope_orig = model.get("rope_scaling_original_context")
        if isinstance(rope_orig, int) and rope_orig > 0:
            hints.append(f"Model trained on {rope_orig:,} ctx and scaled with {rope_type}; the pre-filled context ({fields.get('ctx-size', 'N/A')}) uses that scaling.")

    experts = model.get("expert_count")
    if isinstance(experts, int) and experts > 1:
        used = model.get("expert_used_count") or "?"
        hints.append(
            f"MoE model detected: {experts} experts, {used} active per token. "
            f"If it doesn't fit in VRAM, add extras like `n-cpu-moe = <N>` to keep the first N layers' experts on CPU, "
            f"or `cpu-moe = true` for all MoE weights on CPU."
        )

    arch = summary.get("arch") or ""
    if arch.lower().startswith("gemma"):
        hints.append("Gemma architecture uses sliding-window attention. If ctx feels tight in VRAM, try `swa-full = true` in advanced.")

    return fields, hints


def upsert_section(name: str, values: dict[str, str], extras_text: str) -> None:
    """Create or replace a section. `values` = known form fields (empty strings skipped).
    `extras_text` = raw 'key = value' lines, one per line, appended (last-wins per key)."""
    cp = read_ini()
    if cp.has_section(name):
        cp.remove_section(name)
    cp.add_section(name)

    # Write known keys first, in the order they appear in the schema
    for f in ALL_FIELDS:
        v = values.get(f.key, "")
        if v == "" or v is None:
            continue
        cp.set(name, f.key, str(v))

    # Then parse extras
    for line in extras_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(";"):
            continue
        if "=" not in s:
            continue
        k, _, v = s.partition("=")
        k, v = k.strip(), v.strip()
        if not k:
            continue
        cp.set(name, k, v)

    _atomic_write(cp)


def delete_section(name: str) -> bool:
    cp = read_ini()
    if not cp.has_section(name):
        return False
    cp.remove_section(name)
    _atomic_write(cp)
    return True


def rename_section(old: str, new: str) -> bool:
    if not valid_section_name(new):
        return False
    cp = read_ini()
    if not cp.has_section(old) or cp.has_section(new):
        return False
    items = list(cp.items(old))
    cp.remove_section(old)
    cp.add_section(new)
    for k, v in items:
        cp.set(new, k, v)
    _atomic_write(cp)
    return True


def _atomic_write(cp: configparser.ConfigParser) -> None:
    path = settings.models_ini_path
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        ts = time.strftime("%Y%m%d-%H%M%S")
        backup = path.with_suffix(path.suffix + f".bak-{ts}")
        n = 1
        while backup.exists():
            backup = path.with_suffix(path.suffix + f".bak-{ts}-{n}")
            n += 1
        try:
            shutil.copy(path, backup)  # not copy2: preserve creation-time mtime, not original
        except OSError:
            pass
        _prune_backups()

    buf = io.StringIO()
    cp.write(buf, space_around_delimiters=True)
    text = buf.getvalue()

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _prune_backups() -> None:
    path = settings.models_ini_path
    pattern = f"{path.name}.bak-*"
    try:
        backups = sorted(path.parent.glob(pattern), reverse=True)
    except OSError:
        return
    for old in backups[BACKUPS_TO_KEEP:]:
        try:
            old.unlink()
        except OSError:
            pass


def list_backups() -> list[tuple[str, float, int]]:
    """Return (name, mtime, size) for each backup file, newest first."""
    path = settings.models_ini_path
    out: list[tuple[str, float, int]] = []
    try:
        for p in sorted(path.parent.glob(f"{path.name}.bak-*"), reverse=True):
            s = p.stat()
            out.append((p.name, s.st_mtime, s.st_size))
    except OSError:
        pass
    return out


def raw_text() -> str:
    if not settings.models_ini_path.exists():
        return ""
    return settings.models_ini_path.read_text(encoding="utf-8")


# ---- helper: convert a section's dict into (form_values, extras_text) ----

def split_section_for_form(values: dict[str, str]) -> tuple[dict[str, str], str]:
    form: dict[str, str] = {}
    extras_lines: list[str] = []
    for k, v in values.items():
        if k in ALL_KNOWN_KEYS:
            form[k] = v
        else:
            extras_lines.append(f"{k} = {v}")
    return form, "\n".join(extras_lines)
