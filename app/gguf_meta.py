"""Minimal GGUF v3 metadata reader. Zero deps beyond stdlib."""
from __future__ import annotations

import struct
import threading
import time
from pathlib import Path
from typing import Any

# GGUFValueType from gguf spec
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY = 6, 7, 8, 9
_UINT64, _INT64, _FLOAT64 = 10, 11, 12

_SCALAR_FMT: dict[int, tuple[str, int]] = {
    _UINT8: ("<B", 1), _INT8: ("<b", 1),
    _UINT16: ("<H", 2), _INT16: ("<h", 2),
    _UINT32: ("<I", 4), _INT32: ("<i", 4),
    _UINT64: ("<Q", 8), _INT64: ("<q", 8),
    _FLOAT32: ("<f", 4), _FLOAT64: ("<d", 8),
    _BOOL: ("<?", 1),
}

MAX_ARRAY_ELEMENTS_KEPT = 8
MAX_STRING_LEN = 200_000

# subset of llama.cpp LlamaFileType — enough to name every real-world GGUF quant
FILE_TYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M",
    32: "BF16", 33: "Q4_0_4_4", 34: "Q4_0_4_8", 35: "Q4_0_8_8",
    36: "TQ1_0", 37: "TQ2_0",
}


class GgufMetaError(Exception):
    pass


_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


def _read_string(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    if n > MAX_STRING_LEN:
        chunk = f.read(n)
        return chunk[:MAX_STRING_LEN].decode("utf-8", errors="replace") + "…[truncated]"
    return f.read(n).decode("utf-8", errors="replace")


def _skip_string(f) -> None:
    n = struct.unpack("<Q", f.read(8))[0]
    f.seek(n, 1)


def _read_value(f, vtype: int):
    if vtype in _SCALAR_FMT:
        fmt, size = _SCALAR_FMT[vtype]
        return struct.unpack(fmt, f.read(size))[0]
    if vtype == _STRING:
        return _read_string(f)
    if vtype == _ARRAY:
        subtype = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        if count > MAX_ARRAY_ELEMENTS_KEPT:
            sample = [_read_value(f, subtype) for _ in range(MAX_ARRAY_ELEMENTS_KEPT)]
            remaining = count - MAX_ARRAY_ELEMENTS_KEPT
            if subtype == _STRING:
                for _ in range(remaining):
                    _skip_string(f)
            elif subtype in _SCALAR_FMT:
                _, size = _SCALAR_FMT[subtype]
                f.seek(size * remaining, 1)
            else:
                raise GgufMetaError(f"unsupported nested array subtype {subtype}")
            return {"_array": True, "count": count, "sample": sample}
        return [_read_value(f, subtype) for _ in range(count)]
    raise GgufMetaError(f"unknown value type {vtype}")


def _read_raw_stream(f) -> dict[str, Any]:
    """Parse GGUF metadata from any seekable binary stream.

    Split out from _read_raw so the same parser can run over a range-fetched header
    (io.BytesIO) as well as a local file — metadata lives at the very start of a GGUF,
    so the first ~1 MB is enough to read every KV pair without pulling the weights.
    """
    if True:
        magic = f.read(4)
        if magic != b"GGUF":
            raise GgufMetaError(f"not a GGUF file (magic={magic!r})")
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]
        out: dict[str, Any] = {
            "_gguf_version": version,
            "_tensor_count": tensor_count,
            "_kv_count": kv_count,
        }
        for _ in range(kv_count):
            try:
                key = _read_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                out[key] = _read_value(f, vtype)
            except GgufMetaError as e:
                out["_error"] = f"stopped at KV read: {e}"
                break
            except (struct.error, OSError) as e:
                out["_error"] = f"stopped at KV read: {e}"
                break
    return out


def _read_raw(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return _read_raw_stream(f)


def read_raw_bytes(buf: bytes) -> dict[str, Any]:
    """Parse GGUF metadata out of an in-memory buffer (e.g. a range-fetched header)."""
    import io
    return _read_raw_stream(io.BytesIO(buf))


def read_raw(path: Path) -> dict[str, Any]:
    st = path.stat()
    key = f"{path}|{st.st_mtime_ns}|{st.st_size}"
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached:
            return cached
    raw = _read_raw(path)
    with _CACHE_LOCK:
        _CACHE[key] = raw
    return raw


def _scalar_int(v: Any) -> int | None:
    """Unwrap array-summary dicts / lists to a representative int, or None if not resolvable.
    Some archs (Gemma, MoE variants) encode per-layer values as arrays; the config-panel and
    autoconfig want a scalar. Pick the most-common value from the sample."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, dict) and v.get("_array"):
        sample = v.get("sample") or []
        if sample:
            counts: dict[int, int] = {}
            for x in sample:
                if isinstance(x, (int, float)):
                    counts[int(x)] = counts.get(int(x), 0) + 1
            if counts:
                return max(counts, key=lambda k: counts[k])
    if isinstance(v, list) and v:
        first = v[0]
        if isinstance(first, (int, float)):
            return int(first)
    return None


def _scalar_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    n = _scalar_int(v)
    return float(n) if n is not None else None


def scan_chat_template_features(template: str) -> dict:
    """Detect reasoning/thinking-related capabilities of a Jinja chat template.
    Everything is checked against the raw template source — no execution."""
    if not template or not isinstance(template, str):
        return {}
    t = template.lower()
    return {
        # kwargs the template accepts (found as variable references)
        "accepts_enable_thinking": "enable_thinking" in t,
        "accepts_reasoning_effort": "reasoning_effort" in t,
        "accepts_preserve_thinking": "preserve_thinking" in t,
        # output markers — knowing these tells us reasoning-format should be 'deepseek' so
        # OpenAI-compatible clients (OpenWebUI etc.) can hide the thinking trace nicely
        "uses_think_tags": "<think>" in t,
        "uses_channel_thought": "channel>thought" in t or "channel|>thought" in t or "<|channel|>thought" in t,
    }


def _fmt_params(n: Any) -> str | None:
    if not isinstance(n, (int, float)) or n <= 0:
        return None
    n = float(n)
    if n >= 1e12:
        return f"{n / 1e12:.1f} T"
    if n >= 1e9:
        return f"{n / 1e9:.1f} B"
    if n >= 1e6:
        return f"{n / 1e6:.1f} M"
    if n >= 1e3:
        return f"{n / 1e3:.1f} K"
    return str(int(n))


def summarize(raw: dict[str, Any]) -> dict[str, Any]:
    arch = raw.get("general.architecture", "") or ""

    def a(key: str, default: Any = None) -> Any:
        return raw.get(f"{arch}.{key}", default) if arch else default

    file_type = raw.get("general.file_type")
    quant_name = FILE_TYPE_NAMES.get(file_type, str(file_type) if file_type is not None else None)

    vocab = raw.get("tokenizer.ggml.tokens")
    if isinstance(vocab, dict) and vocab.get("_array"):
        vocab_size = vocab.get("count")
    else:
        vocab_size = a("vocab_size")

    return {
        "arch": arch,
        "general": {
            "name": raw.get("general.name"),
            "description": raw.get("general.description"),
            "author": raw.get("general.author"),
            "license": raw.get("general.license"),
            "url": raw.get("general.url"),
            "quant": quant_name,
            "quant_version": raw.get("general.quantization_version"),
            "params": _fmt_params(raw.get("general.parameter_count")),
            "params_raw": raw.get("general.parameter_count"),
            "gguf_version": raw.get("_gguf_version"),
            "tensor_count": raw.get("_tensor_count"),
            "kv_count": raw.get("_kv_count"),
        },
        "model": {
            "context_length": _scalar_int(a("context_length")),
            "embedding_length": _scalar_int(a("embedding_length")),
            "block_count": _scalar_int(a("block_count")),
            "feed_forward_length": _scalar_int(a("feed_forward_length")),
            "attention_head_count": _scalar_int(a("attention.head_count")),
            # kept raw — autoconfig has _kv_first_int() that samples the array intelligently
            "attention_head_count_kv": a("attention.head_count_kv"),
            "rope_freq_base": _scalar_float(a("rope.freq_base")),
            "rope_scaling_type": a("rope.scaling.type"),
            "rope_scaling_factor": _scalar_float(a("rope.scaling.factor")),
            "rope_scaling_original_context": _scalar_int(a("rope.scaling.original_context_length")),
            "vocab_size": _scalar_int(vocab_size),
            "expert_count": _scalar_int(a("expert_count")),
            "expert_used_count": _scalar_int(a("expert_used_count")),
            "key_length": _scalar_int(a("attention.key_length")),
            "value_length": _scalar_int(a("attention.value_length")),
            "full_attention_interval": _scalar_int(a("full_attention_interval")),
            # Sliding-window attention, declared properly rather than guessed. Gemma-4 sets
            # all four: most layers attend over a short window and use a NARROWER head dim
            # than the global layers, and some layers share another layer's KV entirely.
            # Sizing without these over-estimates the cache by an order of magnitude.
            "sliding_window": _scalar_int(a("attention.sliding_window")),
            "key_length_swa": _scalar_int(a("attention.key_length_swa")),
            "value_length_swa": _scalar_int(a("attention.value_length_swa")),
            "shared_kv_layers": _scalar_int(a("attention.shared_kv_layers")),
            # kept raw: a per-layer array of which layers are local vs global
            "sliding_window_pattern": a("attention.sliding_window_pattern"),
            "ssm_state_size": _scalar_int(a("ssm.state_size")),
            "ssm_inner_size": _scalar_int(a("ssm.inner_size")),
        },
        "tokenizer": {
            "model": raw.get("tokenizer.ggml.model"),
            "pre": raw.get("tokenizer.ggml.pre"),
            "bos_token_id": raw.get("tokenizer.ggml.bos_token_id"),
            "eos_token_id": raw.get("tokenizer.ggml.eos_token_id"),
            "unknown_token_id": raw.get("tokenizer.ggml.unknown_token_id"),
            "padding_token_id": raw.get("tokenizer.ggml.padding_token_id"),
            "add_bos_token": raw.get("tokenizer.ggml.add_bos_token"),
            "add_eos_token": raw.get("tokenizer.ggml.add_eos_token"),
        },
        "chat_template": raw.get("tokenizer.chat_template"),
        "chat_template_features": scan_chat_template_features(raw.get("tokenizer.chat_template") or ""),
    }

