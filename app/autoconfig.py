"""System-aware ini recommendation engine.

Given a GGUF summary + on-disk file size + backend inventory (VRAM, vendor,
baseline flags parsed from the container's CLI), produce a Recommendation
that says: which backend, at what ctx, with which values — and why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import ini

# candidate contexts to try, smallest → largest
_CTX_CANDIDATES = (
    4096, 8192, 12288, 16384, 24576, 32768, 40960, 49152, 57344, 65536,
    81920, 98304, 114688, 131072, 147456, 163840, 196608, 229376, 262144,
    393216, 524288, 786432, 1048576,
)

_CACHE_BYTES_PER_ELEM = {
    "": 2.0, "f16": 2.0, "bf16": 2.0, "f32": 4.0,
    "q8_0": 1.0625, "q5_0": 0.75, "q5_1": 0.8125, "q4_0": 0.6250, "q4_1": 0.6875,
    "iq4_nl": 0.6250,
}

_RESERVE_GB = 0.5   # headroom on each GPU for other allocations
_MODEL_OVERHEAD = 1.00  # file_size × this = model VRAM footprint (Q_K_M loads at ~file size on modern llama.cpp)
_CACHE_DEFAULT = "q8_0"  # symmetric K/V; K stays q8, V could drop to q4 for +20% ctx (future preset)
_SSM_STATE_BYTES = 4 * 1024 * 1024  # ~4 MB per SSM layer, derived from typical state_size×inner_size


def _cache_dtype_bytes(k: str) -> float:
    return _CACHE_BYTES_PER_ELEM.get((k or "").lower(), 2.0)


def _kv_first_int(kv_heads: Any, default: int = 8) -> int:
    """kv_heads may be an int, or a per-layer array dict from GGUF; extract a representative int."""
    if isinstance(kv_heads, int):
        return kv_heads
    if isinstance(kv_heads, dict) and kv_heads.get("_array"):
        sample = kv_heads.get("sample") or []
        # pick the most common value in the sample; for gemma it's usually 8 with a few 1s
        if sample:
            counts: dict[int, int] = {}
            for v in sample:
                counts[int(v)] = counts.get(int(v), 0) + 1
            return max(counts, key=lambda k: counts[k])
    if isinstance(kv_heads, list) and kv_heads:
        return int(kv_heads[0])
    return default


def kv_cache_bytes(arch: str, ctx: int, layers: int, kv_heads: int,
                   head_dim: int, bytes_per_elem: float,
                   key_length: int | None = None, value_length: int | None = None,
                   full_attention_interval: int | None = None,
                   ssm_state_size: int | None = None) -> int:
    """Compute KV cache bytes. Handles:
       - explicit key/value_length overrides (Qwen3.x, Yi, etc.)
       - hybrid attention+SSM (Qwen3.5, Zamba) via full_attention_interval + ssm_state_size
       - Gemma sliding-window carve-out
    """
    if not (ctx > 0 and layers > 0 and kv_heads > 0):
        return 0
    # Use explicit key/value lengths when the model declares them; else fall back to head_dim
    k_dim = int(key_length) if key_length else head_dim
    v_dim = int(value_length) if value_length else head_dim
    if k_dim <= 0 or v_dim <= 0:
        return 0
    per_layer_per_token = kv_heads * (k_dim + v_dim) * bytes_per_elem
    arch_l = (arch or "").lower()

    # Hybrid attention + SSM (Mamba-2 style) — only every Nth layer holds a real KV cache
    is_hybrid = bool(ssm_state_size) or (full_attention_interval and full_attention_interval > 1)
    if is_hybrid and full_attention_interval and full_attention_interval > 1:
        full_layers = max(1, (layers + full_attention_interval - 1) // full_attention_interval)
        ssm_layers = layers - full_layers
        kv_full = full_layers * per_layer_per_token * ctx
        return int(kv_full + ssm_layers * _SSM_STATE_BYTES)

    if arch_l.startswith("gemma") and layers >= 6:
        # gemma-3/4 pattern: ~1-in-6 layers use full attention, rest are SWA with a ~4096-token window
        full_layers = max(1, layers // 6)
        swa_layers = layers - full_layers
        swa_window = 4096
        kv_full = full_layers * per_layer_per_token * ctx
        kv_swa = swa_layers * per_layer_per_token * swa_window
        return int(kv_full + kv_swa)

    return int(layers * per_layer_per_token * ctx)


# ---- baseline parsing (compose command → ini keys) ----

_SHORT_TO_KEY = {
    "-ngl": "ngl", "-fa": "flash-attn", "-ctk": "cache-type-k",
    "-ctv": "cache-type-v", "-np": "parallel", "-c": "ctx-size",
    "-b": "batch-size", "-ub": "ubatch-size", "-t": "threads",
    "-tb": "threads-batch", "-sm": "split-mode", "-mg": "main-gpu",
    "-fit": "fit", "-fitt": "fit-target", "-fitc": "fit-ctx",
    "-cmoe": "cpu-moe", "-ncmoe": "n-cpu-moe",
    "-kvo": "kv-offload",
}


def parse_baseline(cmd_args: list[str]) -> dict[str, str]:
    """Parse the container's llama-server command args into a {ini_key: value} dict."""
    out: dict[str, str] = {}
    n = len(cmd_args)
    for i, a in enumerate(cmd_args):
        key: str | None = None
        if a in _SHORT_TO_KEY:
            key = _SHORT_TO_KEY[a]
        elif a.startswith("--"):
            k = a[2:]
            if k in ini.ALL_KNOWN_KEYS:
                key = k
        if not key:
            continue
        nxt = cmd_args[i + 1] if i + 1 < n else None
        if nxt is not None and not nxt.startswith("-"):
            out[key] = nxt
        else:
            out[key] = "true"
    return out


# ---- recommendation ----

@dataclass
class FitRow:
    ctx: int
    model_gb: float          # GPU-resident weight VRAM (accounts for MoE offload)
    kv_gb: float
    total_gb: float          # model_gb + kv_gb
    fits: bool
    free_gb: float
    offload_kind: str = ""   # "" | "cpu-moe" | "n-cpu-moe"
    n_cpu_moe: int = 0       # populated when offload_kind == "n-cpu-moe"


@dataclass
class BackendPlan:
    name: str
    vendor: str
    vram_gb: float
    rows: list[FitRow]                       # per-ctx breakdown
    max_ctx: int                             # largest ctx that fits with reserve
    fits_at_all: bool

    @property
    def native_fits(self) -> bool:
        return any(r.fits and r.ctx == max(x.ctx for x in self.rows) for r in self.rows) if self.rows else False


@dataclass
class PresetOption:
    key: str                # "fast" | "balanced" | "long-ctx"
    label: str
    icon: str               # lucide name
    backend: str
    ctx: int
    n_cpu_moe: int          # 0 = no offload; layers = all offloaded (cpu-moe=true)
    offload_kind: str       # "" | "cpu-moe" | "n-cpu-moe"
    gpu_layers: int         # layers whose experts stay on GPU
    total_layers: int
    gpu_gb: float           # weights VRAM
    kv_gb: float
    speed_score: float      # 0..1 relative (1.0 = no offload)


@dataclass
class Recommendation:
    plans: list[BackendPlan]
    recommended_backend: str
    recommended_ctx: int
    values: dict[str, str]                   # full-form values
    values_minimal: dict[str, str]           # only non-baseline-redundant values
    baseline_redundant: dict[str, str]       # values already covered by compose baseline
    quirks: list[str]
    unavailable: list[str]                   # which knobs don't apply to this model
    current_diff: list[str]                  # human-readable diff vs existing section (empty if new)
    presets: list[PresetOption] = field(default_factory=list)  # Fast/Balanced/Long-ctx for MoE with offload
    active_preset: str = ""                                     # which preset the current values reflect
    error: str = ""


def _fmt_ctx(n: int) -> str:
    if n >= 1024 and n % 1024 == 0:
        k = n // 1024
        return f"{k}K"
    return f"{n:,}"


def _moe_ratio(expert_count: Any) -> float:
    """Approximate share of a GGUF's weights that live in MoE experts.
    Heuristic: more experts → more of the weights are experts.
      - 8 experts → ~0.75 in experts
      - 16 experts → ~0.85
      - 32-128 experts → ~0.9-0.92
    Clamped to [0.6, 0.92]. Used for sizing cpu-moe / n-cpu-moe recommendations."""
    if not isinstance(expert_count, int) or expert_count < 2:
        return 0.0
    if expert_count >= 64:
        return 0.92
    if expert_count >= 32:
        return 0.90
    if expert_count >= 16:
        return 0.85
    if expert_count >= 8:
        return 0.78
    return 0.65


def _find_fit(model_gb_full: float, kv_gb: float, budget_gb: float,
              layers: int, moe_ratio: float) -> tuple[bool, float, str, int]:
    """Return (fits, gpu_model_gb, offload_kind, n_cpu_moe) for a given (model, kv, budget).

    Strategy: try no offload first; then, if MoE, find the smallest n-cpu-moe
    (fewest layers offloaded) that fits, treating cpu-moe=true as n=layers.
    """
    if model_gb_full + kv_gb <= budget_gb:
        return True, model_gb_full, "", 0
    if moe_ratio <= 0 or layers <= 0:
        return False, model_gb_full, "", 0
    attention_gb = model_gb_full * (1 - moe_ratio)
    expert_per_layer_gb = model_gb_full * moe_ratio / layers
    # smallest n where fits — n counts CPU-offloaded layers
    for n in range(1, layers + 1):
        gpu = attention_gb + max(0, layers - n) * expert_per_layer_gb
        if gpu + kv_gb <= budget_gb:
            if n >= layers:
                return True, attention_gb, "cpu-moe", 0
            return True, gpu, "n-cpu-moe", n
    return False, attention_gb, "cpu-moe", 0  # even full offload can't fit (attention too big for GPU)


def _pareto_frontier(arch: str, layers: int, kv_heads: int, head_dim: int,
                     bytes_per: float, model_gb: float, moe_ratio: float,
                     budget_gb: float, ctx_candidates: list[int],
                     key_length: int | None = None, value_length: int | None = None,
                     full_attention_interval: int | None = None,
                     ssm_state_size: int | None = None) -> list[tuple[int, int, float, float]]:
    """For a MoE model on a given backend, sweep n-cpu-moe from 0..layers.
    Return list of (n_cpu_moe, max_ctx_fitting, gpu_weight_gb, kv_gb_at_max_ctx) — pareto frontier.
    Higher n_cpu_moe → higher max_ctx (more offloaded = less VRAM for weights = more room for KV).
    """
    if moe_ratio <= 0 or layers <= 0:
        return []
    attention_gb = model_gb * (1 - moe_ratio)
    per_layer_gb = model_gb * moe_ratio / layers
    frontier: list[tuple[int, int, float, float]] = []
    for ncm in range(0, layers + 1):
        gpu_layers = layers - ncm
        weight_gb = attention_gb + gpu_layers * per_layer_gb
        if weight_gb >= budget_gb:
            continue  # can't fit even at ctx=0
        # find max ctx that fits with this weight footprint
        best_ctx = 0
        best_kv = 0.0
        for ctx in ctx_candidates:
            kv_gb = kv_cache_bytes(arch, ctx, layers, kv_heads, head_dim, bytes_per,
                                   key_length=key_length, value_length=value_length,
                                   full_attention_interval=full_attention_interval,
                                   ssm_state_size=ssm_state_size) / (1024 ** 3)
            if weight_gb + kv_gb <= budget_gb:
                if ctx > best_ctx:
                    best_ctx = ctx
                    best_kv = kv_gb
        if best_ctx == 0:
            continue
        # only keep this point if it strictly improves ctx over prior (higher ncm) — pareto step
        if frontier and best_ctx <= frontier[-1][1]:
            continue
        frontier.append((ncm, best_ctx, weight_gb, best_kv))
    return frontier


def _presets_from_frontier(frontier: list[tuple[int, int, float, float]], backend: str,
                           layers: int, native_ctx: int) -> list[PresetOption]:
    """Pick Fast / Balanced / Long-ctx from the frontier."""
    if not frontier:
        return []
    # normalize
    max_ctx = max(f[1] for f in frontier)
    min_ncm = min(f[0] for f in frontier)
    max_ncm = max(f[0] for f in frontier)
    fast_min_ctx = 8192

    def _speed_score(ncm: int) -> float:
        return (layers - ncm) / layers if layers > 0 else 0.0

    def _make(key: str, label: str, icon: str, entry: tuple[int, int, float, float]) -> PresetOption:
        ncm, ctx, gpu_gb, kv_gb = entry
        return PresetOption(
            key=key, label=label, icon=icon, backend=backend,
            ctx=ctx, n_cpu_moe=ncm,
            offload_kind=("cpu-moe" if ncm >= layers else ("n-cpu-moe" if ncm > 0 else "")),
            gpu_layers=layers - ncm, total_layers=layers,
            gpu_gb=round(gpu_gb, 2), kv_gb=round(kv_gb, 2),
            speed_score=round(_speed_score(ncm), 3),
        )

    # Fast: highest speed (lowest ncm) where ctx meets a chat minimum
    fast_candidates = [f for f in frontier if f[1] >= fast_min_ctx] or frontier
    fast_entry = min(fast_candidates, key=lambda f: f[0])

    # Long-ctx: max ctx (any ncm)
    long_entry = max(frontier, key=lambda f: (f[1], -f[0]))

    # Balanced: midpoint on the frontier by frontier-index between fast and long
    fi = frontier.index(fast_entry)
    li = frontier.index(long_entry)
    if fi > li: fi, li = li, fi
    mi = (fi + li) // 2
    balanced_entry = frontier[mi]

    out: list[PresetOption] = []
    seen: set[tuple[int, int]] = set()
    for key, label, icon, entry in (
        ("fast", "Fast", "gauge", fast_entry),
        ("balanced", "Balanced", "cpu", balanced_entry),
        ("long-ctx", "Long context", "layers-3", long_entry),
    ):
        sig = (entry[0], entry[1])
        if sig in seen:
            continue
        seen.add(sig)
        out.append(_make(key, label, icon, entry))
    return out


def _find_mmproj(models_dir: "Path | None", section_name: str, subdir: str = "") -> str:
    """Look for a matching mmproj file in the models dir.
    Returns a path RELATIVE to /models (e.g. 'mmproj-model-f16.gguf' or 'granite/mmproj.gguf').
    Match priority:
      1. If model is in a subdir, prefer mmproj in the same subdir
      2. Filename containing the section stem
      3. Only mmproj file in /models (unique fallback)
    """
    if models_dir is None:
        return ""
    try:
        # collect candidates from top-level and one-level subdirs
        candidates: list[tuple[str, str]] = []  # (subdir, filename)
        for p in models_dir.iterdir():
            if p.is_file() and p.suffix.lower() == ".gguf" and "mmproj" in p.name.lower():
                candidates.append(("", p.name))
            elif p.is_dir() and not p.name.startswith("."):
                try:
                    for sp in p.iterdir():
                        if sp.is_file() and sp.suffix.lower() == ".gguf" and "mmproj" in sp.name.lower():
                            candidates.append((p.name, sp.name))
                except OSError:
                    continue
    except OSError:
        return ""
    if not candidates:
        return ""
    stem = section_name.lower()
    stem_key = stem.replace("-", "").replace("_", "").replace(".", "")

    # 1) same subdir as the model (for sharded/co-located layouts)
    if subdir:
        same_subdir = [c for c in candidates if c[0] == subdir]
        if same_subdir:
            c = same_subdir[0]
            return f"/models/{c[0]}/{c[1]}" if c[0] else f"/models/{c[1]}"

    # 2) subdir NAME shares a substantive chunk with the model stem
    # (mmproj is routed to /models/<owner_slug>/, which typically contains the model name)
    def _shares(a: str, b: str, minlen: int = 8) -> bool:
        a2 = a.lower().replace("-", "").replace("_", "").replace(".", "").replace("/", "")
        b2 = b.lower().replace("-", "").replace("_", "").replace(".", "").replace("/", "")
        if not a2 or not b2:
            return False
        return a2[:minlen] in b2 or b2[:minlen] in a2
    for sub, name in candidates:
        if sub and _shares(sub, stem):
            return f"/models/{sub}/{name}"

    # 3) filename shares a substantive chunk with the model stem
    for sub, name in candidates:
        if _shares(name, stem):
            return f"/models/{sub}/{name}" if sub else f"/models/{name}"

    # 4) if there's exactly one mmproj anywhere, use it as a unique fallback
    if len(candidates) == 1:
        sub, name = candidates[0]
        return f"/models/{sub}/{name}" if sub else f"/models/{name}"
    return ""


def analyze(*,
            summary: dict,
            file_size: int,
            backends: list[dict],           # [{name, vendor, vram_gb, baseline: {ini_key: val}}]
            model_rel: str = "",
            current_section: dict[str, str] | None = None,
            preset: str = "balanced",
            models_dir: "Path | None" = None,
            section_name: str = "",
            model_subdir: str = "") -> Recommendation:
    arch = (summary.get("arch") or "").lower()
    m = summary.get("model") or {}
    layers = int(m.get("block_count") or 0)
    heads = int(m.get("attention_head_count") or 1)
    embed = int(m.get("embedding_length") or 0)
    head_dim = embed // heads if heads > 0 else 0
    kv_heads = _kv_first_int(m.get("attention_head_count_kv"), default=heads)
    native_ctx = int(m.get("context_length") or 0)
    experts = m.get("expert_count")
    # explicit K/V lengths + hybrid markers (Qwen 3.5, Zamba, etc.)
    key_length = int(m.get("key_length")) if isinstance(m.get("key_length"), int) else None
    value_length = int(m.get("value_length")) if isinstance(m.get("value_length"), int) else None
    full_attention_interval = int(m.get("full_attention_interval")) if isinstance(m.get("full_attention_interval"), int) else None
    ssm_state_size = int(m.get("ssm_state_size")) if isinstance(m.get("ssm_state_size"), int) else None

    model_gb = file_size / (1024 ** 3) * _MODEL_OVERHEAD
    moe_ratio = _moe_ratio(experts)

    # Plan around q8_0 KV cache: near-identical quality to f16 in practice,
    # and lets us fit ~2× the context. Users can override in the form if they want f16.
    bytes_per = _cache_dtype_bytes(_CACHE_DEFAULT)

    # Candidate ctx values include native and can EXTEND beyond it (via RoPE scaling downstream)
    cands = sorted(set(list(_CTX_CANDIDATES) + ([native_ctx] if native_ctx else [])))
    # cap at 8× native to keep RoPE degradation bounded (still lets us extend a lot)
    if native_ctx:
        cands = [c for c in cands if c <= native_ctx * 8]

    plans: list[BackendPlan] = []
    # remember offload used at the recommended-ctx per backend, for the values dict
    per_backend_offload: dict[str, tuple[str, int]] = {}
    for b in backends:
        rows: list[FitRow] = []
        budget = float(b["vram_gb"]) - _RESERVE_GB
        max_fit = 0
        max_fit_offload: tuple[str, int] = ("", 0)
        for ctx in cands:
            kv_gb = kv_cache_bytes(arch, ctx, layers, kv_heads, head_dim, bytes_per,
                                   key_length=key_length, value_length=value_length,
                                   full_attention_interval=full_attention_interval,
                                   ssm_state_size=ssm_state_size) / (1024 ** 3)
            fits, gpu_model_gb, offload_kind, n_cm = _find_fit(model_gb, kv_gb, budget, layers, moe_ratio)
            total = gpu_model_gb + kv_gb
            rows.append(FitRow(
                ctx=ctx, model_gb=round(gpu_model_gb, 2), kv_gb=round(kv_gb, 2),
                total_gb=round(total, 2), fits=fits,
                free_gb=round(float(b["vram_gb"]) - total, 2),
                offload_kind=offload_kind, n_cpu_moe=n_cm,
            ))
            if fits:
                max_fit = ctx
                max_fit_offload = (offload_kind, n_cm)
        plans.append(BackendPlan(
            name=b["name"], vendor=b.get("vendor", ""),
            vram_gb=float(b["vram_gb"]), rows=rows,
            max_ctx=max_fit, fits_at_all=(max_fit > 0),
        ))
        if max_fit:
            per_backend_offload[b["name"]] = max_fit_offload

    # pick recommendation:
    #   Rule of thumb: pick the largest ctx we can, on the smallest GPU that hosts it,
    #   BUT if MoE requires offload at that ctx, prefer the bigger GPU (less offload = faster).
    recommended: BackendPlan | None = None
    rec_ctx = 0
    is_moe_now = isinstance(experts, int) and experts > 1
    fitting = [p for p in plans if p.fits_at_all]
    if fitting:
        # Best ctx anyone can achieve
        global_max_ctx = max(p.max_ctx for p in fitting)
        top_plans = [p for p in fitting if p.max_ctx >= global_max_ctx]
        # among those hitting the global max ctx, prefer smaller GPU (leaves the big card free)
        # unless MoE + offload needed there → prefer bigger GPU (less offload)
        def _needs_offload_at(p: BackendPlan, ctx: int) -> bool:
            r = next((row for row in p.rows if row.ctx == ctx), None)
            return bool(r and r.offload_kind)
        if is_moe_now and any(_needs_offload_at(p, global_max_ctx) for p in top_plans):
            recommended = sorted(top_plans, key=lambda p: p.vram_gb, reverse=True)[0]
        else:
            recommended = sorted(top_plans, key=lambda p: p.vram_gb)[0]
        rec_ctx = recommended.max_ctx

    # build values
    values: dict[str, str] = {}
    if model_rel:
        values["model"] = model_rel
    if rec_ctx > 0:
        values["ctx-size"] = str(rec_ctx)
    values["ngl"] = "999"
    values["flash-attn"] = "on"
    values["cache-type-k"] = _CACHE_DEFAULT
    values["cache-type-v"] = _CACHE_DEFAULT
    if summary.get("chat_template"):
        values["jinja"] = "true"

    # Multimodal: look for an adjacent mmproj file the user hasn't already set
    if section_name:
        current_mmproj = (current_section or {}).get("mmproj", "").strip()
        if current_mmproj:
            # Respect user's existing choice — echo it so Fill preserves it
            values["mmproj"] = current_mmproj
        else:
            found_mmproj = _find_mmproj(models_dir, section_name, model_subdir)
            if found_mmproj:
                values["mmproj"] = found_mmproj

    rope_type = m.get("rope_scaling_type")

    # MoE offload for the recommended (backend, ctx)
    active_preset = ""
    presets: list[PresetOption] = []
    if recommended and rec_ctx > 0:
        off_kind, n_cm = per_backend_offload.get(recommended.name, ("", 0))
        # If MoE + offload needed, compute the preset frontier on the recommended backend
        # and let the requested preset override ctx / ncm.
        is_moe_now2 = isinstance(experts, int) and experts > 1
        if is_moe_now2 and off_kind:
            budget = float(recommended.vram_gb) - _RESERVE_GB
            frontier = _pareto_frontier(arch, layers, kv_heads, head_dim, bytes_per,
                                        model_gb, moe_ratio, budget, cands,
                                        key_length=key_length, value_length=value_length,
                                        full_attention_interval=full_attention_interval,
                                        ssm_state_size=ssm_state_size)
            presets = _presets_from_frontier(frontier, recommended.name, layers, native_ctx)
            if presets:
                chosen = next((p for p in presets if p.key == preset), presets[len(presets) // 2])
                active_preset = chosen.key
                rec_ctx = chosen.ctx
                off_kind = chosen.offload_kind
                n_cm = chosen.n_cpu_moe
                values["ctx-size"] = str(rec_ctx)
        if off_kind == "cpu-moe":
            values["cpu-moe"] = "true"
        elif off_kind == "n-cpu-moe" and n_cm > 0:
            values["n-cpu-moe"] = str(n_cm)

    # RoPE handling — respect the model's own scaling if declared, otherwise auto-linear
    # for any ctx we chose that exceeds the model's native ctx.
    if rope_type and str(rope_type).lower() not in ("none", ""):
        values["rope-scaling"] = str(rope_type)
        rope_factor = m.get("rope_scaling_factor")
        if isinstance(rope_factor, (int, float)) and rope_factor > 0:
            values["rope-scale"] = str(rope_factor)
    elif rec_ctx > native_ctx and native_ctx > 0:
        values["rope-scaling"] = "linear"
        values["rope-scale"] = f"{round(rec_ctx / native_ctx, 1)}"

    # baseline-redundant: any key that matches the recommended backend's baseline
    baseline_redundant: dict[str, str] = {}
    minimal = dict(values)
    if recommended:
        rec_backend = next((b for b in backends if b["name"] == recommended.name), None)
        base = (rec_backend or {}).get("baseline") or {}
        for k, v in list(values.items()):
            bv = base.get(k)
            if bv is not None and str(bv).lower() == str(v).lower():
                baseline_redundant[k] = v
                minimal.pop(k, None)

    # ensure minimal keeps the essential differentiators even if redundant on paper
    for essential in ("model", "ctx-size", "jinja", "cpu-moe", "n-cpu-moe"):
        if essential in values and essential not in minimal:
            minimal[essential] = values[essential]

    # quirks
    quirks: list[str] = []
    if arch.startswith("gemma"):
        quirks.append("Gemma sliding-window attention: the ctx→KV math above assumes swa-full=false (default). "
                      "Enabling swa-full multiplies full-attn KV ~5× and will OOM.")
    # (MoE hint is emitted later, tailored to whichever offload the recommendation actually applies)
    if not summary.get("chat_template"):
        quirks.append("No embedded chat template — you'll need to set `chat-template` or `chat-template-file` "
                      "manually to get correct multi-turn formatting.")

    # RoPE extension quirk (only when we set linear scaling ourselves)
    if values.get("rope-scaling") == "linear" and (not rope_type or str(rope_type).lower() in ("none", "")) and native_ctx > 0:
        scale = values.get("rope-scale", "?")
        quirks.append(f"Extended ctx from native {_fmt_ctx(native_ctx)} to {_fmt_ctx(rec_ctx)} "
                      f"via `rope-scaling=linear, rope-scale={scale}`. Linear scaling degrades quality gracefully up "
                      f"to ~2× native; beyond that outputs get progressively worse. Drop ctx-size in the form to back off.")

    # unavailable knobs
    unavailable: list[str] = []
    is_moe = isinstance(experts, int) and experts > 1
    if not is_moe:
        unavailable.append("cpu-moe / n-cpu-moe (not MoE)")
    if not rope_type or str(rope_type).lower() == "none":
        unavailable.append("rope-scaling (model doesn't declare one)")

    # MoE-specific quirk: reflect what offload is being applied
    if is_moe:
        if recommended:
            off_kind, n_cm = per_backend_offload.get(recommended.name, ("", 0))
            if off_kind == "cpu-moe":
                quirks.append(f"MoE model ({experts} experts): all expert weights offloaded to CPU via cpu-moe=true. "
                              "Attention stays on GPU. Expect slower generation than a fully-GPU model.")
            elif off_kind == "n-cpu-moe":
                quirks.append(f"MoE model ({experts} experts): first {n_cm} of {layers} layers' experts offloaded to CPU. "
                              "Remaining layers keep experts on GPU for speed. Tune n-cpu-moe up/down to trade VRAM for tok/s.")
            else:
                quirks.append(f"MoE model ({experts} experts): fits fully on GPU at this ctx — no CPU offload needed.")
        else:
            quirks.append(f"MoE model ({experts} experts): does not fit even with all experts offloaded to CPU. "
                          f"You need a bigger GPU, a smaller quant, or a shorter context.")

    # diff vs current section — only report on keys autoconfig actually opinions on.
    # Anything the user set that we don't touch (mmproj, chat-template-file, lora, override-*, etc.)
    # is left alone: not reported as a diff, and Fill/Fill minimal doesn't overwrite it.
    current_diff: list[str] = []
    # Keys we may explicitly displace (e.g. cpu-moe when we set n-cpu-moe instead)
    _displaces = {"cpu-moe", "n-cpu-moe"} if is_moe else set()
    if current_section:
        cur = {k: str(v) for k, v in current_section.items()}
        for k, v in values.items():
            if cur.get(k, "") != v:
                if k in cur:
                    current_diff.append(f"{k}: {cur[k]!r} → {v!r}")
                else:
                    current_diff.append(f"{k}: unset → {v!r}")
        # Only report removals for keys we actively displace
        for k in _displaces:
            if k in cur and cur[k] and k not in values:
                current_diff.append(f"{k}: {cur[k]!r} → unset (superseded)")

    return Recommendation(
        plans=plans,
        recommended_backend=(recommended.name if recommended else ""),
        recommended_ctx=rec_ctx,
        values=values,
        values_minimal=minimal,
        baseline_redundant=baseline_redundant,
        quirks=quirks,
        unavailable=unavailable,
        current_diff=current_diff,
        presets=presets,
        active_preset=active_preset,
    )


def format_ctx(n: int) -> str:
    return _fmt_ctx(n)
