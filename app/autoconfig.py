"""System-aware ini recommendation engine.

Given a GGUF summary + on-disk file size + backend inventory (VRAM, vendor,
baseline flags parsed from the container's CLI), produce a Recommendation
that says: which backend, at what ctx, with which values — and why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import ini

# candidate contexts to try, smallest → largest
_CTX_CANDIDATES = (
    4096, 8192, 12288, 16384, 24576, 32768, 40960, 49152, 57344, 65536,
    73728, 81920, 90112, 98304, 106496, 114688, 122880, 131072, 139264,
    147456, 151552, 155648, 159744, 163840, 172032, 180224, 188416,
    196608, 204800, 212992, 221184, 229376, 237568, 245760, 253952, 262144,
    294912, 327680, 360448, 393216, 425984, 458752, 491520, 524288,
    589824, 655360, 720896, 786432, 851968, 917504, 983040, 1048576,
)
# Every value above is a multiple of 4096 (most are multiples of 8192), which keeps them on
# llama.cpp's internal block-alignment boundaries — the reason arbitrary values like 160000
# behave badly. The upper range used to step by 16384, which was too coarse: a model that
# already fits near its ceiling had only one or two reachable steps left, so the offload
# frontier collapsed to two points and "Balanced" came out identical to "Fast".

_CACHE_BYTES_PER_ELEM = {
    "": 2.0, "f16": 2.0, "bf16": 2.0, "f32": 4.0,
    "q8_0": 1.0625, "q5_0": 0.75, "q5_1": 0.8125, "q4_0": 0.6250, "q4_1": 0.6875,
    "iq4_nl": 0.6250,
}

_RESERVE_PER_GPU = 1.0   # CUDA runtime + driver context + scratch/cuBLAS workspace.
                         # Bumped from 0.5 -> 1.0 for llama.cpp 0.3.0-dev (commit d222767+) which
                         # recognizes MTP nextn tensors and allocates larger cuBLAS workspaces on
                         # inference. Empirical: Qwen3.8-27B at ctx=159744 loaded fine but OOM'd
                         # on first inference (cuBLAS workspace on device 1). At ctx=131072 fits
                         # cleanly. If you pin an older image and want more ctx, drop this back to 0.5.
_MODEL_OVERHEAD_SINGLE = 1.00  # Q_K_M loads at ~file size when everything's on one card
_MODEL_OVERHEAD_SPLIT = 1.08   # +8% for cross-GPU handoffs, duplicated activation buffers, layer-imbalance.
_CACHE_DEFAULT = "q8_0"  # symmetric K/V; K stays q8, V could drop to q4 for +20% ctx (future preset)
_SSM_STATE_BYTES = 4 * 1024 * 1024  # ~4 MB per SSM layer, derived from typical state_size×inner_size
_CPU_LAYER_PENALTY = 20.0  # how much slower one CPU-resident DENSE layer is than a GPU one.
                           # Used only to rank presets, not to decide fit. Dense offload is
                           # brutal compared to MoE expert offload: every token traverses every
                           # CPU layer, whereas MoE only touches a few active experts.
                           # VERIFIED on Qwen3.8-27B-OBLITERATED (65 layers, 2x RTX 5070):
                           #   ngl=999 (all GPU)  -> 34.2 tok/s   (predicted 100%)
                           #   ngl=56  (9 on CPU) ->  9.7 tok/s   (predicted 28%, measured 28.4%)
                           # Still hardware-dependent (RAM bandwidth, PCIe width), so treat the
                           # speed % as a well-calibrated estimate rather than a guarantee.
_MMPROJ_VRAM_MULT = 1.0  # projector weights land in VRAM at ~their file size.
                         # Verified on Qwen3-VL-4B: mmproj-F32.gguf is 1.55 GB on disk and
                         # llama-server allocates 1584.43 MiB for it. (An earlier 2.0 here was
                         # a mis-calibration: that same 1584 MiB was compared against the 0.78 GB
                         # BF16 projector in the directory rather than the F32 one actually
                         # referenced by the preset.)
_MMPROJ_COMPUTE_GB = 0.5  # modality-encoder scratch beyond the projector weights.
                          # This is deliberately small. An earlier 1.8 here was calibrated off a
                          # FAILED allocation line in a log (an attempt at a much larger ctx, on a
                          # different model) and then multiplied by gpu_count — which reserved
                          # 5.34 GB for a 0.87 GB projector and made a working model report
                          # "doesn't fit at any context". Ground truth from a loaded
                          # Qwen3.8-27B-OBLITERATED at ctx=130768: 22.63 GiB used total, of which
                          # model+KV+projector accounts for ~20.96 GiB — so ALL remaining overhead,
                          # both cards' CUDA contexts included, is ~1.67 GiB. _RESERVE_PER_GPU
                          # already covers most of that.


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
                   ssm_state_size: int | None = None,
                   v_bytes_per_elem: float | None = None) -> int:
    """Compute KV cache bytes. Handles:
       - explicit key/value_length overrides (Qwen3.x, Yi, etc.)
       - hybrid attention+SSM (Qwen3.5, Zamba) via full_attention_interval + ssm_state_size
       - Gemma sliding-window carve-out
       - asymmetric K/V quantization (bytes_per_elem is K; v_bytes_per_elem defaults to it)
    """
    if not (ctx > 0 and layers > 0 and kv_heads > 0):
        return 0
    # Use explicit key/value lengths when the model declares them; else fall back to head_dim
    k_dim = int(key_length) if key_length else head_dim
    v_dim = int(value_length) if value_length else head_dim
    if k_dim <= 0 or v_dim <= 0:
        return 0
    v_bytes = bytes_per_elem if v_bytes_per_elem is None else v_bytes_per_elem
    per_layer_per_token = kv_heads * (k_dim * bytes_per_elem + v_dim * v_bytes)
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
    ctx: int                 # per-session ctx (what each user sees)
    total_ctx: int           # ctx * n_sessions — what llama-server allocates as --ctx-size
    model_gb: float          # GPU-resident weight VRAM (accounts for MoE offload)
    kv_gb: float
    total_gb: float          # model_gb + kv_gb
    fits: bool
    free_gb: float
    offload_kind: str = ""   # "" | "cpu-moe" | "n-cpu-moe"
    n_cpu_moe: int = 0       # populated when offload_kind == "n-cpu-moe"
    gpu_pct: int = 100       # share of the model's WEIGHTS resident on the GPU, 0-100.
                             # Measured in bytes rather than layers because for a MoE the
                             # layer count says little: attention stays resident while only
                             # experts move, so "layers on GPU" overstates what is really there.


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
    offload_kind: str       # "" | "cpu-moe" | "n-cpu-moe" | "ngl"
    gpu_layers: int         # layers kept on GPU (for MoE: layers whose experts stay on GPU)
    total_layers: int
    gpu_gb: float           # weights VRAM
    kv_gb: float
    speed_score: float      # 0..1 relative (1.0 = no offload)
    ngl: int = -1           # dense offload only: value to write as `ngl`. -1 = leave at 999 (all)


@dataclass
class Recommendation:
    plans: list[BackendPlan]
    recommended_backend: str
    recommended_ctx: int                     # per-session ctx (each user's window)
    recommended_total_ctx: int = 0           # ctx * n_sessions (llama-server --ctx-size)
    n_sessions: int = 1                      # concurrent parallel slots
    values: dict[str, str] = field(default_factory=dict)              # full-form values
    values_minimal: dict[str, str] = field(default_factory=dict)      # only non-baseline-redundant values
    baseline_redundant: dict[str, str] = field(default_factory=dict)  # values already covered by compose baseline
    quirks: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)              # which knobs don't apply to this model
    current_diff: list[str] = field(default_factory=list)             # human-readable diff vs existing section (empty if new)
    presets: list[PresetOption] = field(default_factory=list)  # Fast/Balanced/Long-ctx samples
    # Every point on the offload frontier, ascending by ctx. The presets above are three
    # samples from this; the UI exposes the whole curve via a slider so any achievable
    # speed/context tradeoff can be picked directly.
    frontier: list[PresetOption] = field(default_factory=list)
    # True when the model fits entirely on the GPU *at its native context*. In that case
    # there is nothing to trade — offloading layers cannot buy context beyond native — so
    # the UI shows a single option instead of three chips that would all be identical.
    fits_full_gpu: bool = False
    native_ctx: int = 0
    # Which preset (if any) the CURRENTLY SAVED ini section corresponds to. Selecting a chip
    # only previews a recommendation — nothing is written until Fill form + Save — so the UI
    # needs to distinguish "previewing" from "actually running" or the two look identical.
    current_preset: str = ""
    has_unsaved: bool = False
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


def _layer_costs(layers: int, n_cpu_moe: int, attention_gb: float,
                 expert_per_layer_gb: float, kv_gb: float) -> list[float]:
    """Per-layer GPU cost: attention + KV share, plus experts above the n-cpu-moe threshold."""
    if layers <= 0:
        return []
    att = attention_gb / layers
    kv = kv_gb / layers
    return [att + kv + (expert_per_layer_gb if i >= n_cpu_moe else 0.0) for i in range(layers)]


def _split_feasible(layers: int, n_cpu_moe: int, attention_gb: float,
                    expert_per_layer_gb: float, kv_gb: float, gpu_count: int,
                    pinned_gb: float, caps: list[float]) -> bool:
    """Does ANY contiguous per-card partition fit? One greedy pass.

    The fit search asks only whether a configuration is placeable, never how to place it
    best. Answering that with _partition_min_max ran a 64-step binary search per call, and
    the search makes thousands of calls per page load — measured at 2239 for one model,
    around nine million inner steps, which turned a snappy preset click into a visible wait.

    Filling each card to capacity in order is optimal for CONTIGUOUS feasibility: taking
    less on an earlier card can only leave more for a later one, never less.
    """
    if gpu_count <= 1 or not caps:
        return True
    costs = _layer_costs(layers, n_cpu_moe, attention_gb, expert_per_layer_gb, kv_gb)
    if not costs:
        return True
    idx, n = 0, len(costs)
    for card in range(gpu_count):
        budget = caps[card] - (pinned_gb if card == 0 else 0.0)
        took = 0
        while idx < n and costs[idx] <= budget:
            budget -= costs[idx]
            idx += 1
            took += 1
        if idx >= n:
            return True
        if took == 0:
            return False          # this card cannot hold even one more layer
    return idx >= n


def _partition_min_max(costs: list[float], k: int, pinned_gb: float,
                       caps: list[float] | None = None) -> list[int]:
    """Split `costs` into k CONTIGUOUS runs, minimizing the heaviest run RELATIVE to its card.

    Device 0 additionally carries `pinned_gb`: the projector, compute buffer and cuBLAS
    workspace are not layer-split, they all land on the main GPU.

    `caps` are per-card capacities. They matter because the goal is not equal bytes, it is
    equal PRESSURE: on a 24 GB card beside a 12 GB card, an even split wastes half the big
    card and overfills the small one. Minimizing load/capacity balances both cases with one
    rule, and reduces to plain byte-balancing when the cards are identical. Falls back to
    equal weighting when capacities are unknown.

    Returns the layer count per device. Binary-searches the ratio and greedily fills, and
    guarantees every device gets at least one layer whenever there are enough layers to go
    round — a device left with zero would be idle hardware.
    """
    if k <= 1 or not costs:
        return [len(costs)]
    n = len(costs)
    caps = list(caps) if caps and len(caps) == k and all(c > 0 for c in caps) else [1.0] * k

    def feasible(ratio: float) -> list[int] | None:
        counts: list[int] = []
        run = 0.0
        used = 0
        for i, c in enumerate(costs):
            idx = len(counts)                       # device the current run belongs to
            budget = caps[idx] * ratio - (pinned_gb if idx == 0 else 0.0)
            layers_left = n - i
            groups_left = k - idx                   # current device plus the ones after it
            # Close the run when the next layer would overflow this card, or when holding on
            # would leave a later card with no layers at all.
            if used > 0 and groups_left > 1 and (run + c > budget or layers_left <= groups_left - 1):
                counts.append(used)
                run, used = 0.0, 0
                idx = len(counts)
                budget = caps[idx] * ratio - (pinned_gb if idx == 0 else 0.0)
            # Checked AFTER any close, and unconditionally: without this the FINAL device
            # accumulates without limit, and the search "succeeds" by piling every layer onto
            # the last card — which is the very imbalance this function exists to prevent.
            if run + c > budget:
                return None
            run += c
            used += 1
        counts.append(used)
        while len(counts) < k:                      # fewer layers than cards
            counts.append(0)
        return counts if len(counts) == k else None

    lo, hi = 0.0, (sum(costs) + pinned_gb) / min(caps)
    best: list[int] | None = None
    for _ in range(64):
        mid = (lo + hi) / 2
        got = feasible(mid)
        if got is not None:
            best, hi = got, mid
        else:
            lo = mid
    if best is None:
        base, rem = divmod(n, k)
        best = [base + (1 if i < rem else 0) for i in range(k)]
    return best


def _balanced_split(layers: int, n_cpu_moe: int, attention_gb: float,
                    expert_per_layer_gb: float, kv_gb: float, gpu_count: int,
                    pinned_gb: float, caps: list[float] | None = None) -> tuple[str, list[float]]:
    """(tensor-split string, per-card GB) for a layer split balanced by BYTES.

    llama.cpp's `split-mode = layer` divides layers by COUNT, weighted by --tensor-split. That
    is correct for a dense model, where every layer costs the same, and badly wrong for a MoE
    under `--n-cpu-moe N`: layers 0..N-1 keep only attention on the GPU while the rest carry
    full experts, so the per-layer cost jumps by an order of magnitude partway through. An even
    count split therefore hands one card nearly all the expensive layers.

    Measured on a 40-layer, 256-expert 35B at n-cpu-moe=24: llama.cpp tried to allocate
    15.0 GiB on device 1 of a 11.9 GiB card while device 0 still had ~9 GiB free. Balancing by
    bytes instead loaded it at 9.7/10.8 GiB across the two cards with the full 256K context.
    """
    if gpu_count < 2 or layers <= 0:
        return "", []
    costs = _layer_costs(layers, n_cpu_moe, attention_gb, expert_per_layer_gb, kv_gb)
    counts = _partition_min_max(costs, gpu_count, pinned_gb, caps)
    # A zero is only legitimate when there are genuinely fewer layers than cards; otherwise
    # something went wrong and an even split is the safer answer than a malformed one.
    if not counts or sum(counts) != layers:
        return "", []
    if any(c <= 0 for c in counts) and layers >= gpu_count:
        return "", []
    loads: list[float] = []
    at = 0
    for i, n in enumerate(counts):
        load = sum(costs[at:at + n]) + (pinned_gb if i == 0 else 0.0)
        loads.append(round(load, 2))
        at += n
    return ",".join(str(c) for c in counts), loads


def _find_fit(model_gb_full: float, kv_gb: float, budget_gb: float,
              layers: int, moe_ratio: float,
              card_ok: "Callable[[float, str, int], bool] | None" = None) -> tuple[bool, float, str, int]:
    """Return (fits, gpu_model_gb, offload_kind, n_layers) for a given (model, kv, budget).

    Strategy: try no offload first; then offload.
      * MoE  -> smallest n-cpu-moe (fewest expert layers moved) that fits; n=layers means cpu-moe
      * dense -> smallest number of whole layers moved to CPU via `ngl`

    For offload_kind == "ngl" the trailing int is the count of CPU-resident LAYERS, not experts.

    The dense branch used to be missing entirely: anything whose weights+KV exceeded the budget
    was reported as "doesn't fit on any backend", even though a dense model runs perfectly well
    with some layers on the CPU — that is exactly what the Fast/Balanced/Long presets do. The
    result was a red X and no options for, say, a 27 GB Q8 on 24 GB of VRAM, when the honest
    answer is "yes, with N layers offloaded, and here is the speed cost".

    `card_ok(gpu_gb, kind, n)` is an optional second test, used on multi-GPU backends to check
    that the per-card split of that configuration actually fits each card. It has to be part
    of the SEARCH, not a filter applied afterwards: the pooled budget and the per-card limit
    are satisfied at different offload levels, so the first configuration that clears the pool
    may still overflow one card while offloading one more layer clears both. Rejecting that
    first candidate instead of continuing produced a band of contexts reported as impossible
    while both smaller AND larger ones fitted — non-monotonic, and wrong.
    """
    def _ok(gpu: float, kind: str, n: int) -> bool:
        if gpu + kv_gb > budget_gb:
            return False
        return card_ok is None or card_ok(gpu, kind, n)

    if _ok(model_gb_full, "", 0):
        return True, model_gb_full, "", 0
    if layers <= 0:
        return False, model_gb_full, "", 0

    if moe_ratio <= 0:
        # Dense: move whole layers to the CPU. Keep at least one on the GPU — a fully
        # offloaded model is just CPU inference and the GPU backend has nothing to do.
        per_layer_gb = model_gb_full / layers
        for cpu_layers in range(1, layers):
            gpu = per_layer_gb * (layers - cpu_layers)
            if _ok(gpu, "ngl", cpu_layers):
                return True, gpu, "ngl", cpu_layers
        return False, model_gb_full, "", 0
    attention_gb = model_gb_full * (1 - moe_ratio)
    expert_per_layer_gb = model_gb_full * moe_ratio / layers
    # smallest n where fits — n counts CPU-offloaded layers
    for n in range(1, layers + 1):
        gpu = attention_gb + max(0, layers - n) * expert_per_layer_gb
        kind = "cpu-moe" if n >= layers else "n-cpu-moe"
        if _ok(gpu, kind, n):
            if n >= layers:
                return True, attention_gb, "cpu-moe", 0
            return True, gpu, "n-cpu-moe", n
    return False, attention_gb, "cpu-moe", 0  # even full offload can't fit (attention too big for GPU)


def _pareto_frontier(arch: str, layers: int, kv_heads: int, head_dim: int,
                     bytes_per: float, model_gb: float, moe_ratio: float,
                     budget_gb: float, ctx_candidates: list[int],
                     n_sessions: int = 1,
                     key_length: int | None = None, value_length: int | None = None,
                     full_attention_interval: int | None = None,
                     ssm_state_size: int | None = None,
                     v_bytes_per_elem: float | None = None,
                     card_ok: "Callable[[float, float, int], bool] | None" = None) -> list[tuple[int, int, float, float]]:
    """For a MoE model on a given backend, sweep n-cpu-moe from 0..layers.

    `card_ok(weight_gb, kv_gb, n_cpu_moe)` is the same per-card feasibility test the fit table
    applies. Without it the frontier can propose a point that clears the pooled budget but
    overflows one card, so a preset card recommends a configuration the table beside it marks
    as not fitting.
    Return list of (n_cpu_moe, max_per_session_ctx_fitting, gpu_weight_gb, kv_gb_at_max_ctx) — pareto frontier.
    Higher n_cpu_moe → higher max_ctx (more offloaded = less VRAM for weights = more room for KV).

    Candidates are interpreted as PER-SESSION ctx. KV is sized against total_ctx = ctx * n_sessions.
    """
    if moe_ratio <= 0 or layers <= 0:
        return []
    attention_gb = model_gb * (1 - moe_ratio)
    per_layer_gb = model_gb * moe_ratio / layers
    frontier: list[tuple[int, int, float, float]] = []
    n = max(1, int(n_sessions))
    for ncm in range(0, layers + 1):
        gpu_layers = layers - ncm
        weight_gb = attention_gb + gpu_layers * per_layer_gb
        if weight_gb >= budget_gb:
            continue  # can't fit even at ctx=0
        # find max per-session ctx that fits with this weight footprint
        best_ctx = 0
        best_kv = 0.0
        for ctx in ctx_candidates:
            kv_gb = kv_cache_bytes(arch, ctx * n, layers, kv_heads, head_dim, bytes_per,
                                   key_length=key_length, value_length=value_length,
                                   full_attention_interval=full_attention_interval,
                                   ssm_state_size=ssm_state_size,
                                   v_bytes_per_elem=v_bytes_per_elem) / (1024 ** 3)
            if card_ok is not None and not card_ok(weight_gb, kv_gb, ncm):
                continue
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


def _dense_frontier(arch: str, layers: int, kv_heads: int, head_dim: int,
                    bytes_per: float, model_gb: float, budget_gb: float,
                    ctx_candidates: list[int], n_sessions: int = 1,
                    key_length: int | None = None, value_length: int | None = None,
                    full_attention_interval: int | None = None,
                    ssm_state_size: int | None = None,
                    v_bytes_per_elem: float | None = None,
                    card_ok: "Callable[[float, float], bool] | None" = None) -> list[tuple[int, int, float, float]]:
    """Context-vs-speed frontier for a DENSE model, by moving whole layers off the GPU.

    MoE models offload expert weights (`--cpu-moe` / `--n-cpu-moe`), which is cheap because
    only a few experts are active per token. A dense model has no experts, so the only lever
    is `--n-gpu-layers` below the total: layers past that stay in host RAM (mmap'd from the
    GGUF) and every single token has to traverse them on the CPU. Freeing VRAM this way buys
    context, but it is far more expensive per layer than MoE offload — see _CPU_LAYER_PENALTY.

    Returns (cpu_layers, max_per_session_ctx, gpu_weight_gb, kv_gb_at_that_ctx), same tuple
    shape as _pareto_frontier so the preset builder can consume either.

    `card_ok(weight_gb, kv_gb)` applies the same per-card feasibility test the fit table uses.
    Without it the frontier proposes points that clear the pooled budget but overflow one
    card, so a preset card could recommend an ngl the table beside it marks as not fitting —
    and, worse, that llama.cpp would OOM.
    """
    if layers <= 0 or model_gb <= 0:
        return []
    per_layer_gb = model_gb / layers
    n = max(1, int(n_sessions))
    frontier: list[tuple[int, int, float, float]] = []
    for cpu_layers in range(0, layers):  # keep at least 1 layer on GPU
        gpu_layers = layers - cpu_layers
        weight_gb = per_layer_gb * gpu_layers
        if weight_gb >= budget_gb:
            continue
        best_ctx, best_kv = 0, 0.0
        for ctx in ctx_candidates:
            kv_gb = kv_cache_bytes(arch, ctx * n, layers, kv_heads, head_dim, bytes_per,
                                   key_length=key_length, value_length=value_length,
                                   full_attention_interval=full_attention_interval,
                                   ssm_state_size=ssm_state_size,
                                   v_bytes_per_elem=v_bytes_per_elem) / (1024 ** 3)
            if weight_gb + kv_gb <= budget_gb and ctx > best_ctx:
                if card_ok is not None and not card_ok(weight_gb, kv_gb):
                    continue
                best_ctx, best_kv = ctx, kv_gb
        if best_ctx == 0:
            continue
        # pareto step: only keep points that strictly improve max ctx
        if frontier and best_ctx <= frontier[-1][1]:
            continue
        frontier.append((cpu_layers, best_ctx, weight_gb, best_kv))
    return frontier


def _frontier_options(frontier: list[tuple[int, int, float, float]], backend: str,
                      layers: int, dense: bool) -> list[PresetOption]:
    """Turn EVERY frontier point into a PresetOption, ascending by ctx.

    Feeds the custom slider, which lets any achievable point on the speed/context curve be
    selected rather than only the three named samples.
    """
    out: list[PresetOption] = []
    for off, ctx, gpu_gb, kv_gb in sorted(frontier, key=lambda f: f[1]):
        if dense:
            gpu_layers = layers - off
            speed = round(layers / (gpu_layers + off * _CPU_LAYER_PENALTY), 3) if layers else 0.0
            out.append(PresetOption(
                key=f"pt{off}", label=f"{gpu_layers}/{layers} layers", icon="sliders",
                backend=backend, ctx=ctx, n_cpu_moe=0,
                offload_kind=("ngl" if off > 0 else ""),
                gpu_layers=gpu_layers, total_layers=layers,
                gpu_gb=round(gpu_gb, 2), kv_gb=round(kv_gb, 2),
                speed_score=speed, ngl=(gpu_layers if off > 0 else 999),
            ))
        else:
            speed = round((layers - off) / layers, 3) if layers else 0.0
            out.append(PresetOption(
                key=f"pt{off}", label=f"ncm {off}", icon="sliders",
                backend=backend, ctx=ctx, n_cpu_moe=off,
                offload_kind=("cpu-moe" if off >= layers else ("n-cpu-moe" if off > 0 else "")),
                gpu_layers=layers - off, total_layers=layers,
                gpu_gb=round(gpu_gb, 2), kv_gb=round(kv_gb, 2),
                speed_score=speed,
            ))
    return out


def _presets_from_dense_frontier(frontier: list[tuple[int, int, float, float]], backend: str,
                                 layers: int) -> list[PresetOption]:
    """Fast / Balanced / Long-context picks from a dense layer-offload frontier."""
    if not frontier:
        return []

    def _speed(cpu_layers: int) -> float:
        # Relative throughput estimate. GPU layer = 1 unit of time, CPU layer = _CPU_LAYER_PENALTY.
        # Dense offload hurts far more than the MoE equivalent: with a 20x penalty, moving just
        # 10% of layers off already costs roughly two thirds of your tokens/sec.
        if layers <= 0:
            return 0.0
        gpu = layers - cpu_layers
        return round(layers / (gpu + cpu_layers * _CPU_LAYER_PENALTY), 3)

    def _make(key: str, label: str, icon: str, e: tuple[int, int, float, float]) -> PresetOption:
        cpu_layers, ctx, gpu_gb, kv_gb = e
        gpu_layers = layers - cpu_layers
        return PresetOption(
            key=key, label=label, icon=icon, backend=backend, ctx=ctx,
            n_cpu_moe=0,
            offload_kind=("ngl" if cpu_layers > 0 else ""),
            gpu_layers=gpu_layers, total_layers=layers,
            gpu_gb=round(gpu_gb, 2), kv_gb=round(kv_gb, 2),
            speed_score=_speed(cpu_layers),
            # 999 keeps llama-server's "everything on GPU" behaviour when nothing is offloaded.
            ngl=(gpu_layers if cpu_layers > 0 else 999),
        )

    fast_entry = min(frontier, key=lambda f: f[0])                 # fewest layers offloaded
    long_entry = max(frontier, key=lambda f: (f[1], -f[0]))        # most context
    fi, li = frontier.index(fast_entry), frontier.index(long_entry)
    if fi > li:
        fi, li = li, fi
    mi = (fi + li) // 2
    # Never let Balanced collapse onto Fast/Long: when they are adjacent the midpoint
    # rounds onto one of them and the chip disappears. Step inward instead.
    if mi in (fi, li) and li - fi >= 2:
        mi = fi + 1
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

    # Balanced: a point STRICTLY between fast and long, so it is never a duplicate of
    # either. Plain midpoint rounding collapses onto fast when the two are adjacent
    # (e.g. a 2-point frontier gives (0+1)//2 == 0), which is why Balanced used to vanish.
    fi = frontier.index(fast_entry)
    li = frontier.index(long_entry)
    if fi > li: fi, li = li, fi
    mi = (fi + li) // 2
    if mi in (fi, li) and li - fi >= 2:
        mi = fi + 1
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


def _looks_like_draft(filename: str) -> bool:
    """Heuristic: is this GGUF a speculative-decoding draft head?

    Match on isolated tokens (`-draft-`, `-mtp-`, or the filename starting/ending with those)
    so we don't false-match main models that have "noMTP" or "nodraft" in their name — those
    are variants explicitly WITHOUT MTP support (e.g. `Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf`).
    """
    low = filename.lower()
    # Explicit "no MTP" or "no draft" variants — these are main models, not drafts
    for negation in ("nomtp", "no-mtp", "no_mtp", "nodraft", "no-draft", "no_draft"):
        if negation in low:
            return False
    # Positive matches — token boundaries only
    for token in ("-draft-", "-draft.", "_draft_", ".draft.", "-mtp-", "_mtp_", ".mtp."):
        if token in low:
            return True
    # Also match "draft" or "mtp" as leading/trailing tokens
    stem = low[:-5] if low.endswith(".gguf") else low
    parts = stem.replace("_", "-").split("-")
    return parts and (parts[0] in ("draft", "mtp") or parts[-1] in ("draft", "mtp"))


def _find_mmproj(models_dir: "Path | None", section_name: str, subdir: str = "") -> str:
    """Look for THIS model's mmproj companion. Returns an absolute /models path, or "".

    Deliberately strict. A projector only counts as this model's companion when it is
    co-located with the model:

      * model in a subdir  -> only that subdir is searched
      * model at top level -> only top-level projectors whose filename contains the
                              model's full stem

    Earlier revisions also tried fuzzy prefix matching on subdir/file names and an
    "if there's only one mmproj anywhere, use it" fallback. Those were flat-layout
    legacy and actively wrong under one-directory-per-model: `Qwen3.8-27B-Q4_K_M`
    picked up the projector belonging to `Qwen3.8-27B-OBLITERATED-Q4_K_M` because
    both normalize to a shared `qwen3827` prefix, which then tripped the multimodal
    ctx cap on a text-only model. Same-directory co-location is the only signal that
    does not cross-contaminate model families.
    """
    if models_dir is None:
        return ""

    def _is_mmproj(name: str) -> bool:
        return name.lower().endswith(".gguf") and "mmproj" in name.lower()

    # Model lives in its own directory: its companion is in there or it has none.
    # A repo often ships several precisions of the same projector (BF16 / F16 / F32).
    # Prefer the SMALLEST: the projector is pinned to the main GPU and competes directly
    # with the KV cache for that card's VRAM, and an F32 projector is typically 2x the
    # size of the BF16 one for no meaningful quality gain.
    if subdir:
        subdir_path = models_dir / subdir
        best: tuple[int, str] | None = None
        try:
            for p in sorted(subdir_path.iterdir()):
                if p.is_file() and _is_mmproj(p.name):
                    sz = p.stat().st_size
                    if best is None or sz < best[0]:
                        best = (sz, p.name)
        except OSError:
            pass
        return f"/models/{subdir}/{best[1]}" if best else ""

    # Flat layout: require the projector filename to carry the model's full stem,
    # so `foo-Q4_K_M.gguf` matches `foo-Q4_K_M-mmproj.gguf` but never a sibling model.
    stem_key = section_name.lower().replace("-", "").replace("_", "").replace(".", "")
    if not stem_key:
        return ""
    try:
        for p in sorted(models_dir.iterdir()):
            if not (p.is_file() and _is_mmproj(p.name)):
                continue
            name_key = p.name.lower().replace("-", "").replace("_", "").replace(".", "")
            if stem_key in name_key:
                return f"/models/{p.name}"
    except OSError:
        pass
    return ""


def analyze(*,
            summary: dict,
            file_size: int,
            backends: list[dict],           # [{name, vendor, vram_gb, baseline: {ini_key: val}}]
            model_rel: str = "",
            current_section: dict[str, str] | None = None,
            preset: str = "",
            n_sessions: int = 1,
            models_dir: "Path | None" = None,
            section_name: str = "",
            model_subdir: str = "",
            mmproj_gb_override: float | None = None) -> Recommendation:
    # Clamp n_sessions to a sensible range for a homelab. Above 8 the per-slot ctx
    # shrinks below usability for real chat, and llama-server continuous batching
    # overhead starts dominating.
    n_sessions = max(1, min(int(n_sessions or 1), 8))
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

    # Model VRAM depends on whether we'll be layer-splitting across multiple GPUs.
    # We check per-backend below; use single-GPU overhead as the base and apply the split
    # multiplier inside the per-backend loop when that backend hosts >1 GPU.
    model_gb_raw = file_size / (1024 ** 3)
    moe_ratio = _moe_ratio(experts)

    # Plan around q8_0 KV cache: near-identical quality to f16 in practice,
    # and lets us fit ~2× the context. Users can override in the form if they want f16.
    bytes_per = _cache_dtype_bytes(_CACHE_DEFAULT)

    # HARD STOP if we have nothing to fit against. Callers drop backends whose VRAM probes
    # as 0 (CPU-only containers, or a GPU probe that failed), so an empty list here is
    # ambiguous with "model too big" downstream — the panel would otherwise tell the user
    # to try a smaller quant when the real problem is that no GPU backend was discovered.
    if not backends:
        return Recommendation(
            plans=[], recommended_backend="", recommended_ctx=0,
            error="No GPU backend available to size against. Either no llama.cpp container was "
                  "discovered, or its VRAM probe returned 0 (CPU-only build, or nvidia-smi / "
                  "rocm-smi not usable inside the container). Check the Containers page: a backend "
                  "must appear there with a non-zero VRAM total before Autoconfig can plan. "
                  "This is not a statement about whether the model would fit.",
        )

    # A backend that is discovered but reports 0 GB is a DIFFERENT failure from one that is
    # genuinely too small, and it must not be reported as the latter. Sizing against zero
    # produces a negative budget, so every context fails and the panel says "doesn't fit at
    # any context" — which reads as a verdict on the model when it is really a missing probe.
    # Vulkan images are the common case: they carry neither nvidia-smi nor rocm-smi, so there
    # is nothing to read and GPU_VRAM has to supply the number.
    # A model larger than VRAM + system RAM cannot run at ANY offload setting: CPU-resident
    # layers live in system memory, so there is nowhere left to put them. Offering it a
    # context estimate at 2% speed is worse than saying nothing, because it reads as "slow
    # but possible" when the honest answer is "not on this machine".
    _ram = max((float(b.get("host_ram_gb") or 0) for b in backends), default=0.0)
    _vram = max((float(b.get("vram_gb") or 0) for b in backends), default=0.0)
    if _ram > 0 and model_gb_raw > (_vram + _ram):
        return Recommendation(
            plans=[], recommended_backend="", recommended_ctx=0,
            error=f"This model needs about {model_gb_raw:.0f} GB, more than this machine's "
                  f"{_vram:.0f} GB VRAM plus {_ram:.0f} GB RAM ({_vram + _ram:.0f} GB total). "
                  "CPU offload moves layers into system memory, so there is no offload setting "
                  "that makes it fit. A smaller quantisation of the same model is the option.",
        )

    _sized = [b for b in backends if float(b.get("vram_gb") or 0) > 0]
    if not _sized:
        _names = ", ".join(str(b.get("name") or "?") for b in backends)
        return Recommendation(
            plans=[], recommended_backend="", recommended_ctx=0,
            error=f"Backend(s) found ({_names}) but none report their VRAM, so there is nothing "
                  "to size against. This is usually a Vulkan build: the image ships no vendor "
                  "SMI tool, so VRAM cannot be probed. Declare it instead — set "
                  "GPU_VRAM=<container>:<GB> on the model-loader service (e.g. "
                  "GPU_VRAM=llama-vulkan:16) and restart it. Nothing here says the model "
                  "does not fit; the size of the card is simply unknown.",
        )
    backends = _sized

    # HARD STOP if the KV cache cannot be sized from this GGUF's metadata.
    #
    # kv_cache_bytes() returns 0 when block_count / attention_head_count_kv / head_dim are
    # missing or zero — which happens on architectures whose keys we don't parse yet. Zero
    # KV silently means "the cache is free", so every candidate fits and the picker happily
    # returns the largest context in the table. That is the worst possible failure: a
    # confident recommendation of e.g. 1048576 for a model that will OOM the instant it
    # loads. Refuse to guess instead — an honest error beats a plausible wrong number.
    if kv_cache_bytes(arch, 4096, layers, kv_heads, head_dim, bytes_per,
                      key_length=key_length, value_length=value_length,
                      full_attention_interval=full_attention_interval,
                      ssm_state_size=ssm_state_size) <= 0:
        missing = [k for k, v in (("block_count", layers),
                                  ("attention_head_count_kv", kv_heads),
                                  ("head_dim (embedding_length / attention_head_count)", head_dim))
                   if not v]
        return Recommendation(
            plans=[], recommended_backend="", recommended_ctx=0,
            error=("Cannot size the KV cache from this model's metadata"
                   + (" — missing/zero: " + ", ".join(missing) if missing else "")
                   + ". Autoconfig will not guess a context size, because an unsized KV cache "
                     "would look free and produce a recommendation that OOMs on load. "
                     "Set ctx-size manually in the form and verify it loads."),
        )

    # NOTE: Autoconfig deliberately does NOT auto-wire speculative decoding / draft
    # models anymore. In practice, community MTP heads (e.g. JonathanColetti's Qwen3.8
    # Uncensored draft) segfaulted llama.cpp with every main quant we tried, and there's
    # no reliable programmatic way to tell a working draft from a broken one before load.
    # The `spec-draft-*` fields are still exposed in the ini schema for manual use if you
    # find a draft that provably works with your llama.cpp build. `_looks_like_draft` is
    # kept only so `ini._is_companion` can still hide draft files from the "unregistered
    # GGUF" list on the Config page.

    # Detect a companion mmproj (multimodal projector — vision, audio, or other) and
    # reserve its real VRAM cost, rather than capping ctx at an arbitrary ceiling.
    #
    # A projector is not layer-split; it loads entirely onto the main GPU, so pooled-VRAM
    # budgeting overshoots by its full size, and the modality encoder allocates scratch on
    # top. An earlier revision handled that by clamping every multimodal model to 32K ctx,
    # which was crude and wrong: it slashed a 262K-native MoE coder to 32K purely because a
    # projector sat in its directory, and (because so many offload points then tied at the
    # same clamped ctx) collapsed the MoE preset frontier to a single option.
    # Reserving the measured cost instead lets the normal fit math decide.
    mmproj_rel = ""
    mmproj_gb = 0.0
    if models_dir is not None and section_name:
        # Size against the projector that will ACTUALLY be loaded. If the preset already
        # names one, that file wins — sizing against a different (possibly smaller) file
        # in the same directory silently under-reserves and the model OOMs on device 0.
        mmproj_rel = (current_section or {}).get("mmproj", "").strip() \
            or _find_mmproj(models_dir, section_name, model_subdir)
        if mmproj_rel:
            try:
                _mp = Path(str(mmproj_rel).replace("/models", str(models_dir), 1))
                mmproj_gb = _mp.stat().st_size / (1024 ** 3)
            except OSError:
                mmproj_gb = 0.0
    # Callers that can see a projector but have no local file (the HF search estimator, which
    # knows sizes from the repo tree but hasn't downloaded anything) pass its size directly.
    # Without this the estimate silently ignores the projector and reads optimistically high.
    if mmproj_gb_override is not None and mmproj_gb_override > 0:
        mmproj_gb = mmproj_gb_override
        mmproj_rel = mmproj_rel or "(remote projector)"
    has_mmproj = bool(mmproj_rel)
    # VRAM pinned to the MAIN GPU for multimodal: projector weights + encoder compute buffer.
    mmproj_vram_gb = (mmproj_gb * _MMPROJ_VRAM_MULT + _MMPROJ_COMPUTE_GB) if has_mmproj else 0.0

    # Candidate ctx values: default cap is the model's native ctx. Linear RoPE extension
    # to 2× is possible but (a) degrades quality noticeably, (b) inflates compute buffers
    # unpredictably. Not worth the OOM risk as an autoconfig default — users who want
    # extension can dial ctx-size up in the form manually.
    cands = sorted(set(list(_CTX_CANDIDATES) + ([native_ctx] if native_ctx else [])))
    if native_ctx:
        cands = [c for c in cands if c <= native_ctx]

    plans: list[BackendPlan] = []
    # remember offload used at the recommended-ctx per backend, for the values dict
    per_backend_offload: dict[str, tuple[str, int]] = {}

    def _fit_all(v_bytes: float, weight_for: str = "",
                 weight_gb: float | None = None) -> tuple[list[BackendPlan], dict[str, tuple[str, int]]]:
        """Run the whole per-backend ctx sweep for a given V-cache element size.
        K always stays at _CACHE_DEFAULT; only V is varied.

        weight_for/weight_gb pin one backend's GPU-resident weight to an already-offloaded
        figure, so the fit table can be recomputed to match a chosen preset instead of
        always showing the everything-on-GPU case."""
        _plans: list[BackendPlan] = []
        _offload: dict[str, tuple[str, int]] = {}
        for b in backends:
            _w = weight_gb if (weight_gb is not None and b["name"] == weight_for) else None
            rows, max_fit, max_fit_offload = _fit_backend(b, v_bytes, _w)
            _plans.append(BackendPlan(
                name=b["name"], vendor=b.get("vendor", ""),
                vram_gb=float(b["vram_gb"]), rows=rows,
                max_ctx=max_fit, fits_at_all=(max_fit > 0),
            ))
            if max_fit:
                _offload[b["name"]] = max_fit_offload
        return _plans, _offload

    def _fit_backend(b: dict, v_bytes: float,
                     weight_gb: float | None = None) -> tuple[list[FitRow], int, tuple[str, int]]:
        rows: list[FitRow] = []
        gpu_count = max(1, int(b.get("gpu_count", 1)))
        # Reserve scales per GPU (each CUDA context takes ~500 MB just to be initialized).
        # NOTE: compute buffer VRAM (prompt-eval scratch, ~1-2 GB at stock ub) is NOT
        # subtracted from budget. Real-world calibration on this rig shows the 1.08
        # split-overhead + 0.5 GB/GPU reserve already over-estimates enough to absorb
        # the compute buffer for models we've tested. Explicitly subtracting compute_gb
        # here over-corrects and drops working ctx picks. If we hit OOMs on models the
        # picker approves, revisit this.
        # The projector is pinned to the main GPU, not layer-split. Under an even layer
        # split every GB pinned to one card costs gpu_count GB of usable POOLED capacity,
        # because the matching share on the other cards cannot be used for it either.
        # Measured: Qwen3-VL's 801 MB projector allocates ~1584 MiB on device 0, and its
        # compute buffer (~1858 MiB) lands there too — so on 2 GPUs a naive pooled
        # subtraction under-reserves by ~2x and the model OOMs on device 0 while the
        # second card still shows free VRAM.
        budget = float(b["vram_gb"]) - _RESERVE_PER_GPU * gpu_count - mmproj_vram_gb
        # Per-card capacities for the single-device feasibility check below. Falls back to an
        # even division of the pool when the sampler has not reported individual cards.
        card_caps = [c - _RESERVE_PER_GPU for c in (b.get("card_vram_gb") or [])]
        if gpu_count > 1 and not card_caps:
            card_caps = [(float(b["vram_gb"]) / gpu_count) - _RESERVE_PER_GPU] * gpu_count
        # The projector, compute buffer and cuBLAS workspace are not layer-split: they all
        # land on the main GPU, so device 0 starts with less room than its siblings.
        pinned_gb = mmproj_vram_gb
        # Model overhead: single-GPU is basically file size; layer-split adds ~5% for cross-card handoffs
        overhead_mul = _MODEL_OVERHEAD_SPLIT if gpu_count > 1 else _MODEL_OVERHEAD_SINGLE
        # When weight_gb is supplied the offload is already baked into it, so pass moe_ratio=0
        # below to stop _find_fit applying a second round of expert offload on top.
        model_gb = model_gb_raw * overhead_mul if weight_gb is None else weight_gb
        eff_moe = moe_ratio if weight_gb is None else 0.0
        max_fit = 0
        max_fit_offload: tuple[str, int] = ("", 0)
        for per_session_ctx in cands:
            # ctx-size llama-server sees = per_session * n_sessions.
            # KV cache is sized against the TOTAL because each slot is allocated
            # a contiguous chunk in the shared cache.
            total_ctx = per_session_ctx * n_sessions
            kv_gb = kv_cache_bytes(arch, total_ctx, layers, kv_heads, head_dim, bytes_per,
                                   key_length=key_length, value_length=value_length,
                                   full_attention_interval=full_attention_interval,
                                   ssm_state_size=ssm_state_size,
                                   v_bytes_per_elem=v_bytes) / (1024 ** 3)
            # The per-card test is handed to the search rather than applied to its answer, so
            # it can keep offloading until BOTH the pool and every individual card are happy.
            def _card_ok(gpu_gb: float, kind: str, n: int,
                         _kv=kv_gb, _caps=card_caps, _pin=pinned_gb) -> bool:
                if gpu_count <= 1 or not _caps:
                    return True
                if eff_moe > 0:
                    att = model_gb * (1 - eff_moe)
                    exp = (model_gb * eff_moe / layers) if layers else 0.0
                    ncm = n if kind == "n-cpu-moe" else (layers if kind == "cpu-moe" else 0)
                else:
                    att, exp, ncm = gpu_gb, 0.0, 0
                return _split_feasible(layers, ncm, att, exp, _kv, gpu_count, _pin, _caps)

            fits, gpu_model_gb, offload_kind, n_cm = _find_fit(
                model_gb, kv_gb, budget, layers, eff_moe, card_ok=_card_ok)
            total = gpu_model_gb + kv_gb
            rows.append(FitRow(
                ctx=per_session_ctx, total_ctx=total_ctx,
                model_gb=round(gpu_model_gb, 2), kv_gb=round(kv_gb, 2),
                total_gb=round(total, 2), fits=fits,
                free_gb=round(float(b["vram_gb"]) - total, 2),
                offload_kind=offload_kind, n_cpu_moe=n_cm,
                gpu_pct=(round(100.0 * gpu_model_gb / model_gb) if model_gb > 0 else 100),
            ))
            if fits:
                max_fit = per_session_ctx
                max_fit_offload = (offload_kind, n_cm)
        return rows, max_fit, max_fit_offload

    # Symmetric q8_0 K/V, always.
    #
    # An earlier revision escalated V to q4_0 when the model could not reach its context
    # ceiling, on the theory that V tolerates quantization better than K (K feeds QK^T
    # through softmax, which amplifies error; V is only weighted-summed afterwards). The
    # VRAM math was real — 2.1 GiB freed on Qwen3.8-27B at 131K — and short-prompt
    # benchmarks looked fine, which is exactly why it shipped.
    #
    # It was a bad trade. There is no CUDA flash-attention kernel for a q4_0 V cache at
    # this model's 256-wide head dim, so attention silently falls back to CPU. Measured
    # on the same 4178-token prompt: q8_0 = 1737 tok/s prompt eval (2.4 s), q4_0 = 115
    # tok/s (36.2 s). A 15x regression, invisible on short prompts because they do almost
    # no attention work. Whether a given (quant, head_dim, arch) combination has a CUDA
    # kernel is not something we can determine from GGUF metadata, so do not guess:
    # trading a predictable slice of context for an unpredictable 15x cliff is not worth
    # it. Users who want asymmetric KV can set `cache-type-v` by hand and benchmark a
    # LONG prompt — short ones will not reveal the fallback.
    v_cache_type = _CACHE_DEFAULT
    plans, per_backend_offload = _fit_all(bytes_per)

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
        # ctx-size llama-server allocates = per-session ctx × n_sessions.
        values["ctx-size"] = str(rec_ctx * n_sessions)
    # ALWAYS emit parallel, even for a single session. llama-server's own default is
    # n_slots = 4, not 1 — leaving this unset silently quarters each slot's context
    # (ctx-size is the shared total, divided across slots).
    values["parallel"] = str(n_sessions)
    if n_sessions > 1:
        # Continuous batching is on by default in recent llama-server, but be explicit
        # so anyone reading the ini can tell we planned for multi-slot serving.
        values["cont-batching"] = "on"
        # When a slot hits its per-session ctx cap, slide oldest tokens out instead
        # of erroring. keep=256 pins roughly one system-prompt's worth of tokens
        # at the start so instructions survive the shift; user tunes this to their
        # actual system-prompt length in the form.
        values["context-shift"] = "on"
        values["keep"] = "256"
        # batch-size is a scheduling knob — 4096 costs negligible VRAM and helps
        # continuous-batching pack incoming prompts. Safe on any hardware that fit
        # the model in the first place.
        values["batch-size"] = "4096"
        # Deliberately NOT bumping ubatch-size. On layer-split multi-GPU running a
        # dense large model, the compute buffer scales as roughly 8 × ub × layers ×
        # hidden bytes. Empirically: 27B (64L, 5120H) at ub=2048 allocates ~4.7 GB
        # of compute buffers across cards. Bumping ub is the actual TTFT lever, but
        # it needs 2+ GB of VRAM headroom which most tight fits don't have. See the
        # quirk below for manual tuning.
    values["ngl"] = "999"
    values["flash-attn"] = "on"
    values["cache-type-k"] = _CACHE_DEFAULT
    # V may have been dropped to q4_0 by the fit search to buy context (see pass 2 above).
    values["cache-type-v"] = v_cache_type
    # Prefix cache reuse — llama-server reuses KV from a previous request when
    # the new prompt shares a prefix. Huge TTFT win for chat continuations and
    # OpenWebUI-style clients that resend the whole history each turn. Costs
    # nothing, no downside, always safe. Off by default in llama-server, which
    # is a footgun for chat use.
    values["cache-reuse"] = "1"
    if summary.get("chat_template"):
        values["jinja"] = "true"

    # Multimodal: look for an adjacent mmproj file the user hasn't already set
    if section_name:
        current_mmproj = (current_section or {}).get("mmproj", "").strip()
        if current_mmproj:
            # Respect user's existing choice — echo it so Fill preserves it
            values["mmproj"] = current_mmproj
        else:
            if mmproj_rel:  # already resolved above, when sizing the VRAM reservation
                values["mmproj"] = mmproj_rel

    rope_type = m.get("rope_scaling_type")

    # MoE offload for the recommended (backend, ctx)
    active_preset = ""
    presets: list[PresetOption] = []
    frontier_opts: list[PresetOption] = []
    # The rule: if the model fits entirely on the GPU AND reaches its native context that
    # way, there is nothing to trade and one option is the whole truth. Offload can only
    # buy context, and there is no context left to buy. Anything short of that — either it
    # can't hold every layer, or it can but not at native ctx — means a real speed/context
    # tradeoff exists and the user should get the choices.
    _fits_full_gpu = False
    if recommended and rec_ctx > 0 and native_ctx > 0:
        _no_offload_ctx = max(
            (r.ctx for r in recommended.rows if r.fits and not r.offload_kind),
            default=0,
        )
        _fits_full_gpu = _no_offload_ctx >= native_ctx
    if recommended and rec_ctx > 0:
        off_kind, n_cm = per_backend_offload.get(recommended.name, ("", 0))
        # If MoE + offload needed, compute the preset frontier on the recommended backend
        # and let the requested preset override ctx / ncm.
        is_moe_now2 = isinstance(experts, int) and experts > 1
        if is_moe_now2 and off_kind:
            rec_backend = next((b for b in backends if b["name"] == recommended.name), None)
            gpu_count = max(1, int((rec_backend or {}).get("gpu_count", 1)))
            overhead_mul = _MODEL_OVERHEAD_SPLIT if gpu_count > 1 else _MODEL_OVERHEAD_SINGLE
            model_gb_rec = model_gb_raw * overhead_mul
            budget = float(recommended.vram_gb) - _RESERVE_PER_GPU * gpu_count - mmproj_vram_gb
            # Same per-card test the fit table applies. For a MoE the split is attention
            # (always resident) plus the experts of every layer at or above n-cpu-moe.
            _mcaps = [c - _RESERVE_PER_GPU for c in ((rec_backend or {}).get("card_vram_gb") or [])]
            if gpu_count > 1 and len(_mcaps) != gpu_count:
                _mcaps = [(float(recommended.vram_gb) / gpu_count) - _RESERVE_PER_GPU] * gpu_count

            def _moe_card_ok(weight_gb: float, kv_gb: float, ncm: int) -> bool:
                if gpu_count <= 1 or not _mcaps:
                    return True
                att = model_gb_rec * (1 - moe_ratio)
                exp = (model_gb_rec * moe_ratio / layers) if layers else 0.0
                return _split_feasible(layers, ncm, att, exp, kv_gb,
                                       gpu_count, mmproj_vram_gb, _mcaps)

            frontier = _pareto_frontier(arch, layers, kv_heads, head_dim, bytes_per,
                                        model_gb_rec, moe_ratio, budget, cands,
                                        n_sessions=n_sessions,
                                        key_length=key_length, value_length=value_length,
                                        full_attention_interval=full_attention_interval,
                                        ssm_state_size=ssm_state_size,
                                        v_bytes_per_elem=_cache_dtype_bytes(v_cache_type),
                                        card_ok=_moe_card_ok)
            presets = _presets_from_frontier(frontier, recommended.name, layers, native_ctx)
            frontier_opts = _frontier_options(frontier, recommended.name, layers, dense=False)
            if presets:
                chosen = next((p for p in presets if p.key == preset), presets[len(presets) // 2])
                active_preset = chosen.key
                rec_ctx = chosen.ctx
                off_kind = chosen.offload_kind
                n_cm = chosen.n_cpu_moe
                values["ctx-size"] = str(rec_ctx * n_sessions)
                # The fit table is deliberately NOT recomputed against the chosen preset's
                # weight. Doing so pinned every row to that one offload level, so the table
                # reported the same GPU-resident weight at every context and claimed 100%
                # resident everywhere — while the preset card above it said 43/65 layers.
                # The per-context search already agrees with the frontier (verified: both
                # pick 22 layers off at 262144), so leaving it alone is both correct and
                # consistent, and each row answers what THAT context actually costs.
        elif not is_moe_now2 and layers > 0:
            # DENSE model: no experts to offload, so trade whole layers for context via
            # `ngl`. Offered whenever the model fits at all, not only when forced — the
            # whole point is letting you choose context over speed deliberately.
            rec_backend = next((b for b in backends if b["name"] == recommended.name), None)
            gpu_count = max(1, int((rec_backend or {}).get("gpu_count", 1)))
            overhead_mul = _MODEL_OVERHEAD_SPLIT if gpu_count > 1 else _MODEL_OVERHEAD_SINGLE
            model_gb_rec = model_gb_raw * overhead_mul
            budget = float(recommended.vram_gb) - _RESERVE_PER_GPU * gpu_count - mmproj_vram_gb
            # Same per-card test the fit table applies, so the presets cannot propose a
            # point the table marks as not fitting.
            _fcaps = [c - _RESERVE_PER_GPU for c in ((rec_backend or {}).get("card_vram_gb") or [])]
            if gpu_count > 1 and len(_fcaps) != gpu_count:
                _fcaps = [(float(recommended.vram_gb) / gpu_count) - _RESERVE_PER_GPU] * gpu_count

            def _front_card_ok(weight_gb: float, kv_gb: float) -> bool:
                if gpu_count <= 1 or not _fcaps:
                    return True
                return _split_feasible(layers, 0, weight_gb, 0.0, kv_gb,
                                       gpu_count, mmproj_vram_gb, _fcaps)

            dfront = _dense_frontier(arch, layers, kv_heads, head_dim, bytes_per,
                                     model_gb_rec, budget, cands, n_sessions=n_sessions,
                                     key_length=key_length, value_length=value_length,
                                     full_attention_interval=full_attention_interval,
                                     ssm_state_size=ssm_state_size,
                                     v_bytes_per_elem=_cache_dtype_bytes(v_cache_type),
                                     card_ok=_front_card_ok)
            presets = _presets_from_dense_frontier(dfront, recommended.name, layers)
            frontier_opts = _frontier_options(dfront, recommended.name, layers, dense=True)
            if presets:
                # Dense offload is OPT-IN. Unlike MoE — where offload is the only way to make
                # the model fit — a dense model already fits with every layer on the GPU, and
                # trading layers for context costs ~3x throughput (measured). So when the
                # caller hasn't explicitly picked a preset, stay on "fast" (no offload) rather
                # than silently spending speed the user never asked to spend.
                want = preset or "fast"
                chosen = next((p for p in presets if p.key == want), presets[0])
                active_preset = chosen.key
                rec_ctx = chosen.ctx
                values["ctx-size"] = str(rec_ctx * n_sessions)
                if chosen.ngl > 0:
                    values["ngl"] = str(chosen.ngl)
                # Same as the MoE branch: the table stays per-context rather than being
                # recomputed at the chosen preset's offload level.
        if off_kind == "cpu-moe":
            values["cpu-moe"] = "true"
        elif off_kind == "n-cpu-moe" and n_cm > 0:
            values["n-cpu-moe"] = str(n_cm)

        # With expert offload on a multi-GPU layer split, an even split is a byte imbalance:
        # layers below n-cpu-moe keep only attention on the GPU, the rest carry full experts.
        # Emit proportions that equalize actual bytes, or one card OOMs while the other idles.
        _gc = max(1, int((rec_backend or {}).get("gpu_count", 1)))
        if _gc > 1 and off_kind in ("n-cpu-moe", "cpu-moe") and layers > 0 and moe_ratio > 0:
            _mg = model_gb_raw * _MODEL_OVERHEAD_SPLIT
            _kv = kv_cache_bytes(arch, rec_ctx * n_sessions, layers, kv_heads, head_dim,
                                 bytes_per, key_length=key_length, value_length=value_length,
                                 full_attention_interval=full_attention_interval,
                                 ssm_state_size=ssm_state_size) / (1024 ** 3)
            _caps = [c - _RESERVE_PER_GPU for c in ((rec_backend or {}).get("card_vram_gb") or [])]
            if len(_caps) != _gc:
                _caps = [(float(recommended.vram_gb) / _gc) - _RESERVE_PER_GPU] * _gc
            _ts, _loads = _balanced_split(
                layers, layers if off_kind == "cpu-moe" else n_cm,
                _mg * (1 - moe_ratio), _mg * moe_ratio / layers,
                _kv, _gc, mmproj_vram_gb, _caps)
            if _ts:
                values["tensor-split"] = _ts
                values["split-mode"] = "layer"

    # Reasoning / thinking — infer from chat-template scanning
    features = summary.get("chat_template_features") or {}
    tpl_kwargs: dict[str, Any] = {}
    if features.get("accepts_enable_thinking"):
        # Use the dedicated --reasoning flag; setting enable_thinking via
        # chat-template-kwargs is deprecated in recent llama.cpp builds.
        values["reasoning"] = "on"
    if features.get("accepts_reasoning_effort"):
        tpl_kwargs["reasoning_effort"] = "medium"
    if tpl_kwargs:
        import json as _json
        values["chat-template-kwargs"] = _json.dumps(tpl_kwargs)
    if features.get("uses_think_tags") or features.get("uses_channel_thought"):
        # Ensures OpenAI-compatible clients (OpenWebUI) put thoughts in message.reasoning_content
        # so they render as a collapsible instead of noise in the answer.
        values["reasoning-format"] = "deepseek"
    if features.get("accepts_preserve_thinking"):
        # Carries reasoning across turns instead of discarding it each message. llama-server
        # itself suggests this at load time for templates that support it ("chat template
        # supports preserving reasoning, consider enabling it via --reasoning-preserve").
        values["reasoning-preserve"] = "on"

    # Multimodal extras — only meaningful when a projector is actually attached.
    if has_mmproj:
        # Keep the projector on GPU. It is small and CPU-side image encoding is slow.
        values["mmproj-offload"] = "on"
        # Bound how many tokens a single image may consume.
        #
        # Vision models with dynamic resolution size their decode buffer from the image,
        # so a big photo allocates far more VRAM than a small one — and that allocation
        # happens at INFERENCE time, long after the fit math approved the load. The model
        # therefore loads fine, serves text fine, and then dies the moment an image
        # arrives, inside mtmd_helper_decode_image_chunk.
        #
        # Measured on Qwen3.8-27B-OBLITERATED (ctx 262144, ngl 43, ~400 MB free on the main
        # GPU): 1536x1536 decoded fine, 2048x2048 hard-OOM'd and killed the instance. With
        # image-max-tokens=1024 both 2048x2048 and 3072x2048 decode correctly and peak VRAM
        # stops moving with resolution — the cap converts an unbounded, input-dependent
        # spike into a fixed cost the fit math can actually live with.
        #
        # 1024 is also the floor llama-server asks for on Qwen-VL ("require at minimum 1024
        # image tokens to function correctly on grounding tasks"), so this pins images to
        # that value rather than trading away accuracy. Raise it if you have spare VRAM and
        # want finer detail on large images.
        values["image-max-tokens"] = "1024"

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

    # Multi-GPU: be explicit about the split strategy rather than relying on the
    # container's CLI baseline to supply it.
    if recommended:
        _rb = next((b for b in backends if b["name"] == recommended.name), None)
        if int((_rb or {}).get("gpu_count", 1)) > 1:
            values["split-mode"] = "layer"

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
    for essential in ("model", "ctx-size", "jinja", "cpu-moe", "n-cpu-moe",
                      "parallel", "cont-batching", "context-shift", "keep",
                      "batch-size", "ubatch-size", "cache-reuse", "reasoning",
                      "reasoning-preserve", "mmproj", "mmproj-offload", "image-max-tokens",
                      # tensor-split is a correctness setting under expert offload, not a
                      # tuning nicety: dropping it restores the even layer split that OOMs.
                      "tensor-split", "split-mode"):
        if essential in values and essential not in minimal:
            minimal[essential] = values[essential]

    # quirks
    quirks: list[str] = []

    # Baseline CONFLICTS — the container's own CLI args win over anything in models.ini,
    # so a preset value that disagrees with the compose command is silently discarded.
    # This bit us hard: `-np 1` in the compose command overrode `parallel = 2` in the ini,
    # so multi-session serving never actually ran, and `-ctv q8_0` overrode `cache-type-v`.
    # Surface it loudly instead of letting the preset look like it took effect.
    if recommended:
        _rb2 = next((b for b in backends if b["name"] == recommended.name), None)
        _base2 = (_rb2 or {}).get("baseline") or {}
        conflicts = [
            f"`{k}`: preset wants {v}, container forces {_base2[k]}"
            for k, v in values.items()
            if k in _base2 and str(_base2[k]).lower() != str(v).lower()
        ]
        if conflicts:
            quirks.append(
                "CONFLICT — the container's CLI args override models.ini, so these preset values "
                "will NOT take effect: " + "; ".join(conflicts) + ". "
                f"Fix by removing those flags from the `{recommended.name}` command in your compose file "
                "so per-model presets can control them. Keep only router-level args there "
                "(--models-dir, --models-preset, --host, --port, --models-max)."
            )

    if arch.startswith("gemma"):
        quirks.append("Gemma sliding-window attention: the ctx→KV math above assumes swa-full=false (default). "
                      "Enabling swa-full multiplies full-attn KV ~5× and will OOM.")
    # (MoE hint is emitted later, tailored to whichever offload the recommendation actually applies)
    if not summary.get("chat_template"):
        quirks.append("No embedded chat template — you'll need to set `chat-template` or `chat-template-file` "
                      "manually to get correct multi-turn formatting.")

    # Chat-template reasoning-capability hints (from template scanning)
    detected: list[str] = []
    if features.get("accepts_enable_thinking"):
        detected.append("`enable_thinking` kwarg (set true/false)")
    if features.get("accepts_reasoning_effort"):
        detected.append("`reasoning_effort` kwarg (low/medium/high)")
    if features.get("accepts_preserve_thinking"):
        detected.append("`preserve_thinking` kwarg (keep reasoning across turns)")
    if detected:
        quirks.append(
            "Chat template supports " + ", ".join(detected) + ". "
            "Thinking is enabled via the dedicated `reasoning = on` flag (setting enable_thinking through "
            "chat-template-kwargs is deprecated in current llama.cpp); anything without a dedicated flag, "
            "such as reasoning_effort, is pre-filled into `chat-template-kwargs`. Override either in the form."
        )
    if features.get("uses_think_tags") or features.get("uses_channel_thought"):
        quirks.append(
            "Model emits <think> or channel-based thought tags. Set `reasoning-format = deepseek` so OpenAI-compatible "
            "clients (OpenWebUI etc.) render thoughts as a collapsible instead of inline in the answer."
        )

    # Multi-session disclosure — makes the ctx-size vs per-session split visible
    if n_sessions > 1 and rec_ctx > 0:
        quirks.append(
            f"Sizing for {n_sessions} concurrent sessions: each user gets {_fmt_ctx(rec_ctx)} of context, "
            f"llama-server allocates ctx-size = {_fmt_ctx(rec_ctx * n_sessions)} total across the -np {n_sessions} slots. "
            f"KV cache is sized against the total; per-session throughput drops roughly linearly with load."
        )
        quirks.append(
            f"When a session hits its {_fmt_ctx(rec_ctx)} cap it will SLIDE: oldest tokens drop, "
            f"generation continues. Set `context-shift = on` and `keep = 256` (tune to your system-prompt "
            f"length in tokens so instructions survive the shift). To fail hard instead of forgetting old "
            f"turns, set `context-shift = off` in the form."
        )
        quirks.append(
            "TTFT tuning for multi-slot: set `batch-size = 4096` (safe, negligible VRAM). "
            "To also improve prompt-eval when a second session arrives mid-generation, bump `ubatch-size` "
            "in the form — this is the main lever, but it's costly. On layer-split multi-GPU, compute-buffer "
            "VRAM grows as ~8 × ubatch × layers × hidden. A dense 27B (64L, 5120H) at ubatch=2048 costs ~4.7 GB "
            "of compute buffers across cards. Don't bump ubatch above 512 unless you have 2+ GB of measured "
            "free VRAM after boot. MoE with CPU offload has much more room to work with."
        )

    # Multi-GPU overhead disclosure — user should know the math accounted for split-mode costs
    if recommended:
        rec_b = next((b for b in backends if b["name"] == recommended.name), None)
        rec_gpu_count = int((rec_b or {}).get("gpu_count", 1))
        if rec_gpu_count > 1:
            quirks.append(
                f"Multi-GPU backend ({rec_gpu_count} cards, layer-split): reserved "
                f"{_RESERVE_PER_GPU * rec_gpu_count:.1f} GB total for CUDA runtime "
                f"(0.5 GB × {rec_gpu_count}) and applied a {int((_MODEL_OVERHEAD_SPLIT - 1) * 100)}% model-VRAM "
                f"multiplier for cross-card handoffs. Real cap will be a bit lower than pure sum-of-VRAMs."
            )

    # Multimodal VRAM accounting
    if has_mmproj:
        quirks.append(
            f"Multimodal model (mmproj companion present — vision, audio, or other modality). "
            f"Reserved {mmproj_vram_gb:.2f} GB for the projector ({mmproj_gb:.2f} GB weights + "
            f"{_MMPROJ_COMPUTE_GB:g} GB encoder scratch). If several projector precisions ship in the "
            "directory the smallest is chosen, since it competes directly with the KV cache.\n"
            "TREAT THIS CTX AS OPTIMISTIC AND VERIFY IT LOADS. The projector and its encoder buffer are "
            "NOT layer-split — both land entirely on the main GPU — so the real limit is that one card, "
            "not the pooled total this estimate is based on. Worse, measured encoder scratch varies ~4x "
            "between models (0.55 GB on a 27B with a 0.87 GB projector vs 2.3 GB on Qwen3-VL-4B with a "
            "0.78 GB one) and is not derivable from GGUF metadata, so no single constant fits all. "
            "If it OOMs on device 0 while the other card still shows free VRAM, that is exactly this "
            "limitation — step ctx-size down until it loads."
        )

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
            _ts = values.get("tensor-split")
            if _ts:
                quirks.append(
                    f"Multi-GPU + expert offload: `tensor-split = {_ts}` balances the split by BYTES, not "
                    "layer count. Layers below n-cpu-moe keep only attention on the GPU while the rest carry "
                    "full experts, so an even split loads one card ~10x heavier than the other and OOMs it "
                    "while the other sits half empty. Don't remove it."
                )
        else:
            quirks.append(f"MoE model ({experts} experts): does not fit even with all experts offloaded to CPU. "
                          f"You need a bigger GPU, a smaller quant, or a shorter context.")

    # Work out which preset the SAVED section currently matches, by comparing the knobs
    # that presets actually set (total ctx plus the offload level). Used to badge the chip
    # that is really running, so a previewed chip can't be mistaken for the live config.
    _current_preset = ""
    if current_section and presets:
        _cur = {k: str(v) for k, v in current_section.items()}
        try:
            _cur_ctx = int(_cur.get("ctx-size") or 0)
        except ValueError:
            _cur_ctx = 0
        _cur_ngl = (_cur.get("ngl") or "").strip()
        _cur_ncm = (_cur.get("n-cpu-moe") or "").strip()
        for _p in presets:
            if _cur_ctx != _p.ctx * n_sessions:
                continue
            if _p.offload_kind == "ngl":
                ok = _cur_ngl == str(_p.ngl)
            elif _p.offload_kind == "n-cpu-moe":
                ok = _cur_ncm == str(_p.n_cpu_moe)
            elif _p.offload_kind == "cpu-moe":
                ok = (_cur.get("cpu-moe") or "").lower() in ("true", "on", "1")
            else:
                ok = _cur_ngl in ("", "999") and not _cur_ncm
            if ok:
                _current_preset = _p.key
                break

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
        recommended_total_ctx=rec_ctx * n_sessions if rec_ctx > 0 else 0,
        n_sessions=n_sessions,
        values=values,
        values_minimal=minimal,
        baseline_redundant=baseline_redundant,
        quirks=quirks,
        unavailable=unavailable,
        current_diff=current_diff,
        presets=presets,
        frontier=frontier_opts,
        fits_full_gpu=_fits_full_gpu,
        native_ctx=native_ctx,
        current_preset=_current_preset,
        has_unsaved=bool(current_diff),
        active_preset=active_preset,
    )


def format_ctx(n: int) -> str:
    return _fmt_ctx(n)
