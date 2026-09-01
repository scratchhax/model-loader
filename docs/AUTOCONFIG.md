# Autoconfig

Autoconfig picks `models.ini` values that fit your live hardware without OOMing llama-server. It reads the GGUF's metadata, probes each backend's VRAM in real time, and tries candidate context sizes from largest to smallest until one fits.

## The VRAM budget

For a given backend with `V` GB of VRAM and `N` GPUs:

```
budget       = V - (RESERVE_PER_GPU × N)     # CUDA runtime + driver context + slack
model_gb     = file_size_gb × overhead_mul   # activation buffers + duplicated allocations
kv_budget    = budget - model_gb
```

Constants (`app/autoconfig.py`):

| Constant | Value | Why |
|---|---|---|
| `_RESERVE_PER_GPU` | `0.5` GB | CUDA runtime + driver context. Scales with GPU count. |
| `_MODEL_OVERHEAD_SINGLE` | `1.00` | Q_K_M loads at ~file size when everything's on one card. |
| `_MODEL_OVERHEAD_SPLIT` | `1.08` | +8% for cross-GPU handoffs, duplicated activation buffers, layer-split imbalance. |
| `_CACHE_DEFAULT` | `q8_0` | 2× smaller KV cache than fp16 with negligible quality loss. |

`_MODEL_OVERHEAD_SPLIT` is calibrated against real measurements on a Qwen 3.8-27B (17 GB Q4_K_M) running on 2× RTX 5070 (24 GB pooled). See "Calibration" below.

## KV cache math

For each candidate ctx size, autoconfig computes the KV cache footprint from GGUF metadata:

```
Standard attention:
  bytes = 2 (K+V) × n_layers × ctx × n_kv_heads × head_dim × cache_bytes

Sliding-window attention (Gemma):
  Full-attention layers pay the full-ctx cost. SWA layers cap at window_size.
  bytes = 2 × Σ_layer (ctx_for_layer × n_kv_heads × head_dim × cache_bytes)

Hybrid attention + SSM (Qwen 3.5/3.6):
  Attention layers use the standard formula.
  SSM layers pay a fixed ~4 MB per layer regardless of ctx.
```

`cache_bytes` per element:
- `f16`: 2 bytes
- `q8_0`: ~1.06 bytes (block-quantized, includes scale overhead)
- `q4_0`: ~0.56 bytes

## Candidate context sizes

The picker tries values from this list (largest first) and returns the first that fits:

```
4096, 8192, 12288, 16384, 24576, 32768, 40960, 49152, 57344, 65536,
73728, 81920, 90112, 98304, 106496, 114688, 122880, 131072, 139264,
147456, 151552, 155648, 159744, 163840, 172032, 180224, 188416,
196608, 212992, 229376, 245760, 262144,
327680, 393216, 458752, 524288, 655360, 786432, 917504, 1048576
```

All values are exact multiples of 1024 (K), chosen to land on llama.cpp's internal block-alignment boundaries. Arbitrary values (e.g. `160000`) get rounded up to the next block boundary internally and may trigger allocation of a larger compute buffer than the ctx alone would suggest.

Values above the model's native ctx (from GGUF `context_length`) are capped at `2 × native_ctx` — llama.cpp's RoPE linear extension is reliable up to ~2× but degrades quickly past that.

## Calibration (Qwen 3.8-27B, 2× RTX 5070, 24 GB pooled)

The 8% split-overhead came from measuring real VRAM usage across candidate ctx sizes:

| ctx | KV formula | Real total VRAM | Real overhead vs formula |
|---|---|---|---|
| 147456 | 5.27 GB | 21.9 GB | model_actual = 16.63 → 1.044× |
| 159744 | 5.70 GB | 22.4 GB | model_actual = 16.70 → 1.048× |
| 163840 | 5.83 GB | OOM | — |
| ~160000 (manual) | ~5.71 GB | OOM | rounded up past a bucket boundary |

Key finding: activation-buffer allocation isn't smooth — it steps in buckets. Between 159744 and 163840 there's a boundary where llama.cpp allocates a larger scratch buffer for attention. The formula predicts they should fit within 1 GB of each other; reality is a ~1.5 GB step.

Autoconfig's 1.08 multiplier is intentionally conservative (over-predicts by ~0.5 GB) so it lands safely below these bucket boundaries. For Qwen 3.8-27B on 2× 5070 this gives 159744 as the pick — real usage 22.4 GB, 1.5 GB free.

## MoE offload strategy

For MoE models (`expert_count > 1`), autoconfig offers three presets on the Config page:

- **Fast** — all experts on GPU, no ctx sacrifice; may need to reduce ctx to fit.
- **Balanced** — half experts on GPU, half offloaded; medium ctx, medium speed.
- **Long-ctx** — all experts on CPU (`-ot exps=CPU`), only attention on GPU; maximum ctx, ~3× slower generation.

The KV cache always stays on GPU regardless of preset.

## Reasoning capability detection

Autoconfig scans the model's chat template (Jinja source in GGUF metadata) for these markers and enables the matching ini fields:

- `{% if enable_thinking %}` → adds `chat-template-kwargs.enable_thinking`
- `reasoning_effort` variable → adds `reasoning-effort` field with values low/medium/high
- `preserve_thinking` → adds `preserve-thinking` toggle

If none are present, the Reasoning tier is hidden.

## What Autoconfig does NOT touch

If you've manually set any of these ini keys, Autoconfig preserves them:
- Any key not in its opinion list (custom sampler, custom quant, exotic flags)
- User-set `n-threads`, `n-threads-batch`, `n-parallel`
- User-set `alias`, `system-prompt`, `chat-template`

Autoconfig only fills the tier of fields it knows about. Everything else you wrote stays exactly as written.

## Overriding Autoconfig

The Config page's Autoconfig panel is a *suggestion*. You edit the form afterwards and save. There is no "apply" step that clobbers your manual edits — the recommendation just pre-fills the form.

To push ctx higher than the recommendation:
1. Pick the value from the Candidate context sizes list above.
2. Save.
3. Restart the container.
4. Watch VRAM usage. If it fits with headroom, great. If it OOMs, drop to the next lower candidate.

Don't type arbitrary values (like `160000`) — llama.cpp rounds them internally and you'll be guessing at the bucket boundary.
