# Autoconfig

Autoconfig picks `models.ini` values that fit your live hardware without OOMing llama-server. It reads the GGUF's metadata, probes each backend's VRAM in real time, and searches candidate context sizes for ones that fit.

Everything below is a *model* of llama.cpp's allocation behaviour, calibrated against measurements on real hardware. It is deliberately conservative. It is not a guarantee, and the doc says where it is least trustworthy.

## The VRAM budget

For a backend with `V` GB of pooled VRAM across `N` GPUs:

```
budget    = V - (RESERVE_PER_GPU × N) - projector_vram
model_gb  = file_size_gb × overhead_mul
kv_budget = budget - model_gb
```

Constants (`app/autoconfig.py`):

| Constant | Value | Why |
|---|---|---|
| `_RESERVE_PER_GPU` | `1.0` GB | CUDA runtime, driver context, scratch and cuBLAS workspace. Scales with GPU count. Raised from 0.5 for llama.cpp 0.3.0-dev, which allocates more per device. |
| `_MODEL_OVERHEAD_SINGLE` | `1.00` | Q_K_M loads at roughly file size when everything is on one card. |
| `_MODEL_OVERHEAD_SPLIT` | `1.08` | +8% for cross-GPU handoffs and duplicated activation buffers. |
| `_CACHE_DEFAULT` | `q8_0` | Half the KV cache of fp16 with negligible quality loss. |
| `_CPU_LAYER_PENALTY` | `20.0` | How much slower one CPU-resident **dense** layer is than a GPU one. Drives the speed estimate only. |
| `_MMPROJ_VRAM_MULT` | `1.0` | A multimodal projector occupies about its file size in VRAM. |
| `_MMPROJ_COMPUTE_GB` | `0.5` | Encoder scratch beyond the projector weights. |

## Pooled VRAM is not enough: the per-card check

This is the part most worth understanding, because getting it wrong produces an OOM that the numbers say shouldn't happen.

**llama.cpp places each layer on exactly one card.** So a configuration can fit the pooled budget comfortably and still fail, because one device has to hold more than it owns. Autoconfig therefore re-checks every candidate against each card individually, using the same split it will actually emit, and rejects any candidate whose heaviest card exceeds that card's capacity.

Per-card capacities come from the live probe. When they are unavailable — a Vulkan backend with no SMI tool, for instance — it falls back to dividing the pool evenly, which is correct only for identical cards.

## Balancing a multi-GPU split by bytes

`split-mode = layer` divides layers **by count**, weighted by `--tensor-split`. That is right for a dense model, where every layer costs the same. It is badly wrong for a MoE under `--n-cpu-moe N`:

- layers `0 .. N-1` keep only attention on the GPU — cheap
- layers `N .. L-1` carry full experts — roughly an order of magnitude more

An even split hands one card nearly all the expensive layers. Measured on a 40-layer, 256-expert 35B at `n-cpu-moe=24`: llama.cpp tried to allocate **15.0 GiB on device 1 of an 11.9 GiB card** while device 0 still had ~9 GiB free.

Autoconfig computes an explicit `tensor-split` that equalises **bytes**, not layer count:

- each layer is costed as attention + its share of KV, plus experts if it is above the `n-cpu-moe` threshold
- the projector and compute buffer are charged to device 0, since they are not layer-split
- the partition minimises the heaviest card **relative to its own capacity**, so a 24 GB card beside a 12 GB one receives proportionally more rather than an equal share
- no `tensor-split` is emitted at all for dense models or single-GPU backends, where an even split is already correct

Balancing that same model by bytes loaded it at 9.7 / 10.8 GiB across the two cards with the full 256K context.

## KV cache math

For each candidate ctx size, the KV footprint is computed from GGUF metadata:

```
Standard attention:
  bytes = 2 (K+V) × n_layers × ctx × n_kv_heads × head_dim × cache_bytes

Sliding-window attention (Gemma):
  Full-attention (global) layers pay the full-ctx cost; SWA (local) layers cap at
  window_size. Every per-layer quantity is read off the SAME repeating period:
  which layers are local, how many KV heads each has, and the head dim it uses.
  bytes = 2 × Σ_layer (ctx_for_layer × n_kv_heads_for_layer × head_dim_for_layer × cache_bytes)

Hybrid attention + SSM (Qwen 3.5/3.6):
  Attention layers use the standard formula.
  SSM layers pay a fixed ~4 MB per layer regardless of ctx.
```

`cache_bytes` per element: `f16` 2 bytes, `q8_0` ~1.06 (block-quantised, includes scale overhead), `q4_0` ~0.56.

> **Gemma declares its KV head count per layer, and the layers are not alike.** On gemma-4-12b the array is `[8,8,8,8,8,1,…]`, aligned with the sliding-window pattern: the *global* layer carries **1** KV head against 8 on the local ones (on 26B-A4B it is 2 against 8). Collapsing that array to a single representative value charges every global layer 8× too much — and the global layers are the only ones whose cost scales with context, so that term dominates the whole estimate. It predicted **14.51 GB** at 208K against a real **1.97 GB**.
>
> `shared_kv_layers` matters too: those layers reuse another layer's cache and allocate none of their own. They are spread through the stack, so the saving applies to local and global layers alike — charging it all to the local layers (the cheap ones) left gemma-4-E4B 70% high.
>
> Both are now read from the declared per-layer arrays, and the result is validated against llama.cpp's own estimator (`llama fit-params`, which reports the memory the allocator will actually request):
>
> | | ours | llama.cpp | delta |
> |---|---|---|---|
> | gemma-4-12b @262K | 2.29 GB | 2.37 GB | −3% |
> | gemma-4-26B-A4B @262K | 2.76 GB | 2.81 GB | −2% |
> | gemma-4-E4B @131K | 1.07 GB | 1.09 GB | −2% |
>
> If you change this math, check it against `llama fit-params` before trusting it. It is in the same container image and takes about three seconds per model.

> **A warning about `cache-type-v`.** Dropping V below `q8_0` saves VRAM and can be catastrophically slow. There is no CUDA flash-attention kernel for a `q4_0` V cache at some head dimensions, so attention silently falls back to the CPU. Measured on a 27B with a 256-wide head dim: prompt eval went from **1737 tok/s to 115 tok/s**, a 15× regression. It is invisible on short prompts. If you change this, benchmark a long one and watch GPU utilisation.

## Concurrent sessions

llama-server divides its context across `--parallel` slots, so `ctx-size` is the **total** and each slot gets `ctx-size / parallel`. The sessions picker (1–8) sets both: pick a per-session context, and Autoconfig writes `ctx-size = per_session × sessions` along with `parallel`.

Each extra session costs real GPU layers, because the KV cache grows with the total. Requests beyond the slot count **queue** rather than fail, so under-provisioning costs latency during bursts while over-provisioning costs speed on every token.

## Candidate context sizes

52 candidates from 4096 to 1048576, all exact multiples of **4096**.

Values are chosen to land on llama.cpp's internal block-alignment boundaries. Arbitrary values (`160000`, say) are rounded up internally and may allocate a larger compute buffer than the context alone suggests.

Values above the model's native context (GGUF `context_length`) are capped at `2 × native_ctx` — RoPE linear extension is reliable to about 2× and degrades quickly past that.

## Offload strategy

**Dense models** buy context by moving whole layers off the GPU with `ngl`. Those layers live in system RAM and every token traverses them on the CPU, so the speed cost is steep — this is what `_CPU_LAYER_PENALTY` models.

**MoE models** offload expert weights instead, with `--n-cpu-moe N` (first N layers' experts on CPU) or `--cpu-moe` (all of them). This is far cheaper per byte than dense offload, because only the active experts are read per token.

The Config page offers four presets, always:

- **Fast** — most GPU-resident, least context
- **Balanced** — middle of the frontier
- **Long context** — maximum context, most offload
- **Custom** — a slider over every point on the fit frontier

When a model fits entirely on the GPU at full context, the presets collapse to that single answer rather than inventing tradeoffs that don't exist. The KV cache always stays on the GPU.

**The speed percentages are an ordering hint, not a benchmark.** They come from `_CPU_LAYER_PENALTY`, calibrated on one dense model. They will reliably tell you Fast beats Long context; they will not tell you your tokens per second, and for MoE models they over-estimate the cost of offload. Benchmark before trusting a number.

## Multimodal projectors

A model is multimodal if its section declares `mmproj`. The projector occupies roughly its file size in VRAM plus ~0.5 GB of encoder scratch, and — importantly — it is **pinned to the main GPU**, not layer-split, so it is charged to device 0 in the per-card check.

A projector is not necessarily a *vision* projector: llama.cpp uses the same `--mmproj` slot for audio encoders. Autoconfig reads the projector's own metadata (`clip.vision.*` versus `clip.audio.*`) to tell them apart.

Vision models also get `image-max-tokens`, because dynamic-resolution models size their decode buffer from the image, and that allocation happens at inference time — long after the fit math approved the load.

## Reasoning capability detection

Autoconfig scans the chat template in the GGUF metadata and sets:

- `reasoning = on` when the template accepts `enable_thinking`. Note this uses the **dedicated flag**; setting `enable_thinking` through `chat-template-kwargs` is deprecated in current llama.cpp.
- `chat-template-kwargs = {"reasoning_effort": "medium"}` when the template accepts a reasoning effort.
- `reasoning-format = deepseek` when the template emits think tags, so OpenAI-compatible clients put thoughts in `reasoning_content` instead of inline in the answer.
- `reasoning-preserve = on` when the template supports carrying reasoning across turns.

## Calibration (Qwen 3.8-27B, 2× RTX 5070, 24 GB pooled)

The split overhead came from measuring real VRAM across candidate context sizes:

| ctx | KV formula | Real total VRAM | Result |
|---|---|---|---|
| 147456 | 5.27 GB | 21.9 GB | fits |
| 159744 | 5.70 GB | 22.4 GB | fits, 1.5 GB free |
| 163840 | 5.83 GB | — | OOM |
| ~160000 (manual) | ~5.71 GB | — | OOM, rounded past a bucket boundary |

Key finding: **activation-buffer allocation steps in buckets, it isn't smooth.** Between 159744 and 163840 llama.cpp allocates a larger attention scratch buffer. The formula predicts those two should sit within 1 GB of each other; reality is a ~1.5 GB step. The 1.08 multiplier is deliberately conservative so picks land below such boundaries.

Two further measurements worth recording:

- `_CPU_LAYER_PENALTY = 20.0` verified on the same model: `ngl=999` gave 34.2 tok/s, `ngl=56` gave 9.7 tok/s. Predicted 28% of full speed, measured 28.4%.
- `-ub 2048` was tried and reverted. It cost **4.7 GB**, not the few hundred megabytes expected, and OOM'd a configuration that otherwise fit.

## What Autoconfig does not touch

Autoconfig fills the fields it has an opinion about and leaves everything else exactly as written — custom samplers, exotic flags, your `n-threads`, your `chat-template`.

The panel is a **suggestion**. It pre-fills the form; nothing reaches `models.ini` until you click Save, and a banner tells you when what you are previewing differs from what is currently running.

## Overriding it

To push context past the recommendation: pick a value from the candidate list, save, restart the container, and watch VRAM. If it fits with headroom, good; if it OOMs, drop to the next candidate down.

Avoid arbitrary values — llama.cpp rounds them internally and you end up guessing at a bucket boundary.
