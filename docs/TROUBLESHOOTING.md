# Troubleshooting

## Container OOMs at the autoconfig-recommended ctx

Autoconfig's default overhead is 8% for multi-GPU. If a real load exceeds this (some fine-tunes, some quant methods, unusual chat templates with huge system prompts), drop to the next lower ctx candidate in the [candidate list](AUTOCONFIG.md#candidate-context-sizes). Don't type arbitrary values — stick to the list, they're aligned to llama.cpp's internal buckets.

If OOMs are systematic across models, edit `app/autoconfig.py` and bump `_MODEL_OVERHEAD_SPLIT` (multi-GPU) or `_MODEL_OVERHEAD_SINGLE` (single-GPU) up by 0.02 and rebuild.

## Dashboard shows no backends

- Confirm your llama container is running: `docker ps | grep llama`.
- Confirm its image tag starts with `ghcr.io/ggml-org/llama.cpp:` — auto-discovery matches on that prefix.
- If your image is a fork or private registry, set the env var: `LLAMA_CONTAINERS=my-llama-1,my-llama-2` on the model-loader service.
- Check model-loader can reach the docker socket: `docker exec model-loader ls -l /var/run/docker.sock` should show the mounted socket.

## GPU stats show "warming up..." forever

The sampler runs every 2 seconds. If it never populates:

- `docker exec <llama-container> nvidia-smi` — must succeed. If it doesn't, your llama container doesn't have GPU access; fix that in your compose (nvidia runtime, `deploy.resources.reservations.devices`).
- For AMD, `rocm-smi` must succeed inside the container.

## VRAM total shows 0 GB (AMD)

Some ROCm stacks report memory allocated as a percentage but don't expose total VRAM through `rocm-smi --json`. Set it explicitly: `GPU_VRAM=llama-7900xt:20`.

## OpenWebUI doesn't see a new backend

OpenWebUI uses `PersistentConfig` — after first startup it ignores env vars and reads from `webui.db`. Editing `OPENAI_API_BASE_URLS` in your compose file changes nothing.

Fix: on the Containers page, look for the amber OpenWebUI card. Click **Reconcile**. This writes directly to `webui.db` via `docker exec`, restarts open-webui, and preserves any manual URL prefixes you'd set (like `NV`, `AMD`, `CPU`).

If the card doesn't appear at all, model-loader can't find your open-webui container. Check the name — it searches for containers matching `open-webui` or `openwebui`.

## Downloads stall / restart from 0%

- Parallel chunks (default 8) request byte ranges. HuggingFace's CDN sometimes rate-limits; the downloader will retry a chunk up to 3 times before failing that job.
- If a job is stuck, cancel it on `/downloads` and re-queue from `/search`. State survives model-loader restarts.
- For gated repos, ensure your HF token is set in `/settings` (stored in `/data/model_loader.db`).

## "500 Internal Server Error" opening a model detail page

Usually means the GGUF has metadata fields Model Loader hasn't seen before. The parser is defensive but not exhaustive. Check `docker logs model-loader` for the stack trace and file an issue with the model repo id — most fixes are a one-line addition to `app/gguf_meta.py`.

## Autoconfig recommends `ctx = 4096` for a model with 128K native

The picker fell all the way to the smallest candidate, meaning the model + fp16 KV cache + reserve exceeds your total VRAM. Options:

1. Use a smaller quant of the same model (Q4_K_M instead of Q6_K).
2. Add a second GPU and use `-sm layer`.
3. Accept partial offload. Autoconfig does this for you now: dense models move whole layers off the GPU via `ngl`, MoE models offload expert weights via `n-cpu-moe`. The presets show the speed cost of each point.

## A setting saves but has no effect

Check your llama.cpp container's `command:` in docker-compose. **Anything on the command line overrides the preset file, silently.** A stray `--ctx-size`, `-np` or `--n-gpu-layers` in compose beats whatever Model Loader writes into `models.ini`, and nothing surfaces the conflict — you just get behaviour that doesn't match the config.

Keep `command:` to `--models-preset`, `--host`, `--port` and `--models-max`. Everything per-model belongs in the ini.

## Multi-GPU OOM even though pooled VRAM looks fine

llama.cpp places each layer on exactly one card, so a config can fit the pool and still blow a single device. This bites hardest on MoE models with expert offload: layers below `n-cpu-moe` keep only attention on the GPU while the rest carry full experts, and `split-mode = layer` divides by layer *count*, so one card gets nearly all the expensive layers.

Autoconfig now emits a byte-balanced `tensor-split` and checks each card individually. If you hand-edited the section, either re-run Autoconfig or set `tensor-split` yourself. The symptom is distinctive: the log shows a failed allocation on one device while `nvidia-smi` shows the other with gigabytes free.

## A model shows in OpenWebUI but fails with "model not found"

OpenWebUI renders its model whitelist rather than intersecting it with what the backend actually serves, so an id left behind by a deleted or renamed model keeps appearing in the picker.

Containers page → the connection shows a **dead model id** warning → click to remove. Rename and delete do this automatically; the warning is for ids that drifted some other way.

## "image input is not supported" when uploading a picture

Either the model has no `mmproj` (check its ini section), or you selected a different model than you meant to — similarly-named text-only and multimodal variants are easy to confuse.

Containers page → **Align capabilities** writes each model's vision flag into OpenWebUI from its ini section, so the upload control only appears where it can work. Models page shows the same per model.

## Autoconfig says "doesn't fit at any context" on a Vulkan backend

The Vulkan image ships neither `nvidia-smi` nor `rocm-smi`, so VRAM can't be probed and the budget computes as zero. Declare it: `GPU_VRAM=<container>:<GB>` on the model-loader service, then restart it.

Current versions say this explicitly rather than reporting a fit failure. If you see the fit-failure wording instead, you're on an older build.

## Renamed a model and now the section says "No GGUF found"

Fixed in current versions — sections resolve to their file via the `model =` path rather than by matching the section name to a filename. If you hit this, you're on an older build; setting `model =` explicitly in the section is the workaround.

## `docker exec model-loader python3 -m app.migrate_layout` fails

Preconditions:
- `/models` inside the container must contain the flat `.gguf` files.
- The user running the container needs write permission on the models directory (usually root, since docker containers run as root by default).

The migration is atomic per-model — a partial failure leaves the already-migrated models done and the rest untouched. Safe to re-run.

## Log-tail freezes / jumps around

The Containers page polls logs every 2 seconds. If you're mid-scroll and the polling replaces the DOM, the browser scroll position resets. The grep filter box (`filter` icon) reduces the update volume enough that this usually stops being annoying. If it doesn't, `docker logs -f <container>` from a terminal is more reliable for long tail sessions.

## Rebuilding the image after code changes

```bash
cd <compose-dir>
docker compose up -d --build model-loader
```

The Dockerfile installs pip deps in a separate layer, so rebuilds after Python edits are ~2-3 seconds.

## Model doesn't appear in `/models` after download

- Check the download finished — the row on `/downloads` should show 100%.
- Check the file landed in the right subdir: `docker exec model-loader ls -la /models/<stem>/`.
- The models page uses stem-directory scanning; if the file is in the flat root (`/models/foo.gguf` instead of `/models/foo/foo.gguf`), run the migration.

## Reset everything

Nuclear option — deletes app state but preserves your GGUFs and `models.ini`:

```bash
docker compose stop model-loader
rm -rf ./model_loader_data
docker compose up -d model-loader
```

You'll need to re-enter your HF token and re-save any prompts, but download history and everything else rebuilds itself.
