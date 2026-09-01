# Model Loader

Personal homelab web app for managing llama.cpp GGUF models and containers. Runs alongside your `llama.cpp` server containers and gives you a browser UI for:

- Searching and downloading GGUFs from Hugging Face (parallel-range downloader, live per-chunk speed, sparklines)
- Managing the `models.ini` preset file (91-field form, tooltips, atomic writes with backups)
- **Autoconfig** — analyzes a GGUF and your live GPU state, then recommends context/offload/RoPE settings that actually fit
- Per-backend hardware stats (GPU util, VRAM, temp, power; container CPU/RSS)
- Live container status + restart, log tail with filter
- Auto-discovery of any `ghcr.io/ggml-org/llama.cpp:*` container on your docker socket
- Adding newly-discovered backends to OpenWebUI's `openai.api_base_urls` (writes directly to its sqlite so `PersistentConfig` accepts them)

**Personal-use, LAN-only. No auth. Mounts `docker.sock` — treat as root-equivalent on the host.**

## New-box install

Prereqs: `docker` + `docker compose` plugin, an existing `docker-compose.yaml` for your llama containers (or use the snippets in the app's `/containers` page), and a `./models` directory that's bind-mounted into your llama containers.

```bash
# clone this repo somewhere near your compose file
git clone <your-remote> ~/ai-lab/model_loader

# add the service to your existing docker-compose.yaml
# (see docker-compose.snippet.yaml in this repo for the block to paste)

# from your compose dir
docker compose up -d --build model-loader

# app lives at http://<host>:8090
```

`bootstrap.sh` in this repo automates the compose-block insertion and first build. From your compose dir:
```bash
bash /path/to/model_loader/bootstrap.sh
```

## Configuration (env vars)

All optional. Defaults in `app/config.py`.

| Var | Default | Purpose |
|---|---|---|
| `MODELS_DIR` | `/models` | Where GGUFs live inside the container |
| `MODELS_INI_PATH` | `/models/models.ini` | Path to the llama-server preset file |
| `DATA_DIR` | `/data` | Sqlite state (HF token, prompts, avatar cache, download history) |
| `LLAMA_CONTAINERS` | (empty) | Optional whitelist. Empty = auto-discover any `ghcr.io/ggml-org/llama.cpp:*` container |
| `GPU_VRAM` | (empty) | Optional per-container VRAM overrides, e.g. `llama-7900xt:20,llama-5070:12`. Auto-probes via `nvidia-smi` / `rocm-smi` if unset. |
| `MAX_CONCURRENT_DOWNLOADS` | `2` | Parallel job cap |
| `BIND_PORT` | `8090` | HTTP port |

## Backups

Everything you care about lives in three places:

1. **This git repo** — source code
2. **`/data/model_loader.db`** on host (mount source) — HF token, saved prompts, download history, avatar cache
3. **`/models/models.ini`** — llama-server preset file (has 10 rolling backups automatically)

The `backup.sh` script tars all three plus your OpenWebUI data into `./backups/<timestamp>.tgz`. Run it from cron or manually.

Models themselves (the GGUFs) are big and separate — `rsync` those with your normal backup strategy.

## Notable design choices

- **One directory per model** (`/models/<stem>/<file>.gguf`) — prevents mmproj filename collisions across models. See `app/migrate_layout.py` for the one-shot script that reorganizes a flat models dir.
- **Explicit `model = /models/<stem>/<file>.gguf`** in every ini section — llama-server takes the value literally, not relative to `--models-dir`.
- **PersistentConfig-aware OpenWebUI sync** — writes to `webui.db` directly (not env vars) because OpenWebUI only reads env once and persists the rest.
- **Every download in a subdir; mmproj auto-fetched** — click Download on a main GGUF and any `*mmproj*.gguf` in the same repo comes along, into the same subdir.

## Files

```
app/
  main.py             FastAPI routes
  services.py         Docker + models-dir + OpenWebUI helpers
  autoconfig.py       KV cache math + preset picker + VRAM fit
  hw.py               Background sampler, nvidia-smi / rocm-smi probes
  hf.py               HF Hub API client (search, tree, mmproj auto-detect)
  gguf_meta.py        Hand-rolled GGUF v3 metadata reader
  ini.py              Model preset ini schema, parser, atomic writes
  downloader.py       Parallel-range download engine, sqlite job history
  db.py               Sqlite prefs
  migrate_layout.py   One-shot: flat -> per-model-subdir layout
  templates/          Jinja2 templates (HTMX + Alpine + Tailwind CDN)
Dockerfile
requirements.txt
docker-compose.snippet.yaml   Service block to paste into your compose
bootstrap.sh                  Fresh-host installer
backup.sh                     Data backup helper
```

## License

Personal use. No warranty. Don't expose to the public internet — no auth, root-equivalent socket access.
