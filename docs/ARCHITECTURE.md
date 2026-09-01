# Architecture

## Stack

- **FastAPI** — HTTP routes, no async DB (sqlite is sync + fast enough).
- **Jinja2** — server-rendered templates.
- **HTMX** — partial swaps for polling, form submits without page reload.
- **Alpine.js** — small client-side interactions (tab switches, modals, clipboard).
- **Tailwind CSS via CDN** — no build step. `theme.css` is a thin extension for one-off colors.
- **docker SDK for Python** — container discovery, exec, stats.
- **sqlite** — app-level state (HF token, prompts, download history, avatar cache).

Everything ships as one Docker image. Deps in `requirements.txt` are pinned. Python 3.12 slim base.

## File layout

```
app/
  main.py             FastAPI routes — one file, ~50 endpoints. Order matches nav.
  services.py         docker SDK wrapper, models-dir helpers, OpenWebUI reconciler.
  autoconfig.py       KV cache math, preset picker, VRAM fit, MoE offload strategy.
  hw.py               Background sampler thread, per-container stats cache.
  hf.py               HF Hub API client (search, tree, avatar cache).
  gguf_meta.py        Hand-rolled GGUF v3 metadata reader (no numpy).
  ini.py              models.ini schema (98 fields, 10 tiers) + parser + writer.
  downloader.py       Parallel-range download engine with sqlite-backed job history.
  db.py               Sqlite prefs (schema in module docstring).
  migrate_layout.py   One-shot: flat GGUF layout -> per-model-subdir + absolute paths.
  config.py           pydantic-settings for env vars.
  utils.py            Small helpers (size formatting, stem parsing, etc.).
  templates/          Jinja2 templates. Underscore prefix = partial (HTMX fragment).
```

## Data flow

### On startup

1. `app/main.py` calls `hw.start_sampler()` — spawns a daemon thread that polls each discovered llama container every 2 seconds and caches stats.
2. Docker client connects to `/var/run/docker.sock`.
3. FastAPI serves. Templates render from cache-warm stats.

### On page render

- Every page extends `base.html`, which includes the nav, palette (for Cmd-K), and Tailwind CDN link.
- Long-polling pages (`/containers`, `/downloads`) use `hx-get` with `hx-trigger="every 2s"` to swap in fresh partials without a full reload.
- Cards remember DOM identity so animations don't restart on each poll.

### On download

1. `POST /download` inserts a row into `downloads` sqlite table with status=`queued`.
2. A background worker (started at first request) picks it up, splits the file into 8 byte-range chunks, opens 8 concurrent HTTPS streams.
3. Each chunk writes to a `.part-N` file. Progress + speed sample every 500ms into an in-memory ring buffer for the sparkline.
4. On completion, chunks are concatenated in order, temp files removed, `status=done`.
5. If a companion `*mmproj*.gguf` exists in the same repo, it's auto-queued into the same subdir.

### On config save

1. Form POST hits `POST /config/section/{name}`.
2. `ini.py` reads current `models.ini`, updates the named section, writes atomically via `os.replace()`.
3. Before overwriting, current file is copied to `models.ini.bak-<timestamp>`. Only the 10 newest backups are kept.

### On OpenWebUI reconcile

1. `POST /containers/sync-openwebui` collects: (a) current llama backends from docker socket, (b) contents of `openai.api_base_urls` from `webui.db` via `docker exec open-webui python3 -c "..."`.
2. Computes diff: missing = backends not in webui, stale = webui URLs pointing at containers that no longer exist.
3. Writes the reconciled list back to `webui.db` (still via docker exec + inline python), preserving user-set URL prefixes.
4. Restarts `open-webui` container to force it to re-read the config.

Why direct DB write? OpenWebUI uses `PersistentConfig` — env vars are only seed values on a virgin DB, subsequent reads come from `webui.db`. Editing compose env vars post-first-boot changes nothing.

The same `docker exec` channel drives three more OpenWebUI features:

- **Per-connection model whitelists** (`openai.api_configs[i].model_ids`), so a CPU backend can offer only the models it can actually run.
- **Dead-id detection.** OpenWebUI *renders* that whitelist rather than intersecting it with what the backend reports, so an id left behind by a deleted or renamed model still appears in the picker and fails with "model not found" only when someone selects it. Model Loader flags those and prunes them automatically after a rename or delete.
- **Vision capability** (`model.meta.capabilities.vision`). Unlike the config table, `model` is ordinary application data read per request, so this needs no restart. The flag is derived from the projector's own metadata rather than the mere presence of an `mmproj`, because llama.cpp uses that same slot for audio encoders.

## Design choices worth calling out

### One directory per model

`/models/<stem>/<file>.gguf` instead of flat `/models/<file>.gguf`. Reason: many models ship a matching `mmproj-*.gguf` (vision projector). In a flat layout, mmproj files from two different models will collide by filename. With per-stem subdirs, they can't.

`app/migrate_layout.py` does the one-shot conversion and rewrites `models.ini` with absolute paths.

### Explicit absolute `model = /models/<stem>/<file>.gguf` in every ini section

`llama-server --models-dir /models --models-preset foo.ini` does *not* join the `model` value with `--models-dir`. It takes the value literally. If you write `model = foo.gguf`, llama-server looks in `$PWD`, not `/models`. Autoconfig always writes absolute paths to avoid this footgun.

### Hand-rolled GGUF metadata parser (no numpy)

`gguf_meta.py` reads the GGUF v3 header + metadata KV pairs manually. Weight tensors are skipped. This avoids pulling numpy (~50 MB) into the image just to parse a few dozen integers. About 200 lines.

### Sampler runs in a thread, not asyncio

Docker SDK is sync. Running `nvidia-smi` via `container.exec_run()` is sync. Wrapping this in an async loop with `run_in_executor` bought us nothing — the thread does the same work with less ceremony. 2-second sample interval is plenty for a homelab dashboard.

### PersistentConfig-aware OpenWebUI sync

See "On OpenWebUI reconcile" above. Writing directly to `webui.db` is the only way that actually works.

### Autoconfig is a suggestion, not an action

The Config page shows Autoconfig's recommendation in a panel, but doesn't apply it — you manually save the form. This means you can pull up Autoconfig for reference without risking overwriting hand-tuned values.

## What's intentionally not here

- **No auth.** Personal-use tool. Reverse-proxy it if you need auth.
- **No queue backend.** In-process asyncio queue for downloads. Rebooting model-loader mid-download means resuming from the last chunk (which is fine since chunks are byte-range).
- **No metrics export.** Prometheus/Grafana are out of scope. The dashboard is the "monitoring UI."
- **No user prefs beyond a global HF token.** Single-user tool.
- **No frontend build.** Tailwind from CDN. No PostCSS, no Vite, no npm at all.
- **No tests.** Correctness is enforced by shipping to production of one (the developer) and fixing what breaks. Do not adopt this policy on a team.
