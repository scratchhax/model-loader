# Strata coexistence plan

The design and build order for letting Strata, llama.cpp and declared GPU tenants
(chatterbox, comfyui) share this box instead of taking turns on it. Work happens on
`feat/strata-coexistence`; `main` stays deployable and gets no commits until the feature
is verified on the live box. Nothing is pushed until it is said so out loud.

## Why the current integration stops short

Strata today is a well-behaved *guest*: label discovery, a card pin, its own log reader,
and a winner-take-all Engine row. But the switch (`services.activate_engine()`) can only
hand the cards over by stopping everything else, autoconfig budgets llama models against
a card's **total** VRAM with no idea Strata exists, the RAM side is uncoordinated (Strata
pins a 35–55 GB expert arena; on ROCm, host-RAM reclaim suspends KFD queues — the
documented `verify: timed out` stalls), and Strata never reaches Open WebUI because sync
skips non-router backends.

## What upstream Strata already offers (verified against Niko1221/Strata docs, 2026-10)

| Primitive | Notes |
| --- | --- |
| `POST /unload`, `POST /load`, `GET /health` | unload ~0.3 s, reload in seconds from the OS page cache; `/health` answers `"loaded"`. `/v1/models` lists the model as `unloaded` like llama's router. |
| `--min-free-vram-mib N` | answer 503 "the GPU is in use" instead of dying on `mtp: buffers do not fit`. |
| `--idle-unload S` | give VRAM and RAM back after S idle seconds. |
| `--before-load "cmd"` | a command run before the model reloads — the hook for asking model-loader to evict llama. |
| `--vram-reserve-mib N` | sizes the expert cache against free VRAM minus a reserve; works on HIP (AMD_HIP.md uses it). |
| `GET /metrics` | per-request `hit_rate`, `pcie_share`, `drafts_offered/accepted`, `ram_blobs`, `file_blobs`, `file_mb`. |
| `POST /v1/vram` elastic shrink | **NVIDIA-only** — not available on the R9700 pair; unload/load is the AMD mechanism. |
| `STRATA_*` env vars | reach the engine by inheritance; no entrypoint change needed. |

## Control matrix with the stock image (no fork)

The stock `docker-entrypoint.sh` (verified inside the image, 2026-10) has **no `CONFIG=`
hook**: it sources `strata-pin/strata.env` (exporting only `GPU`/`GPUS`), runs `setup.py`,
which always launches `serve/server.py` with `$STRATA_DATA/config/strata-<tag>.json`.
That is simpler than the plan assumed — one config, and upstream explicitly blesses
foreign keys in it: `SETUP_KEYS` names what setup writes itself and `carry_over()`
(#629) keeps every other key across a re-setup.

| Knob | Mechanism |
| --- | --- |
| Card pin | existing `strata.env` → `GPU=`/`GPUS=` (unchanged) |
| `--vram-reserve-mib`, `--expert-cache`, … | `args` array in the run config (setup's file) |
| `idle_unload_s`, `min_free_vram_mib`, `before_load` | same file, top-level server keys |
| `STRATA_ARENA_PIN_GIB` etc. | extra lines in `strata.env` |
| `/load` `/unload` `/health` `/metrics` | HTTP on the compose network |

The app is a second, honest writer of exactly the keys it names: `STRATA_CONFIG_KEYS`
(a JSON object in `config.py`) is merged into the run config at app start — atomic
replace, rolling `.bak-*`, never touching setup's keys. The path is derived from docker
inspect twice (the strata container's `$STRATA_DATA` mount, then this app's own mounts
translating the host path). A one-line note on the card shows the delta while the file
does not yet say what the settings say.

## Phases

- [x] **1. Load-state awareness.** `_probe_loaded_model` learns `unloaded` as a distinct
      state ("parked — loads in seconds") instead of "1 configured, none loaded"; the
      Strata card shows it; `_wait_ready` polls `/health` `"loaded"` for Strata because
      `/v1/models` answers 200 while unloaded. (commit 4fb67c6)
- [x] **2. Load/Unload without losing the page cache.** `POST /containers/{name}/strata-load|unload`
      routes + card buttons; the Engine switch gains a soft variant (unload + kfd drain
      wait, container stays warm); a parked target is asked back with /load instead of a
      no-op docker start; refusals surface on the card via `services.strata_error()`.
      (commit e024926)
- [x] **3. Config JSON ownership.** merge writer in `app/strata.py` (atomic replace,
      rolling backups, the `ini.py` pattern), `STRATA_CONFIG_KEYS` setting applied at app
      start, drift note on the card. No `CONFIG=` env line — the image has no such hook;
      the app writes the coexistence keys into setup's own run config, which upstream
      keeps across re-setups (#629).
- [x] **4. Card split.** when Strata is running on card N, autoconfig sees card N's capacity
      as zero (`_fit_card_vram`) so llama placement never counts it; the Engine row gains a
      **Share** button — offered only when the cards demonstrably divide (running Strata on
      a strict subset, or a stopped Strata pinned to one card beside a running llama) —
      that starts the engine and stops neither.
- [ ] **5. Time-share mode (opt-in).** app-managed `min_free_vram_mib` (503 beats OOM),
      `idle_unload_s` (auto give-back), and `before_load` → `POST /internal/yield` on
      model-loader, which stops the llama container so Strata's reload always wins. Pair
      with llama `--sleep-idle-seconds` so both engines yield when idle.
- [ ] **6. RAM coordination.** subtract Strata's live RSS from the RAM pool in autoconfig's
      CPU sizing (today only the static `HOST_RAM_RESERVE_GB`), show the pinned arena in
      the overview, warn near the ROCm reclaim-stall zone.
- [ ] **7. Polish.** speedometer/hero from `GET /metrics` (log regex demoted to fallback);
      Open WebUI sync registers Strata opt-in with known-ids from its own `/v1/models`;
      `ai-lab.vram-reserve-gb` label so declared tenants are subtracted from fit budgets.

## Ground rules

- `main` is untouched until live verification passes; merge is a human decision.
- Nothing is pushed to origin unless explicitly asked.
- The running `model-loader` container is never rebuilt or restarted for development;
  verification is a throwaway `docker build` + a test container on another port.
- One commit per phase; every commit boots the app standalone.
- **The Strata container on this box serves the agent doing this work.** It is never
  stopped, restarted, or sent `/unload` or `/load` during development - that takes the
  agent offline mid-task. GET endpoints (`/health`, `/v1/models`, `/metrics`) are safe
  to read; every control-path test runs against a fake Strata server on localhost.
