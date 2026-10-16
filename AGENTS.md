# AGENTS.md

Model Loader: FastAPI + HTMX + Alpine + Tailwind (all via CDN, no frontend build) web UI managing llama.cpp GGUF models and containers on one personal homelab box. Ships as a single Docker container (python:3.12-slim), deps pinned in `requirements.txt`. Single-user, no auth, LAN-only, mounts the Docker socket.

## Commands

- Tests: `python3 -m pytest` from the repo root (single file: `python3 -m pytest tests/test_ini.py`). Dev-only deps in `requirements-dev.txt` (`pip install -r requirements-dev.txt`); pytest is deliberately NOT in `requirements.txt` — the shipped image stays lean and the Dockerfile only copies `app/`.
- **Every new feature ships with unit tests** (repo rule). Write them characterization-style: pin what the code does today. If a test surfaces a latent bug, report it, don't silently fix it. Tests need no docker and no network — `tests/conftest.py`'s `ini_env` fixture redirects the `settings` singleton's paths to tmp dirs; monkeypatch `app.utils.docker.from_env` for docker-touching code.
- Still no lint, no typecheck, no CI. The pytest suite is the only automated gate; run it before every PR and paste the output in the PR description.
- Deploy/build: run compose from the **parent compose project dir** (`~/ai-lab`), not this repo — this repo is symlinked in as `./model_loader`. `bootstrap.sh` inserts the service into an existing compose file (idempotent, backs it up first).
- Local dev: `pip install -r requirements.txt && uvicorn app.main:app --reload --port 8090`. Defaults in `app/config.py` are container paths (`/models`, `/data`) — override `MODELS_DIR`, `MODELS_INI_PATH`, `DATA_DIR` for host runs.
- One-shot layout migration (idempotent): `docker exec model-loader python3 -m app.migrate_layout`.

## Architecture

- Routes live in `app/routers/` — one module per nav section (overview, power, models, settings, search, downloads, config, containers, prompts, benchmark, strata), each an `APIRouter` included by `app/main.py` at the section's old position. `main.py` (~650 lines) keeps only app-level plumbing: factory, readonly middleware, startup, palette, badges, benchmark run-detail charts, Router-tab endpoints. Shared response plumbing is `app/web.py` (templates + `_toast`/`_QUEUED_CHIP`/`_blocked_chip`); shared placement math is `app/fit.py`. `services.py` (docker SDK, OpenWebUI reconciler) and `autoconfig.py` (VRAM fit math) are the other large modules. Docs' file trees lag behind `app/` — trust the directory.
- Templates: underscore prefix = HTMX partial fragment; pages extend `base.html`. Polling pages use `hx-trigger="every 2s"` partial swaps.
- sqlite schema is inline in `db.init()` — new tables/columns go there.
- `hw.py` sampler is a plain daemon thread over the sync docker SDK; this is deliberate, don't asyncio-ify it (see `docs/ARCHITECTURE.md`).
- `ini.py` writes `models.ini` atomically (`os.replace`) with 10 rolling `.bak` copies. Benchmarks never write to it.

## Gotchas

- **Dev/second instance against a live host: set `ROUTER_READONLY=true`.** A second instance otherwise races the primary over GPU start/stop/switch; the middleware in `main.py` refuses all mutating routes except `/router/`.
- Autoconfig **Fill is destructive within `AUTOCONFIG_DOMAIN`**: it clears keys it wants unset, not just writes values. Don't widen the domain casually.
- llama-server command-line flags **override** `models.ini` silently. Per-model settings belong in the ini; container `command:` stays limited to `--models-preset`, `--host`, `--port`, `--models-max`.
- `model =` values in ini must be **absolute** (`/models/...`); llama-server does not join them with `--models-dir`.
- OpenWebUI config only takes effect via direct writes to `webui.db` (PersistentConfig); compose env vars are seed-only after first boot.
- Non-llama engines (Strata) are discovered by the `ai-lab.engine` container label and are intentionally excluded from autoconfig, log telemetry, and `models.ini`. See `docs/STRATA_COEXISTENCE.md`.
- Vulkan images have no SMI tool: VRAM must be declared via `GPU_VRAM=container:GB` or fit/autoconfig can't size.

## Conventions

- Commit messages: single-line prose summary, capitalized, no conventional-commit prefixes (match `git log`). Work lands on `main` via PRs (merge commits).
- LF line endings enforced via `.gitattributes`.
- Deeper context: `docs/ARCHITECTURE.md` (design rationale), `docs/AUTOCONFIG.md` (fit math + calibration), `docs/TROUBLESHOOTING.md`, `docs/HOST_SENSORS.md`.
