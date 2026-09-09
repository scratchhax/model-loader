from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .config import settings

_LOCK = threading.Lock()
_DB_PATH: Path | None = None


def _path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        _DB_PATH = settings.data_dir / "model_loader.db"
    return _DB_PATH


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_path())
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init() -> None:
    with _LOCK, _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS download_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id      TEXT NOT NULL,
                filename     TEXT NOT NULL,
                dest_path    TEXT NOT NULL,
                total_bytes  INTEGER NOT NULL,
                status       TEXT NOT NULL,
                error        TEXT,
                started_at   REAL NOT NULL,
                completed_at REAL
            );
            CREATE TABLE IF NOT EXISTS avatar_cache (
                owner       TEXT PRIMARY KEY,
                url         TEXT NOT NULL,
                fetched_at  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS update_check (
                filename          TEXT PRIMARY KEY,
                repo_id           TEXT NOT NULL,
                hf_last_modified  TEXT,
                checked_at        REAL NOT NULL
            );
            -- One row per completed llama-server request, scraped from its logs.
            -- UNIQUE(backend, instance, task) is what makes ingestion idempotent: the scraper
            -- re-reads the whole log tail every pass rather than tracking a watermark, and
            -- relies on this constraint to drop what it has already seen.
            CREATE TABLE IF NOT EXISTS req_timing (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            REAL NOT NULL,
                backend       TEXT NOT NULL,
                instance      TEXT NOT NULL,
                task          INTEGER NOT NULL,
                model_path    TEXT NOT NULL DEFAULT '',
                alias         TEXT NOT NULL DEFAULT '',
                spec_type     TEXT NOT NULL DEFAULT '',
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                prompt_tps    REAL    NOT NULL DEFAULT 0,
                gen_tokens    INTEGER NOT NULL DEFAULT 0,
                gen_tps       REAL    NOT NULL DEFAULT 0,
                draft_acc     REAL,
                draft_len     REAL,
                UNIQUE(backend, instance, task)
            );
            CREATE INDEX IF NOT EXISTS req_timing_model
                ON req_timing(model_path, ts DESC);
            CREATE INDEX IF NOT EXISTS req_timing_alias
                ON req_timing(alias, ts DESC);
            -- The arguments one spawned llama-server instance was given. Stored whole, as
            -- JSON, rather than as columns for the flags that seemed interesting: what makes
            -- two runs different is discovered by diffing these, so a fixed column set would
            -- have to be extended every time it guessed wrong.
            -- Benchmark results. Deliberately a dead end: nothing reads these back into a
            -- configuration decision. They exist so a person can look at what their hardware
            -- actually does, which is why every measurement is stored raw rather than reduced
            -- to a score.
            CREATE TABLE IF NOT EXISTS bench_run (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                backend     TEXT NOT NULL,
                status      TEXT NOT NULL,           -- running | done | cancelled | error
                started_at  REAL NOT NULL,
                finished_at REAL,
                reps        INTEGER NOT NULL DEFAULT 1,
                max_tokens  INTEGER NOT NULL DEFAULT 256,
                note        TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS bench_variant (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id   INTEGER NOT NULL,
                alias    TEXT NOT NULL,
                argv_json TEXT NOT NULL DEFAULT '{}',
                load_ms  REAL,
                UNIQUE(run_id, alias)
            );
            CREATE TABLE IF NOT EXISTS bench_result (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                variant_id  INTEGER NOT NULL,
                prompt_name TEXT NOT NULL,
                rep         INTEGER NOT NULL,
                cold        INTEGER NOT NULL DEFAULT 0,
                ttft_ms     REAL,
                ttft_answer_ms REAL,
                truncated   INTEGER NOT NULL DEFAULT 0,
                total_ms    REAL,
                prompt_n    INTEGER,
                prompt_tps  REAL,
                gen_n       INTEGER,
                gen_tps     REAL,
                draft_n     INTEGER,
                draft_acc   REAL,
                peak_vram_json TEXT NOT NULL DEFAULT '[]',
                contended   INTEGER NOT NULL DEFAULT 0,
                err         TEXT NOT NULL DEFAULT '',
                -- The generated text itself. Timings alone cannot answer "is this model any
                -- good at this", and a run that keeps only timings destroys the one artifact
                -- that could ever be graded. Keeping it makes scoring an OFFLINE pass over
                -- stored rows: a rubric can change and re-score history without spending GPU
                -- time again. Reasoning is deliberately not kept - the answer is what gets
                -- graded, and thinking would multiply row size for text no rubric reads.
                response_text TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS bench_result_variant ON bench_result(variant_id);
            -- Results from llama.cpp's own `llama bench`, which is the right tool for raw
            -- prompt-processing and generation throughput: it warms up, repeats, reports a
            -- standard deviation, and sweeps parameters natively. The whole JSON entry is kept
            -- because it carries every setting the run used; the extracted columns exist only
            -- so the common questions can be asked in SQL.
            CREATE TABLE IF NOT EXISTS bench_sweep (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id       INTEGER NOT NULL,
                alias        TEXT NOT NULL DEFAULT '',
                model_type   TEXT NOT NULL DEFAULT '',
                test         TEXT NOT NULL DEFAULT '',
                n_prompt     INTEGER, n_gen INTEGER, n_depth INTEGER,
                avg_ts       REAL,    stddev_ts REAL,
                n_gpu_layers INTEGER, n_cpu_moe INTEGER, n_ubatch INTEGER,
                type_k       TEXT,    type_v TEXT, flash_attn INTEGER,
                split_mode   TEXT,
                raw_json     TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS bench_sweep_run ON bench_sweep(run_id);
            CREATE TABLE IF NOT EXISTS server_config (
                backend    TEXT NOT NULL,
                instance   TEXT NOT NULL,
                alias      TEXT NOT NULL DEFAULT '',
                model_path TEXT NOT NULL DEFAULT '',
                argv_json  TEXT NOT NULL DEFAULT '{}',
                first_seen REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (backend, instance)
            );
            CREATE TABLE IF NOT EXISTS prompts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                body        TEXT NOT NULL,
                created_at  REAL NOT NULL
            );

            -- ------------------------------------------------ capability evals
            -- Separate from bench_* deliberately. Those answer "what does my hardware do"
            -- and go stale whenever a config changes; these answer "what is this model good
            -- at" and go stale only when the model changes. Sharing tables would mean
            -- re-running an eval every time a tensor-split moved, and would force a
            -- throughput measurement to be reduced to a score, which bench_* exists not to
            -- do.
            --
            -- Grading is a SECOND pass, on purpose. A run stores generations; a grader fills
            -- the score columns afterwards. That split is what makes a rubric revisable: the
            -- expensive half (GPU time) is paid once, and the cheap half (scoring) can be
            -- redone over history for free.
            CREATE TABLE IF NOT EXISTS eval_suite (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                slug    TEXT NOT NULL,                  -- humaneval-plus, aider-polyglot
                -- Version is part of the identity rather than a mutable field. Revising a
                -- problem set or a rubric has to produce a NEW suite, otherwise old scores
                -- are silently compared against new criteria and the badge lies.
                version TEXT NOT NULL DEFAULT '1',
                kind    TEXT NOT NULL DEFAULT 'code',   -- code | writing
                title   TEXT NOT NULL DEFAULT '',
                UNIQUE(slug, version)
            );
            CREATE TABLE IF NOT EXISTS eval_case (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                suite_id    INTEGER NOT NULL,
                case_key    TEXT NOT NULL,              -- HumanEval/0
                prompt      TEXT NOT NULL,
                -- Everything the grader needs and the model must never see: unit tests,
                -- entry point, forbidden words, target length. Opaque to the runner by
                -- design, so a new kind of grader needs no schema change.
                grader_json TEXT NOT NULL DEFAULT '{}',
                -- 'quick' marks a fixed subset. A full coding suite is hours per model here
                -- (164 problems x ~1400 tokens is ~3.7 h at 17 tok/s), so a subset is what
                -- makes the feature usable rather than a nicety.
                tier        TEXT NOT NULL DEFAULT 'full',
                ord         INTEGER NOT NULL DEFAULT 0,
                UNIQUE(suite_id, case_key)
            );
            CREATE INDEX IF NOT EXISTS eval_case_suite ON eval_case(suite_id, tier);
            CREATE TABLE IF NOT EXISTS eval_run (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                suite_id    INTEGER NOT NULL,
                backend     TEXT NOT NULL,
                tier        TEXT NOT NULL DEFAULT 'full',
                status      TEXT NOT NULL,              -- running|done|cancelled|error
                started_at  REAL NOT NULL,
                finished_at REAL,
                max_tokens  INTEGER NOT NULL DEFAULT 3072,
                note        TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS eval_result (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id        INTEGER NOT NULL,
                case_id       INTEGER NOT NULL,
                alias         TEXT NOT NULL,
                -- Generation half: written while the run is happening.
                response_text TEXT NOT NULL DEFAULT '',
                gen_n         INTEGER,
                gen_tps       REAL,
                total_ms      REAL,
                truncated     INTEGER NOT NULL DEFAULT 0,
                err           TEXT NOT NULL DEFAULT '',
                -- Grading half: written later, nullable until a grader has run. NULL score
                -- and score 0.0 mean different things - not yet judged, versus judged and
                -- wrong - and a badge that conflates them is worse than no badge.
                graded_at     REAL,
                grader        TEXT NOT NULL DEFAULT '',
                passed        INTEGER,
                score         REAL,
                detail_json   TEXT NOT NULL DEFAULT '{}',
                UNIQUE(run_id, case_id, alias)
            );
            CREATE INDEX IF NOT EXISTS eval_result_run ON eval_result(run_id, alias);
            CREATE INDEX IF NOT EXISTS eval_result_ungraded ON eval_result(run_id, graded_at);
            """
        )
    _add_missing_columns()


def _add_missing_columns() -> None:
    """Add columns that were introduced after a table shipped.

    CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so a new column
    has to be ALTERed in on any database created before it was added.
    """
    wanted = {
        "bench_result": (
            ("ttft_answer_ms", "REAL"),
            ("truncated", "INTEGER NOT NULL DEFAULT 0"),
            ("response_text", "TEXT NOT NULL DEFAULT ''"),
        ),
    }
    with _LOCK, _conn() as c:
        for table, cols in wanted.items():
            try:
                have = {r["name"] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
            except sqlite3.Error:
                continue
            if not have:
                continue  # table not created yet; the CREATE above already has these
            for name, decl in cols:
                if name not in have:
                    try:
                        c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                    except sqlite3.Error:
                        pass


def get_setting(key: str, default: str = "") -> str:
    with _LOCK, _conn() as c:
        row = c.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO kv(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def record_download(*, repo_id: str, filename: str, dest_path: str, total_bytes: int,
                    status: str, error: str | None, started_at: float, completed_at: float | None) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO download_history(repo_id, filename, dest_path, total_bytes, status, error, started_at, completed_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (repo_id, filename, dest_path, total_bytes, status, error, started_at, completed_at),
        )


def recent_downloads(limit: int = 50) -> list[sqlite3.Row]:
    with _LOCK, _conn() as c:
        return list(c.execute(
            "SELECT * FROM download_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall())


def owner_by_filename() -> dict[str, str]:
    """filename -> owner, from successful downloads."""
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT filename, repo_id FROM download_history WHERE status = 'done'"
        ).fetchall()
    return {r["filename"]: (r["repo_id"].split("/", 1)[0] if "/" in r["repo_id"] else r["repo_id"]) for r in rows}


def downloaded_files_by_repo() -> dict[str, list[str]]:
    """{repo_id: [filename, ...]} of successfully-downloaded files, one entry per repo."""
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT repo_id, filename FROM download_history WHERE status = 'done' ORDER BY id DESC"
        ).fetchall()
    out: dict[str, list[str]] = {}
    for r in rows:
        rid = r["repo_id"]
        if not rid:
            continue
        # Preserve insertion order for stable display; skip duplicate filenames per repo.
        lst = out.setdefault(rid, [])
        if r["filename"] not in lst:
            lst.append(r["filename"])
    return out


def get_avatar(owner: str) -> dict | None:
    with _LOCK, _conn() as c:
        r = c.execute("SELECT url, fetched_at FROM avatar_cache WHERE owner = ?", (owner,)).fetchone()
        return {"url": r["url"], "fetched_at": r["fetched_at"]} if r else None


def set_avatar(owner: str, url: str, fetched_at: float) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO avatar_cache(owner, url, fetched_at) VALUES(?, ?, ?) "
            "ON CONFLICT(owner) DO UPDATE SET url = excluded.url, fetched_at = excluded.fetched_at",
            (owner, url, fetched_at),
        )


def all_update_checks() -> dict[str, dict]:
    with _LOCK, _conn() as c:
        rows = c.execute("SELECT filename, repo_id, hf_last_modified, checked_at FROM update_check").fetchall()
    return {r["filename"]: dict(r) for r in rows}


def set_update_check(filename: str, repo_id: str, hf_last_modified: str | None, checked_at: float) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO update_check(filename, repo_id, hf_last_modified, checked_at) VALUES(?, ?, ?, ?) "
            "ON CONFLICT(filename) DO UPDATE SET repo_id = excluded.repo_id, "
            "hf_last_modified = excluded.hf_last_modified, checked_at = excluded.checked_at",
            (filename, repo_id, hf_last_modified, checked_at),
        )


def list_prompts() -> list[dict]:
    with _LOCK, _conn() as c:
        rows = c.execute("SELECT id, name, body, created_at FROM prompts ORDER BY name COLLATE NOCASE").fetchall()
    return [dict(r) for r in rows]


def add_prompt(name: str, body: str) -> int:
    import time as _t
    with _LOCK, _conn() as c:
        cur = c.execute(
            "INSERT INTO prompts(name, body, created_at) VALUES(?, ?, ?)",
            (name, body, _t.time()),
        )
        return cur.lastrowid or 0


def delete_prompt(pid: int) -> bool:
    with _LOCK, _conn() as c:
        cur = c.execute("DELETE FROM prompts WHERE id = ?", (pid,))
        return cur.rowcount > 0


def get_prompt(pid: int) -> dict | None:
    with _LOCK, _conn() as c:
        r = c.execute("SELECT id, name, body FROM prompts WHERE id = ?", (pid,)).fetchone()
        return dict(r) if r else None


def download_records() -> list[dict]:
    """Latest 'done' record per (filename, repo_id)."""
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT filename, repo_id, MAX(id) AS id FROM download_history "
            "WHERE status = 'done' GROUP BY filename, repo_id"
        ).fetchall()
    return [dict(r) for r in rows]


def record_timings(backend: str, samples: list) -> int:
    """Store request samples, ignoring any already held. Returns the number newly inserted.

    INSERT OR IGNORE against UNIQUE(backend, instance, task) is what lets the scraper re-read
    the same log tail on every pass without either duplicating rows or having to remember how
    far it got last time.
    """
    if not samples:
        return 0
    rows = [(s.ts, backend, s.instance, s.task, s.model_path, s.alias, s.spec_type,
             s.prompt_tokens, s.prompt_tps, s.gen_tokens, s.gen_tps,
             s.draft_acc, s.draft_len) for s in samples]
    with _LOCK, _conn() as c:
        before = c.total_changes
        c.executemany(
            "INSERT OR IGNORE INTO req_timing("
            "ts, backend, instance, task, model_path, alias, spec_type, "
            "prompt_tokens, prompt_tps, gen_tokens, gen_tps, draft_acc, draft_len) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return c.total_changes - before


def recent_timings(*, model_path: str = "", alias: str = "",
                   min_gen_tokens: int = 0, limit: int = 400) -> list[sqlite3.Row]:
    """Most recent samples for a model, newest first.

    Matched on alias first, model_path only as a fallback.

    The alias IS the models.ini section name - it is what the router passes as --alias - so it
    is the exact key for "this section's history". Path is broader than intended: two sections
    can point at the same GGUF under completely different settings, and matching on path pooled
    them, so one section's panel reported another section's throughput as its own.
    """
    where, params = ["gen_tokens >= ?"], [int(min_gen_tokens)]
    if alias:
        where.append("alias = ?")
        params.append(alias)
    elif model_path:
        where.append("model_path = ?")
        params.append(model_path)
    else:
        return []
    params.append(int(limit))
    with _LOCK, _conn() as c:
        return list(c.execute(
            "SELECT * FROM req_timing WHERE " + " AND ".join(where)
            + " ORDER BY ts DESC LIMIT ?", params
        ).fetchall())


def record_server_configs(backend: str, configs: list) -> int:
    """Upsert the argv of each spawned instance. Re-ingest overwrites, because a later pass
    over the log may have seen the full argv block where an earlier one caught only the
    announcement line."""
    if not configs:
        return 0
    import json as _json
    rows = [(backend, c.instance, c.alias, c.model_path,
             _json.dumps(c.argv or {}, sort_keys=True), c.first_seen) for c in configs]
    with _LOCK, _conn() as c:
        c.executemany(
            "INSERT INTO server_config(backend, instance, alias, model_path, argv_json, first_seen) "
            "VALUES(?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(backend, instance) DO UPDATE SET "
            "  alias      = CASE WHEN excluded.alias      != '' THEN excluded.alias      ELSE server_config.alias END, "
            "  model_path = CASE WHEN excluded.model_path != '' THEN excluded.model_path ELSE server_config.model_path END, "
            "  argv_json  = CASE WHEN excluded.argv_json  != '{}' THEN excluded.argv_json ELSE server_config.argv_json END, "
            "  first_seen = CASE WHEN server_config.first_seen = 0 THEN excluded.first_seen ELSE server_config.first_seen END",
            rows,
        )
    return len(rows)


def timings_by_instance(*, model_path: str = "", alias: str = "",
                        min_gen_tokens: int = 0) -> list[sqlite3.Row]:
    """Per-instance aggregates for one model, newest instance first.

    Grouping is by instance because the router respawns on any argument change, so an instance
    is exactly one configuration. Percentiles are computed by the caller: SQLite has no median,
    and the row counts here are small enough that sorting in Python is cheaper than faking it.
    """
    where, params = ["gen_tokens >= ?"], [int(min_gen_tokens)]
    if alias:
        where.append("alias = ?")
        params.append(alias)
    elif model_path:
        where.append("model_path = ?")
        params.append(model_path)
    else:
        return []
    with _LOCK, _conn() as c:
        return list(c.execute(
            "SELECT t.instance, t.gen_tps, t.prompt_tps, t.draft_acc, t.draft_len, t.ts, "
            "       t.spec_type, t.backend "
            "FROM req_timing t WHERE " + " AND ".join(where) + " ORDER BY t.ts DESC", params
        ).fetchall())


def server_configs_for(instances: list[str]) -> dict[str, dict]:
    """{instance: argv dict} for the given instances."""
    if not instances:
        return {}
    import json as _json
    qs = ",".join("?" * len(instances))
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT instance, argv_json FROM server_config WHERE instance IN (%s)" % qs,
            list(instances)).fetchall()
    out = {}
    for r in rows:
        try:
            out[r["instance"]] = _json.loads(r["argv_json"] or "{}")
        except ValueError:
            out[r["instance"]] = {}
    return out


# ---------------------------------------------------------------- benchmark

def bench_create_run(backend: str, reps: int, max_tokens: int, started_at: float) -> int:
    with _LOCK, _conn() as c:
        cur = c.execute(
            "INSERT INTO bench_run(backend, status, started_at, reps, max_tokens) "
            "VALUES(?, 'running', ?, ?, ?)", (backend, started_at, int(reps), int(max_tokens)))
        return int(cur.lastrowid)


def bench_finish_run(run_id: int, status: str, finished_at: float, note: str = "") -> None:
    with _LOCK, _conn() as c:
        c.execute("UPDATE bench_run SET status = ?, finished_at = ?, note = ? WHERE id = ?",
                  (status, finished_at, note, int(run_id)))


def bench_add_variant(run_id: int, alias: str, argv_json: str = "{}") -> int:
    with _LOCK, _conn() as c:
        cur = c.execute(
            "INSERT OR IGNORE INTO bench_variant(run_id, alias, argv_json) VALUES(?, ?, ?)",
            (int(run_id), alias, argv_json))
        if cur.lastrowid:
            return int(cur.lastrowid)
        row = c.execute("SELECT id FROM bench_variant WHERE run_id = ? AND alias = ?",
                        (int(run_id), alias)).fetchone()
        return int(row["id"]) if row else 0


def bench_set_load_ms(variant_id: int, load_ms: float | None) -> None:
    with _LOCK, _conn() as c:
        c.execute("UPDATE bench_variant SET load_ms = ? WHERE id = ?", (load_ms, int(variant_id)))


def bench_add_result(variant_id: int, **kw) -> None:
    cols = ("prompt_name", "rep", "cold", "ttft_ms", "ttft_answer_ms", "truncated",
            "total_ms", "prompt_n", "prompt_tps",
            "gen_n", "gen_tps", "draft_n", "draft_acc", "peak_vram_json", "contended", "err",
            "response_text")
    # Every NOT NULL column here that has a meaningful "nothing to say" value. A caller that
    # omits one would otherwise abort the insert with a constraint error part-way through a
    # long run, losing the whole row for the sake of a field with a perfectly good default.
    # prompt_name and rep are deliberately absent: a result row that does not know which
    # prompt or repetition it came from is not worth storing.
    _empty = {"err": "", "response_text": "", "peak_vram_json": "[]",
              "cold": 0, "contended": 0, "truncated": 0}
    vals = [int(variant_id)] + [
        _empty[k] if (kw.get(k) is None and k in _empty) else kw.get(k) for k in cols
    ]
    with _LOCK, _conn() as c:
        c.execute("INSERT INTO bench_result(variant_id, " + ", ".join(cols) + ") "
                  "VALUES(" + ", ".join("?" * (len(cols) + 1)) + ")", vals)


def bench_runs(limit: int = 25) -> list[sqlite3.Row]:
    with _LOCK, _conn() as c:
        return list(c.execute(
            "SELECT r.*, (SELECT COUNT(*) FROM bench_variant v WHERE v.run_id = r.id) AS n_variants, "
            "       (SELECT COUNT(*) FROM bench_result x JOIN bench_variant v2 ON x.variant_id = v2.id "
            "        WHERE v2.run_id = r.id) AS n_results "
            "FROM bench_run r ORDER BY r.id DESC LIMIT ?", (int(limit),)).fetchall())


def bench_run(run_id: int) -> sqlite3.Row | None:
    with _LOCK, _conn() as c:
        return c.execute("SELECT * FROM bench_run WHERE id = ?", (int(run_id),)).fetchone()


def bench_variants(run_id: int) -> list[sqlite3.Row]:
    with _LOCK, _conn() as c:
        return list(c.execute("SELECT * FROM bench_variant WHERE run_id = ? ORDER BY id",
                              (int(run_id),)).fetchall())


def bench_results(run_id: int) -> list[sqlite3.Row]:
    with _LOCK, _conn() as c:
        return list(c.execute(
            "SELECT x.*, v.alias FROM bench_result x JOIN bench_variant v ON x.variant_id = v.id "
            "WHERE v.run_id = ? ORDER BY v.id, x.prompt_name, x.rep", (int(run_id),)).fetchall())


def prompts_by_ids(ids: list[int]) -> list[sqlite3.Row]:
    if not ids:
        return []
    qs = ",".join("?" * len(ids))
    with _LOCK, _conn() as c:
        return list(c.execute("SELECT * FROM prompts WHERE id IN (%s) ORDER BY name" % qs,
                              [int(i) for i in ids]).fetchall())


# Prompts the benchmark needs to be useful out of the box. Seeded once, guarded by a flag
# rather than by "is the table empty", so someone who deletes them does not get them back on
# the next restart. They are ordinary prompts afterwards - editable, deletable, and usable
# from the Prompts page like any other.
_BENCH_SEED_KEY = "bench_prompts_seeded"

_SWA_BLURB = (
    "Sliding-window attention limits each token's view to a fixed span of recent tokens "
    "rather than the whole sequence. Layers alternate between local windows and full global "
    "attention, so only the global layers carry a KV cache that grows with context length. "
    "This keeps memory close to flat as context grows, at some cost to how far information "
    "can travel in a single layer."
)

_BENCH_PROMPTS = (
    ("Bench: creative writing",
     "Write the opening three paragraphs of a short story about a lighthouse keeper who "
     "discovers the light has been going out on its own. Establish the setting and their "
     "state of mind. Do not summarise the plot; write the prose itself."),
    ("Bench: coding",
     "Write a Python function merge_intervals(intervals) that takes a list of (start, end) "
     "tuples and returns them merged and sorted. Handle empty input and touching intervals. "
     "Include a docstring and three test cases."),
    ("Bench: reasoning",
     "A train leaves station A at 14:05 travelling at 80 km/h. A second train leaves station "
     "B, 300 km away, at 14:35 travelling toward A at 100 km/h. At what time do they meet, "
     "and how far from station A? Show your working step by step."),
    # Long input on purpose: prompt processing is a separate cost from generation, and only a
    # prompt big enough to take real time makes the prompt tok/s column mean anything.
    ("Bench: summarise long input",
     "Summarise the following in exactly five bullet points."
     + (chr(10) + chr(10)) + ((_SWA_BLURB + chr(10) + chr(10)) * 12)),
)


def seed_bench_prompts() -> int:
    """Add the built-in benchmark prompts once. Returns how many were added."""
    if get_setting(_BENCH_SEED_KEY, ""):
        return 0
    have = {p["name"] for p in list_prompts()}
    added = 0
    for name, body in _BENCH_PROMPTS:
        if name not in have:
            add_prompt(name, body)
            added += 1
    set_setting(_BENCH_SEED_KEY, "1")
    return added


def bench_add_sweep(run_id: int, alias: str, entry: dict) -> None:
    """Store one `llama bench` result row, raw JSON included."""
    import json as _json
    npr = int(entry.get("n_prompt") or 0)
    ngen = int(entry.get("n_gen") or 0)
    ndep = int(entry.get("n_depth") or 0)
    # Same label llama-bench prints, so a row here and a row in its own output are obviously
    # the same measurement.
    test = (f"pp{npr}" if npr else f"tg{ngen}") + (f" @ d{ndep}" if ndep else "")
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO bench_sweep(run_id, alias, model_type, test, n_prompt, n_gen, n_depth, "
            "avg_ts, stddev_ts, n_gpu_layers, n_cpu_moe, n_ubatch, type_k, type_v, flash_attn, "
            "split_mode, raw_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (int(run_id), alias, str(entry.get("model_type") or ""), test, npr, ngen, ndep,
             entry.get("avg_ts"), entry.get("stddev_ts"),
             entry.get("n_gpu_layers"), entry.get("n_cpu_moe"), entry.get("n_ubatch"),
             str(entry.get("type_k") or ""), str(entry.get("type_v") or ""),
             entry.get("flash_attn"), str(entry.get("split_mode") or ""),
             _json.dumps(entry, sort_keys=True)))


def bench_sweeps(run_id: int) -> list[sqlite3.Row]:
    with _LOCK, _conn() as c:
        return list(c.execute(
            "SELECT * FROM bench_sweep WHERE run_id = ? ORDER BY alias, n_depth, id",
            (int(run_id),)).fetchall())


# ---------------------------------------------------------------- capability evals
#
# Two-phase by design: a run writes generations, a grader writes scores over them later.
# Every function here belongs to one of those halves, or is a read for the UI.


def eval_suite_upsert(slug: str, version: str, kind: str, title: str = "") -> int:
    """Get-or-create a suite. Returns its id.

    Idempotent, so vendoring a problem set can be re-run on every boot without piling up
    duplicates or needing a separate "have I seeded this yet" flag.
    """
    with _LOCK, _conn() as c:
        c.execute("INSERT OR IGNORE INTO eval_suite(slug, version, kind, title) "
                  "VALUES(?, ?, ?, ?)", (slug, version, kind, title))
        row = c.execute("SELECT id FROM eval_suite WHERE slug = ? AND version = ?",
                        (slug, version)).fetchone()
        return int(row["id"]) if row else 0


def eval_suites() -> list[sqlite3.Row]:
    with _LOCK, _conn() as c:
        return list(c.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM eval_case k WHERE k.suite_id = s.id) AS n_cases "
            "FROM eval_suite s ORDER BY s.kind, s.slug, s.version").fetchall())


def eval_case_upsert(suite_id: int, case_key: str, prompt: str,
                     grader_json: str = "{}", tier: str = "full", ord_: int = 0) -> int:
    """Insert or update one case. Returns its id."""
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO eval_case(suite_id, case_key, prompt, grader_json, tier, ord) "
            "VALUES(?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(suite_id, case_key) DO UPDATE SET "
            "  prompt = excluded.prompt, grader_json = excluded.grader_json, "
            "  tier = excluded.tier, ord = excluded.ord",
            (int(suite_id), case_key, prompt, grader_json, tier, int(ord_)))
        row = c.execute("SELECT id FROM eval_case WHERE suite_id = ? AND case_key = ?",
                        (int(suite_id), case_key)).fetchone()
        return int(row["id"]) if row else 0


def eval_cases(suite_id: int, tier: str = "") -> list[sqlite3.Row]:
    """Cases in a suite. tier='quick' returns only the subset; anything else returns all.

    'quick' is a SUBSET of the suite, not a sibling tier, so asking for the full set must not
    filter on tier at all. Filtering `tier = 'full'` would silently drop every case marked
    quick and shrink the suite to whatever nobody flagged.
    """
    sql = "SELECT * FROM eval_case WHERE suite_id = ?"
    args: list = [int(suite_id)]
    if tier == "quick":
        sql += " AND tier = 'quick'"
    sql += " ORDER BY ord, id"
    with _LOCK, _conn() as c:
        return list(c.execute(sql, args).fetchall())


def eval_create_run(suite_id: int, backend: str, tier: str, max_tokens: int,
                    started_at: float) -> int:
    with _LOCK, _conn() as c:
        cur = c.execute(
            "INSERT INTO eval_run(suite_id, backend, tier, status, started_at, max_tokens) "
            "VALUES(?, ?, ?, 'running', ?, ?)",
            (int(suite_id), backend, tier, started_at, int(max_tokens)))
        return int(cur.lastrowid)


def eval_finish_run(run_id: int, status: str, finished_at: float, note: str = "") -> None:
    with _LOCK, _conn() as c:
        c.execute("UPDATE eval_run SET status = ?, finished_at = ?, note = ? WHERE id = ?",
                  (status, finished_at, note, int(run_id)))


def eval_add_result(run_id: int, case_id: int, alias: str, **kw) -> int:
    """Store one generation. Replaces any previous row for the same (run, case, model).

    REPLACE rather than IGNORE because a retried case produces new text, and a grade left
    attached to text that no longer exists is worse than no grade. The score columns are not
    carried over by the replace - they reset to NULL along with the row, which is correct:
    new output has not been judged yet.
    """
    cols = ("response_text", "gen_n", "gen_tps", "total_ms", "truncated", "err")
    _empty = {"response_text": "", "err": "", "truncated": 0}
    vals = [int(run_id), int(case_id), alias] + [
        _empty[k] if (kw.get(k) is None and k in _empty) else kw.get(k) for k in cols
    ]
    with _LOCK, _conn() as c:
        cur = c.execute(
            "INSERT OR REPLACE INTO eval_result(run_id, case_id, alias, " + ", ".join(cols) +
            ") VALUES(" + ", ".join("?" * (len(cols) + 3)) + ")", vals)
        return int(cur.lastrowid)


def eval_grade(result_id: int, grader: str, passed: int | None, score: float | None,
               detail_json: str, graded_at: float) -> None:
    """Write the grading half of a row. Called by the offline pass, never by the runner."""
    with _LOCK, _conn() as c:
        c.execute(
            "UPDATE eval_result SET graded_at = ?, grader = ?, passed = ?, score = ?, "
            "detail_json = ? WHERE id = ?",
            (graded_at, grader, passed, score, detail_json, int(result_id)))


def eval_ungraded(run_id: int = 0, limit: int = 500) -> list[sqlite3.Row]:
    """Rows awaiting a grade, with the case data a grader needs joined in.

    Errored generations are skipped. There is no text to judge, and scoring them zero would
    fold an infrastructure failure into a capability number.
    """
    sql = ("SELECT r.*, k.case_key, k.grader_json, s.slug, s.version, s.kind "
           "FROM eval_result r "
           "JOIN eval_case  k  ON k.id = r.case_id "
           "JOIN eval_run   run ON run.id = r.run_id "
           "JOIN eval_suite s  ON s.id = run.suite_id "
           "WHERE r.graded_at IS NULL AND r.err = ''")
    args: list = []
    if run_id:
        sql += " AND r.run_id = ?"
        args.append(int(run_id))
    sql += " ORDER BY r.id LIMIT ?"
    args.append(int(limit))
    with _LOCK, _conn() as c:
        return list(c.execute(sql, args).fetchall())


def eval_runs(limit: int = 25) -> list[sqlite3.Row]:
    with _LOCK, _conn() as c:
        return list(c.execute(
            "SELECT run.*, s.slug, s.version, s.kind, s.title, "
            "  (SELECT COUNT(*) FROM eval_result r WHERE r.run_id = run.id) AS n_results, "
            "  (SELECT COUNT(DISTINCT r.alias) FROM eval_result r WHERE r.run_id = run.id) "
            "    AS n_models, "
            "  (SELECT COUNT(*) FROM eval_result r WHERE r.run_id = run.id "
            "     AND r.graded_at IS NOT NULL) AS n_graded "
            "FROM eval_run run JOIN eval_suite s ON s.id = run.suite_id "
            "ORDER BY run.id DESC LIMIT ?", (int(limit),)).fetchall())


def eval_run(run_id: int) -> sqlite3.Row | None:
    with _LOCK, _conn() as c:
        return c.execute(
            "SELECT run.*, s.slug, s.version, s.kind, s.title "
            "FROM eval_run run JOIN eval_suite s ON s.id = run.suite_id "
            "WHERE run.id = ?", (int(run_id),)).fetchone()


def eval_results(run_id: int) -> list[sqlite3.Row]:
    with _LOCK, _conn() as c:
        return list(c.execute(
            "SELECT r.*, k.case_key, k.tier FROM eval_result r "
            "JOIN eval_case k ON k.id = r.case_id "
            "WHERE r.run_id = ? ORDER BY r.alias, k.ord, k.id", (int(run_id),)).fetchall())


def eval_model_scores() -> list[sqlite3.Row]:
    """Latest graded score per (model, suite). The row a badge renders from.

    "Latest" is by run id, not by best score: a badge has to describe the model as it is now,
    not on its best day.

    The CTE is load-bearing. Grouping eval_result by (suite, alias) and taking MAX(run_id) in
    the same SELECT looks equivalent and is not - the other aggregates would then span every
    run the model ever did, so one bad early run would drag a good latest one down forever.
    Pinning the run id first and aggregating only that run's rows is the whole difference.
    """
    with _LOCK, _conn() as c:
        return list(c.execute(
            """
            WITH latest AS (
                SELECT run.suite_id AS suite_id, r.alias AS alias, MAX(run.id) AS run_id
                FROM eval_result r
                JOIN eval_run run ON run.id = r.run_id
                WHERE r.graded_at IS NOT NULL
                GROUP BY run.suite_id, r.alias
            )
            SELECT s.slug, s.version, s.kind, s.title, l.alias, l.run_id,
                   run.tier                                       AS tier,
                   COUNT(*)                                       AS n_cases,
                   SUM(CASE WHEN r.passed = 1 THEN 1 ELSE 0 END)  AS n_passed,
                   AVG(r.score)                                   AS mean_score,
                   MAX(r.graded_at)                               AS graded_at
            FROM latest l
            JOIN eval_result r  ON r.run_id = l.run_id AND r.alias = l.alias
            JOIN eval_run  run  ON run.id = l.run_id
            JOIN eval_suite s   ON s.id = run.suite_id
            WHERE r.graded_at IS NOT NULL
            GROUP BY l.run_id, l.alias
            ORDER BY s.slug, l.alias
            """).fetchall())
