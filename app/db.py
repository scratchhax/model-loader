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
                err         TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS bench_result_variant ON bench_result(variant_id);
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


def downloaded_repo_ids() -> set[str]:
    """Set of HF repo ids ('owner/name') that have at least one successfully-downloaded file."""
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT DISTINCT repo_id FROM download_history WHERE status = 'done'"
        ).fetchall()
    return {r["repo_id"] for r in rows if r["repo_id"]}


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


def timing_row_count() -> int:
    with _LOCK, _conn() as c:
        return int(c.execute("SELECT COUNT(*) AS n FROM req_timing").fetchone()["n"])


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
            "gen_n", "gen_tps", "draft_n", "draft_acc", "peak_vram_json", "contended", "err")
    vals = [int(variant_id)] + [kw.get(k) for k in cols]
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
