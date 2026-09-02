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
            CREATE TABLE IF NOT EXISTS prompts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                body        TEXT NOT NULL,
                created_at  REAL NOT NULL
            );
            """
        )


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

    Matched on model_path when given and alias otherwise. Path is preferred because it is what
    llama-server logs directly; an alias only exists if the router's spawn block was inside the
    scraped window.
    """
    where, params = ["gen_tokens >= ?"], [int(min_gen_tokens)]
    if model_path:
        where.append("model_path = ?")
        params.append(model_path)
    elif alias:
        where.append("alias = ?")
        params.append(alias)
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
