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
