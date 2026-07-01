"""SPDX-License-Identifier: Apache-2.0
SQLite persistence (Checklist B-16, B-17).

- WAL mode + foreign keys ON (PRAGMA foreign_keys, cascading deletes)
- per-call connections via a ``connection()`` context manager (no shared handle
  across threads — SQLite serializes anyway, and this avoids "database is locked"
  under the job framework's writers)
- idempotent migrations: ``migrate()`` introspects PRAGMA table_info and ALTERs
  only missing columns, so it is safe to run on every startup
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .logging_config import get_logger

log = get_logger(__name__)

_lock = threading.Lock()

_SCHEMA = {
    "jobs": """
        CREATE TABLE IF NOT EXISTS jobs (
            id          TEXT PRIMARY KEY,
            kind        TEXT NOT NULL,                 -- download|merge|split|build_dataset|train|export|inference
            state       TEXT NOT NULL DEFAULT 'queued',-- queued|running|done|error|cancelled
            progress    REAL NOT NULL DEFAULT 0.0,
            message     TEXT NOT NULL DEFAULT '',
            payload     TEXT NOT NULL DEFAULT '{}',    -- JSON: inputs
            result      TEXT,                          -- JSON: outputs
            error       TEXT,
            created_at  TEXT NOT NULL,
            started_at  TEXT,
            ended_at    TEXT
        )
    """,
    "training_runs": """
        CREATE TABLE IF NOT EXISTS training_runs (
            id            TEXT PRIMARY KEY,
            job_id        TEXT REFERENCES jobs(id) ON DELETE SET NULL,
            model_id      TEXT,
            dataset_id    TEXT,
            config_path   TEXT,
            output_dir    TEXT,
            status        TEXT NOT NULL DEFAULT 'queued',
            best_loss     REAL,
            last_step     INTEGER,
            created_at    TEXT NOT NULL,
            updated_at    TEXT
        )
    """,
    "models": """
        CREATE TABLE IF NOT EXISTS models (
            id          TEXT PRIMARY KEY,
            repo_id     TEXT,
            path        TEXT,
            format      TEXT,          -- separate_expert | merged_expert | runnable
            size_bytes  INTEGER,
            meta        TEXT NOT NULL DEFAULT '{}',
            created_at  TEXT NOT NULL
        )
    """,
    "datasets": """
        CREATE TABLE IF NOT EXISTS datasets (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            source      TEXT,          -- hf | upload | paste
            path        TEXT,
            rows        INTEGER,
            revision    TEXT,
            created_at  TEXT NOT NULL
        )
    """,
    "settings_kv": """
        CREATE TABLE IF NOT EXISTS settings_kv (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """,
    "audit_events": """
        CREATE TABLE IF NOT EXISTS audit_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            actor       TEXT,
            action      TEXT NOT NULL,
            target      TEXT,
            detail      TEXT NOT NULL DEFAULT '{}'
        )
    """,
}


@contextmanager
def connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a short-lived connection with pragmas applied. Commits on success."""
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    """Create tables + run idempotent migrations. Safe to call on every boot."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, connection(db_path) as conn:
        for name, ddl in _SCHEMA.items():
            conn.execute(ddl)
        _migrate(conn)
    log.info("db.initialized", path=str(db_path), tables=list(_SCHEMA))


def _column_set(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _migrate(conn: sqlite3.Connection) -> None:
    """Forward-only, additive migrations keyed by (table, column)."""
    additions = {
        ("jobs", "log_path"): "ALTER TABLE jobs ADD COLUMN log_path TEXT",
    }
    for (table, col), ddl in additions.items():
        if col not in _column_set(conn, table):
            log.info("db.migrate", table=table, column=col)
            conn.execute(ddl)
