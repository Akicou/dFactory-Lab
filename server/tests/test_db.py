"""SPDX-License-Identifier: Apache-2.0"""
from server import db as dbmod


def test_init_creates_tables(tmp_path):
    p = tmp_path / "lab" / "t.sqlite"
    dbmod.init_db(p)
    with dbmod.connection(p) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"jobs", "training_runs", "models", "datasets", "settings_kv", "audit_events"} <= tables


def test_migrations_idempotent(tmp_path):
    p = tmp_path / "t.sqlite"
    dbmod.init_db(p)
    dbmod.init_db(p)  # second run must not error
    with dbmod.connection(p) as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
    assert "log_path" in cols


def test_insert_job(tmp_path):
    p = tmp_path / "t.sqlite"
    dbmod.init_db(p)
    with dbmod.connection(p) as conn:
        conn.execute(
            "INSERT INTO jobs (id, kind, state, created_at) VALUES (?,?,?,?)",
            ("j1", "merge", "queued", "2026-01-01T00:00:00Z"))
        row = conn.execute("SELECT kind, state FROM jobs WHERE id='j1'").fetchone()
    assert row["kind"] == "merge" and row["state"] == "queued"
