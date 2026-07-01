"""SPDX-License-Identifier: Apache-2.0
Background job framework (Checklist B-16..B-18, R-... reconciliation).

A *job* is a unit of long work (download / merge / split / build_dataset / train
/ export / inference). Jobs are submitted as plain Python callables and run on a
thread pool; their state is persisted to SQLite so a UI reload or a server
restart never loses the record.

Callable contract:  fn(job: Job, update: Update) -> Any
  - ``update(progress: float, message: str = "", **partial)`` reports progress
  - raising -> state='error' with the traceback; returning -> state='done'
"""
from __future__ import annotations

import json
import sqlite3
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .logging_config import get_logger

log = get_logger(__name__)

JobFn = Callable[["Job", "Update"], Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    kind: str
    state: str = "queued"
    progress: float = 0.0
    message: str = ""
    payload: dict = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    log_path: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["result"] = self.result if _jsonable(self.result) else str(self.result)
        return d


def _jsonable(v: Any) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


@dataclass
class Update:
    """Progress reporter handed to each job callable."""
    job_id: str
    _set_state: Callable[[str], None]
    _set_progress: Callable[[float, str, dict], None]

    def __call__(self, progress: float, message: str = "", **partial: Any) -> None:
        self._set_progress(float(progress), message, partial)

    def done(self, result: Any = None, message: str = "done") -> None:
        self._set_progress(1.0, message, {"result": result})
        self._set_state("done")


class JobRegistry:
    """Thread-safe registry + SQLite persistence for jobs."""

    def __init__(self, db_path: Path, max_workers: int = 4) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._subs: dict[str, list[Callable[[dict], None]]] = {}
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lab-job")
        self._load()

    # ── persistence ────────────────────────────────────────────────────────
    def _load(self) -> None:
        from . import db as _db
        try:
            with _db.connection(self._db_path) as conn:
                for row in conn.execute("SELECT * FROM jobs"):
                    j = Job(
                        id=row["id"], kind=row["kind"], state=row["state"],
                        progress=row["progress"], message=row["message"],
                        payload=json.loads(row["payload"] or "{}"),
                        result=json.loads(row["result"]) if row["result"] else None,
                        error=row["error"], created_at=row["created_at"],
                        started_at=row["started_at"], ended_at=row["ended_at"],
                        log_path=row["log_path"],
                    )
                    self._jobs[j.id] = j
        except sqlite3.Error as exc:
            log.warning("jobs.load_failed", error=str(exc))

    def _persist(self, job: Job) -> None:
        from . import db as _db
        with _db.connection(self._db_path) as conn:
            conn.execute(
                """INSERT INTO jobs (id, kind, state, progress, message, payload,
                                      result, error, created_at, started_at, ended_at, log_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     state=excluded.state, progress=excluded.progress,
                     message=excluded.message, result=excluded.result,
                     error=excluded.error, started_at=excluded.started_at,
                     ended_at=excluded.ended_at, log_path=excluded.log_path""",
                (job.id, job.kind, job.state, job.progress, job.message,
                 json.dumps(job.payload, default=str),
                 json.dumps(job.result, default=str) if _jsonable(job.result) else None,
                 job.error, job.created_at, job.started_at, job.ended_at, job.log_path),
            )

    # ── public API ─────────────────────────────────────────────────────────
    def submit(self, kind: str, fn: JobFn, payload: Optional[dict] = None,
               job_id: Optional[str] = None) -> str:
        import uuid
        jid = job_id or uuid.uuid4().hex[:12]
        job = Job(id=jid, kind=kind, payload=payload or {})
        with self._lock:
            self._jobs[jid] = job
            self._cancel[jid] = threading.Event()
        self._persist(job)
        self._emit(job)
        future = self._pool.submit(self._run, jid, fn)
        with self._lock:
            self._futures[jid] = future
        log.info("job.submitted", id=jid, kind=kind)
        return jid

    def _run(self, jid: str, fn: JobFn) -> None:
        job = self._jobs[jid]
        job.state = "running"
        job.started_at = _now()
        self._persist(job)
        self._emit(job)

        def set_state(s: str) -> None:
            job.state = s

        def set_progress(p: float, msg: str, partial: dict) -> None:
            job.progress = max(job.progress, min(1.0, p))
            if msg:
                job.message = msg
            if "result" in partial:
                job.result = partial["result"]
            self._persist(job)
            self._emit(job)

        update = Update(job_id=jid, _set_state=set_state, _set_progress=set_progress)
        try:
            result = fn(job, update)
            if job.state != "done":
                job.result = result
                job.state = "done"
                job.progress = 1.0
                job.message = job.message or "done"
        except Exception as exc:  # noqa: BLE001 - job failures are operational, not bugs
            job.state = "error"
            job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            log.error("job.failed", id=jid, kind=job.kind, error=str(exc))
        finally:
            job.ended_at = _now()
            self._persist(job)
            self._emit(job)

    def cancel(self, jid: str) -> bool:
        with self._lock:
            ev = self._cancel.get(jid)
            job = self._jobs.get(jid)
        if ev and job and job.state in ("queued", "running"):
            ev.set()
            job.state = "cancelled"
            job.ended_at = _now()
            self._persist(job)
            self._emit(job)
            return True
        return False

    def is_cancelled(self, jid: str) -> bool:
        with self._lock:
            ev = self._cancel.get(jid)
        return bool(ev and ev.is_set())

    def get(self, jid: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(jid)

    def list(self, kind: Optional[str] = None, limit: int = 100) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if kind:
            jobs = [j for j in jobs if j.kind == kind]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.state in ("queued", "running"))

    # ── subscriptions (SSE feed) ───────────────────────────────────────────
    def subscribe(self, jid: str, cb: Callable[[dict], None]) -> Callable[[], None]:
        with self._lock:
            self._subs.setdefault(jid, []).append(cb)
        # replay current state
        job = self.get(jid)
        if job:
            cb(job.to_dict())

        def unsub() -> None:
            with self._lock:
                lst = self._subs.get(jid, [])
                if cb in lst:
                    lst.remove(cb)
        return unsub

    def _emit(self, job: Job) -> None:
        with self._lock:
            subs = list(self._subs.get(job.id, []))
        payload = job.to_dict()
        for cb in subs:
            try:
                cb(payload)
            except Exception:  # noqa: BLE001
                pass

    # ── lifecycle ──────────────────────────────────────────────────────────
    def reconcile_on_startup(self) -> int:
        """Mark any job left running/queued from a previous process as errored."""
        n = 0
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.state in ("queued", "running"):
                job.state = "error"
                job.error = (job.error or "") + " [reconciled: server restarted during job]"
                job.ended_at = _now()
                self._persist(job)
                n += 1
        if n:
            log.warning("jobs.reconciled", count=n)
        return n

    def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel running jobs and drain the pool on server shutdown."""
        with self._lock:
            running = [j.id for j in self._jobs.values() if j.state == "running"]
        for jid in running:
            self.cancel(jid)
        self._pool.shutdown(wait=False, cancel_futures=True)
        log.info("jobs.shutdown", cancelled=running)


# Module-level singleton, initialized by the app lifespan.
_registry: Optional[JobRegistry] = None


def init_registry(db_path: Path, max_workers: int = 4) -> JobRegistry:
    global _registry
    _registry = JobRegistry(db_path, max_workers=max_workers)
    return _registry


def get_registry() -> JobRegistry:
    if _registry is None:
        raise RuntimeError("JobRegistry not initialized — call init_registry() first")
    return _registry
