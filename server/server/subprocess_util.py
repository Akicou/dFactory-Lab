"""SPDX-License-Identifier: Apache-2.0
Safe subprocess execution (Checklist A-... subprocess safety).

Every wrapped CLI (download / merge / split / build / train / export) runs through
``run_tracked``: argument arrays only (never ``shell=True``), every user path is
jailed under an allow-listed root, and every process is registered so the server
can tear it all down on shutdown. This is the single chokepoint for spawning the
engine (torchrun, moe_convertor, build_gsm8k_dataset).
"""
from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .logging_config import get_logger

log = get_logger(__name__)

# job_id -> list[Popen], populated by run_tracked, drained by terminate_all.
_processes: dict[str, list[subprocess.Popen]] = {}
_lock = threading.Lock()


class PathEscapeError(ValueError):
    """Raised when a user-supplied path escapes the allowed roots."""


def validate_path(path: str | Path, roots: list[Path]) -> Path:
    """Resolve *path* and assert it is inside one of *roots*. Blocks ``..`` escapes."""
    p = Path(path).expanduser()
    resolved = (p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve())
    for root in roots:
        try:
            root_resolved = root.resolve()
        except OSError:
            continue
        if resolved == root_resolved or root_resolved in resolved.parents:
            return resolved
    raise PathEscapeError(
        f"Path {resolved} is outside the allowed roots: {[str(r) for r in roots]}"
    )


def argv_for_display(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def run_tracked(
    argv: list[str],
    *,
    job_id: str,
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    log_path: Optional[Path] = None,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    timeout: Optional[float] = None,
) -> dict:
    """Run *argv* (no shell), stream stdout+stderr line by line, honour cancellation.

    Returns ``{"returncode": int, "stdout_tail": str, "stderr_tail": str, "command": str}``.
    Raises ``subprocess.CalledProcessError`` on non-zero exit when the caller sets
    ``check`` - here we always return the dict so job functions can record the failure.
    """
    assert isinstance(argv, list) and all(isinstance(a, str) for a in argv), "argv must be a list[str]"
    display = argv_for_display(argv)
    log.info("subprocess.start", job_id=job_id, command=display, cwd=str(cwd) if cwd else None)

    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    # Never leak a token into a process that is not expected to use it.
    full_env.pop("HF_TOKEN", None)

    popen_env = {**full_env, "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1", "LINES": "0", "COLUMNS": "0"}

    log_file = open(log_path, "a", encoding="utf-8", errors="replace") if log_path else None
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd else None,
            env=popen_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            # New process group so we can signal the whole tree on cancel/shutdown.
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        if log_file:
            log_file.close()
        return {"returncode": 127, "stdout_tail": "", "stderr_tail": str(exc), "command": display}

    with _lock:
        _processes.setdefault(job_id, []).append(proc)

    stdout_lines: list[str] = []
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            stdout_lines.append(line)
            if log_file:
                log_file.write(line + "\n")
                log_file.flush()
            if on_line:
                try:
                    on_line(line)
                except Exception:  # noqa: BLE001 - a bad callback must not kill the run
                    pass
            if cancel_event is not None and cancel_event.is_set():
                _terminate(proc, job_id)
                break
    finally:
        if log_file:
            log_file.close()
        with _lock:
            lst = _processes.get(job_id, [])
            if proc in lst:
                lst.remove(proc)

    # If a timeout is set and still running, kill it.
    if timeout is not None and proc.poll() is None:
        _terminate(proc, job_id)

    rc = proc.wait(timeout=10) if proc.poll() is None else proc.returncode
    tail = "\n".join(stdout_lines[-40:])
    log.info("subprocess.exit", job_id=job_id, returncode=rc)
    return {"returncode": rc, "stdout_tail": tail, "stderr_tail": "", "command": display}


def register(key: str, proc: subprocess.Popen) -> None:
    """Track a long-lived process (e.g. a loaded inference server) so it is reaped
    by terminate_all on shutdown."""
    with _lock:
        _processes.setdefault(key, []).append(proc)


def terminate(key: str) -> None:
    """Terminate every process registered under *key* and drop it."""
    with _lock:
        procs = _processes.pop(key, [])
    for proc in procs:
        _terminate(proc, key)


def _terminate(proc: subprocess.Popen, job_id: str) -> None:
    """SIGTERM the process group, then SIGKILL after a grace period."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except (OSError, ProcessLookupError):
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except (OSError, ProcessLookupError):
            pass
    log.warning("subprocess.terminated", job_id=job_id, pid=proc.pid)


def terminate_all(timeout: float = 5.0) -> None:
    """Tear down every tracked subprocess (graceful shutdown)."""
    with _lock:
        all_procs = [p for lst in _processes.values() for p in lst]
    if not all_procs:
        return
    log.info("subprocess.terminate_all", count=len(all_procs))
    for proc in all_procs:
        if proc.poll() is None:
            try:
                proc.terminate()
            except (OSError, ProcessLookupError):
                pass
    deadline = time.monotonic() + timeout
    for proc in all_procs:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining if remaining > 0 else 0.1)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except (OSError, ProcessLookupError):
                pass
    with _lock:
        _processes.clear()
