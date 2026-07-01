"""SPDX-License-Identifier: Apache-2.0
System routes (Checklist B-12, B-13, B-9): health, liveness, system info,
shutdown, and the cross-cutting job status / SSE feed.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from .. import __version__, PHASE
from ..auth import auth_required, access_token, verify
from ..logging_config import get_logger
from ..schemas import Health, OK, SystemInfo
from ..services import audit as audit_svc

router = APIRouter(prefix="/api", tags=["system"])
log = get_logger(__name__)


def _uptime(req: Request) -> float:
    return round(time.time() - req.app.state.startup_time, 2)


@router.get("/health", response_model=Health)
async def health(req: Request) -> Health:
    registry = req.app.state.registry
    return Health(
        status="ok",
        version=__version__,
        phase=PHASE,
        bind=req.app.state.settings.host,
        uptime_s=_uptime(req),
        active_jobs=registry.active_count() if registry else 0,
        gpu=req.app.state.hardware if getattr(req.app.state, "hardware", None) else None,
    )


@router.get("/liveness")
async def liveness() -> OK:
    return OK(data={"alive": True})


@router.get("/auth/verify")
async def auth_verify(req: Request) -> OK:
    return OK(data={"auth_required": auth_required(req), "verified": verify(req),
                    "token_present": bool(access_token(req))})


@router.get("/security")
async def security(req: Request) -> OK:
    s = req.app.state.settings
    return OK(data={
        "bind": s.host, "loopback": s.is_loopback, "exposed": not s.is_loopback,
        "auth_required": auth_required(req), "token_present": bool(access_token(req)),
        "engine_present": __import__("pathlib").Path(__file__).resolve().parents[3] and True,
    })


@router.get("/audit")
async def audit_log(req: Request, limit: int = 100) -> OK:
    return OK(data=audit_svc.recent(req.app.state.settings.db_path(), limit=limit))


@router.get("/system", response_model=SystemInfo)
async def system(req: Request) -> SystemInfo:
    import platform, sys
    from ..settings import repo_root
    s = req.app.state.settings
    return SystemInfo(
        python=sys.version.split()[0],
        platform=platform.platform(),
        cpus=__import__("os").cpu_count() or 1,
        hardware=req.app.state.hardware or {},
        data_dir=str(s.data_dir),
        engine_present=(repo_root() / "train.sh").is_file(),
    )


@router.post("/shutdown")
async def shutdown(req: Request) -> OK:
    """Trigger a graceful shutdown of the server and all tracked subprocesses."""
    log.warning("shutdown.requested")
    trigger = getattr(req.app.state, "trigger_shutdown", None)
    if trigger is None:
        return OK(data={"shutting_down": False, "note": "no shutdown hook registered"})
    # Fire after responding so the client gets a reply.
    asyncio.get_running_loop().call_later(0.1, trigger)
    return OK(data={"shutting_down": True})


# ── jobs (cross-cutting; feature routes submit into this registry) ──────────
@router.get("/jobs")
async def list_jobs(req: Request, kind: str | None = None, limit: int = 100) -> Any:
    registry = req.app.state.registry
    return [j.to_dict() for j in registry.list(kind=kind, limit=limit)]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, req: Request) -> Any:
    job = req.app.state.registry.get(job_id)
    if job is None:
        return {"ok": False, "error": "job not found", "id": job_id}
    return job.to_dict()


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, req: Request) -> Any:
    ok = req.app.state.registry.cancel(job_id)
    return {"ok": ok, "id": job_id}


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, req: Request) -> StreamingResponse:
    """Server-Sent Events stream of a job's state. Closes when the job is terminal."""
    import json
    import queue
    q: "queue.Queue[dict | None]" = queue.Queue()
    registry = req.app.state.registry
    unsub = registry.subscribe(job_id, lambda d: q.put(d))

    async def event_gen():
        try:
            while True:
                try:
                    payload = await asyncio.to_thread(q.get, True, 1.0)
                except Exception:  # noqa: BLE001 - timeout, loop again
                    payload = None
                if payload is not None:
                    yield f"data: {json.dumps(payload)}\n\n"
                    if payload.get("state") in ("done", "error", "cancelled"):
                        break
                else:
                    yield ": keepalive\n\n"
        finally:
            unsub()

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
