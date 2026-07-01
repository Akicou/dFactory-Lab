"""SPDX-License-Identifier: Apache-2.0
FastAPI application factory (Checklist B-6, B-7, B-8, B-10, B-11, B-23).

``create_app()`` wires the lifespan (db + job registry + hardware probe), the
middleware stack (security headers, request logging, body-size cap), the system
routes and the Phase 2–6 pipeline route stubs, and serves the built frontend
with an SPA fallback when present.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from . import __version__, PHASE
from .logging_config import get_logger, setup_logging
from .schemas import ErrorDetail, OK

log = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Server"] = "dFactory-Lab"
        return resp


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        resp = await call_next(request)
        ms = round((time.perf_counter() - start) * 1000, 1)
        log.info("http.request", method=request.method, path=request.url.path,
                 status=resp.status_code, ms=ms)
        return resp


class MaxBodyMiddleware(BaseHTTPMiddleware):
    """Reject oversized request bodies (protects upload/dataset endpoints)."""
    MAX = 512 * 1024 * 1024  # 512 MiB ceiling; per-route limits can be tighter

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and int(cl) > self.MAX:
            return JSONResponse(status_code=413, content={"ok": False, "error": "request body too large"})
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from . import db as _db
    from . import jobs as _jobs
    from . import hardware as _hw
    from .settings import get_settings

    s = get_settings()
    s.ensure_dirs()
    _db.init_db(s.db_path())
    registry = _jobs.init_registry(s.db_path())
    registry.reconcile_on_startup()

    app.state.settings = s
    app.state.registry = registry
    app.state.startup_time = time.time()
    log.info("lifespan.startup", host=s.host, port=s.port, data_dir=str(s.data_dir))

    # Hardware probe is best-effort and fast (lazy torch); run it now so
    # /api/health has the GPU summary from the first request.
    try:
        app.state.hardware = _hw.detect_hardware()
    except Exception as exc:  # noqa: BLE001
        log.warning("hardware.probe_failed", error=str(exc))
        app.state.hardware = {"backend": "unknown"}

    yield

    log.info("lifespan.shutdown")
    try:
        registry.shutdown()
    except Exception as exc:  # noqa: BLE001 - resilient shutdown (B-11)
        log.warning("registry.shutdown_failed", error=str(exc))
    from . import subprocess_util
    try:
        subprocess_util.terminate_all()
    except Exception as exc:  # noqa: BLE001
        log.warning("subprocess.teardown_failed", error=str(exc))


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(
        title="dFactory-Lab",
        version=__version__,
        description="Fine-tune, merge & run diffusion LLMs without the CLI.",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    s_cors = []
    from .settings import get_settings
    s = get_settings()
    if s.cors_origins.strip():
        s_cors = [o.strip() for o in s.cors_origins.split(",") if o.strip()]
    else:
        # Loopback dev defaults + the Tauri desktop scheme.
        s_cors = ["http://127.0.0.1", "http://localhost", "tauri://localhost", "http://tauri.localhost"]
    app.add_middleware(
        CORSMiddleware, allow_origins=s_cors, allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )
    app.add_middleware(MaxBodyMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

    from .routes import system, models, datasets, training, pipeline
    app.include_router(system.router)
    app.include_router(models.router)
    app.include_router(datasets.router)
    app.include_router(training.router)
    app.include_router(pipeline.export)
    app.include_router(pipeline.chat)

    @app.exception_handler(HTTPException)
    async def http_exc_handler(request: Request, exc: HTTPException):  # noqa: ANN001
        return JSONResponse(status_code=exc.status_code,
                            content=ErrorDetail(error=str(exc.detail), detail=None).model_dump())

    @app.get("/", response_model=OK)
    async def root() -> OK:
        return OK(data={"name": "dFactory-Lab", "version": __version__, "phase": PHASE,
                        "docs": "/api/docs", "health": "/api/health"})

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve a built frontend (web/dist) with SPA fallback, if present (B-23)."""
    from .settings import repo_root
    dist = repo_root() / "web" / "dist"
    if not dist.is_dir():
        return
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str):  # noqa: ANN201
        if full_path.startswith("api/"):
            return JSONResponse(status_code=404, content={"ok": False, "error": "not found"})
        index = dist / "index.html"
        return FileResponse(index) if index.is_file() else JSONResponse(
            status_code=404, content={"ok": False, "error": "frontend not built"})
