"""SPDX-License-Identifier: Apache-2.0
dFactory-Lab server entrypoint (Checklist B-4, B-5, B-9, B-10, B-11, B-21, B-22).

Self-contained: ``python server/run.py`` boots the FastAPI app under uvicorn,
finds a free port if the requested one is taken, prints the access banner, and
runs until Ctrl+C / ``POST /api/shutdown`` — tearing down every tracked
subprocess (training, merge, download) on the way out. Mirrors Unsloth Studio's
run.py shape without copying its (AGPL) source.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

# Make the package importable when run directly (like Unsloth's run.py).
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

os.environ.setdefault("PYTHONWARNINGS", "ignore")


def _ensure_utf8_stdio() -> None:
    """Windows cp1252 can't encode the banner emoji (or non-ASCII paths) — force UTF-8.
    Mirrors Unsloth main.py's Windows setup (Checklist B-26)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass


_ensure_utf8_stdio()

from server.logging_config import get_logger, setup_logging  # noqa: E402
from server.banner import install_uvicorn_log_rewrite, print_lab_banner, print_port_in_use_notice  # noqa: E402

log = get_logger("server.run")


def _is_port_free(host: str, port: int) -> bool:
    try:
        info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)[0]
        family, socktype, proto, _, sockaddr = info
        with socket.socket(family, socktype, proto) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(sockaddr)
        return True
    except OSError:
        return False


def _find_free_port(host: str, start: int, attempts: int = 20) -> int:
    for off in range(attempts):
        cand = start + off
        if _is_port_free(host, cand):
            return cand
    raise RuntimeError(f"No free port in {start}–{start + attempts - 1}")


def _apply_args_to_env(args: argparse.Namespace) -> None:
    """Args override Settings (which is env-driven)."""
    mapping = {
        "host": "DFACTORY_LAB_HOST", "port": "DFACTORY_LAB_PORT",
        "token": "DFACTORY_LAB_TOKEN", "data_dir": "DFACTORY_LAB_DATA_DIR",
        "log_format": "DFACTORY_LAB_LOG_FORMAT", "log_level": "DFACTORY_LAB_LOG_LEVEL",
    }
    for attr, env in mapping.items():
        val = getattr(args, attr, None)
        if val is not None and val != "":
            os.environ[env] = str(val)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the dFactory-Lab server")
    p.add_argument("--host", default=None, help="Bind host (default 127.0.0.1; 0.0.0.0 = network)")
    p.add_argument("--port", type=int, default=None, help="Bind port (default 8000)")
    p.add_argument("--token", default=None, help="Bootstrap access token (auto-generated if unset)")
    p.add_argument("--data-dir", dest="data_dir", default=None, help="Managed data directory")
    p.add_argument("--log-format", dest="log_format", default=None, choices=["console", "json"])
    p.add_argument("--log-level", dest="log_level", default=None)
    p.add_argument("--silent", action="store_true", help="Suppress the startup banner")
    return p


def _graceful_shutdown(server, shutdown_event: threading.Event) -> None:
    log.info("graceful_shutdown.start")
    if server is not None:
        server.should_exit = True
    try:
        from server import jobs
        # registry may be set by lifespan; tear down if present
        registry = getattr(_app_holder.app.state, "registry", None) if _app_holder.app else None
        if registry is not None:
            registry.shutdown()
    except Exception as exc:  # noqa: BLE001 - resilient (B-11)
        log.warning("graceful_shutdown.jobs_failed", error=str(exc))
    try:
        from server import subprocess_util
        subprocess_util.terminate_all()
    except Exception as exc:  # noqa: BLE001
        log.warning("graceful_shutdown.subprocess_failed", error=str(exc))
    shutdown_event.set()
    log.info("graceful_shutdown.done")


class _AppHolder:
    app = None


_app_holder = _AppHolder()


def run_server(host: str | None = None, port: int | None = None, silent: bool = False) -> None:
    import uvicorn

    args = _build_parser().parse_args([] if host is None and port is None else None)
    _apply_args_to_env(args)
    from server.settings import get_settings
    s = get_settings()
    setup_logging(level=s.log_level, fmt=s.log_format)

    bind_host = host or s.host
    requested_port = port or s.port

    # Free-port handling (B-5).
    if not _is_port_free(bind_host, requested_port):
        original = requested_port
        requested_port = _find_free_port(bind_host, requested_port + 1)
        if not silent:
            print_port_in_use_notice(original, requested_port)

    install_uvicorn_log_rewrite()

    from server.main import create_app
    app = create_app()
    _app_holder.app = app

    ready = threading.Event()
    failed = threading.Event()

    class _ReadyServer(uvicorn.Server):
        async def startup(self, *a, **kw):  # noqa: ANN002, ANN003
            await super().startup(*a, **kw)
            if getattr(self, "started", False) and not self.should_exit:
                ready.set()

    config = uvicorn.Config(app, host=bind_host, port=requested_port,
                            log_level="info", access_log=False, server_header=False)
    server = _ReadyServer(config)
    shutdown_event = threading.Event()

    def trigger_shutdown() -> None:
        _graceful_shutdown(server, shutdown_event)

    # Exposed before first request so POST /api/shutdown works immediately.
    app.state.trigger_shutdown = trigger_shutdown

    def _serve() -> None:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        except Exception as exc:  # noqa: BLE001
            log.error("server.serve_failed", error=str(exc))
            failed.set()
        finally:
            loop.close()
            if not ready.is_set():
                failed.set()

    t = threading.Thread(target=_serve, daemon=True, name="uvicorn")
    t.start()

    # Wait until uvicorn binds (or fails).
    while not ready.is_set():
        if failed.is_set() or not t.is_alive():
            sys.stderr.write("ERROR: dFactory-Lab failed to start.\n")
            sys.exit(1)
        ready.wait(timeout=0.1)

    display_host = bind_host if bind_host not in ("0.0.0.0", "::") else _resolve_lan_ip()
    if not silent:
        print_lab_banner(port=requested_port, bind_host=bind_host,
                         display_host=display_host, token=s.resolved_token if s.token else "",
                         phase=1)

    # Signal handlers (Ctrl+C / SIGTERM / SIGBREAK on Windows).
    def _sig(signum, frame):  # noqa: ANN001
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_DFL)
        _graceful_shutdown(server, shutdown_event)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sig)

    while not shutdown_event.is_set():
        shutdown_event.wait(timeout=1.0)
    t.join(timeout=5.0)


def _resolve_lan_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    try:
        run_server()
    except KeyboardInterrupt:
        pass
