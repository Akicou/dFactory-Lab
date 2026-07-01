"""SPDX-License-Identifier: Apache-2.0
Loaded-model server manager for the Chat page.

Each "loaded" model is one long-lived SGLang server process on its own port
(``python -m sglang.launch_server ...``, per the LLaDA2.1 model card). This module
launches them, polls readiness, routes chat to the right one, and ejects/tears them
down. Loaded state is ephemeral — nothing survives a restart.

On a box without a GPU / sglang, ``load`` surfaces state="error" with the message
instead of crashing. Set ``sglang_simulate`` to drive the whole flow against the
deterministic MockBackend (dev + tests).
"""
from __future__ import annotations

import shlex
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .. import subprocess_util
from ..logging_config import get_logger
from .inference import DiffusionParams, MockBackend, SGLangBackend

log = get_logger(__name__)


@dataclass
class LoadedServer:
    id: str
    model_path: str
    port: int
    url: str
    state: str = "starting"   # starting | ready | error | stopped
    pid: Optional[int] = None
    message: str = ""
    log_path: str = ""
    started_at: float = field(default_factory=time.time)

    def public(self) -> dict:
        d = asdict(self)
        d["name"] = Path(self.model_path).name
        return d


def _free_port(base: int) -> int:
    """First bindable port at/after *base* (best-effort; races are harmless — the
    server just fails to bind and reports state=error)."""
    for port in range(base, base + 512):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port in [{base}, {base + 512})")


def _sglang_argv(model_path: str, port: int, settings) -> list[str]:
    """Build the `python -m sglang.launch_server` command from the model card."""
    python = settings.sglang_python or sys.executable
    argv = [
        python, "-m", "sglang.launch_server",
        "--model-path", str(model_path),
        "--dllm-algorithm", settings.sglang_dllm_algorithm,
        "--trust-remote-code",
        "--tp-size", str(settings.sglang_tp_size),
        "--mem-fraction-static", str(settings.sglang_mem_fraction),
        "--max-running-requests", "1",
        "--attention-backend", settings.sglang_attention_backend,
        "--host", settings.sglang_host,
        "--port", str(port),
    ]
    if settings.sglang_extra_args.strip():
        argv += shlex.split(settings.sglang_extra_args)
    return argv


def _default_spawn(argv: list[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", encoding="utf-8", errors="replace")
    return subprocess.Popen(
        argv, stdout=f, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,  # own process group => clean group kill
    )


def _default_probe(url: str, timeout: float = 3.0) -> bool:
    """True when the SGLang server answers /health with 2xx."""
    import httpx
    try:
        r = httpx.get(f"{url}/health", timeout=timeout)
        return r.status_code < 300
    except Exception:  # noqa: BLE001 - not up yet
        return False


class ServerManager:
    """Owns the set of loaded model servers. Thread-safe; one instance per app."""

    def __init__(self, settings, *, spawn: Callable = _default_spawn,
                 probe: Callable[[str], bool] = _default_probe) -> None:
        self.settings = settings
        self._spawn = spawn
        self._probe = probe
        self._servers: dict[str, LoadedServer] = {}
        self._lock = threading.Lock()

    # ── queries ──────────────────────────────────────────────────────────────
    def list(self) -> list[dict]:
        with self._lock:
            return [s.public() for s in self._servers.values()]

    def get(self, sid: str) -> Optional[LoadedServer]:
        with self._lock:
            return self._servers.get(sid)

    def backend_for(self, sid: str):
        """Return the inference backend for a ready server, or raise ValueError."""
        s = self.get(sid)
        if s is None:
            raise ValueError(f"no such loaded model: {sid}")
        if s.state != "ready":
            raise ValueError(f"model {s.public()['name']} is {s.state}, not ready")
        if self.settings.sglang_simulate:
            b = MockBackend()
            b.load(s.model_path)
            return b
        return SGLangBackend(s.url, timeout=self.settings.sglang_timeout)

    def generate(self, sid: str, messages: list[dict], params: DiffusionParams) -> dict:
        return self.backend_for(sid).generate(messages, params)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def load(self, model_path: str) -> str:
        with self._lock:
            live = [s for s in self._servers.values() if s.state in ("starting", "ready")]
            if len(live) >= self.settings.sglang_max_loaded:
                raise ValueError(
                    f"already at capacity ({self.settings.sglang_max_loaded} loaded); eject one first")
            sid = uuid.uuid4().hex[:8]
            port = _free_port(self.settings.sglang_port_base + len(self._servers))
            url = f"http://{self.settings.sglang_host}:{port}"
            srv = LoadedServer(id=sid, model_path=str(model_path), port=port, url=url)
            srv.log_path = str(Path(self.settings.data_dir) / "logs" / f"sglang-{sid}.log")
            self._servers[sid] = srv

        if self.settings.sglang_simulate:
            srv.state = "ready"
            srv.message = "simulated (mock backend)"
            log.info("servers.load.simulated", id=sid, model=srv.model_path)
            return sid

        try:
            argv = _sglang_argv(model_path, port, self.settings)
            proc = self._spawn(argv, Path(srv.log_path))
            subprocess_util.register(sid, proc)
            srv.pid = proc.pid
            log.info("servers.load.spawned", id=sid, pid=proc.pid, port=port)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the request
            srv.state = "error"
            srv.message = str(exc)
            log.warning("servers.load.spawn_failed", id=sid, error=str(exc))
            return sid

        threading.Thread(target=self._await_ready, args=(sid, proc), daemon=True).start()
        return sid

    def _await_ready(self, sid: str, proc: subprocess.Popen) -> None:
        deadline = time.monotonic() + self.settings.sglang_load_timeout_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:  # process died during startup
                self._set(sid, state="error", message=f"server exited (code {proc.returncode})")
                return
            if self._probe(self.get(sid).url):
                self._set(sid, state="ready", message="")
                log.info("servers.ready", id=sid)
                return
            time.sleep(2.0)
        self._set(sid, state="error", message="timed out waiting for /health")
        subprocess_util.terminate(sid)

    def eject(self, sid: str) -> bool:
        with self._lock:
            srv = self._servers.pop(sid, None)
        if srv is None:
            return False
        subprocess_util.terminate(sid)
        srv.state = "stopped"
        log.info("servers.eject", id=sid)
        return True

    def _set(self, sid: str, **fields) -> None:
        with self._lock:
            srv = self._servers.get(sid)
            if srv is None:
                return
            for k, v in fields.items():
                setattr(srv, k, v)
