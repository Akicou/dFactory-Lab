"""SPDX-License-Identifier: Apache-2.0 - loaded-model ServerManager (no GPU needed)."""
import time

from server.services import servers as srv
from server.services.inference import DiffusionParams, MockBackend, SGLangBackend
from server.settings import get_settings


class FakeProc:
    def __init__(self):
        self.pid = 4321
        self._alive = True
        self.returncode = None

    def poll(self):
        return None if self._alive else self.returncode

    def terminate(self):
        self._alive = False
        self.returncode = 0

    def wait(self, timeout=None):
        self._alive = False
        return 0

    def kill(self):
        self._alive = False


def _mgr(**over):
    s = get_settings()
    for k, v in over.items():
        setattr(s, k, v)
    return s


def test_sglang_argv_from_model_card():
    argv = srv._sglang_argv("/models/LLaDA2.1-mini", 30007, get_settings())
    assert argv[1:3] == ["-m", "sglang.launch_server"]
    assert "--model-path" in argv and "/models/LLaDA2.1-mini" in argv
    assert argv[argv.index("--dllm-algorithm") + 1] == "JointThreshold"
    assert argv[argv.index("--port") + 1] == "30007"
    assert "--trust-remote-code" in argv


def test_load_ready_eject_with_injected_spawn():
    settings = _mgr(sglang_load_timeout_s=5.0)
    proc = FakeProc()
    m = srv.ServerManager(settings, spawn=lambda argv, log: proc, probe=lambda url: True)

    sid = m.load("/models/LLaDA2.1-mini")
    assert m.get(sid).state in ("starting", "ready")

    for _ in range(50):  # readiness poller runs in a daemon thread
        if m.get(sid).state == "ready":
            break
        time.sleep(0.05)
    assert m.get(sid).state == "ready"
    assert isinstance(m.backend_for(sid), SGLangBackend)

    assert m.eject(sid) is True
    assert m.get(sid) is None
    assert proc.poll() == 0  # terminated


def test_simulate_uses_mock_backend():
    m = srv.ServerManager(_mgr(sglang_simulate=True))
    sid = m.load("/models/whatever")
    assert m.get(sid).state == "ready"
    assert isinstance(m.backend_for(sid), MockBackend)
    out = m.generate(sid, [{"role": "user", "content": "hi"}], DiffusionParams())
    assert out["text"]


def test_capacity_limit():
    m = srv.ServerManager(_mgr(sglang_simulate=True, sglang_max_loaded=1))
    m.load("/models/a")
    try:
        m.load("/models/b")
        assert False, "expected capacity error"
    except ValueError as exc:
        assert "capacity" in str(exc)
