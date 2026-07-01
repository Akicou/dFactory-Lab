"""SPDX-License-Identifier: Apache-2.0"""
from server.settings import Settings


def test_defaults():
    s = Settings()
    assert s.host == "127.0.0.1"
    assert s.port == 8000
    assert s.is_loopback is True
    assert "db" in s.dirs()


def test_env_override(monkeypatch):
    monkeypatch.setenv("DFACTORY_LAB_HOST", "0.0.0.0")
    monkeypatch.setenv("DFACTORY_LAB_PORT", "9999")
    s = Settings()
    assert s.host == "0.0.0.0"
    assert s.port == 9999
    assert s.is_loopback is False


def test_resolved_token_autogenerates():
    s = Settings()
    s.token = ""
    assert len(s.resolved_token) > 8
    s2 = Settings()
    s2.token = "fixed-token"
    assert s2.resolved_token == "fixed-token"


def test_engine_env_has_cuda_order():
    env = Settings().engine_env()
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["NNODES"] == "1"
