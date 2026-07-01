"""SPDX-License-Identifier: Apache-2.0
Shared test fixtures. Points the managed data dir at a temp dir per session and
resets the Settings cache so env changes take effect.
"""
import os
import sys
from pathlib import Path

import pytest

_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

_TMP_DATA = _SERVER_DIR / ".lab-test"
os.environ["DFACTORY_LAB_DATA_DIR"] = str(_TMP_DATA)
os.environ.setdefault("DFACTORY_LAB_LOG_FORMAT", "console")


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DFACTORY_LAB_DATA_DIR", str(tmp_path / "lab"))
    from server import settings as _s
    _s.reset_settings_cache()
    yield
    _s.reset_settings_cache()


@pytest.fixture()
def app():
    from server.settings import reset_settings_cache
    reset_settings_cache()
    from server.main import create_app
    return create_app()


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
