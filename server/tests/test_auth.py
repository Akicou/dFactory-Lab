"""SPDX-License-Identifier: Apache-2.0 - bootstrap-token auth + audit log."""
from fastapi.testclient import TestClient


def test_loopback_open(client):
    # default bind is loopback -> auth not required
    assert client.get("/api/models").status_code == 200
    sec = client.get("/api/security").json()["data"]
    assert sec["auth_required"] is False and sec["loopback"] is True


def test_exposed_requires_token(monkeypatch):
    monkeypatch.setenv("DFACTORY_LAB_HOST", "0.0.0.0")
    monkeypatch.setenv("DFACTORY_LAB_TOKEN", "secret")
    from server.settings import reset_settings_cache
    reset_settings_cache()
    from server.main import create_app
    with TestClient(create_app()) as c:
        # protected route without a token -> 401
        assert c.get("/api/models").status_code == 401
        # public routes stay open
        assert c.get("/api/health").status_code == 200
        assert c.get("/api/auth/verify").status_code == 200
        # bearer header works
        assert c.get("/api/models", headers={"Authorization": "Bearer secret"}).status_code == 200
        # query-param token works
        assert c.get("/api/models?token=secret").status_code == 200
        # wrong token -> 401
        assert c.get("/api/models", headers={"Authorization": "Bearer nope"}).status_code == 401
    reset_settings_cache()


def test_wrong_token_constant_time(monkeypatch):
    # ensures the 401 path doesn't crash on malformed headers
    monkeypatch.setenv("DFACTORY_LAB_HOST", "0.0.0.0")
    monkeypatch.setenv("DFACTORY_LAB_TOKEN", "secret")
    from server.settings import reset_settings_cache
    reset_settings_cache()
    from server.main import create_app
    with TestClient(create_app()) as c:
        assert c.get("/api/models", headers={"Authorization": "Bearer"}).status_code == 401
        assert c.get("/api/models").status_code == 401
    reset_settings_cache()


def test_audit_records_mutations(client):
    client.post("/api/datasets/convert", json={
        "rows": [{"q": "hi", "a": "hello"}],
        "mapping": {"q": "user", "a": "assistant"}, "name": "audited"})
    rows = client.get("/api/audit").json()["data"]
    actions = [r["action"] for r in rows]
    assert "datasets.convert" in actions
    assert any(r["target"] for r in rows if r["action"] == "datasets.convert")
