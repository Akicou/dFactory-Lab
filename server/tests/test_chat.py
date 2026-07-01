"""SPDX-License-Identifier: Apache-2.0 - chat / inference routes (mock backend)."""
import json

from server.services import inference as svc


def test_params_route(client):
    r = client.get("/api/chat/params").json()["data"]
    assert r["defaults"]["diffusion_steps"] == 32
    assert "repetition_penalty is not supported" in r["notes"]


def test_completions_uses_diffusion_params(client):
    body = {"model_dir": "/tmp/m", "session": "s1",
            "messages": [{"role": "user", "content": "hello"}],
            "params": {"diffusion_steps": 64, "mask_schedule": "cosine"}}
    r = client.post("/api/chat/completions", json=body)
    assert r.status_code == 200
    data = r.json()["data"]
    assert "64 steps" in data["response"]
    assert data["details"]["diffusion_steps"] == 64
    # history recorded
    hist = client.get("/api/chat/history?session=s1").json()["data"]
    assert hist and hist[-1]["response"] == data["response"]


def test_compare_route(client):
    # No loaded servers => both sides fall back to the global mock backend.
    body = {"left_id": "", "right_id": "",
            "messages": [{"role": "user", "content": "q"}]}
    r = client.post("/api/chat/compare", json=body).json()["data"]
    assert "left" in r and "right" in r
    assert "text" in r["left"]


def test_load_eject_simulated(client, monkeypatch):
    monkeypatch.setenv("DFACTORY_LAB_SGLANG_SIMULATE", "true")
    from server.settings import reset_settings_cache
    reset_settings_cache()
    # rebuild the app's manager with the simulate setting
    from server.services.servers import ServerManager
    from server.settings import get_settings
    client.app.state.servers = ServerManager(get_settings())

    # model_path must resolve under <data_dir>/models
    from pathlib import Path
    model_path = str(Path(client.app.state.settings.data_dir) / "models" / "LLaDA2.1-mini")
    r = client.post("/api/chat/load", json={"model_path": model_path}).json()["data"]
    assert "id" in r, r
    sid = r["id"]
    loaded = client.get("/api/chat/loaded").json()["data"]
    assert any(s["id"] == sid and s["state"] == "ready" for s in loaded)

    # chat routed to the loaded (simulated) server
    res = client.post("/api/chat/completions", json={
        "server_id": sid, "messages": [{"role": "user", "content": "hi"}]}).json()["data"]
    assert res["response"]

    assert client.post("/api/chat/eject", json={"id": sid}).json()["data"]["ok"] is True
    assert client.get("/api/chat/loaded").json()["data"] == []


def test_sglang_backend_maps_openai():
    """SGLangBackend forwards OpenAI sampling knobs and unwraps the response."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["max_tokens"] == 128 and body["messages"][-1]["content"] == "hi"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "pong"}}],
            "usage": {"completion_tokens": 3}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    be = svc.SGLangBackend("http://sglang:30000", client=client)
    out = be.generate([{"role": "user", "content": "hi"}],
                      svc.DiffusionParams(max_new_tokens=128))
    assert out["text"] == "pong" and out["tokens_generated"] == 3
    assert out["backend"] == "sglang"


def test_show_unmasking(client):
    body = {"messages": [{"role": "user", "content": "x"}],
            "params": {"diffusion_steps": 5, "show_unmasking": True}}
    res = client.post("/api/chat/completions", json=body).json()["data"]
    assert res["details"]["unmasking"] is not None
    assert len(res["details"]["unmasking"]) == 5
