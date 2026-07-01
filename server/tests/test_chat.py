"""SPDX-License-Identifier: Apache-2.0 — chat / inference routes (mock backend)."""
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
    body = {"base_dir": "/tmp/b", "tuned_dir": "/tmp/t",
            "messages": [{"role": "user", "content": "q"}]}
    r = client.post("/api/chat/compare", json=body).json()["data"]
    assert "base" in r and "tuned" in r
    assert "text" in r["base"]


def test_show_unmasking(client):
    body = {"messages": [{"role": "user", "content": "x"}],
            "params": {"diffusion_steps": 5, "show_unmasking": True}}
    res = client.post("/api/chat/completions", json=body).json()["data"]
    assert res["details"]["unmasking"] is not None
    assert len(res["details"]["unmasking"]) == 5
