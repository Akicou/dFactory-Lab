"""SPDX-License-Identifier: Apache-2.0 - MoE merge/split round-trip on a tiny fixture."""
import json
from pathlib import Path

import numpy as np
import pytest

safetensors = pytest.importorskip("safetensors")
from safetensors.numpy import save_file as np_save_file

from server.services import moe

NUM_LAYERS, NUM_EXPERTS, FIRST_K = 2, 4, 0
PROJS = ["gate_proj", "up_proj", "down_proj"]


def _separate_state_dict():
    sd = {}
    for L in range(NUM_LAYERS):
        for e in range(NUM_EXPERTS):
            for p in PROJS:
                sd[f"model.layers.{L}.mlp.experts.{e}.{p}.weight"] = np.random.randn(4, 8).astype("float32")
    sd["model.embed_tokens.weight"] = np.random.randn(10, 4).astype("float32")
    return sd


def _write_model_dir(d: Path, sd):
    d.mkdir(parents=True, exist_ok=True)
    np_save_file(sd, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps({
        "num_hidden_layers": NUM_LAYERS, "num_experts": NUM_EXPERTS,
        "first_k_dense_replace": FIRST_K, "vocab_size": 10, "hidden_size": 4,
        "model_type": "llada2_moe_veomni",
    }))


def test_merge_state_dict_shape():
    sd = _separate_state_dict()
    merged = moe.merge_state_dict(sd, num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS,
                                  first_k_dense_replace=FIRST_K)
    # each (layer, proj) collapses to one stacked tensor [num_experts, 4, 8]
    for L in range(NUM_LAYERS):
        for p in PROJS:
            t = merged[f"model.layers.{L}.mlp.experts.{p}"]
            assert t.shape[0] == NUM_EXPERTS
    # passthrough survives
    assert "model.embed_tokens.weight" in merged
    # separate keys are gone
    assert not any("experts.0.gate_proj" in k for k in merged)


def test_split_reverses_merge():
    sd = _separate_state_dict()
    merged = moe.merge_state_dict(sd, num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS,
                                  first_k_dense_replace=FIRST_K)
    split = moe.split_state_dict(merged, num_experts=NUM_EXPERTS)
    assert set(split) == set(_separate_state_dict())


def test_convert_roundtrip_and_format(tmp_path):
    separate = tmp_path / "separate"
    _write_model_dir(separate, _separate_state_dict())
    assert moe.detect_format(separate) == "separate_expert"

    merged_dir = tmp_path / "merged"
    summary = moe.convert(separate, merged_dir, "merge")
    assert summary["mode"] == "merge"
    assert moe.detect_format(merged_dir) == "merged_expert"
    assert (merged_dir / "config.json").is_file()

    split_dir = tmp_path / "split"
    moe.convert(merged_dir, split_dir, "split")
    assert moe.detect_format(split_dir) == "separate_expert"


def test_catalog_route(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.json()["data"]
    ids = {m["id"] for m in data}
    assert {"llada2-mini", "llada2-flash"} <= ids
    mini = next(m for m in data if m["id"] == "llada2-mini")
    assert mini["meta"]["num_experts"] == 256
    assert mini["repo_id"] == "inclusionAI/LLaDA2.0-mini-preview"


def test_local_inventory_and_merge_route(tmp_path, client):
    # write a separate-expert fixture into the managed models dir
    models_dir = Path(client.app.state.settings.data_dir) / "models"
    separate = models_dir / "tiny"
    _write_model_dir(separate, _separate_state_dict())

    inv = client.get("/api/models/local").json()["data"]
    assert any(m["id"] == "tiny" and m["format"] == "separate_expert" for m in inv)

    out = models_dir / "tiny-merged"
    body = {"input_dir": str(separate), "output_dir": str(out)}
    r = client.post("/api/models/merge", json=body)
    assert r.status_code == 200
    jid = r.json()["data"]["job_id"]

    import time
    for _ in range(100):
        j = client.get(f"/api/jobs/{jid}").json()
        if j.get("state") in ("done", "error"):
            break
        time.sleep(0.05)
    assert j["state"] == "done", j
    assert moe.detect_format(out) == "merged_expert"
