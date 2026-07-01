"""SPDX-License-Identifier: Apache-2.0 - export pipeline (split + modeling copy + manifest)."""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file as np_save_file

from server.services import export as svc
from server.services import moe

NUM_LAYERS, NUM_EXPERTS, FIRST_K = 2, 4, 0
PROJS = ["gate_proj", "up_proj", "down_proj"]


def _merged_state_dict():
    """Already-merged expert tensors (what a trained checkpoint looks like)."""
    sd = {}
    for L in range(NUM_LAYERS):
        for p in PROJS:
            sd[f"model.layers.{L}.mlp.experts.{p}"] = np.random.randn(NUM_EXPERTS, 4, 8).astype("float32")
    sd["model.embed_tokens.weight"] = np.random.randn(10, 4).astype("float32")
    return sd


def _write_cfg(d: Path):
    (d / "config.json").write_text(json.dumps({
        "num_hidden_layers": NUM_LAYERS, "num_experts": NUM_EXPERTS,
        "first_k_dense_replace": FIRST_K, "vocab_size": 10, "hidden_size": 4,
        "model_type": "llada2_moe_veomni",
    }))


def _make_ckpt(tmp_path: Path):
    """Fake trained checkpoint dir: output_dir/checkpoints/global_step_10/hf_ckpt."""
    hf = tmp_path / "out" / "checkpoints" / "global_step_10" / "hf_ckpt"
    hf.mkdir(parents=True)
    np_save_file(_merged_state_dict(), str(hf / "model.safetensors"))
    _write_cfg(hf)
    (hf / "tokenizer.json").write_text("{}")
    return tmp_path / "out", hf


def _make_base(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "modeling_llada2_moe.py").write_text("# modeling code")
    (base / "configuration_llada2_moe.py").write_text("# config code")
    return base


def test_find_latest_checkpoint(tmp_path):
    out, hf = _make_ckpt(tmp_path)
    # add an older step that must NOT be picked
    older = tmp_path / "out" / "checkpoints" / "global_step_2" / "hf_ckpt"
    older.mkdir(parents=True)
    _write_cfg(older)
    found = svc.find_latest_checkpoint(out)
    assert found == hf


def test_export_run_roundtrip(tmp_path):
    out, _ = _make_ckpt(tmp_path)
    base = _make_base(tmp_path)
    export_dir = tmp_path / "exports" / "m1"

    logs = []
    res = svc.export_run(source=out, original_base_dir=base, export_dir=export_dir,
                         update=lambda p, m: logs.append(m))

    assert moe.detect_format(export_dir) == "separate_expert"
    assert (export_dir / "modeling_llada2_moe.py").is_file()
    assert (export_dir / "manifest.json").is_file()
    assert res["modeling_files_copied"] == ["modeling_llada2_moe.py", "configuration_llada2_moe.py"]
    assert res["missing"] == []  # complete
    assert "export ready" in logs[-1]


def test_export_route(tmp_path, client):
    out, _ = _make_ckpt(tmp_path)
    base = _make_base(tmp_path)
    s = client.app.state.settings
    # stage under managed dirs so path jailing passes
    managed_out = Path(s.data_dir) / "checkpoints" / "out"
    managed_base = Path(s.data_dir) / "models" / "base"
    shutil.copytree(out, managed_out)
    shutil.copytree(base, managed_base)

    body = {"source": str(managed_out), "original_base_dir": str(managed_base), "export_name": "run1"}
    r = client.post("/api/export", json=body)
    assert r.status_code == 200
    jid = r.json()["data"]["job_id"]
    for _ in range(100):
        j = client.get(f"/api/jobs/{jid}").json()
        if j.get("state") in ("done", "error"):
            break
        time.sleep(0.05)
    assert j["state"] == "done", j
    assert j["result"]["modeling_files_copied"]
