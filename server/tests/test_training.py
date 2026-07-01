"""SPDX-License-Identifier: Apache-2.0 - training config + launcher."""
import time

import yaml

from server.services import training as svc
from server.settings import Settings


def test_build_config_applies_overrides():
    cfg = svc.build_config("llada2-mini", {"train": {"lr": 5e-5, "global_batch_size": 4}})
    assert cfg["train"]["lr"] == 5e-5
    assert cfg["train"]["global_batch_size"] == 4
    # untouched keys keep defaults
    assert cfg["train"]["block_diffusion_mode"] is True
    assert cfg["data"]["noise_range_low"] == 0.3


def test_validate_config():
    ok, _ = svc.validate_config(svc.DEFAULT_CONFIG)
    assert ok is True
    bad = svc.build_config(overrides={"model": {"model_path": ""}})
    ok, errs = svc.validate_config(bad)
    assert ok is False and any("model_path" in e for e in errs)
    noisy = svc.build_config(overrides={"data": {"noise_range_low": 0.9, "noise_range_high": 0.3}})
    _, errs = svc.validate_config(noisy)
    assert any("noise_range_low" in e for e in errs)


def test_yaml_roundtrip(tmp_path):
    cfg = svc.build_config()
    p = svc.write_config_yaml(cfg, tmp_path / "c.yaml")
    loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert loaded["train"]["block_size"] == cfg["train"]["block_size"]
    assert loaded["data"]["text_keys"] == "messages"


def test_build_torchrun_argv_mirrors_train_sh():
    s = Settings()
    s.nnodes = 1
    argv = svc.build_torchrun_argv("/path/cfg.yaml", s)
    assert argv[0] == "torchrun"
    assert "--standalone" in argv
    assert any(a.startswith("--nproc-per-node=") for a in argv)
    assert argv[-2].endswith("train_llada2_bd.py")
    assert argv[-1] == "/path/cfg.yaml"
    # multi-node uses rdzv instead of standalone
    s.nnodes = 2
    argv2 = svc.build_torchrun_argv("/path/cfg.yaml", s)
    assert "--standalone" not in argv2
    assert any(a.startswith("--rdzv_endpoint=") for a in argv2)


def test_parse_metric_line():
    m = svc.parse_metric_line("{'loss': 2.34, 'lr': 1e-05, 'step': 42}")
    assert m["loss"] == 2.34 and m["step"] == 42
    assert svc.parse_metric_line("nothing useful here") is None


def test_estimate_vram():
    est = svc.estimate_vram_gb(16.0, fsdp_shards=4, offload=True)
    assert est["total_gb"] > 0 and est["per_gpu_gb"] > 0
    assert est["per_gpu_gb"] < est["total_gb"]


def test_config_route(client):
    r = client.get("/api/training/config").json()["data"]
    assert "llada2-mini" in r["presets"]
    assert "block_diffusion_mode" in r["diffusion_keys"]


def test_start_dry_run_route(client):
    r = client.post("/api/training/start", json={"preset": "llada2-mini",
                                                 "overrides": {"train": {"output_dir": "./out"}},
                                                 "dry_run": True})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["dry_run"] is True
    jid = data["job_id"]
    for _ in range(100):
        j = client.get(f"/api/jobs/{jid}").json()
        if j.get("state") in ("done", "error"):
            break
        time.sleep(0.05)
    assert j["state"] == "done", j
    assert j["result"]["argv"][0] == "torchrun"
    # run registered in history
    runs = client.get("/api/training/runs").json()["data"]
    assert any(r["job_id"] == jid for r in runs)
