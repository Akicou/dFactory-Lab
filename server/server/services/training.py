"""SPDX-License-Identifier: Apache-2.0
Training engine (Checklist T-*): config builder for the real block-diffusion SFT
YAML schema, a torchrun launcher that mirrors train.sh, log-line metrics parsing,
resume, training-run history, and a rough VRAM precheck.

The actual run needs the VeOmni runtime + torch + GPU + model weights; this module
constructs everything correctly and executes torchrun via the safe subprocess
runner. When torchrun is absent it records a clear 127 error instead of crashing.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from ..logging_config import get_logger
from ..settings import engine_root, repo_root

log = get_logger(__name__)

# ── defaults mirrored verbatim from configs/sft/llada2_mini_bd_sft.yaml ──────
DEFAULT_CONFIG = {
    "model": {
        "config_path": "./configs/model_configs/llada2_mini",
        "model_path": "./LLaDA2.0-mini-preview-moe-merge",
        "tokenizer_path": "./LLaDA2.0-mini-preview-moe-merge",
        "attn_implementation": "sdpa",
        "moe_implementation": "fused",
    },
    "data": {
        "train_path": "./gsm8k_datasets/gsm8k_train.jsonl",
        "data_type": "conversation",
        "datasets_type": "mapping",
        "dataloader_type": "native",
        "max_seq_len": 2048,
        "text_keys": "messages",
        "noise_range_low": 0.3,
        "noise_range_high": 0.8,
        "num_workers": 16,
    },
    "train": {
        "output_dir": "./llada2_mini_bd_sft_outputs",
        "data_parallel_mode": "fsdp2",
        "tensor_parallel_size": 1,
        "ulysses_parallel_size": 1,
        "expert_parallel_size": 1,
        "global_batch_size": 8,
        "micro_batch_size": 1,
        "num_train_epochs": 1,
        "rmpad": False,
        "rmpad_with_pos_ids": False,
        "bsz_warmup_ratio": 0.007,
        "dyn_bsz_margin": 0,
        "dyn_bsz_buffer_size": 200,
        "optimizer": "adamw",
        "beta1": 0.9,
        "beta2": 0.999,
        "lr": 1.0e-5,
        "lr_warmup_ratio": 0.03,
        "lr_decay_style": "cosine",
        "lr_decay_ratio": 1.0,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "enable_mixed_precision": True,
        "enable_gradient_checkpointing": True,
        "enable_full_shard": True,
        "enable_fsdp_offload": True,
        "enable_activation_offload": False,
        "init_device": "meta",
        "broadcast_model_weights_from_rank0": True,
        "enable_full_determinism": False,
        "empty_cache_steps": 500,
        "ckpt_manager": "dcp",
        "load_checkpoint_path": "",
        "save_epochs": 1,
        "save_hf_weights": True,
        "block_diffusion_mode": True,
        "block_size": 32,
        "same_token_labels": True,
        "use_wandb": False,
        "log_steps": 1,
    },
}

def _flash_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["model"].update({
        "config_path": "./configs/model_configs/llada2_flash",
        "model_path": "./LLaDA2.0-flash-preview-moe-merge",
        "tokenizer_path": "./LLaDA2.0-flash-preview-moe-merge"})
    cfg["train"].update({"output_dir": "./llada2_flash_bd_sft_outputs", "global_batch_size": 16})
    return cfg


PRESETS = {
    "llada2-mini": DEFAULT_CONFIG,
    "llada2-flash": _flash_config(),
}

REQUIRED_KEYS = {
    "model": {"model_path", "config_path", "tokenizer_path"},
    "data": {"train_path", "max_seq_len"},
    "train": {"output_dir", "global_batch_size", "lr", "block_diffusion_mode"},
}

# diffusion-specific knobs surfaced first-class in the UI
DIFFUSION_KEYS = ("noise_range_low", "noise_range_high", "block_diffusion_mode",
                  "block_size", "same_token_labels")


def build_config(preset: str = "llada2-mini", overrides: Optional[dict] = None) -> dict:
    """Deep-merge user overrides onto a preset."""
    base = json.loads(json.dumps(PRESETS.get(preset, DEFAULT_CONFIG)))  # deep copy
    overrides = overrides or {}
    for section, vals in overrides.items():
        if section not in base or not isinstance(vals, dict):
            base[section] = vals
            continue
        base[section].update(vals)
    return base


def validate_config(cfg: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for section, keys in REQUIRED_KEYS.items():
        if section not in cfg:
            errors.append(f"missing section: {section}")
            continue
        for k in keys:
            if not cfg[section].get(k):
                errors.append(f"{section}.{k} is required")
    nrl = cfg.get("data", {}).get("noise_range_low")
    nrh = cfg.get("data", {}).get("noise_range_high")
    if nrl is not None and nrh is not None and nrl > nrh:
        errors.append("data.noise_range_low must be <= noise_range_high")
    if cfg.get("train", {}).get("global_batch_size", 0) <= 0:
        errors.append("train.global_batch_size must be > 0")
    return (not errors), errors


def write_config_yaml(cfg: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


# ── torchrun launch (mirrors train.sh) ───────────────────────────────────────
def _detect_gpu_count() -> int:
    for fn in ("nvidia-smi", "rocm-smi"):
        if shutil.which(fn):
            try:
                import subprocess
                out = subprocess.run([fn, "--list-gpus" if fn == "nvidia-smi" else "--showproductname"],
                                     capture_output=True, text=True, timeout=5)
                if fn == "nvidia-smi":
                    return max(1, len([l for l in out.stdout.splitlines() if l.strip()]))
            except Exception:  # noqa: BLE001
                pass
    return 1


def build_torchrun_argv(config_path: str | Path, settings) -> list[str]:
    """Construct the torchrun command exactly like train.sh."""
    nnodes = max(1, settings.nnodes)
    nproc = settings.nproc_per_node if settings.nproc_per_node > 0 else _detect_gpu_count()
    argv = ["torchrun", f"--nnodes={nnodes}", f"--nproc-per-node={nproc}",
            f"--node-rank={settings.node_rank}"]
    if nnodes == 1:
        argv.append("--standalone")
    else:
        argv.append(f"--rdzv_endpoint={settings.master_addr}:{settings.master_port}")
    argv.append(str(engine_root() / "tasks" / "train_llada2_bd.py"))
    argv.append(str(config_path))
    return argv


def build_env(settings) -> dict[str, str]:
    env = settings.engine_env()
    veomni = engine_root() / "VeOmni"
    env["PYTHONPATH"] = str(veomni) + os.pathsep + os.environ.get("PYTHONPATH", "")
    return env


# ── metrics parsing ──────────────────────────────────────────────────────────
_METRIC_RES = [
    re.compile(r"'?loss'?\s*[:=]\s*([0-9.eE+-]+)"),
    re.compile(r"'?lr'?\s*[:=]\s*([0-9.eE+-]+)"),
    re.compile(r"'?(?:step|global_step)'?\s*[:=]?\s*(\d+)"),
    re.compile(r"tokens/s\s*[:=]?\s*([0-9.eE+-]+)", re.I),
]


def parse_metric_line(line: str) -> Optional[dict]:
    """Pull loss/lr/step/tokens_per_s from a trainer log line, else None."""
    m = {rx.search(line) for rx in _METRIC_RES}
    found = [x for x in m if x]
    if not found:
        return None
    out: dict[str, Any] = {}
    for x in found:
        txt = x.group(0).lower()
        val = x.group(1)
        try:
            num = float(val)
        except ValueError:
            continue
        if "loss" in txt:
            out["loss"] = num
        elif "lr" in txt and "url" not in txt:
            out["lr"] = num
        elif "tokens/s" in txt:
            out["tokens_per_s"] = num
        elif "step" in txt:
            out["step"] = int(num)
    return out or None


# ── VRAM precheck (rough) ────────────────────────────────────────────────────
def estimate_vram_gb(num_params_b: float, *, fsdp_shards: int = 1, offload: bool = True) -> dict:
    """Rough total-and-per-GPU VRAM estimate for bf16 AdamW training (bytes/param)."""
    bytes_per_param = 2  # bf16 weights
    bytes_per_param += 2  # bf16 grads
    bytes_per_param += 4 + 4  # adam m, v in fp32
    activations = 2 * 2  # bf16 fwd+bwd activations (very rough)
    total_gb = num_params_b * (bytes_per_param + activations)
    per_gpu = total_gb / max(1, fsdp_shards)
    if offload:
        per_gpu *= 0.45  # optimizer/offload shaves resident GPU memory
    return {"total_gb": round(total_gb, 1), "per_gpu_gb": round(per_gpu, 1),
            "note": "rough estimate; FSDP + CPU offload reduce per-GPU resident memory"}


# ── run orchestration ────────────────────────────────────────────────────────
def register_run(db_path: Path, *, job_id: str, model_id: str, dataset_id: str,
                 config_path: str, output_dir: str) -> str:
    from .. import db as _db
    rid = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + job_id[:6]
    with _db.connection(db_path) as conn:
        conn.execute(
            """INSERT INTO training_runs (id, job_id, model_id, dataset_id, config_path,
                                          output_dir, status, created_at)
               VALUES (?,?,?,?,?,?, 'running', ?)""",
            (rid, job_id, model_id, dataset_id, config_path, output_dir,
             datetime.now(timezone.utc).isoformat()))
    return rid


def launch(config_path: str | Path, settings, *, update: Optional[Callable] = None,
           cancel_event=None, dry_run: bool = False) -> dict:
    """Run torchrun for the given config. Returns the process result (or the
    planned argv/env when dry_run=True)."""
    from .. import subprocess_util
    argv = build_torchrun_argv(config_path, settings)
    env = build_env(settings)
    display = " ".join(shlex.quote(a) for a in argv)
    if dry_run:
        return {"dry_run": True, "argv": argv, "env": env, "command": display}

    log_path = Path(settings.data_dir) / "logs" / f"train-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.log"
    last_metric: dict = {}

    def on_line(line: str) -> None:
        metric = parse_metric_line(line)
        if metric:
            last_metric.update(metric)
            if update:
                update(0.0, f"step {metric.get('step', '?')}", **{"metric": metric})

    if update:
        update(0.01, f"launching: {display}")
    res = subprocess_util.run_tracked(
        argv, job_id=getattr(cancel_event, "name", "train") if cancel_event else "train",
        cwd=str(engine_root()), env=env, log_path=log_path,
        on_line=on_line, cancel_event=cancel_event,
    )
    res["last_metric"] = last_metric
    res["log_path"] = str(log_path)
    return res


def finetune(*, model_source: str, dataset_path: str, settings,
             preset: str = "llada2-mini", overrides: Optional[dict] = None,
             update: Optional[Callable] = None, dry_run: bool = True) -> dict:
    """One-click finetune: merge the model into training format if needed, build
    the config, and launch torchrun. This is the flow the UI drives so the user
    never merges manually (Checklist: merge happens when you run a finetune)."""
    from . import moe

    def log(p: float, msg: str) -> None:
        if update:
            update(p, msg)

    src = Path(model_source)
    fmt = moe.detect_format(src)
    if fmt == "separate_expert":
        merged = src.parent / (src.name + "-merged")
        log(0.05, "merging MoE experts for training")
        moe.convert(src, merged, "merge", on_log=lambda m: log(0.12, m))
        model_path = str(merged)
        log(0.4, "model merged")
    else:
        model_path = str(src)
        log(0.05, f"model already {fmt or 'unknown'}, skipping merge")

    cfg = build_config(preset, overrides)
    cfg["model"]["model_path"] = model_path
    cfg["model"]["tokenizer_path"] = model_path
    cfg["data"]["train_path"] = dataset_path
    ok, errs = validate_config(cfg)
    if not ok:
        raise ValueError("; ".join(errs))
    cfg_path = write_config_yaml(
        cfg, Path(settings.data_dir) / "configs" / f"finetune-{preset}.yaml")
    log(0.45, "config ready")

    # map launch's 0..1 progress into the 0.45..1.0 band
    def _launch_update(p: float, msg: str, **_: object) -> None:
        log(0.45 + 0.55 * max(0.0, p), msg)

    res = launch(cfg_path, settings, update=_launch_update, dry_run=dry_run)
    return {"config_path": str(cfg_path), "model_path": model_path,
            "merged": fmt == "separate_expert", "launch": res}
