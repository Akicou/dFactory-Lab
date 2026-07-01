"""SPDX-License-Identifier: Apache-2.0
Model hub service (Checklist MH-*): catalog of supported dLLMs, local inventory
with format detection, and HuggingFace download as a job.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

from ..logging_config import get_logger
from ..settings import repo_root
from . import moe

log = get_logger(__name__)

# Supported dLLMs (from dFactory README) + their vendored config path.
SUPPORTED_MODELS = [
    {
        "id": "llada2-mini",
        "repo_id": "inclusionAI/LLaDA2.0-mini-preview",
        "name": "LLaDA2.0-mini",
        "size": "16B",
        "config_path": "configs/model_configs/llada2_mini",
    },
    {
        "id": "llada2-flash",
        "repo_id": "inclusionAI/LLaDA2.0-flash-preview",
        "name": "LLaDA2.0-flash",
        "size": "100B",
        "config_path": "configs/model_configs/llada2_flash",
    },
]

_MODELING_FILES = moe._MODELING_FILES


def catalog() -> list[dict]:
    """Supported models with metadata parsed from the vendored config.json."""
    out = []
    for m in SUPPORTED_MODELS:
        item = dict(m)
        cfg_dir = repo_root() / m["config_path"]
        cfg_file = cfg_dir / "config.json"
        meta: dict[str, Any] = {}
        if cfg_file.is_file():
            try:
                c = json.loads(cfg_file.read_text(encoding="utf-8"))
                meta = {k: c.get(k) for k in (
                    "num_hidden_layers", "num_experts", "num_experts_per_tok",
                    "hidden_size", "vocab_size", "pad_token_id", "first_k_dense_replace",
                    "moe_intermediate_size", "score_function")}
            except Exception as exc:  # noqa: BLE001
                log.warning("models.catalog.parse_failed", model=m["id"], error=str(exc))
        item["meta"] = meta
        out.append(item)
    return out


def _scan_model_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    dirs = []
    for entry in root.iterdir():
        if entry.is_dir() and (entry / "config.json").is_file():
            dirs.append(entry)
        # one level of nesting (e.g. <root>/<repo>/<sub>)
        elif entry.is_dir():
            for sub in entry.iterdir():
                if sub.is_dir() and (sub / "config.json").is_file():
                    dirs.append(sub)
    return dirs


def list_local(models_dir: str | Path) -> list[dict]:
    """Inventory downloaded/local models with detected format + size."""
    root = Path(models_dir)
    out = []
    for d in _scan_model_dirs(root):
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        fmt = "unknown"
        meta: dict = {}
        try:
            fmt = moe.detect_format(d)
        except Exception as exc:  # noqa: BLE001
            log.debug("models.detect_failed", dir=str(d), error=str(exc))
        try:
            meta = moe.read_moe_config(d)
        except Exception:  # noqa: BLE001
            pass
        out.append({
            "id": d.name, "path": str(d), "format": fmt,
            "size_bytes": size, "has_modeling": any((d / f).is_file() for f in _MODELING_FILES),
            "num_experts": meta.get("num_experts"),
        })
    return out


def download_model(
    repo_id: str,
    dest_dir: str | Path,
    *,
    token: Optional[str] = None,
    nest_under_repo_name: bool = True,
    update: Optional[Callable] = None,
) -> dict:
    """Download a snapshot. Resolves the true output path (mirroring
    download_hf_model.py, which nests under repo_id.split('/')[1])."""
    from huggingface_hub import snapshot_download
    dest = Path(dest_dir)
    if nest_under_repo_name:
        # mirror scripts/download_hf_model.py so the path matches engine expectations
        sub = repo_id.split("/")[-1]
        real_dest = dest / sub
    else:
        real_dest = dest
    real_dest.mkdir(parents=True, exist_ok=True)
    if update:
        update(0.02, f"downloading {repo_id} -> {real_dest}")
    snapshot_download(
        repo_id=repo_id, local_dir=str(real_dest), token=token or None,
        etag_timeout=30,
    )
    if update:
        update(1.0, "download complete")
    return {"repo_id": repo_id, "path": str(real_dest)}


def merge_model(input_dir: str | Path, output_dir: str | Path,
                update: Optional[Callable] = None) -> dict:
    return moe.convert(input_dir, output_dir, "merge", on_log=_update_to_log(update))


def split_model(input_dir: str | Path, output_dir: str | Path,
                update: Optional[Callable] = None) -> dict:
    return moe.convert(input_dir, output_dir, "split", on_log=_update_to_log(update))


def _update_to_log(update):
    if not update:
        return None
    i = [0]

    def _log(msg: str) -> None:
        i[0] += 1
        update(min(0.95, i[0] * 0.1), msg)
    return _log
