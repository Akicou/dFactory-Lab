"""SPDX-License-Identifier: Apache-2.0
Export / merge-back / packaging pipeline (Checklist E-*).

Automates the post-training steps from the dFactory README (steps 6-7 + packaging):
  1. locate the saved HF checkpoint under output_dir/checkpoints/global_step_XXX/hf_ckpt/
  2. split merged-expert weights back to separate-expert (services.moe)
  3. copy modeling_llada2_moe.py (+ config code) from the ORIGINAL base model dir
  4. verify completeness + write a sha256 manifest
The result dir is a runnable HF-format model.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Callable, Optional

from ..logging_config import get_logger
from . import moe

log = get_logger(__name__)

_STEP_RE = re.compile(r"global_step_(\d+)$")
_REQUIRED_FILES = ["config.json"]
_TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                    "tokenizer.model", "added_tokens.json"]


def find_latest_checkpoint(output_dir: str | Path) -> Optional[Path]:
    """Find output_dir/checkpoints/global_step_<N>/hf_ckpt with the largest N."""
    root = Path(output_dir) / "checkpoints"
    if not root.is_dir():
        # also accept being handed the checkpoint dir directly
        direct = Path(output_dir)
        if (direct / "config.json").is_file() and direct.glob("*.safetensors"):
            return direct
        return None
    candidates = []
    for ckpt in root.glob("global_step_*"):
        m = _STEP_RE.match(ckpt.name)
        hf = ckpt / "hf_ckpt"
        if m and hf.is_dir():
            candidates.append((int(m.group(1)), hf))
    if not candidates:
        return None
    return max(candidates)[1]


def export_run(
    *,
    source: str | Path,
    original_base_dir: str | Path,
    export_dir: str | Path,
    update: Optional[Callable] = None,
    write_manifest: bool = True,
) -> dict:
    """Full export: split + copy modeling + verify + manifest.

    *source* is either a training output_dir (auto-locates the checkpoint) or the
    checkpoint dir itself.
    """
    src = Path(source)

    def _log(p: float, msg: str) -> None:
        log.info("export", msg=msg)
        if update:
            update(p, msg)

    ckpt = find_latest_checkpoint(src)
    if ckpt is None:
        # fall back: treat source itself as a merged checkpoint dir
        if src.is_dir() and (src / "config.json").is_file():
            ckpt = src
        else:
            raise FileNotFoundError(f"No checkpoint found under {src}")
    _log(0.1, f"checkpoint: {ckpt}")

    # split merged -> separate
    out = Path(export_dir)
    _log(0.3, "splitting merged experts -> separate")
    moe.convert(ckpt, out, "split", on_log=lambda m: _log(0.4, m))

    # copy modeling file from original base
    _log(0.7, f"copying modeling code from {original_base_dir}")
    copied = moe.copy_modeling_file(original_base_dir, out)

    # verify
    missing = verify_completeness(out)
    if missing:
        _log(0.9, f"WARNING: export may be incomplete; missing: {missing}")

    # manifest
    manifest = write_sha256_manifest(out) if write_manifest else None
    _log(1.0, f"export ready: {out}")
    return {
        "export_dir": str(out), "checkpoint": str(ckpt),
        "modeling_files_copied": copied, "missing": missing,
        "manifest": manifest,
    }


def verify_completeness(export_dir: str | Path) -> list[str]:
    """Return a list of missing files for a runnable HF model dir."""
    d = Path(export_dir)
    missing = []
    for f in _REQUIRED_FILES:
        if not (d / f).is_file():
            missing.append(f)
    if not any((d / f).is_file() for f in _TOKENIZER_FILES):
        missing.append("tokenizer (none of " + "/".join(_TOKENIZER_FILES) + ")")
    if not any(d.glob("*.safetensors")):
        missing.append("*.safetensors")
    if not (d / "modeling_llada2_moe.py").is_file():
        missing.append("modeling_llada2_moe.py")
    return missing


def write_sha256_manifest(export_dir: str | Path) -> dict:
    """sha256 per file (skip the manifest itself). Best-effort for huge dirs."""
    import json
    d = Path(export_dir)
    manifest: dict[str, str] = {}
    for f in sorted(d.iterdir()):
        if f.is_file() and f.name != "manifest.json":
            manifest[f.name] = _sha256(f)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
