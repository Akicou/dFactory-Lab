"""SPDX-License-Identifier: Apache-2.0
MoE expert merge / split (Checklist M-*, faithfully mirrors scripts/moe_convertor.py).

The upstream ``moe_convertor.py`` implements two exact tensor transforms; we
re-implement them here so the Lab can convert weights **without** the VeOmni
runtime (the script's CLI pulls in ``veomni.models.build_tokenizer/save_model_weights``).
The math is identical:

  merge : model.layers.{L}.mlp.experts.{0..N-1}.{gate,up,down}_proj.weight
          -> model.layers.{L}.mlp.experts.{gate,up,down}_proj   (stacked [N, ...])
  split : the reverse.

torch is used when present (handles bfloat16); otherwise numpy (float types) so
the round-trip is unit-testable without a torch install.
"""
from __future__ import annotations

import glob
import json
import re
import shutil
from pathlib import Path
from typing import Any

from ..logging_config import get_logger

log = get_logger(__name__)

_PROJ_TYPES = ["gate_proj", "up_proj", "down_proj"]
_MERGED_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(" + "|".join(_PROJ_TYPES) + r")$")
_MODELING_FILES = ["modeling_llada2_moe.py", "configuration_llada2_moe.py", "__init__.py"]


def _backend() -> str:
    try:
        import torch  # noqa: F401
        return "torch"
    except Exception:  # noqa: BLE001
        return "numpy"


def _stack(tensors: list[Any]) -> Any:
    if hasattr(tensors[0], "dim"):  # torch
        import torch
        return torch.stack(tensors, dim=0)
    import numpy as np
    return np.stack(tensors, axis=0)


def load_state_dict(input_dir: str | Path) -> tuple[dict[str, Any], str]:
    """Load every ``*.safetensors`` shard in *input_dir* into one dict."""
    files = sorted(glob.glob(str(Path(input_dir) / "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {input_dir}")
    sd: dict[str, Any] = {}
    backend = _backend()
    if backend == "torch":
        from safetensors.torch import safe_open
        for f in files:
            with safe_open(f, framework="pt", device="cpu") as h:
                for k in h.keys():
                    sd[k] = h.get_tensor(k)
    else:
        from safetensors import safe_open
        for f in files:
            with safe_open(f, framework="np", device="cpu") as h:
                for k in h.keys():
                    sd[k] = h.get_tensor(k)
    return sd, backend


def save_state_dict(output_dir: str | Path, sd: dict[str, Any], backend: str) -> Path:
    """Save *sd* to a single ``model.safetensors`` in *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "model.safetensors"
    if backend == "torch":
        from safetensors.torch import save_file
        save_file(sd, str(target))
    else:
        from safetensors.numpy import save_file
        save_file({k: v for k, v in sd.items()}, str(target))
    return target


def merge_state_dict(sd: dict[str, Any], *, num_layers: int, num_experts: int,
                     first_k_dense_replace: int) -> dict[str, Any]:
    """Separate-expert -> merged-expert (mirrors moe_convertor.moe_merge)."""
    new: dict[str, Any] = {}
    processed: set[str] = set()
    for layer in range(first_k_dense_replace, num_layers):
        for proj in _PROJ_TYPES:
            experts = []
            for e in range(num_experts):
                key = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
                if key not in sd:
                    raise KeyError(f"merge: missing expert weight {key}")
                experts.append(sd[key])
                processed.add(key)
            new[f"model.layers.{layer}.mlp.experts.{proj}"] = _stack(experts)
    for k, v in sd.items():
        if k not in processed:
            new[k] = v
    return new


def split_state_dict(sd: dict[str, Any], *, num_experts: int) -> dict[str, Any]:
    """Merged-expert -> separate-expert (mirrors moe_convertor.split_moe_experts)."""
    new: dict[str, Any] = {}
    for k, v in sd.items():
        m = _MERGED_RE.match(k)
        if m:
            layer, proj = m.group(1), m.group(2)
            shape0 = v.shape[0] if hasattr(v, "shape") else v.size(0)
            if shape0 != num_experts:
                raise ValueError(
                    f"split: tensor {k} first dim {shape0} != num_experts {num_experts}")
            for e in range(num_experts):
                new[f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"] = v[e]
        else:
            new[k] = v
    return new


def read_moe_config(model_dir: str | Path) -> dict:
    """Read the MoE-relevant fields from config.json."""
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for required in ("num_hidden_layers", "num_experts", "first_k_dense_replace"):
        if required not in cfg:
            raise KeyError(f"config.json missing required key: {required}")
    return cfg


def detect_format(model_dir: str | Path) -> str:
    """'separate_expert' | 'merged_expert' | 'unknown' based on tensor key shapes."""
    try:
        sd, _ = load_state_dict(model_dir)
    except FileNotFoundError:
        return "unknown"
    has_merged = any(_MERGED_RE.match(k) for k in sd)
    has_separate = any(re.match(r"model\.layers\.\d+\.mlp\.experts\.\d+\.\w+_proj\.weight$", k)
                       for k in sd)
    if has_merged and not has_separate:
        return "merged_expert"
    if has_separate and not has_merged:
        return "separate_expert"
    return "unknown"


def convert(input_dir: str | Path, output_dir: str | Path, mode: str,
            on_log=None) -> dict:
    """Convert a model dir. mode ∈ {'merge','split'}. Returns a summary dict.

    Copies config.json + tokenizer + modeling files alongside the new weights, so
    the output dir is immediately usable (no VeOmni ``save_model_weights`` needed).
    """
    in_dir, out_dir = Path(input_dir), Path(output_dir)
    cfg = read_moe_config(in_dir)
    num_layers = cfg["num_hidden_layers"]
    num_experts = cfg["num_experts"]
    first_k = cfg["first_k_dense_replace"]

    def _log(msg: str) -> None:
        log.info("moe.convert", mode=mode, msg=msg)
        if on_log:
            try:
                on_log(msg)
            except Exception:  # noqa: BLE001
                pass

    _log(f"loading shards from {in_dir} ({mode})")
    sd, backend = load_state_dict(in_dir)
    _log(f"loaded {len(sd)} tensors via {backend}; "
         f"{num_layers} layers, {num_experts} experts (first_k_dense_replace={first_k})")

    if mode == "merge":
        new_sd = merge_state_dict(sd, num_layers=num_layers, num_experts=num_experts,
                                  first_k_dense_replace=first_k)
    elif mode == "split":
        new_sd = split_state_dict(sd, num_experts=num_experts)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    sd.clear()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_state_dict(out_dir, new_sd, backend)
    _copy_assets(in_dir, out_dir)
    _log(f"wrote {len(new_sd)} tensors + assets to {out_dir}")
    return {
        "mode": mode, "input": str(in_dir), "output": str(out_dir),
        "num_layers": num_layers, "num_experts": num_experts,
        "tensors_in": len(new_sd), "backend": backend,
    }


def _copy_assets(in_dir: Path, out_dir: Path) -> None:
    # config.json
    cfg = in_dir / "config.json"
    if cfg.is_file():
        shutil.copy2(cfg, out_dir / "config.json")
    # tokenizer + supporting files
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "tokenizer.model", "added_tokens.json", "generation_config.json"):
        src = in_dir / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)
    # modeling code (trust_remote_code) if present at source
    for name in _MODELING_FILES:
        src = in_dir / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)


def copy_modeling_file(original_base_dir: str | Path, dest_dir: str | Path) -> list[str]:
    """Copy modeling_llada2_moe.py (+ config) from the ORIGINAL base model dir into
    a freshly split export dir (Checklist: the manual README step 7, automated)."""
    copied = []
    for name in _MODELING_FILES:
        src = Path(original_base_dir) / name
        if src.is_file():
            shutil.copy2(src, Path(dest_dir) / name)
            copied.append(name)
    return copied
