"""SPDX-License-Identifier: Apache-2.0
Hardware detection (Checklist H-2, H-19).

Detects CUDA / ROCm / Apple Silicon (MPS) / CPU. ``CUDA_DEVICE_ORDER`` is pinned
to ``PCI_BUS_ID`` *before* any torch import so a GPU index chosen from nvidia-smi
matches the one torch/CUDA resolve (Unsloth's main.py documents this mismatch).
torch is imported lazily and optional - the server boots and reports ``cpu`` when
the ML stack is absent.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any

from .logging_config import get_logger

log = get_logger(__name__)

# Must run before torch creates a CUDA context.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def detect_hardware() -> dict[str, Any]:
    """Best-effort hardware probe. Never raises."""
    info: dict[str, Any] = {
        "backend": "cpu",
        "device_count": 0,
        "devices": [],
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "platform": platform.platform(),
        "torch_version": None,
    }

    torch = _maybe_import_torch()
    if torch is not None:
        info["torch_version"] = torch.__version__
        try:
            if torch.cuda.is_available():
                info["backend"] = "cuda"
                info["device_count"] = torch.cuda.device_count()
                info["devices"] = [
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total": int(torch.cuda.get_device_properties(i).total_memory),
                    }
                    for i in range(torch.cuda.device_count())
                ]
                return info
        except Exception as exc:  # noqa: BLE001
            log.debug("hardware.torch_cuda_probe_failed", error=str(exc))

        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info["backend"] = "mps"
                info["device_count"] = 1
                info["devices"] = [{"index": 0, "name": "Apple Silicon (MPS)"}]
                return info
        except Exception:  # noqa: BLE001
            pass

    # ROCm via rocm-smi (no torch, or torch without ROCm build).
    rocm = _probe_rocm()
    if rocm:
        return rocm

    # NVIDIA via nvidia-smi (no torch build present).
    nvidia = _probe_nvidia_smi()
    if nvidia:
        return nvidia

    info["device_count"] = os.cpu_count() or 1
    return info


def _maybe_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:  # noqa: BLE001 - torch optional at this stage
        return None


def _probe_nvidia_smi() -> dict[str, Any] | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        devices = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                devices.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total": int(parts[2]) * 1024 * 1024,
                })
        if devices:
            return {
                "backend": "cuda",
                "device_count": len(devices),
                "devices": devices,
                "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
                "platform": platform.platform(),
                "torch_version": None,
                "via": "nvidia-smi",
            }
    except Exception as exc:  # noqa: BLE001
        log.debug("hardware.nvidia_smi_failed", error=str(exc))
    return None


def _probe_rocm() -> dict[str, Any] | None:
    rocm_smi = shutil.which("rocm-smi") or os.path.join(
        os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "rocm-smi")
    if not shutil.which(rocm_smi):
        return None
    try:
        out = subprocess.run(
            [rocm_smi, "--showproductname", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        import json
        cards = {k: v for k, v in json.loads(out.stdout).items() if k.startswith("card")}
        devices = [{
            "index": i,
            "name": c.get("Card series", c.get("Card model", "AMD GPU")),
            "memory_total": int(c.get("VRAM Total Memory (B)", 0)),
        } for i, c in enumerate(cards.values())]
        if devices:
            return {
                "backend": "rocm", "device_count": len(devices), "devices": devices,
                "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
                "platform": platform.platform(), "torch_version": None, "via": "rocm-smi",
            }
    except Exception as exc:  # noqa: BLE001
        log.debug("hardware.rocm_failed", error=str(exc))
    return None


def summary() -> dict[str, Any]:
    """Compact summary for /api/health and /api/system."""
    hw = detect_hardware()
    return {
        "backend": hw["backend"],
        "device_count": hw["device_count"],
        "devices": [{"index": d.get("index"), "name": d.get("name")} for d in hw.get("devices", [])],
    }
