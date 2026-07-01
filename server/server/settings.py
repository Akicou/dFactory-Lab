"""SPDX-License-Identifier: Apache-2.0
Central configuration for the dFactory-Lab server.

All settings are environment-driven (see ../.env.example). ``Settings`` is the
single source of truth read by the app factory, the runner and the job framework.
"""
from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DFACTORY_LAB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── server bind ────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8000
    token: str = ""  # bootstrap access token; auto-generated if empty

    # ── storage ────────────────────────────────────────────────────────────
    data_dir: Path = Path(".lab")

    # ── logging ────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "console"  # "console" (dev) | "json" (prod)
    file_log: bool = True

    # ── cors / web ─────────────────────────────────────────────────────────
    cors_origins: str = ""  # comma-separated; "" = loopback/desktop only

    # ── engine / training (mirror train.sh + .env.example) ─────────────────
    # These are surfaced for the training launcher; not prefixed because the
    # engine (torchrun / VeOmni) reads them verbatim from the environment.
    nnodes: int = 1
    nproc_per_node: int = 0  # 0 => auto (GPU count)
    node_rank: int = 0
    master_addr: str = "0.0.0.0"
    master_port: int = 12345

    # ── inference (SGLang) ─────────────────────────────────────────────────
    # Empty url => the deterministic MockBackend. Set to a running SGLang
    # OpenAI server (e.g. http://127.0.0.1:30000) to serve real LLaDA2.1 output.
    sglang_url: str = ""
    sglang_model: str = "default"
    sglang_timeout: float = 600.0

    # Chat "load/eject" launches SGLang servers per model (one process each). These
    # knobs mirror the LLaDA2.1 model-card launch command.
    sglang_python: str = ""            # "" => sys.executable
    sglang_host: str = "127.0.0.1"
    sglang_port_base: int = 30000
    sglang_max_loaded: int = 2         # two = the compare arena
    sglang_mem_fraction: float = 0.8
    sglang_tp_size: int = 1
    sglang_attention_backend: str = "flashinfer"
    sglang_dllm_algorithm: str = "JointThreshold"
    sglang_load_timeout_s: float = 600.0
    sglang_extra_args: str = ""        # extra flags, shlex-split
    sglang_simulate: bool = False      # skip the real process; serve via the mock (dev/no-GPU)

    # ── integrations ───────────────────────────────────────────────────────
    hf_token: str = ""
    wandb_project: str = ""
    wandb_name: str = ""

    @computed_field  # type: ignore[misc]
    @property
    def resolved_token(self) -> str:
        """The effective access token, generating one on first use if unset."""
        if self.token.strip():
            return self.token.strip()
        return secrets.token_urlsafe(16)

    @computed_field  # type: ignore[misc]
    @property
    def is_loopback(self) -> bool:
        return self.host in ("127.0.0.1", "localhost", "::1")

    def dirs(self) -> dict[str, Path]:
        """Managed subdirectories under data_dir (created lazily by ensure_dirs)."""
        root = self.data_dir
        return {
            "root": root,
            "db": root,
            "models": root / "models",
            "datasets": root / "datasets",
            "checkpoints": root / "checkpoints",
            "exports": root / "exports",
            "logs": root / "logs",
        }

    def ensure_dirs(self) -> None:
        for p in self.dirs().values():
            p.mkdir(parents=True, exist_ok=True)

    def db_path(self) -> Path:
        return self.data_dir / "dfactory_lab.sqlite"

    def engine_env(self) -> dict[str, str]:
        """Environment dict for engine subprocesses (torchrun / VeOmni)."""
        env = {
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NNODES": str(self.nnodes),
            "NODE_RANK": str(self.node_rank),
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
        }
        if self.nproc_per_node > 0:
            env["NPROC_PER_NODE"] = str(self.nproc_per_node)
        return env


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    return Settings()


def reset_settings_cache() -> None:
    """Test hook: clear the cached Settings so env changes take effect."""
    get_settings.cache_clear()


def repo_root() -> Path:
    return _repo_root()


def engine_root() -> Path:
    """The vendored dFactory engine directory (repo root)."""
    return _repo_root()
