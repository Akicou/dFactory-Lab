"""SPDX-License-Identifier: Apache-2.0
Pydantic request/response schemas shared across routes.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class OK(BaseModel):
    """Standard success envelope."""
    ok: bool = True
    data: Optional[Any] = None


class ErrorDetail(BaseModel):
    ok: bool = False
    error: str
    detail: Optional[Any] = None


class Health(BaseModel):
    status: str = "ok"  # ok | starting | degraded
    version: str = "0.1.0"
    phase: int = 1
    bind: Optional[str] = None
    uptime_s: float = 0.0
    active_jobs: int = 0
    gpu: Optional[Any] = Field(None, description="detected hardware summary (None until probed)")


class SystemInfo(BaseModel):
    python: str
    platform: str
    cpus: int
    hardware: Any
    data_dir: str
    engine_present: bool


class JobStatus(BaseModel):
    id: str
    kind: str  # download | merge | split | build_dataset | train | export | inference
    state: str  # queued | running | done | error | cancelled
    progress: float = 0.0  # 0..1
    message: str = ""
    created_at: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
