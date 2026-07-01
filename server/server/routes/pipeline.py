"""SPDX-License-Identifier: Apache-2.0
Pipeline route stubs (Phases 2–6).

These routers declare the REST surface the UI will call and return HTTP 501 with
a pointer to the implementing phase, so the contract is stable and discoverable
before the feature lands. Each is replaced by a real implementation in its phase.
"""
from __future__ import annotations

from fastapi import APIRouter

models = APIRouter(prefix="/api/models", tags=["models"])
datasets = APIRouter(prefix="/api/datasets", tags=["datasets"])
training = APIRouter(prefix="/api/training", tags=["training"])
export = APIRouter(prefix="/api/export", tags=["export"])
chat = APIRouter(prefix="/api/chat", tags=["chat"])


def _todo(phase: int, feature: str) -> dict:
    from fastapi import HTTPException
    raise HTTPException(
        status_code=501,
        detail=f"{feature} is implemented in Phase {phase} (see Checklist.md).",
    )


# ── models / hub (Phase 2) ───────────────────────────────────────────────────
@models.get("")
async def models_list(): _todo(2, "GET /api/models")
@models.post("/download")
async def models_download(): _todo(2, "POST /api/models/download")
@models.post("/merge")
async def models_merge(): _todo(2, "POST /api/models/merge")
@models.post("/split")
async def models_split(): _todo(2, "POST /api/models/split")
@models.get("/local")
async def models_local(): _todo(2, "GET /api/models/local")


# ── datasets (Phase 3) ───────────────────────────────────────────────────────
@datasets.get("")
async def datasets_list(): _todo(3, "GET /api/datasets")
@datasets.post("")
async def datasets_create(): _todo(3, "POST /api/datasets")
@datasets.post("/build")
async def datasets_build(): _todo(3, "POST /api/datasets/build")
@datasets.get("/{dataset_id}/preview")
async def datasets_preview(dataset_id: str): _todo(3, "GET /api/datasets/{id}/preview")


# ── training (Phase 4) ───────────────────────────────────────────────────────
@training.get("/config")
async def training_config_schema(): _todo(4, "GET /api/training/config")
@training.post("/start")
async def training_start(): _todo(4, "POST /api/training/start")
@training.get("/runs")
async def training_runs(): _todo(4, "GET /api/training/runs")
@training.get("/runs/{run_id}")
async def training_run(run_id: str): _todo(4, "GET /api/training/runs/{id}")


# ── export (Phase 5) ─────────────────────────────────────────────────────────
@export.post("")
async def export_create(): _todo(5, "POST /api/export")
@export.get("/{export_id}")
async def export_status(export_id: str): _todo(5, "GET /api/export/{id}")


# ── chat / inference (Phase 6) ───────────────────────────────────────────────
@chat.post("/completions")
async def chat_completions(): _todo(6, "POST /api/chat/completions")
@chat.get("/history")
async def chat_history(): _todo(6, "GET /api/chat/history")
