"""SPDX-License-Identifier: Apache-2.0
Pipeline route stubs (Phases 2–6).

These routers declare the REST surface the UI will call and return HTTP 501 with
a pointer to the implementing phase, so the contract is stable and discoverable
before the feature lands. Each is replaced by a real implementation in its phase.
"""
from __future__ import annotations

from fastapi import APIRouter

chat = APIRouter(prefix="/api/chat", tags=["chat"])


def _todo(phase: int, feature: str) -> dict:
    from fastapi import HTTPException
    raise HTTPException(
        status_code=501,
        detail=f"{feature} is implemented in Phase {phase} (see Checklist.md).",
    )


# ── chat / inference (Phase 6) ───────────────────────────────────────────────
@chat.post("/completions")
async def chat_completions(): _todo(6, "POST /api/chat/completions")
@chat.get("/history")
async def chat_history(): _todo(6, "GET /api/chat/history")
