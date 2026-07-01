"""SPDX-License-Identifier: Apache-2.0
Chat / inference routes (Phase 6; replaces /api/chat stubs).
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..logging_config import get_logger
from ..schemas import OK
from ..services import inference as svc

router = APIRouter(prefix="/api/chat", tags=["chat"])
log = get_logger(__name__)


class Message(BaseModel):
    role: str
    content: str


class CompletionsReq(BaseModel):
    model_dir: str | None = None
    messages: list[Message]
    params: svc.DiffusionParams = svc.DiffusionParams()
    session: str = "default"


class CompareReq(BaseModel):
    base_dir: str
    tuned_dir: str
    messages: list[Message]
    params: svc.DiffusionParams = svc.DiffusionParams()


@router.get("/params")
async def params() -> OK:
    p = svc.DiffusionParams()
    return OK(data={"defaults": p.model_dump(),
                    "notes": "dLLM generate() recomputes attention per step (no KV-cache "
                             "speedup); repetition_penalty is not supported."})


@router.post("/completions")
async def completions(body: CompletionsReq, req: Request) -> OK:
    messages = [m.model_dump() for m in body.messages]
    res = svc.generate(messages, body.params, model_dir=body.model_dir)
    if body.model_dir:
        svc.record_turn(req.app.state.settings.data_dir, session=body.session,
                        model_dir=body.model_dir, messages=messages,
                        response=res["text"], params=body.params)
    return OK(data={"response": res["text"], "details": res, "params": body.params.model_dump()})


@router.post("/compare")
async def compare(body: CompareReq) -> OK:
    msgs = [m.model_dump() for m in body.messages]
    base = svc.generate(msgs, body.params, model_dir=body.base_dir)
    tuned = svc.generate(msgs, body.params, model_dir=body.tuned_dir)
    return OK(data={"base": base, "tuned": tuned})


@router.get("/history")
async def history(req: Request, session: str | None = None, limit: int = 50) -> OK:
    rows = svc.read_history(req.app.state.settings.data_dir, session=session, limit=limit)
    return OK(data=rows)
