"""SPDX-License-Identifier: Apache-2.0
Chat / inference routes (Phase 6; replaces /api/chat stubs).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..logging_config import get_logger
from ..schemas import OK
from ..services import inference as svc
from ..services.audit import audit_req
from ..subprocess_util import PathEscapeError, validate_path

router = APIRouter(prefix="/api/chat", tags=["chat"])
log = get_logger(__name__)


class Message(BaseModel):
    role: str
    content: str


class CompletionsReq(BaseModel):
    model_dir: str | None = None
    server_id: str | None = None   # a loaded server (see /load); overrides model_dir routing
    messages: list[Message]
    params: svc.DiffusionParams = svc.DiffusionParams()
    session: str = "default"


class LoadReq(BaseModel):
    model_path: str


class EjectReq(BaseModel):
    id: str


class CompareReq(BaseModel):
    left_id: str
    right_id: str
    messages: list[Message]
    params: svc.DiffusionParams = svc.DiffusionParams()


def _models_root(req: Request) -> Path:
    return (Path(req.app.state.settings.data_dir) / "models").resolve()


def _gen(req: Request, sid: str | None, messages: list[dict],
         params: svc.DiffusionParams, model_dir: str | None = None) -> dict:
    """Generate via a loaded server when sid is given, else the global backend."""
    if sid:
        try:
            return req.app.state.servers.generate(sid, messages, params)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
    return svc.generate(messages, params, model_dir=model_dir)


@router.get("/params")
async def params() -> OK:
    p = svc.DiffusionParams()
    return OK(data={"defaults": p.model_dump(),
                    "notes": "dLLM generate() recomputes attention per step (no KV-cache "
                             "speedup); repetition_penalty is not supported."})


@router.get("/loaded")
async def loaded(req: Request) -> OK:
    return OK(data=req.app.state.servers.list())


@router.post("/load")
async def load(body: LoadReq, req: Request) -> OK:
    try:
        path = validate_path(body.model_path, [_models_root(req)])
    except PathEscapeError as exc:
        return OK(data={"ok": False, "error": str(exc)})
    try:
        sid = req.app.state.servers.load(str(path))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    audit_req(req, action="chat.load", target=str(path), detail={"id": sid})
    return OK(data={"id": sid})


@router.post("/eject")
async def eject(body: EjectReq, req: Request) -> OK:
    ok = req.app.state.servers.eject(body.id)
    audit_req(req, action="chat.eject", target=body.id, detail={"ok": ok})
    return OK(data={"ok": ok})


@router.post("/completions")
async def completions(body: CompletionsReq, req: Request) -> OK:
    messages = [m.model_dump() for m in body.messages]
    res = _gen(req, body.server_id, messages, body.params, model_dir=body.model_dir)
    if body.model_dir or body.server_id:
        svc.record_turn(req.app.state.settings.data_dir, session=body.session,
                        model_dir=body.model_dir or body.server_id or "",
                        messages=messages, response=res["text"], params=body.params)
    return OK(data={"response": res["text"], "details": res, "params": body.params.model_dump()})


@router.post("/compare")
async def compare(body: CompareReq, req: Request) -> OK:
    msgs = [m.model_dump() for m in body.messages]
    # Sequential, like Unsloth's Model Arena (parallel inference is future work).
    left = _gen(req, body.left_id, msgs, body.params)
    right = _gen(req, body.right_id, msgs, body.params)
    return OK(data={"left": left, "right": right})


@router.get("/history")
async def history(req: Request, session: str | None = None, limit: int = 50) -> OK:
    rows = svc.read_history(req.app.state.settings.data_dir, session=session, limit=limit)
    return OK(data=rows)
