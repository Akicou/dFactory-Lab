"""SPDX-License-Identifier: Apache-2.0
Model hub + MoE routes (Phase 2; replaces the /api/models stubs).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..logging_config import get_logger
from ..schemas import OK
from ..services import models as svc
from ..services.audit import audit_req
from ..subprocess_util import PathEscapeError, validate_path

router = APIRouter(prefix="/api/models", tags=["models"])
log = get_logger(__name__)


class DownloadReq(BaseModel):
    repo_id: str
    dest: Optional[str] = None  # defaults to <data_dir>/models
    nest_under_repo_name: bool = True


class ConvertReq(BaseModel):
    input_dir: str
    output_dir: str


def _read_roots(req: Request) -> list[Path]:
    s = req.app.state.settings
    return [Path(s.data_dir).resolve(), Path(__file__).resolve().parents[3]]


def _write_root(req: Request) -> Path:
    return (Path(req.app.state.settings.data_dir) / "models").resolve()


@router.get("")
async def catalog() -> OK:
    return OK(data=svc.catalog())


@router.get("/local")
async def local(req: Request) -> OK:
    s = req.app.state.settings
    return OK(data=svc.list_local(s.data_dir / "models"))


@router.get("/detect")
async def detect(path: str) -> OK:
    return OK(data={"path": path, "format": svc.moe.detect_format(path)})


@router.post("/download")
async def download(body: DownloadReq, req: Request) -> OK:
    s = req.app.state.settings
    dest = Path(body.dest) if body.dest else (Path(s.data_dir) / "models")
    try:
        validate_path(dest, _read_roots(req))
    except PathEscapeError as exc:
        return OK(data={"ok": False, "error": str(exc)})

    def fn(job, update):
        return svc.download_model(body.repo_id, dest, token=s.hf_token or None,
                                  nest_under_repo_name=body.nest_under_repo_name, update=update)

    jid = req.app.state.registry.submit("download", fn, payload={"repo_id": body.repo_id})
    audit_req(req, action="models.download", target=body.repo_id, detail={"job_id": jid})
    return OK(data={"job_id": jid, "kind": "download"})


@router.post("/merge")
async def merge(body: ConvertReq, req: Request) -> OK:
    return _submit_convert(req, "merge", body.input_dir, body.output_dir)


@router.post("/split")
async def split(body: ConvertReq, req: Request) -> OK:
    return _submit_convert(req, "split", body.input_dir, body.output_dir)


def _submit_convert(req: Request, mode: str, input_dir: str, output_dir: str) -> OK:
    roots = _read_roots(req)
    try:
        in_resolved = validate_path(input_dir, roots)
        out_resolved = validate_path(output_dir, [_write_root(req)])
    except PathEscapeError as exc:
        return OK(data={"ok": False, "error": str(exc)})

    fn = (svc.merge_model if mode == "merge" else svc.split_model)

    def _job(job, update):
        return fn(in_resolved, out_resolved, update=update)

    jid = req.app.state.registry.submit(mode, _job,
                                        payload={"input": str(in_resolved), "output": str(out_resolved)})
    audit_req(req, action=f"models.{mode}", target=str(in_resolved),
              detail={"job_id": jid, "output": str(out_resolved)})
    return OK(data={"job_id": jid, "kind": mode})
