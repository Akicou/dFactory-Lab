"""SPDX-License-Identifier: Apache-2.0
Export routes (Phase 5; replaces /api/export stubs).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..logging_config import get_logger
from ..schemas import OK
from ..services import export as svc
from ..services.audit import audit_req
from ..subprocess_util import PathEscapeError, validate_path

router = APIRouter(prefix="/api/export", tags=["export"])
log = get_logger(__name__)


class ExportReq(BaseModel):
    source: str                 # training output_dir OR a checkpoint dir
    original_base_dir: str      # pre-merge base model dir (for the modeling file)
    export_name: str


def _roots(req: Request) -> list[Path]:
    s = req.app.state.settings
    return [Path(s.data_dir).resolve(), Path(__file__).resolve().parents[3]]


@router.post("")
async def export_model(body: ExportReq, req: Request) -> OK:
    s = req.app.state.settings
    export_dir = (Path(s.data_dir) / "exports" / body.export_name).resolve()
    try:
        validate_path(body.source, _roots(req))
        validate_path(body.original_base_dir, _roots(req))
        validate_path(export_dir, [Path(s.data_dir).resolve()])
    except PathEscapeError as exc:
        return OK(data={"ok": False, "error": str(exc)})

    def fn(job, update):
        return svc.export_run(source=body.source, original_base_dir=body.original_base_dir,
                              export_dir=export_dir, update=update)

    jid = req.app.state.registry.submit("export", fn,
                                        payload={"source": body.source, "export_name": body.export_name})
    audit_req(req, action="export.run", target=body.export_name,
              detail={"job_id": jid, "source": body.source})
    return OK(data={"job_id": jid, "kind": "export", "export_dir": str(export_dir)})


@router.get("/{job_id}")
async def export_status(job_id: str, req: Request) -> OK:
    job = req.app.state.registry.get(job_id)
    if job is None:
        return OK(data={"ok": False, "error": "job not found"})
    return OK(data=job.to_dict())
