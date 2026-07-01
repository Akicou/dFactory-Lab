"""SPDX-License-Identifier: Apache-2.0
Training routes (Phase 4; replaces /api/training stubs).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..logging_config import get_logger
from ..schemas import OK
from ..services import training as svc
from ..services.audit import audit_req

router = APIRouter(prefix="/api/training", tags=["training"])
log = get_logger(__name__)


class ValidateReq(BaseModel):
    config: dict


class StartReq(BaseModel):
    preset: str = "llada2-mini"
    overrides: dict | None = None
    model_id: str | None = None
    dataset_id: str | None = None
    dry_run: bool = False


class FinetuneReq(BaseModel):
    model_source: str
    dataset_path: str
    preset: str = "llada2-mini"
    overrides: dict | None = None
    dry_run: bool = True
    model_id: str | None = None
    dataset_id: str | None = None


@router.get("/config")
async def config_schema() -> OK:
    return OK(data={
        "presets": list(svc.PRESETS.keys()),
        "default": svc.DEFAULT_CONFIG,
        "diffusion_keys": list(svc.DIFFUSION_KEYS),
        "required_keys": {k: sorted(v) for k, v in svc.REQUIRED_KEYS.items()},
    })


@router.post("/config/validate")
async def validate_config(body: ValidateReq) -> OK:
    ok, errors = svc.validate_config(body.config)
    return OK(data={"valid": ok, "errors": errors})


@router.get("/vram")
async def vram(num_params_b: float, gpus: int = 1, offload: bool = True) -> OK:
    return OK(data=svc.estimate_vram_gb(num_params_b, fsdp_shards=max(1, gpus), offload=offload))


@router.get("/runs")
async def runs(req: Request) -> OK:
    from .. import db as _db
    s = req.app.state.settings
    with _db.connection(s.db_path()) as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM training_runs ORDER BY created_at DESC")]
    return OK(data=rows)


@router.get("/runs/{run_id}")
async def run(run_id: str, req: Request) -> OK:
    from .. import db as _db
    s = req.app.state.settings
    with _db.connection(s.db_path()) as conn:
        row = conn.execute("SELECT * FROM training_runs WHERE id=?", (run_id,)).fetchone()
    return OK(data=dict(row) if row else None)


@router.post("/start")
async def start(body: StartReq, req: Request) -> OK:
    s = req.app.state.settings
    cfg = svc.build_config(body.preset, body.overrides)
    ok, errors = svc.validate_config(cfg)
    if not ok:
        return OK(data={"ok": False, "errors": errors})

    cfg_dir = Path(s.data_dir) / "configs"
    cfg_path = svc.write_config_yaml(cfg, cfg_dir / f"{body.preset}-{__import__('time').strftime('%Y%m%d%H%M%S')}.yaml")

    def fn(job, update):
        return svc.launch(cfg_path, s, update=update, dry_run=body.dry_run)

    jid = req.app.state.registry.submit("train", fn, payload={"preset": body.preset, "config": str(cfg_path)})
    rid = svc.register_run(s.db_path(), job_id=jid, model_id=body.model_id or "",
                           dataset_id=body.dataset_id or "", config_path=str(cfg_path),
                           output_dir=cfg["train"]["output_dir"])
    audit_req(req, action="training.start", target=rid,
              detail={"preset": body.preset, "config": str(cfg_path), "dry_run": body.dry_run})
    return OK(data={"job_id": jid, "run_id": rid, "config_path": str(cfg_path),
                    "dry_run": body.dry_run})


@router.post("/finetune")
async def finetune(body: FinetuneReq, req: Request) -> OK:
    """One-click finetune: auto-merge the model, then train (the merge step runs
    for real; the torchrun step is dry-run unless a GPU runtime is present)."""
    from ..subprocess_util import PathEscapeError, validate_path
    s = req.app.state.settings
    roots = [Path(s.data_dir).resolve(), Path(__file__).resolve().parents[3]]
    try:
        validate_path(body.model_source, roots)
        validate_path(body.dataset_path, roots)
    except PathEscapeError as exc:
        return OK(data={"ok": False, "error": str(exc)})

    def fn(job, update):
        return svc.finetune(model_source=body.model_source, dataset_path=body.dataset_path,
                            preset=body.preset, overrides=body.overrides, settings=s,
                            update=update, dry_run=body.dry_run)

    jid = req.app.state.registry.submit("train", fn,
                                        payload={"model_source": body.model_source, "preset": body.preset})
    rid = svc.register_run(s.db_path(), job_id=jid, model_id=body.model_id or "",
                           dataset_id=body.dataset_id or "", config_path="finetune",
                           output_dir="(orchestrated)")
    audit_req(req, action="training.finetune", target=rid,
              detail={"model_source": body.model_source, "dataset_path": body.dataset_path,
                      "dry_run": body.dry_run})
    return OK(data={"job_id": jid, "run_id": rid})
