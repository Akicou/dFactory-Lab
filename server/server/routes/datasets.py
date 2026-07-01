"""SPDX-License-Identifier: Apache-2.0
Dataset routes (Phase 3; replaces /api/datasets stubs).
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from pydantic import BaseModel

from ..logging_config import get_logger
from ..schemas import OK
from ..services import datasets as svc

router = APIRouter(prefix="/api/datasets", tags=["datasets"])
log = get_logger(__name__)


class ConvertReq(BaseModel):
    rows: list[dict]
    mapping: dict[str, str]            # column -> role (user/assistant/system/messages)
    name: str
    eval_ratio: float = 0.0            # optional held-out split


class BuildReq(BaseModel):
    preset: str = "gsm8k"              # gsm8k | hf
    repo_id: str | None = None
    config: str | None = None
    name: str = "gsm8k"
    strip_cot: bool = False


def _datasets_root(req: Request) -> Path:
    return Path(req.app.state.settings.data_dir) / "datasets"


@router.get("")
async def list_datasets(req: Request) -> OK:
    from .. import db as _db
    s = req.app.state.settings
    with _db.connection(s.db_path()) as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM datasets ORDER BY created_at DESC")]
    return OK(data=rows)


@router.post("/convert")
async def convert(body: ConvertReq, req: Request) -> OK:
    records = svc.convert_rows(body.rows, body.mapping)
    valid, invalid = [], []
    for rec in records:
        ok, errs = svc.validate_record(rec)
        (valid if ok else invalid).append((rec, errs))
    out_dir = _datasets_root(req) / body.name
    train_path = svc.write_jsonl([r for r, _ in valid], out_dir / "train.jsonl")
    res = {"stats": svc.stats([r for r, _ in valid]),
           "invalid_count": len(invalid), "train_path": str(train_path)}
    if body.eval_ratio and 0 < body.eval_ratio < 1 and len(valid) > 1:
        k = max(1, int(len(valid) * body.eval_ratio))
        eval_path = svc.write_jsonl([r for r, _ in valid[-k:]], out_dir / "eval.jsonl")
        res["eval_path"] = str(eval_path)
    did = svc.register_dataset(req.app.state.settings.db_path(), name=body.name,
                               source="convert", path=str(out_dir), rows=len(valid))
    return OK(data={"id": did, **res})


@router.post("/build")
async def build(body: BuildReq, req: Request) -> OK:
    s = req.app.state.settings
    out_dir = _datasets_root(req) / body.name

    def fn(job, update):
        if body.preset == "gsm8k":
            return svc.build_gsm8k(out_dir, strip_cot=body.strip_cot, update=update)
        if body.preset == "hf":
            splits = svc.from_hf(body.repo_id or "", body.config)
            written = {}
            total_rows = 0
            mapping = {"question": "user", "answer": "assistant"}  # default; HF varies
            for sp, rows in splits.items():
                recs = svc.convert_rows(rows, mapping) if rows and "question" in rows[0] else \
                    [{"messages": list(r.values())[:1]} for r in rows]
                p = svc.write_jsonl(recs, out_dir / f"{sp}.jsonl")
                written[sp] = str(p)
                total_rows += len(recs)
            return {"out_dir": str(out_dir), "splits": written, "rows": total_rows}
        raise ValueError(f"unknown preset {body.preset}")

    jid = req.app.state.registry.submit("build_dataset", fn,
                                        payload={"preset": body.preset, "name": body.name})
    return OK(data={"job_id": jid, "kind": "build_dataset", "out_dir": str(out_dir)})


@router.get("/{dataset_id}/preview")
async def preview(dataset_id: str, req: Request, n: int = 10) -> OK:
    from .. import db as _db
    s = req.app.state.settings
    with _db.connection(s.db_path()) as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
    if not row:
        return OK(data={"ok": False, "error": "dataset not found"})
    dpath = Path(row["path"])
    # prefer train.jsonl, else any .jsonl in the dir
    jsonl = dpath / "train.jsonl"
    if not jsonl.is_file():
        candidates = sorted(dpath.glob("*.jsonl"))
        jsonl = candidates[0] if candidates else None
    if not jsonl or not jsonl.is_file():
        return OK(data={"ok": False, "error": "no jsonl found", "path": str(dpath)})
    records = svc.read_jsonl(jsonl)[:n]
    return OK(data={"dataset": dict(row), "file": str(jsonl),
                    "records": records, "stats": svc.stats(svc.read_jsonl(jsonl))})


@router.post("/upload")
async def upload(req: Request, file: UploadFile = File(...)) -> OK:
    raw = (await file.read()).decode("utf-8", errors="replace")
    # accept jsonl (one record/line) or a json array
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not records:
        try:
            records = json.loads(raw)
        except json.JSONDecodeError:
            records = []
    name = Path(file.filename or "upload").stem
    out_dir = _datasets_root(req) / name
    path = svc.write_jsonl(records, out_dir / "train.jsonl")
    did = svc.register_dataset(req.app.state.settings.db_path(), name=name,
                               source="upload", path=str(out_dir), rows=len(records))
    return OK(data={"id": did, "rows": len(records), "path": str(path)})
