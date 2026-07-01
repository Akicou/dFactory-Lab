"""SPDX-License-Identifier: Apache-2.0
Dataset preparation (Checklist D-*).

Converts any tabular source into the conversational ``messages`` JSONL the dFactory
trainer expects (data_type:conversation, text_keys:messages), mirroring
scripts/build_gsm8k_dataset.py's format_gsm8k_example_to_messages but with a
configurable column->role mapper. Validates the messages schema and writes
train/eval JSONL splits under the managed data dir.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..logging_config import get_logger

log = get_logger(__name__)

VALID_ROLES = {"system", "user", "assistant"}


def to_messages(row: dict, mapping: dict[str, str]) -> dict:
    """Map columns of *row* to a {messages:[...]} record per *mapping* (col->role).

    A column may map to a single role (question->user) or to 'messages' if the
    column already holds a list of {role,content} turns.
    """
    messages: list[dict[str, str]] = []
    # preserve a stable order: system first, then input order otherwise.
    ordered = sorted(mapping.items(), key=lambda kv: (kv[1] != "system",))
    for col, role in ordered:
        if col not in row:
            continue
        val = row[col]
        if role == "messages":
            if isinstance(val, list):
                messages.extend(val)
            continue
        content = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
        messages.append({"role": role, "content": content})
    return {"messages": messages}


def convert_rows(rows: Iterable[dict], mapping: dict[str, str]) -> list[dict]:
    return [to_messages(r, mapping) for r in rows]


def validate_record(rec: dict) -> tuple[bool, list[str]]:
    """A record is valid if it has a non-empty messages list of {role,content}."""
    errors: list[str] = []
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False, ["missing or empty 'messages'"]
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            errors.append(f"turn {i}: must have role and content")
            continue
        if m["role"] not in VALID_ROLES:
            errors.append(f"turn {i}: invalid role {m['role']!r}")
        if not isinstance(m["content"], str) or not m["content"].strip():
            errors.append(f"turn {i}: empty content")
    return (not errors), errors


def write_jsonl(records: list[dict], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return p


def read_jsonl(path: str | Path) -> list[dict]:
    out: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def stats(records: list[dict]) -> dict:
    from collections import Counter
    roles = Counter()
    lengths = []
    for rec in records:
        for m in rec.get("messages", []):
            roles[m.get("role", "?")] += 1
            lengths.append(len(m.get("content", "")))
    n = len(lengths)
    avg = sum(lengths) / n if n else 0
    return {
        "rows": len(records), "turns": n, "roles": dict(roles),
        "avg_chars": round(avg, 1),
        "max_chars": max(lengths) if lengths else 0,
    }


# ── sources ──────────────────────────────────────────────────────────────────
def from_rows(rows: list[dict], mapping: dict[str, str]) -> list[dict]:
    return convert_rows(rows, mapping)


def from_jsonl_file(path: str | Path) -> list[dict]:
    return read_jsonl(path)


def from_hf(repo_id: str, config: Optional[str] = None, split: Optional[str] = None) -> dict:
    """Load a HF dataset (needs the `datasets` extra). Returns {split: [rows]}."""
    from datasets import load_dataset
    ds = load_dataset(repo_id, config) if config else load_dataset(repo_id)
    return {sp: list(ds[sp]) for sp in ds.keys()}


def build_gsm8k(out_dir: str | Path, *, strip_cot: bool = False,
                update: Optional[Callable] = None) -> dict:
    """Wrap scripts/build_gsm8k_dataset.py: openai/gsm8k -> conversational JSONL.

    Mirrors format_gsm8k_example_to_messages (question->user, answer->assistant).
    Optionally strips the '#### <answer>' chain-of-thought marker from the assistant
    content. Writes gsm8k_{split}.jsonl under out_dir.
    """
    from datasets import load_dataset
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if update:
        update(0.1, "loading openai/gsm8k (main)")
    ds = load_dataset("openai/gsm8k", "main")
    mapping = {"question": "user", "answer": "assistant"}
    written = {}
    splits = dict(ds)
    for i, (split, rows) in enumerate(splits.items()):
        records = []
        for r in rows:
            ans = r["answer"]
            if strip_cot and "####" in ans:
                ans = ans.rsplit("####", 1)[0].rstrip()
            records.append(to_messages({"question": r["question"], "answer": ans}, mapping))
        path = write_jsonl(records, out / f"gsm8k_{split}.jsonl")
        written[split] = str(path)
        if update:
            update(0.1 + 0.9 * (i + 1) / len(splits), f"wrote {split}: {len(records)} rows")
    provenance = {"source": "openai/gsm8k", "config": "main", "strip_cot": strip_cot}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {"out_dir": str(out), "splits": written, "provenance": provenance}


def register_dataset(db_path: Path, *, name: str, source: str, path: str,
                     rows: int, revision: Optional[str] = None) -> str:
    """Persist a dataset record. Returns the dataset id."""
    did = uuid.uuid4().hex[:12]
    from datetime import datetime, timezone
    from .. import db as _db
    with _db.connection(db_path) as conn:
        conn.execute(
            "INSERT INTO datasets (id,name,source,path,rows,revision,created_at) VALUES (?,?,?,?,?,?,?)",
            (did, name, source, path, rows, revision, datetime.now(timezone.utc).isoformat()))
    return did
