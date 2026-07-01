"""SPDX-License-Identifier: Apache-2.0
Audit log (Checklist A-* audit): append-only record of model/data/credential ops.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def audit(db_path: Path, *, actor: str = "system", action: str,
          target: str = "", detail: Optional[dict[str, Any]] = None) -> None:
    from .. import db as _db
    with _db.connection(db_path) as conn:
        conn.execute(
            "INSERT INTO audit_events (ts, actor, action, target, detail) VALUES (?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), actor, action, target,
             json.dumps(detail or {}, default=str)),
        )


def audit_req(request, *, action: str, target: str = "", detail: Optional[dict] = None) -> None:
    """Convenience: audit against the request's settings db path."""
    s = getattr(request.app.state, "settings", None)
    if s is not None:
        audit(s.db_path(), action=action, target=target, detail=detail)


def recent(db_path: Path, limit: int = 100) -> list[dict]:
    from .. import db as _db
    with _db.connection(db_path) as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM audit_events ORDER BY id DESC LIMIT ?", (limit,))]
