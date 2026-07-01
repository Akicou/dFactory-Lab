"""SPDX-License-Identifier: Apache-2.0
Bootstrap-token auth (Checklist A-*).

Loopback binds (the default) are open for local development. When the server is
exposed (0.0.0.0 / :: / tunnel), every /api/* request - except a small
allowlist - must present the bootstrap access token (Authorization: Bearer <token>
or ?token=). This mirrors Unsloth Studio's model without copying its code.
"""
from __future__ import annotations

import hmac
from typing import Iterable

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# paths that remain reachable without a token (health, liveness, auth, docs)
_PUBLIC_PREFIXES = (
    "/api/health", "/api/liveness", "/api/auth", "/api/openapi", "/api/docs",
    "/api/redoc",
)


def auth_required(request: Request) -> bool:
    """Auth is enforced only when the bind is exposed (not loopback)."""
    s = getattr(request.app.state, "settings", None)
    if s is None:
        return False
    return not s.is_loopback


def access_token(request: Request) -> str:
    return getattr(request.app.state, "access_token", "") or ""


def verify(request: Request) -> bool:
    """True if the request presents a valid token (or none is required)."""
    if not auth_required(request):
        return True
    tok = access_token(request)
    if not tok:  # nothing configured -> fail open is unsafe; treat as required-but-unset
        return False
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        presented = header[7:].strip()
    else:
        presented = (request.query_params.get("token") or "").strip()
    return bool(presented) and hmac.compare_digest(presented, tok)


def _is_public(path: str) -> bool:
    return path == "/" or not path.startswith("/api/") or path.startswith(_PUBLIC_PREFIXES)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or _is_public(request.url.path) or verify(request):
            return await call_next(request)
        return JSONResponse(status_code=401,
                            content={"ok": False, "error": "unauthorized: bootstrap token required"})
