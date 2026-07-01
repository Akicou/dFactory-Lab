"""SPDX-License-Identifier: Apache-2.0
Structured logging via structlog (Checklist B-24, B-25).

- console renderer in dev (readable, colored when a TTY)
- JSON renderer in prod (one object per line, log-shipper friendly)
- a ``filter_sensitive_data`` processor scrubs tokens / secrets from every event
- ``get_logger(__name__)`` is the single factory used across the codebase
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

import structlog

_SENSITIVE_KEYS = {
    "token", "authorization", "password", "secret", "api_key", "hf_token",
    "access_token", "refresh_token", "cookie",
}
_REDACT = "***REDACTED***"

# Greppy patterns for values that should never leak even as free-form strings.
_TOKEN_RES = [
    re.compile(r"(hf_[A-Za-z0-9]{16,})"),          # HuggingFace tokens
    re.compile(r"(Bearer\s+[A-Za-z0-9._\-]+)"),
    re.compile(r"(sk-[A-Za-z0-9]{16,})"),
]


def _scrub_value(v: Any) -> Any:
    if isinstance(v, str):
        for rx in _TOKEN_RES:
            v = rx.sub(_REDACT, v)
        return v
    return v


def filter_sensitive_data(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """structlog processor: redact sensitive keys + token-shaped values."""
    out: dict[str, Any] = {}
    for k, v in event_dict.items():
        if k.lower() in _SENSITIVE_KEYS:
            out[k] = _REDACT
        elif isinstance(v, dict):
            out[k] = {
                kk: (_REDACT if kk.lower() in _SENSITIVE_KEYS else _scrub_value(vv))
                for kk, vv in v.items()
            }
        else:
            out[k] = _scrub_value(v)
    return out


def setup_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Configure stdlib + structlog logging. Idempotent."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        filter_sensitive_data,
        timestamper,
    ]

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=_stdout_supports_color())

    structlog.configure(
        processors=shared + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging through the same renderer so uvicorn / engine logs
    # share the sensitive-data scrub and timestamp format.
    handler = logging.StreamHandler()
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared,
        )
    )
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) and
               isinstance(h.formatter, structlog.stdlib.ProcessorFormatter)
               for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(log_level)

    # Quiet noisy libs.
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _stdout_supports_color() -> bool:
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
    try:
        import sys
        return sys.stdout.isatty()
    except (AttributeError, OSError, ValueError):
        return False


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
