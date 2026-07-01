"""SPDX-License-Identifier: Apache-2.0
Startup banner (Checklist B-21, B-22).

Stdlib-only (safe to import before the app). Mirrors the Unsloth Studio banner
shape — loopback URL, external URL when bound to the wildcard, API base, health
URL, and the bootstrap token — but branded for dFactory-Lab. ANSI accents lean
warm (terracotta/amber) to preview the locked Editorial Paper direction.
"""
from __future__ import annotations

import os
import sys


def _color_ok() -> bool:
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
    try:
        return sys.stdout.isatty()
    except (AttributeError, OSError, ValueError):
        return False


# Warm palette previewing Editorial Paper: ink + terracotta accent + muted rule.
_INK = "\033[38;5;235;1m"      # near-black ink (headline)
_ACCENT = "\033[38;5;166;1m"   # terracotta/saffron accent
_MUTED = "\033[38;5;245m"      # warm gray (rules, secondary)
_LINK = "\033[38;5;94;1m"      # deep terracotta link
_WARN = "\033[38;5;202;1m"     # brighter terracotta for warnings
_RESET = "\033[0m"


def _s(text: str, code: str, use: bool) -> str:
    return f"{code}{text}{_RESET}" if use else text


def _url_host(host: str) -> str:
    return f"[{host}]" if (":" in host and not host.startswith("[")) else host


def print_port_in_use_notice(original: int, new: int) -> None:
    msg = f"Port {original} is in use; using port {new} instead."
    print(_s(f"  ⚠ {msg}", _WARN, _color_ok()))


def print_lab_banner(
    *,
    port: int,
    bind_host: str,
    display_host: str,
    token: str = "",
    phase: int = 1,
) -> None:
    use = _color_ok()
    rule = _s("─" * 56, _MUTED, use)

    loopback_url = f"http://127.0.0.1:{port}"
    external_url = f"http://{_url_host(display_host)}:{port}"
    listen_all = bind_host in ("0.0.0.0", "::")
    primary = loopback_url if (listen_all or bind_host in ("127.0.0.1", "localhost", "::1")) else external_url

    lines = [
        "",
        _s("🧪  dFactory-Lab", _INK, use) + _s(f"   (Phase {phase})", _MUTED, use),
        rule,
        _s("  Open on this machine:", _MUTED, use),
        _s(f"    {primary}", _LINK, use),
    ]
    if listen_all and display_host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "::"):
        lines += ["", _s("  From another device on your network:", _MUTED, use),
                  _s(f"    {external_url}", _LINK, use)]

    lines += [
        "",
        _s("  API & health:", _MUTED, use),
        _s(f"    {primary}/api", _ACCENT, use),
        _s(f"    {primary}/api/health", _MUTED, use),
    ]

    if token:
        lines += ["", _s("  Bootstrap access token:", _MUTED, use),
                  _s(f"    {token}", _ACCENT, use)]

    lines += [
        rule,
        _s("  To stop: press Ctrl+C (Control+C, not Command+C, on macOS).", _WARN, use),
        rule,
        "",
    ]
    print("\n".join(lines))


def install_uvicorn_log_rewrite() -> None:
    """Rename uvicorn's 'Uvicorn running on' line to 'dFactory-Lab running on'."""
    import logging
    import re

    class _Rewrite(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.msg if isinstance(record.msg, str) else ""
                if msg.startswith("Uvicorn running on "):
                    record.msg = "dFactory-Lab running on " + msg[len("Uvicorn running on "):]
                if isinstance(record.msg, str):
                    record.msg = re.sub(r"\(Press CTRL\+C to quit\)",
                                        "(To stop: press Ctrl+C)", record.msg)
            except Exception:  # noqa: BLE001
                pass
            return True

    filt = _Rewrite()
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).addFilter(filt)
