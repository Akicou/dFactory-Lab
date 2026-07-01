"""SPDX-License-Identifier: Apache-2.0
Inference & chat for diffusion LLMs (Checklist I-*, DL-*).

Defines the dLLM generation surface (denoising steps + masking schedule, not the
autoregressive KV-cache params), an InferenceBackend interface, and a deterministic
MockBackend so the chat API is fully exercisable without the model runtime. A real
backend (transformers + modeling_llada2_moe.LLaDA2MoeModelLM.generate, or sglang)
plugs into the same interface when torch + the model are available.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field

from ..logging_config import get_logger

log = get_logger(__name__)


class DiffusionParams(BaseModel):
    """Generation params specific to diffusion LLMs.

    Note: unlike autoregressive decoders, the LLaDA2 generate() recomputes
    attention each step (no incremental KV-cache speedup), and historically does
    NOT accept repetition_penalty — so it is intentionally absent here.
    """
    diffusion_steps: int = Field(32, ge=1, le=1024, description="denoising iterations")
    mask_schedule: str = Field("linear", description="linear|geometric|cosine")
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    max_new_tokens: int = Field(512, ge=1, le=8192)
    seed: Optional[int] = None
    show_unmasking: bool = False  # return per-step unmasking for the playground


class InferenceBackend(Protocol):
    loaded_dir: Optional[str]

    def load(self, model_dir: str | Path) -> None: ...
    def generate(self, messages: list[dict], params: DiffusionParams) -> dict: ...


def apply_chat_template(messages: list[dict], model_dir: Optional[str | Path] = None) -> str:
    """Best-effort chat template; falls back to role-tagged concat without a tokenizer."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True) if model_dir else None
        if tok is not None and getattr(tok, "chat_template", None):
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as exc:  # noqa: BLE001
        log.debug("inference.template_fallback", error=str(exc))
    return "".join(f"<|{m['role']}|>{m['content']}" for m in messages)


class MockBackend:
    """Deterministic stand-in backend. Echoes the prompt + chosen diffusion params
    so the chat pipeline is testable end-to-end before the model runtime exists."""

    def __init__(self) -> None:
        self.loaded_dir: Optional[str] = None

    def load(self, model_dir: str | Path) -> None:
        self.loaded_dir = str(model_dir)
        log.info("inference.mock.load", dir=self.loaded_dir)

    def generate(self, messages: list[dict], params: DiffusionParams) -> dict:
        prompt = apply_chat_template(messages, self.loaded_dir)
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = (f"[mock dLLM · {params.diffusion_steps} steps · {params.mask_schedule}] "
                f"reply to: {last_user[:120]}")
        steps = [
            {"step": i, "unmasked_frac": round((i + 1) / params.diffusion_steps, 3)}
            for i in range(params.diffusion_steps)
        ] if params.show_unmasking else None
        return {
            "text": text, "tokens_generated": len(text.split()),
            "diffusion_steps": params.diffusion_steps, "prompt_chars": len(prompt),
            "unmasking": steps,
        }


_backend: Optional[InferenceBackend] = None


def get_backend() -> InferenceBackend:
    """Return the active inference backend (MockBackend until a real one is wired)."""
    global _backend
    if _backend is None:
        _backend = MockBackend()
    return _backend


def set_backend(b: InferenceBackend) -> None:
    global _backend
    _backend = b


# ── history ──────────────────────────────────────────────────────────────────
def record_turn(data_dir: str | Path, *, session: str, model_dir: str,
                messages: list[dict], response: str, params: DiffusionParams) -> None:
    chat_dir = Path(data_dir) / "chat"
    chat_dir.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(), "session": session,
           "model_dir": model_dir, "messages": messages, "response": response,
           "params": params.model_dump()}
    with (chat_dir / f"{session}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_history(data_dir: str | Path, session: Optional[str] = None, limit: int = 50) -> list[dict]:
    chat_dir = Path(data_dir) / "chat"
    if not chat_dir.is_dir():
        return []
    files = [chat_dir / f"{session}.jsonl"] if session else sorted(chat_dir.glob("*.jsonl"))
    out: list[dict] = []
    for fp in files:
        if not fp.is_file():
            continue
        for line in fp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out[-limit:]


def generate(messages: list[dict], params: DiffusionParams,
             model_dir: Optional[str] = None) -> dict:
    b = get_backend()
    if model_dir:
        b.load(model_dir)
    return b.generate(messages, params)
