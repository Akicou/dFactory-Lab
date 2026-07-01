# Contributing to dFactory-Lab

Thanks for helping build a lab console for fine-tuning diffusion LLMs. The full plan of record is
[`Checklist.md`](./Checklist.md) — most contributions should map to an item there.

## Workflow

1. Pick an open Checklist item (or a whole phase) and mention it in your PR (`Closes Checklist: §4 T-12`).
2. Branch from `main`: `feat/<area>-<short>`, e.g. `feat/moe-merge-endpoint`.
3. Each **phase** is committed as one or more commits; don't squash across phase boundaries.
4. Open a PR against `main` using the PR template.

## Repo areas

| Area | Path | Stack |
|------|------|-------|
| Engine (vendored, from dFactory) | `configs/`, `models/`, `scripts/`, `tasks/`, `train.sh` | Python, Apache-2.0 |
| Distributed framework | `VeOmni/` (submodule) | Python |
| Backend server | `server/` (planned) | Python, FastAPI |
| Frontend | `web/` (planned) | TypeScript, React, Tauri |

## Local checks

**Engine / backend**
```bash
ruff check .
pytest -q
PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH sh train.sh tasks/train_llada2_bd.py configs/sft/llada2_mini_bd_sft.yaml
```

**Frontend** (once `web/` exists)
```bash
cd web && npm ci && npm run typecheck && npm run build
```

## Rules

- **Never commit** model weights (`.safetensors/.bin/.pt/.ckpt/.gguf`), datasets, or secrets. CI rejects them.
- Keep shell scripts LF (`.gitattributes` enforces this).
- The dFactory engine stays **Apache-2.0**. Unsloth Studio is **AGPL-3.0** — study its patterns, do **not**
  copy its source into this repo.
- UI must follow the locked **design system** (distinct; no generic black/blue AI-slop look). See Checklist §7.

## License

By contributing you agree your contributions are licensed Apache-2.0.
