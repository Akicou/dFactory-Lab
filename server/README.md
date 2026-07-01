# `server/` — dFactory-Lab backend (planned)

FastAPI application that **starts the same way as Unsloth Studio** (`run.py` → uvicorn, with a startup banner and
bootstrap token) but **drives the dFactory pipeline** for the user: download → MoE merge → dataset build →
block-diffusion SFT → MoE split → modeling-file copy → packaging → inference.

> Status: **planned (Phase 1)**. See [`../Checklist.md`](../Checklist.md), sections **Backend Core & Server
> Lifecycle**, **MoE Weight Processing**, **Training Engine**, **Export Pipeline**, **Inference & Chat**.

Responsibilities (built across phases):

- App factory + lifecycle + startup banner + graceful teardown of child training processes
- Background job framework (download / merge / train / split / export / inference) with SSE/WebSocket progress
- SQLite-backed registry (models, datasets, runs, checkpoints, exports, settings)
- REST surface: `/api/models/*`, `/api/datasets/*`, `/api/training/*`, `/api/export/*`, `/api/chat/*`, `/api/hardware/*`
- Safe subprocess execution of `train.sh`, `torchrun`, and `scripts/moe_convertor.py` (no shell interpolation, jailed paths)
