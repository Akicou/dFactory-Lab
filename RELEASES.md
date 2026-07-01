# Releases

## v0.1.0 — full lab scaffold (Phases 0–10)

First end-to-end build of dFactory-Lab: the dFactory dLLM fine-tuning engine
wrapped in a FastAPI server + an Editorial-Paper React UI, so the whole
download → MoE-merge → dataset → block-diffusion SFT → split → export → chat
pipeline runs without the CLI.

| Phase | Commit | What landed |
|---|---|---|
| 0 | `4fb01f7` | Foundation: vendored dFactory engine, GitHub meta, **Checklist.md** (403 items) |
| 1 | `f00539c` | FastAPI server skeleton — run.py + banner, settings, structlog, SQLite, job framework, hardware probe, health/shutdown, graceful teardown |
| 2 | `00d2d04` | Model hub + MoE merge/split (faithful reimplementation of `moe_convertor.py`; format detection; HF download) |
| 3 | `aac97c1` | Datasets — conversational conversion + column mapper, validation, GSM8K build, preview |
| 4 | `8e87f67` | Training engine — full YAML config builder (real keys incl. diffusion knobs), torchrun launcher, metrics parse, VRAM precheck, dry-run |
| 5 | `46b981d` | Export — checkpoint discovery, auto split + modeling-file copy, completeness check, sha256 manifest |
| 6 | `a370ba3` | Inference & chat — dLLM params (steps/masking), backend interface + mock, history, base-vs-tuned compare |
| 7 | `15201ac` | Frontend shell + locked **Editorial Paper** design system, live Dashboard |
| 8 | `5d3a0f6` | Feature screens — Models, Datasets, Training, Export, Chat (live API; diffusion playground) |
| 9 | `73fdec6` | Auth (bootstrap token), audit log, real Settings UI, PACKAGING.md |
| 10 | _this_ | CI runs the suite; convenience runner; full-stack quickstart; release |

### Verified locally
- 49 backend tests green (`pytest server/tests`).
- `npm run build` green (30 modules).
- Full-stack smoke: server serves `web/dist` at `/` (+ SPA fallback), and
  `/api/health`, `/api/models` (256-expert catalog), `/api/chat/completions`
  (per-step unmasking) all respond.

### Needs a real ML environment (not exercised here)
- Live HuggingFace download (network + large weights).
- An actual `torchrun` block-diffusion SFT run (GPU + `torch` + the `VeOmni`
  submodule). The server builds the exact argv/env and dry-run is verified; a
  real run requires the runtime.
- A real inference backend (transformers / sglang) — the mock backend exercises
  the full chat pipeline; `set_backend()` plugs in the model runtime.
- Tauri desktop installers (Rust toolchain; see `PACKAGING.md`).
