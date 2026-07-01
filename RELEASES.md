# Releases

## v0.1.0

First end-to-end build of dFactory-Lab. The dFactory diffusion-LLM fine-tuning
engine is wrapped in a FastAPI server and a small web UI, so the whole pipeline
runs from a browser: download, MoE merge, dataset prep, block-diffusion SFT,
split, packaging, and chat.

### What is included

- FastAPI server that starts like a normal dev server and serves the built UI.
  Settings, structured logging, a SQLite store, a background job framework,
  hardware detection, health and shutdown endpoints, and graceful teardown of
  child processes.
- Model hub with the LLaDA2.0 mini and flash catalog, local inventory, format
  detection, HuggingFace download, and MoE expert merge and split. The merge and
  split reimplement the exact transforms in the upstream converter, so they work
  without the VeOmni runtime.
- Dataset prep with a configurable column to messages mapper, schema validation,
  the GSM8K preset, and a live preview.
- Training engine with a YAML config builder for every real config key (including
  the diffusion knobs), a torchrun launcher that mirrors train.sh, metrics
  parsing, a VRAM estimate, and dry-run mode.
- Export pipeline that finds the checkpoint, splits it back to separate experts,
  copies the modeling file, verifies completeness, and writes a sha256 manifest.
- Chat with diffusion generation params (denoising steps and masking schedule),
  history, and base versus fine-tuned compare.
- React UI in a paper-toned theme: dashboard, models, datasets, training, export,
  chat with a per-step unmasking view, and settings.
- Bootstrap-token auth when the server is exposed, an audit log, CI that runs the
  test suite, and packaging notes for the desktop path.

### Verified locally

- 49 backend tests pass with pytest.
- The web build passes (npm run build).
- The server serves the built UI, and the health, models, and chat endpoints
  respond with real data. Screenshots in the README are produced by
  web/scripts/screenshot.mjs.

### Needs a real ML environment

These were not exercised on the build machine:

- A live HuggingFace download (network and large weights).
- A real torchrun block-diffusion SFT run (GPU, torch, and the VeOmni submodule).
  The server builds the exact command and dry-run is verified.
- A real inference backend (transformers or sglang). A mock backend exercises the
  full chat pipeline, and a real backend plugs into the same interface.
- Tauri desktop installers (Rust toolchain, see PACKAGING.md).
