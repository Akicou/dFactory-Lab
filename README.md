<div align="center">

# dFactory-Lab

A lab console for fine-tuning, merging, and running diffusion LLMs (dLLMs)
without touching the CLI.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Engine: dFactory](https://img.shields.io/badge/engine-dFactory-yellow)](https://github.com/inclusionAI/dFactory)
[![Models: LLaDA2](https://img.shields.io/badge/models-LLaDA2.0%20mini%20%2F%20flash-orange)](https://huggingface.co/inclusionAI)

</div>

dFactory-Lab wraps the [dFactory](https://github.com/inclusionAI/dFactory) fine-tuning engine in a FastAPI server and a small web UI, so the whole diffusion-LLM workflow runs from a browser instead of a shell.

![Dashboard](web/screenshots/dashboard.png)

## What it does

dFactory trains diffusion LLMs (`LLaDA2.0-mini` 16B, `LLaDA2.0-flash` 100B) with block-diffusion supervised fine-tuning on top of ByteDance [VeOmni](https://github.com/ByteDance-Seed/VeOmni) and `torchrun`. Doing that by hand is a chain of CLI steps. dFactory-Lab drives the same steps from the UI:

1. download the base model
2. merge the MoE experts into the training format
3. build a dataset into conversational JSONL
4. train with block-diffusion SFT
5. split the checkpoint back into separate experts
6. copy the modeling file and package a runnable model
7. chat with the fine-tuned model

The fine-tuning, merge, split, and diffusion code comes from dFactory (Apache-2.0). The management UI takes cues from Unsloth Studio, whose source is AGPL-3.0 and is studied here, not copied.

## Screenshots

| Dashboard | Models |
|---|---|
| ![Dashboard](web/screenshots/dashboard.png) | ![Models](web/screenshots/models.png) |
| Training | Chat |
| ![Training](web/screenshots/training.png) | ![Chat](web/screenshots/chat.png) |

Regenerate them with `npm run screenshot` (in `web/`, with the server running). The script uses Chrome when available and falls back to the bundled Chromium.

## Repository layout

```
configs/        model and SFT YAML configs (from dFactory)
models/         LLaDA2-MoE modeling code (from dFactory)
scripts/        download, MoE merge/split, dataset build (from dFactory)
tasks/          block-diffusion SFT entrypoints (from dFactory)
train.sh        torchrun launcher (from dFactory)
VeOmni/         submodule: ByteDance distributed training framework
server/         FastAPI backend that drives the pipeline
web/            React UI (Vite, TypeScript, Tailwind)
docs/           docs and the upstream dFactory README
Checklist.md    the build checklist
```

## Quick start

```bash
git clone https://github.com/Akicou/dFactory-Lab.git
cd dFactory-Lab
git submodule update --init      # optional: only needed for a real GPU training run

# backend
python -m venv .venv
.venv/Scripts/python -m pip install -e 'server[dev,engine]'   # Linux/macOS: .venv/bin/python

# frontend
cd web && npm ci && npm run build && cd ..

# run: serves the UI at / and the API at /api
.venv/Scripts/python server/run.py
# open http://127.0.0.1:8000
```

For an actual block-diffusion training run you also need `torch` and the VeOmni runtime:

```bash
cd VeOmni && uv sync --extra gpu && source .venv/bin/activate && cd ..
PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH sh train.sh tasks/train_llada2_bd.py configs/sft/llada2_mini_bd_sft.yaml
```

See [`PACKAGING.md`](./PACKAGING.md) for the desktop (Tauri) path and [`Checklist.md`](./Checklist.md) for the full build plan.

## Status

First end-to-end build. The server, the UI, and the MoE merge/split round trip are covered by tests. A real training run needs a GPU and the VeOmni runtime; the server still builds the exact `torchrun` command, and the rest of the pipeline is exercised through tests and dry runs. Details in [`RELEASES.md`](./RELEASES.md).

## Design

The UI is a clean white theme with soft corners, lucide line icons, and quiet motion. It is its own design rather than a reskin of another tool.

## License

Apache-2.0. The dFactory engine is &copy; inclusionAI. See [`LICENSE`](./LICENSE) and [`LEGAL.md`](./LEGAL.md).
