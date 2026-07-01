<div align="center">

# 🧪 dFactory-Lab

**A lab console for fine-tuning, merging & running discrete/diffusion LLMs (dLLMs) — without ever touching the CLI.**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Engine: dFactory](https://img.shields.io/badge/Engine-dFactory%20(inclusionAI)-yellow)](https://github.com/inclusionAI/dFactory)
[![Status: Phase 0](https://img.shields.io/badge/Status-Phase%200%20Foundation-lightgrey)](./Checklist.md)
[![Models: LLaDA2](https://img.shields.io/badge/Models-LLaDA2.0%20mini%20%2F%20flash-orange)](https://huggingface.co/inclusionAI)

</div>

---

## What is this?

**dFactory-Lab** wraps the [**dFactory**](https://github.com/inclusionAI/dFactory) dLLM fine-tuning engine
(diffusion LLMs — `LLaDA2.0-mini` 16B, `LLaDA2.0-flash` 100B — trained with **block-diffusion SFT** over
ByteDance [**VeOmni**](https://github.com/ByteDance-Seed/VeOmni) + `torchrun`) in a **server + UI**, so a user can
download → merge → prepare data → train → split → export → chat **entirely from a browser / desktop app**.

Conceptually it is **"Unsloth Studio, but purpose-built for diffusion LLMs."** The Unsloth Studio architecture
(FastAPI backend + Tauri/React frontend) is studied as the **UI/management reference**; the **fine-tuning, MoE
merge/split and block-diffusion logic all come from dFactory**.

> **Provenance & licensing.** The fine-tuning engine under `configs/`, `models/`, `scripts/`, `tasks/`, `train.sh`
> is copied from **dFactory** and remains **Apache-2.0** (see [`LICENSE`](./LICENSE), [`LEGAL.md`](./LEGAL.md),
> [`docs/upstream/dFactory-README.md`](./docs/upstream/dFactory-README.md)). **Unsloth Studio is AGPL-3.0**; we study
> its patterns but **do not copy its source** — dFactory-Lab is written from scratch and is Apache-2.0.

## The pipeline the Lab automates

These are the manual CLI steps from dFactory's README. **The Lab server performs all of them for the user:**

| # | Stage | dFactory source the Lab wraps |
|---|-------|-------------------------------|
| 1 | Download base model | `scripts/download_hf_model.py` |
| 2 | **Merge** MoE experts (separate → stacked) | `scripts/moe_convertor.py --mode merge` |
| 3 | Build dataset → conversational JSONL | `scripts/build_gsm8k_dataset.py` |
| 4 | Configure training (YAML) | `configs/sft/*.yaml` |
| 5 | **Train** block-diffusion SFT | `train.sh` + `tasks/train_llada2_bd.py` |
| 6 | **Split** checkpoint (stacked → separate) | `scripts/moe_convertor.py --mode split` |
| 7 | Copy modeling file into the export dir | `modeling_llada2_moe.py` |
| 8 | **Chat / inference** | served dLLM |

## Repository layout

```
dFactory-Lab/
├── configs/        # model + SFT YAML configs (from dFactory)
├── models/         # LLaDA2-MoE modeling code (from dFactory)
├── scripts/        # download / MoE merge+split / dataset build (from dFactory)
├── tasks/          # block-diffusion SFT training entrypoints (from dFactory)
├── train.sh        # torchrun launcher (from dFactory)
├── VeOmni/         # submodule — ByteDance distributed training framework
├── server/         # (planned) FastAPI backend that drives the pipeline
├── web/            # (planned) React + Tauri frontend (distinct, non-slop design)
├── docs/           # docs + upstream dFactory README
└── Checklist.md    # the build checklist (~300 items, all phases)
```

## Status

This repository is at **Phase 0 — Foundation**. The engine is vendored and the full build roadmap is captured in
[`Checklist.md`](./Checklist.md). Server and UI are built phase-by-phase; each phase ends with a git commit.

## Quick start (engine only, today)

```bash
git clone <this-repo>.git --recursive        # --recursive fetches the VeOmni submodule
cd dFactory-Lab

# environment (option A — uv, recommended)
cd VeOmni && uv sync --extra gpu && source .venv/bin/activate && cd ..

# environment (option B — pip)
pip install -e VeOmni/

# the Lab server (planned) will do everything below for you; until then, the raw CLI works:
PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH sh train.sh tasks/train_llada2_bd.py configs/sft/llada2_mini_bd_sft.yaml
```

## Design direction (TBD — picked in Phase 7)

The UI **must be visually distinct** from Unsloth/dFactory and **must avoid the generic "AI-slop" black + blue
gradient SaaS look.** Three candidate directions are proposed in `Checklist.md → Design System`; one will be
locked before frontend work begins.

## Roadmap

See [`Checklist.md`](./Checklist.md) for the full ~300-item, phase-by-phase plan. Phases:

`0` Foundation · `1` Repo + backend skeleton · `2` Model hub + MoE merge/split · `3` Datasets & prep ·
`4` Training engine · `5` Export/merge/packaging · `6` Inference & chat · `7` Frontend shell + design ·
`8` Frontend features · `9` Auth/security/observability/distribution · `10` Testing/QA/release

## License

Apache-2.0. The dFactory engine is © inclusionAI (Apache-2.0). See [`LICENSE`](./LICENSE) and [`LEGAL.md`](./LEGAL.md).
