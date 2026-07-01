# `web/` - dFactory-Lab frontend (planned)

React + Tauri desktop (and web) UI for the Lab - **distinct by design**. It is **not** a clone of Unsloth Studio;
it shares the *management-app pattern* (FastAPI client, sidebar shell, job dashboards) but ships its **own design
token system** and deliberately rejects the generic black + blue "AI-slop" aesthetic.

> Status: **planned (Phase 7–8)**. See [`../Checklist.md`](../Checklist.md), sections **Frontend App Shell & UX
> Foundation**, **Design System & Visual Direction**, **Frontend Feature Surfaces**.

Screens (built across phases):

- **Models/Hub** - catalog, download progress, merge/split actions, local registry
- **Datasets** - ingest, column→messages mapping, preview, train/eval splits
- **Training** - YAML config editor (every real key from `configs/sft/*.yaml`), live metrics, checkpoints, resume
- **Export** - target picker, automated split + modeling-file copy, packaging, download
- **Chat** - dLLM-specific params (denoising steps, masking schedule), history, base-vs-finetuned compare
- **Diffusion Playground** (differentiator) - step-by-step unmasking visualization
- **Settings** - GPU selection, HF token, paths, theme, locale
