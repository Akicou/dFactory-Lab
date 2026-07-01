# syntax=docker/dockerfile:1
#
# dFactory-Lab GPU image for RunPod (and any CUDA host).
# Builds the web UI, then layers it on a PyTorch+CUDA runtime with the VeOmni
# engine so block-diffusion SFT actually runs on the pod's GPU.

# ── stage 1: build the web frontend ─────────────────────────────────────────
FROM node:22-bookworm-slim AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ── stage 2: runtime (CUDA + torch + VeOmni + app) ──────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TOKENIZERS_PARALLELISM=false \
    TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    DFACTORY_LAB_HOST=0.0.0.0 \
    DFACTORY_LAB_PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ByteDance VeOmni distributed-training framework (the engine shells out to it).
# Cloned to /app/VeOmni because the server's launcher sets PYTHONPATH=<repo>/VeOmni.
RUN git clone --depth 1 https://github.com/ByteDance-Seed/VeOmni.git /app/VeOmni \
    && pip install -e /app/VeOmni/

# Server + engine extras (fastapi/uvicorn/pydantic/structlog + huggingface_hub/datasets/safetensors).
COPY server/ ./server/
RUN pip install -e "./server[dev,engine]"

# Vendored dFactory engine + the built UI.
COPY configs/ ./configs/
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY tasks/ ./tasks/
COPY train.sh ./train.sh
COPY --from=web /web/dist ./web/dist

ENV PYTHONPATH=/app/VeOmni

EXPOSE 8000
# Serve the UI at / and the API at /api on 0.0.0.0 (RunPod proxy reaches it).
CMD ["python", "server/run.py", "--host", "0.0.0.0", "--port", "8000"]
