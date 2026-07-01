# Deploy dFactory-Lab on a RunPod GPU pod

This runs the whole lab (UI + API + real block-diffusion training) on a GPU pod.
The pod pulls the Docker image, runs the server on port 8000, and trains on the
GPU. Models, datasets and checkpoints live on a network volume so they survive
pod restarts.

## Image

```
ghcr.io/akicou/dfactory-lab:latest
```

(Public read is enabled. RunPod can pull it with no credentials.)

## Start command

The image's default command is the start command, so leave RunPod's
"Docker Cmd / Entrypoint override" empty, OR set it explicitly:

```
python server/run.py --host 0.0.0.0 --port 8000
```

## Pod settings

| Setting | Value |
|---|---|
| Template | Custom: `ghcr.io/akicou/dfactory-lab:latest` |
| GPU | RTX 4090 (24 GB) minimum for mini (16B) with FSDP+CPU offload; more / multi-GPU for flash (100B) |
| Exposed port | `8000/http` (add an exposed port so the RunPod proxy URL works) |
| Volume | Network volume mounted at `/workspace` (persists weights/data) |
| Env | see below |

## Environment variables

Set these in the pod's env section:

```
DFACTORY_LAB_HOST=0.0.0.0
DFACTORY_LAB_PORT=8000
DFACTORY_LAB_TOKEN=<choose a long random string>   # required: 0.0.0.0 bind is token-gated
DFACTORY_LAB_DATA_DIR=/workspace/lab                # put state on the network volume
HF_TOKEN=<optional, for gated repos>
```

Because the bind is `0.0.0.0`, the server requires a bootstrap token on every
`/api/*` request (except health/liveness/auth/docs). Open the proxy URL and pass
the token as `Authorization: Bearer <DFACTORY_LAB_TOKEN>` or `?token=<...>`.

## First run

1. Pod boots, server prints the access banner (loopback + proxy URL + token).
2. Open the RunPod proxy URL for port 8000 in a browser.
3. **Models** tab: download `LLaDA2.0-mini`.
4. **Datasets** tab: build GSM8K (or upload your own).
5. **Training** tab: pick the model + dataset, set hyperparams, **Start finetune**
   (the server merges the MoE experts for you, then runs torchrun on the GPU).
6. **Save model** tab: split the checkpoint, copy the modeling file, package.
7. **Chat** tab: run the fine-tuned model.

Everything is written under `/workspace/lab` (the network volume).

---

## Build and push the image yourself

Run this where Docker is available. `whoami` here is your GitHub login, used as
the GHCR namespace.

```bash
# 1. whoami -> your GitHub username (namespace for the image)
USER=$(gh api user --jq .login)        # -> Akicou
echo "$USER"

# 2. one-time: give the gh token the GHCR write scope (opens a browser)
gh auth refresh -h github.com -s write:packages

# 3. log Docker into GHCR with that token
gh auth token | docker login ghcr.io -u "$USER" --password-stdin

# 4. build + push
docker build -t ghcr.io/$USER/dfactory-lab:latest .
docker push ghcr.io/$USER/dfactory-lab:latest

# 5. (optional) make the package public so RunPod can pull it without creds
gh api -X PATCH /user/packages/container/dfactory-lab -f visibility=public
```

Tag a pinned version too if you want reproducible pods:

```bash
docker tag ghcr.io/$USER/dfactory-lab:latest ghcr.io/$USER/dfactory-lab:0.1.0
docker push ghcr.io/$USER/dfactory-lab:0.1.0
```

## Notes

- The image is large (CUDA + torch + VeOmni). First pull on a pod is slow; the
  network volume keeps it cached across restarts on the same region.
- `train.sh` / torchrun honor `NNODES`, `NPROC_PER_NODE`, `MASTER_ADDR`,
  `MASTER_PORT` — set them as env vars for multi-GPU / multi-pod runs.
- To run CPU-only (no GPU, just the UI + API + merge/split): the same image works
  with any pod; training will fail at the `torchrun` step without a GPU.
