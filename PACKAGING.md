# Packaging & distribution

dFactory-Lab ships as **two artifacts**: a Python **server** (`server/`) and a
**web frontend** (`web/`). The server can serve the built frontend directly, or
the frontend can run as a standalone **Tauri desktop** app talking to the server
in `--api-only` mode.

## 1. Web (primary) - served by the backend

```bash
# build the frontend
cd web && npm ci && npm run build        # -> web/dist/

# run the backend (serves web/dist at /, with SPA fallback)
cd ..
python server/run.py                     # http://127.0.0.1:8000
```

The FastAPI app's `_mount_frontend()` serves `web/dist/index.html` at `/` and on
any unknown client route; `/api/*` is the JSON API. In dev, `npm run dev`
(Vite :5174) proxies `/api` → `127.0.0.1:8000`.

## 2. Desktop (Tauri) - planned

The desktop path wraps the same `web/` build in a Tauri 2 shell and launches the
server in `--api-only` mode (no SPA, CORS locked to `tauri://localhost`). It
requires the Rust toolchain (`rustup`) and the Tauri CLI. Scaffold:

```bash
cd web
npm i -D @tauri-apps/cli@^2
npx tauri init                              # produces src-tauri/  (Cargo.toml, tauri.conf.json)
# configure tauri.conf.json: productName "dFactory-Lab", beforeBuildCommand "npm run build",
#   frontendDist "../dist", devUrl "http://localhost:5174"
npx tauri build                             # -> installers per platform
```

Restrictive CSP (`default-src 'self'; connect-src 'self' http://127.0.0.1:*`)
and the loopback-only server bind keep the desktop app local by default.

## 3. Security posture at distribution time

- Default bind is **127.0.0.1** (loopback) - open locally, no token needed.
- Network exposure (`-H 0.0.0.0` / tunnel) is opt-in and then **token-gated**:
  every `/api/*` request (bar health/liveness/auth/docs) needs
  `Authorization: Bearer <token>` or `?token=`. The bootstrap token is printed in
  the startup banner.
- Subprocess safety: every engine CLI (`torchrun`, `moe_convertor`,
  `build_gsm8k`) runs with argument arrays (no shell) and user paths are jailed
  under the managed data dir (see `server/subprocess_util.py`).
- Credentials (HF token) are read from env / settings, never logged
  (`filter_sensitive_data`), never committed.

## 4. Server packaging

```bash
cd server && pip install -e .[dev,engine]   # or:  uv sync --extra engine
python run.py --host 127.0.0.1 --port 8000
```

Engine extras (`huggingface_hub`, `datasets`, `PyYAML`) enable download/dataset
flows; the training run additionally needs `torch` + the `VeOmni` submodule
(`git submodule update --init --recursive`).
