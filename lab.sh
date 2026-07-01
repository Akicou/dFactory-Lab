#!/usr/bin/env bash
# dFactory-Lab convenience runner: builds the frontend if needed, then starts
# the server (which serves the SPA at / and the API at /api/*).
set -euo pipefail
cd "$(dirname "$0")"

# pick a python (venv if present, else system)
PY="${DFACTORY_PY:-}"
if [ -z "$PY" ]; then
  for cand in ".venv/Scripts/python.exe" ".venv/bin/python" "python3" "python"; do
    if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then PY="$cand"; break; fi
  done
fi

# build the frontend if dist is missing
if [ ! -f web/dist/index.html ] && command -v npm >/dev/null 2>&1; then
  echo "→ building frontend (web/dist)…"
  (cd web && npm ci && npm run build)
fi

exec "$PY" server/run.py "$@"
