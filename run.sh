#!/usr/bin/env bash
# Start the vision service in the foreground.
#
# caffeinate is not optional in practice: a sleeping Mac mini looks exactly like
# bad wifi from the Pi's side, and that is a miserable thing to debug at 11pm.
set -euo pipefail

cd "$(dirname "$0")"

: "${VS_BACKEND:=real}"          # fail loudly if models are missing, rather than inventing furniture
: "${VS_HOST:=0.0.0.0}"
: "${VS_PORT:=8080}"
: "${VS_DEVICE:=mps}"
export VS_BACKEND VS_HOST VS_PORT VS_DEVICE

PY="${PYTHON:-python3}"
if [[ -d .venv ]]; then PY=".venv/bin/python"; fi

# --workers 1 is deliberate. There is one GPU; a second worker process would
# load a second copy of every model and then fight for it.
exec caffeinate -i "$PY" -m uvicorn vision_service.app:app \
    --host "$VS_HOST" --port "$VS_PORT" \
    --workers 1 \
    --timeout-keep-alive 75 \
    --log-level info
