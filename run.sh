#!/usr/bin/env bash
# Start the vision service in the foreground.
#
# You normally don't call this directly: launchd does (see `make start`).
# `make run` calls it too, for watching the service in a terminal.
set -euo pipefail

cd "$(dirname "$0")"

# Local overrides (VS_PORT, VS_DEVICE, ...). Optional and gitignored.
if [[ -f .env ]]; then set -a; source .env; set +a; fi

: "${VS_HOST:=0.0.0.0}"
: "${VS_PORT:=8080}"
: "${VS_DEVICE:=mps}"
export VS_HOST VS_PORT VS_DEVICE

if [[ ! -x .venv/bin/python ]]; then
    echo "run.sh: no .venv here — run 'make install' first" >&2
    exit 1
fi

# caffeinate is not optional in practice: a sleeping Mac mini looks exactly like
# bad wifi from the Pi's side, and that is a miserable thing to debug at 11pm.
# -w $$ keeps the Mac awake for exactly as long as this PID lives. `exec` below
# keeps the same PID, so caffeinate exits with the server, and launchd's
# stop/restart signals go straight to uvicorn for a clean shutdown.
caffeinate -i -w $$ &

# --workers 1 is deliberate. There is one GPU; a second worker process would
# load a second copy of every model and then fight for it.
exec .venv/bin/python -m uvicorn vision_service.app:app \
    --host "$VS_HOST" --port "$VS_PORT" \
    --workers 1 \
    --timeout-keep-alive 75 \
    --log-level info
