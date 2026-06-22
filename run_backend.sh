#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

cd "$ROOT_DIR/backend"
"$PYTHON_BIN" -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload --no-access-log
