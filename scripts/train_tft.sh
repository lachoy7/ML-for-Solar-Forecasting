#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

MAX_EPOCHS="${1:-50}"
ACCELERATOR="${2:-auto}"
NUM_WORKERS="${3:-0}"

python -m ml_for_solar.tft \
  --max-epochs "$MAX_EPOCHS" \
  --accelerator "$ACCELERATOR" \
  --num-workers "$NUM_WORKERS"
