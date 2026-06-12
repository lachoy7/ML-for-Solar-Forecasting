#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

CSV="${1:-data/train_x.csv}"
NUM_WORKERS="${2:-0}"

python -m ml_for_solar.dataloading \
  --csv "$CSV" \
  --num-workers "$NUM_WORKERS"
