#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

PREDICTIONS="${1:-outputs/predictions/tft_test_predictions.csv}"
REFERENCE="${2:-data/train_x_old.csv}"

python -m ml_for_solar.evaluation \
  --predictions "$PREDICTIONS" \
  --reference "$REFERENCE"
