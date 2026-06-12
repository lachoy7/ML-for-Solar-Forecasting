#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

INPUT="${1:-data/raw/daily/21.48722, -158.02806.csv}"
OUTPUT="${2:-data/train_x.csv}"

python -m ml_for_solar.preprocess \
  --input "$INPUT" \
  --output "$OUTPUT"
