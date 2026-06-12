#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

NSRDB_PATH="${1:-data/raw/nsrdb}"
OUTPUT_DIR="${2:-data/raw/daily}"

python -m ml_for_solar.wrangling \
  --nsrdb-path "$NSRDB_PATH" \
  --output-dir "$OUTPUT_DIR"
