#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [[ -z "${NREL_API_KEY:-}" || -z "${NREL_EMAIL:-}" ]]; then
  echo "Set NREL_API_KEY and NREL_EMAIL before running this script." >&2
  exit 1
fi

python -m ml_for_solar.fetch_nsrdb --location-ids "$@"
