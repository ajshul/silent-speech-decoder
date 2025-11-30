#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/preprocess.sh [DATA_ROOT] [INDEX] [FEATURES_ROOT]
# Defaults align with docs/QUICKSTART.md.

ROOT=${1:-data/emg_data}
INDEX=${2:-results/index.parquet}
FEATURES_ROOT=${3:-results/features}
shift 3 || true
EXTRA_ARGS=("$@")

echo "EMG features -> ${FEATURES_ROOT}/emg"
python -m src.data.preprocessing \
  --mode emg \
  --root "${ROOT}" \
  --index "${INDEX}" \
  --out "${FEATURES_ROOT}/emg" \
  "${EXTRA_ARGS[@]}"

echo "Teacher features -> ${FEATURES_ROOT}/teacher"
python -m src.data.preprocessing \
  --mode teacher \
  --root "${ROOT}" \
  --index "${INDEX}" \
  --out "${FEATURES_ROOT}/teacher" \
  "${EXTRA_ARGS[@]}"
