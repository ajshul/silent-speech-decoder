#!/usr/bin/env bash
set -euo pipefail

# Example:
# ./scripts/eval.sh --checkpoint results/checkpoints/mps_fast_ctc/best.pt --splits voiced_parallel_data silent_parallel_data

python -m src.evaluation.evaluate "$@"
