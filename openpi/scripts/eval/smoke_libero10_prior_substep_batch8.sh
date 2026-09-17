#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

MEMORY_VARIANT="${MEMORY_VARIANT:-success}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

MEMORY_VARIANT="${MEMORY_VARIANT}" \
RUN_ROOT="${RUN_ROOT:-logs/smoke_prior_substep_${MEMORY_VARIANT}_batch8_${RUN_ID}}" \
NUM_TRIALS_PER_TASK=8 \
TASK_IDS_CSV="${TASK_IDS_CSV:-0}" \
SAVE_VIDEOS="${SAVE_VIDEOS:-0}" \
SAVE_EPISODE_DATA=0 \
BATCH_SIZE=8 \
bash scripts/eval/run_libero10_prior_substep_batch8.sh
