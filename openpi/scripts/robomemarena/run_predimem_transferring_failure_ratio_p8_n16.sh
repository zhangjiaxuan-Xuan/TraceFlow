#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export UPPER_GPU="${UPPER_GPU:-0}"
export LOWER_GPU="${LOWER_GPU:-1}"
export UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}"
export LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}"
export ENV_WORKERS="${ENV_WORKERS:-64}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
export SEED="${SEED:-7}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export START_ROUND="${START_ROUND:-1}"
export END_ROUND="${END_ROUND:-10}"

exec bash "${ROOT}/scripts/robomemarena/run_predimem_transferring_inherited_cl_ratio_candidate.sh" \
  failure p8_n16
