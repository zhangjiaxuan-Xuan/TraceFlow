#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

export UPPER_GUIDANCE_HISTORY_ENABLED="1"
export UPPER_GUIDANCE_HISTORY_LOCK_OBSERVATIONS="${UPPER_GUIDANCE_HISTORY_LOCK_OBSERVATIONS:-2}"
export UPPER_GUIDANCE_HISTORY_ADVANCE_CONFIRMATIONS="${UPPER_GUIDANCE_HISTORY_ADVANCE_CONFIRMATIONS:-2}"
export UPPER_GUIDANCE_HISTORY_MAX_ROLLBACK="${UPPER_GUIDANCE_HISTORY_MAX_ROLLBACK:-0.10}"
export UPPER_GUIDANCE_HISTORY_MAX_ADVANCE="${UPPER_GUIDANCE_HISTORY_MAX_ADVANCE:-0.35}"
export RUN_ROOT="${RUN_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/eval_upper_knnlm_history/counting_tdense5_upper_history_seed${SEED:-50}_${STAMP}}"

exec bash "${ROOT}/scripts/robomemarena/run_predimem_counting_upper_knnlm_only.sh"
