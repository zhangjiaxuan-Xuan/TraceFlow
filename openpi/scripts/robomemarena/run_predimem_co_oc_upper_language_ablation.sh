#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: $0 counting|occlusion baseline|knnlm}"
ARM="${2:?Usage: $0 counting|occlusion baseline|knnlm}"
case "${SUITE}" in counting|occlusion) ;; *) echo "SUITE must be counting or occlusion" >&2; exit 2 ;; esac
case "${ARM}" in baseline|knnlm) ;; *) echo "ARM must be baseline or knnlm" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

export HEAD_VARIANTS="lower"
export MEMORY_GUIDANCE_DISABLE_SUITE_GATE="1"
export UPPER_GUIDANCE_ENABLED="$([[ "${ARM}" == "knnlm" ]] && echo 1 || echo 0)"
export UPPER_GUIDANCE_TOP_K="16"
export UPPER_GUIDANCE_TEMPERATURE="0.07"
export UPPER_GUIDANCE_MIN_TASK_PURITY="0.5"
export UPPER_GUIDANCE_MIN_SUBTASK_CONFIDENCE="0.8"
export UPPER_GUIDANCE_INTERPOLATION="0.8"
export UPPER_GUIDANCE_BANK_PER_PRIMITIVE="16"
export UPPER_GUIDANCE_BANK_SEED="17"
export UPPER_GUIDANCE_DEVICE="cpu"

export SEED="${SEED:-50}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-51}"
export UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}"
export LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}"
export ENV_WORKERS="${ENV_WORKERS:-64}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export RECORD_MEMORY_DATA="${RECORD_MEMORY_DATA:-0}"
export RUN_ROOT="${RUN_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/eval_upper_knnlm_co_oc/${SUITE}_${ARM}_v1_lower_seed${SEED}_${STAMP}}"

exec bash "${ROOT}/scripts/robomemarena/run_predimem_suite_conditioned_dual_gpu.sh" "${SUITE}"
