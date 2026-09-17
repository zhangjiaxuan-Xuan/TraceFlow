#!/usr/bin/env bash
set -euo pipefail

SAMPLE_ID=${1:?Usage: $0 sample0|sample1|sample2|sample3|sample4|sample5}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FULL26_ROOT="${FULL26_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32}"

case "${SAMPLE_ID}" in
  # Default center point.
  sample0) HORIZON=4; CONFIRMATIONS=2; SIM_TOL=0.02; PURITY_TOL=0.10; REINFORCEMENTS=2 ;;
  # Fast, permissive feedback.
  sample1) HORIZON=3; CONFIRMATIONS=1; SIM_TOL=0.04; PURITY_TOL=0.20; REINFORCEMENTS=1 ;;
  # Short and conservative.
  sample2) HORIZON=3; CONFIRMATIONS=2; SIM_TOL=0.01; PURITY_TOL=0.05; REINFORCEMENTS=1 ;;
  # Persistent center-tolerance feedback.
  sample3) HORIZON=6; CONFIRMATIONS=2; SIM_TOL=0.02; PURITY_TOL=0.10; REINFORCEMENTS=3 ;;
  # Slow confirmation with permissive drift tolerance.
  sample4) HORIZON=6; CONFIRMATIONS=3; SIM_TOL=0.04; PURITY_TOL=0.20; REINFORCEMENTS=2 ;;
  # Long window but strict retrieval consistency.
  sample5) HORIZON=6; CONFIRMATIONS=2; SIM_TOL=0.01; PURITY_TOL=0.05; REINFORCEMENTS=2 ;;
  *) echo "Unknown sample: ${SAMPLE_ID}" >&2; exit 2 ;;
esac

export TASK_IDS="${TASK_IDS:-6,7,8,9,10,15,16}"
export MEMORY_ALLOWED_TASK_IDS="${MEMORY_ALLOWED_TASK_IDS:-6,7,8,9,10,15,16}"
export UPPER_GUIDANCE_ALLOWED_TASK_IDS="${UPPER_GUIDANCE_ALLOWED_TASK_IDS:-6,7,8,9,10,15,16}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
export SEED="${SEED:-50}"
export UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}"
export LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}"
export ENV_WORKERS="${ENV_WORKERS:-64}"
export SAVE_VIDEO="${SAVE_VIDEO:-0}"
export RECORD_MEMORY_DATA=0

export UPPER_GUIDANCE_FEEDBACK_MODE="${UPPER_GUIDANCE_FEEDBACK_MODE:-control}"
export UPPER_GUIDANCE_FEEDBACK_HORIZON="${HORIZON}"
export UPPER_GUIDANCE_FEEDBACK_CONFIRMATIONS="${CONFIRMATIONS}"
export UPPER_GUIDANCE_FEEDBACK_SIMILARITY_TOLERANCE="${SIM_TOL}"
export UPPER_GUIDANCE_FEEDBACK_PURITY_TOLERANCE="${PURITY_TOL}"
export UPPER_GUIDANCE_FEEDBACK_MAX_REINFORCEMENTS="${REINFORCEMENTS}"

RUN_NAME="${SAMPLE_ID}_h${HORIZON}_c${CONFIRMATIONS}_s${SIM_TOL}_p${PURITY_TOL}_r${REINFORCEMENTS}_seed${SEED}"
export RUN_ROOT="${RUN_ROOT:-${FULL26_ROOT}/eval/feedback_bayes_round1/${RUN_NAME}}"

exec bash "${ROOT}/scripts/robomemarena/run_predimem_counting_history_knn_self.sh" fusion_fusion_action
