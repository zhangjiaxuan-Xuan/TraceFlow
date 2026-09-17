#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:?Usage: $0 sequence|transferring|counting|occlusion}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
case "${SUITE}" in
  sequence) TASK_IDS="1,2,3,22" ;;
  transferring) TASK_IDS="18,19,25,26" ;;
  counting) TASK_IDS="6,7,8,9,10,15,16" ;;
  occlusion) TASK_IDS="4,5,11,12,13,14,17,20,21,23,24" ;;
  *) echo "Unknown suite: ${SUITE}" >&2; exit 2 ;;
esac

AOSS_ROOT="${AOSS_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5_balanced_v2_4096/predimem_4096_fp32}"
VLA_CKPT="${VLA_CKPT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/checkpoints/vla_alltask_pytorch}"
RUN_ROOT="${RUN_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5_balanced_v2_4096/eval_retrieval_audit/${SUITE}_seed${SEED:-50}_ep${EPISODES_PER_TASK:-10}_$(date -u +%Y%m%d_%H%M%S)}"
TASK_PROMPTS="${TASK_PROMPTS:-${ROOT}/../RoboMemArena/evaluation_benchmark/openpi_minimal_runtime/task_prompts.py}"

for path in \
  "${AOSS_ROOT}/heads/fusion/best.pt" \
  "${AOSS_ROOT}/memory/fusion/gpm_memory.index" \
  "${AOSS_ROOT}/memory/fusion/gpm_memory_meta_dense_frame_v3.pt" \
  "${AOSS_ROOT}/memory/shared/gpm_memory_actions.npz" \
  "${TASK_PROMPTS}"; do
  [[ -f "${path}" ]] || { echo "Missing balanced Full26 audit artifact: ${path}" >&2; exit 2; }
done

export AOSS_ROOT VLA_CKPT RUN_ROOT TASK_IDS
export VLA_CONFIG="${VLA_CONFIG:-pi05_robomemarena_all26_reactive}"
export ACTION_STATS="${ACTION_STATS:-${VLA_CKPT}/assets/robomemarena/all26_pi05_reactive/norm_stats.json}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/runtime/openpi_data}"
export MODE=v1 HEAD_VARIANTS=fusion MEMORY_ADMISSION=success
export MEMORY_TOP_K="${MEMORY_TOP_K:-16}"
export MEMORY_META_BASENAME=gpm_memory_meta_dense_frame_v3.pt
export MEMORY_ALIGNMENT_TAG=dense_frame_v3-real-query-audit
export MEMORY_ACTION_ALIGNMENT=dense_frame_v3
export MEMORY_EXACT_TASK_GATE=0
export MEMORY_ALLOWED_TASK_IDS=""
# cap<=0 disables clipping and is therefore unbounded. Keep retrieval active
# while making this diagnostic rollout effectively identical to Base.
export MEMORY_GUIDANCE_NORM_CAP=0.000001
export MEMORY_GUIDANCE_TOTAL_NORM_CAP=0.000001
export MEMORY_GUIDANCE_SUITE_GATE_PATH=""
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}" SEED="${SEED:-50}"
export UPPER_GPU="${UPPER_GPU:-0}" LOWER_GPU="${LOWER_GPU:-1}"
export UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}" LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}"
export ENV_WORKERS="${ENV_WORKERS:-32}"
export SAVE_VIDEO=0 RECORD_MEMORY_DATA=1 MEMORY_TRACE_LEVEL=light
export PREDIMEM_COMPONENT_TIMING=1

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  export PREFLIGHT_ONLY=1
  bash "${ROOT}/scripts/robomemarena/run_predimem_dual_gpu_batch8_env32.sh" v1
  echo "Real-query retrieval audit preflight passed: suite=${SUITE} tasks=${TASK_IDS}"
  exit 0
fi

bash "${ROOT}/scripts/robomemarena/run_predimem_dual_gpu_batch8_env32.sh" v1

"${PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}" \
  "${ROOT}/scripts/analysis/audit_arena_memory_retrieval.py" \
  --run-root "${RUN_ROOT}" --head fusion --task-prompts "${TASK_PROMPTS}" \
  --task-ids "${TASK_IDS}" --episodes all --call-stride 1 \
  --workers "${AUDIT_WORKERS:-32}" \
  --output "${RUN_ROOT}/real_eval_retrieval_audit.json"

echo "Real-query retrieval audit complete: ${RUN_ROOT}"
