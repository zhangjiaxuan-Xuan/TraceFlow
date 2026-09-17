#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
AOSS_ROOT="${AOSS_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5_balanced_v2_4096/predimem_4096_fp32}"
VLA_CKPT="${VLA_CKPT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/checkpoints/vla_alltask_pytorch}"
RUN_ROOT="${RUN_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5_balanced_v2_4096/online_query_monitor/seed${SEED:-50}_$(date -u +%Y%m%d_%H%M%S)}"

# One full inference batch. CO/OC receive six of the eight diagnostic slots.
TASK_IDS="${TASK_IDS:-1,19,6,15,4,12,13,17}"
HEAD="${AOSS_ROOT}/heads/fusion/best.pt"
META="${AOSS_ROOT}/memory/fusion/gpm_memory_meta_dense_frame_v3.pt"
INDEX="${AOSS_ROOT}/memory/fusion/gpm_memory.index"
ACTIONS="${AOSS_ROOT}/memory/shared/gpm_memory_actions.npz"

for path in "${PYTHON}" "${HEAD}" "${META}" "${INDEX}" "${ACTIONS}"; do
  [[ -f "${path}" ]] || { echo "Missing online-monitor artifact: ${path}" >&2; exit 2; }
done

export AOSS_ROOT VLA_CKPT RUN_ROOT TASK_IDS
export VLA_CONFIG="${VLA_CONFIG:-pi05_robomemarena_all26_reactive}"
export ACTION_STATS="${ACTION_STATS:-${VLA_CKPT}/assets/robomemarena/all26_pi05_reactive/norm_stats.json}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/runtime/openpi_data}"
export MODE=v1 HEAD_VARIANTS=fusion MEMORY_ADMISSION=success
export MEMORY_TOP_K=16 MEMORY_META_BASENAME=gpm_memory_meta_dense_frame_v3.pt
export MEMORY_ACTION_ALIGNMENT=dense_frame_v3 MEMORY_ALIGNMENT_TAG=dense-frame-v3-online-query-monitor-v1
export MEMORY_ALLOWED_TASK_IDS="" MEMORY_EXACT_TASK_GATE=0
# cap<=0 means unbounded in V1. A tiny positive cap preserves retrieval while
# making the monitored rollout effectively Base-policy behavior.
export MEMORY_GUIDANCE_NORM_CAP=0.000001 MEMORY_GUIDANCE_TOTAL_NORM_CAP=0.000001
export MEMORY_GUIDANCE_SUITE_GATE_PATH=""
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}" SEED="${SEED:-50}"
export MAX_STEPS="${MAX_STEPS:-120}" POST_GOAL_STEPS=0
export UPPER_GPU="${UPPER_GPU:-0}" LOWER_GPU="${LOWER_GPU:-1}"
export UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-8}" LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-8}"
export ENV_WORKERS="${ENV_WORKERS:-8}" SAVE_VIDEO=0
export RECORD_MEMORY_DATA=1 MEMORY_TRACE_LEVEL=bank
export TRACE_LOCAL_WORKERS="${TRACE_LOCAL_WORKERS:-8}" TRACE_TRANSFER_WORKERS="${TRACE_TRANSFER_WORKERS:-8}"
export PREDIMEM_COMPONENT_TIMING=1

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  export PREFLIGHT_ONLY=1
  bash "${ROOT}/scripts/robomemarena/run_predimem_dual_gpu_batch8_env32.sh" v1
  echo "Online query monitor preflight passed: tasks=${TASK_IDS} run=${RUN_ROOT}"
  exit 0
fi

if [[ "${POSTPROCESS_ONLY:-0}" != "1" ]]; then
  bash "${ROOT}/scripts/robomemarena/run_predimem_dual_gpu_batch8_env32.sh" v1
fi

SERVER_LOG="${RUN_ROOT}/fusion/lower_server.log"
grep -F "task_head=${HEAD}" "${SERVER_LOG}" >/dev/null || {
  echo "Runtime did not load the requested head: ${HEAD}" >&2; exit 3;
}
grep -F "action_alignment=dense_frame_v3" "${SERVER_LOG}" >/dev/null || {
  echo "Runtime alignment mismatch; expected dense_frame_v3" >&2; exit 3;
}

"${PYTHON}" "${ROOT}/scripts/analysis/audit_predimem_online_query_replay.py" \
  --run-root "${RUN_ROOT}" --head-variant fusion \
  --head-checkpoint "${HEAD}" --memory-meta "${META}" --faiss-index "${INDEX}" \
  --top-k 16 --device cpu --online-only \
  --output "${RUN_ROOT}/fusion/online_query_replay_audit.json" \
  2>&1 | tee "${RUN_ROOT}/fusion/online_query_replay.log"

echo "Online query monitor complete: ${RUN_ROOT}"
