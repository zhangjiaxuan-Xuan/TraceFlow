#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
cd "${OPENPI_ROOT}"

SUITE="${SUITE:?SUITE is required}"
EVAL_MODE="${EVAL_MODE:?EVAL_MODE must be base or fixed_prior}"
POLICY_DIR="${POLICY_DIR:?POLICY_DIR is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
GPU="${GPU:-0}"
PORT="${PORT:-8600}"
SEED="${SEED:-7}"
EPISODE_START="${EPISODE_START:-0}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
FIXED_PRIOR_TOP_K="${FIXED_PRIOR_TOP_K:-8}"

case "${SUITE}" in
  libero_10) task_count=10 ;;
  libero_90) task_count=90 ;;
  *) echo "SUITE must be libero_10 or libero_90" >&2; exit 2 ;;
esac
case "${EVAL_MODE}" in
  base) use_memory=0; use_lcm=0 ;;
  fixed_prior) use_memory=1; use_lcm=1 ;;
  *) echo "EVAL_MODE must be base or fixed_prior" >&2; exit 2 ;;
esac
[[ "${BATCH_SIZE}" == "8" ]] || { echo "BATCH_SIZE must be 8" >&2; exit 2; }
[[ "${FIXED_PRIOR_TOP_K}" =~ ^[1-9][0-9]*$ ]] || { echo "FIXED_PRIOR_TOP_K must be positive" >&2; exit 2; }
[[ -f "${POLICY_DIR}/model.safetensors" ]] || { echo "Missing policy model: ${POLICY_DIR}/model.safetensors" >&2; exit 1; }

if [[ "${EVAL_MODE}" == "fixed_prior" ]]; then
  required=(
    "${TASK_HEAD_CKPT:?TASK_HEAD_CKPT is required for fixed_prior}"
    "${MEMORY_META_PATH:?MEMORY_META_PATH is required for fixed_prior}"
    "${FAISS_INDEX_PATH:?FAISS_INDEX_PATH is required for fixed_prior}"
    "${MEMORY_ACTIONS_PATH:?MEMORY_ACTIONS_PATH is required for fixed_prior}"
    "${OPENPI_ROOT}/checkpoints/lcm.pt"
  )
  for path in "${required[@]}"; do
    [[ -f "${path}" ]] || { echo "Missing fixed-prior artifact: ${path}" >&2; exit 1; }
  done
fi

task_ids_csv="$(seq -s, 0 $((task_count - 1)))"
eval_log="${RUN_ROOT}/eval/${SUITE}.jsonl"
resume=0
if [[ -f "${eval_log}" ]]; then
  resume=1
elif [[ -d "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty RUN_ROOT without an eval log: ${RUN_ROOT}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}/eval" "${RUN_ROOT}/indexes" \
  "${RUN_ROOT}/libero_config" "${RUN_ROOT}/cache/numba" "${RUN_ROOT}/cache/matplotlib"

CUDA_VISIBLE_DEVICES="${GPU}" \
SERVER_CUDA_VISIBLE_DEVICES="${GPU}" \
CLIENT_CUDA_VISIBLE_DEVICES="${GPU}" \
MUJOCO_EGL_DEVICE_ID=0 \
LIBERO_CONFIG_PATH="${RUN_ROOT}/libero_config" \
NUMBA_CACHE_DIR="${RUN_ROOT}/cache/numba" \
MPLCONFIGDIR="${RUN_ROOT}/cache/matplotlib" \
PORT="${PORT}" \
POLICY_DIR="${POLICY_DIR}" \
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-}" \
MEMORY_META_PATH="${MEMORY_META_PATH:-}" \
FAISS_INDEX_PATH="${FAISS_INDEX_PATH:-}" \
MEMORY_ACTIONS_PATH="${MEMORY_ACTIONS_PATH:-}" \
MEMORY_TOP_K="${FIXED_PRIOR_TOP_K}" \
LOG_DIR="${RUN_ROOT}/eval" \
RESULTS_TXT="${RUN_ROOT}/eval/results.txt" \
VIDEO_ROOT="${RUN_ROOT}/videos" \
EPISODE_DATA_ROOT="" \
TASK_IDS_CSV="${task_ids_csv}" \
NUM_TRIALS_PER_TASK="${EPISODES_PER_TASK}" \
EPISODE_START="${EPISODE_START}" \
SEED="${SEED}" \
SAVE_VIDEOS=0 \
RESUME="${resume}" \
INFERENCE_BATCH_SIZE=8 \
INFERENCE_BATCH_WAIT_MS=20 \
LIBERO_CLIENTS_PER_SUITE=8 \
USE_MEMORY="${use_memory}" \
USE_LCM="${use_lcm}" \
USE_MEMORY_GUIDANCE=0 \
MEMORY_GUIDANCE_ONLY=0 \
MEMORY_PRIOR_SUBSTEP_GUIDANCE=0 \
USE_NEGATIVE_GUIDANCE=0 \
  bash scripts/eval/run_libero_eval.sh "${SUITE}"

"${PYTHON}" scripts/analysis/summarize_libero_eval_run.py \
  --eval-log "${eval_log}" \
  --output-dir "${RUN_ROOT}/indexes" \
  --expected-tasks "${task_count}" \
  --expected-episodes-per-task "${EPISODES_PER_TASK}"

