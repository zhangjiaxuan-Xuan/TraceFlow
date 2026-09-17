#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
cd "${OPENPI_ROOT}"

POLICY_DIR="${POLICY_DIR:?POLICY_DIR is required}"
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:?TASK_HEAD_CKPT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
GPU="${GPU:-0}"
PORT="${PORT:-8200}"
SEED="${SEED:-7}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-100}"
RESUME="${RESUME:-auto}"

[[ "${SEED}" == "7" ]] || { echo "SEED is fixed to 7" >&2; exit 2; }
[[ "${BATCH_SIZE}" == "8" ]] || { echo "BATCH_SIZE is fixed to 8" >&2; exit 2; }

required=(
  "${POLICY_DIR}/model.safetensors"
  "${TASK_HEAD_CKPT}"
  "${MEMORY_META_PATH:?MEMORY_META_PATH is required}"
  "${FAISS_INDEX_PATH:?FAISS_INDEX_PATH is required}"
  "${MEMORY_ACTIONS_PATH:?MEMORY_ACTIONS_PATH is required}"
  "${NEGATIVE_MEMORY_META_PATH:?NEGATIVE_MEMORY_META_PATH is required}"
  "${NEGATIVE_FAISS_INDEX_PATH:?NEGATIVE_FAISS_INDEX_PATH is required}"
  "${NEGATIVE_MEMORY_ACTIONS_PATH:?NEGATIVE_MEMORY_ACTIONS_PATH is required}"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "Missing fixed experiment artifact: ${path}" >&2; exit 1; }
done

eval_log="${RUN_ROOT}/eval/libero_10.jsonl"
if [[ "${RESUME}" == "auto" ]]; then
  RESUME=0
  [[ -f "${eval_log}" ]] && RESUME=1
fi
if [[ "${RESUME}" == "0" && -d "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty RUN_ROOT without resume: ${RUN_ROOT}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU}" \
SERVER_CUDA_VISIBLE_DEVICES="${GPU}" \
CLIENT_CUDA_VISIBLE_DEVICES="${GPU}" \
MUJOCO_EGL_DEVICE_ID=0 \
LIBERO_CONFIG_PATH="${RUN_ROOT}/libero_config" \
NUMBA_CACHE_DIR="${RUN_ROOT}/cache/numba" \
MPLCONFIGDIR="${RUN_ROOT}/cache/matplotlib" \
PORT="${PORT}" \
POLICY_DIR="${POLICY_DIR}" \
TASK_HEAD_CKPT="${TASK_HEAD_CKPT}" \
MEMORY_META_PATH="${MEMORY_META_PATH}" \
FAISS_INDEX_PATH="${FAISS_INDEX_PATH}" \
MEMORY_ACTIONS_PATH="${MEMORY_ACTIONS_PATH}" \
NEGATIVE_MEMORY_META_PATH="${NEGATIVE_MEMORY_META_PATH}" \
NEGATIVE_FAISS_INDEX_PATH="${NEGATIVE_FAISS_INDEX_PATH}" \
NEGATIVE_MEMORY_ACTIONS_PATH="${NEGATIVE_MEMORY_ACTIONS_PATH}" \
LOG_DIR="${RUN_ROOT}/eval" \
RESULTS_TXT="${RUN_ROOT}/eval/results.txt" \
VIDEO_ROOT="${RUN_ROOT}/eval/videos" \
EPISODE_DATA_ROOT="" \
TASK_IDS_CSV="0,1,2,3,4,5,6,7,8,9" \
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
SEED=7 \
SAVE_VIDEOS="${SAVE_VIDEOS:-0}" \
RESUME="${RESUME}" \
INFERENCE_BATCH_SIZE=8 \
INFERENCE_BATCH_WAIT_MS=20 \
LIBERO_CLIENTS_PER_SUITE=8 \
USE_MEMORY=1 \
MEMORY_REFRESH_EVERY=1 \
MEMORY_TOP_K=8 \
USE_LCM=0 \
USE_MEMORY_GUIDANCE=0 \
MEMORY_GUIDANCE_ONLY=1 \
MEMORY_GUIDANCE_TIME_VERSION=v0 \
MEMORY_PRIOR_SUBSTEP_GUIDANCE=0 \
MEMORY_GUIDANCE_NUM_STEPS=10 \
MEMORY_GUIDANCE_LAMBDA_MAX=0.20 \
MEMORY_GUIDANCE_T_CUT=0.30 \
MEMORY_GUIDANCE_SIGMA=0.30 \
MEMORY_GUIDANCE_NORM_CAP=0.20 \
MEMORY_GUIDANCE_MIN_SIMILARITY=-1.0 \
USE_NEGATIVE_GUIDANCE=1 \
NEGATIVE_MEMORY_MIN_SIMILARITY=-1.0 \
NEGATIVE_MEMORY_MIN_CONFIDENCE=0.0 \
NEGATIVE_MEMORY_TOP_K=8 \
NEGATIVE_GUIDANCE_BETA=0.10 \
NEGATIVE_GUIDANCE_SIGMA=0.30 \
NEGATIVE_GUIDANCE_NORM_CAP=0.10 \
MEMORY_GUIDANCE_TOTAL_NORM_CAP=0.20 \
bash scripts/eval/run_libero_eval.sh libero_10

"${PYTHON}" scripts/analysis/summarize_libero_eval_run.py \
  --eval-log "${eval_log}" \
  --output-dir "${RUN_ROOT}/indexes" \
  --expected-tasks 10 \
  --expected-episodes-per-task "${NUM_TRIALS_PER_TASK}"
