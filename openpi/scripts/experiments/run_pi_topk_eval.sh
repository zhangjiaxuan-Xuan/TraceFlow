#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
cd "${OPENPI_ROOT}"

METHOD="${METHOD:?METHOD is required and must be guidance_negative}"
case "${METHOD}" in
  guidance_negative) use_lcm=0; guidance_only=1; use_negative=1 ;;
  *) echo "Invalid METHOD=${METHOD}" >&2; exit 2 ;;
esac

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${$}}"
MEMORY_TOP_K="${MEMORY_TOP_K:?MEMORY_TOP_K is required}"
GPU="${GPU:-0}"
PORT="${PORT:-8200}"
SUITES="${SUITES:-libero_10}"
TASK_IDS_CSV="${TASK_IDS_CSV:-}"

[[ "${MEMORY_TOP_K}" =~ ^(1|4|8|16|32)$ ]] || { echo "Invalid MEMORY_TOP_K=${MEMORY_TOP_K}" >&2; exit 2; }
[[ "${SEED:-7}" == "7" ]] || { echo "SEED is fixed to 7" >&2; exit 2; }
[[ "${EPISODE_START:-100}" == "100" ]] || { echo "EPISODE_START is fixed to 100" >&2; exit 2; }
[[ "${NUM_TRIALS_PER_TASK:-50}" == "50" ]] || { echo "NUM_TRIALS_PER_TASK is fixed to 50" >&2; exit 2; }
[[ "${BATCH_SIZE:-8}" == "8" ]] || { echo "BATCH_SIZE is fixed to 8" >&2; exit 2; }

required=(
  "${POLICY_DIR:?POLICY_DIR is required}/model.safetensors"
  "${TASK_HEAD_CKPT:?TASK_HEAD_CKPT is required}"
  "${MEMORY_META_PATH:?MEMORY_META_PATH is required}"
  "${FAISS_INDEX_PATH:?FAISS_INDEX_PATH is required}"
  "${MEMORY_ACTIONS_PATH:?MEMORY_ACTIONS_PATH is required}"
)
required+=(
  "${NEGATIVE_MEMORY_META_PATH:?NEGATIVE_MEMORY_META_PATH is required}"
  "${NEGATIVE_FAISS_INDEX_PATH:?NEGATIVE_FAISS_INDEX_PATH is required}"
  "${NEGATIVE_MEMORY_ACTIONS_PATH:?NEGATIVE_MEMORY_ACTIONS_PATH is required}"
)
[[ "${NEGATIVE_MEMORY_TOP_K:?NEGATIVE_MEMORY_TOP_K is required}" =~ ^(1|4|8)$ ]] || {
  echo "Invalid NEGATIVE_MEMORY_TOP_K=${NEGATIVE_MEMORY_TOP_K}" >&2
  exit 2
}
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "Missing fixed artifact: ${path}" >&2; exit 1; }
done

IFS=',' read -r -a suite_array <<<"${SUITES}"
total_clients="${TOTAL_LIBERO_CLIENTS:-8}"
[[ "${total_clients}" == "8" ]] || {
  echo "TOTAL_LIBERO_CLIENTS is fixed to 8 for this protocol, got ${total_clients}" >&2
  exit 2
}
(( total_clients >= ${#suite_array[@]} )) || {
  echo "TOTAL_LIBERO_CLIENTS=${total_clients} is smaller than suite count ${#suite_array[@]}" >&2
  exit 2
}
clients_per_suite=$(( total_clients / ${#suite_array[@]} ))
(( clients_per_suite * ${#suite_array[@]} == total_clients )) || {
  echo "TOTAL_LIBERO_CLIENTS=${total_clients} must be divisible by suite count ${#suite_array[@]}" >&2
  exit 2
}
eval_logs=()
for suite in "${suite_array[@]}"; do
  eval_logs+=("${RUN_ROOT}/eval/${suite}.jsonl")
done
resume="${RESUME:-auto}"
if [[ "${resume}" == "auto" ]]; then
  resume=0
  all_logs_exist=1
  for log in "${eval_logs[@]}"; do
    [[ -f "${log}" ]] || all_logs_exist=0
  done
  [[ "${all_logs_exist}" == "1" ]] && resume=1
fi
if [[ "${resume}" == "0" && -d "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty RUN_ROOT without resume: ${RUN_ROOT}" >&2
  exit 1
fi

env \
NEGATIVE_MEMORY_META_PATH="${NEGATIVE_MEMORY_META_PATH}" \
NEGATIVE_FAISS_INDEX_PATH="${NEGATIVE_FAISS_INDEX_PATH}" \
NEGATIVE_MEMORY_ACTIONS_PATH="${NEGATIVE_MEMORY_ACTIONS_PATH}" \
NEGATIVE_MEMORY_TOP_K="${NEGATIVE_MEMORY_TOP_K}" \
CUDA_VISIBLE_DEVICES="${GPU}" \
SERVER_CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES:-${GPU}}" \
CLIENT_CUDA_VISIBLE_DEVICES="${CLIENT_CUDA_VISIBLE_DEVICES:-${GPU}}" \
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}" \
LIBERO_CONFIG_PATH="${RUN_ROOT}/libero_config" \
NUMBA_CACHE_DIR="${RUN_ROOT}/cache/numba" \
MPLCONFIGDIR="${RUN_ROOT}/cache/matplotlib" \
PORT="${PORT}" \
POLICY_DIR="${POLICY_DIR}" \
TASK_HEAD_CKPT="${TASK_HEAD_CKPT}" \
MEMORY_META_PATH="${MEMORY_META_PATH}" \
FAISS_INDEX_PATH="${FAISS_INDEX_PATH}" \
MEMORY_ACTIONS_PATH="${MEMORY_ACTIONS_PATH}" \
LOG_DIR="${RUN_ROOT}/eval" \
RUN_ID="${RUN_ID}" \
RESULTS_TXT="${RUN_ROOT}/eval/results.txt" \
VIDEO_ROOT="${RUN_ROOT}/eval/videos" \
EPISODE_DATA_ROOT="" \
TASK_IDS_CSV="${TASK_IDS_CSV}" \
NUM_TRIALS_PER_TASK=50 \
EPISODE_START=100 \
SEED=7 \
SAVE_VIDEOS=0 \
FAIL_ON_EPISODE_ERROR=1 \
RESUME="${resume}" \
INFERENCE_BATCH_SIZE=8 \
INFERENCE_BATCH_WAIT_MS=20 \
LIBERO_CLIENTS_PER_SUITE="${clients_per_suite}" \
USE_MEMORY=1 \
MEMORY_REFRESH_EVERY=1 \
MEMORY_TOP_K="${MEMORY_TOP_K}" \
USE_LCM="${use_lcm}" \
USE_MEMORY_GUIDANCE=0 \
MEMORY_GUIDANCE_ONLY="${guidance_only}" \
MEMORY_GUIDANCE_TIME_VERSION=v1 \
MEMORY_PRIOR_SUBSTEP_GUIDANCE=0 \
MEMORY_GUIDANCE_NUM_STEPS=10 \
MEMORY_GUIDANCE_LAMBDA_MAX=0.20 \
MEMORY_GUIDANCE_T_CUT=0.30 \
MEMORY_GUIDANCE_SIGMA=0.30 \
MEMORY_GUIDANCE_NORM_CAP=0.20 \
MEMORY_GUIDANCE_MIN_SIMILARITY=-1.0 \
USE_NEGATIVE_GUIDANCE="${use_negative}" \
NEGATIVE_MEMORY_MIN_SIMILARITY=-1.0 \
NEGATIVE_MEMORY_MIN_CONFIDENCE=0.0 \
NEGATIVE_GUIDANCE_BETA=0.10 \
NEGATIVE_GUIDANCE_SIGMA=0.30 \
NEGATIVE_GUIDANCE_NORM_CAP=0.10 \
MEMORY_GUIDANCE_TOTAL_NORM_CAP=0.20 \
MEMORY_GUIDANCE_TRACE_DIR="${RUN_ROOT}/traces/${RUN_ID}" \
MEMORY_GUIDANCE_TRACE_LEVEL=light \
bash scripts/eval/run_libero_eval.sh "${suite_array[@]}"

for suite in "${suite_array[@]}"; do
  case "${suite}" in
    libero_spatial|libero_object|libero_goal|libero_10) expected_tasks=10 ;;
    libero_90) expected_tasks=90 ;;
    *) echo "Unsupported top-k suite: ${suite}" >&2; exit 2 ;;
  esac
  "${PYTHON}" scripts/analysis/summarize_libero_eval_run.py \
    --eval-log "${RUN_ROOT}/eval/${suite}.jsonl" \
    --output-dir "${RUN_ROOT}/indexes/${suite}" \
    --expected-tasks "${expected_tasks}" \
    --expected-episodes-per-task 50
done
