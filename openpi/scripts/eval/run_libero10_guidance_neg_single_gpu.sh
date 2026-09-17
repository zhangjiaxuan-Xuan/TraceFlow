#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

GPU="${GPU:-0}"
PORT="${PORT:-8200}"
POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-logs/libero10_guidance_pos_neg_single_gpu_${RUN_ID}}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-100}"
TASK_IDS_CSV="${TASK_IDS_CSV:-0,1,2,3,4,5,6,7,8,9}"
EXPECTED_TASKS="$(awk -F, '{print NF}' <<<"${TASK_IDS_CSV}")"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
SAVE_EPISODE_DATA="${SAVE_EPISODE_DATA:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"

if [[ "${BATCH_SIZE}" -le 0 || "${BATCH_SIZE}" -gt "${NUM_TRIALS_PER_TASK}" ]]; then
  echo "BATCH_SIZE must be between 1 and NUM_TRIALS_PER_TASK." >&2
  exit 1
fi

if [[ -d "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to mix data into non-empty RUN_ROOT: ${RUN_ROOT}" >&2
  exit 1
fi

episode_data_root=""
if [[ "${SAVE_EPISODE_DATA}" == "1" ]]; then
  episode_data_root="${RUN_ROOT}/episode_data"
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
LOG_DIR="${RUN_ROOT}/eval" \
RESULTS_TXT="${RUN_ROOT}/eval/results.txt" \
VIDEO_ROOT="${RUN_ROOT}/eval/videos" \
EPISODE_DATA_ROOT="${episode_data_root}" \
EPISODE_DATA_MODE=all \
TASK_IDS_CSV="${TASK_IDS_CSV}" \
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
SAVE_VIDEOS="${SAVE_VIDEOS}" \
INFERENCE_BATCH_SIZE="${BATCH_SIZE}" \
INFERENCE_BATCH_WAIT_MS=20 \
LIBERO_CLIENTS_PER_SUITE="${BATCH_SIZE}" \
USE_MEMORY=1 \
USE_LCM=0 \
MEMORY_GUIDANCE_ONLY=1 \
MEMORY_GUIDANCE_NUM_STEPS=10 \
USE_NEGATIVE_GUIDANCE=1 \
NEGATIVE_MEMORY_MIN_SIMILARITY="${NEGATIVE_MEMORY_MIN_SIMILARITY:-0.975}" \
NEGATIVE_MEMORY_TOP_K="${NEGATIVE_MEMORY_TOP_K:-4}" \
NEGATIVE_MEMORY_MIN_CONFIDENCE="${NEGATIVE_MEMORY_MIN_CONFIDENCE:-0.75}" \
NEGATIVE_GUIDANCE_NORM_CAP="${NEGATIVE_GUIDANCE_NORM_CAP:-0.10}" \
bash scripts/eval/run_libero_eval.sh libero_10

validate_args=()
if [[ "${SAVE_VIDEOS}" == "1" ]]; then validate_args+=(--require-videos); fi
if [[ "${SAVE_EPISODE_DATA}" == "1" ]]; then validate_args+=(--require-trajectories); fi
/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python scripts/analysis/summarize_libero_eval_run.py \
  --eval-log "${RUN_ROOT}/eval/libero_10.jsonl" \
  --output-dir "${RUN_ROOT}/indexes" \
  --expected-tasks "${EXPECTED_TASKS}" \
  --expected-episodes-per-task "${NUM_TRIALS_PER_TASK}" \
  "${validate_args[@]}"

echo "Run complete: ${RUN_ROOT}"
