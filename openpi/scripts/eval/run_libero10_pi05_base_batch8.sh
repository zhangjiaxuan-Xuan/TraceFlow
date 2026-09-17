#!/usr/bin/env bash
set -euo pipefail

if [[ "${OPTIMUS_BASE_BATCH8_SNAPSHOT:-0}" != "1" ]]; then
  original_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
  export OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${original_script_dir}/../.." >/dev/null 2>&1 && pwd)}"
  script_snapshot="${TMPDIR:-/tmp}/optimus_base_batch8_${$}.sh"
  cp -- "${BASH_SOURCE[0]}" "${script_snapshot}"
  export OPTIMUS_BASE_BATCH8_SNAPSHOT=1
  exec bash "${script_snapshot}" "$@"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

GPU="${GPU:-0}"
PORT="${PORT:-8200}"
POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-logs/libero10_pi05_base_batch8_${RUN_ID}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-100}"
TASK_IDS_CSV="${TASK_IDS_CSV:-0,1,2,3,4,5,6,7,8,9}"
EXPECTED_TASKS="$(awk -F, '{print NF}' <<<"${TASK_IDS_CSV}")"
SEED="${SEED:-7}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
SAVE_EPISODE_DATA="${SAVE_EPISODE_DATA:-0}"
RESUME="${RESUME:-0}"
RESUME_MIN_IDLE_SECONDS="${RESUME_MIN_IDLE_SECONDS:-180}"

if [[ "${BATCH_SIZE}" -le 0 || "${BATCH_SIZE}" -gt "${NUM_TRIALS_PER_TASK}" ]]; then
  echo "BATCH_SIZE must be between 1 and NUM_TRIALS_PER_TASK." >&2
  exit 1
fi
if [[ ! -f "${POLICY_DIR}/model.safetensors" ]]; then
  echo "Missing pi0.5 PyTorch checkpoint: ${POLICY_DIR}/model.safetensors" >&2
  exit 1
fi
if [[ "${RESUME}" == "1" ]]; then
  if [[ ! -f "${RUN_ROOT}/eval/libero_10.jsonl" ]]; then
    echo "RESUME=1 requires ${RUN_ROOT}/eval/libero_10.jsonl" >&2
    exit 1
  fi
  log_idle_seconds=$(( $(date +%s) - $(stat -c %Y "${RUN_ROOT}/eval/libero_10.jsonl") ))
  if (( log_idle_seconds < RESUME_MIN_IDLE_SECONDS )); then
    echo "Resume log changed ${log_idle_seconds}s ago; wait for the existing remote run to stop." >&2
    exit 1
  fi
  if [[ "${RUN_ROOT}" != *libero10_pi05_base_batch8* ]]; then
    echo "RUN_ROOT does not look like a base batch8 run: ${RUN_ROOT}" >&2
    exit 1
  fi
  if [[ -d "${RUN_ROOT}/eval/videos" ]]; then
    SAVE_VIDEOS=1
  elif [[ "${SAVE_VIDEOS}" == "1" ]]; then
    echo "Cannot enable videos while resuming a run that has no existing video directory." >&2
    exit 1
  fi
  if [[ -d "${RUN_ROOT}/episode_data" ]]; then
    SAVE_EPISODE_DATA=1
  elif [[ "${SAVE_EPISODE_DATA}" == "1" ]]; then
    echo "Cannot enable episode data while resuming a run that has no existing episode_data directory." >&2
    exit 1
  fi
elif [[ -d "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to mix data into non-empty RUN_ROOT: ${RUN_ROOT}" >&2
  exit 1
fi

episode_data_root=""
if [[ "${SAVE_EPISODE_DATA}" == "1" ]]; then
  episode_data_root="${RUN_ROOT}/episode_data"
fi

validate_args=()
if [[ "${SAVE_VIDEOS}" == "1" ]]; then validate_args+=(--require-videos); fi
if [[ "${SAVE_EPISODE_DATA}" == "1" ]]; then validate_args+=(--require-trajectories); fi
if [[ "${RESUME}" == "1" ]] && /path/to/user/miniforge3/envs/optimusvla-openpi/bin/python scripts/analysis/summarize_libero_eval_run.py \
  --eval-log "${RUN_ROOT}/eval/libero_10.jsonl" \
  --output-dir "${RUN_ROOT}/indexes" \
  --expected-tasks "${EXPECTED_TASKS}" \
  --expected-episodes-per-task "${NUM_TRIALS_PER_TASK}" \
  "${validate_args[@]}"; then
  echo "Run is already complete; no policy server was started: ${RUN_ROOT}"
  exit 0
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
SEED="${SEED}" \
SAVE_VIDEOS="${SAVE_VIDEOS}" \
RESUME="${RESUME}" \
RUN_ID="${RUN_ID}" \
INFERENCE_BATCH_SIZE="${BATCH_SIZE}" \
INFERENCE_BATCH_WAIT_MS=20 \
LIBERO_CLIENTS_PER_SUITE="${BATCH_SIZE}" \
USE_MEMORY=0 \
USE_LCM=0 \
MEMORY_GUIDANCE_ONLY=0 \
USE_NEGATIVE_GUIDANCE=0 \
bash scripts/eval/run_libero_eval.sh libero_10

/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python scripts/analysis/summarize_libero_eval_run.py \
  --eval-log "${RUN_ROOT}/eval/libero_10.jsonl" \
  --output-dir "${RUN_ROOT}/indexes" \
  --expected-tasks "${EXPECTED_TASKS}" \
  --expected-episodes-per-task "${NUM_TRIALS_PER_TASK}" \
  "${validate_args[@]}"

echo "Pi0.5 base batch evaluation complete: ${RUN_ROOT}"
echo "Task summary: ${RUN_ROOT}/indexes/task_summary.tsv"
