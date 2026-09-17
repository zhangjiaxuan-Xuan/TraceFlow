#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
cd "${OPENPI_ROOT}"

MEMORY_VARIANT="${MEMORY_VARIANT:-success}"
case "${MEMORY_VARIANT}" in
  success) USE_NEGATIVE_GUIDANCE=0 ;;
  success_fail) USE_NEGATIVE_GUIDANCE=1 ;;
  *) echo "MEMORY_VARIANT must be success or success_fail, got: ${MEMORY_VARIANT}" >&2; exit 2 ;;
esac

MEMORY_GUIDANCE_VERSION="${MEMORY_GUIDANCE_VERSION:-v2}"
MEMORY_GUIDANCE_V05_DYNAMIC_NFE="${MEMORY_GUIDANCE_V05_DYNAMIC_NFE:-0}"
case "${MEMORY_GUIDANCE_VERSION}" in
  v0) VERSION_TAG="_v0" ;;
  v0_5)
    case "${MEMORY_GUIDANCE_V05_DYNAMIC_NFE}" in
      0) VERSION_TAG="_v05_fixed" ;;
      1) VERSION_TAG="_v05_dynamic_v2" ;;
      *) echo "MEMORY_GUIDANCE_V05_DYNAMIC_NFE must be 0 or 1, got: ${MEMORY_GUIDANCE_V05_DYNAMIC_NFE}" >&2; exit 2 ;;
    esac
    ;;
  v1) VERSION_TAG="_v1" ;;
  v2) VERSION_TAG="" ;;
  *) echo "MEMORY_GUIDANCE_VERSION must be v0, v0_5, v1, or v2, got: ${MEMORY_GUIDANCE_VERSION}" >&2; exit 2 ;;
esac

GPU="${GPU:-0}"
PORT="${PORT:-8200}"
POLICY_DIR="${POLICY_DIR:-/path/to/local/data/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-${OPENPI_ROOT}/checkpoints/gpm_task_head.pt}"
MEMORY_META_PATH="${MEMORY_META_PATH:-${OPENPI_ROOT}/memory/gpm_memory_meta.pt}"
FAISS_INDEX_PATH="${FAISS_INDEX_PATH:-${OPENPI_ROOT}/memory/gpm_memory.index}"
MEMORY_ACTIONS_PATH="${MEMORY_ACTIONS_PATH:-${OPENPI_ROOT}/memory/gpm_memory_actions.npz}"
NEGATIVE_MEMORY_META_PATH="${NEGATIVE_MEMORY_META_PATH:-${OPENPI_ROOT}/memory/negative/gpm_negative_memory_meta.pt}"
NEGATIVE_FAISS_INDEX_PATH="${NEGATIVE_FAISS_INDEX_PATH:-${OPENPI_ROOT}/memory/negative/gpm_negative_memory.index}"
NEGATIVE_MEMORY_ACTIONS_PATH="${NEGATIVE_MEMORY_ACTIONS_PATH:-${OPENPI_ROOT}/memory/negative/gpm_negative_memory_actions.npz}"
MEMORY_SOURCE_TAG="${MEMORY_SOURCE_TAG:-official}"
if [[ ! "${MEMORY_SOURCE_TAG}" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "MEMORY_SOURCE_TAG must contain only letters, digits, underscores, or hyphens." >&2
  exit 2
fi
SOURCE_TAG=""
if [[ "${MEMORY_SOURCE_TAG}" != "official" ]]; then SOURCE_TAG="_${MEMORY_SOURCE_TAG}"; fi
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-logs/libero10_guidance_only${VERSION_TAG}_${MEMORY_VARIANT}_batch8${SOURCE_TAG}_${RUN_ID}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-100}"
TASK_IDS_CSV="${TASK_IDS_CSV:-0,1,2,3,4,5,6,7,8,9}"
EXPECTED_TASKS="$(awk -F, '{print NF}' <<<"${TASK_IDS_CSV}")"
SEED="${SEED:-7}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
SAVE_EPISODE_DATA="${SAVE_EPISODE_DATA:-0}"
RESUME="${RESUME:-0}"
RESUME_MIN_IDLE_SECONDS="${RESUME_MIN_IDLE_SECONDS:-180}"

if [[ "${BATCH_SIZE}" -ne 8 ]]; then
  echo "This validation entrypoint requires BATCH_SIZE=8." >&2
  exit 2
fi
if [[ "${NUM_TRIALS_PER_TASK}" -lt "${BATCH_SIZE}" ]]; then
  echo "NUM_TRIALS_PER_TASK must be at least ${BATCH_SIZE}." >&2
  exit 2
fi

required_files=(
  "${POLICY_DIR}/model.safetensors"
  "${TASK_HEAD_CKPT}"
  "${MEMORY_META_PATH}"
  "${FAISS_INDEX_PATH}"
  "${MEMORY_ACTIONS_PATH}"
)
if [[ "${MEMORY_VARIANT}" == "success_fail" ]]; then
  required_files+=(
    "${NEGATIVE_MEMORY_META_PATH}"
    "${NEGATIVE_FAISS_INDEX_PATH}"
    "${NEGATIVE_MEMORY_ACTIONS_PATH}"
  )
fi
for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then echo "Missing required file: ${path}" >&2; exit 1; fi
done
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight passed: guidance=${MEMORY_GUIDANCE_VERSION} source=${MEMORY_SOURCE_TAG} head=${TASK_HEAD_CKPT} index=${FAISS_INDEX_PATH}"
  exit 0
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
  if [[ "${MEMORY_GUIDANCE_VERSION}" == "v0" && "${RUN_ROOT}" != *guidance_only_v0_* ]]; then
    echo "RUN_ROOT does not match V0 guidance: ${RUN_ROOT}" >&2
    exit 1
  fi
  if [[ "${MEMORY_GUIDANCE_VERSION}" == "v0_5" && "${RUN_ROOT}" != *guidance_only_v05_* ]]; then
    echo "RUN_ROOT does not match V0.5 guidance: ${RUN_ROOT}" >&2
    exit 1
  fi
  if [[ "${MEMORY_GUIDANCE_VERSION}" != "v0_5" && "${RUN_ROOT}" == *guidance_only_v05_* ]]; then
    echo "Refusing to resume a V0.5 run with ${MEMORY_GUIDANCE_VERSION}: ${RUN_ROOT}" >&2
    exit 1
  fi
  if [[ "${MEMORY_GUIDANCE_VERSION}" == "v0_5" ]]; then
    expected_nfe_tag="_v05_fixed_"
    if [[ "${MEMORY_GUIDANCE_V05_DYNAMIC_NFE}" == "1" ]]; then expected_nfe_tag="_v05_dynamic_v2_"; fi
    if [[ "${RUN_ROOT}" != *"${expected_nfe_tag}"* ]]; then
      echo "RUN_ROOT does not match V0.5 dynamic-NFE mode: ${RUN_ROOT}" >&2
      exit 1
    fi
  fi
  if [[ "${MEMORY_GUIDANCE_VERSION}" != "v0" && "${RUN_ROOT}" == *guidance_only_v0_* ]]; then
    echo "Refusing to resume a V0 run with ${MEMORY_GUIDANCE_VERSION}: ${RUN_ROOT}" >&2
    exit 1
  fi
  case "${MEMORY_VARIANT}:${RUN_ROOT}" in
    success_fail:*success_fail*) ;;
    success:*success_batch8*) ;;
    *) echo "RUN_ROOT does not match MEMORY_VARIANT=${MEMORY_VARIANT}: ${RUN_ROOT}" >&2; exit 1 ;;
  esac
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
EPISODE_DATA_ROOT="${episode_data_root}" \
EPISODE_DATA_MODE=all \
TASK_IDS_CSV="${TASK_IDS_CSV}" \
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
SEED="${SEED}" \
SAVE_VIDEOS="${SAVE_VIDEOS}" \
RESUME="${RESUME}" \
RUN_ID="${RUN_ID}" \
INFERENCE_BATCH_SIZE=8 \
INFERENCE_BATCH_WAIT_MS="${INFERENCE_BATCH_WAIT_MS:-20}" \
LIBERO_CLIENTS_PER_SUITE=8 \
USE_MEMORY=1 \
USE_LCM=0 \
USE_MEMORY_GUIDANCE=0 \
MEMORY_GUIDANCE_ONLY=1 \
MEMORY_GUIDANCE_TIME_VERSION="${MEMORY_GUIDANCE_VERSION}" \
MEMORY_PRIOR_SUBSTEP_GUIDANCE=0 \
MEMORY_GUIDANCE_NUM_STEPS=10 \
MEMORY_GUIDANCE_LAMBDA_MAX="${MEMORY_GUIDANCE_LAMBDA_MAX:-0.20}" \
MEMORY_GUIDANCE_T_CUT="${MEMORY_GUIDANCE_T_CUT:-0.30}" \
MEMORY_GUIDANCE_SIGMA="${MEMORY_GUIDANCE_SIGMA:-0.30}" \
MEMORY_GUIDANCE_NORM_CAP="${MEMORY_GUIDANCE_NORM_CAP:-0.20}" \
MEMORY_GUIDANCE_MIN_SIMILARITY="${MEMORY_GUIDANCE_MIN_SIMILARITY:--1.0}" \
MEMORY_GUIDANCE_V2_MAGNITUDE_CAP="${MEMORY_GUIDANCE_V2_MAGNITUDE_CAP:-0.10}" \
MEMORY_GUIDANCE_V05_DYNAMIC_NFE="${MEMORY_GUIDANCE_V05_DYNAMIC_NFE}" \
MEMORY_GUIDANCE_V05_FINE_RATIO="${MEMORY_GUIDANCE_V05_FINE_RATIO:-1.20}" \
MEMORY_GUIDANCE_V05_FINE_SCALE="${MEMORY_GUIDANCE_V05_FINE_SCALE:-0.20}" \
MEMORY_GUIDANCE_V05_FINE_CONFIRM_STEPS="${MEMORY_GUIDANCE_V05_FINE_CONFIRM_STEPS:-2}" \
MEMORY_GUIDANCE_V05_DYNAMIC_GUIDANCE_SCALE="${MEMORY_GUIDANCE_V05_DYNAMIC_GUIDANCE_SCALE:-0.50}" \
USE_NEGATIVE_GUIDANCE="${USE_NEGATIVE_GUIDANCE}" \
NEGATIVE_MEMORY_MIN_SIMILARITY="${NEGATIVE_MEMORY_MIN_SIMILARITY:-0.975}" \
NEGATIVE_MEMORY_MIN_CONFIDENCE="${NEGATIVE_MEMORY_MIN_CONFIDENCE:-0.75}" \
NEGATIVE_MEMORY_TOP_K="${NEGATIVE_MEMORY_TOP_K:-4}" \
NEGATIVE_GUIDANCE_NORM_CAP="${NEGATIVE_GUIDANCE_NORM_CAP:-0.10}" \
bash scripts/eval/run_libero_eval.sh libero_10

/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python scripts/analysis/summarize_libero_eval_run.py \
  --eval-log "${RUN_ROOT}/eval/libero_10.jsonl" \
  --output-dir "${RUN_ROOT}/indexes" \
  --expected-tasks "${EXPECTED_TASKS}" \
  --expected-episodes-per-task "${NUM_TRIALS_PER_TASK}" \
  "${validate_args[@]}"

echo "Guidance-only ${MEMORY_GUIDANCE_VERSION} ${MEMORY_VARIANT} batch=8 evaluation complete: ${RUN_ROOT}"
echo "Task summary: ${RUN_ROOT}/indexes/task_summary.tsv"
