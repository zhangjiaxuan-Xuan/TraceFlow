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

MEMORY_PRIOR_GUIDANCE_VERSION="${MEMORY_PRIOR_GUIDANCE_VERSION:-v1}"
case "${MEMORY_PRIOR_GUIDANCE_VERSION}" in
  v1) VERSION_TAG="" ;;
  v2) VERSION_TAG="_v2" ;;
  v3_prior_only) VERSION_TAG="_v3_prior_only" ;;
  v3_prior_decay) VERSION_TAG="_v3_prior_decay" ;;
  v3_prior_decay_joint) VERSION_TAG="_v3_prior_decay_joint" ;;
  v3_re_prior_decay) VERSION_TAG="_v3_re_prior_decay" ;;
  *)
    echo "MEMORY_PRIOR_GUIDANCE_VERSION must be v1, v2, v3_prior_only, "\
"v3_prior_decay, v3_prior_decay_joint, or v3_re_prior_decay" >&2
    exit 2
    ;;
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
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-logs/libero10_prior_substep${VERSION_TAG}_${MEMORY_VARIANT}_batch8_${RUN_ID}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LIBERO_CLIENTS_PER_SUITE="${LIBERO_CLIENTS_PER_SUITE:-8}"
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
if [[ "${LIBERO_CLIENTS_PER_SUITE}" -lt "${BATCH_SIZE}" ]]; then
  echo "LIBERO_CLIENTS_PER_SUITE must be at least BATCH_SIZE=${BATCH_SIZE}." >&2
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
if [[ "${MEMORY_PRIOR_GUIDANCE_VERSION}" == v3_* ]]; then
  if [[ -n "${MEMORY_GUIDANCE_V2_MAGNITUDE_CAP:-}" \
    && "${MEMORY_GUIDANCE_V2_MAGNITUDE_CAP}" != "0" \
    && "${MEMORY_GUIDANCE_V2_MAGNITUDE_CAP}" != "0.0" ]]; then
    echo "V3 prior-anchor guidance fixes MEMORY_GUIDANCE_V2_MAGNITUDE_CAP=0." >&2
    exit 2
  fi
  MEMORY_GUIDANCE_V2_MAGNITUDE_CAP=0
else
  MEMORY_GUIDANCE_V2_MAGNITUDE_CAP="${MEMORY_GUIDANCE_V2_MAGNITUDE_CAP:-0.10}"
fi
MEMORY_PRIOR_GUIDANCE_FINAL_SCALE="${MEMORY_PRIOR_GUIDANCE_FINAL_SCALE:-0.01}"
if [[ "${MEMORY_PRIOR_GUIDANCE_VERSION}" == "v3_re_prior_decay" ]]; then
  NFE_FLOOR="${NFE_FLOOR:-3}"
  if (( NFE_FLOOR < 3 )); then
    echo "V3-re requires NFE_FLOOR >= 3, got: ${NFE_FLOOR}" >&2
    exit 2
  fi
else
  NFE_FLOOR="${NFE_FLOOR:-1}"
fi
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight passed: prior=${MEMORY_PRIOR_GUIDANCE_VERSION} batch=${BATCH_SIZE} "\
"envs=${LIBERO_CLIENTS_PER_SUITE} final_scale=${MEMORY_PRIOR_GUIDANCE_FINAL_SCALE} "\
"nfe_floor=${NFE_FLOOR}"
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
  if [[ "${RUN_ROOT}" != *"prior_substep${VERSION_TAG}_"* ]]; then
    echo "RUN_ROOT does not match ${MEMORY_PRIOR_GUIDANCE_VERSION} guidance: ${RUN_ROOT}" >&2
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
PREDIMEM_NFE_STATS_PATH="${RUN_ROOT}/nfe_stats.jsonl" \
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
LIBERO_CLIENTS_PER_SUITE="${LIBERO_CLIENTS_PER_SUITE}" \
USE_MEMORY=1 \
USE_LCM=0 \
USE_MEMORY_GUIDANCE=0 \
MEMORY_GUIDANCE_ONLY=0 \
MEMORY_PRIOR_SUBSTEP_GUIDANCE=1 \
MEMORY_PRIOR_GUIDANCE_VERSION="${MEMORY_PRIOR_GUIDANCE_VERSION}" \
MEMORY_PRIOR_GUIDANCE_FINAL_SCALE="${MEMORY_PRIOR_GUIDANCE_FINAL_SCALE}" \
NFE_FLOOR="${NFE_FLOOR}" \
MEMORY_GUIDANCE_V2_MAGNITUDE_CAP="${MEMORY_GUIDANCE_V2_MAGNITUDE_CAP}" \
MEMORY_JOINT_FAILURE_PRIOR="${MEMORY_JOINT_FAILURE_PRIOR:-0.50}" \
USE_NEGATIVE_GUIDANCE="${USE_NEGATIVE_GUIDANCE}" \
NEGATIVE_MEMORY_MIN_SIMILARITY="${NEGATIVE_MEMORY_MIN_SIMILARITY:-0.975}" \
NEGATIVE_MEMORY_TOP_K="${NEGATIVE_MEMORY_TOP_K:-4}" \
NEGATIVE_GUIDANCE_NORM_CAP="${NEGATIVE_GUIDANCE_NORM_CAP:-0.10}" \
bash scripts/eval/run_libero_eval.sh libero_10

/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python scripts/analysis/summarize_libero_eval_run.py \
  --eval-log "${RUN_ROOT}/eval/libero_10.jsonl" \
  --output-dir "${RUN_ROOT}/indexes" \
  --expected-tasks "${EXPECTED_TASKS}" \
  --expected-episodes-per-task "${NUM_TRIALS_PER_TASK}" \
  "${validate_args[@]}"

if [[ "${MEMORY_PRIOR_GUIDANCE_VERSION}" == "v3_re_prior_decay" ]]; then
  /path/to/user/miniforge3/envs/optimusvla-openpi/bin/python scripts/analysis/summarize_adaptive_nfe.py \
    --stats-file "${RUN_ROOT}/nfe_stats.jsonl" \
    --libero-eval-log "${RUN_ROOT}/eval/libero_10.jsonl" \
    --output "${RUN_ROOT}/nfe_summary.json"
fi

echo "Prior-substep ${MEMORY_PRIOR_GUIDANCE_VERSION} ${MEMORY_VARIANT} batch=8 evaluation complete: ${RUN_ROOT}"
echo "Task summary: ${RUN_ROOT}/indexes/task_summary.tsv"
