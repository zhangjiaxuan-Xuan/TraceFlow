#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
cd "${OPENPI_ROOT}"

MODE="${MODE:?MODE is required}"
POLICY_DIR="${POLICY_DIR:?POLICY_DIR is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
GPU="${GPU:-0}"
PORT="${PORT:-8200}"
SEED="${SEED:-7}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-100}"
RESUME="${RESUME:-auto}"
TOPK_SELECTION="${TOPK_SELECTION:?TOPK_SELECTION is required}"
TOPK_SELECTION_SHA256="${TOPK_SELECTION_SHA256:?TOPK_SELECTION_SHA256 is required}"

IFS=$'\t' read -r SELECTED_POSITIVE_TOP_K SELECTED_NEGATIVE_TOP_K _ < <(
  "${PYTHON}" -m openpi.experiments.topk_contract \
    --path "${TOPK_SELECTION}" --sha256 "${TOPK_SELECTION_SHA256}" --consumer pi --field tsv
)

[[ "${BATCH_SIZE}" == "8" ]] || { echo "BATCH_SIZE must be 8" >&2; exit 2; }
[[ -f "${POLICY_DIR}/model.safetensors" ]] || { echo "Missing policy: ${POLICY_DIR}/model.safetensors" >&2; exit 1; }
case "${MODE}" in
  policy) USE_MEMORY=0; USE_LCM=0; GUIDANCE_ONLY=0; USE_NEGATIVE=0 ;;
  prior) USE_MEMORY=1; USE_LCM=1; GUIDANCE_ONLY=0; USE_NEGATIVE=0 ;;
  v0_success) USE_MEMORY=1; USE_LCM=0; GUIDANCE_ONLY=1; USE_NEGATIVE=0 ;;
  v0_success_failure) USE_MEMORY=1; USE_LCM=0; GUIDANCE_ONLY=1; USE_NEGATIVE=1 ;;
  *) echo "Unknown MODE=${MODE}" >&2; exit 2 ;;
esac

require_selected_k() {
  local name="$1"
  local expected="$2"
  local actual="${!name-}"
  if [[ -n "${actual}" && "${actual}" != "${expected}" ]]; then
    echo "${name} conflicts with frozen top-k selection: expected ${expected}, got ${actual}" >&2
    exit 2
  fi
  printf -v "${name}" '%s' "${expected}"
  export "${name}"
}
case "${MODE}" in
  prior)
    FIXED_PRIOR_TOP_K="${FIXED_PRIOR_TOP_K:?FIXED_PRIOR_TOP_K is required for the fixed prior baseline}"
    require_selected_k MEMORY_TOP_K "${FIXED_PRIOR_TOP_K}"
    ;;
  v0_success) require_selected_k MEMORY_TOP_K "${SELECTED_POSITIVE_TOP_K}" ;;
  v0_success_failure)
    require_selected_k MEMORY_TOP_K "${SELECTED_POSITIVE_TOP_K}"
    require_selected_k NEGATIVE_MEMORY_TOP_K "${SELECTED_NEGATIVE_TOP_K}"
    ;;
esac

if [[ "${USE_MEMORY}" == "1" ]]; then
  required=(
    "${TASK_HEAD_CKPT:?TASK_HEAD_CKPT is required}"
    "${MEMORY_META_PATH:?MEMORY_META_PATH is required}"
    "${FAISS_INDEX_PATH:?FAISS_INDEX_PATH is required}"
    "${MEMORY_ACTIONS_PATH:?MEMORY_ACTIONS_PATH is required}"
  )
  if [[ "${USE_NEGATIVE}" == "1" ]]; then
    required+=(
      "${NEGATIVE_MEMORY_META_PATH:?NEGATIVE_MEMORY_META_PATH is required}"
      "${NEGATIVE_FAISS_INDEX_PATH:?NEGATIVE_FAISS_INDEX_PATH is required}"
      "${NEGATIVE_MEMORY_ACTIONS_PATH:?NEGATIVE_MEMORY_ACTIONS_PATH is required}"
    )
  fi
  for path in "${required[@]}"; do [[ -f "${path}" ]] || { echo "Missing artifact: ${path}" >&2; exit 1; }; done
fi

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
TASK_HEAD_CKPT="${TASK_HEAD_CKPT:-}" \
MEMORY_META_PATH="${MEMORY_META_PATH:-}" \
FAISS_INDEX_PATH="${FAISS_INDEX_PATH:-}" \
MEMORY_ACTIONS_PATH="${MEMORY_ACTIONS_PATH:-}" \
MEMORY_TOP_K="${MEMORY_TOP_K:-8}" \
NEGATIVE_MEMORY_META_PATH="${NEGATIVE_MEMORY_META_PATH:-}" \
NEGATIVE_FAISS_INDEX_PATH="${NEGATIVE_FAISS_INDEX_PATH:-}" \
NEGATIVE_MEMORY_ACTIONS_PATH="${NEGATIVE_MEMORY_ACTIONS_PATH:-}" \
LOG_DIR="${RUN_ROOT}/eval" \
RESULTS_TXT="${RUN_ROOT}/eval/results.txt" \
VIDEO_ROOT="${RUN_ROOT}/eval/videos" \
EPISODE_DATA_ROOT="" \
TASK_IDS_CSV="0,1,2,3,4,5,6,7,8,9" \
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
SEED="${SEED}" \
SAVE_VIDEOS="${SAVE_VIDEOS:-0}" \
RESUME="${RESUME}" \
INFERENCE_BATCH_SIZE=8 \
LIBERO_CLIENTS_PER_SUITE=8 \
USE_MEMORY="${USE_MEMORY}" \
USE_LCM="${USE_LCM}" \
MEMORY_GUIDANCE_ONLY="${GUIDANCE_ONLY}" \
MEMORY_GUIDANCE_TIME_VERSION=v0 \
MEMORY_GUIDANCE_NUM_STEPS=10 \
USE_NEGATIVE_GUIDANCE="${USE_NEGATIVE}" \
NEGATIVE_MEMORY_MIN_SIMILARITY=-1.0 \
NEGATIVE_MEMORY_MIN_CONFIDENCE=0.0 \
NEGATIVE_MEMORY_TOP_K="${NEGATIVE_MEMORY_TOP_K:-4}" \
bash scripts/eval/run_libero_eval.sh libero_10

"${PYTHON}" scripts/analysis/summarize_libero_eval_run.py \
  --eval-log "${eval_log}" \
  --output-dir "${RUN_ROOT}/indexes" \
  --expected-tasks 10 \
  --expected-episodes-per-task "${NUM_TRIALS_PER_TASK}"
