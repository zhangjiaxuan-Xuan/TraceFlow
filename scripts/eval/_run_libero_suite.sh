#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${1:?config name required}"
SUITE="${2:?suite required}"
POSITIVE_RELATIVE="${3:?positive bank path required}"
NEGATIVE_RELATIVE="${4:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
tf_init_interpreters

case "${CONFIG_NAME}" in
  libero_spatial|libero_goal) groups=(common.libero_head libero.b6500_positive); positive_k=8 ;;
  libero_object|libero_10) groups=(common.libero_head libero.b50_positive libero.ncpi_negative); positive_k=16 ;;
  *) echo "Unknown LIBERO release config: ${CONFIG_NAME}" >&2; exit 2 ;;
esac

ASSET_ROOT="$(tf_assets "${groups[@]}")"
POLICY_DIR="$(tf_checkpoint pi05)"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT must identify this suite output directory}"
LIBERO_SOURCE_ROOT="${LIBERO_SOURCE_ROOT:-${TRACEFLOW_ROOT}/third_party/LIBERO}"
[[ -f "${LIBERO_SOURCE_ROOT}/libero/libero/__init__.py" ]] || {
  echo "Missing LIBERO checkout; run scripts/setup/fetch_benchmarks.sh" >&2
  exit 2
}

negative=0
[[ -n "${NEGATIVE_RELATIVE}" ]] && negative=1
export OPENPI_ROOT OPENPI_PYTHON LIBERO_PYTHON LIBERO_SOURCE_ROOT POLICY_DIR RUN_ROOT
export LIBERO_CONFIG_PATH="${TRACEFLOW_LIBERO_CONFIG_PATH:-${RUN_ROOT}/libero_config}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
export SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-900}"
export TASK_HEAD_CKPT="${ASSET_ROOT}/common/libero/gpm_task_head.pt"
export MEMORY_META_PATH="${ASSET_ROOT}/${POSITIVE_RELATIVE}/gpm_memory_meta.pt"
export FAISS_INDEX_PATH="${ASSET_ROOT}/${POSITIVE_RELATIVE}/gpm_memory.index"
export MEMORY_ACTIONS_PATH="${ASSET_ROOT}/${POSITIVE_RELATIVE}/gpm_memory_actions.npz"
export MEMORY_TOP_K="${positive_k}"
export USE_MEMORY=1 USE_LCM=0 USE_MEMORY_GUIDANCE=0 MEMORY_GUIDANCE_ONLY=1
export MEMORY_GUIDANCE_TIME_VERSION=v1 MEMORY_GUIDANCE_NUM_STEPS=10
export MEMORY_GUIDANCE_LAMBDA_MAX=0.20 MEMORY_GUIDANCE_T_CUT=0.30 MEMORY_GUIDANCE_SIGMA=0.30
export MEMORY_GUIDANCE_NORM_CAP=0.20 MEMORY_GUIDANCE_TOTAL_NORM_CAP=0.20
export MEMORY_GUIDANCE_MIN_SIMILARITY=-1.0
export SEED="${SEED:-7}" NUM_TRIALS_PER_TASK="${EPISODES_PER_TASK:-50}"
export INFERENCE_BATCH_SIZE="${BATCH_SIZE:-16}" LIBERO_CLIENTS_PER_SUITE="${ENV_WORKERS:-32}"
export SERVER_CUDA_VISIBLE_DEVICES="${GPU:-0}" CLIENT_CUDA_VISIBLE_DEVICES="${GPU:-0}"
export SAVE_VIDEOS="${SAVE_VIDEOS:-1}" RESUME="${RESUME:-0}"
export LOG_DIR="${RUN_ROOT}/eval" RESULTS_TXT="${RUN_ROOT}/eval/results.txt" VIDEO_ROOT="${RUN_ROOT}/videos"
if [[ "${negative}" == "1" ]]; then
  export USE_NEGATIVE_GUIDANCE=1 NEGATIVE_MEMORY_TOP_K=8
  export NEGATIVE_MEMORY_META_PATH="${ASSET_ROOT}/${NEGATIVE_RELATIVE}/gpm_negative_memory_meta.pt"
  export NEGATIVE_FAISS_INDEX_PATH="${ASSET_ROOT}/${NEGATIVE_RELATIVE}/gpm_negative_memory.index"
  export NEGATIVE_MEMORY_ACTIONS_PATH="${ASSET_ROOT}/${NEGATIVE_RELATIVE}/gpm_negative_memory_actions.npz"
  export NEGATIVE_MEMORY_MIN_SIMILARITY=-1.0 NEGATIVE_MEMORY_MIN_CONFIDENCE=0.0
  export NEGATIVE_GUIDANCE_BETA=0.10 NEGATIVE_GUIDANCE_SIGMA=0.30 NEGATIVE_GUIDANCE_NORM_CAP=0.10
else
  export USE_NEGATIVE_GUIDANCE=0
fi

mkdir -p "${RUN_ROOT}"
exec bash "${OPENPI_ROOT}/scripts/eval/run_libero_eval.sh" "${SUITE}"
