#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
tf_print_config_if_requested libero_plus10 "${1:-}" && exit 0
tf_init_interpreters

PLUS_ROOT="${LIBERO_PLUS_ROOT:-${TRACEFLOW_ROOT}/third_party/LIBERO-plus}"
if [[ ! -f "${PLUS_ROOT}/libero/libero/benchmark/task_classification.json" || ! -d "${PLUS_ROOT}/libero/libero/assets" ]]; then
  bash "${TRACEFLOW_ROOT}/scripts/setup/fetch_benchmarks.sh" plus
fi
ASSET_ROOT="$(tf_assets common.libero_head libero.libero10_positive libero.libero10_negative)"
POLICY_DIR="$(tf_checkpoint pi05)"
RUN_ROOT="${RUN_ROOT:-$(tf_default_output libero_plus10)}"

export OPENPI_ROOT OPENPI_PYTHON LIBERO_PYTHON POLICY_DIR RUN_ROOT
export LIBERO_PLUS_ROOT="${PLUS_ROOT}" LIBERO_SOURCE_ROOT="${PLUS_ROOT}"
export LIBERO_CONFIG_PATH="${TRACEFLOW_LIBERO_CONFIG_PATH:-${RUN_ROOT}/libero_config}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
export SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-900}"
export TASK_HEAD_CKPT="${ASSET_ROOT}/common/libero/gpm_task_head.pt"
export BANK_ROOT="${ASSET_ROOT}/libero/libero10"
export POSITIVE_BANK="${BANK_ROOT}/positive"
export MEMORY_META_PATH="${POSITIVE_BANK}/gpm_memory_meta.pt"
export FAISS_INDEX_PATH="${POSITIVE_BANK}/gpm_memory.index"
export MEMORY_ACTIONS_PATH="${POSITIVE_BANK}/gpm_memory_actions.npz"
export NEGATIVE_MEMORY_META_PATH="${BANK_ROOT}/negative/gpm_negative_memory_meta.pt"
export NEGATIVE_FAISS_INDEX_PATH="${BANK_ROOT}/negative/gpm_negative_memory.index"
export NEGATIVE_MEMORY_ACTIONS_PATH="${BANK_ROOT}/negative/gpm_negative_memory_actions.npz"
export MODE="${MODE:-guidance}" EVAL_SCOPE="${EVAL_SCOPE:-full}" MEMORY_BANK_SCOPE=libero10
export MEMORY_TOP_K=16 NEGATIVE_MEMORY_TOP_K=8 MEMORY_GUIDANCE_TIME_VERSION=v1
export MEMORY_GUIDANCE_NORM_CAP=0.20 MEMORY_GUIDANCE_TOTAL_NORM_CAP=0.20
export NEGATIVE_MEMORY_MIN_SIMILARITY=-1.0 NEGATIVE_MEMORY_MIN_CONFIDENCE=0.0
export NEGATIVE_GUIDANCE_BETA=0.10 NEGATIVE_GUIDANCE_NORM_CAP=0.10
export SEED="${SEED:-7}" NUM_TRIALS_PER_TASK="${EPISODES_PER_TASK:-1}"
export INFERENCE_BATCH_SIZE="${BATCH_SIZE:-8}"
export LIBERO_CLIENTS_PER_SUITE="${LIBERO_CLIENTS_PER_SUITE:-${ENV_WORKERS:-16}}"
export LOG_DIR="${RUN_ROOT}"
exec bash "${OPENPI_ROOT}/scripts/eval/run_libero_plus10_v1_bcpi_lowcost.sh"
