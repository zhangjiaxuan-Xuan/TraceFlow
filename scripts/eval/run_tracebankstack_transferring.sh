#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
if [[ "${1:-}" == "--print-config" || "${2:-}" == "--print-config" ]]; then
  PYTHONPATH="${TRACEFLOW_ROOT}/src" "${PYTHON:-python3}" -m traceflow.cli config tracebankstack_transferring
  exit 0
fi
MODE="${1:-all}"
case "${MODE}" in collect|success|failure|joint|all) ;; *) echo "Usage: $0 collect|success|failure|joint|all" >&2; exit 2 ;; esac
tf_init_interpreters
ASSET_ROOT="$(tf_assets cl.transferring)"
UPPER_CKPT="$(tf_checkpoint predimem-upper)"
VLA_CKPT="$(tf_checkpoint predimem-vla)"
BASE_ROOT="${ASSET_ROOT}/arena/extra8"
CAMPAIGN_BASE="${CAMPAIGN_ROOT:-$(tf_default_output tracebankstack_transferring)}"

export ROOT="${OPENPI_ROOT}" OPENPI_PY="${OPENPI_PYTHON}" OPENPI_PYTHON
export AOSS_ROOT="${BASE_ROOT}" PREDIMEM_RUNTIME_ROOT="$(dirname "$(dirname "${VLA_CKPT}")")"
export VLA_CKPT VLM_CKPT="${UPPER_CKPT}" CAMPAIGN_BASE
export FIXED_META="${BASE_ROOT}/memory/fusion/gpm_memory_meta.pt"
export FIXED_INDEX="${BASE_ROOT}/memory/fusion/gpm_memory.index"
export FIXED_ACTIONS="${BASE_ROOT}/memory/shared/gpm_memory_actions.npz"
export FIXED_HEAD="${BASE_ROOT}/heads/fusion/best.pt"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
export ROUNDS="${ROUNDS:-1}" START_ROUND="${START_ROUND:-1}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}" SEED="${SEED:-7}"
export UPPER_GPU="${UPPER_GPU:-0}" LOWER_GPU="${LOWER_GPU:-1}"
export UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}" LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}"
export ENV_WORKERS="${ENV_WORKERS:-64}" SAVE_VIDEO="${SAVE_VIDEO:-1}"

run_group() {
  local group="$1"
  bash "${OPENPI_ROOT}/scripts/robomemarena/run_predimem_transferring_inherited_cl.sh" \
    "${POSITIVE_TOP_K:-16}" "${group}"
}
case "${MODE}" in
  collect) run_group collect ;;
  success) run_group success ;;
  failure) run_group failure ;;
  joint) run_group both ;;
  all) run_group collect; run_group success; run_group failure; run_group both ;;
esac
