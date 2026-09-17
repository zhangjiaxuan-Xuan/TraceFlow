#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
if [[ "${1:-}" == "--print-config" || "${2:-}" == "--print-config" ]]; then
  PYTHONPATH="${TRACEFLOW_ROOT}/src" "${PYTHON:-python3}" -m traceflow.cli config tracebankstack_libero10
  exit 0
fi
MODE="${1:-all}"
case "${MODE}" in collect|success|failure|joint|all) ;; *) echo "Usage: $0 collect|success|failure|joint|all" >&2; exit 2 ;; esac
tf_init_interpreters
ASSET_ROOT="$(tf_assets common.libero_head cl.libero10)"
POLICY_DIR="$(tf_checkpoint pi05)"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$(tf_default_output tracebankstack_libero10)}"
LIBERO_SOURCE_ROOT="${LIBERO_SOURCE_ROOT:-${TRACEFLOW_ROOT}/third_party/LIBERO}"
export OPENPI_ROOT OPENPI_PYTHON LIBERO_PYTHON POLICY_DIR LIBERO_SOURCE_ROOT
export TASK_HEAD_CKPT="${ASSET_ROOT}/common/libero/gpm_task_head.pt"
export INITIAL_BANK_ROOT="${ASSET_ROOT}/cl/libero10"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
export ROUNDS="${ROUNDS:-1}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
export SEED="${SEED:-7}" GPU="${GPU:-0}" PORT="${PORT:-8200}"
export INFERENCE_BATCH_SIZE="${BATCH_SIZE:-16}" ENV_WORKERS="${ENV_WORKERS:-32}"
export TASK_IDS_CSV="${TASK_IDS_CSV:-0,1,2,3,4,5,6,7,8,9}"
export MAX_ENV_STEPS="${MAX_ENV_STEPS:-0}" SMOKE="${SMOKE:-0}"

collect() {
  MODE=base RUN_ROOT="${CAMPAIGN_ROOT}/collector" EPISODE_DATA_MODE=all \
    bash "${OPENPI_ROOT}/scripts/eval/run_pi_v1_self_cl_round.sh"
}
branch() {
  local public_name="$1" internal_name="$2"
  OUTPUT_ROOT="${CAMPAIGN_ROOT}/${public_name}" BRANCH="${internal_name}" \
    bash "${OPENPI_ROOT}/scripts/experiments/run_pi_v1_self_cl_libero10.sh" \
      --rounds "${ROUNDS}" --episodes-per-task "${EPISODES_PER_TASK}"
}

case "${MODE}" in
  collect) collect ;;
  success) branch success success_only ;;
  failure) branch failure failure_only ;;
  joint) branch joint all ;;
  all) collect; branch success success_only; branch failure failure_only; branch joint all ;;
esac
