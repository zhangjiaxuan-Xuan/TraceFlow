#!/usr/bin/env bash
set -euo pipefail

# Full26 Fusion-V1 validation using the offline two-metric fitted gate.
# This is the deployable gate validation. It does not claim to execute the
# future Upper-V0/Fusion-V1 runtime mixture.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
FIT_CONFIG="${TWO_METRIC_FIT_CONFIG:-${ROOT}/docs/report/data/predimem_two_metric_gate_v1/gate_config.json}"
GATE_PATH="${TWO_METRIC_COMPAT_GATE:-${ROOT}/docs/report/data/predimem_two_metric_gate_v1/compiled_arena_v1_gate.json}"

[[ -f "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -f "${FIT_CONFIG}" ]] || { echo "Missing fitted gate: ${FIT_CONFIG}" >&2; exit 2; }

"${PYTHON}" -m optimus_eval.compile_two_metric_gate_compat \
  --fit-config "${FIT_CONFIG}" \
  --output "${GATE_PATH}"

export MEMORY_GUIDANCE_SUITE_GATE_PATH="${GATE_PATH}"
export MODE="v1"
export HEAD_VARIANTS="fusion"
export MEMORY_ADMISSION="success"
export MEMORY_TOP_K="${MEMORY_TOP_K:-16}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-51}"
export TASK_IDS="${TASK_IDS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}"
export SEED="${SEED:-50}"
export UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}"
export LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}"
export ENV_WORKERS="${ENV_WORKERS:-64}"
export UPPER_GPU="${UPPER_GPU:-0}"
export LOWER_GPU="${LOWER_GPU:-1}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export RECORD_MEMORY_DATA="${RECORD_MEMORY_DATA:-0}"
export PREDIMEM_COMPONENT_TIMING="${PREDIMEM_COMPONENT_TIMING:-1}"

exec bash "${ROOT}/scripts/robomemarena/run_predimem_arena_full26_separability_gate_dual_gpu.sh" v1
