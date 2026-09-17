#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: $0 v0|v1}"
case "${MODE}" in v0|v1) ;; *) echo "MODE must be v0 or v1" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE_PATH="${MEMORY_GUIDANCE_SUITE_GATE_PATH:-${ROOT}/configs/robomemarena/arena_suite_guidance_gate_v1.json}"
[[ -f "${GATE_PATH}" ]] || { echo "Missing suite gate: ${GATE_PATH}" >&2; exit 2; }

RUN_ROOT="${RUN_ROOT:-${RUN_ROOT_BASE:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/eval_separability_gate}/${MODE}_k16_batch32_env64_seed50_trials51_$(date -u +%Y%m%d_%H%M%S)}"

MEMORY_GUIDANCE_SUITE_GATE_PATH="${GATE_PATH}" \
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}" \
PREDIMEM_COMPONENT_TIMING="${PREDIMEM_COMPONENT_TIMING:-1}" \
ENV_WORKERS="${ENV_WORKERS:-64}" \
UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}" \
LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}" \
MEMORY_TOP_K="${MEMORY_TOP_K:-16}" \
EPISODES_PER_TASK="${EPISODES_PER_TASK:-51}" \
SEED="${SEED:-50}" \
RUN_ROOT="${RUN_ROOT}" \
bash "${ROOT}/scripts/robomemarena/run_predimem_arena_full26_v0_v1_dual_gpu.sh" "${MODE}"
