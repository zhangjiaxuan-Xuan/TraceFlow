#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${1:?config required}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
tf_init_interpreters

case "${CONFIG_NAME}" in
  arena_sequence)
    group=arena.extra8; relative=arena/extra8; tasks=1,2,3,22; head=upper; mode=v0; episodes=26; gate="" ;;
  arena_transferring)
    group=arena.extra8; relative=arena/extra8; tasks=18,19,25,26; head=fusion; mode=v1; episodes=26; gate="" ;;
  arena_counting)
    group=arena.counting; relative=arena/counting; tasks=6,7,8,9,10,15,16; head=fusion; mode=v1; episodes=51
    gate="${OPENPI_ROOT}/configs/robomemarena/arena_counting_v1_1_gate.json" ;;
  arena_occlusion)
    group=arena.occlusion; relative=arena/occlusion; tasks=4,5,11,12,13,14,17,20,21,23,24; head=fusion; mode=v1; episodes=51
    gate="${OPENPI_ROOT}/configs/robomemarena/arena_occlusion_v1_1_gate.json" ;;
  *) echo "Unknown Arena config: ${CONFIG_NAME}" >&2; exit 2 ;;
esac

ASSET_ROOT="$(tf_assets "${group}")"
UPPER_CKPT="$(tf_checkpoint predimem-upper)"
VLA_CKPT="$(tf_checkpoint predimem-vla)"
SELF_ROOT="${ASSET_ROOT}/${relative}"
RUN_ROOT="${RUN_ROOT:-$(tf_default_output "${CONFIG_NAME}")}"

export ROOT="${OPENPI_ROOT}" OPENPI_PY="${OPENPI_PYTHON}" OPENPI_PYTHON PREDIMEM_PYTHON
export SELF_ROOT AOSS_ROOT="${SELF_ROOT}" SOURCE_ROOT="$(dirname "$(dirname "${VLA_CKPT}")")"
export VLA_CKPT VLM_CKPT="${UPPER_CKPT}" HEAD_VARIANTS="${head}"
export TASK_IDS="${TASK_IDS:-${tasks}}" MEMORY_ALLOWED_TASK_IDS="${MEMORY_ALLOWED_TASK_IDS:-${tasks}}"
export UPPER_GUIDANCE_ALLOWED_TASK_IDS="${UPPER_GUIDANCE_ALLOWED_TASK_IDS:-${tasks}}"
export MEMORY_EXACT_TASK_GATE="${MEMORY_EXACT_TASK_GATE:-1}"
export MEMORY_TOP_K="${MEMORY_TOP_K:-16}" NEGATIVE_MEMORY_TOP_K="${NEGATIVE_MEMORY_TOP_K:-8}"
export PREDIMEM_MODE="${mode}" EPISODES_PER_TASK="${EPISODES_PER_TASK:-${episodes}}" SEED="${SEED:-50}"
export UPPER_GPU="${UPPER_GPU:-0}" LOWER_GPU="${LOWER_GPU:-1}"
export UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}" LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}"
export ENV_WORKERS="${ENV_WORKERS:-64}" SAVE_VIDEO="${SAVE_VIDEO:-1}"
export RUN_ROOT RESUME="${RESUME:-0}" PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
export MEMORY_GUIDANCE_NORM_CAP="${MEMORY_GUIDANCE_NORM_CAP:-0.20}"
export MEMORY_GUIDANCE_TOTAL_NORM_CAP="${MEMORY_GUIDANCE_TOTAL_NORM_CAP:-0.20}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
if [[ -n "${gate}" ]]; then export MEMORY_GUIDANCE_SUITE_GATE_PATH="${gate}"; fi

exec bash "${OPENPI_ROOT}/scripts/robomemarena/run_predimem_v1_tdense5_common_dual_gpu.sh"
