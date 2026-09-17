#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/pi05_counting_occlusion_tdense5_v1_seed50}"

export TASK_IDS="${TASK_IDS:-4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,23,24}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
export SEED="${SEED:-50}"
export GPU="${GPU:-0}"
export PORT="${PORT:-8720}"
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
export INFERENCE_BATCH_WAIT_MS="${INFERENCE_BATCH_WAIT_MS:-100}"
export ENV_WORKERS="${ENV_WORKERS:-72}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export REPLAN_STEPS="${REPLAN_STEPS:-10}"
export NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
export MAX_STEPS="${MAX_STEPS:-2500}"
export POST_GOAL_STEPS="${POST_GOAL_STEPS:-200}"
export MEMORY_TOP_K="${MEMORY_TOP_K:-8}"
export MEMORY_PROGRESS_WINDOW="${MEMORY_PROGRESS_WINDOW:-0.20}"
export MEMORY_GUIDANCE_NORM_CAP="${MEMORY_GUIDANCE_NORM_CAP:-0.20}"
export MEMORY_GUIDANCE_TOTAL_NORM_CAP="${MEMORY_GUIDANCE_TOTAL_NORM_CAP:-0.20}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  export TASK_IDS="${SMOKE_TASK_IDS:-4,6}"
  export EPISODES_PER_TASK="${SMOKE_EPISODES_PER_TASK:-1}"
  export ENV_WORKERS="${SMOKE_ENV_WORKERS:-2}"
  export SAVE_VIDEO="${SMOKE_SAVE_VIDEO:-0}"
  CAMPAIGN_ROOT="${SMOKE_CAMPAIGN_ROOT:-/tmp/pi05_co_oc_tdense5_v1_smoke_${USER:-user}}"
fi

export RUN_ROOT="${RUN_ROOT:-${CAMPAIGN_ROOT}/eval}"
if [[ "${PREFLIGHT_ONLY:-0}" != "1" && -f "${RUN_ROOT}/results.txt" ]]; then
  echo "[skip] completed Pi0.5 Counting/Occlusion evaluation: ${RUN_ROOT}"
  exit 0
fi
if [[ -f "${RUN_ROOT}/run_config.json" ]]; then
  export RESUME=1
else
  export RESUME=0
fi

exec bash "${ROOT}/scripts/eval/run_pi05_arena_v1_dense5_budget1024_batch8.sh"
