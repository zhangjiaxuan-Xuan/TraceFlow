#!/usr/bin/env bash
set -euo pipefail

# Clean replication of the seed-7 k8 collector after the Upper per-environment
# ordering fix. "none" means no online outcome-memory admission; the fixed
# producer-matched positive demonstration bank remains enabled.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/transferring_self_cl_k8_seed7/collector_upperfix}"
resume=0
[[ -f "${RUN_ROOT}/run_config.json" ]] && resume=1

UPPER_GPU="${UPPER_GPU:-0}" LOWER_GPU="${LOWER_GPU:-1}" \
SEED=7 TASK_IDS=18,19,25,26 EPISODES_PER_TASK=50 \
UPPER_BATCH_SIZE=32 LOWER_BATCH_SIZE=32 ENV_WORKERS=96 \
MEMORY_TOP_K=8 NEGATIVE_MEMORY_TOP_K=8 \
SAVE_VIDEO=1 RECORD_MEMORY_DATA=1 MEMORY_TRACE_LEVEL=full \
RUN_ROOT="${RUN_ROOT}" RESUME="${resume}" PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}" \
  bash "${ROOT}/scripts/robomemarena/run_predimem_arena_transferring_ablation_dual_gpu.sh" collect
