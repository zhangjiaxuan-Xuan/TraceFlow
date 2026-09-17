#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

# Offline-ranked local candidate. This remains an online short test, not a
# claimed optimum: the task-level surrogate only weakly beats its null model.
export UPPER_GUIDANCE_HISTORY_ENABLED="0"
export UPPER_GUIDANCE_TEMPORAL_ENABLED="1"
export UPPER_GUIDANCE_TEMPORAL_POSTERIOR="${UPPER_GUIDANCE_TEMPORAL_POSTERIOR:-0.75}"
export UPPER_GUIDANCE_TEMPORAL_PURITY="${UPPER_GUIDANCE_TEMPORAL_PURITY:-0.35}"
export UPPER_GUIDANCE_TEMPORAL_EVIDENCE_DECAY="${UPPER_GUIDANCE_TEMPORAL_EVIDENCE_DECAY:-0.95}"
export UPPER_GUIDANCE_TEMPORAL_ADVANCE_EVIDENCE="${UPPER_GUIDANCE_TEMPORAL_ADVANCE_EVIDENCE:-0.45}"
export UPPER_GUIDANCE_TEMPORAL_SAME_STAGE_BUDGET="${UPPER_GUIDANCE_TEMPORAL_SAME_STAGE_BUDGET:-2}"
export UPPER_GUIDANCE_TEMPORAL_MAX_ROLLBACK="${UPPER_GUIDANCE_TEMPORAL_MAX_ROLLBACK:-0.15}"
export UPPER_GUIDANCE_TEMPORAL_MAX_ADVANCE="${UPPER_GUIDANCE_TEMPORAL_MAX_ADVANCE:-0.40}"

export EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
export SEED="${SEED:-50}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export RECORD_MEMORY_DATA="0"
export RUN_ROOT="${RUN_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/eval_upper_knnlm_temporal_v2/counting_tdense5_upper_temporal_v2_seed${SEED}_${STAMP}}"

exec bash "${ROOT}/scripts/robomemarena/run_predimem_counting_upper_knnlm_only.sh"
