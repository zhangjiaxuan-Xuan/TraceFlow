#!/usr/bin/env bash
set -euo pipefail

ARM=${1:?Usage: $0 upper_fusion_action|fusion_fusion_action|fusion_only|upper_only}
case "${ARM}" in
  upper_fusion_action|fusion_fusion_action|fusion_only|upper_only) ;;
  *) echo "Unknown experiment arm: ${ARM}" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SELF_ROOT=${SELF_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_joint_v6_tdense5/predimem_2048_fp32}
SOURCE_ROOT=${SOURCE_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower}
STAMP="$(date -u +%Y%m%d_%H%M%S)"

export ROOT
export AOSS_ROOT="${SELF_ROOT}"
export VLA_CKPT="${SOURCE_ROOT}/checkpoints/vla_alltask_pytorch"
export VLM_CKPT="${VLM_CKPT:-/path/to/local/data/robomemarena/models/PrediMem/vlm_tasks1to26_ckpt74500}"
export MEMORY_META_BASENAME=gpm_memory_meta_dense_frame_v3.pt
export MEMORY_ALIGNMENT_TAG=joint-tdense5-stage-frame-v3-upper-hold10
export TASK_IDS="${TASK_IDS:-1,2,3,18,19,22,25,26}"
export MEMORY_ALLOWED_TASK_IDS="${TASK_IDS}"
export UPPER_GUIDANCE_ALLOWED_TASK_IDS="${TASK_IDS}"
export EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
export SEED="${SEED:-50}"
export UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-32}"
export LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-32}"
export ENV_WORKERS="${ENV_WORKERS:-64}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export RECORD_MEMORY_DATA=0
export REPLAN_STEPS=10
export VLM_INTERVAL=5

export UPPER_GUIDANCE_ENABLED=1
export UPPER_GUIDANCE_MANIFEST="${SELF_ROOT}/features/anchors_stride5.jsonl"
export UPPER_GUIDANCE_FEATURES="${SELF_ROOT}/features/upper_dense5_held/upper_features.npy"
export UPPER_GUIDANCE_TOP_K="${UPPER_GUIDANCE_TOP_K:-16}"
export UPPER_GUIDANCE_TEMPERATURE="${UPPER_GUIDANCE_TEMPERATURE:-0.07}"
export UPPER_GUIDANCE_INTERPOLATION="${UPPER_GUIDANCE_INTERPOLATION:-0.8}"
export UPPER_GUIDANCE_BANK_PER_PRIMITIVE="${UPPER_GUIDANCE_BANK_PER_PRIMITIVE:-16}"
export UPPER_GUIDANCE_HISTORY_ENABLED=0
export UPPER_GUIDANCE_TEMPORAL_ENABLED=1
export UPPER_GUIDANCE_TEMPORAL_EVIDENCE_DECAY="${UPPER_GUIDANCE_TEMPORAL_EVIDENCE_DECAY:-0.95}"
export UPPER_GUIDANCE_TEMPORAL_ADVANCE_EVIDENCE="${UPPER_GUIDANCE_TEMPORAL_ADVANCE_EVIDENCE:-0.45}"
export UPPER_GUIDANCE_TEMPORAL_SAME_STAGE_BUDGET="${UPPER_GUIDANCE_TEMPORAL_SAME_STAGE_BUDGET:-2}"
export UPPER_GUIDANCE_TEMPORAL_MAX_ROLLBACK="${UPPER_GUIDANCE_TEMPORAL_MAX_ROLLBACK:-0.15}"
export UPPER_GUIDANCE_TEMPORAL_MAX_ADVANCE="${UPPER_GUIDANCE_TEMPORAL_MAX_ADVANCE:-0.40}"

case "${ARM}" in
  upper_fusion_action|upper_only)
    export UPPER_GUIDANCE_HEAD="${SELF_ROOT}/heads/upper/best.pt"
    export UPPER_GUIDANCE_TEMPORAL_POSTERIOR="${UPPER_GUIDANCE_TEMPORAL_POSTERIOR:-0.75}"
    export UPPER_GUIDANCE_TEMPORAL_PURITY="${UPPER_GUIDANCE_TEMPORAL_PURITY:-0.35}"
    export UPPER_GUIDANCE_LOWER_FEATURES=""
    export UPPER_GUIDANCE_UPPER_AGE=""
    export LOWER_FEATURE_EXPORT_ENABLED=0
    export LOWER_FEATURE_HEAD_CKPT=""
    ;;
  fusion_fusion_action|fusion_only)
    export UPPER_GUIDANCE_HEAD="${SELF_ROOT}/heads/fusion/best.pt"
    export UPPER_GUIDANCE_LOWER_FEATURES="${SELF_ROOT}/features/lower_tdense5/pooled_prefix.npy"
    export UPPER_GUIDANCE_UPPER_AGE="${SELF_ROOT}/features/upper_dense5_held/upper_age.npy"
    export LOWER_FEATURE_EXPORT_ENABLED=1
    export LOWER_FEATURE_HEAD_CKPT="${SELF_ROOT}/heads/fusion/best.pt"
    export UPPER_GUIDANCE_TEMPORAL_POSTERIOR="${UPPER_GUIDANCE_TEMPORAL_POSTERIOR:-0.50}"
    export UPPER_GUIDANCE_TEMPORAL_PURITY="${UPPER_GUIDANCE_TEMPORAL_PURITY:-0.25}"
    ;;
esac

case "${ARM}" in
  upper_fusion_action|fusion_fusion_action)
    export PREDIMEM_MODE=v1
    export HEAD_VARIANTS=fusion
    export MEMORY_ADMISSION=success
    export TASK_HEAD_CKPT="${SELF_ROOT}/heads/fusion/best.pt"
    export MEMORY_TOP_K=8
    export MEMORY_GUIDANCE_NORM_CAP=0.20
    export MEMORY_GUIDANCE_TOTAL_NORM_CAP=0.20
    MODE=v1
    ;;
  fusion_only|upper_only)
    export HEAD_VARIANTS=base
    export MEMORY_ADMISSION=none
    MODE=base
    ;;
esac

export RUN_ROOT="${RUN_ROOT:-${SELF_ROOT}/eval/history_knn_self_${ARM}_seed${SEED}_${STAMP}}"
exec bash "${ROOT}/scripts/robomemarena/run_predimem_dual_gpu_batch8_env32.sh" "${MODE}"
