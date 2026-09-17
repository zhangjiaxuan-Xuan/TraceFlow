#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SELF_ROOT="${SELF_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_joint_v3/predimem_2048_fp32}"
export SELF_ROOT
export MEMORY_META_BASENAME=gpm_memory_meta_anchor_forward_v1.pt
export MEMORY_ALIGNMENT_TAG=anchor-forward-v1
export RUN_ROOT="${RUN_ROOT:-${SELF_ROOT}/eval/v3_best_anchor_forward_v1_$(date -u +%Y%m%d_%H%M%S)}"

exec bash "${ROOT}/scripts/robomemarena/run_predimem_v3_best_joint_head_dual_gpu.sh"
