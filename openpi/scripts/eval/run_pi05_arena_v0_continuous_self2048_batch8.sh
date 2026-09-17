#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ANCHOR_ALIGNED=1
export MEMORY_ACTION_ALIGNMENT=continuous_frame_v2
export SELF_ROOT="${SELF_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_self_v3_anchors64/pi05_finetuned}"

exec bash "${SCRIPT_DIR}/run_pi05_arena_self_head.sh" v0 original2048_fp32 "$@"
