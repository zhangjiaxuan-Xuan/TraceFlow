#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ANCHOR_ALIGNED=1
export MEMORY_ACTION_ALIGNMENT=continuous_frame_v2
export EXPECTED_TEMPORAL_WINDOW=1
export SELF_ROOT="${SELF_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_self_v3_anchors64_single_frame/pi05_finetuned}"

exec bash "${ROOT}/scripts/eval/run_pi05_arena_self_head.sh" v3_re original2048_fp32
