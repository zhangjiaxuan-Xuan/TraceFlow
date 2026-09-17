#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ANCHOR_ALIGNED=1
export MEMORY_ACTION_ALIGNMENT=dense_frame_v3
export EXPECTED_TEMPORAL_WINDOW=3
export SELF_ROOT="${SELF_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_self_v5_dense5_budget/pi05_finetuned}"

exec bash "${ROOT}/scripts/eval/run_pi05_arena_self_head.sh" v1 budget1024_fp32
