#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export ANCHOR_ALIGNED=1
export NFE_FLOOR="${NFE_FLOOR:-3}"
export ENV_WORKERS="${ENV_WORKERS:-16}"
export INFERENCE_BATCH_SIZE=8
export INFERENCE_BATCH_WAIT_MS="${INFERENCE_BATCH_WAIT_MS:-5}"
exec bash "${SCRIPT_DIR}/run_pi05_arena_self_head.sh" v3_re original2048_fp32 "$@"
