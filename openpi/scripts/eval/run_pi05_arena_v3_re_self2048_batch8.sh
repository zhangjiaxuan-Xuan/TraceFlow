#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export NFE_FLOOR="${NFE_FLOOR:-3}"
exec bash "${SCRIPT_DIR}/run_pi05_arena_self_head.sh" v3_re original2048_fp32 "$@"
