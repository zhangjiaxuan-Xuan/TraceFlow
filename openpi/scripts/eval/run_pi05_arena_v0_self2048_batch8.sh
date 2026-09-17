#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
exec bash "${SCRIPT_DIR}/run_pi05_arena_self_head.sh" v0 original2048_fp32 "$@"
