#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SPEC="${1:?Usage: $0 original2048_fp32|capacity256_fp32}"
shift
exec bash "${SCRIPT_DIR}/run_pi05_arena_self_head.sh" v3 "${SPEC}" "$@"
