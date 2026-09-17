#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export MEMORY_MODE=v3
exec bash "${SCRIPT_DIR}/run_pi05_robomemarena_extra8_finetuned_batch8.sh" "$@"
