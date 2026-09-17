#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
if [[ "${1:-}" == "--print-config" ]]; then
  PYTHONPATH="${SCRIPT_DIR}/../../src" "${PYTHON:-python3}" -m traceflow.cli config arena_occlusion
  exit 0
fi
exec bash "${SCRIPT_DIR}/_run_arena.sh" arena_occlusion

