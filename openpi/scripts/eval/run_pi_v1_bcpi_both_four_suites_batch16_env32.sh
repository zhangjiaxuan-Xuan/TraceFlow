#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MODE=both exec bash "${SCRIPT_DIR}/run_pi_v1_bcpi_four_suites_dynamic.sh" "$@"
