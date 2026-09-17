#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/path/to/user/miniforge3/envs/optimusvla-analysis/bin/python}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MANIFEST [--dry-run]" >&2
  exit 2
fi

cd "${OPENPI_ROOT}/.."
PYTHONPATH="${OPENPI_ROOT}/src:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${OPENPI_ROOT}/scripts/experiments/run_cross_policy_long_eval.py" "$@"
