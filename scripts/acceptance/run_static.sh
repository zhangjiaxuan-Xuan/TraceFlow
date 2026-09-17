#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
if [[ -n "${TRACEFLOW_TEST_PYTHON:-}" ]]; then
  TEST_PYTHON="${TRACEFLOW_TEST_PYTHON}"
elif command -v conda >/dev/null 2>&1 && [[ -x "$(conda info --base)/envs/traceflow-openpi/bin/python" ]]; then
  TEST_PYTHON="$(conda info --base)/envs/traceflow-openpi/bin/python"
else
  TEST_PYTHON="$(command -v python3)"
fi
find "${ROOT}" -type f -name '*.sh' -not -path '*/.git/*' -print0 | xargs -0 -n1 bash -n
PYTHONPATH="${ROOT}/src" "${TEST_PYTHON}" -m compileall -q "${ROOT}/src" "${ROOT}/tools" "${ROOT}/tests"
PYTHONPATH="${ROOT}/src" "${TEST_PYTHON}" -m pytest "${ROOT}/tests"
PYTHONPATH="${ROOT}/src" "${TEST_PYTHON}" "${ROOT}/tools/acceptance_scan.py" "${ROOT}"
echo "Static acceptance passed."
