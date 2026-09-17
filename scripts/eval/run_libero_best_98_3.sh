#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
if [[ "${1:-}" == "--print-config" ]]; then
  for name in libero_spatial libero_object libero_goal libero_10; do
    PYTHONPATH="${TRACEFLOW_ROOT}/src" "${PYTHON:-python3}" -m traceflow.cli config "${name}"
  done
  exit 0
fi

ROOT_RUN="${RUN_ROOT:-$(tf_default_output libero_best_98_3)}"
mkdir -p "${ROOT_RUN}"
echo "98.3% is a best-of-suite envelope from independent configurations; it is not a single 1966/2000 run."

RUN_ROOT="${ROOT_RUN}/libero_spatial" bash "${SCRIPT_DIR}/_run_libero_suite.sh" \
  libero_spatial libero_spatial libero/b6500_positive
RUN_ROOT="${ROOT_RUN}/libero_object" bash "${SCRIPT_DIR}/_run_libero_suite.sh" \
  libero_object libero_object libero/b50_positive libero/ncpi_negative
RUN_ROOT="${ROOT_RUN}/libero_goal" bash "${SCRIPT_DIR}/_run_libero_suite.sh" \
  libero_goal libero_goal libero/b6500_positive
RUN_ROOT="${ROOT_RUN}/libero_10" bash "${SCRIPT_DIR}/_run_libero_suite.sh" \
  libero_10 libero_10 libero/b50_positive libero/ncpi_negative

tf_init_interpreters
"${OPENPI_PYTHON}" -m traceflow.cli summarize-libero "${ROOT_RUN}" "${ROOT_RUN}/summary.json"
echo "Summary: ${ROOT_RUN}/summary.json"

