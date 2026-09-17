#!/usr/bin/env bash
set -euo pipefail

TRACEFLOW_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${TRACEFLOW_ROOT}/openpi"

tf_conda_root() {
  if [[ -n "${CONDA_ROOT:-}" ]]; then
    printf '%s\n' "${CONDA_ROOT}"
  else
    local conda_bin
    conda_bin="${CONDA_EXE:-$(command -v conda || true)}"
    [[ -n "${conda_bin}" ]] || { echo "Conda is unavailable; set CONDA_ROOT or interpreter overrides." >&2; return 2; }
    "${conda_bin}" info --base
  fi
}

tf_init_interpreters() {
  local conda_root
  conda_root="$(tf_conda_root)"
  export OPENPI_PYTHON="${OPENPI_PYTHON:-${conda_root}/envs/traceflow-openpi/bin/python}"
  export LIBERO_PYTHON="${LIBERO_PYTHON:-${conda_root}/envs/traceflow-libero/bin/python}"
  export PREDIMEM_PYTHON="${PREDIMEM_PYTHON:-${conda_root}/envs/traceflow-predimem/bin/python}"
  [[ -x "${OPENPI_PYTHON}" ]] || { echo "Missing OpenPI interpreter: ${OPENPI_PYTHON}" >&2; return 2; }
  export PYTHONPATH="${TRACEFLOW_ROOT}/src:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
}

tf_print_config_if_requested() {
  local config="$1"
  if [[ "${2:-}" == "--print-config" ]]; then
    PYTHONPATH="${TRACEFLOW_ROOT}/src" "${PYTHON:-python3}" -m traceflow.cli config "${config}"
    return 0
  fi
  return 1
}

tf_assets() {
  "${OPENPI_PYTHON}" -m traceflow.cli assets "$@"
}

tf_checkpoint() {
  "${OPENPI_PYTHON}" -m traceflow.cli checkpoints "$1"
}

tf_default_output() {
  local name="$1"
  printf '%s\n' "${TRACEFLOW_OUTPUT_ROOT:-${TRACEFLOW_ROOT}/outputs}/${name}_$(date -u +%Y%m%d_%H%M%S)"
}

