#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
# Prefer Conda when both commands exist: older mamba releases print a formatted
# table for `info --base` instead of the machine-readable path Conda returns.
SOLVER="${MAMBA_EXE:-${CONDA_EXE:-$(command -v conda || command -v mamba || true)}}"
[[ -n "${SOLVER}" && -x "${SOLVER}" ]] || { echo "Install Conda or Mamba first." >&2; exit 2; }
UV="${UV:-$(command -v uv || true)}"
[[ -n "${UV}" && -x "${UV}" ]] || { echo "Install uv first (https://docs.astral.sh/uv/)." >&2; exit 2; }
if [[ -z "${CONDA_ROOT:-}" ]]; then
  solver_base="$("${SOLVER}" info --base)"
  if [[ "${solver_base}" == /* ]]; then
    CONDA_ROOT="${solver_base}"
  else
    CONDA_ROOT="$(printf '%s\n' "${solver_base}" | awk -F ' : ' '/base environment|root prefix/ {print $2; exit}')"
  fi
fi
[[ "${CONDA_ROOT}" == /* && -d "${CONDA_ROOT}" ]] || {
  echo "Could not determine an absolute Conda root from ${SOLVER}: ${CONDA_ROOT}" >&2
  exit 2
}
UV_INDEX_URL="${UV_INDEX_URL:-${PIP_INDEX_URL:-https://pypi.org/simple}}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

create() {
  local name="$1" version="$2" requirements="$3" python_path
  python_path="${CONDA_ROOT}/envs/${name}/bin/python"
  if [[ ! -x "${python_path}" ]]; then
    "${SOLVER}" create -y -n "${name}" "python=${version}" pip
  fi
  "${UV}" pip install --python "${python_path}" --index-url "${UV_INDEX_URL}" -r "${ROOT}/${requirements}"
  "${UV}" pip install --python "${python_path}" --index-url "${UV_INDEX_URL}" -e "${ROOT}"
}

create traceflow-openpi 3.11 requirements/openpi.txt
create traceflow-libero 3.10 requirements/libero.txt
create traceflow-predimem 3.10 requirements/predimem.txt

"${UV}" pip install --python "${CONDA_ROOT}/envs/traceflow-openpi/bin/python" \
  -e "${ROOT}/openpi/packages/openpi-client"
bash "${ROOT}/scripts/setup/fetch_benchmarks.sh" libero
echo "TraceFlow environments are ready under ${CONDA_ROOT}/envs."
