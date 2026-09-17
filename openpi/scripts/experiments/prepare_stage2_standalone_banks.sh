#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${OPENPI_ROOT}/.." >/dev/null 2>&1 && pwd)"
SMOL_ROOT="${PROJECT_ROOT}/smolvla"
PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
SCOPE="${EXPERIMENT_SCOPE:-formal}"
GROUP="${MEMORY_GROUP:-ALL}"
DEVICE="${DEVICE:-cuda}"
ARTIFACT_ROOT="${OPENPI_ROOT}/artifacts/cl_data_energy"
PROVIDER_DIR="${SMOL_ROOT}/artifacts/provider_control"
HEAD="${OPENPI_ROOT}/checkpoints/gpm_task_head.pt"

case "${SCOPE}:${GROUP}" in
  formal:N|formal:S|formal:ALL|special:N) ;;
  *) echo "Allowed preparation targets: formal:N/S/ALL or special:N" >&2; exit 2 ;;
esac

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

cache_special_n() {
  local manifest="${PROVIDER_DIR}/pi_success_failure.jsonl"
  local features="${ARTIFACT_ROOT}/standalone_features/special/N"
  require_file "${manifest}"
  "${PYTHON}" "${SMOL_ROOT}/scripts/subset_feature_cache.py" \
    --source-manifest "${ARTIFACT_ROOT}/manifests/new_pi.jsonl" \
    --source-feature-dir "${ARTIFACT_ROOT}/features/new_pi" \
    --target-manifest "${manifest}" \
    --output-dir "${features}"
}

build_bank() {
  local scope="$1" group="$2" manifest="$3" features="$4"
  local output="${ARTIFACT_ROOT}/standalone_banks/${scope}/${group}"
  require_file "${manifest}"
  require_file "${features}/pooled_prefix.npy"
  require_file "${features}/completed.npy"
  require_file "${features}/cache_state.json"
  if [[ -f "${output}/build_summary.json" ]]; then
    echo "Existing completed bank retained: ${output}"
    return
  fi
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing partial output directory: ${output}" >&2
    exit 1
  fi
  "${PYTHON}" "${OPENPI_ROOT}/scripts/memory/build_cl_memory_bank.py" \
    --group "${group}" \
    --manifest "${manifest}" \
    --feature-dir "${features}" \
    --checkpoint "${HEAD}" \
    --output-dir "${output}" \
    --admission both \
    --device "${DEVICE}"
}

require_file "${HEAD}"
if [[ "${SCOPE}" == "special" ]]; then
  if [[ ! -f "${PROVIDER_DIR}/pi_success_failure.jsonl" ]]; then
    "${PYTHON}" "${SMOL_ROOT}/scripts/prepare_provider_control.py"
  fi
  cache_special_n
  build_bank special N \
    "${PROVIDER_DIR}/pi_success_failure.jsonl" \
    "${ARTIFACT_ROOT}/standalone_features/special/N"
  exit 0
fi

if [[ "${GROUP}" == "N" || "${GROUP}" == "ALL" ]]; then
  build_bank formal N \
    "${ARTIFACT_ROOT}/manifests/new_pi.jsonl" \
    "${ARTIFACT_ROOT}/features/new_pi"
fi
if [[ "${GROUP}" == "S" || "${GROUP}" == "ALL" ]]; then
  build_bank formal S \
    "${ARTIFACT_ROOT}/manifests/smol.jsonl" \
    "${ARTIFACT_ROOT}/features/smol"
fi
