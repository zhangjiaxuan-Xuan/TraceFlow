#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-all}"
case "${TARGET}" in libero|plus|all) ;; *) echo "Usage: $0 libero|plus|all" >&2; exit 2 ;; esac
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
THIRD_PARTY="${TRACEFLOW_THIRD_PARTY:-${ROOT}/third_party}"
LIBERO_REVISION="${LIBERO_REVISION:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
PLUS_REVISION="${LIBERO_PLUS_REVISION:-4976dc30028e805ff8094b55501d532c48fec182}"
mkdir -p "${THIRD_PARTY}"

clone_pinned() {
  local url="$1" revision="$2" destination="$3" marker="$4" temporary
  if [[ -f "${destination}/${marker}" ]]; then return 0; fi
  [[ ! -e "${destination}" ]] || { echo "Refusing to overwrite incomplete ${destination}" >&2; return 2; }
  temporary="$(mktemp -d "${THIRD_PARTY}/.clone.XXXXXX")"
  trap 'rm -rf "${temporary}"' RETURN
  git clone --filter=blob:none --no-checkout "${url}" "${temporary}"
  git -c safe.directory="${temporary}" -C "${temporary}" checkout --detach "${revision}"
  mv "${temporary}" "${destination}"
  trap - RETURN
}

if [[ "${TARGET}" == "libero" || "${TARGET}" == "all" ]]; then
  clone_pinned https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_REVISION}" \
    "${THIRD_PARTY}/LIBERO" libero/libero/__init__.py
  libero_python="${CONDA_ROOT:-$(conda info --base)}/envs/traceflow-libero/bin/python"
  "${UV:-$(command -v uv)}" pip install --python "${libero_python}" --no-deps -e "${THIRD_PARTY}/LIBERO"
fi

if [[ "${TARGET}" == "plus" || "${TARGET}" == "all" ]]; then
  clone_pinned https://github.com/sylvestf/LIBERO-plus.git "${PLUS_REVISION}" \
    "${THIRD_PARTY}/LIBERO-plus" libero/libero/benchmark/task_classification.json
  PLUS_ROOT="${THIRD_PARTY}/LIBERO-plus"
  if [[ ! -d "${PLUS_ROOT}/libero/libero/assets" ]]; then
    CACHE="${TRACEFLOW_DOWNLOAD_CACHE:-${XDG_CACHE_HOME:-${HOME}/.cache}/traceflow/downloads}"
    ARCHIVE="${CACHE}/libero-plus-assets.zip"
    mkdir -p "${CACHE}"
    curl --fail --location --continue-at - --output "${ARCHIVE}" \
      https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip
    [[ "$(stat -c '%s' "${ARCHIVE}")" == "6395849578" ]] || { echo "LIBERO-Plus asset size mismatch" >&2; exit 1; }
    echo "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf  ${ARCHIVE}" | sha256sum -c -
    temporary="$(mktemp -d "${CACHE}/libero-plus.XXXXXX")"
    trap 'rm -rf "${temporary}"' EXIT
    unzip -q "${ARCHIVE}" -d "${temporary}"
    source_assets="${temporary}/assets"
    [[ -d "${source_assets}" ]] || source_assets="${temporary}/libero/libero/assets"
    [[ -d "${source_assets}" ]] || { echo "assets.zip has an unexpected layout" >&2; exit 1; }
    mv "${source_assets}" "${PLUS_ROOT}/libero/libero/assets"
  fi
fi
