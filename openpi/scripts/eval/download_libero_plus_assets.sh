#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)}"
PLUS_ROOT="${LIBERO_PLUS_ROOT:-${OPENPI_ROOT}/third_party/LIBERO-plus}"
CACHE_ROOT="${LIBERO_PLUS_ASSET_CACHE:-/path/to/storage/datasets/robotics/LIBERO-plus/assets_cache}"
ARCHIVE="${CACHE_ROOT}/assets.zip"
PARTS="${LIBERO_PLUS_ASSET_PARTS:-8}"
PARALLEL="${LIBERO_PLUS_ASSET_PARALLEL:-8}"
URL="https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip"
EXPECTED_SIZE=6395849578
EXPECTED_SHA256="96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"

[[ -f "${PLUS_ROOT}/libero/libero/__init__.py" ]] || {
  echo "Invalid LIBERO_PLUS_ROOT: ${PLUS_ROOT}" >&2
  exit 2
}
[[ "${PARTS}" =~ ^[1-9][0-9]*$ ]] || { echo "LIBERO_PLUS_ASSET_PARTS must be positive" >&2; exit 2; }
[[ "${PARALLEL}" =~ ^[1-9][0-9]*$ ]] || { echo "LIBERO_PLUS_ASSET_PARALLEL must be positive" >&2; exit 2; }
command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }
command -v unzip >/dev/null || { echo "unzip is required" >&2; exit 2; }
mkdir -p "${CACHE_ROOT}"

if [[ -f "${ARCHIVE}" ]]; then
  size="$(stat -c '%s' "${ARCHIVE}")"
  if [[ "${size}" -ne "${EXPECTED_SIZE}" ]]; then
    echo "Removing incomplete archive (${size}/${EXPECTED_SIZE} bytes)."
    rm -f "${ARCHIVE}"
  fi
fi

if [[ ! -f "${ARCHIVE}" ]]; then
  work="${CACHE_ROOT}/parts_${PARTS}"
  mkdir -p "${work}"
  chunk_size=$(( (EXPECTED_SIZE + PARTS - 1) / PARTS ))

  download_part() {
    local index="$1" start="$2" end="$3" part expected have request_start more got
    part="${work}/part_$(printf '%03d' "${index}")"
    expected=$((end - start + 1))
    while true; do
      have=0
      [[ -f "${part}" ]] && have="$(stat -c '%s' "${part}")"
      if ((have == expected)); then return 0; fi
      if ((have > expected)); then
        echo "Oversized part ${index}: ${have}/${expected}" >&2
        return 1
      fi
      request_start=$((start + have))
      more="${part}.more.$$"
      rm -f "${more}"
      downloaded=0
      for attempt in {1..20}; do
        if curl --fail --silent --show-error --location --http1.1 --retry 3 --retry-delay 2 \
          -H "Range: bytes=${request_start}-${end}" --output "${more}" "${URL}"; then
          downloaded=1
          break
        fi
        rm -f "${more}"
      done
      ((downloaded == 1)) || return 1
      got="$(stat -c '%s' "${more}")"
      if ((got <= 0 || have + got > expected)); then
        echo "Invalid response for part ${index}: existing=${have} received=${got} expected=${expected}" >&2
        rm -f "${more}"
        return 1
      fi
      cat "${more}" >> "${part}"
      rm -f "${more}"
    done
  }

  pids=()
  for ((index=0; index<PARTS; index++)); do
    start=$((index * chunk_size))
    end=$((start + chunk_size - 1))
    if ((end >= EXPECTED_SIZE)); then end=$((EXPECTED_SIZE - 1)); fi
    printf 'Queueing part %d/%d: bytes %d-%d\n' "$((index + 1))" "${PARTS}" "${start}" "${end}"
    download_part "${index}" "${start}" "${end}" &
    pids+=("$!")
    if ((${#pids[@]} == PARALLEL)); then
      for pid in "${pids[@]}"; do wait "${pid}"; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
  for ((index=0; index<PARTS; index++)); do
    part="${work}/part_$(printf '%03d' "${index}")"
    [[ -f "${part}" ]] || { echo "Missing downloaded part: ${part}" >&2; exit 1; }
  done
  cat "${work}"/part_??? > "${ARCHIVE}.tmp"
  [[ "$(stat -c '%s' "${ARCHIVE}.tmp")" -eq "${EXPECTED_SIZE}" ]] || {
    echo "Combined archive size mismatch." >&2
    exit 1
  }
  mv "${ARCHIVE}.tmp" "${ARCHIVE}"
fi

[[ "$(stat -c '%s' "${ARCHIVE}")" -eq "${EXPECTED_SIZE}" ]] || {
  echo "Archive size mismatch: ${ARCHIVE}" >&2
  exit 1
}
actual_sha256="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
[[ "${actual_sha256}" == "${EXPECTED_SHA256}" ]] || {
  echo "Archive SHA256 mismatch: ${actual_sha256}" >&2
  exit 1
}
echo "Archive verified: ${ARCHIVE}"
rm -rf "${CACHE_ROOT}/parts_${PARTS}"

target="${PLUS_ROOT}/libero/libero/assets"
if [[ -d "${target}" ]]; then
  echo "Assets already extracted at ${target}; leaving existing directory unchanged."
  exit 0
fi

extract_root="${CACHE_ROOT}/.extract.$$.tmp"
rm -rf "${extract_root}"
mkdir -p "${extract_root}"
trap 'rm -rf "${extract_root}"' EXIT
unzip -q "${ARCHIVE}" -d "${extract_root}"
source_assets="${extract_root}/assets"
if [[ ! -d "${source_assets}" ]]; then
  source_assets="${extract_root}/libero/libero/assets"
fi
if [[ ! -d "${source_assets}" ]]; then
  while IFS= read -r candidate; do
    if [[ -d "${candidate}/new_objects" && -d "${candidate}/scenes" ]]; then
      source_assets="${candidate}"
      break
    fi
  done < <(find "${extract_root}" -type d -name assets -print)
fi
[[ -d "${source_assets}" ]] || {
  echo "Could not find assets/ in the downloaded archive." >&2
  find "${extract_root}" -maxdepth 3 -type d >&2
  exit 1
}
mv "${source_assets}" "${target}"
echo "Extracted Plus assets to ${target}"
