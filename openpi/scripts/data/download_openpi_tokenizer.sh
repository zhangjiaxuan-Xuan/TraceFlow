#!/usr/bin/env bash
set -euo pipefail

OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/path/to/local/data/openpi}"
TARGET="${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"
PARTIAL="${TARGET}.partial"
EXPECTED_SIZE=4264023
URL="https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"

mkdir -p "$(dirname "${TARGET}")"
if [[ -f "${TARGET}" && "$(stat -c '%s' "${TARGET}")" -eq "${EXPECTED_SIZE}" ]]; then
  echo "Tokenizer already available: ${TARGET}"
  exit 0
fi

curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
  --output "${PARTIAL}" "${URL}"

actual_size="$(stat -c '%s' "${PARTIAL}")"
if [[ "${actual_size}" -ne "${EXPECTED_SIZE}" ]]; then
  echo "Tokenizer size mismatch: expected=${EXPECTED_SIZE} actual=${actual_size}" >&2
  exit 1
fi
mv "${PARTIAL}" "${TARGET}"
chmod a+rw "${TARGET}"
echo "Tokenizer ready: ${TARGET}"
