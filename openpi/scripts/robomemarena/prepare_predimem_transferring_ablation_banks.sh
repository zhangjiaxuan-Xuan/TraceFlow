#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENPI_PY="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
COLLECTOR_ROOT="${COLLECTOR_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/transferring_memory_eval/collector_success_20260808_030251/fusion}"
POS_ROOT="${POS_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/transferring_memory_success}"
NEG_ROOT="${NEG_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/transferring_memory_failure}"
WORKERS="${WORKERS:-16}"

extra=()
[[ "${OVERWRITE:-0}" == "1" ]] && extra+=(--overwrite)
"${OPENPI_PY}" "${ROOT}/scripts/robomemarena/build_predimem_recorded_bank.py" \
  --run-root "${COLLECTOR_ROOT}" --label success --output "${POS_ROOT}" \
  --workers "${WORKERS}" "${extra[@]}"
"${OPENPI_PY}" "${ROOT}/scripts/robomemarena/build_predimem_recorded_bank.py" \
  --run-root "${COLLECTOR_ROOT}" --label failure --output "${NEG_ROOT}" \
  --workers "${WORKERS}" "${extra[@]}"
