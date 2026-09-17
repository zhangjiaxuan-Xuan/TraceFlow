#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:?Usage: $0 success|both}"
case "${GROUP}" in success|both) ;; *) echo "GROUP must be success or both" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
START_ROUND=4 ROUNDS=10 \
  bash "${ROOT}/scripts/robomemarena/run_predimem_transferring_inherited_cl.sh" 8 "${GROUP}"
