#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:?Usage: $0 success|failure|both p16_n8|p8_n16}"
CANDIDATE="${2:?Usage: $0 success|failure|both p16_n8|p8_n16}"
case "${GROUP}" in success|failure|both) ;; *) echo "GROUP must be success, failure, or both" >&2; exit 2 ;; esac
case "${CANDIDATE}" in p16_n8|p8_n16) ;; *) echo "Invalid candidate: ${CANDIDATE}" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RATIO_CANDIDATES="${CANDIDATE}" \
  exec bash "${ROOT}/scripts/robomemarena/run_predimem_transferring_inherited_cl_delayed_ratio.sh" "${GROUP}"
