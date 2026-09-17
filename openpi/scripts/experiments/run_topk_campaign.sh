#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
PYTHON="${CAMPAIGN_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-analysis/bin/python}"
MANIFEST="${1:?usage: run_topk_campaign.sh CAMPAIGN_MANIFEST [extra args]}"
shift

exec "${PYTHON}" "${SCRIPT_DIR}/run_topk_campaign.py" --manifest "${MANIFEST}" "$@"
