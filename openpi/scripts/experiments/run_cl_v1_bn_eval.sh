#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
if [[ "${CL_GROUP_TAG:-}" != "BN" ]]; then
  echo "V1 CL control is restricted to CL_GROUP_TAG=BN." >&2
  exit 2
fi
export CL_GUIDANCE_VERSION=v1
exec bash "${SCRIPT_DIR}/run_cl_v0_eval.sh" "$@"
