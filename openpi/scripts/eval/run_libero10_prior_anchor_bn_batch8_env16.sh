#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
BN_BANK="${OPENPI_ROOT}/artifacts/cl_data_energy/banks/BN/both"
SUMMARY="${BN_BANK}/build_summary.json"

case "${MEMORY_PRIOR_GUIDANCE_VERSION:-}" in
  v3_prior_only|v3_prior_decay|v3_prior_decay_joint|v3_re_prior_decay) ;;
  *) echo "BN prior-anchor entrypoint requires a V3 prior-anchor version." >&2; exit 2 ;;
esac
if [[ "${MEMORY_PRIOR_GUIDANCE_VERSION}" == "v3_re_prior_decay" ]]; then
  export NFE_FLOOR="${NFE_FLOOR:-3}"
  NFE_TAG="_nfe_floor${NFE_FLOOR}"
else
  export NFE_FLOOR="${NFE_FLOOR:-1}"
  NFE_TAG=""
fi
if [[ -n "${MEMORY_VARIANT:-}" && "${MEMORY_VARIANT}" != "success_fail" ]]; then
  echo "BN prior-anchor evaluation fixes MEMORY_VARIANT=success_fail." >&2
  exit 2
fi
[[ -f "${SUMMARY}" ]] || { echo "Missing completed BN/both bank: ${SUMMARY}" >&2; exit 1; }

python3 - "${SUMMARY}" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
identity = (summary.get("group"), summary.get("admission"))
baseline = summary.get("baseline_items", {})
admitted = summary.get("admitted_items", {})
output = summary.get("output", {})
counts = (
    int(baseline.get("positive", -1)),
    int(baseline.get("negative", -1)),
    int(admitted.get("positive", -1)),
    int(admitted.get("negative", -1)),
    int(output.get("positive", {}).get("items", -1)),
    int(output.get("negative", {}).get("items", -1)),
)
expected = (6500, 0, 9503, 497, 16003, 497)
if identity != ("BN", "both") or counts != expected:
    raise SystemExit(
        f"BN/both bank mismatch: identity={identity}, counts={counts}, expected={expected}"
    )
print("BN/both bank verified: positive=16003 (B6500 + N9503), negative=N497")
PY

export MEMORY_VARIANT=success_fail
export TASK_HEAD_CKPT="${OPENPI_ROOT}/checkpoints/gpm_task_head.pt"
export MEMORY_META_PATH="${BN_BANK}/positive/gpm_memory_meta.pt"
export FAISS_INDEX_PATH="${BN_BANK}/positive/gpm_memory.index"
export MEMORY_ACTIONS_PATH="${BN_BANK}/positive/gpm_memory_actions.npz"
export NEGATIVE_MEMORY_META_PATH="${BN_BANK}/negative/gpm_negative_memory_meta.pt"
export NEGATIVE_FAISS_INDEX_PATH="${BN_BANK}/negative/gpm_negative_memory.index"
export NEGATIVE_MEMORY_ACTIONS_PATH="${BN_BANK}/negative/gpm_negative_memory_actions.npz"
export MEMORY_GUIDANCE_V2_MAGNITUDE_CAP=0
export MEMORY_PRIOR_GUIDANCE_FINAL_SCALE="${MEMORY_PRIOR_GUIDANCE_FINAL_SCALE:-0.01}"
export MEMORY_JOINT_FAILURE_PRIOR="${MEMORY_JOINT_FAILURE_PRIOR:-0.50}"
export BATCH_SIZE=8
export LIBERO_CLIENTS_PER_SUITE=16
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-logs/libero10_prior_substep_${MEMORY_PRIOR_GUIDANCE_VERSION}${NFE_TAG}_BN_both_batch8_env16_${RUN_ID}}"

exec bash "${SCRIPT_DIR}/run_libero10_prior_substep_batch8.sh" "$@"
