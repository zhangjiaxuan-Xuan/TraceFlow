#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SELF_ROOT="${SELF_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_joint_v3/predimem_2048_fp32}"
OPENPI_PY="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
BEST="${SELF_ROOT}/best_variant.json"

[[ -f "${BEST}" ]] || { echo "Missing joint-conditioned best variant: ${BEST}" >&2; exit 2; }
BEST_VARIANT="$("${OPENPI_PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["variant"])' "${BEST}")"
"${OPENPI_PY}" - "${SELF_ROOT}/heads/${BEST_VARIANT}/best.pt" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected_upper = "predimem_trajectory_ordered_runtime_generate_subtask_hidden_v4"
if not bool(checkpoint.get("joint_conditioned")):
    raise SystemExit("PrediMem validation requires a joint-conditioned retrieval head")
if checkpoint.get("lower_conditioning_protocol") != "upper_generated_subtask_v1":
    raise SystemExit("Lower features were not conditioned on upper-generated subtasks")
if checkpoint.get("upper_feature_protocol") != expected_upper:
    raise SystemExit(
        f"Upper feature protocol mismatch: {checkpoint.get('upper_feature_protocol')} != {expected_upper}"
    )
if not checkpoint.get("subtask_records_sha256"):
    raise SystemExit("Joint retrieval head lacks subtask provenance")
print(f"Verified joint-conditioned PrediMem head: variant={checkpoint['variant']}")
PY

export SELF_ROOT
exec bash "${ROOT}/scripts/robomemarena/run_predimem_v3_best_self_head_dual_gpu.sh"
