#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower}"
SELF_ROOT="${SELF_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_self_v2/predimem_2048_fp32}"
BEST="${SELF_ROOT}/best_variant.json"
OPENPI_PY="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
VLA_CKPT="${VLA_CKPT:-${SOURCE_ROOT}/checkpoints/vla_alltask_pytorch}"
VLM_CKPT="${VLM_CKPT:-/path/to/storage/datasets/robotics/RoboMemArena/models/PrediMem/vlm_tasks1to26_ckpt74500}"

[[ -f "${BEST}" ]] || { echo "Missing trained best-variant record: ${BEST}" >&2; exit 2; }
BEST_VARIANT="$("${OPENPI_PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["variant"])' "${BEST}")"
case "${BEST_VARIANT}" in lower|upper|fusion) ;; *) echo "Invalid best variant: ${BEST_VARIANT}" >&2; exit 2 ;; esac

"${OPENPI_PY}" - "${SELF_ROOT}/heads/${BEST_VARIANT}/best.pt" "${VLA_CKPT}" "${VLM_CKPT}" <<'PY'
from pathlib import Path
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
variant = checkpoint["variant"]
if int(checkpoint["out_dim"]) != 2048:
    raise SystemExit(f"PrediMem final head must be 2048D, got {checkpoint['out_dim']}")
if variant in ("lower", "fusion"):
    expected = Path(checkpoint["lower_producer_policy_dir"]).resolve()
    actual = Path(sys.argv[2]).resolve()
    if expected != actual:
        raise SystemExit(f"PrediMem lower producer mismatch: {expected} != {actual}")
if variant in ("upper", "fusion"):
    expected = Path(checkpoint["upper_producer_checkpoint"]).resolve()
    actual = Path(sys.argv[3]).resolve()
    if expected != actual:
        raise SystemExit(f"PrediMem upper producer mismatch: {expected} != {actual}")
print(f"Verified PrediMem self-model identity: variant={variant}")
PY

export AOSS_ROOT="${SELF_ROOT}"
export HEAD_VARIANTS="${BEST_VARIANT}"
export VLA_CKPT VLM_CKPT
export ACTION_STATS="${VLA_CKPT}/assets/robomemarena/extra8_pi05_reactive/norm_stats.json"
export RUN_ROOT="${RUN_ROOT:-${SELF_ROOT}/eval/v3_best_${BEST_VARIANT}_$(date -u +%Y%m%d_%H%M%S)}"

echo "PrediMem best self-head guidance: variant=${BEST_VARIANT} root=${SELF_ROOT}"
exec bash "${ROOT}/scripts/robomemarena/run_predimem_dual_gpu_batch8_env32.sh" v3
