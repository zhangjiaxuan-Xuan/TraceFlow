#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: $0 v0|v1|v3|v3_prior_only|v3_prior_decay|v3_1|v3_re [original2048_fp32|budget1024_fp32|capacity256_fp32]}"
SPEC="${2:-original2048_fp32}"
case "${MODE}" in
  v0|v1|v3|v3_prior_only|v3_prior_decay|v3_1|v3_re) ;;
  *) echo "Invalid Pi self-head memory mode: ${MODE}" >&2; exit 2 ;;
esac
case "${SPEC}" in
  original2048_fp32|budget1024_fp32|capacity256_fp32) ;;
  *) echo "Invalid Pi self-head spec: ${SPEC}" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SELF_ROOT="${SELF_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_self_v2/pi05_finetuned}"
ARTIFACT_ROOT="${SELF_ROOT}/${SPEC}"
DEFAULT_POLICY_DIR="${ROOT}/checkpoints/pi05_robomemarena_extra8_reactive/extra8_reactive_fullft_seed42_finite_loader/30000_pytorch"
LEGACY_POLICY_DIR="/path/to/local/CVPR26-OptimusVLA/openpi/checkpoints/pi05_robomemarena_extra8_reactive/extra8_reactive_fullft_seed42_finite_loader/30000_pytorch"
if [[ ! -f "${DEFAULT_POLICY_DIR}/model.safetensors" && -f "${LEGACY_POLICY_DIR}/model.safetensors" ]]; then
  DEFAULT_POLICY_DIR="${LEGACY_POLICY_DIR}"
fi
POLICY_DIR="${POLICY_DIR:-${DEFAULT_POLICY_DIR}}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
ANCHOR_ALIGNED="${ANCHOR_ALIGNED:-0}"
MEMORY_ACTION_ALIGNMENT="${MEMORY_ACTION_ALIGNMENT:-auto}"
EXPECTED_TEMPORAL_WINDOW="${EXPECTED_TEMPORAL_WINDOW:-3}"
if [[ "${ANCHOR_ALIGNED}" == "1" ]]; then
  [[ "${MEMORY_ACTION_ALIGNMENT}" != "auto" ]] || MEMORY_ACTION_ALIGNMENT="anchor_forward_v1"
  MEMORY_META_NAME="gpm_memory_meta_anchor_forward_v1.pt"
  if [[ "${MEMORY_ACTION_ALIGNMENT}" == "dense_frame_v3" ]]; then
    MEMORY_META_NAME="gpm_memory_meta_dense_frame_v3.pt"
    ALIGNMENT_TAG="_dense_frame_v3_frames${EXPECTED_TEMPORAL_WINDOW}"
  elif [[ "${MEMORY_ACTION_ALIGNMENT}" == "continuous_frame_v2" ]]; then
    ALIGNMENT_TAG="_continuous_frame_v2_frames${EXPECTED_TEMPORAL_WINDOW}"
  else
    ALIGNMENT_TAG="_anchor_forward_v1"
  fi
else
  [[ "${MEMORY_ACTION_ALIGNMENT}" != "auto" ]] || MEMORY_ACTION_ALIGNMENT="legacy_progress"
  MEMORY_META_NAME="gpm_memory_meta.pt"
  ALIGNMENT_TAG=""
fi

for path in \
  "${ARTIFACT_ROOT}/heads/lower/best.pt" \
  "${ARTIFACT_ROOT}/memory/lower/${MEMORY_META_NAME}" \
  "${ARTIFACT_ROOT}/memory/lower/gpm_memory.index" \
  "${ARTIFACT_ROOT}/memory/shared/gpm_memory_actions.npz" \
  "${ARTIFACT_ROOT}/best_variant.json"; do
  [[ -f "${path}" ]] || { echo "Missing Pi self-head artifact: ${path}" >&2; exit 2; }
done
"${OPENPI_PYTHON}" - "${ARTIFACT_ROOT}/heads/lower/best.pt" \
  "${ARTIFACT_ROOT}/best_variant.json" "${POLICY_DIR}" "${SPEC}" \
  "${ARTIFACT_ROOT}/memory/lower/${MEMORY_META_NAME}" "${ANCHOR_ALIGNED}" \
  "${MEMORY_ACTION_ALIGNMENT}" "${EXPECTED_TEMPORAL_WINDOW}" <<'PY'
import json
from pathlib import Path
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
selection = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_dims = {
    "original2048_fp32": 2048,
    "budget1024_fp32": 1024,
    "capacity256_fp32": 256,
}
expected_dim = expected_dims[sys.argv[4]]
if int(checkpoint["out_dim"]) != expected_dim:
    raise SystemExit(f"Head dimension mismatch: {checkpoint['out_dim']} != {expected_dim}")
if selection.get("variant") != "lower":
    raise SystemExit(f"Expected lower self-head, got {selection.get('variant')!r}")
producer = Path(checkpoint["lower_producer_policy_dir"]).resolve()
policy = Path(sys.argv[3]).resolve()
if producer != policy:
    raise SystemExit(f"Pi self-head producer mismatch: {producer} != {policy}")
metadata = torch.load(sys.argv[5], map_location="cpu", weights_only=False)
anchor_aligned = sys.argv[6] == "1"
alignment = sys.argv[7]
expected_temporal_window = int(sys.argv[8])
checkpoint_temporal_window = int(checkpoint.get("temporal_window", 1))
anchor_count = sum("anchor_frame" in entry for entry in metadata)
if anchor_aligned and anchor_count != len(metadata):
    raise SystemExit(
        f"Anchor metadata is incomplete: {anchor_count}/{len(metadata)} entries have anchor_frame"
    )
if not anchor_aligned and anchor_count:
    raise SystemExit(
        f"Legacy evaluation received anchor-aware metadata: {anchor_count}/{len(metadata)}"
    )
if alignment in ("continuous_frame_v2", "dense_frame_v3") and not anchor_aligned:
    raise SystemExit(f"{alignment} requires anchor-aligned metadata")
if alignment in ("continuous_frame_v2", "dense_frame_v3"):
    actual_window = checkpoint_temporal_window
    expected_offsets = [0] if expected_temporal_window == 1 else [-20, -10, 0]
    expected_lower_dim = 2048 * expected_temporal_window
    if expected_temporal_window not in (1, 3):
        raise SystemExit(f"Expected temporal window must be 1 or 3, got {expected_temporal_window}")
    if actual_window != expected_temporal_window:
        raise SystemExit(
            f"{alignment} temporal window mismatch: {actual_window} != {expected_temporal_window}"
        )
    if list(checkpoint.get("temporal_offsets", [0])) != expected_offsets:
        raise SystemExit(
            f"Unexpected temporal offsets: {checkpoint.get('temporal_offsets')}"
        )
    if int(checkpoint["lower_dim"]) != expected_lower_dim:
        raise SystemExit(
            f"Temporal head lower_dim must be {expected_lower_dim}, got {checkpoint['lower_dim']}"
        )
print(
    "Verified Pi self-head identity: "
    f"spec={sys.argv[4]} out_dim={expected_dim} policy={policy} "
    f"action_alignment={alignment} temporal_window={checkpoint_temporal_window}"
)
PY

if [[ "${RESUME:-0}" == "1" && -z "${RUN_ROOT:-}" ]]; then
  echo "RESUME=1 requires explicit RUN_ROOT." >&2
  exit 2
fi

NFE_TAG=""
if [[ "${MODE}" =~ ^(v3_prior_only|v3_prior_decay|v3_1|v3_re)$ ]]; then
  export NFE_FLOOR="${NFE_FLOOR:-3}"
  NFE_TAG="_nfe_floor${NFE_FLOOR}"
fi
if [[ -z "${RUN_ROOT:-}" ]]; then
  export RUN_ROOT="${ROOT}/logs/robomemarena_extra8_pi05_finetuned_ckpt30000_memory_${MODE}_self_${SPEC}${NFE_TAG}${ALIGNMENT_TAG}_batch8_$(date -u +%Y%m%d_%H%M%S)"
fi
export MEMORY_MODE="${MODE}"
export CHECKPOINT="${POLICY_DIR}"
export TASK_HEAD_CKPT="${ARTIFACT_ROOT}/heads/lower/best.pt"
export MEMORY_META_PATH="${ARTIFACT_ROOT}/memory/lower/${MEMORY_META_NAME}"
export FAISS_INDEX_PATH="${ARTIFACT_ROOT}/memory/lower/gpm_memory.index"
export MEMORY_ACTIONS_PATH="${ARTIFACT_ROOT}/memory/shared/gpm_memory_actions.npz"
export MEMORY_PROVENANCE_TAG="lower-self-checkpoint+${SPEC}${ALIGNMENT_TAG}"
export MEMORY_ACTION_ALIGNMENT

exec bash "${ROOT}/scripts/eval/run_pi05_robomemarena_extra8_finetuned_batch8.sh"
