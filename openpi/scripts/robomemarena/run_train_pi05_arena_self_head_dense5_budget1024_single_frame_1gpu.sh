#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENPI_PY="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
GPU="${GPU:-0}"

SOURCE_ROOT="${SOURCE_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_self_v5_dense5_budget/pi05_finetuned}"
MANIFEST="${MANIFEST:-${SOURCE_ROOT}/features/anchors_stride5.jsonl}"
FEATURE_DIR="${FEATURE_DIR:-${SOURCE_ROOT}/features/lower}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_self_v5_dense5_budget_single_frame/pi05_finetuned}"
ARTIFACT_ROOT="${OUTPUT_ROOT}/budget1024_fp32"

for path in \
  "${OPENPI_PY}" \
  "${MANIFEST}" \
  "${FEATURE_DIR}/pooled_prefix.npy" \
  "${FEATURE_DIR}/completed.npy" \
  "${FEATURE_DIR}/cache_state.json"; do
  [[ -e "${path}" ]] || { echo "Missing completed dense5 source artifact: ${path}" >&2; exit 2; }
done

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export OPENPI_TORCH_COMPILE=0
export PYTHONUNBUFFERED=1

"${OPENPI_PY}" - "${MANIFEST}" "${FEATURE_DIR}" <<'PY'
import json
import sys

import numpy as np

manifest, feature_dir = sys.argv[1:]
rows = sum(1 for line in open(manifest, encoding="utf-8") if line.strip())
features = np.load(f"{feature_dir}/pooled_prefix.npy", mmap_mode="r")
completed = np.load(f"{feature_dir}/completed.npy", mmap_mode="r")
state = json.load(open(f"{feature_dir}/cache_state.json", encoding="utf-8"))
assert features.shape == (rows, 6144), features.shape
assert len(completed) == rows and int(completed.sum()) == rows
assert int(state["base_feature_dim"]) == 2048
assert int(state["temporal_window"]) == 3
assert list(state["temporal_offsets"]) == [-20, -10, 0]
print(
    f"Verified reusable dense5 cache: rows={rows} source_dim=6144 "
    "single_frame_view=features[:, -2048:]",
    flush=True,
)
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Pi dense5 budget1024 single-frame head preflight passed: ${ARTIFACT_ROOT}"
  exit 0
fi

"${OPENPI_PY}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA preflight failed: PyTorch cannot access the GPU")
print(
    f"CUDA preflight passed: {torch.cuda.get_device_name(0)} "
    f"free={torch.cuda.mem_get_info()[0] / 2**30:.1f}GiB",
    flush=True,
)
PY

echo "[stage 1/2] Train single-frame 2048-1024-1024 retrieval head"
"${OPENPI_PY}" "${ROOT}/scripts/robomemarena/train_predimem_three_heads.py" \
  --manifest "${MANIFEST}" \
  --lower-features "${FEATURE_DIR}/pooled_prefix.npy" \
  --lower-feature-tail-dim 2048 \
  --output-root "${ARTIFACT_ROOT}" \
  --device cuda:0 \
  --variants lower \
  --hidden-dim 1024 \
  --out-dim 1024 \
  --bank-dtype fp32 \
  --progress-coordinate stage \
  --positive-progress-radius "${POSITIVE_PROGRESS_RADIUS:-0.025}" \
  --progress-recall-tolerance "${PROGRESS_RECALL_TOLERANCE:-0.04}" \
  --require-cross-trajectory-positive \
  --temporal-regression-weight "${TEMPORAL_REGRESSION_WEIGHT:-0.5}" \
  --temporal-regression-scale "${TEMPORAL_REGRESSION_SCALE:-0.04}" \
  --deploy-all-anchors \
  --skip-compression-study \
  --resume \
  --epochs "${HEAD_EPOCHS:-30}" \
  --steps-per-epoch "${HEAD_STEPS_PER_EPOCH:-100}" \
  --batch-size "${HEAD_BATCH_SIZE:-512}"

echo "[stage 2/2] Build dense-frame metadata and report temporal diagnostics"
"${OPENPI_PY}" "${ROOT}/scripts/robomemarena/build_anchor_aligned_memory_meta.py" \
  --manifest "${MANIFEST}" \
  --source-meta "${ARTIFACT_ROOT}/memory/lower/gpm_memory_meta.pt" \
  --output-meta "${ARTIFACT_ROOT}/memory/lower/gpm_memory_meta_dense_frame_v3.pt" \
  --protocol dense_frame_v3 \
  --overwrite

"${OPENPI_PY}" - "${ARTIFACT_ROOT}/best_variant.json" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding="utf-8"))["metrics"]
print(
    "Dense5 budget1024 single-frame diagnostics: "
    f"task_stage_R@1={float(metrics['task_stage_recall@1']):.6f} "
    f"frame_MAE={float(metrics['stage_frame_mae@1']):.3f} "
    f"frame_P95={float(metrics['stage_frame_p95@1']):.3f}",
    flush=True,
)
PY

echo "Pi0.5 dense5 budget1024 single-frame self-head complete: ${ARTIFACT_ROOT}"
