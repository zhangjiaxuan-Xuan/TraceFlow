#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARENA_ROOT="${ROOT}/../RoboMemArena"
OPENPI_PY="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
GPU="${GPU:-0}"

RAW_ROOT="${RAW_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/raw}"
SELECTION="${SELECTION:-${RAW_ROOT}/selection_sequence-transferring_all_seeds.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_self_v4_dense10/pi05_finetuned}"
MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/features/anchors_stride10.jsonl}"
FEATURE_DIR="${FEATURE_DIR:-${OUTPUT_ROOT}/features/lower}"
ARTIFACT_ROOT="${OUTPUT_ROOT}/original2048_fp32"
DEFAULT_POLICY_DIR="${ROOT}/checkpoints/pi05_robomemarena_extra8_reactive/extra8_reactive_fullft_seed42_finite_loader/30000_pytorch"
LEGACY_POLICY_DIR="/path/to/local/CVPR26-OptimusVLA/openpi/checkpoints/pi05_robomemarena_extra8_reactive/extra8_reactive_fullft_seed42_finite_loader/30000_pytorch"
if [[ ! -f "${DEFAULT_POLICY_DIR}/model.safetensors" && -f "${LEGACY_POLICY_DIR}/model.safetensors" ]]; then
  DEFAULT_POLICY_DIR="${LEGACY_POLICY_DIR}"
fi
POLICY_DIR="${POLICY_DIR:-${DEFAULT_POLICY_DIR}}"
CONFIG_NAME="${CONFIG_NAME:-pi05_robomemarena_extra8_reactive}"

for path in "${OPENPI_PY}" "${SELECTION}" "${POLICY_DIR}/model.safetensors"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export OPENPI_TORCH_COMPILE=0
export PYTHONUNBUFFERED=1
mkdir -p "${MANIFEST%/*}" "${FEATURE_DIR}"

if [[ ! -f "${MANIFEST}" ]]; then
  "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/prepare_predimem_extra8_anchors.py" \
    --selection "${SELECTION}" \
    --raw-root "${RAW_ROOT}" \
    --task-config "${ARENA_ROOT}/evaluation_benchmark/async_vlm26_reference/fullvlm_v2_26_memory_tasks.json" \
    --task-prompts "${ARENA_ROOT}/evaluation_benchmark/openpi_minimal_runtime/task_prompts.py" \
    --output "${MANIFEST}" \
    --anchor-stride 10 \
    --history-offsets=-20,-10,0 \
    --workers "${DATA_WORKERS:-24}"
fi

"${OPENPI_PY}" - "${MANIFEST}" <<'PY'
import json
import sys
from collections import defaultdict

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
by_action = defaultdict(list)
for row in rows:
    by_action[row["action_id"]].append(row)
assert len(by_action) == 800, len(by_action)
for action_id, items in by_action.items():
    items.sort(key=lambda row: row["global_frame_index"])
    frames = [int(row["global_frame_index"]) for row in items]
    assert frames[0] == 0, (action_id, frames[0])
    assert frames[-1] == int(items[-1]["trajectory_length"]) - 1
    assert all(0 < right - left <= 10 for left, right in zip(frames, frames[1:]))
    for row in items:
        assert int(row["stage_frame_index"]) == (
            int(row["global_frame_index"]) - int(row["stage_start_frame"])
        )
        assert [item["offset"] for item in row["temporal_context"]] == [-20, -10, 0]
print(f"Verified dense10 temporal manifest: trajectories={len(by_action)} rows={len(rows)}")
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Pi dense10 training preflight passed: ${OUTPUT_ROOT}"
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

echo "[stage 1/3] Resume dense temporal feature cache"
"${OPENPI_PY}" "${ROOT}/scripts/memory/cache_prior_head_features.py" \
  --manifest "${MANIFEST}" \
  --output-dir "${FEATURE_DIR}" \
  --policy-dir "${POLICY_DIR}" \
  --config-name "${CONFIG_NAME}" \
  --device cuda \
  --batch-size "${FEATURE_BATCH_SIZE:-96}" \
  --auto-batch \
  --min-batch-size "${FEATURE_MIN_BATCH_SIZE:-8}" \
  --feature-dim 2048

"${OPENPI_PY}" - "${FEATURE_DIR}/completed.npy" <<'PY'
import sys
import numpy as np

completed = np.load(sys.argv[1], mmap_mode="r")
count = int(completed.sum())
if count != len(completed):
    raise RuntimeError(f"Feature cache incomplete after extraction: {count}/{len(completed)}")
print(f"Verified complete dense temporal feature cache: {count}/{len(completed)}", flush=True)
PY

echo "[stage 2/3] Train stage-relative 6144-1024-2048 retrieval head"
"${OPENPI_PY}" "${ROOT}/scripts/robomemarena/train_predimem_three_heads.py" \
  --manifest "${MANIFEST}" \
  --lower-features "${FEATURE_DIR}/pooled_prefix.npy" \
  --output-root "${ARTIFACT_ROOT}" \
  --device cuda:0 \
  --variants lower \
  --hidden-dim 1024 \
  --out-dim 2048 \
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

echo "[stage 3/3] Build dense-frame metadata and enforce the temporal gate"
"${OPENPI_PY}" "${ROOT}/scripts/robomemarena/build_anchor_aligned_memory_meta.py" \
  --manifest "${MANIFEST}" \
  --source-meta "${ARTIFACT_ROOT}/memory/lower/gpm_memory_meta.pt" \
  --output-meta "${ARTIFACT_ROOT}/memory/lower/gpm_memory_meta_dense_frame_v3.pt" \
  --protocol dense_frame_v3 \
  --overwrite

"${OPENPI_PY}" - "${ARTIFACT_ROOT}/best_variant.json" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
metrics = record["metrics"]
mae = float(metrics["stage_frame_mae@1"])
p95 = float(metrics["stage_frame_p95@1"])
recall = float(metrics["task_stage_recall@1"])
print(
    f"Dense temporal gate: task_stage_R@1={recall:.6f} "
    f"frame_MAE={mae:.3f} frame_P95={p95:.3f}",
    flush=True,
)
if recall < 0.99 or mae > 6.0 or p95 > 6.0:
    raise SystemExit(
        "Dense temporal gate failed; artifacts were preserved, but evaluation is blocked."
    )
PY

echo "Pi0.5 dense10 self-head complete: ${ARTIFACT_ROOT}"
