#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARENA_ROOT="${ROOT}/../RoboMemArena"
OPENPI_PY="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
GPU="${GPU:-0}"

RAW_ROOT="${RAW_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/raw}"
SELECTION="${SELECTION:-${RAW_ROOT}/selection_sequence-transferring_all_seeds.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_self_v3_anchors64/pi05_finetuned}"
MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/features/anchors64.jsonl}"
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
    --anchors-per-trajectory 64 \
    --history-offsets=-20,-10,0 \
    --workers "${DATA_WORKERS:-24}"
fi

"${OPENPI_PY}" - "${MANIFEST}" <<'PY'
import json
import sys
from collections import Counter

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
counts = Counter(row["action_id"] for row in rows)
assert len(counts) == 800, len(counts)
assert min(counts.values()) == 64 and max(counts.values()) == 64, (
    min(counts.values()),
    max(counts.values()),
)
assert len(rows) == 51_200, len(rows)
assert all(
    [item["offset"] for item in row["temporal_context"]] == [-20, -10, 0]
    for row in rows
)
print(f"Verified anchors64 temporal manifest: trajectories={len(counts)} rows={len(rows)}")
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Pi anchors64 training preflight passed: ${OUTPUT_ROOT}"
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

echo "[stage 1/3] Resume temporal feature cache"
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
print(f"Verified complete temporal feature cache: {count}/{len(completed)}", flush=True)
PY

echo "[stage 2/3] Train 6144-1024-2048 retrieval head"
"${OPENPI_PY}" "${ROOT}/scripts/robomemarena/train_predimem_three_heads.py" \
  --manifest "${MANIFEST}" \
  --lower-features "${FEATURE_DIR}/pooled_prefix.npy" \
  --output-root "${ARTIFACT_ROOT}" \
  --device cuda:0 \
  --variants lower \
  --hidden-dim 1024 \
  --out-dim 2048 \
  --bank-budget 64 \
  --bank-dtype fp32 \
  --resume \
  --epochs "${HEAD_EPOCHS:-30}" \
  --steps-per-epoch "${HEAD_STEPS_PER_EPOCH:-100}" \
  --batch-size "${HEAD_BATCH_SIZE:-512}"

echo "[stage 3/3] Build continuous-frame memory metadata"
"${OPENPI_PY}" "${ROOT}/scripts/robomemarena/build_anchor_aligned_memory_meta.py" \
  --manifest "${MANIFEST}" \
  --source-meta "${ARTIFACT_ROOT}/memory/lower/gpm_memory_meta.pt" \
  --output-meta "${ARTIFACT_ROOT}/memory/lower/gpm_memory_meta_anchor_forward_v1.pt" \
  --overwrite

echo "Pi0.5 anchors64 self-head complete: ${ARTIFACT_ROOT}"
