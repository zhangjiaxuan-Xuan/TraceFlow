#!/usr/bin/env bash
set -euo pipefail

TRAIN_STARTED_AT="$(date -Is)"
on_exit() {
  status=$?
  echo "[$(date -Is)] PrediMem training process exited status=${status} started=${TRAIN_STARTED_AT}" >&2
}
on_signal() {
  signal="$1"
  status="$2"
  echo "[$(date -Is)] PrediMem training received signal=${signal}" >&2
  exit "${status}"
}
trap on_exit EXIT
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARENA_ROOT="${ARENA_ROOT:-${ROOT}/../RoboMemArena}"
OPENPI_PY="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
VLM_PY="${VLM_PY:-/path/to/user/miniforge3/envs/predimem-vlm/bin/python}"
GPU="${GPU:-0}"

RAW_ROOT="${RAW_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/raw}"
TASK_IDS_CSV="${TASK_IDS_CSV:-1,2,3,18,19,22,25,26}"
EXPECTED_TRAJECTORIES="${EXPECTED_TRAJECTORIES:-800}"
TASK_IDS_CSV="${TASK_IDS_CSV:-1,2,3,18,19,22,25,26}"
EXPECTED_TRAJECTORIES="${EXPECTED_TRAJECTORIES:-800}"
SELECTION="${SELECTION:-${RAW_ROOT}/selection_sequence-transferring_all_seeds.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_joint_v6_tdense5/predimem_2048_fp32}"
DENSE_MANIFEST="${DENSE_MANIFEST:-${OUTPUT_ROOT}/features/anchors_stride5.jsonl}"
UPPER_MANIFEST="${UPPER_MANIFEST:-${OUTPUT_ROOT}/features/upper_stride10.jsonl}"
UPPER_SOURCE_DIR="${UPPER_SOURCE_DIR:-${OUTPUT_ROOT}/features/upper_stride10}"
UPPER_ALIGNED_DIR="${UPPER_ALIGNED_DIR:-${OUTPUT_ROOT}/features/upper_dense5_held}"
LOWER_DIR="${LOWER_DIR:-${OUTPUT_ROOT}/features/lower_tdense5}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/train_joint_tdense5.log}"

SOURCE_ROOT="${SOURCE_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower}"
VLA_CKPT="${VLA_CKPT:-${SOURCE_ROOT}/checkpoints/vla_alltask_pytorch}"
VLA_CONFIG="${VLA_CONFIG:-pi05_robomemarena_extra8_reactive}"
VLM_CKPT="${VLM_CKPT:-/path/to/storage/datasets/robotics/RoboMemArena/models/PrediMem/vlm_tasks1to26_ckpt74500}"
TASK_CONFIG="${TASK_CONFIG:-${ARENA_ROOT}/evaluation_benchmark/async_vlm26_reference/fullvlm_v2_26_memory_tasks.json}"
TASK_PROMPTS="${TASK_PROMPTS:-${ARENA_ROOT}/evaluation_benchmark/openpi_minimal_runtime/task_prompts.py}"

for path in \
  "${OPENPI_PY}" "${VLM_PY}" "${SELECTION}" \
  "${VLA_CKPT}/model.safetensors" "${VLM_CKPT}/config.json" \
  "${TASK_CONFIG}" "${TASK_PROMPTS}"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done

mkdir -p "${LOG_DIR}" "${DENSE_MANIFEST%/*}"
export TASK_IDS_CSV EXPECTED_TRAJECTORIES
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/packages/openpi-client/src:${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export OPENPI_TORCH_COMPILE=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${DENSE_MANIFEST}" ]]; then
  "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/prepare_predimem_extra8_anchors.py" \
    --selection "${SELECTION}" --raw-root "${RAW_ROOT}" \
    --task-config "${TASK_CONFIG}" --task-prompts "${TASK_PROMPTS}" \
    --output "${DENSE_MANIFEST}" --task-ids "${TASK_IDS_CSV}" \
    --expected-trajectories "${EXPECTED_TRAJECTORIES}" --anchor-stride 5 \
    --history-offsets=-20,-10,0 --workers "${DATA_WORKERS:-24}"
fi
if [[ ! -f "${UPPER_MANIFEST}" ]]; then
  "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/prepare_predimem_extra8_anchors.py" \
    --selection "${SELECTION}" --raw-root "${RAW_ROOT}" \
    --task-config "${TASK_CONFIG}" --task-prompts "${TASK_PROMPTS}" \
    --output "${UPPER_MANIFEST}" --task-ids "${TASK_IDS_CSV}" \
    --expected-trajectories "${EXPECTED_TRAJECTORIES}" --anchor-stride 10 \
    --history-offsets=0 --workers "${DATA_WORKERS:-24}"
fi

"${OPENPI_PY}" - "${DENSE_MANIFEST}" "${UPPER_MANIFEST}" <<'PY'
import json
import sys
from collections import defaultdict

def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

dense, upper = map(read, sys.argv[1:])
for rows, stride, offsets in ((dense, 5, [-20, -10, 0]), (upper, 10, [0])):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["action_id"]].append(row)
        assert [item["offset"] for item in row["temporal_context"]] == offsets
    expected = int(__import__("os").environ["EXPECTED_TRAJECTORIES"])
    assert len(grouped) == expected, (len(grouped), expected)
    for action_id, items in grouped.items():
        frames = [int(row["global_frame_index"]) for row in items]
        assert frames[0] == 0 and frames[-1] == int(items[-1]["trajectory_length"]) - 1
        assert all(0 < b - a <= stride for a, b in zip(frames, frames[1:])), action_id
print(f"Verified manifests: dense5={len(dense)} upper_stride10={len(upper)}")
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "PrediMem joint Tdense5 training preflight passed: ${OUTPUT_ROOT}"
  exit 0
fi

exec > >(tee -a "${RUN_LOG}") 2>&1
"${OPENPI_PY}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA preflight failed")
print(f"CUDA={torch.cuda.get_device_name(0)} free={torch.cuda.mem_get_info()[0] / 2**30:.1f}GiB", flush=True)
PY

if [[ -z "${UPPER_BATCH_SIZE:-}" ]]; then
  gpu_total_gib="$("${VLM_PY}" -c 'import torch; print(torch.cuda.get_device_properties(0).total_memory / 2**30)')"
  UPPER_BATCH_SIZE="$("${OPENPI_PY}" -c \
    'import sys; total=float(sys.argv[1]); print(48 if total >= 70 else (24 if total >= 35 else 8))' \
    "${gpu_total_gib}")"
fi
echo "PrediMem Upper batch=${UPPER_BATCH_SIZE} (override with UPPER_BATCH_SIZE=N)"

echo "[$(date -Is)] Stage 1/5: causal Upper cache at the real 10-step request cadence"
"${VLM_PY}" "${ROOT}/scripts/robomemarena/cache_predimem_upper_features.py" \
  --manifest "${UPPER_MANIFEST}" --checkpoint "${VLM_CKPT}" \
  --task-config "${TASK_CONFIG}" --output-dir "${UPPER_SOURCE_DIR}" \
  --device cuda:0 --batch-size "${UPPER_BATCH_SIZE}" \
  --workers "${DATA_WORKERS:-24}" --n-recent "${N_RECENT:-5}" \
  --progress-every "${UPPER_PROGRESS_EVERY:-1000}" \
  --merge-distance "${D_MERGE:-6}" --keyframe-max "${K_MAX:-0}" \
  --max-new-tokens "${UPPER_MAX_NEW_TOKENS:-256}" --feature-dim 4096 \
  --generate-subtasks

echo "[$(date -Is)] Stage 2/5: causal hold Upper features onto dense5 Lower rows"
"${OPENPI_PY}" "${ROOT}/scripts/robomemarena/align_predimem_upper_cache.py" \
  --source-manifest "${UPPER_MANIFEST}" --target-manifest "${DENSE_MANIFEST}" \
  --source-cache-dir "${UPPER_SOURCE_DIR}" --output-dir "${UPPER_ALIGNED_DIR}" \
  --age-normalizer-steps 20

echo "[$(date -Is)] Stage 3/5: Upper-conditioned three-frame Lower cache"
"${OPENPI_PY}" "${ROOT}/scripts/memory/cache_prior_head_features.py" \
  --manifest "${DENSE_MANIFEST}" --output-dir "${LOWER_DIR}" \
  --policy-dir "${VLA_CKPT}" --config-name "${VLA_CONFIG}" \
  --device cuda --batch-size "${LOWER_BATCH_SIZE:-64}" --auto-batch \
  --min-batch-size "${LOWER_MIN_BATCH_SIZE:-8}" --feature-dim 2048 \
  --progress-every-batches "${LOWER_PROGRESS_EVERY_BATCHES:-5}" \
  --prompt-overrides "${UPPER_ALIGNED_DIR}/subtasks.jsonl"

echo "[$(date -Is)] Stage 4/5: train 6144+4096 -> 1024 -> 2048 fusion retrieval head"
"${OPENPI_PY}" "${ROOT}/scripts/robomemarena/train_predimem_three_heads.py" \
  --manifest "${DENSE_MANIFEST}" \
  --lower-features "${LOWER_DIR}/pooled_prefix.npy" \
  --upper-features "${UPPER_ALIGNED_DIR}/upper_features.npy" \
  --upper-age "${UPPER_ALIGNED_DIR}/upper_age.npy" \
  --output-root "${OUTPUT_ROOT}" --device cuda:0 --variants lower,upper,fusion \
  --hidden-dim 1024 --out-dim 2048 --bank-dtype fp32 \
  --progress-coordinate stage --positive-progress-radius "${POSITIVE_PROGRESS_RADIUS:-0.025}" \
  --progress-recall-tolerance "${PROGRESS_RECALL_TOLERANCE:-0.04}" \
  --require-cross-trajectory-positive --temporal-regression-weight "${TEMPORAL_REGRESSION_WEIGHT:-0.5}" \
  --temporal-regression-scale "${TEMPORAL_REGRESSION_SCALE:-0.04}" \
  --deploy-all-anchors --skip-compression-study --require-joint-conditioning --resume \
  --retrieval-device "${HEAD_RETRIEVAL_DEVICE:-auto}" \
  --retrieval-query-batch-size "${HEAD_RETRIEVAL_QUERY_BATCH_SIZE:-4096}" \
  --projection-batch-size "${HEAD_PROJECTION_BATCH_SIZE:-8192}" \
  --epochs "${HEAD_EPOCHS:-30}" --steps-per-epoch "${HEAD_STEPS_PER_EPOCH:-100}" \
  --batch-size "${HEAD_BATCH_SIZE:-512}"

echo "[$(date -Is)] Stage 5/5: dense-frame action alignment metadata"
for variant in lower upper fusion; do
  "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/build_anchor_aligned_memory_meta.py" \
    --manifest "${DENSE_MANIFEST}" \
    --source-meta "${OUTPUT_ROOT}/memory/${variant}/gpm_memory_meta.pt" \
    --output-meta "${OUTPUT_ROOT}/memory/${variant}/gpm_memory_meta_dense_frame_v3.pt" \
    --protocol dense_frame_v3 --overwrite
done

echo "[$(date -Is)] PrediMem joint Tdense5 head complete: ${OUTPUT_ROOT}"
