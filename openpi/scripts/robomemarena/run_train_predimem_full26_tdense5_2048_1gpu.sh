#!/usr/bin/env bash
set -euo pipefail

# Full positive demonstration bank for the 26-task branch. Existing
# Sequence/Transferring producer caches are reused; only the remaining tasks
# are encoded and then merged by trajectory/frame identity.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/raw}"
ALL_SELECTION="${ALL_SELECTION:-${RAW_ROOT}/selection_sequence-occlusion-counting-transferring_all_seeds.json}"
TASK_IDS_CSV="${TASK_IDS_CSV:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}"
EXPECTED_TRAJECTORIES="${EXPECTED_TRAJECTORIES:-2600}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_full26_tdense5/predimem_2048_fp32}"
VLA_CONFIG="${VLA_CONFIG:-pi05_robomemarena_all26_reactive}"
UPPER_BATCH_SIZE="${UPPER_BATCH_SIZE:-48}"
LOWER_BATCH_SIZE="${LOWER_BATCH_SIZE:-128}"
REUSE_ROOT="${REUSE_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/retrieval_joint_v6_tdense5/predimem_2048_fp32}"
REUSE_TASK_IDS="${REUSE_TASK_IDS:-1,2,3,18,19,22,25,26}"
NEW_TASK_IDS="${NEW_TASK_IDS:-4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,23,24}"
OPENPI_PY="${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}"
VLM_PY="${VLM_PY:-/path/to/user/miniforge3/envs/predimem-vlm/bin/python}"
TASK_CONFIG="${TASK_CONFIG:-${ROOT}/../RoboMemArena/evaluation_benchmark/async_vlm26_reference/fullvlm_v2_26_memory_tasks.json}"
TASK_PROMPTS="${TASK_PROMPTS:-${ROOT}/../RoboMemArena/evaluation_benchmark/openpi_minimal_runtime/task_prompts.py}"
VLA_CKPT="${VLA_CKPT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower/checkpoints/vla_alltask_pytorch}"
VLM_CKPT="${VLM_CKPT:-/path/to/storage/datasets/robotics/RoboMemArena/models/PrediMem/vlm_tasks1to26_ckpt74500}"
VLA_NORM_SOURCE="${VLA_NORM_SOURCE:-/path/to/local/data/robomemarena/models/PrediMem/vla_alltask/assets/policy_assets/norm_stats.json}"
VLA_NORM_TARGET="${VLA_CKPT}/assets/robomemarena/all26_pi05_reactive/norm_stats.json"
VLA_REUSE_NORM="${VLA_CKPT}/assets/robomemarena/extra8_pi05_reactive/norm_stats.json"
DENSE_MANIFEST="${OUTPUT_ROOT}/features/anchors_stride5.jsonl"
UPPER_MANIFEST="${OUTPUT_ROOT}/features/upper_stride10.jsonl"
NEW_DENSE_MANIFEST="${OUTPUT_ROOT}/features/new_anchors_stride5.jsonl"
NEW_UPPER_MANIFEST="${OUTPUT_ROOT}/features/new_upper_stride10.jsonl"
NEW_ROOT="${OUTPUT_ROOT}/features/new_cache"
PIPELINE_LOG_DIR="${OUTPUT_ROOT}/logs"
PIPELINE_LOG="${PIPELINE_LOG:-${PIPELINE_LOG_DIR}/full26_pipeline.log}"

TRAIN_STARTED_AT="$(date -Is)"
on_exit() {
  status=$?
  echo "[$(date -Is)] Full26 pipeline exited status=${status} started=${TRAIN_STARTED_AT}" >&2
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

export PYTHONPATH="${ROOT}/packages/openpi-client/src:${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OPENPI_TORCH_COMPILE=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${PIPELINE_LOG_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1
echo "[$(date -Is)] Full26 pipeline start: lower_batch=${LOWER_BATCH_SIZE} head_batch=${HEAD_BATCH_SIZE:-512}"

cache_ready() {
  "${OPENPI_PY}" -c '
import json, pathlib, sys
import numpy as np
root, manifest, names = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).resolve(), sys.argv[3].split(",")
state_path = root / "upper_cache_state.json"
if not state_path.is_file(): raise SystemExit(1)
state = json.load(state_path.open())
if not state.get("complete") or pathlib.Path(state.get("manifest", "")).resolve() != manifest: raise SystemExit(1)
rows = int(state.get("rows", -1))
for name in names:
    path = root / name
    if not path.is_file(): raise SystemExit(1)
    if name.endswith(".npy") and len(np.load(path, mmap_mode="r")) != rows: raise SystemExit(1)
' "$1" "$2" "$3" 2>/dev/null
}

[[ -f "${ALL_SELECTION}" ]] || {
  echo "Missing all-task selection: ${ALL_SELECTION}" >&2
  exit 2
}
[[ -f "${VLA_NORM_SOURCE}" ]] || { echo "Missing official all-task norm stats: ${VLA_NORM_SOURCE}" >&2; exit 2; }
[[ -f "${VLA_REUSE_NORM}" ]] || { echo "Missing reused extra8 norm stats: ${VLA_REUSE_NORM}" >&2; exit 2; }
if [[ -f "${VLA_NORM_TARGET}" ]]; then
  cmp -s "${VLA_NORM_SOURCE}" "${VLA_NORM_TARGET}" || {
    echo "Existing all26 norm stats differ from the official all-task checkpoint" >&2
    exit 2
  }
else
  mkdir -p "${VLA_NORM_TARGET%/*}"
  cp "${VLA_NORM_SOURCE}" "${VLA_NORM_TARGET}"
fi
cmp -s "${VLA_NORM_TARGET}" "${VLA_REUSE_NORM}" || {
  echo "Cannot reuse extra8 Lower cache: extra8/all26 normalization differs" >&2
  exit 2
}

export RAW_ROOT TASK_IDS_CSV EXPECTED_TRAJECTORIES SELECTION OUTPUT_ROOT VLA_CONFIG
SELECTION="${ALL_SELECTION}"
export SELECTION

mkdir -p "${OUTPUT_ROOT}/features"
if [[ ! -f "${DENSE_MANIFEST}" ]]; then
  "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/prepare_predimem_extra8_anchors.py" \
    --selection "${ALL_SELECTION}" --raw-root "${RAW_ROOT}" --task-config "${TASK_CONFIG}" \
    --task-prompts "${TASK_PROMPTS}" --output "${DENSE_MANIFEST}" --task-ids "${TASK_IDS_CSV}" \
    --expected-trajectories "${EXPECTED_TRAJECTORIES}" --anchor-stride 5 \
    --history-offsets=-20,-10,0 --workers "${DATA_WORKERS:-48}"
fi
if [[ ! -f "${UPPER_MANIFEST}" ]]; then
  "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/prepare_predimem_extra8_anchors.py" \
    --selection "${ALL_SELECTION}" --raw-root "${RAW_ROOT}" --task-config "${TASK_CONFIG}" \
    --task-prompts "${TASK_PROMPTS}" --output "${UPPER_MANIFEST}" --task-ids "${TASK_IDS_CSV}" \
    --expected-trajectories "${EXPECTED_TRAJECTORIES}" --anchor-stride 10 \
    --history-offsets=0 --workers "${DATA_WORKERS:-48}"
fi

if [[ ! -f "${NEW_DENSE_MANIFEST}" ]]; then
  "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/prepare_predimem_extra8_anchors.py" \
    --selection "${ALL_SELECTION}" --raw-root "${RAW_ROOT}" --task-config "${TASK_CONFIG}" \
    --task-prompts "${TASK_PROMPTS}" --output "${NEW_DENSE_MANIFEST}" --task-ids "${NEW_TASK_IDS}" \
    --anchor-stride 5 --history-offsets=-20,-10,0 --workers "${DATA_WORKERS:-48}"
fi
if [[ ! -f "${NEW_UPPER_MANIFEST}" ]]; then
  "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/prepare_predimem_extra8_anchors.py" \
    --selection "${ALL_SELECTION}" --raw-root "${RAW_ROOT}" --task-config "${TASK_CONFIG}" \
    --task-prompts "${TASK_PROMPTS}" --output "${NEW_UPPER_MANIFEST}" --task-ids "${NEW_TASK_IDS}" \
    --anchor-stride 10 --history-offsets=0 --workers "${DATA_WORKERS:-48}"
fi

upper_cache_complete=0
if [[ -f "${NEW_ROOT}/upper_source/upper_cache_state.json" && -f "${NEW_ROOT}/upper_source/upper_features.npy" && -f "${NEW_ROOT}/upper_source/subtasks.jsonl" ]]; then
  upper_cache_complete="$("${OPENPI_PY}" -c 'import json,sys; print(int(bool(json.load(open(sys.argv[1])).get("complete", False))))' "${NEW_ROOT}/upper_source/upper_cache_state.json")"
fi
if [[ "${upper_cache_complete}" != "1" ]]; then
  mkdir -p "${NEW_ROOT}"
  "${VLM_PY}" "${ROOT}/scripts/robomemarena/cache_predimem_upper_features.py" \
    --manifest "${NEW_UPPER_MANIFEST}" --checkpoint "${VLM_CKPT}" --task-config "${TASK_CONFIG}" \
    --output-dir "${NEW_ROOT}/upper_source" --device cuda:0 --batch-size "${UPPER_BATCH_SIZE:-48}" \
    --workers "${DATA_WORKERS:-48}" --n-recent "${N_RECENT:-5}" --progress-every "${UPPER_PROGRESS_EVERY:-1000}" \
    --merge-distance "${D_MERGE:-6}" --keyframe-max "${K_MAX:-0}" --max-new-tokens "${UPPER_MAX_NEW_TOKENS:-256}" \
    --feature-dim 4096 --generate-subtasks
fi

if cache_ready "${OUTPUT_ROOT}/features/upper_stride10" "${UPPER_MANIFEST}" \
  "upper_features.npy,upper_completed.npy,subtasks.jsonl"; then
  echo "[$(date -Is)] Reusing completed full26 Upper merge"
else
  "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/merge_predimem_feature_cache.py" \
    --kind upper --full-manifest "${UPPER_MANIFEST}" \
    --base-manifest "${REUSE_ROOT}/features/upper_stride10.jsonl" --base-cache "${REUSE_ROOT}/features/upper_stride10" \
    --extra-manifest "${NEW_UPPER_MANIFEST}" --extra-cache "${NEW_ROOT}/upper_source" \
    --output-cache "${OUTPUT_ROOT}/features/upper_stride10"
fi

if cache_ready "${OUTPUT_ROOT}/features/upper_dense5_held" "${DENSE_MANIFEST}" \
  "upper_features.npy,upper_age.npy,upper_source_indices.npy,subtasks.jsonl"; then
  echo "[$(date -Is)] Reusing completed full26 dense5 Upper alignment"
else
  "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/align_predimem_upper_cache.py" \
    --source-manifest "${UPPER_MANIFEST}" --target-manifest "${DENSE_MANIFEST}" \
    --source-cache-dir "${OUTPUT_ROOT}/features/upper_stride10" \
    --output-dir "${OUTPUT_ROOT}/features/upper_dense5_held" --age-normalizer-steps 20
fi
if cache_ready "${NEW_ROOT}/upper_dense5_held" "${NEW_DENSE_MANIFEST}" \
  "upper_features.npy,upper_age.npy,upper_source_indices.npy,subtasks.jsonl"; then
  echo "[$(date -Is)] Reusing completed new-task dense5 Upper alignment"
else
  "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/align_predimem_upper_cache.py" \
    --source-manifest "${UPPER_MANIFEST}" --target-manifest "${NEW_DENSE_MANIFEST}" \
    --source-cache-dir "${OUTPUT_ROOT}/features/upper_stride10" \
    --output-dir "${NEW_ROOT}/upper_dense5_held" --age-normalizer-steps 20
fi

lower_cache_complete=0
if [[ -f "${NEW_ROOT}/lower/cache_state.json" && -f "${NEW_ROOT}/lower/pooled_prefix.npy" && -f "${NEW_ROOT}/lower/completed.npy" ]]; then
  lower_cache_complete="$("${OPENPI_PY}" -c 'import json,sys,numpy as np; s=json.load(open(sys.argv[1])); p=sys.argv[2]; a=np.load(p); print(int(int(s.get("rows", -1)) == len(a) and bool(np.all(a))))' "${NEW_ROOT}/lower/cache_state.json" "${NEW_ROOT}/lower/completed.npy" 2>/dev/null || true)"
fi
if [[ "${lower_cache_complete}" != "1" ]]; then
  "${OPENPI_PY}" "${ROOT}/scripts/memory/cache_prior_head_features.py" \
    --manifest "${NEW_DENSE_MANIFEST}" --output-dir "${NEW_ROOT}/lower" \
    --policy-dir "${VLA_CKPT}" --config-name "${VLA_CONFIG}" --device cuda \
    --batch-size "${LOWER_BATCH_SIZE:-96}" --auto-batch --min-batch-size "${LOWER_MIN_BATCH_SIZE:-8}" \
    --feature-dim 2048 --progress-every-batches "${LOWER_PROGRESS_EVERY_BATCHES:-5}" \
    --prompt-overrides "${NEW_ROOT}/upper_dense5_held/subtasks.jsonl"
fi

"${OPENPI_PY}" "${ROOT}/scripts/robomemarena/merge_predimem_feature_cache.py" \
  --kind lower --full-manifest "${DENSE_MANIFEST}" \
  --base-manifest "${REUSE_ROOT}/features/anchors_stride5.jsonl" --base-cache "${REUSE_ROOT}/features/lower_tdense5" \
  --extra-manifest "${NEW_DENSE_MANIFEST}" --extra-cache "${NEW_ROOT}/lower" \
  --output-cache "${OUTPUT_ROOT}/features/lower_tdense5" \
  --prompt-overrides "${OUTPUT_ROOT}/features/upper_dense5_held/subtasks.jsonl" \
  --config-name "${VLA_CONFIG}"

export DENSE_MANIFEST UPPER_MANIFEST UPPER_SOURCE_DIR="${OUTPUT_ROOT}/features/upper_stride10" \
  UPPER_ALIGNED_DIR="${OUTPUT_ROOT}/features/upper_dense5_held" LOWER_DIR="${OUTPUT_ROOT}/features/lower_tdense5" \
  UPPER_BATCH_SIZE LOWER_BATCH_SIZE VLA_CKPT VLA_CONFIG VLM_CKPT TASK_CONFIG TASK_PROMPTS
exec bash "${ROOT}/scripts/robomemarena/run_train_predimem_joint_tdense5_2048_1gpu.sh"
