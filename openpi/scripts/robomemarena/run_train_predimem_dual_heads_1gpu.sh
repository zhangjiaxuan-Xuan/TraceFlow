#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARENA_ROOT="${ROOT}/../RoboMemArena"
OPENPI_PY=${OPENPI_PY:-/path/to/user/miniforge3/envs/optimusvla-openpi/bin/python}
VLM_PY=${VLM_PY:-/path/to/user/miniforge3/envs/predimem-vlm/bin/python}
GPU=${GPU:-0}

AOSS_ROOT=${AOSS_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower}
RAW_ROOT=${RAW_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/raw}
SELECTION=${SELECTION:-${RAW_ROOT}/selection_sequence-transferring_all_seeds.json}
VLM_CKPT=${VLM_CKPT:-/path/to/local/data/robomemarena/models/PrediMem/vlm_tasks1to26_ckpt74500}
VLA_JAX_CKPT=${VLA_JAX_CKPT:-/path/to/local/data/robomemarena/models/PrediMem/vla_alltask}
VLA_PT_CKPT=${VLA_PT_CKPT:-${AOSS_ROOT}/checkpoints/vla_alltask_pytorch}
MANIFEST=${MANIFEST:-${AOSS_ROOT}/features/anchors16.jsonl}
LOWER_DIR=${LOWER_DIR:-${AOSS_ROOT}/features/lower}
UPPER_DIR=${UPPER_DIR:-${AOSS_ROOT}/features/upper}

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OPENPI_TORCH_COMPILE=0
mkdir -p "${AOSS_ROOT}/checkpoints" "${LOWER_DIR}" "${UPPER_DIR}"

for executable in "${OPENPI_PY}" "${VLM_PY}"; do
  [[ -x "${executable}" ]] || { echo "Missing Python environment: ${executable}"; exit 2; }
done
[[ -f "${SELECTION}" ]] || { echo "Missing 8-task selection: ${SELECTION}"; exit 2; }
[[ -d "${VLM_CKPT}" && -d "${VLA_JAX_CKPT}" ]] || { echo "Missing PrediMem checkpoint"; exit 2; }

if [[ ! -f "${MANIFEST}" ]]; then
  "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/prepare_predimem_extra8_anchors.py" \
    --selection "${SELECTION}" --raw-root "${RAW_ROOT}" \
    --task-config "${ARENA_ROOT}/evaluation_benchmark/async_vlm26_reference/fullvlm_v2_26_memory_tasks.json" \
    --task-prompts "${ARENA_ROOT}/evaluation_benchmark/openpi_minimal_runtime/task_prompts.py" \
    --output "${MANIFEST}" --anchors-per-trajectory 16 --workers "${DATA_WORKERS:-24}"
fi

if [[ ! -f "${VLA_PT_CKPT}/model.safetensors" ]]; then
  SITE="$("${OPENPI_PY}" -c 'import site; print(site.getsitepackages()[0])')"
  cp -r "${ROOT}/src/openpi/models_pytorch/transformers_replace/." "${SITE}/transformers/"
  "${OPENPI_PY}" "${ROOT}/examples/convert_jax_model_to_pytorch.py" \
    --checkpoint-dir "${VLA_JAX_CKPT}" \
    --config-name pi05_robomemarena_extra8_reactive \
    --output-path "${VLA_PT_CKPT}" --precision bfloat16
fi
OFFICIAL_NORM_STATS="${VLA_JAX_CKPT}/assets/policy_assets/norm_stats.json"
CONFIG_NORM_DIR="${VLA_PT_CKPT}/assets/robomemarena/extra8_pi05_reactive"
[[ -f "${OFFICIAL_NORM_STATS}" ]] || { echo "Missing official PrediMem norm stats: ${OFFICIAL_NORM_STATS}"; exit 2; }
mkdir -p "${CONFIG_NORM_DIR}"
cp "${OFFICIAL_NORM_STATS}" "${CONFIG_NORM_DIR}/norm_stats.json"

if [[ "${SMOKE:-0}" == "1" ]]; then
  "${OPENPI_PY}" -c "import numpy as n; assert n.load('${LOWER_DIR}/completed.npy').sum() >= 1"
  "${VLM_PY}" -c "import numpy as n; assert n.load('${UPPER_DIR}/upper_completed.npy').sum() >= 1"
  SMOKE_ROOT=${SMOKE_ROOT:-/tmp/predimem_runtime_smoke_$$}
  "${OPENPI_PY}" "${ROOT}/scripts/robomemarena/build_predimem_runtime_smoke_artifacts.py" \
    --manifest "${MANIFEST}" \
    --lower-features "${LOWER_DIR}/pooled_prefix.npy" \
    --upper-features "${UPPER_DIR}/upper_features.npy" \
    --output-root "${SMOKE_ROOT}"
  echo "PrediMem dual-head training entrypoint smoke passed: ${SMOKE_ROOT}"
  exit 0
fi

if [[ ! -f "${LOWER_DIR}/completed.npy" ]] || \
   ! "${OPENPI_PY}" -c "import numpy as n; assert n.load('${LOWER_DIR}/completed.npy').all()" 2>/dev/null; then
  "${OPENPI_PY}" "${ROOT}/scripts/memory/cache_prior_head_features.py" \
    --manifest "${MANIFEST}" --output-dir "${LOWER_DIR}" \
    --policy-dir "${VLA_PT_CKPT}" --config-name pi05_robomemarena_extra8_reactive \
    --device cuda --batch-size "${LOWER_BATCH_SIZE:-16}" --feature-dim 2048
fi

if [[ ! -f "${UPPER_DIR}/upper_completed.npy" ]] || \
   ! "${VLM_PY}" -c "import numpy as n; assert n.load('${UPPER_DIR}/upper_completed.npy').all()" 2>/dev/null; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${VLM_PY}" \
    "${ROOT}/scripts/robomemarena/cache_predimem_upper_features.py" \
    --manifest "${MANIFEST}" --checkpoint "${VLM_CKPT}" --output-dir "${UPPER_DIR}" \
    --batch-size "${UPPER_BATCH_SIZE:-2}" --workers "${DATA_WORKERS:-16}" --device cuda:0
fi

"${OPENPI_PY}" "${ROOT}/scripts/robomemarena/train_predimem_three_heads.py" \
  --manifest "${MANIFEST}" \
  --lower-features "${LOWER_DIR}/pooled_prefix.npy" \
  --upper-features "${UPPER_DIR}/upper_features.npy" \
  --output-root "${AOSS_ROOT}" --device cuda:0 \
  --epochs "${HEAD_EPOCHS:-30}" --steps-per-epoch "${HEAD_STEPS_PER_EPOCH:-100}" \
  --batch-size "${HEAD_BATCH_SIZE:-512}"
