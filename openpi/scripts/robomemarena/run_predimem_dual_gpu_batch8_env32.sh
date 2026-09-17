#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?Usage: $0 base|v0|v1|v3}
case "${MODE}" in base|v0|v1|v3) ;; *) echo "MODE must be base, v0, v1, or v3" >&2; exit 2 ;; esac

if [[ "${PREDIMEM_DUAL_RUN_SNAPSHOT:-0}" != "1" ]]; then
  export PREDIMEM_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  snapshot="$(mktemp /tmp/predimem_dual_gpu.XXXXXX.sh)"
  cp "$0" "${snapshot}"
  chmod +x "${snapshot}"
  export PREDIMEM_DUAL_RUN_SNAPSHOT=1
  exec bash "${snapshot}" "${MODE}"
fi

ROOT=${ROOT:-${PREDIMEM_PROJECT_ROOT:?PREDIMEM_PROJECT_ROOT is required}}
ARENA_ROOT=${ARENA_ROOT:-${ROOT}/../RoboMemArena}
cd "${ROOT}"
BENCHMARK_ROOT="${ARENA_ROOT}/evaluation_benchmark"
OPENPI_PY=${OPENPI_PY:-python}
VLM_PY=${VLM_PY:-${PREDIMEM_PYTHON:-python}}
UPPER_GPU=${UPPER_GPU:-0}
LOWER_GPU=${LOWER_GPU:-1}
UPPER_PORT=${UPPER_PORT:-8830}
LOWER_PORT=${LOWER_PORT:-8831}
UPPER_HOST=${UPPER_HOST:-127.0.0.1}
LOWER_HOST=${LOWER_HOST:-127.0.0.1}
EXTERNAL_UPPER_SERVER=${EXTERNAL_UPPER_SERVER:-0}
UPPER_BATCH_SIZE=${UPPER_BATCH_SIZE:-16}
UPPER_BATCH_WAIT_MS=${UPPER_BATCH_WAIT_MS:-50}
LOWER_BATCH_SIZE=${LOWER_BATCH_SIZE:-16}
LOWER_BATCH_WAIT_MS=${LOWER_BATCH_WAIT_MS:-100}
LOWER_PROBE_CROSS_ENV_BATCH=${LOWER_PROBE_CROSS_ENV_BATCH:-0}
ENV_WORKERS=${ENV_WORKERS:-32}
TASK_IDS=${TASK_IDS:-1,2,3,18,19,22,25,26}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-50}
SEED=${SEED:-7}
HEAD_VARIANTS=${HEAD_VARIANTS:-"lower upper fusion"}
REPLAN_STEPS=${REPLAN_STEPS:-10}
VLM_INTERVAL=${VLM_INTERVAL:-5}
N_RECENT=${N_RECENT:-5}
MAX_STEPS=${MAX_STEPS:-2500}
NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-10}
POST_GOAL_STEPS=${POST_GOAL_STEPS:-200}
SAVE_VIDEO=${SAVE_VIDEO:-1}
RECORD_MEMORY_DATA=${RECORD_MEMORY_DATA:-0}
MEMORY_TRACE_LEVEL=${MEMORY_TRACE_LEVEL:-full}
TRACE_STAGING_BASE=${TRACE_STAGING_BASE:-${RUN_ROOT:-${TMPDIR:-/tmp}}/trace_staging}
TRACE_LOCAL_WORKERS=${TRACE_LOCAL_WORKERS:-8}
TRACE_TRANSFER_WORKERS=${TRACE_TRANSFER_WORKERS:-8}
TRACE_TRANSFER_BATCH_SIZE=${TRACE_TRANSFER_BATCH_SIZE:-1024}
TRACE_FLUSH_TIMEOUT_SECONDS=${TRACE_FLUSH_TIMEOUT_SECONDS:-1800}
RESUME=${RESUME:-0}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
ALLOW_SAME_GPU=${ALLOW_SAME_GPU:-0}
SERVER_START_TIMEOUT_SECONDS=${SERVER_START_TIMEOUT_SECONDS:-1200}
WORKER_TIMEOUT_SECONDS=${WORKER_TIMEOUT_SECONDS:-172800}
WORKER_POOL_RETRIES=${WORKER_POOL_RETRIES:-5}
OPENPI_TORCH_COMPILE=${OPENPI_TORCH_COMPILE:-0}
MEMORY_GUIDANCE_NORM_CAP=${MEMORY_GUIDANCE_NORM_CAP:-0.20}
MEMORY_GUIDANCE_TOTAL_NORM_CAP=${MEMORY_GUIDANCE_TOTAL_NORM_CAP:-${MEMORY_GUIDANCE_NORM_CAP}}
MEMORY_GUIDANCE_LAMBDA_MAX=${MEMORY_GUIDANCE_LAMBDA_MAX:-0.20}
MEMORY_PRIOR_GUIDANCE_FINAL_SCALE=${MEMORY_PRIOR_GUIDANCE_FINAL_SCALE:-0.01}
MEMORY_ADMISSION=${MEMORY_ADMISSION:-success}
MEMORY_TOP_K=${MEMORY_TOP_K:-8}
MEMORY_ALLOWED_TASK_IDS=${MEMORY_ALLOWED_TASK_IDS:-}
MEMORY_EXACT_TASK_GATE=${MEMORY_EXACT_TASK_GATE:-0}
NEGATIVE_MEMORY_TOP_K=${NEGATIVE_MEMORY_TOP_K:-8}
NEGATIVE_MEMORY_MIN_SIMILARITY=${NEGATIVE_MEMORY_MIN_SIMILARITY:-0.975}
NEGATIVE_MEMORY_MIN_CONFIDENCE=${NEGATIVE_MEMORY_MIN_CONFIDENCE:-0.75}
POSITIVE_MEMORY_META_PATH=${POSITIVE_MEMORY_META_PATH:-}
POSITIVE_FAISS_INDEX_PATH=${POSITIVE_FAISS_INDEX_PATH:-}
POSITIVE_MEMORY_ACTIONS_PATH=${POSITIVE_MEMORY_ACTIONS_PATH:-}
NEGATIVE_MEMORY_META_PATH=${NEGATIVE_MEMORY_META_PATH:-}
NEGATIVE_FAISS_INDEX_PATH=${NEGATIVE_FAISS_INDEX_PATH:-}
NEGATIVE_MEMORY_ACTIONS_PATH=${NEGATIVE_MEMORY_ACTIONS_PATH:-}
TASK_HEAD_CKPT=${TASK_HEAD_CKPT:-}
PROTOCOL_LOCK_FILE=${PROTOCOL_LOCK_FILE:-}
GUIDANCE_ADAPTER_PATH=${GUIDANCE_ADAPTER_PATH:-}
UPPER_GUIDANCE_ENABLED=${UPPER_GUIDANCE_ENABLED:-0}
UPPER_GUIDANCE_HEAD=${UPPER_GUIDANCE_HEAD:-}
UPPER_GUIDANCE_MANIFEST=${UPPER_GUIDANCE_MANIFEST:-}
UPPER_GUIDANCE_FEATURES=${UPPER_GUIDANCE_FEATURES:-}
UPPER_GUIDANCE_LOWER_FEATURES=${UPPER_GUIDANCE_LOWER_FEATURES:-}
UPPER_GUIDANCE_UPPER_AGE=${UPPER_GUIDANCE_UPPER_AGE:-}
UPPER_GUIDANCE_TOP_K=${UPPER_GUIDANCE_TOP_K:-16}
UPPER_GUIDANCE_TEMPERATURE=${UPPER_GUIDANCE_TEMPERATURE:-0.07}
UPPER_GUIDANCE_MIN_TASK_PURITY=${UPPER_GUIDANCE_MIN_TASK_PURITY:-0.0}
UPPER_GUIDANCE_MIN_SUBTASK_CONFIDENCE=${UPPER_GUIDANCE_MIN_SUBTASK_CONFIDENCE:-0.0}
UPPER_GUIDANCE_INTERPOLATION=${UPPER_GUIDANCE_INTERPOLATION:-0.8}
UPPER_GUIDANCE_BANK_PER_PRIMITIVE=${UPPER_GUIDANCE_BANK_PER_PRIMITIVE:-16}
UPPER_GUIDANCE_BANK_SEED=${UPPER_GUIDANCE_BANK_SEED:-17}
UPPER_GUIDANCE_DEVICE=${UPPER_GUIDANCE_DEVICE:-cpu}
UPPER_GUIDANCE_ALLOWED_TASK_IDS=${UPPER_GUIDANCE_ALLOWED_TASK_IDS:-}
UPPER_GUIDANCE_HISTORY_ENABLED=${UPPER_GUIDANCE_HISTORY_ENABLED:-0}
UPPER_GUIDANCE_HISTORY_LOCK_OBSERVATIONS=${UPPER_GUIDANCE_HISTORY_LOCK_OBSERVATIONS:-2}
UPPER_GUIDANCE_HISTORY_ADVANCE_CONFIRMATIONS=${UPPER_GUIDANCE_HISTORY_ADVANCE_CONFIRMATIONS:-2}
UPPER_GUIDANCE_HISTORY_MAX_ROLLBACK=${UPPER_GUIDANCE_HISTORY_MAX_ROLLBACK:-0.10}
UPPER_GUIDANCE_HISTORY_MAX_ADVANCE=${UPPER_GUIDANCE_HISTORY_MAX_ADVANCE:-0.35}
UPPER_GUIDANCE_TEMPORAL_ENABLED=${UPPER_GUIDANCE_TEMPORAL_ENABLED:-0}
UPPER_GUIDANCE_FEEDBACK_MODE=${UPPER_GUIDANCE_FEEDBACK_MODE:-off}
UPPER_GUIDANCE_FEEDBACK_HORIZON=${UPPER_GUIDANCE_FEEDBACK_HORIZON:-4}
UPPER_GUIDANCE_FEEDBACK_CONFIRMATIONS=${UPPER_GUIDANCE_FEEDBACK_CONFIRMATIONS:-2}
UPPER_GUIDANCE_FEEDBACK_SIMILARITY_TOLERANCE=${UPPER_GUIDANCE_FEEDBACK_SIMILARITY_TOLERANCE:-0.02}
UPPER_GUIDANCE_FEEDBACK_PURITY_TOLERANCE=${UPPER_GUIDANCE_FEEDBACK_PURITY_TOLERANCE:-0.10}
UPPER_GUIDANCE_FEEDBACK_MAX_REINFORCEMENTS=${UPPER_GUIDANCE_FEEDBACK_MAX_REINFORCEMENTS:-2}
UPPER_GUIDANCE_FEEDBACK_LOWER_RESCUE_ENABLED=${UPPER_GUIDANCE_FEEDBACK_LOWER_RESCUE_ENABLED:-0}
UPPER_GUIDANCE_FEEDBACK_LOWER_RESCUE_CONFIRMATIONS=${UPPER_GUIDANCE_FEEDBACK_LOWER_RESCUE_CONFIRMATIONS:-2}
UPPER_GUIDANCE_FEEDBACK_LOWER_RESCUE_MIN_MARGIN=${UPPER_GUIDANCE_FEEDBACK_LOWER_RESCUE_MIN_MARGIN:-0.05}
UPPER_GUIDANCE_LOWER_GLOBAL_MODE=${UPPER_GUIDANCE_LOWER_GLOBAL_MODE:-off}
UPPER_GUIDANCE_LOWER_GLOBAL_TOP_K=${UPPER_GUIDANCE_LOWER_GLOBAL_TOP_K:-16}
UPPER_GUIDANCE_LOWER_GLOBAL_PROBE_CANDIDATES=${UPPER_GUIDANCE_LOWER_GLOBAL_PROBE_CANDIDATES:-4}
UPPER_GUIDANCE_LOWER_GLOBAL_MIN_POSTERIOR=${UPPER_GUIDANCE_LOWER_GLOBAL_MIN_POSTERIOR:-0.35}
UPPER_GUIDANCE_LOWER_GLOBAL_MIN_SCORE=${UPPER_GUIDANCE_LOWER_GLOBAL_MIN_SCORE:-0.45}
UPPER_GUIDANCE_LOWER_GLOBAL_MIN_MARGIN=${UPPER_GUIDANCE_LOWER_GLOBAL_MIN_MARGIN:-0.03}
UPPER_GUIDANCE_LOWER_GLOBAL_PROGRESS_SCALE=${UPPER_GUIDANCE_LOWER_GLOBAL_PROGRESS_SCALE:-0.15}
UPPER_STAGE_GUIDANCE_CONFIG=${UPPER_STAGE_GUIDANCE_CONFIG:-}
UPPER_GUIDANCE_TEMPORAL_POSTERIOR=${UPPER_GUIDANCE_TEMPORAL_POSTERIOR:-0.75}
UPPER_GUIDANCE_TEMPORAL_PURITY=${UPPER_GUIDANCE_TEMPORAL_PURITY:-0.35}
UPPER_GUIDANCE_TEMPORAL_EVIDENCE_DECAY=${UPPER_GUIDANCE_TEMPORAL_EVIDENCE_DECAY:-0.95}
UPPER_GUIDANCE_TEMPORAL_ADVANCE_EVIDENCE=${UPPER_GUIDANCE_TEMPORAL_ADVANCE_EVIDENCE:-0.45}
UPPER_GUIDANCE_TEMPORAL_SAME_STAGE_BUDGET=${UPPER_GUIDANCE_TEMPORAL_SAME_STAGE_BUDGET:-2}
UPPER_GUIDANCE_TEMPORAL_MAX_ROLLBACK=${UPPER_GUIDANCE_TEMPORAL_MAX_ROLLBACK:-0.15}
UPPER_GUIDANCE_TEMPORAL_MAX_ADVANCE=${UPPER_GUIDANCE_TEMPORAL_MAX_ADVANCE:-0.40}
LOWER_FEATURE_EXPORT_ENABLED=${LOWER_FEATURE_EXPORT_ENABLED:-0}
LOWER_FEATURE_HEAD_CKPT=${LOWER_FEATURE_HEAD_CKPT:-}

AOSS_ROOT=${AOSS_ROOT:-/path/to/storage/datasets/robotics/RoboMemArena/derived/predimem_dual_tower}
MEMORY_META_BASENAME=${MEMORY_META_BASENAME:-gpm_memory_meta.pt}
MEMORY_ALIGNMENT_TAG=${MEMORY_ALIGNMENT_TAG:-legacy-progress}
MEMORY_ACTION_ALIGNMENT=${MEMORY_ACTION_ALIGNMENT:-auto}
VLM_CKPT=${VLM_CKPT:-/path/to/local/data/robomemarena/models/PrediMem/vlm_tasks1to26_ckpt74500}
VLM_PROCESSOR_DIR=${VLM_PROCESSOR_DIR:-${VLM_CKPT}}
VLA_CKPT=${VLA_CKPT:-${AOSS_ROOT}/checkpoints/vla_alltask_pytorch}
VLA_CONFIG=${VLA_CONFIG:-pi05_robomemarena_extra8_reactive}
ACTION_STATS=${ACTION_STATS:-${VLA_CKPT}/assets/robomemarena/extra8_pi05_reactive/norm_stats.json}
TASK_CONFIG=${TASK_CONFIG:-${BENCHMARK_ROOT}/async_vlm26_reference/fullvlm_v2_26_memory_tasks.json}
OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${AOSS_ROOT}/runtime/openpi_data}
RUN_ROOT=${RUN_ROOT:-${AOSS_ROOT}/eval/${MODE}_dual_gpu_batch8_env32_$(date -u +%Y%m%d_%H%M%S)}

[[ "${EXTERNAL_UPPER_SERVER}" == "0" || "${EXTERNAL_UPPER_SERVER}" == "1" ]] || {
  echo "EXTERNAL_UPPER_SERVER must be 0 or 1." >&2
  exit 2
}
[[ "${EXTERNAL_UPPER_SERVER}" == "1" || "${UPPER_GPU}" != "${LOWER_GPU}" || "${ALLOW_SAME_GPU}" == "1" ]] || {
  echo "UPPER_GPU and LOWER_GPU must differ; ALLOW_SAME_GPU=1 is smoke-only." >&2
  exit 2
}
[[ "${UPPER_BATCH_SIZE}" -ge 2 && "${UPPER_BATCH_SIZE}" -le 64 \
  && "${LOWER_BATCH_SIZE}" -ge 2 && "${LOWER_BATCH_SIZE}" -le 64 ]] || {
  echo "UPPER_BATCH_SIZE and LOWER_BATCH_SIZE must be in [2, 64]. Production defaults to 16." >&2
  exit 2
}
[[ "${LOWER_PROBE_CROSS_ENV_BATCH}" == "0" || "${LOWER_PROBE_CROSS_ENV_BATCH}" == "1" ]] || {
  echo "LOWER_PROBE_CROSS_ENV_BATCH must be 0 or 1." >&2
  exit 2
}
if [[ "${LOWER_PROBE_CROSS_ENV_BATCH}" == "1" && "${LOWER_FEATURE_EXPORT_ENABLED}" != "1" ]]; then
  echo "Cross-environment Lower probe batching requires LOWER_FEATURE_EXPORT_ENABLED=1." >&2
  exit 2
fi
case "${MEMORY_ADMISSION}" in none|success|failure|both) ;; *)
  echo "MEMORY_ADMISSION must be none, success, failure, or both." >&2; exit 2 ;; esac
if [[ "${MODE}" == "base" && "${MEMORY_ADMISSION}" != "none" ]]; then
  echo "MODE=base requires MEMORY_ADMISSION=none." >&2
  exit 2
fi
[[ "${MEMORY_TOP_K}" -ge 1 && "${NEGATIVE_MEMORY_TOP_K}" -ge 1 ]] || {
  echo "Memory top-k values must be positive." >&2; exit 2
}
[[ "${WORKER_POOL_RETRIES}" =~ ^[0-9]+$ ]] || {
  echo "WORKER_POOL_RETRIES must be a non-negative integer." >&2
  exit 2
}
IFS=',' read -r -a task_array <<<"${TASK_IDS}"
task_count=${#task_array[@]}
[[ "${task_count}" -gt 0 && "${ENV_WORKERS}" -ge 1 ]] || {
  echo "TASK_IDS must be non-empty and ENV_WORKERS must be positive." >&2
  exit 2
}
case "${RECORD_MEMORY_DATA}:${MEMORY_TRACE_LEVEL}" in
  0:full|0:light|0:bank|1:full|1:light|1:bank) ;;
  *) echo "RECORD_MEMORY_DATA must be 0/1 and MEMORY_TRACE_LEVEL must be full/light/bank" >&2; exit 2 ;;
esac
[[ "${TRACE_LOCAL_WORKERS}" -eq 8 && "${TRACE_TRANSFER_WORKERS}" -eq 8 ]] || {
  echo "PrediMem staged recording requires exactly 8 local and 8 transfer workers." >&2; exit 2
}
[[ "${TRACE_TRANSFER_BATCH_SIZE}" -ge "${TRACE_TRANSFER_WORKERS}" ]] || {
  echo "TRACE_TRANSFER_BATCH_SIZE must be at least TRACE_TRANSFER_WORKERS." >&2; exit 2
}
for path in \
  "${OPENPI_PY}" "${VLM_PY}" "${VLM_CKPT}/config.json" "${VLA_CKPT}/model.safetensors" \
  "${ACTION_STATS}" "${TASK_CONFIG}" "${ROOT}/optimus_eval/predimem_upper_server.py" \
  "${ROOT}/optimus_eval/predimem_dual_worker.py" \
  "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done
if [[ "${UPPER_GUIDANCE_ENABLED}" == "1" ]]; then
  [[ -n "${UPPER_GUIDANCE_HEAD}" && -f "${UPPER_GUIDANCE_HEAD}" ]] || { echo "Missing upper guidance head: ${UPPER_GUIDANCE_HEAD}" >&2; exit 2; }
  [[ -n "${UPPER_GUIDANCE_MANIFEST}" && -f "${UPPER_GUIDANCE_MANIFEST}" ]] || { echo "Missing upper guidance manifest: ${UPPER_GUIDANCE_MANIFEST}" >&2; exit 2; }
  [[ -n "${UPPER_GUIDANCE_FEATURES}" && -f "${UPPER_GUIDANCE_FEATURES}" ]] || { echo "Missing upper guidance features: ${UPPER_GUIDANCE_FEATURES}" >&2; exit 2; }
  if [[ "${UPPER_GUIDANCE_LOWER_GLOBAL_MODE}" != "off" ]]; then
    [[ -n "${UPPER_GUIDANCE_LOWER_FEATURES}" && -f "${UPPER_GUIDANCE_LOWER_FEATURES}" ]] || {
      echo "Lower global feedback requires UPPER_GUIDANCE_LOWER_FEATURES." >&2; exit 2;
    }
    [[ -n "${UPPER_GUIDANCE_UPPER_AGE}" && -f "${UPPER_GUIDANCE_UPPER_AGE}" ]] || {
      echo "Lower global feedback requires UPPER_GUIDANCE_UPPER_AGE." >&2; exit 2;
    }
    [[ "${LOWER_FEATURE_EXPORT_ENABLED}" == "1" ]] || {
      echo "Lower global feedback requires LOWER_FEATURE_EXPORT_ENABLED=1." >&2; exit 2;
    }
  fi
fi
if [[ -n "${UPPER_STAGE_GUIDANCE_CONFIG}" ]]; then
  [[ -f "${UPPER_STAGE_GUIDANCE_CONFIG}" ]] || {
    echo "Missing stage-conditioned guidance config: ${UPPER_STAGE_GUIDANCE_CONFIG}" >&2; exit 2;
  }
  [[ "${UPPER_GUIDANCE_ENABLED}" == "1" ]] || {
    echo "Stage-conditioned guidance requires UPPER_GUIDANCE_ENABLED=1." >&2; exit 2;
  }
  [[ "${UPPER_GUIDANCE_HISTORY_ENABLED}:${UPPER_GUIDANCE_TEMPORAL_ENABLED}:${UPPER_GUIDANCE_FEEDBACK_MODE}" == "0:0:off" ]] || {
    echo "Stage-conditioned guidance replaces legacy history/temporal state machines." >&2; exit 2;
  }
  [[ "${UPPER_GUIDANCE_LOWER_GLOBAL_MODE}" != "off" ]] || {
    echo "Stage-conditioned guidance requires Lower counterfactual feedback." >&2; exit 2;
  }
fi
[[ "${UPPER_GUIDANCE_HISTORY_ENABLED}:${UPPER_GUIDANCE_TEMPORAL_ENABLED}" != "1:1" ]] || {
  echo "Upper hard-history and temporal-v2 gates are mutually exclusive." >&2
  exit 2
}
case "${UPPER_GUIDANCE_LOWER_GLOBAL_MODE}" in off|shadow|control) ;; *)
  echo "UPPER_GUIDANCE_LOWER_GLOBAL_MODE must be off, shadow, or control." >&2; exit 2 ;; esac
for variant in ${HEAD_VARIANTS}; do
  case "${variant}" in
    lower|upper|fusion) ;;
    base) [[ "${MEMORY_ADMISSION}" == "none" ]] || { echo "HEAD_VARIANTS=base requires MEMORY_ADMISSION=none" >&2; exit 2; } ;;
    *) echo "Invalid head variant: ${variant}" >&2; exit 2 ;;
  esac
  head_path="${TASK_HEAD_CKPT:-${AOSS_ROOT}/heads/${variant}/best.pt}"
  positive_meta="${POSITIVE_MEMORY_META_PATH:-${AOSS_ROOT}/memory/${variant}/${MEMORY_META_BASENAME}}"
  positive_index="${POSITIVE_FAISS_INDEX_PATH:-${AOSS_ROOT}/memory/${variant}/gpm_memory.index}"
  positive_actions="${POSITIVE_MEMORY_ACTIONS_PATH:-${AOSS_ROOT}/memory/shared/gpm_memory_actions.npz}"
  if [[ "${MEMORY_ADMISSION}" != "none" ]]; then
    [[ -f "${head_path}" ]] || { echo "Missing ${variant} head: ${head_path}" >&2; exit 2; }
  fi
  if [[ "${MEMORY_ADMISSION}" != "none" && "${MEMORY_ADMISSION}" != "failure" ]]; then
    for path in "${positive_meta}" "${positive_index}" "${positive_actions}"; do
      [[ -f "${path}" ]] || { echo "Missing positive ${variant} artifact: ${path}" >&2; exit 2; }
    done
  fi
  if [[ "${MEMORY_ADMISSION}" != "none" && "${MEMORY_ADMISSION}" != "success" ]]; then
    for path in "${NEGATIVE_MEMORY_META_PATH}" "${NEGATIVE_FAISS_INDEX_PATH}" "${NEGATIVE_MEMORY_ACTIONS_PATH}"; do
      [[ -f "${path}" ]] || { echo "Missing negative ${variant} artifact: ${path}" >&2; exit 2; }
    done
  fi
  if [[ "${MEMORY_META_BASENAME}" == "gpm_memory_meta_anchor_forward_v1.pt" ]]; then
    provenance="${AOSS_ROOT}/memory/${variant}/${MEMORY_META_BASENAME}.json"
    [[ -f "${provenance}" ]] || { echo "Missing ${variant} anchor provenance: ${provenance}" >&2; exit 2; }
    "${OPENPI_PY}" -c \
      'import json,sys; r=json.load(open(sys.argv[1])); assert r["protocol"]=="anchor_forward_v1" and int(r["entries"])>0, r' \
      "${provenance}"
  fi
done
if [[ "${MEMORY_ADMISSION}" != "none" ]]; then
  [[ -f "${AOSS_ROOT}/memory/shared/gpm_memory_actions.npz" ]] || {
    echo "Missing shared action store" >&2
    exit 2
  }
fi

memory_scope="${MEMORY_ALLOWED_TASK_IDS:-all}"
upper_guidance_protocol="off"
if [[ "${UPPER_GUIDANCE_ENABLED}" == "1" ]]; then
  upper_guidance_protocol="knnlm-k${UPPER_GUIDANCE_TOP_K}-lambda${UPPER_GUIDANCE_INTERPOLATION}-purity${UPPER_GUIDANCE_MIN_TASK_PURITY}-posterior${UPPER_GUIDANCE_MIN_SUBTASK_CONFIDENCE}"
  if [[ -n "${UPPER_GUIDANCE_LOWER_FEATURES}" ]]; then
    [[ -f "${UPPER_GUIDANCE_LOWER_FEATURES}" ]] || { echo "Missing Fusion lower feature bank: ${UPPER_GUIDANCE_LOWER_FEATURES}" >&2; exit 2; }
    [[ -f "${UPPER_GUIDANCE_UPPER_AGE}" ]] || { echo "Missing Fusion upper-age bank: ${UPPER_GUIDANCE_UPPER_AGE}" >&2; exit 2; }
    [[ "${LOWER_FEATURE_EXPORT_ENABLED}" == "1" ]] || { echo "Fusion language guidance requires LOWER_FEATURE_EXPORT_ENABLED=1" >&2; exit 2; }
    [[ -f "${LOWER_FEATURE_HEAD_CKPT}" ]] || { echo "Missing Fusion feature-export head: ${LOWER_FEATURE_HEAD_CKPT}" >&2; exit 2; }
    upper_guidance_protocol+="-fusion-causal"
  fi
  if [[ "${UPPER_GUIDANCE_HISTORY_ENABLED}" == "1" ]]; then
    upper_guidance_protocol+="-history-lock${UPPER_GUIDANCE_HISTORY_LOCK_OBSERVATIONS}-advance${UPPER_GUIDANCE_HISTORY_ADVANCE_CONFIRMATIONS}-rollback${UPPER_GUIDANCE_HISTORY_MAX_ROLLBACK}-jump${UPPER_GUIDANCE_HISTORY_MAX_ADVANCE}"
  elif [[ "${UPPER_GUIDANCE_TEMPORAL_ENABLED}" == "1" ]]; then
    upper_guidance_protocol+="-temporal-v2-p${UPPER_GUIDANCE_TEMPORAL_POSTERIOR}-q${UPPER_GUIDANCE_TEMPORAL_PURITY}-decay${UPPER_GUIDANCE_TEMPORAL_EVIDENCE_DECAY}-evidence${UPPER_GUIDANCE_TEMPORAL_ADVANCE_EVIDENCE}-budget${UPPER_GUIDANCE_TEMPORAL_SAME_STAGE_BUDGET}-rollback${UPPER_GUIDANCE_TEMPORAL_MAX_ROLLBACK}-jump${UPPER_GUIDANCE_TEMPORAL_MAX_ADVANCE}"
  fi
  if [[ "${UPPER_GUIDANCE_FEEDBACK_MODE}" != "off" ]]; then
    upper_guidance_protocol+="-feedback-${UPPER_GUIDANCE_FEEDBACK_MODE}-h${UPPER_GUIDANCE_FEEDBACK_HORIZON}-c${UPPER_GUIDANCE_FEEDBACK_CONFIRMATIONS}-sim${UPPER_GUIDANCE_FEEDBACK_SIMILARITY_TOLERANCE}-purity${UPPER_GUIDANCE_FEEDBACK_PURITY_TOLERANCE}-reinforce${UPPER_GUIDANCE_FEEDBACK_MAX_REINFORCEMENTS}"
    if [[ "${UPPER_GUIDANCE_FEEDBACK_LOWER_RESCUE_ENABLED}" == "1" ]]; then
      upper_guidance_protocol+="-lower-rescue-c${UPPER_GUIDANCE_FEEDBACK_LOWER_RESCUE_CONFIRMATIONS}-margin${UPPER_GUIDANCE_FEEDBACK_LOWER_RESCUE_MIN_MARGIN}"
    fi
  fi
  if [[ "${UPPER_GUIDANCE_LOWER_GLOBAL_MODE}" != "off" ]]; then
    upper_guidance_protocol+="-lower-global-${UPPER_GUIDANCE_LOWER_GLOBAL_MODE}-k${UPPER_GUIDANCE_LOWER_GLOBAL_TOP_K}-m${UPPER_GUIDANCE_LOWER_GLOBAL_PROBE_CANDIDATES}-p${UPPER_GUIDANCE_LOWER_GLOBAL_MIN_POSTERIOR}-s${UPPER_GUIDANCE_LOWER_GLOBAL_MIN_SCORE}-margin${UPPER_GUIDANCE_LOWER_GLOBAL_MIN_MARGIN}-ps${UPPER_GUIDANCE_LOWER_GLOBAL_PROGRESS_SCALE}"
  fi
  if [[ -n "${UPPER_STAGE_GUIDANCE_CONFIG}" ]]; then
    upper_stage_guidance_hash="$(sha256sum "${UPPER_STAGE_GUIDANCE_CONFIG}" | cut -c1-12)"
    upper_guidance_protocol+="-stage-conditioned-${upper_stage_guidance_hash}"
    if [[ "${WILL_GUIDANCE_ORACLE_COMPLETION:-0}" == "1" ]]; then
      upper_guidance_protocol+="-diagnostic-oracle-completion"
    fi
  fi
fi
protocol="PrediMem-official-async-semantics+dual-gpu+upper-dynamic-batch${UPPER_BATCH_SIZE}+lower-dynamic-batch${LOWER_BATCH_SIZE}+env${ENV_WORKERS}+probe-crossenv${LOWER_PROBE_CROSS_ENV_BATCH}+no-vlm-png-dump+${MEMORY_ALIGNMENT_TAG}+memory-tasks-${memory_scope}+upper-guidance-${upper_guidance_protocol}"
config_json="$("${OPENPI_PY}" -c \
  'import json,sys; print(json.dumps(dict(mode=sys.argv[1],tasks=sys.argv[2],episodes=int(sys.argv[3]),seed=int(sys.argv[4]),heads=sys.argv[5],upper_batch=int(sys.argv[6]),lower_batch=int(sys.argv[7]),env_workers=int(sys.argv[8]),vlm_interval=int(sys.argv[9]),replan_steps=int(sys.argv[10]),max_steps=int(sys.argv[11]),save_video=int(sys.argv[12]),vlm_ckpt=sys.argv[13],vla_ckpt=sys.argv[14],protocol=sys.argv[15],guidance_norm_cap=float(sys.argv[16]),guidance_total_norm_cap=float(sys.argv[17]),memory_meta=sys.argv[18],memory_admission=sys.argv[19],memory_top_k=int(sys.argv[20]),negative_memory_top_k=int(sys.argv[21]),record_memory_data=int(sys.argv[22]),memory_trace_level=sys.argv[23],guidance_suite_gate=sys.argv[24],memory_action_alignment=sys.argv[25],upper_guidance_enabled=int(sys.argv[26]),upper_guidance_head=sys.argv[27],upper_guidance_manifest=sys.argv[28],upper_guidance_features=sys.argv[29],upper_guidance_top_k=int(sys.argv[30]),upper_guidance_temperature=float(sys.argv[31]),upper_guidance_min_task_purity=float(sys.argv[32]),upper_guidance_min_subtask_confidence=float(sys.argv[33]),upper_guidance_interpolation=float(sys.argv[34]),upper_guidance_bank_per_primitive=int(sys.argv[35]),upper_guidance_bank_seed=int(sys.argv[36]),guidance_lambda_max=float(sys.argv[37]),prior_guidance_final_scale=float(sys.argv[38]),lower_probe_cross_env_batch=int(sys.argv[39])),sort_keys=True))' \
  "${MODE}" "${TASK_IDS}" "${EPISODES_PER_TASK}" "${SEED}" "${HEAD_VARIANTS}" \
  "${UPPER_BATCH_SIZE}" "${LOWER_BATCH_SIZE}" "${ENV_WORKERS}" "${VLM_INTERVAL}" \
  "${REPLAN_STEPS}" "${MAX_STEPS}" "${SAVE_VIDEO}" "${VLM_CKPT}" "${VLA_CKPT}" "${protocol}" \
  "${MEMORY_GUIDANCE_NORM_CAP}" "${MEMORY_GUIDANCE_TOTAL_NORM_CAP}" "${MEMORY_META_BASENAME}" \
  "${MEMORY_ADMISSION}" "${MEMORY_TOP_K}" "${NEGATIVE_MEMORY_TOP_K}" \
  "${RECORD_MEMORY_DATA}" "${MEMORY_TRACE_LEVEL}" "${MEMORY_GUIDANCE_SUITE_GATE_PATH:-}" \
  "${MEMORY_ACTION_ALIGNMENT}" "${UPPER_GUIDANCE_ENABLED}" "${UPPER_GUIDANCE_HEAD}" \
  "${UPPER_GUIDANCE_MANIFEST}" "${UPPER_GUIDANCE_FEATURES}" "${UPPER_GUIDANCE_TOP_K}" \
  "${UPPER_GUIDANCE_TEMPERATURE}" "${UPPER_GUIDANCE_MIN_TASK_PURITY}" \
  "${UPPER_GUIDANCE_MIN_SUBTASK_CONFIDENCE}" "${UPPER_GUIDANCE_INTERPOLATION}" \
  "${UPPER_GUIDANCE_BANK_PER_PRIMITIVE}" "${UPPER_GUIDANCE_BANK_SEED}" \
  "${MEMORY_GUIDANCE_LAMBDA_MAX}" "${MEMORY_PRIOR_GUIDANCE_FINAL_SCALE}" \
  "${LOWER_PROBE_CROSS_ENV_BATCH}")"
if [[ "${RESUME}" == "1" ]]; then
  [[ -f "${RUN_ROOT}/run_config.json" ]] || { echo "RESUME=1 requires ${RUN_ROOT}/run_config.json" >&2; exit 2; }
  # Batch size and environment count are throughput controls. They may be
  # increased on resume without changing the experiment or memory identity.
  # Configs created before Upper kNN guidance was introduced omitted these
  # fields. Canonicalize that historical absence to the disabled defaults;
  # an enabled request still differs and remains non-resumable.
  normalize_config='import json,re,sys; r=json.load(sys.stdin); [r.pop(k, None) for k in ("upper_batch", "lower_batch", "env_workers")]; defaults={"upper_guidance_enabled":0,"upper_guidance_head":"","upper_guidance_manifest":"","upper_guidance_features":"","upper_guidance_top_k":16,"upper_guidance_temperature":0.07,"upper_guidance_min_task_purity":0.0,"upper_guidance_min_subtask_confidence":0.0,"upper_guidance_interpolation":0.8,"upper_guidance_bank_per_primitive":16,"upper_guidance_bank_seed":17,"guidance_lambda_max":0.2,"prior_guidance_final_scale":0.01}; [r.setdefault(k,v) for k,v in defaults.items()]; p=r.get("protocol", ""); p=re.sub(r"upper-dynamic-batch[0-9]+", "upper-dynamic-batch*", p); p=re.sub(r"lower-dynamic-batch[0-9]+", "lower-dynamic-batch*", p); p=re.sub(r"\+env[0-9]+\+", "+env*+", p); p += "+upper-guidance-off" if "+upper-guidance-" not in p else ""; r["protocol"]=p; print(json.dumps(r, sort_keys=True, separators=(",", ":")))'
  stored="$("${OPENPI_PY}" -c "${normalize_config}" <"${RUN_ROOT}/run_config.json")"
  requested="$(printf '%s' "${config_json}" | "${OPENPI_PY}" -c "${normalize_config}")"
  [[ "${stored}" == "${requested}" ]] || { echo "Resume configuration mismatch." >&2; exit 2; }
elif [[ "${PREFLIGHT_ONLY}" != "1" ]]; then
  if [[ -e "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to mix results into non-empty RUN_ROOT: ${RUN_ROOT}" >&2
    exit 2
  fi
  mkdir -p "${RUN_ROOT}"
  printf '%s\n' "${config_json}" >"${RUN_ROOT}/run_config.json"
fi
if [[ -n "${PROTOCOL_LOCK_FILE}" && "${PREFLIGHT_ONLY}" != "1" ]]; then
  [[ -f "${PROTOCOL_LOCK_FILE}" ]] || { echo "Missing protocol lock: ${PROTOCOL_LOCK_FILE}" >&2; exit 2; }
  if [[ "${RESUME}" == "1" ]]; then
    cmp -s "${PROTOCOL_LOCK_FILE}" "${RUN_ROOT}/protocol_lock.json" || {
      echo "Resume protocol lock mismatch: ${RUN_ROOT}/protocol_lock.json" >&2
      exit 2
    }
  else
    cp "${PROTOCOL_LOCK_FILE}" "${RUN_ROOT}/protocol_lock.json"
  fi
fi
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "Preflight passed: mode=${MODE} heads=${HEAD_VARIANTS} eval_tasks=${TASK_IDS} memory_allowed_tasks=${MEMORY_ALLOWED_TASK_IDS:-all} seed=${SEED} episodes=${EPISODES_PER_TASK} batch=${UPPER_BATCH_SIZE}/${LOWER_BATCH_SIZE} envs=${ENV_WORKERS} resume=${RESUME}"
  exit 0
fi

LIBERO_CONFIG_DIR="${RUN_ROOT}/libero_config"
mkdir -p \
  "${RUN_ROOT}/logs" "${RUN_ROOT}/stdout" "${RUN_ROOT}/cache/numba" \
  "${RUN_ROOT}/cache/matplotlib" "${LIBERO_CONFIG_DIR}/datasets" "${OPENPI_DATA_HOME}"
cat >"${LIBERO_CONFIG_DIR}/config.yaml" <<EOF
benchmark_root: ${BENCHMARK_ROOT}/libero_fork/libero
bddl_files: ${BENCHMARK_ROOT}/libero_fork/libero/bddl_files
init_states: ${BENCHMARK_ROOT}/libero_fork/libero/init_files
datasets: ${LIBERO_CONFIG_DIR}/datasets
assets: ${BENCHMARK_ROOT}/libero_fork/libero/assets
EOF

export OPENPI_ROOT="${ROOT}"
export OPENPI_INFERENCE_ROOT="${ROOT}"
export TARGET_LIBERO_PATH="${BENCHMARK_ROOT}/libero_fork"
export OPENPI_DATA_HOME OPENPI_TORCH_COMPILE
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYOPENGL_PLATFORM=egl MUJOCO_GL=egl
export PYTHONPATH="${ROOT}/src:${BENCHMARK_ROOT}/async_vlm26_reference:${BENCHMARK_ROOT}/openpi_minimal_runtime:${BENCHMARK_ROOT}/scripts:${BENCHMARK_ROOT}/libero_fork:${ROOT}/packages/openpi-client/src:${ROOT}:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"
export NUMBA_CACHE_DIR="${RUN_ROOT}/cache/numba"
export MPLCONFIGDIR="${RUN_ROOT}/cache/matplotlib"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Keep the platform-facing terminal active during long Counting episodes. The
# upper server emits this heartbeat only after a real inference batch finishes.
exec 3>&1

upper_pid=""
lower_pid=""
transfer_pid=""
worker_pids=()
cleanup() {
  for pid in "${worker_pids[@]:-}"; do
    [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
  done
  [[ -n "${lower_pid}" ]] && kill "${lower_pid}" 2>/dev/null || true
  [[ -n "${transfer_pid}" ]] && kill "${transfer_pid}" 2>/dev/null || true
  [[ -n "${upper_pid}" ]] && kill "${upper_pid}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

upper_server_args=(
  --checkpoint "${VLM_CKPT}"
  --processor-dir "${VLM_PROCESSOR_DIR}"
  --task-config "${TASK_CONFIG}"
  --port "${UPPER_PORT}"
  --device cuda:0
  --batch-size "${UPPER_BATCH_SIZE}"
  --batch-wait-ms "${UPPER_BATCH_WAIT_MS}"
  --max-new-tokens "${MAX_NEW_TOKENS:-256}"
  --merge-distance "${D_MERGE:-6}"
  --keyframe-max "${K_MAX:-0}"
)
if [[ "${UPPER_GUIDANCE_ENABLED}" == "1" ]]; then
  upper_server_args+=(
    --upper-guidance-enabled
    --upper-guidance-head "${UPPER_GUIDANCE_HEAD}"
    --upper-guidance-manifest "${UPPER_GUIDANCE_MANIFEST}"
    --upper-guidance-features "${UPPER_GUIDANCE_FEATURES}"
    --upper-guidance-top-k "${UPPER_GUIDANCE_TOP_K}"
    --upper-guidance-temperature "${UPPER_GUIDANCE_TEMPERATURE}"
    --upper-guidance-min-task-purity "${UPPER_GUIDANCE_MIN_TASK_PURITY}"
    --upper-guidance-min-subtask-confidence "${UPPER_GUIDANCE_MIN_SUBTASK_CONFIDENCE}"
    --upper-guidance-interpolation "${UPPER_GUIDANCE_INTERPOLATION}"
    --upper-guidance-bank-per-primitive "${UPPER_GUIDANCE_BANK_PER_PRIMITIVE}"
    --upper-guidance-bank-seed "${UPPER_GUIDANCE_BANK_SEED}"
    --upper-guidance-device "${UPPER_GUIDANCE_DEVICE}"
    --upper-guidance-allowed-task-ids "${UPPER_GUIDANCE_ALLOWED_TASK_IDS}"
    --upper-guidance-feedback-mode "${UPPER_GUIDANCE_FEEDBACK_MODE}"
    --upper-guidance-feedback-horizon "${UPPER_GUIDANCE_FEEDBACK_HORIZON}"
    --upper-guidance-feedback-confirmations "${UPPER_GUIDANCE_FEEDBACK_CONFIRMATIONS}"
    --upper-guidance-feedback-similarity-tolerance "${UPPER_GUIDANCE_FEEDBACK_SIMILARITY_TOLERANCE}"
    --upper-guidance-feedback-purity-tolerance "${UPPER_GUIDANCE_FEEDBACK_PURITY_TOLERANCE}"
    --upper-guidance-feedback-max-reinforcements "${UPPER_GUIDANCE_FEEDBACK_MAX_REINFORCEMENTS}"
    --upper-guidance-lower-global-mode "${UPPER_GUIDANCE_LOWER_GLOBAL_MODE}"
    --upper-guidance-lower-global-top-k "${UPPER_GUIDANCE_LOWER_GLOBAL_TOP_K}"
    --upper-guidance-lower-global-probe-candidates "${UPPER_GUIDANCE_LOWER_GLOBAL_PROBE_CANDIDATES}"
    --upper-guidance-lower-global-min-posterior "${UPPER_GUIDANCE_LOWER_GLOBAL_MIN_POSTERIOR}"
    --upper-guidance-lower-global-min-score "${UPPER_GUIDANCE_LOWER_GLOBAL_MIN_SCORE}"
    --upper-guidance-lower-global-min-margin "${UPPER_GUIDANCE_LOWER_GLOBAL_MIN_MARGIN}"
    --upper-guidance-lower-global-progress-scale "${UPPER_GUIDANCE_LOWER_GLOBAL_PROGRESS_SCALE}"
  )
  if [[ -n "${UPPER_STAGE_GUIDANCE_CONFIG}" ]]; then
    upper_server_args+=(--upper-stage-guidance-config "${UPPER_STAGE_GUIDANCE_CONFIG}")
  fi
  if [[ "${UPPER_GUIDANCE_FEEDBACK_LOWER_RESCUE_ENABLED}" == "1" ]]; then
    upper_server_args+=(
      --upper-guidance-feedback-lower-rescue-enabled
      --upper-guidance-feedback-lower-rescue-confirmations "${UPPER_GUIDANCE_FEEDBACK_LOWER_RESCUE_CONFIRMATIONS}"
      --upper-guidance-feedback-lower-rescue-min-margin "${UPPER_GUIDANCE_FEEDBACK_LOWER_RESCUE_MIN_MARGIN}"
    )
  fi
  if [[ -n "${UPPER_GUIDANCE_LOWER_FEATURES}" ]]; then
    upper_server_args+=(
      --upper-guidance-lower-features "${UPPER_GUIDANCE_LOWER_FEATURES}"
      --upper-guidance-upper-age "${UPPER_GUIDANCE_UPPER_AGE}"
    )
  fi
  if [[ "${UPPER_GUIDANCE_HISTORY_ENABLED}" == "1" ]]; then
    upper_server_args+=(
      --upper-guidance-history-enabled
      --upper-guidance-history-lock-observations "${UPPER_GUIDANCE_HISTORY_LOCK_OBSERVATIONS}"
      --upper-guidance-history-advance-confirmations "${UPPER_GUIDANCE_HISTORY_ADVANCE_CONFIRMATIONS}"
      --upper-guidance-history-max-rollback "${UPPER_GUIDANCE_HISTORY_MAX_ROLLBACK}"
      --upper-guidance-history-max-advance "${UPPER_GUIDANCE_HISTORY_MAX_ADVANCE}"
    )
  elif [[ "${UPPER_GUIDANCE_TEMPORAL_ENABLED}" == "1" ]]; then
    upper_server_args+=(
      --upper-guidance-temporal-enabled
      --upper-guidance-temporal-posterior "${UPPER_GUIDANCE_TEMPORAL_POSTERIOR}"
      --upper-guidance-temporal-purity "${UPPER_GUIDANCE_TEMPORAL_PURITY}"
      --upper-guidance-temporal-evidence-decay "${UPPER_GUIDANCE_TEMPORAL_EVIDENCE_DECAY}"
      --upper-guidance-temporal-advance-evidence "${UPPER_GUIDANCE_TEMPORAL_ADVANCE_EVIDENCE}"
      --upper-guidance-temporal-same-stage-budget "${UPPER_GUIDANCE_TEMPORAL_SAME_STAGE_BUDGET}"
      --upper-guidance-temporal-max-rollback "${UPPER_GUIDANCE_TEMPORAL_MAX_ROLLBACK}"
      --upper-guidance-temporal-max-advance "${UPPER_GUIDANCE_TEMPORAL_MAX_ADVANCE}"
    )
  fi
fi
if [[ "${EXTERNAL_UPPER_SERVER}" == "1" ]]; then
  echo "external upper: ${UPPER_HOST}:${UPPER_PORT}" >"${RUN_ROOT}/logs/upper_server.log"
  "${OPENPI_PY}" "${ROOT}/scripts/eval/wait_for_websocket.py" \
    --host "${UPPER_HOST}" --port "${UPPER_PORT}" --timeout "${SERVER_START_TIMEOUT_SECONDS}" \
    >"${RUN_ROOT}/logs/upper_ready.log" 2>&1 || {
      cat "${RUN_ROOT}/logs/upper_ready.log" >&2
      exit 3
    }
else
  PREDIMEM_HEARTBEAT_FD=3 PREDIMEM_HEARTBEAT_EVERY_BATCHES="${HEARTBEAT_EVERY_BATCHES:-1}" \
  CUDA_VISIBLE_DEVICES="${UPPER_GPU}" "${VLM_PY}" -m optimus_eval.predimem_upper_server \
    "${upper_server_args[@]}" >"${RUN_ROOT}/logs/upper_server.log" 2>&1 &
  upper_pid=$!
  "${OPENPI_PY}" "${ROOT}/scripts/eval/wait_for_websocket.py" \
    --host "${UPPER_HOST}" --port "${UPPER_PORT}" --pid "${upper_pid}" --timeout "${SERVER_START_TIMEOUT_SECONDS}" \
    >"${RUN_ROOT}/logs/upper_ready.log" 2>&1 || {
      tail -n 200 "${RUN_ROOT}/logs/upper_server.log" >&2
      exit 3
    }
fi

for variant in ${HEAD_VARIANTS}; do
  # Re-anchor after a long Upper cold start. Some managed launchers invalidate
  # the inherited cwd handle while the process is waiting, which makes the
  # next Python interpreter fail during getpath initialization.
  cd "${ROOT}"
  variant_root="${RUN_ROOT}/${variant}"
  if [[ "${RESUME}" == "1" && -f "${variant_root}/results.txt" ]]; then
    if [[ "${RECORD_MEMORY_DATA}" == "1" && ! -f "${variant_root}/memory_records/index.jsonl" ]]; then
      "${OPENPI_PY}" -m optimus_eval.index_predimem_memory_records \
        --run-root "${variant_root}" \
        --task-ids "${TASK_IDS}" \
        --episodes-per-task "${EPISODES_PER_TASK}" \
        --trace-level "${MEMORY_TRACE_LEVEL}"
    fi
    echo "[skip] completed ${variant}: ${variant_root}"
    continue
  fi
  mkdir -p "${variant_root}"
  destination_trace_dir="${variant_root}/memory_records/traces"
  trace_dir="${destination_trace_dir}"
  local_trace_dir=""
  if [[ "${RECORD_MEMORY_DATA}:${MEMORY_TRACE_LEVEL}" == "1:bank" ]]; then
    run_trace_id="$(printf '%s' "${RUN_ROOT}" | sha256sum | cut -c1-20)"
    local_trace_dir="${TRACE_STAGING_BASE}/${run_trace_id}/${variant}/memory_records/traces"
    mkdir -p "${local_trace_dir}" "${destination_trace_dir}"
    "${OPENPI_PY}" -m optimus_eval.predimem_trace_transfer \
      --source "${local_trace_dir}" --destination "${destination_trace_dir}" \
      --workers "${TRACE_TRANSFER_WORKERS}" \
      --batch-size "${TRACE_TRANSFER_BATCH_SIZE}" \
      > >(tee -a "${variant_root}/trace_transfer.log" >&3) 2>&1 &
    transfer_pid=$!
  fi
  head_path="${TASK_HEAD_CKPT:-${AOSS_ROOT}/heads/${variant}/best.pt}"
  positive_meta="${POSITIVE_MEMORY_META_PATH:-${AOSS_ROOT}/memory/${variant}/${MEMORY_META_BASENAME}}"
  positive_index="${POSITIVE_FAISS_INDEX_PATH:-${AOSS_ROOT}/memory/${variant}/gpm_memory.index}"
  positive_actions="${POSITIVE_MEMORY_ACTIONS_PATH:-${AOSS_ROOT}/memory/shared/gpm_memory_actions.npz}"
  server=(
    "${OPENPI_PY}" "${ROOT}/scripts/serve_policy.py"
    --env LIBERO --port "${LOWER_PORT}" --seed "${SEED}"
    --inference-batch-size "${LOWER_BATCH_SIZE}"
    --inference-batch-wait-ms "${LOWER_BATCH_WAIT_MS}"
    --use-memory
    --action-norm-stats-path "${ACTION_STATS}"
    --action-use-quantile-norm
    --memory-top-k "${MEMORY_TOP_K}"
    --memory-progress-window "${MEMORY_PROGRESS_WINDOW:-0.20}"
    --memory-action-alignment "${MEMORY_ACTION_ALIGNMENT}"
    --memory-retrieval-backend "${MEMORY_RETRIEVAL_BACKEND:-auto}"
    --align-mode hybrid --mixture-mode gaussian --nfe-min 1 --nfe-max 10
  )
  probe_worker_args=()
  if [[ "${LOWER_PROBE_CROSS_ENV_BATCH}" == "1" ]]; then
    probe_worker_args+=(--lower-probe-cross-env-batch)
  fi
  if [[ "${MEMORY_ADMISSION}" == "none" ]]; then
    server=(
      "${OPENPI_PY}" "${ROOT}/scripts/serve_policy.py"
      --env LIBERO --port "${LOWER_PORT}" --seed "${SEED}"
      --inference-batch-size "${LOWER_BATCH_SIZE}"
      --inference-batch-wait-ms "${LOWER_BATCH_WAIT_MS}"
      --no-use-memory
    )
  elif [[ "${MEMORY_ADMISSION}" != "failure" ]]; then
    server+=(--use-positive-memory --task-head-ckpt "${head_path}"
      --memory-meta-path "${positive_meta}"
      --faiss-index-path "${positive_index}"
      --memory-actions-path "${positive_actions}")
  else
    server+=(--no-use-positive-memory --task-head-ckpt "${head_path}")
  fi
  if [[ "${LOWER_PROBE_CROSS_ENV_BATCH}" == "1" ]]; then
    server+=(--separate-lower-probe-batches)
  fi
  if [[ "${LOWER_FEATURE_EXPORT_ENABLED}" == "1" ]]; then
    server+=(--export-lower-retrieval-feature)
    if [[ "${MEMORY_ADMISSION}" == "none" ]]; then
      server+=(--task-head-ckpt "${LOWER_FEATURE_HEAD_CKPT}")
    elif [[ "${head_path}" != "${LOWER_FEATURE_HEAD_CKPT}" ]]; then
      echo "Action and Fusion-language paths must use the same fusion head checkpoint" >&2
      exit 2
    fi
  fi
  if [[ "${MEMORY_ADMISSION}" != "none" && "${MEMORY_ADMISSION}" != "failure" ]]; then
    server+=(
      --memory-top-k "${MEMORY_TOP_K}"
      --memory-allowed-task-ids "${MEMORY_ALLOWED_TASK_IDS}"
    )
    if [[ "${MEMORY_EXACT_TASK_GATE}" == "1" ]]; then
      server+=(--memory-exact-task-gate)
    fi
  fi
  if [[ "${MEMORY_ADMISSION}" != "none" && "${MEMORY_ADMISSION}" != "success" ]]; then
    server+=(
      --use-negative-guidance
      --negative-memory-meta-path "${NEGATIVE_MEMORY_META_PATH}"
      --negative-faiss-index-path "${NEGATIVE_FAISS_INDEX_PATH}"
      --negative-memory-actions-path "${NEGATIVE_MEMORY_ACTIONS_PATH}"
      --negative-memory-top-k "${NEGATIVE_MEMORY_TOP_K}"
      --negative-memory-min-similarity "${NEGATIVE_MEMORY_MIN_SIMILARITY}"
      --negative-memory-min-confidence "${NEGATIVE_MEMORY_MIN_CONFIDENCE}"
    )
  fi
  if [[ "${MEMORY_ADMISSION}" == "none" ]]; then
    :
  elif [[ "${MODE}" == "v0" ]]; then
    server+=(--memory-guidance-only --memory-guidance-time-version v0 --memory-guidance-num-steps 10)
  elif [[ "${MODE}" == "v1" ]]; then
    server+=(
      --memory-guidance-only
      --memory-guidance-time-version v1
      --memory-guidance-num-steps 10
      --memory-guidance-lambda-max "${MEMORY_GUIDANCE_LAMBDA_MAX}"
      --memory-prior-guidance-final-scale "${MEMORY_PRIOR_GUIDANCE_FINAL_SCALE}"
      --memory-guidance-norm-cap "${MEMORY_GUIDANCE_NORM_CAP}"
      --memory-guidance-total-norm-cap "${MEMORY_GUIDANCE_TOTAL_NORM_CAP}"
    )
  else
    server+=(
      --memory-prior-substep-guidance
      --memory-prior-guidance-version v3_prior_decay
      --memory-guidance-v2-magnitude-cap 0
      --memory-guidance-lambda-max 0.20
      --memory-prior-guidance-final-scale 0.01
    )
  fi
  if [[ -n "${MEMORY_GUIDANCE_SUITE_GATE_PATH:-}" ]]; then
    server+=(--memory-guidance-suite-gate-path "${MEMORY_GUIDANCE_SUITE_GATE_PATH}")
  fi
  if [[ -n "${GUIDANCE_ADAPTER_PATH}" ]]; then
    [[ -e "${GUIDANCE_ADAPTER_PATH}" ]] || { echo "Missing GUIDANCE_ADAPTER_PATH: ${GUIDANCE_ADAPTER_PATH}" >&2; exit 2; }
    server+=(--guidance-adapter-path "${GUIDANCE_ADAPTER_PATH}")
  fi
  if [[ "${RECORD_MEMORY_DATA}" == "1" ]]; then
    [[ -n "${local_trace_dir}" ]] && trace_dir="${local_trace_dir}"
    mkdir -p "${trace_dir}"
    server+=(
      --memory-guidance-trace-dir "${trace_dir}"
      --memory-guidance-trace-level "${MEMORY_TRACE_LEVEL}"
      --memory-guidance-trace-workers "${TRACE_LOCAL_WORKERS}"
    )
  fi
  server+=(policy:checkpoint --policy.config "${VLA_CONFIG}" --policy.dir "${VLA_CKPT}")
  PREDIMEM_NFE_STATS_PATH="${variant_root}/nfe_stats.jsonl" \
    CUDA_VISIBLE_DEVICES="${LOWER_GPU}" "${server[@]}" >"${variant_root}/lower_server.log" 2>&1 &
  lower_pid=$!
  "${OPENPI_PY}" "${ROOT}/scripts/eval/wait_for_websocket.py" \
    --port "${LOWER_PORT}" --pid "${lower_pid}" --timeout "${SERVER_START_TIMEOUT_SECONDS}" \
    >"${variant_root}/lower_ready.log" 2>&1 || {
      tail -n 200 "${variant_root}/lower_server.log" >&2
      exit 3
    }

  worker_pids=()
  video_arg=--save-video
  [[ "${SAVE_VIDEO}" == "1" ]] || video_arg=--no-save-video
  worker_log="${RUN_ROOT}/stdout/${variant}_dynamic_pool.log"
  touch "${worker_log}"
  trace_worker_args=()
  [[ -n "${local_trace_dir}" ]] && trace_worker_args+=(--trace-root "${local_trace_dir}")
  echo "[worker-pool] invocation=$(date -u +%Y-%m-%dT%H:%M:%SZ) retries=${WORKER_POOL_RETRIES}" | tee -a "${worker_log}"
  failed=1
  for ((pool_attempt=0; pool_attempt<=WORKER_POOL_RETRIES; pool_attempt++)); do
    echo "[worker-pool] attempt=$((pool_attempt + 1))/$((WORKER_POOL_RETRIES + 1))" | tee -a "${worker_log}"
    timeout --signal=TERM --kill-after=60 "${WORKER_TIMEOUT_SECONDS}" \
      "${VLM_PY}" -m optimus_eval.predimem_dual_worker \
      --benchmark-root "${BENCHMARK_ROOT}" \
      --task-config "${TASK_CONFIG}" \
      --task-ids "${TASK_IDS}" \
      --num-workers "${ENV_WORKERS}" \
      --episodes "${EPISODES_PER_TASK}" \
      --seed "${SEED}" \
      --upper-host "${UPPER_HOST}" \
      --upper-port "${UPPER_PORT}" \
      --pi-host "${LOWER_HOST}" \
      --pi-port "${LOWER_PORT}" \
      --output-root "${variant_root}" \
      --variant "${MODE}-${variant}" \
      --n-recent "${N_RECENT}" \
      --vlm-interval "${VLM_INTERVAL}" \
      --replan-steps "${REPLAN_STEPS}" \
      --num-steps-wait "${NUM_STEPS_WAIT}" \
      --max-steps "${MAX_STEPS}" \
      --post-goal-steps "${POST_GOAL_STEPS}" \
      "${probe_worker_args[@]}" \
      "${trace_worker_args[@]}" \
      "${video_arg}" >>"${worker_log}" 2>&1 &
    worker_pid=$!
    worker_pids=("${worker_pid}")
    service_failed=""
    while kill -0 "${worker_pid}" 2>/dev/null; do
      upper_state="external"
      if [[ "${EXTERNAL_UPPER_SERVER}" != "1" ]]; then
        upper_state="$(ps -o stat= -p "${upper_pid}" 2>/dev/null | tr -d ' ' || true)"
      fi
      lower_state="$(ps -o stat= -p "${lower_pid}" 2>/dev/null | tr -d ' ' || true)"
      if [[ "${EXTERNAL_UPPER_SERVER}" != "1" && ( -z "${upper_state}" || "${upper_state}" == Z* ) ]]; then
        service_failed="upper"
        break
      fi
      if [[ -z "${lower_state}" || "${lower_state}" == Z* ]]; then
        service_failed="lower"
        break
      fi
      perl -e 'select undef, undef, undef, 2' || true
    done
    if [[ -n "${service_failed}" ]]; then
      echo "[worker-pool] ${service_failed} server exited; aborting attempt so the outer supervisor can resume" | tee -a "${worker_log}"
      kill "${worker_pid}" 2>/dev/null || true
      wait "${worker_pid}" 2>/dev/null || true
      worker_pids=()
      exit 7
    fi
    if wait "${worker_pid}"; then
      worker_pids=()
      failed=0
      break
    fi
    worker_pids=()
    echo "[worker-pool] incomplete attempt=$((pool_attempt + 1)); requeueing missing episodes" | tee -a "${worker_log}"
  done
  if [[ "${failed}" != "0" ]]; then
    echo "${variant} workers failed after $((WORKER_POOL_RETRIES + 1)) pool attempts; inspect ${RUN_ROOT}/stdout" >&2
    exit 5
  fi
  if [[ -n "${local_trace_dir}" ]]; then
    "${OPENPI_PY}" -m optimus_eval.wait_predimem_trace_staging \
      --run-root "${variant_root}" --trace-root "${local_trace_dir}" \
      --producer-pid "${lower_pid}" --timeout-seconds "${TRACE_FLUSH_TIMEOUT_SECONDS}"
  fi
  kill "${lower_pid}" 2>/dev/null || true
  wait "${lower_pid}" 2>/dev/null || true
  lower_pid=""
  if [[ -n "${transfer_pid}" ]]; then
    wait "${transfer_pid}" || {
      tail -n 200 "${variant_root}/trace_transfer.log" >&2
      exit 6
    }
    transfer_pid=""
    [[ -f "${destination_trace_dir}/TRANSFER_COMPLETE.json" ]] || {
      echo "Trace transfer did not publish completion marker: ${destination_trace_dir}" >&2
      exit 6
    }
  fi

  "${OPENPI_PY}" -m optimus_eval.summarize_robomemarena "${variant_root}" \
    --task-ids "${TASK_IDS}" \
    --episodes-per-task "${EPISODES_PER_TASK}" | tee "${variant_root}/results.txt"
  if [[ "${RECORD_MEMORY_DATA}" == "1" ]]; then
    "${OPENPI_PY}" -m optimus_eval.index_predimem_memory_records \
      --run-root "${variant_root}" \
      --task-ids "${TASK_IDS}" \
      --episodes-per-task "${EPISODES_PER_TASK}" \
      --trace-level "${MEMORY_TRACE_LEVEL}"
  fi
  if [[ "${MODE}" == "v3" ]]; then
    "${OPENPI_PY}" -m optimus_eval.summarize_predimem_nfe "${variant_root}" \
      --task-ids "${TASK_IDS}" | tee "${variant_root}/nfe_results.txt"
  fi
done

echo "RUN_ROOT=${RUN_ROOT}"
