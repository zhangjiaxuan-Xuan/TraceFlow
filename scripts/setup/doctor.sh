#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null 2>&1 && pwd)"
source "${ROOT}/scripts/lib/common.sh"
failures=0
check() { if "$@" >/dev/null 2>&1; then printf '[ok] %s\n' "$*"; else printf '[fail] %s\n' "$*"; failures=$((failures + 1)); fi; }

check command -v conda
check command -v uv
if tf_init_interpreters; then
  check "${OPENPI_PYTHON}" -c 'import torch, faiss, h5py, modelscope, huggingface_hub'
  check "${LIBERO_PYTHON}" -c 'import mujoco, robosuite, yaml'
  check "${PREDIMEM_PYTHON}" -c 'import torch, transformers; from transformers import Qwen3VLForConditionalGeneration'
else
  failures=$((failures + 1))
fi
check nvidia-smi
check env MUJOCO_GL=egl "${LIBERO_PYTHON}" -c 'import mujoco; m=mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type=\"plane\" size=\"1 1 .1\"/></worldbody></mujoco>"); d=mujoco.MjData(m); c=mujoco.GLContext(16,16); c.make_current(); mujoco.mj_forward(m,d); c.free()'
check test -f "${ROOT}/RoboMemArena/evaluation_benchmark/async_vlm26_reference/fullvlm_v2_26_memory_tasks.json"
check test -f "${ROOT}/third_party/LIBERO/libero/libero/__init__.py"

if [[ "${TRACEFLOW_DOCTOR_ASSETS:-0}" == "1" ]] && [[ -x "${OPENPI_PYTHON:-}" ]]; then
  check "${OPENPI_PYTHON}" -m traceflow.cli assets common.libero_head
fi
if [[ "${TRACEFLOW_DOCTOR_CHECKPOINTS:-0}" == "1" ]] && [[ -x "${OPENPI_PYTHON:-}" ]]; then
  check "${OPENPI_PYTHON}" -m traceflow.cli checkpoints pi05
fi

if ((failures)); then
  echo "Doctor found ${failures} problem(s)." >&2
  exit 1
fi
echo "TraceFlow doctor passed."
