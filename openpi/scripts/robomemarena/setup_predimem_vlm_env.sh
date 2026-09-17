#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX=${ENV_PREFIX:-/path/to/user/miniforge3/envs/predimem-vlm}
MAMBA=${MAMBA:-/path/to/user/miniforge3/bin/mamba}
UV=${UV:-$(command -v uv)}
UV_INDEX_URL=${UV_INDEX_URL:-https://pypi.org/simple}
LOCAL_PROXY=${LOCAL_PROXY:-http://127.0.0.1:7897}
[[ -n "${UV}" && -x "${UV}" ]] || { echo "uv executable not found" >&2; exit 2; }
if [[ -n "${LOCAL_PROXY}" ]]; then
  export HTTP_PROXY="${LOCAL_PROXY}" HTTPS_PROXY="${LOCAL_PROXY}" ALL_PROXY="${LOCAL_PROXY}"
  export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${MAMBA}" create -y -p "${ENV_PREFIX}" python=3.10 pip
fi
"${UV}" pip install --python "${ENV_PREFIX}/bin/python" \
  --index-url "${UV_INDEX_URL}" \
  "numpy==1.26.4" "torch==2.7.1" "torchvision==0.22.1" "transformers==5.3.0" \
  "accelerate>=1.8" "pillow>=10" "imageio>=2.37" "imageio-ffmpeg>=0.6" "tqdm>=4.67" \
  "robosuite==1.4.0" "mujoco==3.9.0" "h5py>=3.11" "scipy>=1.13" \
  "opencv-python-headless>=4.10" "easydict>=1.13" "cloudpickle>=3" \
  "tyro>=0.9" "websockets>=11" "msgpack>=1.0.5" "dm-tree>=0.1.8" \
  "termcolor>=2" "bddl==1.0.1" "gym==0.25.2" "matplotlib>=3.8" \
  "hydra-core>=1.3" "future>=1.0"

ROBOSUITE_DIR="$("${ENV_PREFIX}/bin/python" -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("robosuite").origin).parent)')"
if [[ ! -f "${ROBOSUITE_DIR}/macros_private.py" ]]; then
  cp "${ROBOSUITE_DIR}/macros.py" "${ROBOSUITE_DIR}/macros_private.py"
fi
sed -i 's/^CACHE_NUMBA = True$/CACHE_NUMBA = False/' "${ROBOSUITE_DIR}/macros_private.py"

"${ENV_PREFIX}/bin/python" - <<'PY'
import torch
import transformers
from transformers import Qwen3VLForConditionalGeneration
print({"torch": torch.__version__, "transformers": transformers.__version__,
       "qwen3_vl": Qwen3VLForConditionalGeneration.__name__})
PY
