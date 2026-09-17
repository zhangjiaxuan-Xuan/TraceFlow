#!/usr/bin/env bash
set -euo pipefail

: "${OPENPI_ROOT:?OPENPI_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${LIBERO_PYTHON:?LIBERO_PYTHON is required}"
: "${PYTHONPATH_VALUE:?PYTHONPATH_VALUE is required}"

if "${LIBERO_PYTHON}" - <<'PY'
import yaml
if yaml.__version__ != "6.0.3":
    raise SystemExit(f"TraceFlow requires PyYAML 6.0.3, found {yaml.__version__}")
print(f"LIBERO runtime ready: yaml={yaml.__file__}")
PY
then
  export PYTHONPATH="${PYTHONPATH_VALUE}"
  return 0
fi

# Legacy fallback for an explicitly supplied, checksum-verified wheel. Public
# installs obtain PyYAML through requirements/libero.txt and do not need this.
LIBERO_RUNTIME_SITE="${RUN_ROOT}/runtime/python310/pyyaml_6_0_3"
PY_YAML_WHEEL="${PY_YAML_WHEEL:-${OPENPI_ROOT}/third_party/wheels/pyyaml-6.0.3-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl}"
[[ -f "${PY_YAML_WHEEL}" ]] || {
  echo "PyYAML 6.0.3 is missing from the LIBERO environment; rerun scripts/setup/create_envs.sh" >&2
  return 1
}
"${LIBERO_PYTHON}" "${OPENPI_ROOT}/scripts/eval/bootstrap_libero_runtime.py" \
  --wheel "${PY_YAML_WHEEL}" \
  --target "${LIBERO_RUNTIME_SITE}"
PYTHONPATH_VALUE="${LIBERO_RUNTIME_SITE}:${PYTHONPATH_VALUE}"
export LIBERO_RUNTIME_SITE PYTHONPATH_VALUE
export PYTHONPATH="${PYTHONPATH_VALUE}"
LIBERO_RUNTIME_SITE="${LIBERO_RUNTIME_SITE}" "${LIBERO_PYTHON}" - <<'PY'
import os
from pathlib import Path
import yaml

runtime = Path(os.environ["LIBERO_RUNTIME_SITE"]).resolve()
module = Path(yaml.__file__).resolve()
if runtime not in module.parents:
    raise SystemExit(f"PyYAML was not loaded from the private runtime: {module}")
print(f"Private LIBERO runtime ready: yaml={module}")
PY
