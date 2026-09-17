#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import tempfile


ORIGINAL = '''        assert MUJOCO_EGL_DEVICE_ID.isdigit() and (
            MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES
        ), "MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id specified in CUDA_VISIBLE_DEVICES"'''

PATCHED = '''        _cvd_list = [x.strip() for x in CUDA_VISIBLE_DEVICES.split(",")]
        assert MUJOCO_EGL_DEVICE_ID.isdigit() and (
            int(MUJOCO_EGL_DEVICE_ID) == 0 or MUJOCO_EGL_DEVICE_ID in _cvd_list
        ), "MUJOCO_EGL_DEVICE_ID must be 0 or one of the device ids in CUDA_VISIBLE_DEVICES"'''


def binding_utils_path() -> Path:
    distribution = importlib.metadata.distribution("robosuite")
    return Path(distribution.locate_file("robosuite/utils/binding_utils.py")).resolve()


def patch_binding_utils(*, check: bool) -> Path:
    path = binding_utils_path()
    source = path.read_text(encoding="utf-8")
    if PATCHED in source:
        return path
    if ORIGINAL not in source:
        raise RuntimeError(
            f"Unsupported robosuite binding_utils.py at {path}; refusing an unverified patch"
        )
    if check:
        raise RuntimeError(f"robosuite EGL logical-device patch is required: {path}")

    updated = source.replace(ORIGINAL, PATCHED, 1)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(updated)
    temporary.chmod(path.stat().st_mode)
    temporary.replace(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the validated Mem-project robosuite EGL logical-device fix"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = patch_binding_utils(check=args.check)
    print(f"robosuite EGL logical-device patch verified: {path}")


if __name__ == "__main__":
    main()
