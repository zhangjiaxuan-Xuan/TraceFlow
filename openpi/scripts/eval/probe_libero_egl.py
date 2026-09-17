from __future__ import annotations

import json
import os

import numpy as np
import robosuite  # noqa: F401  # Triggers robosuite's EGL environment validation.
import mujoco


def main() -> None:
    xml = """
    <mujoco>
      <visual><global offwidth="64" offheight="64"/></visual>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    with mujoco.Renderer(model, height=64, width=64) as renderer:
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        frame = renderer.render()
    if frame.shape != (64, 64, 3) or not np.isfinite(frame).all():
        raise RuntimeError(f"Invalid EGL frame: shape={frame.shape}")
    print(
        json.dumps(
            {
                "ok": True,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                "frame_shape": list(frame.shape),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
