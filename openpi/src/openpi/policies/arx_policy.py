"""ARX AC One bimanual policy transforms for TraceFlow real-world data."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class ArxInputs(transforms.DataTransformFn):
    """Decode native ARX state and bimanual EEF-delta action tensors."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != 14:
            raise ValueError(f"Expected ARX state [..., 14], got {state.shape}")
        input_images = data["images"]
        if set(input_images) != set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected exactly {self.EXPECTED_CAMERAS}, got {tuple(input_images)}")

        def decode(image: np.ndarray) -> np.ndarray:
            value = np.asarray(image)
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(value * 255.0, 0, 255).astype(np.uint8)
            if value.ndim == 3 and value.shape[0] == 3:
                value = einops.rearrange(value, "c h w -> h w c")
            if value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError(f"Expected RGB image, got {value.shape}")
            return value

        result = {
            "image": {
                "base_0_rgb": decode(input_images["cam_high"]),
                "left_wrist_0_rgb": decode(input_images["cam_left_wrist"]),
                "right_wrist_0_rgb": decode(input_images["cam_right_wrist"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": state,
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 14:
                raise ValueError(f"Expected ARX EEF actions [..., 14], got {actions.shape}")
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class ArxOutputs(transforms.DataTransformFn):
    """Return native right-first ARX bimanual EEF deltas."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14], dtype=np.float32)}
