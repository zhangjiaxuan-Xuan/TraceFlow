from __future__ import annotations

import numpy as np

from openpi.policies.arx_policy import ArxInputs
from openpi.policies.arx_policy import ArxOutputs


def test_arx_inputs_preserve_right_first_state_and_eef_action() -> None:
    state = np.arange(14, dtype=np.float32)
    actions = np.arange(70, dtype=np.float32).reshape(5, 14)
    image = np.zeros((3, 32, 48), dtype=np.uint8)
    output = ArxInputs()(
        {
            "state": state,
            "actions": actions,
            "images": {
                "cam_high": image,
                "cam_left_wrist": image,
                "cam_right_wrist": image,
            },
            "prompt": "test",
        }
    )
    np.testing.assert_array_equal(output["state"], state)
    np.testing.assert_array_equal(output["actions"], actions)
    assert output["image"]["base_0_rgb"].shape == (32, 48, 3)
    np.testing.assert_array_equal(ArxOutputs()({"actions": actions})["actions"], actions)
