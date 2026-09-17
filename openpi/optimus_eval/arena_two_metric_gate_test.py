from __future__ import annotations

import numpy as np
import torch

from optimus_eval.arena_full_length_alignment import _normalized_curve
from optimus_eval.fit_predimem_two_metric_gate import _weighted_logistic_fit


def test_normalized_curve_uses_episode_endpoint() -> None:
    calls = [
        {"policy_call_idx": 10, "alignment": 0.0},
        {"policy_call_idx": 20, "alignment": 0.5},
        {"policy_call_idx": 30, "alignment": 1.0},
    ]
    curve = _normalized_curve(calls, "alignment", np.asarray([0.0, 0.25, 0.5, 1.0]))
    np.testing.assert_allclose(curve, [0.0, 0.25, 0.5, 1.0])


def test_weighted_logistic_fit_is_monotone() -> None:
    fit = _weighted_logistic_fit(
        np.asarray([0.9, 1.1, 1.5, 1.6]),
        np.asarray([0.0, 0.0, 1.0, 1.0]),
        ["Counting", "Occlusion", "Transferring", "Sequence"],
        l2=0.05,
        device=torch.device("cpu"),
    )
    assert fit["slope"] > 0
    assert np.all(np.diff(fit["probability"]) > 0)
    assert 1.1 < fit["threshold"] < 1.5
