from __future__ import annotations

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def _model() -> PI0Pytorch:
    model = object.__new__(PI0Pytorch)
    model.memory_guidance_lambda_max = 0.2
    model.memory_guidance_norm_cap = 0.2
    model.memory_guidance_suite_gate = {
        "6": {
            "suite": "Counting",
            "v0_lambda_max": 0.056,
            "v1_norm_cap": 0.057,
            "t_cut": 0.6,
        }
    }
    return model


def test_v0_gate_scales_raw_field_without_defining_a_cap() -> None:
    profile = _model()._memory_guidance_suite_profile({"task_id": 6}, "v0")
    assert profile == {"lambda_max": 0.056, "t_cut": 0.6, "norm_cap": 0.2}


def test_v1_gate_preserves_lambda_and_changes_relative_cap() -> None:
    profile = _model()._memory_guidance_suite_profile({"task_id": 6}, "v1")
    assert profile == {"lambda_max": 0.2, "t_cut": 0.6, "norm_cap": 0.057}


def test_disabled_gate_preserves_legacy_path() -> None:
    model = _model()
    model.memory_guidance_suite_gate = None
    assert model._memory_guidance_suite_profile({"task_id": 6}, "v1") is None
