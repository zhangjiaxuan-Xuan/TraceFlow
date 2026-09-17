import math
from types import SimpleNamespace

import pytest
import torch

import openpi.models_pytorch.pi0_pytorch as pi0_module
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.task_head.memory_init import MemoryInitProvider


def test_continuous_nfe_preserves_v1_rounding() -> None:
    provider = object.__new__(MemoryInitProvider)
    provider.nfe_min = 1
    provider.nfe_max = 10
    similarity = 7.0 / 15.0

    assert math.isclose(provider._nfe_continuous_from_similarity(similarity), 3.4)
    assert provider._nfe_from_similarity(similarity) == 3


def test_full_interval_schedule_preserves_v1_integer_nfe_and_time() -> None:
    steps, dt = PI0Pytorch._full_interval_schedule(3, 10)

    assert steps == 3
    assert math.isclose(float(dt.sum()), 1.0)
    torch.testing.assert_close(torch.from_numpy(dt), torch.full((3,), 1.0 / 3, dtype=torch.float64))


def test_full_interval_schedule_clamps_nfe_to_configured_maximum() -> None:
    steps, dt = PI0Pytorch._full_interval_schedule(12, 10)

    assert steps == 10
    assert math.isclose(float(dt.sum()), 1.0)
    torch.testing.assert_close(torch.from_numpy(dt), torch.full((10,), 0.1, dtype=torch.float64))


def test_v1_effective_noise_uses_similarity_alpha_and_memory_variance() -> None:
    provider = object.__new__(MemoryInitProvider)
    provider.mixture_mode = "gaussian"
    provider.noise_low = 0.2
    provider.noise_high = 1.0
    provider.sigma_min = 0.05
    blocks = torch.tensor([[[0.0]], [[2.0]]]).numpy()
    weights = torch.tensor([[[0.5]], [[0.5]]]).numpy()

    # similarity=1 gives alpha=0.2; the weighted memory variance is 1.
    actual = provider._effective_prior_noise_rms(blocks, weights, similarity=1.0)

    assert math.isclose(actual, 0.2, rel_tol=1e-6)


def test_direction_first_mix_sets_direction_before_speed_cap() -> None:
    base = torch.tensor([[[1.0, 0.0]]])
    guidance = torch.tensor([[[0.0, 1.0]]])

    mixed = PI0Pytorch._mix_guidance_direction_first(base, guidance, magnitude_cap=0.10)

    expected_direction = torch.nn.functional.normalize(base + guidance, dim=-1)
    actual_direction = torch.nn.functional.normalize(mixed, dim=-1)
    torch.testing.assert_close(actual_direction, expected_direction)
    torch.testing.assert_close(mixed.flatten(1).norm(dim=1), torch.tensor([1.1]))


def test_direction_first_mix_can_preserve_base_speed_exactly() -> None:
    base = torch.tensor([[[3.0, 4.0]]])
    guidance = torch.tensor([[[-2.0, 3.0]]])

    mixed = PI0Pytorch._mix_guidance_direction_first(base, guidance, magnitude_cap=0.0)

    torch.testing.assert_close(mixed.flatten(1).norm(dim=1), base.flatten(1).norm(dim=1))


def test_direction_first_mix_handles_exact_velocity_cancellation() -> None:
    base = torch.tensor([[[3.0, 4.0]]])

    mixed = PI0Pytorch._mix_guidance_direction_first(base, -base, magnitude_cap=0.0)

    torch.testing.assert_close(mixed, base)


def test_relative_guidance_cap_bounds_full_chunk_norm() -> None:
    base = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    guidance = torch.tensor([[[0.0, 8.0], [6.0, 0.0]]])

    capped = PI0Pytorch._cap_guidance_relative_to_base(base, guidance, 0.5)

    torch.testing.assert_close(
        capped.flatten(1).norm(dim=1),
        0.5 * base.flatten(1).norm(dim=1),
    )


def test_relative_guidance_cap_preserves_subthreshold_guidance() -> None:
    base = torch.tensor([[[3.0, 4.0]]])
    guidance = torch.tensor([[[0.1, 0.2]]])

    capped = PI0Pytorch._cap_guidance_relative_to_base(base, guidance, 0.5)

    torch.testing.assert_close(capped, guidance)


def test_prior_anchor_decay_reaches_exact_final_scale_for_adaptive_nfe() -> None:
    scales = [
        PI0Pytorch._prior_anchor_guidance_scale(step, 3, 0.20, 0.01)
        for step in range(3)
    ]

    assert scales == [0.20, 0.10500000000000001, 0.010000000000000009]
    assert math.isclose(
        PI0Pytorch._prior_anchor_guidance_scale(0, 1, 0.20, 0.01),
        0.01,
    )


def test_prior_velocity_correction_uses_sampler_sign() -> None:
    noisy_prior = torch.tensor([[[1.0, -1.0]]])
    guidance_velocity = torch.tensor([[[0.25, -0.50]]])

    corrected = PI0Pytorch._apply_prior_velocity_correction(noisy_prior, guidance_velocity)

    torch.testing.assert_close(corrected, torch.tensor([[[0.75, -0.50]]]))


def test_joint_outcome_guidance_uses_one_balanced_success_posterior() -> None:
    model = object.__new__(PI0Pytorch)
    model.__dict__.update(
        memory_guidance_lambda_max=0.2,
        memory_guidance_t_cut=0.3,
        memory_guidance_sigma=1.0,
        negative_guidance_sigma=1.0,
        memory_joint_failure_prior=0.5,
    )
    x_t = torch.zeros((1, 1, 1))
    positive_blocks = torch.tensor([[[1.0]]])
    negative_blocks = torch.tensor([[[-1.0]]])

    guidance, debug = model._joint_outcome_memory_flow_guidance(
        x_t,
        positive_blocks,
        torch.ones(1),
        negative_blocks,
        torch.ones(1),
        1.0,
        scale_override=1.0,
        return_debug=True,
    )

    # Equal evidence gives P(failure|x)=0.5. The success-posterior action score
    # is 0.5 * (1 - -1) = 1; velocity coordinates use the opposite sign.
    torch.testing.assert_close(guidance, torch.tensor([[[-1.0]]]))
    torch.testing.assert_close(debug["failure_posterior"], torch.tensor([0.5]))


def test_joint_outcome_guidance_uses_class_prior_in_shared_denominator() -> None:
    model = object.__new__(PI0Pytorch)
    model.__dict__.update(
        memory_guidance_lambda_max=0.2,
        memory_guidance_t_cut=0.3,
        memory_guidance_sigma=1.0,
        negative_guidance_sigma=1.0,
        memory_joint_failure_prior=0.2,
    )

    guidance, debug = model._joint_outcome_memory_flow_guidance(
        torch.zeros((1, 1, 1)),
        torch.tensor([[[1.0]]]),
        torch.ones(1),
        torch.tensor([[[-1.0]]]),
        torch.ones(1),
        1.0,
        scale_override=1.0,
        return_debug=True,
    )

    torch.testing.assert_close(debug["failure_posterior"], torch.tensor([0.2]))
    torch.testing.assert_close(guidance, torch.tensor([[[-0.4]]]))


def test_joint_outcome_guidance_rejects_invalid_failure_prior() -> None:
    model = object.__new__(PI0Pytorch)
    model.__dict__.update(
        memory_guidance_lambda_max=0.2,
        memory_guidance_t_cut=0.3,
        memory_guidance_sigma=1.0,
        negative_guidance_sigma=1.0,
        memory_joint_failure_prior=0.0,
    )

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        model._joint_outcome_memory_flow_guidance(
            torch.zeros((1, 1, 1)),
            torch.ones((1, 1, 1)),
            torch.ones(1),
            -torch.ones((1, 1, 1)),
            torch.ones(1),
            1.0,
        )


def _run_prior_anchor_sampler(
    monkeypatch,
    *,
    use_flow_decay: bool,
    use_joint_guidance: bool = False,
    compact_active: bool = False,
    flow_guidance_norm_cap: float | None = None,
):
    class FakeSession:
        nfe_adapt = 3
        s_global = 1.0

        def __init__(self, **_kwargs):
            pass

        def maybe_refresh(self, **_kwargs):
            return False

        def sample_chunk(self, _progress):
            return torch.tensor([[1.0, 0.0]])

        def sample_clean_chunk(self, _progress):
            return torch.tensor([[0.8, 0.0]])

        def guidance_tensors(self, _progress, _device):
            return torch.ones((1, 1, 2)), torch.ones(1), {}

    monkeypatch.setattr(pi0_module, "ActionMemorySession", FakeSession)
    class FakeNegativeProvider:
        def query_blocks(self, *_args, **_kwargs):
            return -torch.ones((1, 1, 2)), torch.ones(1), {}

    model = object.__new__(PI0Pytorch)
    model.__dict__.update(
        config=SimpleNamespace(action_horizon=1, action_dim=2),
        memory_provider=SimpleNamespace(mixture_mode="gaussian"),
        task_head=torch.nn.Identity(),
        memory_top_k=1,
        memory_refresh_every=1,
        memory_refresh_sim_threshold=0.0,
        memory_guidance_min_similarity=-1.0,
        memory_nfe_max=10,
        memory_guidance_lambda_max=0.20,
        memory_prior_guidance_final_scale=0.01,
        use_negative_guidance=use_joint_guidance,
        negative_memory_provider=FakeNegativeProvider() if use_joint_guidance else None,
        negative_memory_top_k=1,
        negative_memory_min_similarity=0.975,
        negative_memory_min_confidence=0.75,
        memory_joint_failure_prior=0.5,
        _batched_memory_states={},
        _batch_runtime=[
            {
                "environment_id": "env-0",
                "reset": True,
                "progress": 0.0,
                "seed": 7,
            }
        ],
    )
    scales = []

    def guidance(_x, _v, *_args, **kwargs):
        scale = float(kwargs["scale_override"])
        scales.append(scale)
        return torch.tensor([[[0.0, scale]]]), {}

    model._memory_flow_guidance = guidance
    model._joint_outcome_memory_flow_guidance = guidance
    model.denoise_step = lambda _state, _masks, _cache, x_t, _time: torch.tensor(
        [[[1.0, 0.0]]],
        dtype=x_t.dtype,
    )
    cache = SimpleNamespace(
        key_cache=[torch.zeros((1, 1, 1, 1))],
        value_cache=[torch.zeros((1, 1, 1, 1))],
    )
    result = model._sample_actions_batched_prior_anchor_guidance(
        device=torch.device("cpu"),
        state=torch.zeros((1, 1)),
        prefix_pad_masks=torch.ones((1, 1), dtype=torch.bool),
        past_key_values=cache,
        outputs_embeds=torch.tensor([[[1.0, 0.0]]]),
        noise=None,
        use_flow_decay=use_flow_decay,
        use_joint_guidance=use_joint_guidance,
        compact_active=compact_active,
        flow_guidance_norm_cap=flow_guidance_norm_cap,
    )
    return model, result, scales


def test_prior_anchor_only_corrects_prior_then_uses_base_flow(monkeypatch) -> None:
    model, result, scales = _run_prior_anchor_sampler(monkeypatch, use_flow_decay=False)

    assert scales == [0.20]
    torch.testing.assert_close(result, torch.tensor([[[0.0, -0.2]]]))
    assert model._batched_memory_states["env-0"]["prior_anchor_schedule"]["flow_scales"] == []


def test_prior_anchor_decay_preserves_flow_speed_and_anneals(monkeypatch) -> None:
    model, result, scales = _run_prior_anchor_sampler(monkeypatch, use_flow_decay=True)

    assert len(scales) == 4
    assert math.isclose(scales[0], 0.20)
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(scales[1:], [0.20, 0.105, 0.01], strict=True)
    )
    assert result[0, 0, 0] > 0.0
    assert result[0, 0, 1] < -0.2
    schedule = model._batched_memory_states["env-0"]["prior_anchor_schedule"]
    assert schedule["direction_first_magnitude_cap"] == 0.0
    assert math.isclose(schedule["flow_scales"][-1], 0.01)


def test_prior_anchor_v31_records_recursive_guidance_cap(monkeypatch) -> None:
    model, result, _ = _run_prior_anchor_sampler(
        monkeypatch,
        use_flow_decay=True,
        flow_guidance_norm_cap=0.5,
    )

    assert torch.isfinite(result).all()
    schedule = model._batched_memory_states["env-0"]["prior_anchor_schedule"]
    assert schedule["flow_guidance_norm_cap"] == 0.5
    assert schedule["direction_first_magnitude_cap"] == 0.0


def test_prior_anchor_compaction_preserves_nfe_and_output(monkeypatch) -> None:
    dense_model, dense_result, dense_scales = _run_prior_anchor_sampler(
        monkeypatch,
        use_flow_decay=True,
        compact_active=False,
    )
    compact_model, compact_result, compact_scales = _run_prior_anchor_sampler(
        monkeypatch,
        use_flow_decay=True,
        compact_active=True,
    )

    assert dense_model._last_adaptive_nfe_batch["logical_nfe"] == [3]
    assert compact_model._last_adaptive_nfe_batch["logical_nfe"] == [3]
    assert dense_scales == compact_scales
    torch.testing.assert_close(dense_result, compact_result)


def test_prior_anchor_joint_uses_joint_field_for_prior_and_decay(monkeypatch) -> None:
    model, result, scales = _run_prior_anchor_sampler(
        monkeypatch,
        use_flow_decay=True,
        use_joint_guidance=True,
    )

    assert len(scales) == 4
    assert torch.isfinite(result).all()
    schedule = model._batched_memory_states["env-0"]["prior_anchor_schedule"]
    assert schedule["mode"] == "prior_decay_joint"
    assert schedule["joint_failure_prior"] == 0.5


def test_v0_positive_guidance_bypasses_norm_cap() -> None:
    model = object.__new__(PI0Pytorch)
    model.__dict__.update(
        memory_guidance_lambda_max=0.2,
        memory_guidance_t_cut=0.3,
        memory_guidance_sigma=0.3,
        memory_guidance_norm_cap=0.2,
    )
    x_t = torch.zeros((1, 1, 2))
    v_base = torch.ones_like(x_t)
    blocks = torch.ones((1, 1, 2))
    weights = torch.ones(1)

    raw, _ = model._memory_flow_guidance(
        x_t, v_base, blocks, weights, 1.0, apply_norm_cap=False
    )
    capped, _ = model._memory_flow_guidance(
        x_t, v_base, blocks, weights, 1.0, apply_norm_cap=True
    )

    assert raw.flatten(1).norm(dim=1).item() > v_base.flatten(1).norm(dim=1).item()
    torch.testing.assert_close(
        capped.flatten(1).norm(dim=1),
        0.2 * v_base.flatten(1).norm(dim=1),
    )


def test_v0_negative_guidance_bypasses_norm_cap() -> None:
    model = object.__new__(PI0Pytorch)
    model.__dict__.update(
        memory_guidance_lambda_max=0.2,
        memory_guidance_t_cut=0.3,
        negative_guidance_beta=1.0,
        negative_guidance_sigma=0.3,
        negative_guidance_norm_cap=0.1,
    )
    x_t = torch.zeros((1, 1, 2))
    v_base = torch.ones_like(x_t)
    blocks = torch.ones((1, 1, 2))
    weights = torch.ones(1)

    raw, _ = model._negative_memory_flow_guidance(
        x_t, v_base, blocks, weights, 1.0, apply_norm_cap=False
    )
    capped, _ = model._negative_memory_flow_guidance(
        x_t, v_base, blocks, weights, 1.0, apply_norm_cap=True
    )

    assert raw.flatten(1).norm(dim=1).item() > v_base.flatten(1).norm(dim=1).item()
    torch.testing.assert_close(
        capped.flatten(1).norm(dim=1),
        0.1 * v_base.flatten(1).norm(dim=1),
    )


def test_v05_velocity_ratio_uses_conservative_floor_skip() -> None:
    base = torch.tensor([[[1.0, 0.0]]])

    assert PI0Pytorch._v05_extra_skip(3.2 * base, base).item() == 2
    assert PI0Pytorch._v05_extra_skip(1.99 * base, base).item() == 0
    assert PI0Pytorch._v05_extra_skip(2.0 * base, base).item() == 1


def test_v05_positive_raw_guidance_ignores_time_decay_and_cutoff() -> None:
    model = object.__new__(PI0Pytorch)
    model.__dict__.update(
        memory_guidance_lambda_max=0.2,
        memory_guidance_t_cut=0.3,
        memory_guidance_sigma=1.0,
        memory_guidance_norm_cap=0.2,
    )
    x_t = torch.zeros((1, 1, 1))
    v_base = torch.ones_like(x_t)
    blocks = torch.ones((1, 1, 1))
    weights = torch.ones(1)

    scheduled, _ = model._memory_flow_guidance(
        x_t, v_base, blocks, weights, 0.1, apply_norm_cap=False
    )
    raw, _ = model._memory_flow_guidance(
        x_t, v_base, blocks, weights, 0.1, apply_norm_cap=False, scale_override=1.0
    )

    torch.testing.assert_close(scheduled, torch.zeros_like(scheduled))
    torch.testing.assert_close(raw, -torch.ones_like(raw))


def test_v05_negative_raw_guidance_keeps_failure_beta() -> None:
    model = object.__new__(PI0Pytorch)
    model.__dict__.update(
        memory_guidance_lambda_max=0.2,
        memory_guidance_t_cut=0.3,
        negative_guidance_beta=0.1,
        negative_guidance_sigma=1.0,
        negative_guidance_norm_cap=0.1,
    )
    x_t = torch.zeros((1, 1, 1))
    v_base = torch.ones_like(x_t)
    blocks = torch.ones((1, 1, 1))
    weights = torch.ones(1)

    raw, _ = model._negative_memory_flow_guidance(
        x_t, v_base, blocks, weights, 0.1, apply_norm_cap=False, scale_override=1.0
    )

    torch.testing.assert_close(raw, torch.full_like(raw, 0.1))


def _run_v05_constant_guidance(dynamic_nfe: bool, guidance_ratio: float) -> tuple[int, torch.Tensor]:
    model = object.__new__(PI0Pytorch)
    model.__dict__.update(
        memory_guidance_v05_dynamic_nfe=dynamic_nfe,
        memory_guidance_v05_fine_ratio=1.2,
        memory_guidance_v05_fine_scale=0.2,
        memory_guidance_v05_fine_confirm_steps=2,
        memory_guidance_v05_dynamic_guidance_scale=0.5,
        debug_memory=False,
    )
    calls = []

    def denoise_step(*_args):
        calls.append(1)
        return torch.ones((1, 1, 1))

    def positive_guidance(_x, v_flow, *_args, **_kwargs):
        return guidance_ratio * v_flow, {}

    model.denoise_step = denoise_step
    model._memory_flow_guidance = positive_guidance
    state = torch.zeros((1, 1))
    x_t = torch.zeros((1, 1, 1))
    states = {"env": {"call_count": 0}}
    result = model._sample_actions_batched_memory_guidance_v05(
        state=state,
        prefix_pad_masks=torch.ones((1, 1), dtype=torch.bool),
        past_key_values=None,
        x_t=x_t,
        positive=[(torch.ones((1, 1, 1)), torch.ones(1))],
        negative=[None],
        active_ids=["env"],
        states=states,
        num_steps=10,
    )
    assert states["env"]["call_count"] == 1
    return len(calls), result


def test_v05_fixed_nfe_always_runs_ten_steps() -> None:
    calls, result = _run_v05_constant_guidance(dynamic_nfe=False, guidance_ratio=2.2)

    assert calls == 10
    assert torch.isfinite(result).all()


def test_v05_dynamic_grid_has_three_step_aggressive_path() -> None:
    assert PI0Pytorch._v05_dynamic_next_grid(10, fine_region=False) == 4
    assert PI0Pytorch._v05_dynamic_next_grid(4, fine_region=False) == 1
    assert PI0Pytorch._v05_dynamic_next_grid(1, fine_region=False) == 0


def test_v05_dynamic_nfe_uses_arc_length_budget() -> None:
    calls, result = _run_v05_constant_guidance(dynamic_nfe=True, guidance_ratio=4.0)

    assert calls == 3
    # v_actual RMS is 3.0 and progress is [0.6, 0.3, 0.1], so the normalized
    # sum RMS(v_actual) * dt is exactly one.
    torch.testing.assert_close(result, torch.full_like(result, -1.0))


def test_v05_dynamic_fine_region_walks_each_point_one_grid_step() -> None:
    calls, result = _run_v05_constant_guidance(dynamic_nfe=True, guidance_ratio=1.1)

    assert calls == 10
    torch.testing.assert_close(result, torch.full_like(result, -1.0))
