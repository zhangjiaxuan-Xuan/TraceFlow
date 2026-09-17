import torch

from openpi.models_pytorch.pi0_pytorch import GuidanceResidualAdapter, PI0Pytorch


def test_adapter_is_zero_initialized_and_trainable() -> None:
    adapter = GuidanceResidualAdapter(action_dim=7, hidden_dim=16)
    x = torch.randn(3, 10, 7)
    base = torch.randn_like(x)
    guidance = torch.randn_like(x)
    time = torch.rand(3)
    output = adapter(x, base, guidance, time)
    torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward()
    assert adapter.net[-1].weight.grad is not None


def test_v1_training_guidance_obeys_relative_norm_cap() -> None:
    batch, top_k, horizon, action_dim = 4, 8, 10, 7
    x = torch.randn(batch, horizon, action_dim)
    base = torch.randn_like(x)
    blocks = torch.randn(batch, top_k, horizon, action_dim)
    weights = torch.rand(batch, top_k)
    guidance = PI0Pytorch.compute_training_memory_guidance(
        x,
        base,
        blocks,
        weights,
        torch.ones(batch),
        version="v1",
        lambda_max=0.2,
        t_cut=0.3,
        sigma=0.3,
        norm_cap=0.2,
    )
    guidance_norm = guidance.flatten(1).norm(dim=1)
    base_norm = base.flatten(1).norm(dim=1)
    assert torch.all(guidance_norm <= 0.2 * base_norm + 1e-5)


def test_guidance_is_disabled_at_and_below_cutoff() -> None:
    x = torch.zeros(2, 3, 2)
    base = torch.ones_like(x)
    blocks = torch.ones(2, 1, 3, 2)
    guidance = PI0Pytorch.compute_training_memory_guidance(
        x,
        base,
        blocks,
        torch.ones(2, 1),
        torch.tensor([0.1, 0.3]),
        version="v0",
        lambda_max=0.2,
        t_cut=0.3,
        sigma=0.3,
        norm_cap=0.0,
    )
    torch.testing.assert_close(guidance, torch.zeros_like(guidance))
