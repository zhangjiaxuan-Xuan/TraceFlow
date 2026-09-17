from __future__ import annotations

from types import SimpleNamespace

import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def _runtime(*records):
    return [
        {"environment_id": environment_id, "reset": reset}
        for environment_id, reset in records
    ]


def test_temporal_features_are_isolated_by_environment_and_reset() -> None:
    model = SimpleNamespace(
        task_head=SimpleNamespace(temporal_window=3, lower_dim=6),
        _batch_runtime=_runtime(("a", True), ("b", True)),
        _memory_inference_serial=1,
    )
    first = PI0Pytorch._temporal_memory_lower_features(
        model,
        torch.tensor([[1.0, 2.0], [10.0, 20.0]]),
    )
    torch.testing.assert_close(first[0], torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 2.0]))
    torch.testing.assert_close(first[1], torch.tensor([10.0, 20.0, 10.0, 20.0, 10.0, 20.0]))

    model._batch_runtime = _runtime(("b", False), ("a", False))
    model._memory_inference_serial = 2
    second = PI0Pytorch._temporal_memory_lower_features(
        model,
        torch.tensor([[30.0, 40.0], [3.0, 4.0]]),
    )
    torch.testing.assert_close(second[0], torch.tensor([10.0, 20.0, 10.0, 20.0, 30.0, 40.0]))
    torch.testing.assert_close(second[1], torch.tensor([1.0, 2.0, 1.0, 2.0, 3.0, 4.0]))

    model._batch_runtime = _runtime(("a", True))
    model._memory_inference_serial = 3
    reset = PI0Pytorch._temporal_memory_lower_features(
        model,
        torch.tensor([[5.0, 6.0]]),
    )
    torch.testing.assert_close(reset[0], torch.tensor([5.0, 6.0, 5.0, 6.0, 5.0, 6.0]))
