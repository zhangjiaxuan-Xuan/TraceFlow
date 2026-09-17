from pathlib import Path

import numpy as np
import pytest
import torch

from openpi.training.guidance_cache import GuidanceTrainingCache


def _write_cache(path: Path) -> None:
    np.savez(
        path,
        episode_index=np.asarray([1, 1, 3], dtype=np.int64),
        frame_index=np.asarray([2, 5, 0], dtype=np.int64),
        blocks=np.arange(3 * 2 * 4 * 7, dtype=np.float16).reshape(3, 2, 4, 7),
        weights=np.asarray([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]], dtype=np.float32),
    )


def test_lookup_preserves_requested_batch_order(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    _write_cache(path)
    cache = GuidanceTrainingCache(path)
    blocks, weights = cache.lookup(
        {
            "episode_index": torch.tensor([3, 1]),
            "frame_index": torch.tensor([0, 2]),
        },
        device=torch.device("cpu"),
    )
    assert blocks.shape == (2, 2, 4, 7)
    torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5], [0.7, 0.3]]))


def test_lookup_uses_nearest_anchor_in_same_episode(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    _write_cache(path)
    cache = GuidanceTrainingCache(path)
    _, weights = cache.lookup(
        {"episode_index": torch.tensor([1]), "frame_index": torch.tensor([4])},
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(weights, torch.tensor([[0.4, 0.6]]))


def test_lookup_rejects_missing_episode(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    _write_cache(path)
    cache = GuidanceTrainingCache(path)
    with pytest.raises(KeyError, match="no anchor"):
        cache.lookup(
            {"episode_index": torch.tensor([2]), "frame_index": torch.tensor([4])},
            device=torch.device("cpu"),
        )
