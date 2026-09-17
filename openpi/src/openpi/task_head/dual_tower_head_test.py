from __future__ import annotations

import pytest
import torch

from openpi.task_head.dual_tower_head import DualTowerRetrievalHead


@pytest.mark.parametrize("variant", ["lower", "upper", "fusion"])
def test_dual_tower_head_outputs_normalized_embeddings(variant: str) -> None:
    head = DualTowerRetrievalHead(variant=variant, lower_dim=4, upper_dim=6, hidden=8, out_dim=3)
    lower = torch.randn(5, 4)
    upper = torch.randn(5, 6)
    result = head(lower, upper, torch.linspace(0, 1, 5), torch.ones(5))
    assert result.shape == (5, 3)
    torch.testing.assert_close(torch.linalg.vector_norm(result, dim=-1), torch.ones(5))


def test_fusion_masks_unavailable_upper_feature() -> None:
    head = DualTowerRetrievalHead(variant="fusion", lower_dim=4, upper_dim=6, hidden=8, out_dim=3)
    lower = torch.randn(2, 4)
    upper_a = torch.randn(2, 6)
    upper_b = torch.randn(2, 6)
    unavailable = torch.zeros(2)
    first = head(lower, upper_a, torch.ones(2), unavailable)
    second = head(lower, upper_b, torch.ones(2), unavailable)
    torch.testing.assert_close(first, second)


def test_upper_head_rejects_missing_feature() -> None:
    head = DualTowerRetrievalHead(variant="upper", lower_dim=4, upper_dim=6, hidden=8, out_dim=3)
    with pytest.raises(ValueError, match="upper_features"):
        head(torch.randn(1, 4), None)
