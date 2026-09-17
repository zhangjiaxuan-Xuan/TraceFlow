from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from openpi.task_head.task_head_mlp import TaskHeadMLP


VALID_HEAD_VARIANTS = ("lower", "upper", "fusion")


class DualTowerRetrievalHead(nn.Module):
    """Project lower, upper, or causally fused VLM features into one retrieval space."""

    def __init__(
        self,
        *,
        variant: str,
        lower_dim: int = 2048,
        upper_dim: int = 4096,
        hidden: int = 1024,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        variant = str(variant).lower()
        if variant not in VALID_HEAD_VARIANTS:
            raise ValueError(f"Unknown dual-tower head variant: {variant}")
        self.variant = variant
        self.lower_dim = int(lower_dim)
        self.upper_dim = int(upper_dim)
        self.hidden = int(hidden)
        self.out_dim = int(out_dim)

        if variant == "lower":
            self.projector = TaskHeadMLP(self.lower_dim, self.hidden, self.out_dim)
        elif variant == "upper":
            self.projector = TaskHeadMLP(self.upper_dim, self.hidden, self.out_dim)
        else:
            branch_dim = max(128, self.hidden // 2)
            self.lower_projector = nn.Sequential(
                nn.Linear(self.lower_dim, branch_dim),
                nn.GELU(),
                nn.LayerNorm(branch_dim),
            )
            self.upper_projector = nn.Sequential(
                nn.Linear(self.upper_dim, branch_dim),
                nn.GELU(),
                nn.LayerNorm(branch_dim),
            )
            # age and availability are explicit so stale asynchronous upper features
            # cannot silently look identical to fresh features.
            self.fusion = nn.Sequential(
                nn.Linear(2 * branch_dim + 2, self.hidden),
                nn.GELU(),
                nn.LayerNorm(self.hidden),
                nn.Linear(self.hidden, self.out_dim),
            )

    def forward(
        self,
        lower_features: torch.Tensor | None,
        upper_features: torch.Tensor | None = None,
        upper_age: torch.Tensor | None = None,
        upper_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.variant == "lower":
            if lower_features is None:
                raise ValueError("lower retrieval head requires lower_features")
            return F.normalize(self.projector(lower_features.float()), dim=-1)
        if self.variant == "upper":
            if upper_features is None:
                raise ValueError("upper retrieval head requires upper_features")
            return F.normalize(self.projector(upper_features.float()), dim=-1)

        if lower_features is None or upper_features is None:
            raise ValueError("fusion retrieval head requires lower_features and upper_features")
        batch = lower_features.shape[0]
        if upper_age is None:
            upper_age = torch.zeros(batch, device=lower_features.device, dtype=torch.float32)
        if upper_available is None:
            upper_available = torch.ones(batch, device=lower_features.device, dtype=torch.float32)
        age = torch.clamp(upper_age.float().reshape(batch, 1), min=0.0, max=1.0)
        available = upper_available.float().reshape(batch, 1)
        upper = upper_features.float() * available
        fused = torch.cat(
            (
                self.lower_projector(lower_features.float()),
                self.upper_projector(upper),
                age,
                available,
            ),
            dim=-1,
        )
        return F.normalize(self.fusion(fused), dim=-1)


def checkpoint_payload(
    head: DualTowerRetrievalHead,
    state_dict: dict[str, torch.Tensor],
    **extra,
) -> dict:
    return {
        "head_type": "dual_tower",
        "variant": head.variant,
        "lower_dim": head.lower_dim,
        "upper_dim": head.upper_dim,
        "hidden": head.hidden,
        "out_dim": head.out_dim,
        "state_dict": state_dict,
        **extra,
    }
