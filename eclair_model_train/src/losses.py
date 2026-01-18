# /src/losses.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class FocalLossConfig:
    gamma: float = 2.0
    alpha: Optional[torch.Tensor] = None  # per-class weights or None
    ignore_index: int = -100


class FocalLoss(torch.nn.Module):
    """
    Multi-class focal loss implemented on top of cross entropy.
    """

    def __init__(self, cfg: FocalLossConfig):
        super().__init__()
        self.gamma = float(cfg.gamma)
        self.ignore_index = int(cfg.ignore_index)
        if cfg.alpha is not None:
            self.register_buffer("alpha", cfg.alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [N, C], target: [N]
        # compute CE per element
        ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
            ignore_index=self.ignore_index,
            weight=self.alpha,
        )
        # mask ignored
        valid = target != self.ignore_index
        if valid.sum() == 0:
            return logits.sum() * 0.0

        ce_valid = ce[valid]
        pt = torch.exp(-ce_valid)  # = softmax prob of the true class
        loss = ((1.0 - pt) ** self.gamma) * ce_valid
        return loss.mean()
