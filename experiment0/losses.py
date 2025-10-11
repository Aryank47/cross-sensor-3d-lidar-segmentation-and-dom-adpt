# ================================================
# FILE: losses.py
# ================================================
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Multi-class soft Dice computed per-class then averaged.
        logits: [N, C]
        target: [N] with class indices
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        # mask ignore
        mask = target != self.ignore_index
        if mask.sum() == 0:
            return logits.new_tensor(0.0)
        probs = probs[mask]
        target = target[mask]
        # one-hot
        target_1h = F.one_hot(target, num_classes=num_classes).float()

        dims = (0,)  # sum over points
        intersection = torch.sum(probs * target_1h, dim=dims)
        cardinality = torch.sum(probs + target_1h, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[List[float]] = None, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))
        else:
            self.alpha = None
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none", ignore_index=self.ignore_index,
                             weight=self.alpha)
        with torch.no_grad():
            pt = torch.exp(-ce)  # pt = softmax probability of the true class
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100, weight: Optional[List[float]] = None):
        super().__init__()
        w = torch.tensor(weight, dtype=torch.float) if weight is not None else None
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=w)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, target)


class ComboLoss(nn.Module):
    def __init__(self, ce: Optional[nn.Module], focal: Optional[nn.Module], dice: Optional[nn.Module],
                 weights=(0.0, 1.0, 1.0)):
        super().__init__()
        self.ce = ce
        self.focal = focal
        self.dice = dice
        self.w = weights

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = 0.0
        if self.ce is not None:
            total = total + self.w[0] * self.ce(logits, target)
        if self.focal is not None:
            total = total + self.w[1] * self.focal(logits, target)
        if self.dice is not None:
            total = total + self.w[2] * self.dice(logits, target)
        return total


def make_loss(name: str, num_classes: int, ignore_index: int, focal_gamma: float = 2.0,
              focal_alpha: Optional[List[float]] = None, dice_smooth: float = 1.0) -> nn.Module:
    name = name.lower()
    if name == "ce":
        return CrossEntropyLoss(ignore_index=ignore_index)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=focal_alpha, ignore_index=ignore_index)
    if name == "dice":
        return SoftDiceLoss(smooth=dice_smooth, ignore_index=ignore_index)
    if name in ("focal_dice", "dice_focal"):
        return ComboLoss(
            ce=None,
            focal=FocalLoss(gamma=focal_gamma, alpha=focal_alpha, ignore_index=ignore_index),
            dice=SoftDiceLoss(smooth=dice_smooth, ignore_index=ignore_index),
            weights=(0.0, 1.0, 1.0),
        )
    raise ValueError(f"Unknown loss: {name}")


