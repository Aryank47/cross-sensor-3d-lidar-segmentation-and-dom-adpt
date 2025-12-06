from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(
        self, smooth: float = 1.0, ignore_index: int = -100, reduction: str = "mean"
    ):
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
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[List[float]] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))
        else:
            self.alpha = None
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
            ignore_index=self.ignore_index,
            weight=self.alpha,
        )
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


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax Loss - Direct IoU optimization
    Reference: Berman et al., CVPR 2018

    Simplified implementation for multi-class segmentation.
    For full implementation, see: https://github.com/bermanmaxim/LovaszSoftmax
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, C] unnormalized logits
            target: [N] class indices
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)  # [N, C]

        # Flatten for lovasz computation
        num_classes = logits.shape[1]

        losses = []
        for c in range(num_classes):
            if c == self.ignore_index:
                continue

            # Binary mask for class c
            fg = (target == c).float()  # Foreground

            # Ignore pixels
            if self.ignore_index >= 0:
                valid = target != self.ignore_index
                fg = fg[valid]
                class_probs = probs[valid, c]
            else:
                class_probs = probs[:, c]

            if fg.sum() == 0:
                continue

            # Lovász hinge for this class (errors sorted by confidence)
            errors = (fg - class_probs).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]

            # Compute Lovász extension
            inter = fg_sorted.cumsum(0)
            union = fg_sorted.sum() + (1.0 - fg_sorted).cumsum(0)
            iou = inter / union

            # Gradient of IoU: add consecutive differences
            grad = torch.cat([iou[:1], iou[1:] - iou[:-1]])
            loss = torch.sum(grad * errors_sorted)
            losses.append(loss)

        return sum(losses) / len(losses) if losses else logits.new_tensor(0.0)


class ComboLoss(nn.Module):
    """Updated to support CE, Focal, DICE, and Lovász combinations"""

    def __init__(
        self,
        ce: Optional[nn.Module] = None,
        focal: Optional[nn.Module] = None,
        dice: Optional[nn.Module] = None,
        lovasz: Optional[nn.Module] = None,
        weights=(0.0, 1.0, 1.0, 0.0),
    ):
        super().__init__()
        self.ce = ce
        self.focal = focal
        self.dice = dice
        self.lovasz = lovasz
        self.w = weights  # (w_ce, w_focal, w_dice, w_lovasz)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = 0.0
        if self.ce is not None and self.w[0] > 0:
            total = total + self.w[0] * self.ce(logits, target)
        if self.focal is not None and self.w[1] > 0:
            total = total + self.w[1] * self.focal(logits, target)
        if self.dice is not None and self.w[2] > 0:
            total = total + self.w[2] * self.dice(logits, target)
        if self.lovasz is not None and self.w[3] > 0:
            total = total + self.w[3] * self.lovasz(logits, target)
        return total


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss (Cui et al., CVPR 2019)"""

    def __init__(
        self, samples_per_cls: List[int], beta: float = 0.9999, ignore_index: int = -100
    ):
        super().__init__()

        counts = np.asarray(samples_per_cls, dtype=np.float64)

        # Cui et al. effective number
        effective_num = 1.0 - np.power(beta, counts)

        # Avoid division by zero when count == 0
        eps = 1e-8
        effective_num = np.where(effective_num > eps, effective_num, eps)

        weights = (1.0 - beta) / effective_num

        # Normalise to sum to num_classes
        weights = weights / weights.sum() * len(weights)

        self.register_buffer("weight", torch.tensor(weights, dtype=torch.float32))
        self.ignore_index = ignore_index 

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure weight lives on the same device as logits/targets
        weight = self.weight.to(logits.device)
        return F.cross_entropy(
            logits, target, weight=weight, ignore_index=self.ignore_index
        )


def make_loss(
    name: str,
    num_classes: int,
    ignore_index: int,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[List[float]] = None,
    dice_smooth: float = 1.0,
    samples_per_cls: Optional[List[int]] = None,  # NEW
    cb_beta: float = 0.9999,  # NEW
) -> nn.Module:
    name = name.lower()
    if name == "ce":
        return CrossEntropyLoss(ignore_index=ignore_index)
    if name == "focal":
        return FocalLoss(
            gamma=focal_gamma, alpha=focal_alpha, ignore_index=ignore_index
        )
    if name == "dice":
        return SoftDiceLoss(smooth=dice_smooth, ignore_index=ignore_index)

    # NEW: Class-Balanced Loss
    if name == "class_balanced" or name == "cb":
        if samples_per_cls is None:
            raise ValueError("samples_per_cls required for class_balanced loss")
        return ClassBalancedLoss(
            samples_per_cls=samples_per_cls, beta=cb_beta, ignore_index=ignore_index
        )

    # NEW: Lovász Loss
    if name == "lovasz":
        return LovaszSoftmaxLoss(ignore_index=ignore_index)

    if name in ("focal_dice", "dice_focal"):
        return ComboLoss(
            ce=None,
            focal=FocalLoss(
                gamma=focal_gamma, alpha=focal_alpha, ignore_index=ignore_index
            ),
            dice=SoftDiceLoss(smooth=dice_smooth, ignore_index=ignore_index),
            lovasz=None,
            weights=(0.0, 1.0, 1.0, 0.0),  # Updated
        )

    # NEW: DICE + Lovász combination
    if name == "dice_lovasz":
        return ComboLoss(
            ce=None,
            focal=None,
            dice=SoftDiceLoss(smooth=dice_smooth, ignore_index=ignore_index),
            lovasz=LovaszSoftmaxLoss(ignore_index=ignore_index),
            weights=(0.0, 0.0, 1.0, 1.0),
        )

    raise ValueError(f"Unknown loss: {name}")
