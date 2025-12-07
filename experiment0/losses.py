from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# class SoftDiceLoss(nn.Module):
#     def __init__(
#         self, smooth: float = 1.0, ignore_index: int = -100, reduction: str = "mean"
#     ):
#         super().__init__()
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.reduction = reduction

#     def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         Multi-class soft Dice computed per-class then averaged.
#         logits: [N, C]
#         target: [N] with class indices
#         """
#         num_classes = logits.shape[1]
#         probs = F.softmax(logits, dim=1)
#         # mask ignore
#         mask = target != self.ignore_index
#         if mask.sum() == 0:
#             return logits.new_tensor(0.0)
#         probs = probs[mask]
#         target = target[mask]
#         # one-hot
#         target_1h = F.one_hot(target, num_classes=num_classes).float()

#         dims = (0,)  # sum over points
#         intersection = torch.sum(probs * target_1h, dim=dims)
#         cardinality = torch.sum(probs + target_1h, dim=dims)
#         dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
#         loss = 1.0 - dice
#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         else:
#             return loss


class SoftDiceLoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = -100,
        powerize: bool = True,
        use_tmask: bool = True,
        eps: float = 1e-12,
        reduction: str = "mean",  # << added back
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.powerize = powerize
        self.use_tmask = use_tmask
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Multi-class soft Dice computed per-class then averaged.
        logits: [N, C]
        target: [N] with class indices
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        if self.ignore_index is not None:
            mask = target != self.ignore_index
            if not mask.any():
                return logits.new_tensor(0.0)
            probs = probs[mask]
            target = target[mask]

        target_1h = F.one_hot(target, num_classes=num_classes).float()

        intersection = (probs * target_1h).sum(dim=0)  # [C]

        if self.powerize:
            union = probs.pow(2).sum(dim=0) + target_1h.sum(dim=0)
        else:
            union = probs.sum(dim=0) + target_1h.sum(dim=0)

        dice_per_class = (2.0 * intersection + self.smooth) / (
            union + self.smooth + self.eps
        )

        if self.use_tmask:
            tmask = (target_1h.sum(dim=0) > 0).float()
        else:
            tmask = torch.ones_like(union)

        dice_per_class = dice_per_class * tmask
        denom = tmask.sum().clamp_min(1.0)

        if self.reduction == "none":
            return 1.0 - (dice_per_class / denom)

        dice = dice_per_class.sum() / denom  # scalar

        if self.reduction == "mean":
            return 1.0 - dice  # scalar, normal case
        elif self.reduction == "sum":
            return (1.0 - dice) * denom  # not super meaningful, but defined
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def compute_class_balanced_alpha(
    samples_per_cls: Sequence[int],
    beta: float,
) -> torch.Tensor:
    """
    Compute class-balanced weights (alpha) from Cui et al. (CVPR 2019)
    using the 'effective number of samples' formula:

        w_c = (1 - beta) / (1 - beta^{n_c})

    Then normalized so that mean(w_c) = 1 (sum = num_classes).

    Args:
        samples_per_cls: list/seq with one entry per class (including ignore if you want).
        beta: float in [0.9, 0.9999], e.g. 0.999 or 0.9999.

    Returns:
        alpha: torch.FloatTensor of shape [num_classes]
    """
    samples = np.asarray(samples_per_cls, dtype=np.float32)

    # Avoid division by zero for classes with 0 samples (e.g., ignore label 0)
    # They don't matter anyway because you mask them out before the loss.
    samples_safe = samples.copy()
    samples_safe[samples_safe <= 0] = 1.0

    effective_num = 1.0 - np.power(beta, samples_safe)
    weights = (1.0 - beta) / effective_num

    # Normalize so average weight is 1.0 (sum = num_classes)
    weights = weights / np.sum(weights) * len(weights)

    alpha = torch.from_numpy(weights).float()
    return alpha


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Sequence[float]] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        # Store alpha as-is (list or tensor); we'll fix device/dtype in forward
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Prepare class weights on the same device as logits
        weight = None
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                weight = self.alpha.to(device=logits.device, dtype=logits.dtype)
            else:
                # e.g. list/tuple from config
                weight = torch.tensor(
                    self.alpha, device=logits.device, dtype=logits.dtype
                )

        ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
            ignore_index=self.ignore_index,
            weight=weight,
        )

        with torch.no_grad():
            pt = torch.exp(-ce)  # p_t

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
    samples_per_cls: Optional[List[int]] = None,
    cb_beta: float = 0.9999,
    focal_use_cb_alpha: bool = True,
) -> nn.Module:
    name = name.lower()

    # ---- CE ----
    if name == "ce":
        return CrossEntropyLoss(ignore_index=ignore_index)

    # ---- FOCAL (plain) ----
    if name == "focal":
        alpha_tensor = None

        if focal_alpha is not None:
            # explicit α from CLI/config
            alpha_tensor = torch.tensor(focal_alpha, dtype=torch.float32)
        elif focal_use_cb_alpha and samples_per_cls is not None:
            # class-balanced α using effective number
            alpha_tensor = compute_class_balanced_alpha(samples_per_cls, cb_beta)

        return FocalLoss(
            gamma=focal_gamma,
            alpha=alpha_tensor,
            ignore_index=ignore_index,
        )

    # ---- DICE (LiDOG-style defaults) ----
    if name == "dice":
        return SoftDiceLoss(
            smooth=dice_smooth,
            ignore_index=ignore_index,
            # powerize/use_tmask already default to True in ctor
        )

    # ---- Class-balanced CE ----
    if name in ("class_balanced", "cb"):
        if samples_per_cls is None:
            raise ValueError("samples_per_cls required for class_balanced loss")
        return ClassBalancedLoss(
            samples_per_cls=samples_per_cls,
            beta=cb_beta,
            ignore_index=ignore_index,
        )

    # ---- Lovász ----
    if name == "lovasz":
        return LovaszSoftmaxLoss(ignore_index=ignore_index)

    # ---- Focal + DICE combo ----
    if name in ("focal_dice", "dice_focal"):
        # Reuse the same α logic as plain Focal
        alpha_tensor = None
        if focal_alpha is not None:
            alpha_tensor = torch.tensor(focal_alpha, dtype=torch.float32)
        elif focal_use_cb_alpha and samples_per_cls is not None:
            alpha_tensor = compute_class_balanced_alpha(samples_per_cls, cb_beta)

        focal = FocalLoss(
            gamma=focal_gamma,
            alpha=alpha_tensor,
            ignore_index=ignore_index,
        )
        dice = SoftDiceLoss(
            smooth=dice_smooth,
            ignore_index=ignore_index,
        )
        return ComboLoss(
            ce=None,
            focal=focal,
            dice=dice,
            lovasz=None,
            weights=(0.0, 1.0, 1.0, 0.0),
        )

    # ---- DICE + Lovász combo ----
    if name == "dice_lovasz":
        dice = SoftDiceLoss(
            smooth=dice_smooth,
            ignore_index=ignore_index,
        )
        lovasz = LovaszSoftmaxLoss(ignore_index=ignore_index)
        return ComboLoss(
            ce=None,
            focal=None,
            dice=dice,
            lovasz=lovasz,
            weights=(0.0, 0.0, 1.0, 1.0),
        )

    raise ValueError(f"Unknown loss: {name}")
