from typing import List, Optional, Sequence, Union

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
        reduction: str = "none",
        label_smoothing_eps: Optional[float] = 0.25,
        apply_label_smoothing: bool = False,
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.powerize = powerize
        self.use_tmask = use_tmask
        self.eps = eps
        self.reduction = reduction
        self.label_smoothing_eps = label_smoothing_eps
        self.apply_label_smoothing = apply_label_smoothing

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

        # LiDOG-style label smoothing
        if self.label_smoothing_eps is not None and self.apply_label_smoothing:
            max_val = 1.0 - self.label_smoothing_eps
            min_val = self.label_smoothing_eps / (num_classes - 1)
            target_soft = torch.empty_like(target_1h)
            target_soft[target_1h == 1] = max_val
            target_soft[target_1h == 0] = min_val
        else:
            target_soft = target_1h

        intersection = (probs * target_soft).sum(dim=0)  # [C]

        if self.powerize:
            union = probs.pow(2).sum(dim=0) + target_soft.sum(dim=0)
        else:
            union = probs.sum(dim=0) + target_soft.sum(dim=0)

        dice_per_class = (2.0 * intersection + self.smooth) / (
            union + self.smooth + self.eps
        )

        if self.use_tmask:
            tmask = (target_soft.sum(dim=0) > 0).float()
        else:
            tmask = torch.ones_like(union)

        dice_per_class = dice_per_class * tmask
        denom = tmask.sum().clamp_min(1.0)

        if self.reduction == "none":
            return 1.0 - (dice_per_class / denom)

        dice = dice_per_class.sum() / denom  # scalar

        if self.reduction == "mean":
            return 1.0 - dice  # scalar, normal case
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class HardDICELoss(nn.Module):
    """
    LiDOG-style DICELoss with hard one-hot labels.

    This is conceptually equivalent to LiDOG's DICELoss:
      - softmax over classes
      - per-class Dice computed as:
            Dice_c = 2 * sum(p_c * g_c) / (sum(p_c^2) + sum(g_c))   (if powerize=True)
                   or 2 * sum(p_c * g_c) / (sum(p_c)   + sum(g_c))   (if powerize=False)
      - averaged over classes with at least one target pixel (tmask)

    Supports:
      - logits: [N, C] or [B, C, ...]
      - target: [N] or [B, ...]
      - ignore_index: label to ignore
    """

    def __init__(
        self,
        ignore_index: int = -100,
        powerize: bool = True,
        use_tmask: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.powerize = powerize
        self.use_tmask = use_tmask
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() < 2:
            raise ValueError("logits must have shape [N, C, ...]")

        device = logits.device
        n, c = logits.shape[0], logits.shape[1]

        # Flatten [N, C, ...] -> [P, C], [P]
        if logits.dim() > 2:
            logits = logits.view(n, c, -1).permute(0, 2, 1).reshape(-1, c)
            target = target.view(-1)
        else:
            logits = logits.view(-1, c)
            target = target.view(-1)

        # Mask ignore_index
        if self.ignore_index is not None:
            valid = target != self.ignore_index
            if not valid.any():
                return logits.new_tensor(0.0)
            logits = logits[valid]
            target = target[valid]

        # One-hot targets and softmax predictions
        probs = F.softmax(logits, dim=-1)  # [P_valid, C]
        target_1h = F.one_hot(target, num_classes=c).float()  # [P_valid, C]

        # Intersection per class
        intersection = (probs * target_1h).sum(dim=0)  # [C]

        # Union term (LiDOG-style)
        if self.powerize:
            union = probs.pow(2).sum(dim=0) + target_1h.sum(dim=0) + self.eps
        else:
            union = probs.sum(dim=0) + target_1h.sum(dim=0) + self.eps

        # Optionally ignore classes with no ground-truth pixels
        if self.use_tmask:
            tmask = (target_1h.sum(dim=0) > 0).float()  # [C]
        else:
            tmask = torch.ones_like(union)

        dice_per_class = 2.0 * intersection / union  # [C]
        # Mean over present classes
        denom = tmask.sum().clamp_min(1.0)
        dice_mean = (dice_per_class * tmask).sum() / denom

        loss = 1.0 - dice_mean
        return loss.to(device)


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


# class FocalLoss(nn.Module):
#     def __init__(
#         self,
#         gamma: float = 2.0,
#         alpha: Optional[Sequence[float]] = None,
#         ignore_index: int = -100,
#     ):
#         super().__init__()
#         self.gamma = gamma
#         # Store alpha as-is (list or tensor); we'll fix device/dtype in forward
#         self.alpha = alpha
#         self.ignore_index = ignore_index

#     def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         # Prepare class weights on the same device as logits
#         weight = None
#         if self.alpha is not None:
#             if isinstance(self.alpha, torch.Tensor):
#                 weight = self.alpha.to(device=logits.device, dtype=logits.dtype)
#             else:
#                 # e.g. list/tuple from config
#                 weight = torch.tensor(
#                     self.alpha, device=logits.device, dtype=logits.dtype
#                 )

#         ce = F.cross_entropy(
#             logits,
#             target,
#             reduction="none",
#             ignore_index=self.ignore_index,
#             weight=weight,
#         )

#         with torch.no_grad():
#             pt = torch.exp(-ce)  # p_t

#         loss = ((1 - pt) ** self.gamma) * ce
#         return loss.mean()


class FocalLoss(nn.Module):
    """
    Multi-class focal loss for semantic segmentation.

    Implements the standard formulation from:
      Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017).

    For each valid element:
        FL = - alpha_y * (1 - p_t)^gamma * log(p_t)

    where:
        p_t = softmax(logits)[y]

    Args:
        gamma: focusing parameter (>= 0). Typical value: 2.0
        alpha: None (no weighting) or 1D tensor / list of shape [num_classes]
               giving per-class alpha_y.
        ignore_index: label id to ignore (no loss / gradient).
        reduction: "mean", "sum", or "none"
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[Sequence[float], torch.Tensor]] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")

        self.gamma = gamma
        self.ignore_index = ignore_index
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

        # Store alpha as a buffer if provided (so it moves with .to(device))
        if alpha is not None:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha_tensor.dim() != 1:
                raise ValueError(
                    "alpha must be a 1D tensor / list of shape [num_classes]"
                )
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.alpha = None  # type: ignore[attr-defined]

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape [N, C, ...] or [N, C]
            target: Tensor of shape [N, ...] or [N]
                    with integer class indices 0..C-1 (or ignore_index).
        Returns:
            Scalar loss (if reduction != 'none') or loss per element.
        """
        if logits.dim() < 2:
            raise ValueError("logits must have shape [N, C, ...]")

        n, c = logits.shape[0], logits.shape[1]

        # Flatten spatial dims: [N, C, d1, ..., dk] -> [M, C]
        if logits.dim() > 2:
            logits = logits.view(n, c, -1)
            logits = logits.permute(0, 2, 1)  # [N, M, C]
            logits = logits.reshape(-1, c)  # [M, C]
            target = target.view(-1)  # [M]
        else:
            logits = logits.reshape(-1, c)
            target = target.view(-1)

        # Mask out ignore_index
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            if valid_mask.sum() == 0:
                # No valid pixels/points in this batch
                return logits.new_tensor(0.0)
            logits = logits[valid_mask]
            target = target[valid_mask]

        # Compute log softmax and gather log p_t for the true class
        log_probs = F.log_softmax(logits, dim=1)  # [M_valid, C]
        # log_p_t shape: [M_valid]
        log_p_t = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()  # p_t in (0, 1]

        # Alpha weighting per class (if provided)
        if self.alpha is not None:
            # self.alpha: [C], target: [M_valid] -> alpha_t: [M_valid]
            alpha = self.alpha.to(logits.device)
            if alpha.numel() != c:
                raise ValueError(
                    f"alpha has {alpha.numel()} elements, "
                    f"but logits has {c} classes"
                )
            alpha_t = alpha[target]
        else:
            alpha_t = 1.0

        # Focal loss
        # FL = - alpha_t * (1 - p_t)^gamma * log(p_t)
        focal_term = (1.0 - p_t) ** self.gamma
        loss = -alpha_t * focal_term * log_p_t  # [M_valid]

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            # "none": return per-element losses in the original flattened space
            return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100, weight: Optional[List[float]] = None):
        super().__init__()
        w = torch.tensor(weight, dtype=torch.float) if weight is not None else None
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=w)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, target)


# class LovaszSoftmaxLoss(nn.Module):
#     """
#     Lovász-Softmax Loss - Direct IoU optimization
#     Reference: Berman et al., CVPR 2018

#     Simplified implementation for multi-class segmentation.
#     For full implementation, see: https://github.com/bermanmaxim/LovaszSoftmax
#     """

#     def __init__(self, ignore_index: int = -100):
#         super().__init__()
#         self.ignore_index = ignore_index

#     def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             logits: [N, C] unnormalized logits
#             target: [N] class indices
#         """
#         # Get probabilities
#         probs = F.softmax(logits, dim=1)  # [N, C]

#         # Flatten for lovasz computation
#         num_classes = logits.shape[1]

#         losses = []
#         for c in range(num_classes):
#             if c == self.ignore_index:
#                 continue

#             # Binary mask for class c
#             fg = (target == c).float()  # Foreground

#             # Ignore pixels
#             if self.ignore_index >= 0:
#                 valid = target != self.ignore_index
#                 fg = fg[valid]
#                 class_probs = probs[valid, c]
#             else:
#                 class_probs = probs[:, c]

#             if fg.sum() == 0:
#                 continue

#             # Lovász hinge for this class (errors sorted by confidence)
#             errors = (fg - class_probs).abs()
#             errors_sorted, perm = torch.sort(errors, descending=True)
#             fg_sorted = fg[perm]

#             # Compute Lovász extension
#             inter = fg_sorted.cumsum(0)
#             union = fg_sorted.sum() + (1.0 - fg_sorted).cumsum(0)
#             iou = inter / union

#             # Gradient of IoU: add consecutive differences
#             grad = torch.cat([iou[:1], iou[1:] - iou[:-1]])
#             loss = torch.sum(grad * errors_sorted)
#             losses.append(loss)

#         return sum(losses) / len(losses) if losses else logits.new_tensor(0.0)


class LovaszSoftmaxLoss(nn.Module):
    """
    Multi-class Lovasz-Softmax loss (Berman et al., CVPR 2018).

    This implementation is equivalent to the reference code's
    `lovasz_softmax` / `lovasz_softmax_flat` with:
      - classes='present'
      - per_image=False

    It supports:
      - logits of shape [N, C]  (e.g., point-wise)
      - logits of shape [B, C, H, W]  (e.g., image/voxel grid)

    `ignore_index` is the label that should be excluded from the loss.
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:
                - [N, C]   OR
                - [B, C, H, W]
            target:
                - [N]      OR
                - [B, H, W]
        """
        if logits.ndim == 2:
            # [N, C] case (e.g., Minkowski point cloud)
            probs = F.softmax(logits, dim=1)  # [N, C]
            probas_flat = probs
            labels_flat = target.view(-1)
        else:
            # [B, C, H, W] or [B, C, D, H, W] generalization
            probs = F.softmax(logits, dim=1)
            C = probs.size(1)

            # Move channels to last dim and flatten everything except C
            permute_dims = (0, *range(2, probs.ndim), 1)
            probas_flat = probs.permute(*permute_dims).contiguous().view(-1, C)
            labels_flat = target.view(-1)

        if self.ignore_index >= 0:
            valid = labels_flat != self.ignore_index
            probas_flat = probas_flat[valid]
            labels_flat = labels_flat[valid]

        return self._lovasz_softmax_flat(probas_flat, labels_flat)

    @staticmethod
    def _lovasz_softmax_flat(
        probas: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Equivalent to the original `lovasz_softmax_flat` with classes='present'.

        Args:
            probas: [P, C] probabilities (after softmax)
            labels: [P]   integer class labels
        """
        if probas.numel() == 0:
            # no valid pixels/points
            return probas.new_tensor(0.0)

        C = probas.size(1)
        losses = []

        for c in range(C):
            fg = (labels == c).float()  # foreground mask for class c

            # classes='present' behaviour: skip if class not present
            if fg.sum() == 0:
                continue

            # predicted probabilities for class c
            class_pred = probas[:, c]

            # Lovasz hinge-style error: |fg - p|
            errors = (fg - class_pred).abs()

            # sort descending by error
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]

            grad = LovaszSoftmaxLoss._lovasz_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, grad))

        if not losses:
            # no present classes (should be rare)
            return probas.new_tensor(0.0)

        return sum(losses) / len(losses)

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        """
        Gradient of the Lovasz extension w.r.t. sorted errors.
        Matches the reference `lovasz_grad` implementation.
        """
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1.0 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union

        if jaccard.numel() > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]

        return jaccard


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
        if sum(self.w) <= 0:
            raise ValueError("ComboLoss: at least one weight must be > 0.")

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

        if total is None:
            # Shouldn't happen if we validated in __init__
            return logits.new_tensor(0.0)
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
            alpha_tensor = list(focal_alpha)  # ensure it's a simple sequence

        return FocalLoss(
            gamma=focal_gamma,
            alpha=alpha_tensor,
            ignore_index=ignore_index,
            reduction="mean",
        )

    # ---- DICE (LiDOG-style defaults) ----
    if name == "dice":
        return SoftDiceLoss(
            smooth=dice_smooth,
            ignore_index=ignore_index,
            # powerize/use_tmask already default to True in ctor
        )

    if name == "dice_hard":
        return HardDICELoss(
            ignore_index=ignore_index,
            powerize=True,  # to match LiDOG default for powerize
            use_tmask=True,
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
        # Same plain focal as above
        alpha_tensor = None
        if focal_alpha is not None:
            alpha_tensor = list(focal_alpha)

        focal = FocalLoss(
            gamma=focal_gamma,
            alpha=alpha_tensor,
            ignore_index=ignore_index,
            reduction="mean",
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
