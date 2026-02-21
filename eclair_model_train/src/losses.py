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


# --- Lovasz-Softmax (multiclass) + warmup wrapper ---


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of the Lovasz extension w.r.t sorted errors.
    gt_sorted: [P] float/bool (1 for foreground class, 0 otherwise), sorted by error desc.
    """
    p = gt_sorted.numel()
    if p == 0:
        return gt_sorted

    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_softmax_flat(
    probas: torch.Tensor,
    labels: torch.Tensor,
    *,
    classes: str = "present",
) -> torch.Tensor:
    """
    probas: [N, C] softmax probabilities
    labels: [N] int64 in [0..C-1]
    """
    if probas.numel() == 0:
        return probas.sum() * 0.0

    C = probas.shape[1]
    losses = []

    for c in range(C):
        fg = (labels == c).float()  # [N]
        if classes == "present" and fg.sum() == 0:
            continue

        class_pred = probas[:, c]  # [N]
        errors = (fg - class_pred).abs()  # [N]
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]

        grad = _lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))

    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(torch.nn.Module):
    """
    Multi-class Lovasz-Softmax loss. Ignores ignore_index.
    """

    def __init__(self, *, ignore_index: int = -100, classes: str = "present"):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.classes = str(classes)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [N, C], target: [N]
        valid = target != self.ignore_index
        if valid.sum() == 0:
            return logits.sum() * 0.0

        logits_v = logits[valid]
        target_v = target[valid].to(torch.int64)

        probas = F.softmax(logits_v, dim=1)
        return _lovasz_softmax_flat(probas, target_v, classes=self.classes)


@dataclass
class LovaszWarmupConfig:
    enabled: bool = False
    weight: float = 0.5
    warmup_epochs: int = 10
    ramp_epochs: int = 10
    classes: str = "present"


class FocalLovaszLoss(torch.nn.Module):
    """
    Wrap focal + lovasz with a warmup/ramp schedule.
    """

    def __init__(
        self,
        *,
        focal: torch.nn.Module,
        lovasz: torch.nn.Module,
        cfg: LovaszWarmupConfig,
    ):
        super().__init__()
        self.focal = focal
        self.lovasz = lovasz
        self.cfg = cfg
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def current_weight(self) -> float:
        if not self.cfg.enabled:
            return 0.0
        e = self._epoch
        if e <= int(self.cfg.warmup_epochs):
            return 0.0
        ramp = max(0, int(self.cfg.ramp_epochs))
        if ramp == 0:
            return float(self.cfg.weight)
        t = (e - int(self.cfg.warmup_epochs)) / float(ramp)
        t = max(0.0, min(1.0, t))
        return float(self.cfg.weight) * t

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lf = self.focal(logits, target)
        w = self.current_weight()
        if w <= 0.0:
            return lf
        ll = self.lovasz(logits, target)
        return lf + (w * ll)


def soft_dice_loss_2d(
    logits: torch.Tensor,  # [B,C,H,W]
    target: torch.Tensor,  # [B,H,W]
    *,
    ignore_index: int,
    smooth: float = 1.0,
    classes: str = "present",  # present | all
) -> torch.Tensor:
    # mask ignore
    valid = target != ignore_index
    if valid.sum() == 0:
        return logits.sum() * 0.0

    B, C, H, W = logits.shape
    probs = torch.softmax(logits, dim=1)

    tgt = target.clone()
    tgt[~valid] = 0

    onehot = torch.zeros((B, C, H, W), device=logits.device, dtype=probs.dtype)
    onehot.scatter_(1, tgt.unsqueeze(1), 1.0)
    onehot = onehot * valid.unsqueeze(1).to(onehot.dtype)
    probs = probs * valid.unsqueeze(1).to(probs.dtype)

    # per-class dice
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    den = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + smooth) / (den + smooth)

    if classes == "present":
        present = onehot.sum(dim=(0, 2, 3)) > 0
        if present.any():
            dice = dice[present]
    return 1.0 - dice.mean()
