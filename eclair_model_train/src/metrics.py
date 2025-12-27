from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class MetricResult:
    miou: float
    macro_f1: float
    per_class_iou: List[float]
    per_class_f1: List[float]


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = -100):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.mat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        pred, target: [N] int64
        """
        pred = pred.view(-1).to(torch.int64).cpu()
        target = target.view(-1).to(torch.int64).cpu()
        mask = target != self.ignore_index
        if mask.sum() == 0:
            return
        pred = pred[mask]
        target = target[mask]
        k = self.num_classes
        idx = target * k + pred
        bins = torch.bincount(idx, minlength=k * k)
        self.mat += bins.view(k, k)

    def reset(self) -> None:
        self.mat.zero_()

    def compute(self) -> MetricResult:
        mat = self.mat.to(torch.float64)

        tp = torch.diag(mat)
        fp = mat.sum(dim=0) - tp
        fn = mat.sum(dim=1) - tp

        denom_iou = tp + fp + fn
        iou = torch.where(denom_iou > 0, tp / denom_iou, torch.zeros_like(denom_iou))

        denom_f1 = 2 * tp + fp + fn
        f1 = torch.where(denom_f1 > 0, (2 * tp) / denom_f1, torch.zeros_like(denom_f1))

        miou = float(iou.mean().item())
        macro_f1 = float(f1.mean().item())

        return MetricResult(
            miou=miou,
            macro_f1=macro_f1,
            per_class_iou=[float(x) for x in iou.tolist()],
            per_class_f1=[float(x) for x in f1.tolist()],
        )
