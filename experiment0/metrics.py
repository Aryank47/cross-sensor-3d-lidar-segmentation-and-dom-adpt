# ================================================
# FILE: metrics.py
# ================================================
from typing import Dict, List, Optional

import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.long)

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        num = self.num_classes
        y_pred = y_pred.reshape(-1).to('cpu', non_blocking=True)
        y_true = y_true.reshape(-1).to('cpu', non_blocking=True)

        # guard both sides
        valid = (y_true >= 0) & (y_true < num) & (y_pred >= 0) & (y_pred < num)
        if valid.any():
            inds = (y_true[valid] * num + y_pred[valid]).to(torch.int64)
            self.mat += torch.bincount(inds, minlength=num*num).reshape(num, num)

    def value(self) -> torch.Tensor:
        return self.mat


def per_class_iou(conf: torch.Tensor, ignore_ids: Optional[List[int]] = None) -> torch.Tensor:
    K = conf.shape[0]
    ious = torch.zeros(K)
    for c in range(K):
        if ignore_ids and c in ignore_ids:
            ious[c] = float('nan')
            continue
        tp = conf[c, c].float()
        fp = conf[:, c].sum().float() - tp
        fn = conf[c, :].sum().float() - tp
        denom = tp + fp + fn
        ious[c] = (tp / denom) if denom > 0 else float('nan')
    return ious


def per_class_f1(conf: torch.Tensor, ignore_ids: Optional[List[int]] = None) -> torch.Tensor:
    K = conf.shape[0]
    f1 = torch.zeros(K)
    for c in range(K):
        if ignore_ids and c in ignore_ids:
            f1[c] = float('nan')
            continue
        tp = conf[c, c].float()
        fp = conf[:, c].sum().float() - tp
        fn = conf[c, :].sum().float() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1[c] = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else float('nan')
    return f1


def compute_scores(confusion: ConfusionMatrix, ignore_ids: Optional[List[int]] = None, label_set: Optional[List[int]] = None) -> Dict:
    conf = confusion.value().float()
    iou_pc = per_class_iou(conf, ignore_ids)
    f1_pc = per_class_f1(conf, ignore_ids)

    # define which labels to average over
    if label_set is None:
        valid = torch.ones_like(iou_pc, dtype=torch.bool)
    else:
        valid = torch.zeros_like(iou_pc, dtype=torch.bool)
        for idx in label_set:
            if ignore_ids and idx in ignore_ids:
                continue
            valid[idx] = True

    # mask out NaNs
    valid &= ~torch.isnan(iou_pc)
    valid &= ~torch.isnan(f1_pc)

    miou = torch.mean(iou_pc[valid]).item() if valid.any() else 0.0
    macrof1 = torch.mean(f1_pc[valid]).item() if valid.any() else 0.0

    return {
        "mIoU": miou,
        "macroF1": macrof1,
        "IoU_per_class": iou_pc.tolist(),
        "F1_per_class": f1_pc.tolist(),
        "confusion": conf.cpu().tolist(),
    }


def map_labels_tensor(labels: torch.Tensor, mapping: Dict[int, int]) -> torch.Tensor:
    # mapping is a python dict int->int; make a vectorized map
    device = labels.device
    max_id = max(mapping.keys()) if mapping else 0
    lut_size = max(max_id + 1, int(labels.max().item()) + 1)
    lut = torch.zeros(lut_size, dtype=torch.long, device=device)
    for k, v in mapping.items():
        if k < lut_size:
            lut[k] = int(v)
    out = torch.zeros_like(labels)
    valid = labels < lut_size
    out[valid] = lut[labels[valid]]
    return out



