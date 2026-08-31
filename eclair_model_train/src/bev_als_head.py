from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bev_als_config import ALSBEVConfig


class ALSHeightSlicedProjector(nn.Module):
    def __init__(self, cfg: ALSBEVConfig):
        super().__init__()
        self.cfg = cfg
        self.reduce = nn.Sequential(
            nn.Linear(cfg.input_channels, cfg.pre_pool_dim, bias=False),
            nn.LayerNorm(cfg.pre_pool_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _scatter_max(feats: torch.Tensor, flat: torch.Tensor, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        d = int(feats.shape[1])
        valid = torch.zeros(size, dtype=torch.bool, device=feats.device)
        if flat.numel() == 0:
            return feats.new_zeros((size, d)), valid
        valid[flat] = True
        if hasattr(torch.Tensor, "scatter_reduce_"):
            out = feats.new_full((size, d), float("-inf"))
            out.scatter_reduce_(0, flat[:, None].expand(-1, d), feats, reduce="amax", include_self=True)
            out = out.masked_fill(~valid[:, None], 0.0)
            return out, valid

        # Differentiable compatibility fallback for older PyTorch builds.
        order = torch.argsort(flat)
        flat_s = flat.index_select(0, order)
        feats_s = feats.index_select(0, order)
        uniq, counts = torch.unique_consecutive(flat_s, return_counts=True)
        chunks = torch.split(feats_s, counts.detach().cpu().tolist())
        maxima = torch.stack([chunk.max(dim=0).values for chunk in chunks], dim=0)
        out = feats.new_zeros((size, d)).index_copy(0, uniq, maxima)
        return out, valid

    def forward(
        self,
        sparse_tensor,
        *,
        meters_per_voxel: float,
        center_xy_m: torch.Tensor,
        height_edges_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feats = self.reduce(sparse_tensor.F)
        coords = sparse_tensor.C.to(device=feats.device, dtype=torch.int64)
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError("BEV-ALS expects Minkowski coordinates [N,4].")
        if int(coords.shape[0]) != int(feats.shape[0]):
            raise ValueError("BEV-ALS block8 coordinate/feature counts differ.")
        if tuple(int(x) for x in sparse_tensor.tensor_stride) != (1, 1, 1):
            raise ValueError(f"BEV-ALS v1 requires block8 tensor_stride=1; got {sparse_tensor.tensor_stride}")

        center = center_xy_m.to(device=feats.device, dtype=torch.float32)
        edges = height_edges_m.to(device=feats.device, dtype=torch.float32)
        if center.ndim != 2 or tuple(center.shape[1:]) != (2,):
            raise ValueError("BEV-ALS center_xy_m must have shape [B,2].")
        if edges.ndim != 2 or tuple(edges.shape[1:]) != (self.cfg.height_slices - 1,):
            raise ValueError(
                f"BEV-ALS height_edges_m must have shape [B,{self.cfg.height_slices - 1}]."
            )
        b = coords[:, 0]
        batch_size = int(center.shape[0])
        if coords.numel() and (int(b.min()) < 0 or int(b.max()) >= batch_size):
            raise ValueError("BEV-ALS frame batch size does not cover sparse coordinates.")
        xyz = coords[:, 1:4].to(torch.float32) * float(meters_per_voxel)
        rel_xy = xyz[:, :2] - center.index_select(0, b)
        half = float(self.cfg.half_extent_m)
        in_bounds = (rel_xy[:, 0] >= -half) & (rel_xy[:, 0] < half) & (rel_xy[:, 1] >= -half) & (rel_xy[:, 1] < half)
        size = int(self.cfg.grid_size)
        px = torch.floor((rel_xy[:, 0] + half) / float(self.cfg.resolution_m)).to(torch.int64)
        py = torch.floor((rel_xy[:, 1] + half) / float(self.cfg.resolution_m)).to(torch.int64)
        if self.cfg.y_flip:
            py = (size - 1) - py
        sample_edges = edges.index_select(0, b)
        height_slice = (xyz[:, 2:3] > sample_edges).sum(dim=1).to(torch.int64)
        height_slice = height_slice.clamp(0, self.cfg.height_slices - 1)

        b2 = b[in_bounds]
        s2 = height_slice[in_bounds]
        x2 = px[in_bounds]
        y2 = py[in_bounds]
        f2 = feats[in_bounds]
        flat = (((b2 * self.cfg.height_slices + s2) * size + y2) * size + x2).to(torch.int64)
        total = batch_size * self.cfg.height_slices * size * size
        dense, valid = self._scatter_max(f2, flat, total)
        dense = dense.view(batch_size, self.cfg.height_slices, size, size, self.cfg.pre_pool_dim)
        dense = dense.permute(0, 1, 4, 2, 3).reshape(
            batch_size,
            self.cfg.height_slices * self.cfg.pre_pool_dim,
            size,
            size,
        )
        valid_grid = valid.view(batch_size, self.cfg.height_slices, size, size)
        sample_total = torch.bincount(b, minlength=batch_size).to(torch.float32)
        sample_in_bounds = torch.bincount(b[in_bounds], minlength=batch_size).to(torch.float32)
        per_sample_fraction = sample_in_bounds / sample_total.clamp_min(1.0)
        diagnostics = {
            "feature_in_bounds_fraction": in_bounds.float().mean().detach() if in_bounds.numel() else feats.new_tensor(1.0),
            "feature_in_bounds_fraction_per_sample": per_sample_fraction.detach(),
            "feature_valid_cells": valid_grid.sum().detach().to(torch.float32),
        }
        return dense, diagnostics


class ALSBEVDecoder(nn.Module):
    def __init__(self, cfg: ALSBEVConfig, num_classes: int):
        super().__init__()
        in_ch = cfg.height_slices * cfg.pre_pool_dim
        h = cfg.decoder_hidden_dim
        layers = []
        for i in range(3):
            layers.extend(
                [
                    nn.Conv2d(in_ch if i == 0 else h, h, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(h),
                    nn.ReLU(inplace=True),
                ]
            )
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(h, cfg.height_slices * num_classes, kernel_size=1)
        self.height_slices = cfg.height_slices
        self.num_classes = int(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.out(self.body(x))
        b, _, h, w = logits.shape
        return logits.view(b, self.height_slices, self.num_classes, h, w)


class ALSBEVHead(nn.Module):
    def __init__(self, cfg: ALSBEVConfig, num_classes: int):
        super().__init__()
        self.projector = ALSHeightSlicedProjector(cfg)
        self.decoder = ALSBEVDecoder(cfg, num_classes)

    def forward(self, sparse_tensor, *, meters_per_voxel: float, frames: Dict[str, torch.Tensor]):
        dense, diag = self.projector(
            sparse_tensor,
            meters_per_voxel=meters_per_voxel,
            center_xy_m=frames["center_xy_m"],
            height_edges_m=frames["height_edges_m"],
        )
        return self.decoder(dense), diag


def als_bev_multilabel_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    occupied: torch.Tensor,
    cfg: ALSBEVConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if logits.shape != target.shape:
        raise ValueError(f"BEV-ALS logits/target shape mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}")
    if occupied.shape != (logits.shape[0], logits.shape[1], logits.shape[3], logits.shape[4]):
        raise ValueError("BEV-ALS occupied mask shape mismatch.")
    target_f = target.to(device=logits.device, dtype=logits.dtype)
    occ = occupied.to(device=logits.device, dtype=torch.bool)
    mask = occ.unsqueeze(2).expand_as(target_f)
    if bool(mask.any()):
        bce = F.binary_cross_entropy_with_logits(logits[mask], target_f[mask])
    else:
        bce = logits.sum() * 0.0

    probs = torch.sigmoid(logits)
    m = occ.unsqueeze(2).to(logits.dtype)
    dims = (0, 3, 4)
    inter = (probs * target_f * m).sum(dim=dims)
    pred_sum = (probs * m).sum(dim=dims)
    target_sum = (target_f * m).sum(dim=dims)
    dice_sc = (2.0 * inter + 1.0) / (pred_sum + target_sum + 1.0)
    present = target_sum > 0
    dice_loss = (1.0 - dice_sc[present]).mean() if bool(present.any()) else logits.sum() * 0.0
    total = float(cfg.bce_weight) * bce + float(cfg.dice_weight) * dice_loss

    with torch.no_grad():
        pred = probs >= float(cfg.metric_threshold)
        truth = target_f > 0.5
        p = pred & mask
        t = truth & mask
        metric_dims = (0, 3, 4)
        tp_sc = (p & t).sum(dim=metric_dims).to(torch.float32)
        fp_sc = (p & ~t).sum(dim=metric_dims).to(torch.float32)
        fn_sc = (~p & t).sum(dim=metric_dims).to(torch.float32)
        tp = tp_sc.sum()
        fp = fp_sc.sum()
        fn = fn_sc.sum()
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    return total, {
        "bce": bce.detach(),
        "dice_loss": dice_loss.detach(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_by_slice_class": tp_sc,
        "fp_by_slice_class": fp_sc,
        "fn_by_slice_class": fn_sc,
        "positive_support": target_sum.detach(),
    }
