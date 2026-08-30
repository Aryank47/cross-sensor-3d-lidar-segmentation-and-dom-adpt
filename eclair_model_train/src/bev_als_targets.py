from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .bev_als_config import ALSBEVConfig


@dataclass(frozen=True)
class ALSBEVFrame:
    center_xy_m: np.ndarray
    height_edges_m: np.ndarray
    half_extent_m: float
    resolution_m: float
    height: int
    width: int


@dataclass
class ALSBEVTarget:
    target_u8: np.ndarray
    occupied_u8: np.ndarray
    frame: ALSBEVFrame
    diagnostics: Dict[str, np.ndarray | float | int]


def build_als_bev_frame(
    coords_vox_int32: np.ndarray,
    *,
    meters_per_voxel: float,
    cfg: ALSBEVConfig,
    labels_i64: Optional[np.ndarray] = None,
    ignore_index: int = -100,
) -> ALSBEVFrame:
    coords = np.asarray(coords_vox_int32)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError("BEV-ALS frame requires non-empty [N,3] voxel coordinates.")
    xyz_m = coords.astype(np.float64, copy=False) * float(meters_per_voxel)
    center = 0.5 * (xyz_m[:, :2].min(axis=0) + xyz_m[:, :2].max(axis=0))
    z = xyz_m[:, 2]
    if labels_i64 is not None:
        labels = np.asarray(labels_i64, dtype=np.int64)
        valid = labels != int(ignore_index)
        z_for_edges = z[valid] if np.any(valid) else z
    else:
        z_for_edges = z
    edges = np.quantile(z_for_edges, [0.25, 0.50, 0.75]).astype(np.float32)
    size = int(cfg.grid_size)
    return ALSBEVFrame(
        center_xy_m=np.ascontiguousarray(center, dtype=np.float32),
        height_edges_m=np.ascontiguousarray(edges, dtype=np.float32),
        half_extent_m=float(cfg.half_extent_m),
        resolution_m=float(cfg.resolution_m),
        height=size,
        width=size,
    )


def build_als_bev_target(
    coords_vox_int32: np.ndarray,
    labels_i64: np.ndarray,
    *,
    num_classes: int,
    meters_per_voxel: float,
    cfg: ALSBEVConfig,
    ignore_index: int,
) -> ALSBEVTarget:
    coords = np.asarray(coords_vox_int32, dtype=np.int32)
    labels = np.asarray(labels_i64, dtype=np.int64)
    if coords.shape[0] != labels.shape[0]:
        raise ValueError("BEV-ALS coordinate and label counts differ.")
    frame = build_als_bev_frame(
        coords,
        meters_per_voxel=meters_per_voxel,
        cfg=cfg,
        labels_i64=labels,
        ignore_index=ignore_index,
    )
    xyz = coords.astype(np.float64, copy=False) * float(meters_per_voxel)
    lo = frame.center_xy_m.astype(np.float64) - float(frame.half_extent_m)
    px = np.floor((xyz[:, 0] - lo[0]) / float(frame.resolution_m)).astype(np.int64)
    py = np.floor((xyz[:, 1] - lo[1]) / float(frame.resolution_m)).astype(np.int64)
    valid_label = (labels != int(ignore_index)) & (labels >= 0) & (labels < int(num_classes))
    in_bounds = (px >= 0) & (px < frame.width) & (py >= 0) & (py < frame.height)
    valid = valid_label & in_bounds
    if cfg.y_flip:
        py = (frame.height - 1) - py
    slices = np.searchsorted(frame.height_edges_m.astype(np.float64), xyz[:, 2], side="left").astype(np.int64)
    slices = np.clip(slices, 0, int(cfg.height_slices) - 1)

    target = np.zeros((cfg.height_slices, num_classes, frame.height, frame.width), dtype=np.uint8)
    occupied = np.zeros((cfg.height_slices, frame.height, frame.width), dtype=np.uint8)
    if np.any(valid):
        s = slices[valid]
        c = labels[valid]
        yy = py[valid]
        xx = px[valid]
        target[s, c, yy, xx] = 1
        occupied[s, yy, xx] = 1

    class_before = np.bincount(labels[valid_label], minlength=num_classes)[:num_classes].astype(np.int64)
    class_in_bounds = np.bincount(labels[valid], minlength=num_classes)[:num_classes].astype(np.int64)
    class_retention = np.ones(num_classes, dtype=np.float32)
    present = class_before > 0
    class_retention[present] = class_in_bounds[present] / class_before[present]
    positives = target.sum(axis=(2, 3), dtype=np.int64)
    multi = target.sum(axis=1) > 1
    multi_count = int(multi.sum())
    occupied_count = int(occupied.sum())
    diagnostics: Dict[str, np.ndarray | float | int] = {
        "input_valid_voxels": int(valid_label.sum()),
        "in_bounds_valid_voxels": int(valid.sum()),
        "in_bounds_fraction": float(valid.sum()) / max(1, int(valid_label.sum())),
        "occupied_cells_per_slice": occupied.sum(axis=(1, 2), dtype=np.int64),
        "empty_slices": int((occupied.sum(axis=(1, 2)) == 0).sum()),
        "multi_label_cells": multi_count,
        "multi_label_fraction": float(multi_count) / max(1, occupied_count),
        "class_before": class_before,
        "class_in_bounds": class_in_bounds,
        "class_retention": class_retention,
        "positive_cells_per_slice_class": positives,
        "observed_z_min_m": float(xyz[:, 2].min()),
        "observed_z_max_m": float(xyz[:, 2].max()),
    }
    return ALSBEVTarget(target_u8=target, occupied_u8=occupied, frame=frame, diagnostics=diagnostics)
