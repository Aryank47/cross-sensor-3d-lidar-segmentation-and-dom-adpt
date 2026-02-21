# src/bev_labels.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .bev_head import BEVHeadConfig


def _grid_for_level(bev_cfg: BEVHeadConfig, level: str) -> Tuple[int, int, float, float]:
    """
    Returns (H, W, xGrid, yGrid).
    If bev_cfg.img_size_by_level is set, use square H=W=size and compute grid sizes from bounds.
    Else, use res_m.
    """
    p = bev_cfg.projector
    x_min, x_max = float(p.x_min_m), float(p.x_max_m)
    y_min, y_max = float(p.y_min_m), float(p.y_max_m)

    if bev_cfg.img_size_by_level is not None and level in bev_cfg.img_size_by_level:
        size = int(bev_cfg.img_size_by_level[level])
        H = W = max(1, size)
        xGrid = (x_max - x_min) / float(W)
        yGrid = (y_max - y_min) / float(H)
        return H, W, xGrid, yGrid

    res = float(p.res_m)
    W = max(1, int(np.ceil((x_max - x_min) / res)))
    H = max(1, int(np.ceil((y_max - y_min) / res)))
    return H, W, res, res


def build_bev_labels_and_selected_idx(
    *,
    coords_vox_int32: np.ndarray,  # [Nv,3] voxel coords (x,y,z) ints (NO batch dim)
    labels_vox_i64: np.ndarray,  # [Nv] voxel labels (train ids; includes ignore_index)
    voxel_ignore_index: int,
    bev_cfg: BEVHeadConfig,
    level: str,
    meters_per_voxel: float,
    rng: Optional[np.random.Generator],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LiDOG-like BEV labels from voxels:
      - convert voxel coords -> meters
      - filter x/y/z bounds
      - project to pixels (with optional y_flip)
      - collision handled with bev_cfg.label_pool (default 'last' matches LiDOG getBEVImageNew assignment behavior)
    Returns:
      bev_labels: [H,W] int64 filled with bev_cfg.ignore_index where empty
      bev_selected_idx: [H,W] int64 filled with -1 where empty (stores voxel index 0..Nv-1)
    """
    p = bev_cfg.projector
    x_min, x_max = float(p.x_min_m), float(p.x_max_m)
    y_min, y_max = float(p.y_min_m), float(p.y_max_m)
    z_min, z_max = float(p.z_min_m), float(p.z_max_m)
    y_flip = bool(p.y_flip)

    H, W, xGrid, yGrid = _grid_for_level(bev_cfg, level)

    bev_ignore = int(bev_cfg.ignore_index)
    out_lbl = np.full((H, W), bev_ignore, dtype=np.int64)
    out_idx = np.full((H, W), -1, dtype=np.int64)

    if coords_vox_int32.size == 0:
        return out_lbl, out_idx

    coords = coords_vox_int32.astype(np.float32, copy=False) * float(meters_per_voxel)
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    lab = labels_vox_i64.astype(np.int64, copy=False)

    valid = (lab != int(voxel_ignore_index)) & (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max) & (z > z_min) & (z < z_max)
    if not np.any(valid):
        return out_lbl, out_idx

    v_idx = np.where(valid)[0]
    x = x[valid]
    y = y[valid]
    lab = lab[valid]

    px = np.floor((x - x_min) / float(xGrid)).astype(np.int64)
    py = np.floor((y - y_min) / float(yGrid)).astype(np.int64)

    px = np.clip(px, 0, W - 1)
    py = np.clip(py, 0, H - 1)

    if y_flip:
        py = (H - 1) - py

    # collision policy
    policy = str(bev_cfg.label_pool).lower()
    if policy not in ("last", "first", "random", "majority"):
        raise ValueError(f"Unknown BEV label_pool='{policy}' (expected last|first|random|majority)")

    flat = py * W + px

    if policy == "last":
        # LiDOG-like: last assignment wins (deterministic given input order)
        out_lbl[py, px] = lab
        out_idx[py, px] = v_idx
        return out_lbl, out_idx

    if policy == "first":
        # first assignment wins: choose first per flat cell in stable sorted order
        order = np.argsort(flat, kind="mergesort")
        flat_s = flat[order]
        keep = np.ones_like(flat_s, dtype=bool)
        keep[1:] = flat_s[1:] != flat_s[:-1]
        sel = order[keep]
        out_lbl[py[sel], px[sel]] = lab[sel]
        out_idx[py[sel], px[sel]] = v_idx[sel]
        return out_lbl, out_idx

    if policy == "random":
        if rng is None:
            rng = np.random.default_rng(1337)
        r = rng.random(flat.shape[0]).astype(np.float32, copy=False)

        # pick max random score per flat cell
        order = np.lexsort((r, flat))  # sort by flat then r
        flat_s = flat[order]
        # take last in each group = max r
        keep = np.ones_like(flat_s, dtype=bool)
        keep[:-1] = flat_s[:-1] != flat_s[1:]
        sel = order[keep]
        out_lbl[py[sel], px[sel]] = lab[sel]
        out_idx[py[sel], px[sel]] = v_idx[sel]
        return out_lbl, out_idx

    # majority vote (slower, but useful ablation)
    order = np.argsort(flat, kind="mergesort")
    flat_s = flat[order]
    lab_s = lab[order]
    vid_s = v_idx[order]

    start = 0
    while start < flat_s.size:
        f = flat_s[start]
        end = start + 1
        while end < flat_s.size and flat_s[end] == f:
            end += 1

        group = lab_s[start:end]
        # choose majority label
        mx = int(group.max())
        bc = np.bincount(group.astype(np.int64, copy=False), minlength=mx + 1)
        lbl = int(bc.argmax())

        # for selected_idx: pick the first voxel in this group with that label
        mask = group == lbl
        chosen = start + int(np.argmax(mask))  # first True
        chosen_vid = int(vid_s[chosen])

        py0 = int(f // W)
        px0 = int(f % W)
        out_lbl[py0, px0] = lbl
        out_idx[py0, px0] = chosen_vid

        start = end

    return out_lbl, out_idx
