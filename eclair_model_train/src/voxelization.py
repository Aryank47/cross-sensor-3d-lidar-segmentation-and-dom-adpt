# src/voxelization.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import MinkowskiEngine as ME
import numpy as np
import torch


@dataclass(frozen=True)
class VoxelizationConfig:
    feat_pool: str = "sample_first"  # sample_first | mean_all | max_all | random
    label_pool: str = "first"  # first | majority
    random_seed_offset: int = 0  # used for feat_pool=random

    @staticmethod
    def from_cfg(data_cfg: Dict[str, Any]) -> "VoxelizationConfig":
        vx = data_cfg.get("voxelization", None) or {}
        # Backward-compat: allow old key data.voxel_feat_mode if present
        old = str(data_cfg.get("voxel_feat_mode", "")).strip().lower()
        feat_pool = str(vx.get("feat_pool", old or "sample_first")).lower()

        # normalize a few aliases
        aliases = {
            "first": "sample_first",
            "sample": "sample_first",
            "mean": "mean_all",
            "max": "max_all",
            "random_one": "random",
        }
        feat_pool = aliases.get(feat_pool, feat_pool)

        label_pool = str(vx.get("label_pool", "first")).lower()
        label_pool = {"majority_vote": "majority", "vote": "majority"}.get(label_pool, label_pool)

        return VoxelizationConfig(
            feat_pool=feat_pool,
            label_pool=label_pool,
            random_seed_offset=int(vx.get("random_seed_offset", 0)),
        )


def _scatter_mean(feats_p: torch.Tensor, inv: torch.Tensor, nv: int) -> torch.Tensor:
    # feats_p: [Np, C]
    sums = torch.zeros((nv, feats_p.shape[1]), dtype=torch.float32)
    sums.index_add_(0, inv, feats_p.float())
    counts = torch.bincount(inv, minlength=nv).clamp_min(1).to(torch.float32).unsqueeze(1)
    return sums / counts


def _scatter_max(feats_p: torch.Tensor, inv: torch.Tensor, nv: int) -> torch.Tensor:
    # robust: use scatter_reduce_ if available; else fallback to numpy grouping
    if hasattr(torch.Tensor, "scatter_reduce_"):
        out = torch.full((nv, feats_p.shape[1]), -1e30, dtype=torch.float32)
        idx = inv.view(-1, 1).expand(-1, feats_p.shape[1])
        out.scatter_reduce_(0, idx, feats_p.float(), reduce="amax", include_self=True)
        return out
    # fallback (slower but correct)
    feats_np = feats_p.detach().cpu().numpy().astype(np.float32, copy=False)
    inv_np = inv.detach().cpu().numpy().astype(np.int64, copy=False)
    out = np.full((nv, feats_np.shape[1]), -1e30, dtype=np.float32)
    order = np.argsort(inv_np, kind="mergesort")
    inv_s = inv_np[order]
    feats_s = feats_np[order]
    start = 0
    while start < inv_s.size:
        v = inv_s[start]
        end = start + 1
        while end < inv_s.size and inv_s[end] == v:
            end += 1
        out[v] = feats_s[start:end].max(axis=0)
        start = end
    return torch.from_numpy(out)


def _pool_random_one(
    *,
    inv_np: np.ndarray,
    nv: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Pick one point index per voxel uniformly at random.
    Returns sel_point_idx_per_voxel: [nv] indices into the point array.
    """
    N = inv_np.shape[0]
    # Random score per point; pick max score per voxel
    r = rng.random(N).astype(np.float32, copy=False)
    order = np.lexsort((r, inv_np))  # sort by inv then r
    inv_s = inv_np[order]
    # take last occurrence of each voxel id
    sel = np.empty((nv,), dtype=np.int64)
    start = 0
    while start < inv_s.size:
        v = inv_s[start]
        end = start + 1
        while end < inv_s.size and inv_s[end] == v:
            end += 1
        sel[v] = order[end - 1]
        start = end
    return sel


def _pool_labels_majority(
    *,
    labels_p: np.ndarray,  # [Np] int64
    inv_np: np.ndarray,  # [Np] int64
    nv: int,
    ignore_index: int,
    num_classes_hint: Optional[int] = None,
) -> np.ndarray:
    """
    Majority vote per voxel. If all labels in a voxel are ignore_index -> ignore_index.
    """
    order = np.argsort(inv_np, kind="mergesort")
    inv_s = inv_np[order]
    lab_s = labels_p[order]

    out = np.full((nv,), int(ignore_index), dtype=np.int64)

    start = 0
    while start < inv_s.size:
        v = int(inv_s[start])
        end = start + 1
        while end < inv_s.size and inv_s[end] == v:
            end += 1

        group = lab_s[start:end]
        group = group[group != ignore_index]
        if group.size > 0:
            mx = int(group.max())
            K = max((num_classes_hint or 0), mx + 1)
            # bincount needs non-negative
            bc = np.bincount(group.astype(np.int64, copy=False), minlength=K)
            out[v] = int(bc.argmax())

        start = end

    return out


def voxelize_from_q(
    *,
    q_int32: np.ndarray,  # [Np,3] int32
    feats_p_f32: np.ndarray,  # [Np,C] float32
    labels_p_i64: Optional[np.ndarray],  # [Np] int64 (train ids), optional
    ignore_index: int,
    cfg: VoxelizationConfig,
    rng: Optional[np.random.Generator] = None,
    return_maps: bool = True,
    num_classes_hint: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Quantize + pool features (and optionally labels) per voxel.
    Returns ME-ready coords/features and (optional) inverse_map.
    """
    q_t = torch.from_numpy(np.ascontiguousarray(q_int32, dtype=np.int32)).int().contiguous()
    out = ME.utils.sparse_quantize(q_t, return_index=True, return_inverse=True)
    if not (isinstance(out, (tuple, list)) and len(out) >= 3):
        raise RuntimeError("ME sparse_quantize did not return inverse_map; check MinkowskiEngine version.")
    coords_u_t, unique_map_t, inverse_map_t = out[0], out[1], out[2]

    coords_u = coords_u_t.cpu().numpy().astype(np.int32, copy=False)  # [Nv,3]
    unique_map = unique_map_t.cpu().numpy().astype(np.int64, copy=False)  # [Nv]
    inverse_map = inverse_map_t.cpu().numpy().astype(np.int64, copy=False)  # [Np]
    nv = int(coords_u.shape[0])

    feats_p_t = torch.from_numpy(np.ascontiguousarray(feats_p_f32, dtype=np.float32))
    inv_t = torch.from_numpy(inverse_map).long()

    feat_pool = cfg.feat_pool
    if feat_pool == "sample_first":
        feats_u = feats_p_f32[unique_map]
    elif feat_pool == "mean_all":
        feats_u = _scatter_mean(feats_p_t, inv_t, nv).cpu().numpy().astype(np.float32, copy=False)
    elif feat_pool == "max_all":
        feats_u = _scatter_max(feats_p_t, inv_t, nv).cpu().numpy().astype(np.float32, copy=False)
    elif feat_pool == "random":
        if rng is None:
            rng = np.random.default_rng(1337 + int(cfg.random_seed_offset))
        sel = _pool_random_one(inv_np=inverse_map, nv=nv, rng=rng)
        feats_u = feats_p_f32[sel]
    else:
        raise ValueError(f"Unknown voxelization.feat_pool='{feat_pool}'")

    labels_u: Optional[np.ndarray] = None
    if labels_p_i64 is not None:
        lab_pool = cfg.label_pool
        if lab_pool == "first":
            labels_u = labels_p_i64[unique_map].astype(np.int64, copy=False)
        elif lab_pool == "majority":
            labels_u = _pool_labels_majority(
                labels_p=labels_p_i64.astype(np.int64, copy=False),
                inv_np=inverse_map,
                nv=nv,
                ignore_index=int(ignore_index),
                num_classes_hint=num_classes_hint,
            )
        else:
            raise ValueError(f"Unknown voxelization.label_pool='{lab_pool}'")

    return {
        "coords_u": coords_u,
        "feats_u": feats_u,
        "labels_u": labels_u,
        "unique_map": unique_map,
        "inverse_map": inverse_map if return_maps else None,
    }
