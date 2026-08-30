from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml

from .config_loader import load_yaml
from .model import build_model
from .utils import read_las_arrays_robust

try:
    import MinkowskiEngine as ME
except Exception as e:
    raise RuntimeError("MinkowskiEngine is required.") from e


# -------------------------
# Logging utilities
# -------------------------
def setup_logger(out_dir: Path, name: str, level=logging.INFO) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(out_dir / "run.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# YAML mapping loaders
# -------------------------
def load_yaml_map_int_dict(path: str) -> Dict[int, int]:
    mp = yaml.safe_load(Path(path).read_text())
    out: Dict[int, int] = {}
    for k, v in mp.items():
        out[int(k)] = int(v)
    return out


def load_yaml_lut_fixed(path: str, *, size: int, default: int = 0) -> np.ndarray:
    """
    Fixed-size LUT with safe default.
    - For DALES native labels: use size=256, default=0 (ignore)
    - For pred_to_common override: use size=out_channels, default=0 (ignore)
    """
    mp = yaml.safe_load(Path(path).read_text())
    lut = np.full((size,), int(default), dtype=np.int64)
    for k, v in mp.items():
        kk = int(k)
        if 0 <= kk < size:
            lut[kk] = int(v)
    return lut


def build_pred_to_common_from_eclair_map(
    eclair_native_to_common: Dict[int, int],
    *,
    out_channels: int,
) -> np.ndarray:
    """
    Model predicts train_id in [0..out_channels-1].
    ECLAIR native ids are expected to be train_id + 1.
    """
    pred_to_common = np.zeros((out_channels,), dtype=np.int64)
    for train_id in range(out_channels):
        native_id = train_id + 1
        pred_to_common[train_id] = int(eclair_native_to_common.get(native_id, 0))
    return pred_to_common


# -------------------------
# Metrics
# -------------------------
def confusion_from_labels(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    ignore_index: int = 0,
) -> np.ndarray:
    assert gt.shape == pred.shape
    mask = gt != ignore_index
    gt = gt[mask].astype(np.int64, copy=False)
    pred = pred[mask].astype(np.int64, copy=False)

    gt = np.clip(gt, 0, num_classes - 1)
    pred = np.clip(pred, 0, num_classes - 1)

    idx = gt * num_classes + pred
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape((num_classes, num_classes))


def iou_from_confusion(
    cm: np.ndarray, ignore_index: int = 0
) -> Tuple[float, List[float]]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    denom = tp + fp + fn + 1e-12
    iou = (tp / denom).tolist()
    valid = [i for i in range(cm.shape[0]) if i != ignore_index]
    miou = float(np.mean([iou[i] for i in valid])) if valid else float("nan")
    return miou, iou


# -------------------------
# DALES reading
# -------------------------
# def read_dales_las(path: Path) -> Dict[str, np.ndarray]:
#     # NOTE: laspy.read loads all points; for very large tiles ensure memory is sufficient.
#     las = laspy.read(str(path))

#     xyz = las.xyz.astype(np.float32, copy=False)
#     dims = set(las.point_format.dimension_names)
#     # intensity
#     if "intensity" in dims:
#         intensity = np.asarray(las["intensity"])
#         intensity = getattr(intensity, "array", intensity)
#         intensity = np.asarray(intensity).astype(np.float32, copy=False)
#     else:
#         intensity = np.zeros((xyz.shape[0],), dtype=np.float32)
#     print(
#         "DALES intensity stats:",
#         float(intensity.min()),
#         float(intensity.max()),
#         float(np.median(intensity)),
#     )

#     # returns
#     def _dim(name: str, default: int) -> np.ndarray:
#         if name in dims:
#             arr = las[name]
#             arr = getattr(arr, "array", arr)
#             return np.asarray(arr)
#         return np.full((xyz.shape[0],), default, dtype=np.uint8)

#     rn = _dim("return_number", 1).astype(np.int64, copy=False)

#     nor = _dim("number_of_returns", 1).astype(np.int64, copy=False)
#     # rn = (
#     #     np.asarray(las.return_number).astype(np.int64, copy=False)
#     #     if "return_number" in dims
#     #     else np.ones((xyz.shape[0],), dtype=np.int64)
#     # )
#     # nor = (
#     #     np.asarray(las.number_of_returns).astype(np.int64, copy=False)
#     #     if "number_of_returns" in dims
#     #     else np.ones((xyz.shape[0],), dtype=np.int64)
#     # )

#     # classification / GT
#     gt_key = "classification" if "classification" in dims else "raw_classification"
#     cls = las[gt_key]
#     cls = getattr(cls, "array", cls)
#     cls = np.asarray(cls).astype(np.int64, copy=False)

#     return {
#         "xyz": xyz,
#         "intensity": intensity,
#         "return_number": rn,
#         "number_of_returns": nor,
#         "cls": cls,
#     }


def read_dales_las(path: Path) -> Dict[str, np.ndarray]:
    """
    Robust reader wrapper for DALES inference.
    Matches old output signature: keys = [xyz, intensity, return_number, number_of_returns, cls]
    """
    data = read_las_arrays_robust(path)

    # Preserve original logging behavior
    print(
        "DALES intensity stats:",
        float(data["intensity"].min()),
        float(data["intensity"].max()),
        float(np.median(data["intensity"])),
    )

    return {
        "xyz": data["xyz"],
        "intensity": data["intensity"],
        "return_number": data["return_number"],
        "number_of_returns": data["number_of_returns"],
        "cls": data[
            "native_labels"
        ],  # <--- Renamed 'native_labels' to 'cls' to match your old code
    }


# -------------------------
# Patch iterator (non-overlapping default)
# -------------------------
def iter_xy_patches_indices(
    xyz: np.ndarray, patch_size_m: float, patch_stride_m: float
) -> Iterable[np.ndarray]:
    x = xyz[:, 0]
    y = xyz[:, 1]
    xmin, ymin = float(x.min()), float(y.min())

    gx = np.floor((x - xmin) / patch_stride_m).astype(np.int32)
    gy = np.floor((y - ymin) / patch_stride_m).astype(np.int32)

    key = (gx.astype(np.int64) << 32) | (gy.astype(np.int64) & 0xFFFFFFFF)
    order = np.argsort(key, kind="mergesort")
    key_sorted = key[order]

    start = 0
    n = len(order)
    while start < n:
        k = key_sorted[start]
        end = start + 1
        while end < n and key_sorted[end] == k:
            end += 1

        idx = order[start:end]
        cx = int(gx[idx[0]])
        cy = int(gy[idx[0]])
        x0 = xmin + cx * patch_stride_m
        y0 = ymin + cy * patch_stride_m
        x1 = x0 + patch_size_m
        y1 = y0 + patch_size_m

        inside = idx[(x[idx] >= x0) & (x[idx] < x1) & (y[idx] >= y0) & (y[idx] < y1)]
        if inside.size > 0:
            yield inside

        start = end


# -------------------------
# ECLAIR-consistent feature builder (locks dim=11)
# -------------------------
def _one_hot_1_based(values_1_based: np.ndarray, k: int) -> np.ndarray:
    v = values_1_based.astype(np.int64, copy=False)
    v = np.clip(v, 1, k) - 1
    out = np.zeros((v.shape[0], k), dtype=np.float32)
    out[np.arange(v.shape[0]), v] = 1.0
    return out


def scale_intensity_0_1(intensity: np.ndarray, divisor: float) -> np.ndarray:
    x = intensity.astype(np.float32, copy=False)
    # If DALES is already normalized, don't squash it by 65535
    if float(np.max(x)) <= 1.0 + 1e-3:
        return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    return np.clip(x / float(divisor), 0.0, 1.0).astype(np.float32, copy=False)


# -------------------------
# Optional: linearity proxy (feature surgery / structure-aware reps)
# -------------------------
def compute_linearity(
    xyz: np.ndarray,
    k: int,
    logger: logging.Logger,
    *,
    max_points: int = 50000,
    seed: int = 1337,
) -> Optional[np.ndarray]:
    """
    Returns linearity in [0,1] per point using local covariance eigenvalues.
    Requires sklearn; if missing, returns None.
    """
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
    except Exception:
        logger.warning("[lin] sklearn not available -> linearity-based modes disabled.")
        return None

    n = xyz.shape[0]
    if n < 3:
        return np.zeros((n,), dtype=np.float32)

    rng = np.random.default_rng(seed)
    k_eff = min(k, n)

    def _linearity_core(x: np.ndarray) -> np.ndarray:
        nn = NearestNeighbors(n_neighbors=min(k_eff, x.shape[0]), algorithm="auto")
        nn.fit(x)
        _, inds = nn.kneighbors(x, return_distance=True)
        lin = np.zeros((x.shape[0],), dtype=np.float32)
        for i in range(x.shape[0]):
            neigh = x[inds[i]]
            c = neigh - neigh.mean(axis=0, keepdims=True)
            cov = (c.T @ c) / max(1, c.shape[0])
            w = np.linalg.eigvalsh(cov)
            w = np.sort(w)[::-1]  # λ1>=λ2>=λ3
            l1, l2 = float(w[0]), float(w[1])
            lin[i] = (l1 - l2) / (l1 + 1e-6)
        return np.clip(lin, 0.0, 1.0).astype(np.float32, copy=False)

    # Hard bound: if too many points, compute on a subset and assign via 1-NN
    if n > int(max_points):
        sub = rng.choice(n, size=int(max_points), replace=False)
        xyz_sub = xyz[sub]
        lin_sub = _linearity_core(xyz_sub)
        nn1 = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn1.fit(xyz_sub)
        _, ind1 = nn1.kneighbors(xyz, return_distance=True)
        return lin_sub[ind1[:, 0]].astype(np.float32, copy=False)

    return _linearity_core(xyz)


# -------------------------
# Preprocessing spec (scientifically valid)
# -------------------------
@dataclass(frozen=True)
class PreprocSpec:
    # Baseline matches training:
    # - local coords per patch (always applied in code)
    # - coord_norm_factor default 10
    # - voxel_size default 0.05 (in normalized coords)
    coord_norm_factor: float = 10.0
    voxel_size: float = 0.05  # in normalized space, like training
    z_scale: float = (
        1.0  # anisotropic scaling on normalized z (generic; targets span_z shift)
    )

    # Feature transforms (dim locked to 11)
    intensity_mode: str = (
        "as_is"  # as_is|zero|constant|proxy_z|minmax_z|proxy_linearity|quantile_match
    )
    intensity_constant: float = 0.5  # in [0,1]
    returns_mode: str = "as_is"  # onehot|zero|constant|drop_rn|drop_nor
    returns_k: int = 5
    intensity_divisor: float = 65535.0

    # Voxel representative strategy (does NOT change eval set)
    # Model input has 1 feature vector per occupied voxel.
    # We choose how to compute that feature vector from all points in the voxel.
    voxel_feat_mode: str = (
        "sample_first"  # sample_first|sample_random|sample_max_intensity|sample_max_linearity|mean_all|mean_thin
    )
    mean_thin_p: float = 0.7  # used if voxel_feat_mode == mean_thin
    knn_k: int = 16  # for linearity modes
    linearity_max_points: int = 50000
    linearity_seed: int = 1337

    # TTA
    tta_mode: str = "none"  # none|rot4

    # Patch filter (IMPORTANT: does NOT affect "full" metrics, only "filtered" metrics)
    patch_filter_mode: str = "none"  # none|min_points|min_occ_vox
    min_points: int = 1000
    min_occ_vox: int = 1000

    # Intensity reference for quantile matching (optional)
    # If not provided, quantile_match is skipped (no hallucination).
    has_intensity_ref: bool = False

    def to_id(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.md5(payload).hexdigest()[:10]


def _canon(v):
    # Canonicalize floats to avoid 0.04 vs 0.04000000000000001 mismatches
    if isinstance(v, float):
        return round(v, 8)
    if isinstance(v, dict):
        return {k: _canon(v[k]) for k in sorted(v.keys())}
    if isinstance(v, list):
        return [_canon(x) for x in v]
    return v


def spec_key_from_dict(d: Dict) -> Tuple:
    # Stable tuple key over ALL spec fields
    items = []
    for k in sorted(d.keys()):
        items.append((k, _canon(d[k])))
    return tuple(items)


def spec_key_from_spec(s: PreprocSpec) -> Tuple:
    return spec_key_from_dict(dataclasses.asdict(s))


# -------------------------
# Rotation TTA
# -------------------------
def rotate_z(xyz: np.ndarray, deg: int) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (xyz @ R.T).astype(np.float32, copy=False)


# -------------------------
# Quantile matching LUT (requires ECLAIR intensity samples)
# -------------------------
def build_quantile_lut(
    source_vals_0_1: np.ndarray, bins: int = 4096
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - p_grid: percentiles in [0,1]
      - q_source: source quantiles (same length)
    """
    source_vals_0_1 = np.clip(source_vals_0_1.astype(np.float32, copy=False), 0.0, 1.0)
    p_grid = np.linspace(0.0, 1.0, bins, dtype=np.float32)
    q_source = np.quantile(source_vals_0_1, p_grid).astype(np.float32)
    return p_grid, q_source


def apply_quantile_match(
    target_vals_0_1: np.ndarray, p_grid: np.ndarray, q_source: np.ndarray
) -> np.ndarray:
    """
    Monotone mapping: i' = F_S^{-1}(F_T(i)).
    Implemented by mapping target values -> target CDF percentile -> source quantile.
    """
    t = np.clip(target_vals_0_1.astype(np.float32, copy=False), 0.0, 1.0)
    # compute target percentiles by ranking approximation via its own quantiles
    # (This is robust and avoids histogram bin artifacts.)
    q_target = np.quantile(t, p_grid).astype(np.float32)
    # For each value t, find p such that q_target[p] ~ t, then output q_source[p]
    p_of_t = np.interp(t, q_target, p_grid).astype(np.float32)
    out = np.interp(p_of_t, p_grid, q_source).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


# -------------------------
# EDA-alignment stats per patch
# -------------------------
def patch_alignment_stats(
    xyz_local: np.ndarray,
    xyz_norm: np.ndarray,
    intensity_0_1: np.ndarray,
    voxel_size: float,
) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    n = xyz_local.shape[0]
    stats["n_points"] = float(n)
    if n == 0:
        return stats

    # intensity diagnostics
    stats["inten_min"] = float(np.min(intensity_0_1))
    stats["inten_med"] = float(np.median(intensity_0_1))
    stats["inten_max"] = float(np.max(intensity_0_1))
    # degeneracy proxy: unique fraction on a rounded grid
    inten_rounded = np.round(intensity_0_1, 4)
    stats["inten_unique_frac"] = float(len(np.unique(inten_rounded)) / max(1, n))

    # z span in meters (local coords)
    z = xyz_local[:, 2]
    stats["z_span_m"] = float(np.max(z) - np.min(z))

    # voxel occupancy
    q = np.floor(xyz_norm / float(voxel_size)).astype(np.int32, copy=False)
    q = np.ascontiguousarray(q, dtype=np.int32)
    keys = (
        q[:, 0].astype(np.int64) * 73856093
        ^ q[:, 1].astype(np.int64) * 19349663
        ^ q[:, 2].astype(np.int64) * 83492791
    )
    uniq, cnt = np.unique(keys, return_counts=True)
    stats["occ_vox"] = float(len(uniq))
    stats["pts_per_vox_mean"] = float(np.mean(cnt))
    stats["pts_per_vox_med"] = float(np.median(cnt))
    return stats


LINEARITY_INTEN_MODES = {"proxy_linearity"}
LINEARITY_VOX_MODES = {"sample_max_linearity"}


def spec_uses_linearity(spec: PreprocSpec) -> bool:
    return (spec.intensity_mode in LINEARITY_INTEN_MODES) or (
        spec.voxel_feat_mode in LINEARITY_VOX_MODES
    )


# -------------------------
# Build per-point features (dim=11) then per-voxel features (dim=11)
# -------------------------


def build_point_features_dim11(
    xyz_norm: np.ndarray,
    intensity_raw: np.ndarray,
    rn: np.ndarray,
    nor: np.ndarray,
    spec: PreprocSpec,
    logger: logging.Logger,
    *,
    p_grid: Optional[np.ndarray] = None,
    q_source: Optional[np.ndarray] = None,
    linearity: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - intensity_0_1_used (N,)
      - feats_points (N, 11)
    """
    # baseline intensity scaled like training
    inten_0_1 = scale_intensity_0_1(intensity_raw, spec.intensity_divisor)

    # intensity transform
    if spec.intensity_mode == "as_is":
        inten_used = inten_0_1
    elif spec.intensity_mode == "zero":
        inten_used = np.zeros_like(inten_0_1)
    elif spec.intensity_mode == "constant":
        inten_used = np.full_like(inten_0_1, float(spec.intensity_constant))
    elif spec.intensity_mode == "proxy_z":
        # use normalized z (already in normalized coords); squash to [0,1] using robust percentiles
        z = xyz_norm[:, 2].astype(np.float32, copy=False)
        lo, hi = np.quantile(z, 0.01), np.quantile(z, 0.99)
        inten_used = np.clip((z - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)
    elif spec.intensity_mode == "minmax_z":
        z = xyz_norm[:, 2].astype(np.float32, copy=False)
        z0, z1 = float(np.min(z)), float(np.max(z))
        inten_used = np.clip((z - z0) / (z1 - z0 + 1e-6), 0.0, 1.0).astype(np.float32)
    elif spec.intensity_mode == "proxy_linearity":
        if linearity is None:
            logger.warning(
                "[inten] proxy_linearity requested but linearity unavailable -> using constant."
            )
            inten_used = np.full_like(inten_0_1, float(spec.intensity_constant))
        else:
            inten_used = np.clip(linearity.astype(np.float32, copy=False), 0.0, 1.0)
    elif spec.intensity_mode == "quantile_match":
        # scientifically valid ONLY if:
        # - we have a source reference (ECLAIR intensity samples)
        # - target intensity is not degenerate
        if (p_grid is None) or (q_source is None):
            logger.warning(
                "[inten] quantile_match requested but no ECLAIR intensity reference -> using as_is."
            )
            inten_used = inten_0_1
        else:
            # detect degeneracy
            if float(np.max(inten_0_1)) <= 1e-6 and float(np.min(inten_0_1)) >= -1e-6:
                logger.warning(
                    "[inten] DALES intensity appears degenerate (~0). Quantile match impossible -> using constant."
                )
                inten_used = np.full_like(inten_0_1, float(spec.intensity_constant))
            else:
                inten_used = apply_quantile_match(inten_0_1, p_grid, q_source)
    else:
        raise ValueError(f"Unknown intensity_mode: {spec.intensity_mode}")

    inten_used = inten_used.reshape(-1, 1).astype(np.float32, copy=False)

    # returns one-hot like training (k=5)
    rn_oh = _one_hot_1_based(rn, spec.returns_k)
    nor_oh = _one_hot_1_based(nor, spec.returns_k)

    # returns transform (dim preserved)
    if spec.returns_mode == "as_is":
        pass
    elif spec.returns_mode == "drop":
        rn_oh[:] = 0.0
        nor_oh[:] = 0.0
    elif spec.returns_mode == "const1":
        rn_oh[:] = 0.0
        nor_oh[:] = 0.0
        rn_oh[:, 0] = 1.0
        nor_oh[:, 0] = 1.0
    elif spec.returns_mode == "uniform":
        rn_oh[:] = 1.0 / float(spec.returns_k)
        nor_oh[:] = 1.0 / float(spec.returns_k)
    else:
        raise ValueError(f"Unknown returns_mode: {spec.returns_mode}")

    feats_points = np.concatenate([inten_used, rn_oh, nor_oh], axis=1).astype(
        np.float32, copy=False
    )
    assert feats_points.shape[1] == (
        1 + 2 * spec.returns_k
    ), "Feature dim must stay locked."

    return inten_used.reshape(-1), feats_points


def voxelize_with_inverse(
    xyz_norm: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - q_u (M,3) unique voxel coords (int32)
      - unique_idx (M,) indices of representative points (training-like)
      - inv (N,) mapping each point -> voxel index in [0..M-1]
    """
    q = np.floor(xyz_norm / float(voxel_size)).astype(np.int32, copy=False)
    q = np.ascontiguousarray(q, dtype=np.int32)
    q_t = torch.from_numpy(q).int().contiguous()
    # ME returns:
    #   - unique coords tensor (unused here)
    #   - unique indices into original points
    #   - inverse map point->unique index
    _, unique_idx_t, inv_t = ME.utils.sparse_quantize(
        q_t, return_index=True, return_inverse=True
    )
    unique_idx = unique_idx_t.cpu().numpy().astype(np.int64, copy=False)
    inv = inv_t.cpu().numpy().astype(np.int64, copy=False)
    q_u = q[unique_idx]
    q_u = np.ascontiguousarray(q_u, dtype=np.int32)
    return q_u, unique_idx, inv


def build_voxel_features_dim11(
    feats_points: np.ndarray,
    inv: np.ndarray,
    unique_idx: np.ndarray,
    spec: PreprocSpec,
    logger: logging.Logger,
    *,
    intensity_0_1_used: Optional[np.ndarray] = None,
    linearity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build one feature vector per occupied voxel (M,11) without changing eval set.
    Modes:
      - sample_first: training-like (use unique_idx returned by sparse_quantize)
      - sample_random: pick random point within each voxel
      - sample_max_intensity: pick max intensity within voxel (requires intensity_0_1_used)
      - sample_max_linearity: pick max linearity within voxel (requires linearity)
      - mean_all: mean of all points within voxel (soft features)
      - mean_thin: mean of a random subset within voxel (keeps at least one point)
    """
    M = unique_idx.shape[0]
    if spec.voxel_feat_mode == "sample_first":
        return feats_points[unique_idx]

    # group points by voxel using inv sorting
    order = np.argsort(inv, kind="mergesort")
    inv_sorted = inv[order]

    out = np.zeros((M, feats_points.shape[1]), dtype=np.float32)

    start = 0
    N = inv_sorted.shape[0]
    while start < N:
        v = inv_sorted[start]
        end = start + 1
        while end < N and inv_sorted[end] == v:
            end += 1
        pts_idx = order[start:end]  # indices into original points for voxel v

        if spec.voxel_feat_mode == "sample_random":
            chosen = np.random.choice(pts_idx, size=1)[0]
            out[v] = feats_points[chosen]
        elif spec.voxel_feat_mode == "sample_max_intensity":
            if intensity_0_1_used is None:
                logger.warning(
                    "[vox] sample_max_intensity needs intensity -> falling back to sample_first for this voxel."
                )
                out[v] = feats_points[unique_idx[v]]
            else:
                chosen = pts_idx[np.argmax(intensity_0_1_used[pts_idx])]
                out[v] = feats_points[chosen]
        elif spec.voxel_feat_mode == "sample_max_linearity":
            if linearity is None:
                logger.warning(
                    "[vox] sample_max_linearity needs linearity -> falling back to sample_first for this voxel."
                )
                out[v] = feats_points[unique_idx[v]]
            else:
                chosen = pts_idx[np.argmax(linearity[pts_idx])]
                out[v] = feats_points[chosen]
        elif spec.voxel_feat_mode == "mean_all":
            out[v] = feats_points[pts_idx].mean(axis=0)
        elif spec.voxel_feat_mode == "mean_thin":
            p = float(spec.mean_thin_p)
            if p >= 1.0 or pts_idx.size == 1:
                out[v] = feats_points[pts_idx].mean(axis=0)
            else:
                keep = np.random.rand(pts_idx.size) < p
                if not np.any(keep):
                    keep[np.random.randint(0, pts_idx.size)] = (
                        True  # ensure at least one
                    )
                out[v] = feats_points[pts_idx[keep]].mean(axis=0)
        else:
            raise ValueError(f"Unknown voxel_feat_mode: {spec.voxel_feat_mode}")

        start = end

    return out.astype(np.float32, copy=False)


# -------------------------
# Model loading (no hallucination)
# -------------------------
def load_model_from_config(
    *,
    config_path: str,
    ckpt_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, bool, int]:
    """
    Returns: (model, amp_enabled, out_channels)
    """
    cfg = load_yaml(config_path)

    model_cfg = cfg["model"]
    out_channels = int(model_cfg["out_channels"])

    model = build_model(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=out_channels,
        D=int(model_cfg.get("D", 3)),
    ).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    sd = (
        state["model_state"]
        if isinstance(state, dict) and "model_state" in state
        else state
    )
    model.load_state_dict(sd, strict=True)
    model.eval()

    amp = bool(cfg.get("run", {}).get("amp", True)) and (device.type == "cuda")
    return model, amp, out_channels


# -------------------------
# Infer logits per point (keeps eval set fixed!)
# -------------------------
@torch.inference_mode()
def infer_patch_logits_per_point(
    model: torch.nn.Module,
    xyz_norm: np.ndarray,
    feats_points: np.ndarray,
    intensity_used: np.ndarray,
    spec: PreprocSpec,
    device: torch.device,
    logger: logging.Logger,
    *,
    amp: bool,
    linearity: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Returns logits per ORIGINAL point (N,C) as a torch.Tensor on `device`.
    """
    q_u, unique_idx, inv = voxelize_with_inverse(xyz_norm, spec.voxel_size)
    feats_vox = build_voxel_features_dim11(
        feats_points=feats_points,
        inv=inv,
        unique_idx=unique_idx,
        spec=spec,
        logger=logger,
        intensity_0_1_used=intensity_used,
        linearity=linearity,
    )

    # ME coords require batch column
    coords = np.concatenate(
        [np.zeros((q_u.shape[0], 1), dtype=np.int32), q_u],
        axis=1,
    )
    coords_t = (
        torch.from_numpy(np.ascontiguousarray(coords, dtype=np.int32)).int().to(device)
    )
    feats_t = (
        torch.from_numpy(np.ascontiguousarray(feats_vox, dtype=np.float32))
        .float()
        .to(device)
    )

    st = ME.SparseTensor(features=feats_t, coordinates=coords_t, device=device)
    with torch.autocast(device_type=device.type, enabled=amp):
        out = model(st)
        logits_vox = out.F  # (M,C) on device

    inv_t = torch.from_numpy(inv).long().to(device)
    logits_pts = logits_vox[inv_t]  # (N,C)

    # Drop references ASAP (helps long sweeps)
    del st, out, logits_vox, coords_t, feats_t, inv_t
    return logits_pts


def infer_with_tta(
    model: torch.nn.Module,
    xyz_local: np.ndarray,
    intensity_raw: np.ndarray,
    rn: np.ndarray,
    nor: np.ndarray,
    spec: PreprocSpec,
    device: torch.device,
    logger: logging.Logger,
    *,
    amp: bool,
    p_grid: Optional[np.ndarray] = None,
    q_source: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Returns pred_train per point (N,) and alignment stats for the *unrotated* geometry.
    """
    # local coords like training
    xyz_centered = xyz_local - xyz_local.min(axis=0, keepdims=True)
    xyz_local = xyz_centered.astype(np.float32, copy=False)

    # normalized coords like training (+ z anisotropic scaling)
    xyz_norm = (xyz_local / float(spec.coord_norm_factor)).astype(
        np.float32, copy=False
    )
    xyz_norm = np.ascontiguousarray(xyz_norm, dtype=np.float32)
    xyz_norm[:, 2] *= float(spec.z_scale)

    # compute bounded linearity if needed
    linearity = None
    if spec_uses_linearity(spec):
        linearity = compute_linearity(
            xyz_local,
            spec.knn_k,
            logger,
            max_points=int(spec.linearity_max_points),
            seed=int(spec.linearity_seed),
        )

    # per-point features (dim=11)
    inten_used, feats_points = build_point_features_dim11(
        xyz_norm=xyz_norm,
        intensity_raw=intensity_raw,
        rn=rn,
        nor=nor,
        spec=spec,
        logger=logger,
        p_grid=p_grid,
        q_source=q_source,
        linearity=linearity,
    )

    # alignment stats for logging
    stats = patch_alignment_stats(
        xyz_local=xyz_local,
        xyz_norm=xyz_norm,
        intensity_0_1=inten_used,
        voxel_size=spec.voxel_size,
    )

    if spec.tta_mode == "none":
        logits = infer_patch_logits_per_point(
            model=model,
            xyz_norm=xyz_norm,
            feats_points=feats_points,
            intensity_used=inten_used,
            spec=spec,
            device=device,
            logger=logger,
            linearity=linearity,
            amp=amp,
        )
        pred_train = torch.argmax(logits, dim=1).to(dtype=torch.int64).cpu().numpy()
        del logits
        return pred_train, stats

    if spec.tta_mode == "rot4":
        logits_acc = None
        for deg in (0, 90, 180, 270):
            xyz_r = rotate_z(xyz_local, deg)
            xyz_r = xyz_r - xyz_r.min(axis=0, keepdims=True)
            xyz_rn = (xyz_r / float(spec.coord_norm_factor)).astype(
                np.float32, copy=False
            )
            xyz_rn = np.ascontiguousarray(xyz_rn, dtype=np.float32)
            xyz_rn[:, 2] *= float(spec.z_scale)

            # Rebuild features using rotated coords (proxy_z depends on z; keep consistent)
            # Reuse linearity if available; it was computed on xyz_local. For strictness,
            # recompute on rotated coords only if needed (costly). We'll keep same linearity.
            inten_r, feats_r = build_point_features_dim11(
                xyz_norm=xyz_rn,
                intensity_raw=intensity_raw,
                rn=rn,
                nor=nor,
                spec=spec,
                logger=logger,
                p_grid=p_grid,
                q_source=q_source,
                linearity=linearity,
            )
            lg = infer_patch_logits_per_point(
                model=model,
                xyz_norm=xyz_rn,
                feats_points=feats_r,
                intensity_used=inten_r,
                spec=spec,
                device=device,
                logger=logger,
                linearity=linearity,
                amp=amp,
            )
            logits_acc = lg if logits_acc is None else (logits_acc + lg)
            del lg
        assert logits_acc is not None
        logits_acc = logits_acc / 4.0
        pred_train = torch.argmax(logits_acc, dim=1).to(dtype=torch.int64).cpu().numpy()
        del logits_acc
        return pred_train, stats

    raise ValueError(f"Unknown tta_mode: {spec.tta_mode}")


# -------------------------
# Mapping: model output IDs -> common IDs (no hallucination)
# -------------------------
def derive_pred_to_common_map(
    eclair_native_to_common: np.ndarray,
    *,
    try_import: bool,
    undefined_id: int,
    ignore_index: int,
    logger: logging.Logger,
) -> Optional[np.ndarray]:
    """
    Your mapping_eclair_to_common.yaml is native_id -> common_id.
    But the model outputs TRAIN IDs (0..out_channels-1), not native IDs.
    We derive train_id -> common_id by importing eclair_native_to_train_ids from your repo.
    If import fails, return None and require the user to pass --mapping_pred_to_common.
    """
    if not try_import:
        return None
    try:
        # Try a few common import roots (HPC runs often differ in PYTHONPATH)
        candidates = [
            "src.label_maps",
            "eclair_model_train.src.label_maps",
            "label_maps",
        ]
        eclair_native_to_train_ids = None
        for mod in candidates:
            try:
                m = importlib.import_module(mod)
                eclair_native_to_train_ids = getattr(m, "eclair_native_to_train_ids")
                break
            except Exception:
                continue
        if eclair_native_to_train_ids is None:
            raise ImportError(
                f"Could not import eclair_native_to_train_ids from any of {candidates}"
            )
    except Exception as e:
        logger.warning(
            f"[map] Could not import src.label_maps.eclair_native_to_train_ids: {e}"
        )
        return None

    native_ids = np.arange(len(eclair_native_to_common), dtype=np.int64)
    train_ids = eclair_native_to_train_ids(
        native_ids, undefined_id=undefined_id, ignore_index=ignore_index
    )

    # train_id -> native_id (expect mostly one-to-one for {1..11} -> {0..10})
    train_to_native: Dict[int, int] = {}
    for nid, tid in enumerate(train_ids.tolist()):
        if tid == ignore_index:
            continue
        if tid < 0:
            continue
        # If collisions exist, we keep the first and warn.
        if tid in train_to_native and train_to_native[tid] != nid:
            logger.warning(
                f"[map] train_id {tid} maps to multiple native ids ({train_to_native[tid]}, {nid}). Keeping first."
            )
            continue
        train_to_native[tid] = nid

    if not train_to_native:
        logger.warning(
            "[map] Derived empty train_to_native map. Cannot create pred_to_common."
        )
        return None

    max_tid = max(train_to_native.keys())
    pred_to_common = np.zeros((max_tid + 1,), dtype=np.int64)
    for tid, nid in train_to_native.items():
        nid_clamped = int(np.clip(nid, 0, len(eclair_native_to_common) - 1))
        pred_to_common[tid] = int(eclair_native_to_common[nid_clamped])

    logger.info(f"[map] Derived pred_to_common for train ids 0..{max_tid}")
    return pred_to_common


# -------------------------
# Evaluation
# -------------------------
@dataclass
class EvalTotals:
    cm_full: np.ndarray
    cm_filtered: np.ndarray
    n_points_full: int
    n_points_filtered: int
    n_patches_full: int
    n_patches_filtered: int
    n_patches_filtered_out: int
    align_stats_sum: Dict[str, float]
    align_stats_n: int


def empty_totals(C: int) -> EvalTotals:
    return EvalTotals(
        cm_full=np.zeros((C, C), dtype=np.int64),
        cm_filtered=np.zeros((C, C), dtype=np.int64),
        n_points_full=0,
        n_points_filtered=0,
        n_patches_full=0,
        n_patches_filtered=0,
        n_patches_filtered_out=0,
        align_stats_sum={},
        align_stats_n=0,
    )


def add_align_stats(t: EvalTotals, stats: Dict[str, float]) -> None:
    t.align_stats_n += 1
    for k, v in stats.items():
        t.align_stats_sum[k] = t.align_stats_sum.get(k, 0.0) + float(v)


def meets_patch_filter(
    xyz_norm: np.ndarray,
    spec: PreprocSpec,
) -> bool:
    n = xyz_norm.shape[0]
    if spec.patch_filter_mode == "none":
        return True
    if spec.patch_filter_mode == "min_points":
        return n >= int(spec.min_points)
    if spec.patch_filter_mode == "min_occ_vox":
        q = np.floor(xyz_norm / float(spec.voxel_size)).astype(np.int32, copy=False)
        q = np.ascontiguousarray(q, dtype=np.int32)
        keys = (
            q[:, 0].astype(np.int64) * 73856093
            ^ q[:, 1].astype(np.int64) * 19349663
            ^ q[:, 2].astype(np.int64) * 83492791
        )
        occ = len(np.unique(keys))
        return occ >= int(spec.min_occ_vox)
    raise ValueError(f"Unknown patch_filter_mode: {spec.patch_filter_mode}")


def evaluate_run(
    *,
    model: torch.nn.Module,
    dales_files: List[Path],
    map_dales_to_common: np.ndarray,
    pred_to_common: np.ndarray,
    common_class_names: List[str],
    spec: PreprocSpec,
    device: torch.device,
    logger: logging.Logger,
    patch_size_m: float,
    patch_stride_m: float,
    p_grid: Optional[np.ndarray],
    q_source: Optional[np.ndarray],
    amp: bool,
) -> Dict:
    C = len(common_class_names)
    totals = empty_totals(C)

    t0 = time.time()
    logger.info(f"[run] spec={dataclasses.asdict(spec)}")
    logger.info(
        f"[data] n_files={len(dales_files)} patch={patch_size_m} stride={patch_stride_m}"
    )

    for fi, fp in enumerate(dales_files, 1):
        raw = read_dales_las(fp)
        xyz = raw["xyz"]
        inten = raw["intensity"]
        rn = raw["return_number"]
        nor = raw["number_of_returns"]
        gt_native = raw["cls"]

        # map DALES native -> common (clamp)
        gt_native_clip = np.clip(gt_native, 0, 255)
        gt_common = map_dales_to_common[gt_native_clip]

        logger.info(f"[file] {fi}/{len(dales_files)} {fp.name} n_points={xyz.shape[0]}")

        patch_i = 0
        for idx in iter_xy_patches_indices(xyz, patch_size_m, patch_stride_m):
            patch_i += 1

            xyz_p = xyz[idx]
            inten_p = inten[idx]
            rn_p = rn[idx]
            nor_p = nor[idx]
            gt_p = gt_common[idx]

            # local coords and normalized coords (for filter decisions)
            xyz_local = xyz_p - xyz_p.min(axis=0, keepdims=True)
            xyz_norm_for_filter = (xyz_local / float(spec.coord_norm_factor)).astype(
                np.float32, copy=False
            )
            xyz_norm_for_filter = np.ascontiguousarray(
                xyz_norm_for_filter, dtype=np.float32
            )
            xyz_norm_for_filter[:, 2] *= float(spec.z_scale)

            # FULL scope always
            pred_train, a_stats = infer_with_tta(
                model=model,
                xyz_local=xyz_p,
                intensity_raw=inten_p,
                rn=rn_p,
                nor=nor_p,
                spec=spec,
                device=device,
                logger=logger,
                p_grid=p_grid,
                q_source=q_source,
                amp=amp,
            )
            add_align_stats(totals, a_stats)

            pred_train = pred_train.astype(np.int64, copy=False)
            if int(pred_train.max()) >= pred_to_common.shape[0]:
                raise RuntimeError(
                    f"pred_to_common too small: pred_max={int(pred_train.max())} "
                    f"but lut_size={pred_to_common.shape[0]}"
                )
            pred_common = pred_to_common[pred_train]

            cm = confusion_from_labels(gt_p, pred_common, C, ignore_index=0)
            totals.cm_full += cm
            totals.n_points_full += int(gt_p.shape[0])
            totals.n_patches_full += 1

            # FILTERED scope (analysis only; does not bias "full" deltas)
            keep_patch = meets_patch_filter(xyz_norm_for_filter, spec)
            if keep_patch:
                totals.cm_filtered += cm
                totals.n_points_filtered += int(gt_p.shape[0])
                totals.n_patches_filtered += 1
            else:
                totals.n_patches_filtered_out += 1

            # periodic logs
            if patch_i % 25 == 0:
                logger.info(
                    f"[patch] {fp.name} patch_i={patch_i} n={len(idx)} "
                    f"occ_vox={a_stats.get('occ_vox', -1):.0f} "
                    f"ppv_mean={a_stats.get('pts_per_vox_mean', -1):.2f} "
                    f"inten_med={a_stats.get('inten_med', -1):.4f} "
                    f"inten_unique_frac={a_stats.get('inten_unique_frac', -1):.6f} "
                    f"filtered_kept={totals.n_patches_filtered} filtered_out={totals.n_patches_filtered_out}"
                )
                if device.type == "cuda":
                    logger.info(
                        f"[cuda] alloc={torch.cuda.memory_allocated()/1e9:.3f}GB "
                        f"reserved={torch.cuda.memory_reserved()/1e9:.3f}GB"
                    )
        # file-level cleanup helps long sweeps
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    miou_full, iou_full = iou_from_confusion(totals.cm_full, ignore_index=0)
    miou_filt, iou_filt = iou_from_confusion(totals.cm_filtered, ignore_index=0)

    runtime = time.time() - t0

    # aggregate alignment stats
    align_avg = {}
    if totals.align_stats_n > 0:
        for k, v in totals.align_stats_sum.items():
            align_avg[k] = float(v / totals.align_stats_n)

    return {
        "spec": dataclasses.asdict(spec),
        "runtime_s": runtime,
        "full": {
            "miou": miou_full,
            "iou_per_class": iou_full,
            "n_points": totals.n_points_full,
            "n_patches": totals.n_patches_full,
        },
        "filtered": {
            "miou": miou_filt,
            "iou_per_class": iou_filt,
            "n_points": totals.n_points_filtered,
            "n_patches_kept": totals.n_patches_filtered,
            "n_patches_dropped": totals.n_patches_filtered_out,
            "filter_mode": spec.patch_filter_mode,
        },
        "align_avg": align_avg,
    }


# -------------------------
# Run plan generation (staged + limited combos)
# -------------------------
def make_run_plan(base: PreprocSpec, mode: str, max_runs: int) -> List[PreprocSpec]:
    runs: List[PreprocSpec] = [base]

    # -------------------------
    # "Kill Switch" runs (must-run, science-first)
    # -------------------------
    # 1) Geom-only: remove intensity corruption, use permutation-invariant voxel features
    runs.append(
        dataclasses.replace(base, intensity_mode="zero", voxel_feat_mode="mean_all")
    )

    # 2) Statistical alignment: only meaningful when has_intensity_ref=True (will be gated later)
    runs.append(
        dataclasses.replace(
            base, intensity_mode="quantile_match", voxel_feat_mode="mean_all"
        )
    )

    # 3) Structural invariance: slow mode (will be gated unless --include_slow_modes)
    runs.append(
        dataclasses.replace(
            base,
            intensity_mode="proxy_linearity",
            voxel_feat_mode="sample_max_linearity",
        )
    )

    # 4) View invariance: rot4 TTA  mean aggregation
    runs.append(dataclasses.replace(base, tta_mode="rot4", voxel_feat_mode="mean_all"))

    intensity_pool = [
        ("as_is", {}),
        ("zero", {}),
        ("constant", {"intensity_constant": 0.5}),
        ("proxy_z", {}),
        ("minmax_z", {}),
        # SLOW/OPTIONAL: only include if enabled externally
        ("proxy_linearity", {}),
        ("quantile_match", {}),  # only valid if has_intensity_ref=True
    ]
    returns_pool = [
        ("as_is", {}),
        ("drop", {}),
        ("const1", {}),
        ("uniform", {}),
    ]
    coord_pool = [0.8, 1.0, 1.2]  # scale coord_norm_factor
    voxel_pool = [0.8, 1.0, 1.2]  # scale voxel_size
    zscale_pool = [0.8, 1.0, 1.2]
    voxfeat_pool = [
        ("sample_first", {}),
        ("sample_random", {}),
        ("sample_max_intensity", {}),
        # SLOW/OPTIONAL: only include if enabled externally
        ("sample_max_linearity", {}),
        ("mean_all", {}),
        ("mean_thin", {"mean_thin_p": 0.7}),
    ]
    tta_pool = ["none", "rot4"]

    filter_pool = [
        ("none", {}),
        ("min_points", {"min_points": 1000}),
        ("min_occ_vox", {"min_occ_vox": 1000}),
    ]

    def add(r: PreprocSpec):
        if len(runs) < max_runs:
            runs.append(r)

    if mode == "staged":
        # single-factor ablations (clean science)
        for im, kw in intensity_pool:
            add(dataclasses.replace(base, intensity_mode=im, **kw))
        for rm, kw in returns_pool:
            add(dataclasses.replace(base, returns_mode=rm, **kw))
        for m in coord_pool:
            add(dataclasses.replace(base, coord_norm_factor=base.coord_norm_factor * m))
        for m in voxel_pool:
            add(dataclasses.replace(base, voxel_size=base.voxel_size * m))
        for zs in zscale_pool:
            add(dataclasses.replace(base, z_scale=zs))
        for vm, kw in voxfeat_pool:
            add(dataclasses.replace(base, voxel_feat_mode=vm, **kw))
        for tm in tta_pool:
            add(dataclasses.replace(base, tta_mode=tm))
        for fm, kw in filter_pool:
            add(dataclasses.replace(base, patch_filter_mode=fm, **kw))

        # limited combos (motivated by EDA: intensity collapse + sparsity + z-span)
        combos = [
            # robust to missing intensity + voxel mean aggregation + TTA
            dataclasses.replace(
                base,
                intensity_mode="constant",
                intensity_constant=0.5,
                voxel_feat_mode="mean_all",
                tta_mode="rot4",
            ),
            # preserve thin structures proxy: linearity rep + smaller voxels
            dataclasses.replace(
                base,
                voxel_feat_mode="sample_max_linearity",
                voxel_size=base.voxel_size * 0.8,
                intensity_mode="constant",
            ),
            # z-span compensation + smaller voxel + mean
            dataclasses.replace(
                base,
                z_scale=1.2,
                voxel_size=base.voxel_size * 0.8,
                voxel_feat_mode="mean_all",
                intensity_mode="constant",
            ),
        ]
        for c in combos:
            add(c)

    elif mode == "full":
        # broader cross product but capped
        for im, imkw in intensity_pool:
            for rm, rmkw in returns_pool:
                for cm in coord_pool:
                    for vmult in voxel_pool:
                        for zs in zscale_pool:
                            for vfm, vfkw in voxfeat_pool:
                                for tm in tta_pool:
                                    for fm, fmkw in filter_pool:
                                        if len(runs) >= max_runs:
                                            break
                                        runs.append(
                                            dataclasses.replace(
                                                base,
                                                intensity_mode=im,
                                                returns_mode=rm,
                                                coord_norm_factor=base.coord_norm_factor
                                                * cm,
                                                voxel_size=base.voxel_size * vmult,
                                                z_scale=zs,
                                                voxel_feat_mode=vfm,
                                                tta_mode=tm,
                                                patch_filter_mode=fm,
                                                **imkw,
                                                **rmkw,
                                                **vfkw,
                                                **fmkw,
                                            )
                                        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # de-dup
    seen = set()
    uniq: List[PreprocSpec] = []
    for r in runs:
        rid = r.to_id()
        if rid in seen:
            continue
        uniq.append(r)
        seen.add(rid)
    return uniq[:max_runs]


def make_decisive_returns_plan_full(
    base: PreprocSpec,
    *,
    c: float,
    has_ref: bool,
    include_slow: bool,
) -> List[PreprocSpec]:
    """
    High-signal plan intended for comparing new checkpoints.
    ~32 runs (plus quantile_match + slow modes if enabled).
    """
    runs: List[PreprocSpec] = []

    def add(s: PreprocSpec) -> None:
        runs.append(s)

    # -------------------------
    # Group A: Baselines / sanity
    # -------------------------
    add(base)
    add(dataclasses.replace(base, returns_mode="drop"))
    add(dataclasses.replace(base, returns_mode="uniform"))
    add(dataclasses.replace(base, returns_mode="const1"))

    add(dataclasses.replace(base, voxel_feat_mode="mean_all"))
    add(dataclasses.replace(base, voxel_feat_mode="mean_all", returns_mode="drop"))

    add(dataclasses.replace(base, voxel_feat_mode="mean_thin", mean_thin_p=0.7))
    add(dataclasses.replace(base, voxel_feat_mode="sample_max_intensity"))

    add(dataclasses.replace(base, tta_mode="rot4"))
    add(dataclasses.replace(base, tta_mode="rot4", voxel_feat_mode="mean_all"))

    # -------------------------
    # Group B: Intensity robustness
    # -------------------------
    add(dataclasses.replace(base, intensity_mode="constant", intensity_constant=c))
    add(
        dataclasses.replace(
            base, intensity_mode="constant", intensity_constant=c, returns_mode="drop"
        )
    )

    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="mean_all",
        )
    )
    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="mean_all",
            returns_mode="drop",
        )
    )

    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="mean_all",
            tta_mode="rot4",
        )
    )
    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="mean_all",
            tta_mode="rot4",
            returns_mode="drop",
        )
    )

    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="sample_max_intensity",
        )
    )

    add(dataclasses.replace(base, intensity_mode="zero", voxel_feat_mode="mean_all"))
    add(dataclasses.replace(base, intensity_mode="proxy_z", voxel_feat_mode="mean_all"))
    add(
        dataclasses.replace(base, intensity_mode="minmax_z", voxel_feat_mode="mean_all")
    )

    if has_ref:
        add(
            dataclasses.replace(
                base, intensity_mode="quantile_match", voxel_feat_mode="mean_all"
            )
        )
        add(
            dataclasses.replace(
                base,
                intensity_mode="quantile_match",
                voxel_feat_mode="mean_all",
                tta_mode="rot4",
            )
        )
        add(
            dataclasses.replace(
                base,
                intensity_mode="quantile_match",
                voxel_feat_mode="mean_all",
                tta_mode="rot4",
                returns_mode="drop",
            )
        )

    # -------------------------
    # Group C: Geometry sweeps around "best recipe"
    # -------------------------
    best = dataclasses.replace(
        base,
        intensity_mode="constant",
        intensity_constant=c,
        voxel_feat_mode="mean_all",
        tta_mode="rot4",
    )
    best_drop = dataclasses.replace(best, returns_mode="drop")

    # returns ablation on best
    for rm in ("as_is", "drop", "uniform", "const1"):
        add(dataclasses.replace(best, returns_mode=rm))

    # voxel size tweaks
    for vm in (0.8, 1.2):
        add(dataclasses.replace(best, voxel_size=base.voxel_size * vm))
        add(dataclasses.replace(best_drop, voxel_size=base.voxel_size * vm))

    # coord norm tweaks
    for cm in (0.8, 1.2):
        add(dataclasses.replace(best, coord_norm_factor=base.coord_norm_factor * cm))

    # z anisotropy tweaks
    for zs in (0.8, 1.2):
        add(dataclasses.replace(best, z_scale=zs))
        add(dataclasses.replace(best_drop, z_scale=zs))

    # -------------------------
    # Group D: Optional slow structural modes
    # -------------------------
    if include_slow:
        add(
            dataclasses.replace(
                base,
                intensity_mode="proxy_linearity",
                voxel_feat_mode="sample_max_linearity",
                tta_mode="none",
            )
        )
        add(
            dataclasses.replace(
                base,
                intensity_mode="proxy_linearity",
                voxel_feat_mode="sample_max_linearity",
                tta_mode="rot4",
            )
        )

    # De-dup deterministically by canonical spec
    seen = set()
    uniq: List[PreprocSpec] = []
    for s in runs:
        k = spec_key_from_spec(s)
        if k in seen:
            continue
        uniq.append(s)
        seen.add(k)

    return uniq


def make_decisive_returns_plan_minimal(
    base: PreprocSpec,
    *,
    c: float,
    has_ref: bool,
) -> List[PreprocSpec]:
    """
    Minimal decisive plan: ~12-16 runs.
    Goal: fast checkpoint comparison without missing the main failure modes.
    """
    runs: List[PreprocSpec] = []

    def add(s: PreprocSpec) -> None:
        runs.append(s)

    # 1) Baseline
    add(base)

    # 2) Baseline returns ablation (most common DALES generalization failure)
    add(dataclasses.replace(base, returns_mode="drop"))

    # 3) Best-known recipe (+ returns drop variant)
    best = dataclasses.replace(
        base,
        intensity_mode="constant",
        intensity_constant=c,
        voxel_feat_mode="mean_all",
        tta_mode="rot4",
    )
    add(best)
    add(dataclasses.replace(best, returns_mode="drop"))

    # 4) Constant intensity with training-like voxel rep (tests: only intensity mismatch)
    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="sample_first",
        )
    )
    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="sample_first",
            returns_mode="drop",
        )
    )

    # 5) Remove intensity entirely (tests: intensity collapse / corruption)
    add(dataclasses.replace(base, intensity_mode="zero", voxel_feat_mode="mean_all"))

    # 6) Two geometry-proxy intensities (tests: z-span/domain shift)
    add(dataclasses.replace(base, intensity_mode="proxy_z", voxel_feat_mode="mean_all"))
    add(
        dataclasses.replace(base, intensity_mode="minmax_z", voxel_feat_mode="mean_all")
    )

    # 7) Voxel rep stress tests (thin structures)
    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="sample_max_intensity",
        )
    )
    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="mean_thin",
            mean_thin_p=0.7,
        )
    )

    # 8) rot4 on baseline (sanity: view invariance helps even without other changes)
    add(dataclasses.replace(base, tta_mode="rot4"))

    # 9) Quantile match only if ref exists (optional but very high-signal)
    if has_ref:
        add(
            dataclasses.replace(
                base, intensity_mode="quantile_match", voxel_feat_mode="mean_all"
            )
        )
        add(
            dataclasses.replace(
                base,
                intensity_mode="quantile_match",
                voxel_feat_mode="mean_all",
                tta_mode="rot4",
            )
        )

    # De-dup deterministically
    seen = set()
    uniq: List[PreprocSpec] = []
    for s in runs:
        k = spec_key_from_spec(s)
        if k in seen:
            continue
        uniq.append(s)
        seen.add(k)

    return uniq


def make_decisive_plan(
    base: PreprocSpec,
    *,
    c: float,
    has_ref: bool,
    decisive_size: str,
    include_slow: bool,
) -> List[PreprocSpec]:
    if decisive_size == "minimal":
        # Slow modes deliberately excluded in minimal; they’re expensive and not required for checkpoint triage.
        return make_decisive_returns_plan_minimal(base, c=c, has_ref=has_ref)

    if decisive_size == "full":
        return make_decisive_returns_plan_full(
            base, c=c, has_ref=has_ref, include_slow=include_slow
        )

    raise ValueError(f"Unknown decisive_size: {decisive_size}")


# -------------------------
# Main
# -------------------------


def fit_lut_to_out_channels(
    pred_to_common: np.ndarray, out_channels: int, *, logger: logging.Logger
) -> np.ndarray:
    """Pad/trim LUT so pred_to_common.shape[0] == out_channels (model output classes)."""
    if pred_to_common.shape[0] == out_channels:
        return pred_to_common
    if pred_to_common.shape[0] < out_channels:
        padded = np.zeros((out_channels,), dtype=np.int64)
        padded[: pred_to_common.shape[0]] = pred_to_common
        logger.warning(
            f"[map] pred_to_common padded: {pred_to_common.shape[0]} -> {out_channels}"
        )
        return padded
    # larger than out_channels: trim (should be fine as long as model never outputs >= out_channels)
    logger.warning(
        f"[map] pred_to_common trimmed: {pred_to_common.shape[0]} -> {out_channels}"
    )
    return pred_to_common[:out_channels].astype(np.int64, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dales_root", required=True, help="Folder with DALES .las/.laz")
    ap.add_argument(
        "--ckpt",
        required=True,
        help="Torch checkpoint (must be a torch.nn.Module for this script)",
    )
    ap.add_argument("--mapping_dales_native_to_common", required=True)
    ap.add_argument("--mapping_eclair_native_to_common", required=True)
    ap.add_argument(
        "--mapping_pred_to_common",
        default=None,
        help="Optional: model output id -> common id YAML. "
        "Use if auto-derivation fails.",
    )
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--patch_size_m", type=float, default=100.0)
    ap.add_argument("--patch_stride_m", type=float, default=100.0)

    # baseline must match training
    ap.add_argument("--coord_norm_factor", type=float, default=10.0)
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--returns_k", type=int, default=5)
    ap.add_argument("--intensity_divisor", type=float, default=65535.0)

    ap.add_argument("--mode", default="staged", choices=["staged", "full"])
    ap.add_argument("--max_runs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--include_slow_modes",
        action="store_true",
        help="Include slow modes (proxy_linearity / sample_max_linearity). Default: off.",
    )
    ap.add_argument(
        "--pending_specs_json",
        default=None,
        help="Optional path to pending_specs.json. If set, run ONLY these specs (resume mode).",
    )

    # mapping derivation knobs
    ap.add_argument(
        "--try_import_label_maps",
        action="store_true",
        help="Try to import src.label_maps to derive pred->common.",
    )
    ap.add_argument("--eclair_undefined_id", type=int, default=0)
    ap.add_argument("--eclair_ignore_index", type=int, default=-100)

    # optional intensity reference (ECLAIR) for quantile matching
    ref_grp = ap.add_mutually_exclusive_group()
    ref_grp.add_argument(
        "--eclair_intensity_ref_npz",
        default=None,
        help="Optional .npz containing array 'intensity' in RAW uint16 scale, "
        "or 'intensity_0_1' already scaled. Enables quantile_match.",
    )
    ref_grp.add_argument(
        "--eclair_intensity_ref_json",
        default=None,
        help="Optional .json produced by compute_intensity_reference.py containing "
        "'probs' and 'quantiles' in [0,1]. Enables quantile_match.",
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Training YAML config used for the checkpoint/model.",
    )
    ap.add_argument(
        "--plan",
        required=True,
        choices=["sweep", "decisive_returns"],
        help="Which run plan to execute. sweep=default make_run_plan, decisive_returns only the ablations.",
    )
    ap.add_argument(
        "--decisive_size",
        default="minimal",
        choices=["minimal", "full"],
        help="Size of the decisive plan. minimal=~12-16 runs, full=~32-37 runs.",
    )

    args = ap.parse_args()
    if abs(float(args.patch_stride_m) - float(args.patch_size_m)) > 1e-6:
        raise RuntimeError(
            "patch_stride_m must equal patch_size_m unless you implement overlap fusion "
            "(otherwise points get double-counted and IoU is corrupted)."
        )

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    master = setup_logger(out_root, "preproc_sweep.master", level=logging.INFO)

    set_all_seeds(args.seed)
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    # load mappings
    map_dales = load_yaml_lut_fixed(
        args.mapping_dales_native_to_common, size=256, default=0
    )
    # Keep BOTH:
    # 1) dict for the simple fallback assumption native_id = train_id  1
    eclair_native_to_common_dict = load_yaml_map_int_dict(
        args.mapping_eclair_native_to_common
    )
    # 2) LUT for robust derivation through eclair_native_to_train_ids (import-based)
    eclair_native_to_common_lut = load_yaml_lut_fixed(
        args.mapping_eclair_native_to_common, size=256, default=0
    )

    # derive pred_to_common
    # model + amp + out_channels (from config)
    model, amp, out_channels = load_model_from_config(
        config_path=args.config,
        ckpt_path=args.ckpt,
        device=device,
    )

    # pred_to_common
    if args.mapping_pred_to_common is not None:
        pred_to_common = load_yaml_lut_fixed(
            args.mapping_pred_to_common, size=out_channels, default=0
        )
        master.info(
            f"[map] Using provided mapping_pred_to_common: {args.mapping_pred_to_common}"
        )
    else:
        pred_to_common = None
        if args.try_import_label_maps:
            pred_to_common = derive_pred_to_common_map(
                eclair_native_to_common_lut,
                try_import=True,
                undefined_id=int(args.eclair_undefined_id),
                ignore_index=int(args.eclair_ignore_index),
                logger=master,
            )
            if pred_to_common is not None:
                pred_to_common = fit_lut_to_out_channels(
                    pred_to_common, out_channels, logger=master
                )
                master.info(
                    f"[map] Derived pred_to_common via imported label maps (out_channels={out_channels})"
                )

        if pred_to_common is None:
            pred_to_common = build_pred_to_common_from_eclair_map(
                eclair_native_to_common=eclair_native_to_common_dict,
                out_channels=out_channels,
            )
            pred_to_common = fit_lut_to_out_channels(
                pred_to_common, out_channels, logger=master
            )
            master.info(
                f"[map] Fallback pred_to_common using native_id=train_id+1 (out_channels={out_channels})"
            )

    if pred_to_common is None:
        raise RuntimeError(
            "Could not construct model-output-id -> common mapping.\n"
            "Provide --mapping_pred_to_common YAML (output ids -> common ids), "
            "or run with --try_import_label_maps if your repo exposes src.label_maps.eclair_native_to_train_ids."
        )

    common_class_names = [
        "ignore",
        "ground",
        "vegetation",
        "buildings",
        "wires",
        "poles",
        "fence",
        "vehicle",
    ]
    C = len(common_class_names)

    # HARD VALIDATION: prevent silent clipping corruption
    if int(map_dales.max()) >= C:
        raise RuntimeError(
            f"mapping_dales_native_to_common outputs id {int(map_dales.max())} but C={C}. "
            f"Fix mapping or expand common_class_names."
        )
    if int(pred_to_common.max()) >= C:
        raise RuntimeError(
            f"pred_to_common outputs id {int(pred_to_common.max())} but C={C}. "
            f"Fix mapping_eclair_to_common / pred mapping."
        )

    # list DALES files
    droot = Path(args.dales_root)
    files = sorted(list(droot.rglob("*.las")) + list(droot.rglob("*.laz")))

    if not files:
        raise RuntimeError(f"No LAS/LAZ files found under {droot}")

    # optional quantile ref
    p_grid = None
    q_source = None
    has_ref = False
    median_inten = 0.5  # default if no ref provided
    if args.eclair_intensity_ref_json is not None:
        ref = json.loads(Path(args.eclair_intensity_ref_json).read_text())
        if ("probs" not in ref) or ("quantiles" not in ref):
            raise RuntimeError(
                "Intensity ref JSON must contain keys: 'probs', 'quantiles'"
            )
        p = np.asarray(ref["probs"], dtype=np.float32)
        q = np.asarray(ref["quantiles"], dtype=np.float32)
        median_inten = float(q[len(q) // 2])  # works for 1001 or any odd length
        if p.shape != q.shape or p.ndim != 1:
            raise RuntimeError(
                f"Invalid intensity ref shapes: probs={p.shape} quantiles={q.shape}"
            )
        # enforce sorted p_grid and valid ranges
        order = np.argsort(p)
        p = p[order]
        q = q[order]
        p_grid = np.clip(p, 0.0, 1.0).astype(np.float32, copy=False)
        q_source = np.clip(q, 0.0, 1.0).astype(np.float32, copy=False)
        has_ref = True
        master.info(
            f"[inten] Loaded ECLAIR intensity reference JSON from {args.eclair_intensity_ref_json} "
            f"(n={p_grid.size})"
        )

    elif args.eclair_intensity_ref_npz is not None:
        npz = np.load(args.eclair_intensity_ref_npz)
        if "intensity_0_1" in npz:
            src = np.asarray(npz["intensity_0_1"]).astype(np.float32, copy=False)
        elif "intensity" in npz:
            raw = np.asarray(npz["intensity"]).astype(np.float32, copy=False)
            src = np.clip(raw / float(args.intensity_divisor), 0.0, 1.0).astype(
                np.float32
            )
        else:
            raise RuntimeError(
                "eclair_intensity_ref_npz must contain 'intensity' or 'intensity_0_1'"
            )
        p_grid, q_source = build_quantile_lut(src, bins=4096)
        has_ref = True
        master.info(
            f"[inten] Loaded ECLAIR intensity reference from {args.eclair_intensity_ref_npz} (n={src.size})"
        )

    # baseline spec (MATCH TRAINING)
    base = PreprocSpec(
        coord_norm_factor=float(args.coord_norm_factor),
        voxel_size=float(args.voxel_size),
        z_scale=1.0,
        intensity_mode="as_is",
        intensity_constant=median_inten,
        returns_mode="as_is",
        returns_k=int(args.returns_k),
        intensity_divisor=float(args.intensity_divisor),
        voxel_feat_mode="sample_first",  # closest to your training cache behavior
        mean_thin_p=0.7,
        knn_k=16,
        tta_mode="none",
        patch_filter_mode="none",
        min_points=1000,
        min_occ_vox=1000,
        has_intensity_ref=bool(has_ref),
    )

    # -----------------------------
    # Resume mode: run only pending run_ids, but keep the ORIGINAL plan ordering/indices
    # -----------------------------
    pending_dicts = None
    if args.pending_specs_json is not None:
        pending_path = Path(args.pending_specs_json)
        pending_dicts = json.loads(pending_path.read_text())
        master.info(
            f"[plan] RESUME pending-only from {pending_path} n_specs={len(pending_dicts)}"
        )

    if args.plan == "decisive_returns":
        c = float(
            base.intensity_constant
        )  # ECLAIR median if REF_JSON is provided; else default
        # plan = [
        #     # 1) Baseline
        #     dataclasses.replace(
        #         base,
        #         intensity_mode="as_is",
        #         voxel_feat_mode="sample_first",
        #         tta_mode="none",
        #     ),
        #     # 1b) Baseline with returns removed
        #     dataclasses.replace(
        #         base,
        #         intensity_mode="as_is",
        #         voxel_feat_mode="sample_first",
        #         tta_mode="none",
        #         returns_mode="drop",
        #     ),
        #     # 2) Constant intensity (ECLAIR median), training-like voxel rep
        #     dataclasses.replace(
        #         base,
        #         intensity_mode="constant",
        #         intensity_constant=c,
        #         voxel_feat_mode="sample_first",
        #         tta_mode="none",
        #     ),
        #     # 2b) run_007 style, returns removed  ✅ IMPORTANT ADD
        #     dataclasses.replace(
        #         base,
        #         intensity_mode="constant",
        #         intensity_constant=c,
        #         voxel_feat_mode="sample_first",
        #         tta_mode="none",
        #         returns_mode="drop",
        #     ),
        #     # 3) Constant + mean_all
        #     dataclasses.replace(
        #         base,
        #         intensity_mode="constant",
        #         intensity_constant=c,
        #         voxel_feat_mode="mean_all",
        #         tta_mode="none",
        #     ),
        #     # 4) Constant + mean_all + rot4 (best)
        #     dataclasses.replace(
        #         base,
        #         intensity_mode="constant",
        #         intensity_constant=c,
        #         voxel_feat_mode="mean_all",
        #         tta_mode="rot4",
        #     ),
        #     # 4b) Best with returns removed
        #     dataclasses.replace(
        #         base,
        #         intensity_mode="constant",
        #         intensity_constant=c,
        #         voxel_feat_mode="mean_all",
        #         tta_mode="rot4",
        #         returns_mode="drop",
        #     ),
        #     dataclasses.replace(
        #         base,
        #         intensity_mode="constant",
        #         intensity_constant=c,
        #         voxel_feat_mode="sample_first",
        #         tta_mode="none",
        #         returns_mode="drop",
        #     ),
        #     dataclasses.replace(
        #     base,
        #     intensity_mode="constant",
        #     intensity_constant=c,
        #     voxel_feat_mode="mean_all",
        #     tta_mode="none",
        #     returns_mode="drop",
        # ),
        # ]

        plan = make_decisive_plan(
            base,
            c=c,
            has_ref=has_ref,
            decisive_size=args.decisive_size,
            include_slow=bool(args.include_slow_modes),
        )
    else:
        # Build plan, then gate invalid/slow modes deterministically
        plan = make_run_plan(base, mode=args.mode, max_runs=args.max_runs)
        gated: List[PreprocSpec] = []
        for s in plan:
            # quantile_match only if we actually have a ref
            if (s.intensity_mode == "quantile_match") and (not has_ref):
                continue
            # slow modes are opt-in
            if (not args.include_slow_modes) and (
                (s.intensity_mode == "proxy_linearity")
                or (s.voxel_feat_mode == "sample_max_linearity")
            ):
                continue
            gated.append(s)
        plan = gated[: args.max_runs]

    if args.plan == "decisive_returns":
        master.info(
            f"[plan] decisive_size={args.decisive_size} n_runs={len(plan)} device={device.type}"
        )
    else:
        master.info(f"[plan] mode={args.mode} n_runs={len(plan)} device={device.type}")
    master.info(f"[paths] dales_root={droot} ckpt={args.ckpt}")
    master.info(
        f"[baseline] coord_norm_factor={base.coord_norm_factor} voxel_size={base.voxel_size} returns_k={base.returns_k}"
    )

    pending_set = None
    if pending_dicts is not None:
        # Build plan index: canonical spec -> run_id
        plan_index: Dict[Tuple, str] = {}
        collisions = 0
        for s in plan:
            k = spec_key_from_spec(s)
            rid = s.to_id()
            if k in plan_index and plan_index[k] != rid:
                collisions += 1
            plan_index[k] = rid

        if collisions:
            master.warning(
                f"[resume] plan_index had {collisions} canonical collisions (unexpected, but continuing)."
            )

        pending_set = set()
        missing = []
        for d in pending_dicts:
            k = spec_key_from_dict(d)
            rid = plan_index.get(k)
            if rid is None:
                missing.append(d)
            else:
                pending_set.add(rid)

        master.info(
            f"[resume] pending_matched={len(pending_set)} pending_missing={len(missing)}"
        )

        # SANITY CHECK: log missing specs loudly (do not silently proceed)
        if missing:
            master.error(
                "[resume] Some pending specs did NOT match the current plan. "
                "This usually means your sweep generation changed, or spec fields differ."
            )
            # Print compact view
            for i, md in enumerate(missing, 1):
                master.error(f"[resume][missing {i}] {md}")
            raise RuntimeError(
                "Pending specs do not match the generated plan. "
                "Fix plan drift or regenerate pending_specs.json from this exact code version."
            )

    # run sweep
    results: Dict[str, Dict] = {}
    baseline_id = plan[0].to_id()

    for i, spec in enumerate(plan, 1):
        # ensure quantile_match marked correctly
        if spec.intensity_mode == "quantile_match" and not has_ref:
            # keep as-is but will be logged; still run (no hallucination)
            pass

        run_id = spec.to_id()

        # In resume mode, only execute pending run_ids, but preserve original i (folder numbering)
        if pending_set is not None and run_id not in pending_set:
            continue

        run_dir = out_root / f"run_{i:03d}_{run_id}"
        logger = setup_logger(run_dir, f"preproc_sweep.{run_id}", level=logging.INFO)
        (run_dir / "spec.json").write_text(
            json.dumps(dataclasses.asdict(spec), indent=2)
        )

        try:
            res = evaluate_run(
                model=model,
                dales_files=files,
                map_dales_to_common=map_dales,
                pred_to_common=pred_to_common,
                common_class_names=common_class_names,
                spec=spec,
                device=device,
                logger=logger,
                patch_size_m=args.patch_size_m,
                patch_stride_m=args.patch_stride_m,
                p_grid=p_grid,
                q_source=q_source,
                amp=amp,
            )
            (run_dir / "metrics.json").write_text(json.dumps(res, indent=2))
            results[run_id] = {"failed": False, **res}
        except Exception as e:
            logger.exception(f"[run] FAILED: {e}")
            results[run_id] = {
                "failed": True,
                "error": str(e),
                "spec": dataclasses.asdict(spec),
            }
            continue

        full = res["full"]
        master.info(
            f"[done] {i}/{len(plan)} run_id={run_id} full_miou={full['miou']:.4f} "
            f"wires={full['iou_per_class'][4]:.4f} poles={full['iou_per_class'][5]:.4f} "
            f"time={res['runtime_s']:.1f}s"
        )
        # run-level cleanup (important for long sweeps)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # -----------------------------
    # IMPORTANT: Always rebuild "results" from disk before final aggregation.
    # This makes resume mode write combined results.csv/json for ALL completed runs.
    # -----------------------------
    results = {}
    done = 0
    for rd in sorted(out_root.glob("run_[0-9][0-9][0-9]_*")):
        mp = rd / "metrics.json"
        if not mp.exists():
            continue
        try:
            res = json.loads(mp.read_text())
            rid = rd.name.split("_")[-1]
            results[rid] = {"failed": False, **res}
            done += 1
        except Exception:
            continue

    master.info(f"[reload] completed_from_disk={done}")

    # compute deltas vs baseline (FULL scope + FILTERED scope separately)
    baseline = results.get(baseline_id)
    if baseline is None or baseline.get("failed"):
        master.error("[baseline] missing/failed -> cannot compute deltas.")
        return

    b_full_miou = float(baseline["full"]["miou"])
    b_full_iou = np.array(baseline["full"]["iou_per_class"], dtype=np.float64)

    b_f_miou = float(baseline["filtered"]["miou"])
    b_f_iou = np.array(baseline["filtered"]["iou_per_class"], dtype=np.float64)

    rows = []
    for run_id, r in results.items():
        if r.get("failed"):
            continue

        full = r["full"]
        filt = r["filtered"]

        miou = float(full["miou"])
        iou = np.array(full["iou_per_class"], dtype=np.float64)
        d_miou = miou - b_full_miou
        d_iou = iou - b_full_iou

        miou_f = float(filt["miou"])
        iou_f = np.array(filt["iou_per_class"], dtype=np.float64)
        d_miou_f = miou_f - b_f_miou
        d_iou_f = iou_f - b_f_iou

        rows.append(
            {
                "run_id": run_id,
                "full_miou": miou,
                "full_miou_delta": float(d_miou),
                "full_wires": float(iou[4]),
                "full_wires_delta": float(d_iou[4]),
                "full_poles": float(iou[5]),
                "full_poles_delta": float(d_iou[5]),
                "full_fence": float(iou[6]),
                "full_fence_delta": float(d_iou[6]),
                "full_vehicle": float(iou[7]),
                "full_vehicle_delta": float(d_iou[7]),
                "filt_miou": miou_f,
                "filt_miou_delta": float(d_miou_f),
                "filt_wires_delta": float(d_iou_f[4]),
                "filt_poles_delta": float(d_iou_f[5]),
                "n_points_full": int(full["n_points"]),
                "n_patches_full": int(full["n_patches"]),
                "n_patches_kept": int(filt["n_patches_kept"]),
                "n_patches_dropped": int(filt["n_patches_dropped"]),
                "runtime_s": float(r["runtime_s"]),
                "spec": r["spec"],
                "align_avg": r.get("align_avg", {}),
            }
        )

    rows.sort(key=lambda x: x["full_miou_delta"], reverse=True)

    (out_root / "results.json").write_text(
        json.dumps({"baseline_id": baseline_id, "rows": rows}, indent=2)
    )

    header = (
        "run_id,full_miou,full_miou_delta,full_wires,full_wires_delta,full_poles,full_poles_delta,"
        "full_fence,full_fence_delta,full_vehicle,full_vehicle_delta,"
        "filt_miou,filt_miou_delta,filt_wires_delta,filt_poles_delta,"
        "n_points_full,n_patches_full,n_patches_kept,n_patches_dropped,runtime_s"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['run_id']},{r['full_miou']:.6f},{r['full_miou_delta']:.6f},"
            f"{r['full_wires']:.6f},{r['full_wires_delta']:.6f},"
            f"{r['full_poles']:.6f},{r['full_poles_delta']:.6f},"
            f"{r['full_fence']:.6f},{r['full_fence_delta']:.6f},"
            f"{r['full_vehicle']:.6f},{r['full_vehicle_delta']:.6f},"
            f"{r['filt_miou']:.6f},{r['filt_miou_delta']:.6f},"
            f"{r['filt_wires_delta']:.6f},{r['filt_poles_delta']:.6f},"
            f"{r['n_points_full']},{r['n_patches_full']},{r['n_patches_kept']},{r['n_patches_dropped']},"
            f"{r['runtime_s']:.3f}"
        )
    (out_root / "results.csv").write_text("\n".join(lines))

    master.info(f"[saved] {out_root / 'results.csv'}")
    master.info(f"[saved] {out_root / 'results.json'}")
    master.info("[done] scientific preproc sweep complete.")


if __name__ == "__main__":
    main()

# python - <<'PY'
# import numpy as np, laspy
# from pathlib import Path

# def clipped_stats(path, k=5):
#     las = laspy.read(str(path))
#     rn  = np.asarray(las.return_number)
#     nor = np.asarray(las.number_of_returns)
#     bad = np.mean(rn > nor)
#     if bad > 0:
#         print("WARN: rn>nor fraction", bad)


#     return {
#         "file": path.name,
#         "rn_max": int(rn.max()),
#         "nor_max": int(nor.max()),
#         "rn_clip_frac": float(np.mean(rn > k)),
#         "nor_clip_frac": float(np.mean(nor > k)),
#         "rn_uniques": np.unique(rn)[:20].tolist(),
#         "nor_uniques": np.unique(nor)[:20].tolist(),
#     }

# # DALES sample
# ps=[Path("/scratch/m23csa510/dales/dales/all/5080_54400.las"),
# Path("/scratch/m23csa510/dales/dales/all/5185_54390.las"),
# Path("/scratch/m23csa510/dales/dales/all/5190_54400.las"),
# Path("/csehome/m23csa510/lidar_experiments/datasets/eclair/pointclouds/pointcloud_102.laz"),
# Path("/csehome/m23csa510/lidar_experiments/datasets/eclair/pointclouds/pointcloud_1020.laz"),
# Path("/csehome/m23csa510/lidar_experiments/datasets/eclair/pointclouds/pointcloud_1022.laz"),]
# for i,p in enumerate(ps,1):
#     print("######")
#     print(clipped_stats(p, k=5))
#     print()
# PY
