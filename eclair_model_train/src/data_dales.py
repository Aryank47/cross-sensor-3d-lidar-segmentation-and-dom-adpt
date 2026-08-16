# src/data_dales.py
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import MinkowskiEngine as ME
import numpy as np
import torch

from .augment import AugmentConfig, augment_xyz
from .bev_head import BEVHeadConfig
from .bev_labels import build_bev_labels_and_selected_idx
from .features import FeatureConfig, build_features
from .mix3d import (
    MIX_SKIP_NO_DONOR,
    MIX_SKIP_OUTPUT_BUDGET,
    MIX_SKIP_PROBABILITY,
    ALSMix3DConfig,
    MixDiagnostics,
    choose_different_source_index,
    compose_crop_replace,
    should_apply_mix,
    stable_seed,
    unchanged_mix_result,
)
from .utils import atomic_save_torch, read_las_arrays_robust
from .voxelization import VoxelizationConfig, voxelize_from_q


def _build_label_lut(
    mapping: Mapping[int, int],
    *,
    ignore_index: int,
    max_label: int = 255,
) -> np.ndarray:
    """
    Build a LUT so mapping is O(1) and vectorized:
      lut[native_label] -> train_label or ignore_index
    Any label not present in mapping will become ignore_index by default.
    """
    lut = np.full((max_label + 1,), int(ignore_index), dtype=np.int64)
    for k, v in mapping.items():
        kk = int(k)
        vv = int(v)
        if 0 <= kk <= max_label:
            lut[kk] = vv
    return lut


@dataclass
class DalesPatchConfig:
    make_local_coords: bool = True
    coord_norm_factor: float = 10.0
    voxel_size: float = 0.05  # in normalized space, like ECLAIR


@dataclass
class DalesPreprocConfig:
    # intensity preprocessing (label-free)
    intensity_mode: str = "none"  # none | quantile_match | robust_standardize | constant
    intensity_constant: float = 0.5  # used if mode == constant
    # quantile match uses reference quantiles + target quantiles
    # both are arrays over probs in [0..1]
    ref_quantiles: Optional[np.ndarray] = None
    ref_probs: Optional[np.ndarray] = None
    tgt_quantiles: Optional[np.ndarray] = None
    tgt_probs: Optional[np.ndarray] = None

    # geometry/density preprocessing (label-free)
    # basic, cheap option: cap voxels per XY cell with height stratification
    use_height_xy_cap: bool = False
    xy_cell_size_m: float = 2.0  # in meters
    height_bins: int = 5
    caps_per_bin: Tuple[int, ...] = (
        4000,
        3000,
        2500,
        2500,
        2500,
    )  # low->high (example)
    intensity_divisor_override: Optional[float] = None
    # ---- Memory safety ----
    # Hard cap on number of unique voxels per sample AFTER sparse_quantize.
    # If exceeded, we randomly subsample voxels (label-preserving, label-free).
    # If set, cap number of unique voxels in TILE MODE (debug/eval) after voxelization.
    # Default None to avoid silent voxel dropping.
    max_voxels: Optional[int] = None

    # NEW: make voxel cap class-aware to protect rare classes.
    # When True, we keep all voxels belonging to rare_class_ids (if they fit),
    # and only downsample the rest.
    class_aware_max_voxels: bool = False
    # Train IDs of rare classes you want to protect.
    # For your mapping and class_names:
    #   0 ground, 1 vegetation, 2 cars, 3 trucks, 4 buildings,
    #   5 poles, 6 power_lines, 7 fences
    rare_class_ids: Tuple[int, ...] = (2, 3, 5, 6, 7)


@dataclass
class DalesCropConfig:
    # Primary crop size (meters)
    crop_size_xy_m: float = 20.0
    crop_min_points: int = 2000
    crops_per_tile_per_epoch: int = 8

    # Rare-aware center sampling
    rare_center_prob: float = 0.5
    # If empty, we fall back to preproc.rare_class_ids
    rare_center_class_ids: Tuple[int, ...] = (5, 6)  # poles, power_lines by default

    # OOM safety
    max_voxels_per_crop_soft: int = 120000  # target budget; shrink crop until we meet this
    max_voxels_per_crop_hard: int = 160000  # hard budget (last-resort fallback)
    resample_tries: int = 25

    # Adaptive shrinking (instead of voxel dropping)
    min_crop_size_xy_m: float = 8.0
    shrink_ratio: float = 0.8
    max_shrink_steps: int = 5

    # Very last resort: rare-preserving point subsample if STILL too big
    hard_max_points: int = 200000

    # Optional early guard (saves time by shrinking before voxelizing)
    max_points_per_crop_soft: Optional[int] = None

    # NEW: optional class-weighted rare centering (train-id -> weight)
    # Example: {6: 3.0, 5: 2.0, 3: 0.25}
    rare_center_class_weights: Optional[Dict[int, float]] = None
    # NEW: optional frequency compensation inside rare set:
    # p(class) ∝ weight / (count ** beta). beta=0 means "ignore counts".
    rare_center_balance_beta: float = 0.0

    # NEW: bundle multiple crops from the same tile into one __getitem__ call.
    # This avoids re-loading the same LAS/LAZ and redoing per-tile setup for each crop.
    # Effective crops/epoch is still crops_per_tile_per_epoch, but each tile is loaded
    # only ceil(crops_per_tile_per_epoch / crops_per_item) times.
    crops_per_item: int = 1


def _points_to_sparse(
    *,
    xyz_m: np.ndarray,
    raw: Dict[str, np.ndarray],
    point_idx: np.ndarray,
    patch_cfg: DalesPatchConfig,
    feat_cfg: FeatureConfig,
    ignore_index: int,
    label_lut: np.ndarray,
    voxel_cfg: VoxelizationConfig,
    rng: Optional[np.random.Generator] = None,
    crop_center_xy_m: Optional[Tuple[float, float]] = None,
    do_aug: bool = False,
    aug_cfg: Optional[AugmentConfig] = None,
    return_prepared: bool = False,
) -> Dict[str, Any]:
    xyz = xyz_m[point_idx].astype(np.float64, copy=False)

    # --- LiDOG-faithful crop centering (XY only): center crop at (0,0) ---
    # This is a rigid translation; it does NOT distort geometry.
    if crop_center_xy_m is not None:
        cx, cy = float(crop_center_xy_m[0]), float(crop_center_xy_m[1])
        xyz = xyz.copy()
        xyz[:, 0] -= cx
        xyz[:, 1] -= cy

    # Apply augmentation ONLY on the crop points (much faster than augmenting the full tile)
    if do_aug and (aug_cfg is not None) and bool(aug_cfg.enabled):
        if rng is None:
            rng = np.random.default_rng()
        xyz = augment_xyz(xyz.astype(np.float32, copy=False), aug_cfg, rng)

    xyz_norm = (xyz / float(patch_cfg.coord_norm_factor)).astype(np.float32, copy=False)

    intensity = raw.get("intensity", None)
    intensity_scaled = None
    if feat_cfg.use_intensity and intensity is not None:
        # Support either raw [0..65535] or pre-scaled [0..1] intensity.
        intensity_crop = intensity[point_idx].astype(np.float32, copy=False)
        if float(np.max(intensity_crop)) <= 1.0 + 1e-3:
            intensity_scaled = intensity_crop
        else:
            div = float(feat_cfg.intensity_divisor)
            intensity_scaled = intensity_crop / div
        intensity_scaled = np.clip(intensity_scaled, 0.0, 1.0).astype(np.float32, copy=False)

    feats = build_features(
        xyz_local=xyz_norm,
        intensity=intensity_scaled,
        return_number=raw["return_number"][point_idx],
        number_of_returns=raw["number_of_returns"][point_idx],
        rgb=None,
        cfg=feat_cfg,
    )

    # quantize coords
    q = np.floor(xyz_norm / float(patch_cfg.voxel_size)).astype(np.int32, copy=False)
    q = np.ascontiguousarray(q, dtype=np.int32)

    # per-point labels in TRAIN-ID space
    y_native_pts = raw["native_labels"][point_idx].astype(np.int64, copy=False)
    y_safe_pts = np.clip(y_native_pts, 0, 255).astype(np.int64, copy=False)
    y_train_pts = label_lut[y_safe_pts].astype(np.int64, copy=False)

    if rng is None:
        rng = np.random.default_rng()
    vx = voxelize_from_q(
        q_int32=q,
        feats_p_f32=feats.astype(np.float32, copy=False),
        labels_p_i64=y_train_pts,
        ignore_index=int(ignore_index),
        cfg=voxel_cfg,
        rng=rng,
        return_maps=False,
        num_classes_hint=None,
    )
    # NOTE: we will pass voxel_cfg from dataset (see below), not via patch_cfg

    coords_t = torch.from_numpy(np.ascontiguousarray(vx["coords_u"], dtype=np.int32)).int()
    feats_t = torch.from_numpy(np.ascontiguousarray(vx["feats_u"], dtype=np.float32)).float()
    labels_t = torch.from_numpy(np.ascontiguousarray(vx["labels_u"], dtype=np.int64)).long()
    out = {"coords": coords_t, "feats": feats_t, "labels": labels_t}
    if return_prepared:
        out["_prepared"] = {
            "xyz": np.ascontiguousarray(xyz),
            "y_train": np.ascontiguousarray(y_train_pts, dtype=np.int64),
            "intensity": (None if intensity_scaled is None else np.ascontiguousarray(intensity_scaled, dtype=np.float32)),
            "return_number": np.ascontiguousarray(raw["return_number"][point_idx], dtype=np.int64),
            "number_of_returns": np.ascontiguousarray(raw["number_of_returns"][point_idx], dtype=np.int64),
            "rgb": None,
        }
    return out


def _pick_center_index(
    *,
    y_train_pts: np.ndarray,
    ignore_index: int,
    crop_cfg: DalesCropConfig,
    fallback_rare_ids: Tuple[int, ...],
    rng: np.random.Generator,
) -> Tuple[int, bool]:
    """Pick a center point index, optionally biased towards rare classes."""
    n = y_train_pts.shape[0]
    if n == 0:
        return 0, False

    rare_ids = crop_cfg.rare_center_class_ids
    if rare_ids is None or len(rare_ids) == 0:
        rare_ids = fallback_rare_ids

    use_rare = rng.random() < float(crop_cfg.rare_center_prob)
    if use_rare:
        mask_rare = (y_train_pts != ignore_index) & np.isin(y_train_pts, np.asarray(rare_ids, dtype=np.int64))
        cand = np.where(mask_rare)[0]
        if cand.size > 0:
            # If no weights provided: keep current behavior
            wmap = getattr(crop_cfg, "rare_center_class_weights", None)
            if not wmap:
                return int(rng.choice(cand)), True

            # Build class list present in candidate points
            y_cand = y_train_pts[cand]
            classes, counts = np.unique(y_cand, return_counts=True)

            # Compute weights per class (default 1.0 if not specified)
            w = np.array([float(wmap.get(int(c), 1.0)) for c in classes], dtype=np.float64)
            beta = float(getattr(crop_cfg, "rare_center_balance_beta", 0.0))
            if beta != 0.0:
                w = w / np.maximum(1.0, counts.astype(np.float64) ** beta)

            w = np.clip(w, 0.0, None)
            if not np.isfinite(w).all() or w.sum() <= 0:
                return int(rng.choice(cand)), True

            p = w / w.sum()
            chosen_class = int(rng.choice(classes.astype(np.int64), p=p))
            cand_cls = cand[y_cand == chosen_class]
            return int(rng.choice(cand_cls)), True

    return int(rng.integers(0, n)), False


def _points_in_xy_square(
    xyz_m: np.ndarray,
    *,
    cx: float,
    cy: float,
    half: float,
) -> np.ndarray:
    """Return indices of points inside an axis-aligned square crop in XY."""
    m = (np.abs(xyz_m[:, 0] - cx) <= half) & (np.abs(xyz_m[:, 1] - cy) <= half)
    return np.where(m)[0]


def _subsample_points_keep_rare(
    *,
    idx: np.ndarray,
    y_train_pts: np.ndarray,
    ignore_index: int,
    rare_ids: Tuple[int, ...],
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Last resort only.
    Keep all rare points if possible, then fill with random common points.
    """
    if idx.size <= max_points:
        return idx

    y = y_train_pts[idx]
    rare_mask = (y != ignore_index) & np.isin(y, np.asarray(rare_ids, dtype=np.int64))
    rare_idx = idx[rare_mask]
    common_idx = idx[~rare_mask]

    if rare_idx.size >= max_points:
        keep = rng.choice(rare_idx, size=max_points, replace=False)
        keep.sort()
        return keep

    remaining = max_points - rare_idx.size
    if common_idx.size <= remaining:
        keep = np.concatenate([rare_idx, common_idx])
        keep.sort()
        return keep

    common_keep = rng.choice(common_idx, size=remaining, replace=False)
    keep = np.concatenate([rare_idx, common_keep])
    keep.sort()
    return keep


def _find_dales_files(root: Union[str, Path]) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"DALES root not found: {root}")
    # recursively find LAS/LAZ
    files = sorted(list(root.rglob("*.las")) + list(root.rglob("*.laz")))
    if not files:
        raise RuntimeError(f"No .las/.laz files found under: {root}")
    return files


def _read_dales_las(path: Path) -> Dict[str, np.ndarray]:
    """
    Robust reader wrapper for DALES training.
    """
    # The robust reader returns keys: xyz, intensity, return_number, number_of_returns, native_labels
    # This matches your old _read_dales_las output structure perfectly.
    return read_las_arrays_robust(path)


def _quantile_match(
    x: np.ndarray,
    ref_q: np.ndarray,
    ref_p: np.ndarray,
    tgt_q: np.ndarray,
    tgt_p: np.ndarray,
) -> np.ndarray:
    """
    Map x by matching target CDF -> source CDF using quantiles.
    We approximate:
        p = F_T(x)   via interp on (tgt_q, tgt_p)
        x' = F_S^{-1}(p) via interp on (ref_p, ref_q)
    """
    x = x.astype(np.float32, copy=False)
    # clamp to target quantile range to avoid extreme extrapolation
    x_clamped = np.clip(x, tgt_q[0], tgt_q[-1])
    p = np.interp(x_clamped, tgt_q, tgt_p)
    x_mapped = np.interp(p, ref_p, ref_q)
    return x_mapped.astype(np.float32, copy=False)


def _robust_standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    med = np.median(x)
    q1 = np.quantile(x, 0.25)
    q3 = np.quantile(x, 0.75)
    iqr = max(float(q3 - q1), eps)
    z = (x - med) / iqr
    # rescale into [0,1]-ish range (optional); here keep as standardized
    return z.astype(np.float32, copy=False)


def _height_xy_cap(
    xyz_m: np.ndarray,
    feats: np.ndarray,
    labels: np.ndarray,
    *,
    xy_cell_size_m: float,
    height_bins: int,
    caps_per_bin: Sequence[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Label-free thinning AFTER voxel unique selection.
    Works on per-voxel representatives (xyz_m corresponds to those reps).
    """
    assert xyz_m.ndim == 2 and xyz_m.shape[1] == 3
    z = xyz_m[:, 2].astype(np.float32)
    # build height bin edges by quantiles (robust to scale shifts)
    qs = np.linspace(0.0, 1.0, height_bins + 1)
    edges = np.quantile(z, qs)
    # ensure monotone edges
    edges[0] -= 1e-3
    edges[-1] += 1e-3

    x_cell = np.floor(xyz_m[:, 0] / float(xy_cell_size_m)).astype(np.int32)
    y_cell = np.floor(xyz_m[:, 1] / float(xy_cell_size_m)).astype(np.int32)

    keep = np.zeros((xyz_m.shape[0],), dtype=bool)

    for b in range(height_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (z >= lo) & (z < hi)
        idx = np.where(m)[0]
        if idx.size == 0:
            continue
        cap = int(caps_per_bin[b]) if b < len(caps_per_bin) else int(caps_per_bin[-1])

        # group by XY cell
        keys = (x_cell[idx].astype(np.int64) << 32) ^ (y_cell[idx].astype(np.int64) & 0xFFFFFFFF)
        # shuffle within bin for random selection
        perm = rng.permutation(idx.size)
        idx_shuf = idx[perm]
        keys_shuf = keys[perm]

        # stable sort by key to scan groups
        order = np.argsort(keys_shuf, kind="mergesort")
        idx_sorted = idx_shuf[order]
        keys_sorted = keys_shuf[order]

        start = 0
        while start < idx_sorted.size:
            k = keys_sorted[start]
            end = start + 1
            while end < idx_sorted.size and keys_sorted[end] == k:
                end += 1
            group = idx_sorted[start:end]
            if group.size <= cap:
                keep[group] = True
            else:
                chosen = rng.choice(group, size=cap, replace=False)
                keep[chosen] = True
            start = end

    # if thinning is too aggressive, fallback to keeping all
    if keep.sum() < max(1000, xyz_m.shape[0] // 10):
        return xyz_m, feats, labels

    return xyz_m[keep], feats[keep], labels[keep]


def _sha1_hex(x: bytes) -> str:
    return hashlib.sha1(x).hexdigest()


def _hash_np(a: Optional[np.ndarray]) -> Optional[str]:
    if a is None:
        return None
    aa = np.ascontiguousarray(a)
    return _sha1_hex(aa.view(np.uint8).tobytes())


def _cache_key_for_dales(
    *,
    path: Path,
    patch_cfg: DalesPatchConfig,
    feat_cfg: FeatureConfig,
    preproc: DalesPreprocConfig,
    cache_key_extra: Optional[str],
    ignore_index: int,
    label_map: Optional[Dict[int, int]],
    voxel_cfg: VoxelizationConfig,
) -> str:
    """
    Stable cache key for a DALES file given feature/voxelization/preproc settings.
    NOTE: includes preproc array *digests* (not full arrays) to keep key compact.
    """
    key_obj = {
        "v": "dales_cache_v2",
        "path": str(path.resolve()),
        "patch": {
            "make_local_coords": bool(patch_cfg.make_local_coords),
            "coord_norm_factor": float(patch_cfg.coord_norm_factor),
            "voxel_size": float(patch_cfg.voxel_size),
        },
        "features": {
            "use_intensity": bool(feat_cfg.use_intensity),
            "intensity_divisor": float(feat_cfg.intensity_divisor),
            "returns_onehot_k": int(feat_cfg.returns_onehot_k),
            "use_rgb": bool(feat_cfg.use_rgb),
            "include_coords": bool(feat_cfg.include_coords),
        },
        "preproc": {
            "intensity_mode": str(preproc.intensity_mode),
            "intensity_constant": float(preproc.intensity_constant),
            "intensity_divisor_override": (
                None if preproc.intensity_divisor_override is None else float(preproc.intensity_divisor_override)
            ),
            "use_height_xy_cap": bool(preproc.use_height_xy_cap),
            "xy_cell_size_m": float(preproc.xy_cell_size_m),
            "height_bins": int(preproc.height_bins),
            "caps_per_bin": [int(x) for x in preproc.caps_per_bin],
            "ref_quantiles_sha1": _hash_np(preproc.ref_quantiles),
            "ref_probs_sha1": _hash_np(preproc.ref_probs),
            "tgt_quantiles_sha1": _hash_np(preproc.tgt_quantiles),
            "tgt_probs_sha1": _hash_np(preproc.tgt_probs),
            "max_voxels": (None if preproc.max_voxels is None else int(preproc.max_voxels)),
            "class_aware_max_voxels": bool(preproc.class_aware_max_voxels),
            "rare_class_ids": [int(x) for x in preproc.rare_class_ids],
        },
        "extra": cache_key_extra,
        "labels": {
            "ignore_index": int(ignore_index),
            "label_policy": "native_to_train_lut",
        },
        "label_map": (None if label_map is None else dict(sorted((int(k), int(v)) for k, v in label_map.items()))),
        "voxelization": {
            "feat_pool": str(voxel_cfg.feat_pool),
            "label_pool": str(voxel_cfg.label_pool),
            "random_seed_offset": int(voxel_cfg.random_seed_offset),
        },
    }
    s = json.dumps(key_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha1_hex(s)[:24]


def _cache_key_for_dales_raw(*, path: Path) -> str:
    st = path.stat()
    key_obj = {
        "v": "dales_raw_cache_v2",
        "path": str(path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    s = json.dumps(key_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha1_hex(s)[:24]


def _ensure_raw_dtypes(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    # CRITICAL: keep xyz as float64 until after localizing/centering.
    out["xyz"] = raw["xyz"].astype(np.float64, copy=False)

    out["intensity"] = raw["intensity"].astype(np.float32, copy=False) if raw.get("intensity", None) is not None else None

    out["return_number"] = raw["return_number"].astype(np.int64, copy=False)
    out["number_of_returns"] = raw["number_of_returns"].astype(np.int64, copy=False)
    out["native_labels"] = raw["native_labels"].astype(np.int64, copy=False)

    # Keep rgb if present (prevents silent breakage if use_rgb ever enabled)
    out["rgb"] = raw["rgb"].astype(np.float32, copy=False) if raw.get("rgb", None) is not None else None

    return out


def _cap_voxels(
    coords_u: np.ndarray,
    feats_u: np.ndarray,
    labels_u: np.ndarray,
    *,
    max_voxels: int,
    ignore_index: int,
    class_aware: bool,
    rare_ids: Sequence[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(coords_u.shape[0])
    if max_voxels <= 0 or n <= max_voxels:
        return coords_u, feats_u, labels_u

    if not class_aware:
        keep = rng.choice(n, size=max_voxels, replace=False)
        keep.sort()
        return coords_u[keep], feats_u[keep], labels_u[keep]

    rare_ids_arr = np.asarray(list(rare_ids), dtype=np.int64)
    valid = labels_u != int(ignore_index)
    is_rare = valid & np.isin(labels_u, rare_ids_arr)
    rare_idx = np.where(is_rare)[0]
    common_idx = np.where(~is_rare)[0]

    if rare_idx.size >= max_voxels:
        keep = rng.choice(rare_idx, size=max_voxels, replace=False)
        keep.sort()
        return coords_u[keep], feats_u[keep], labels_u[keep]

    remaining = max_voxels - rare_idx.size
    if common_idx.size <= remaining:
        keep = np.concatenate([rare_idx, common_idx])
        keep.sort()
        return coords_u[keep], feats_u[keep], labels_u[keep]

    common_keep = rng.choice(common_idx, size=remaining, replace=False)
    keep = np.concatenate([rare_idx, common_keep])
    keep.sort()
    return coords_u[keep], feats_u[keep], labels_u[keep]


class DalesTiles(torch.utils.data.Dataset):
    """
    Each LAS/LAZ file is treated as one sample (tile/patch).
    Outputs ME-ready (coords, feats, labels) after voxel quantization and unique selection.
    """

    def __init__(
        self,
        *,
        dales_root: str | Path,
        patch_cfg: DalesPatchConfig,
        feat_cfg: FeatureConfig,
        aug_cfg: AugmentConfig,
        is_train: bool,
        ignore_index: int = -100,
        preproc: Optional[DalesPreprocConfig] = None,
        seed: int = 1234,
        use_cache: bool = True,
        cache_root: Optional[str | Path] = None,
        cache_subdir: str = "dales",
        cache_key_extra: Optional[str] = None,
        require_cache: bool = False,
        write_cache: bool = False,
        split_name: Optional[str] = None,
        files: Optional[List[Path]] = None,
        label_map: Optional[Dict[int, int]] = None,
        cache_kind: str = "voxel",  # "voxel" (old) or "raw" (new)
        # NEW: crop sampling (training-time)
        sampling_mode: str = "tiles",  # "tiles" | "crops"
        crop_cfg: DalesCropConfig = DalesCropConfig(),
        voxel_cfg: Optional[VoxelizationConfig] = None,
        bev_cfg: Optional[BEVHeadConfig] = None,
        mix3d_cfg: Optional[ALSMix3DConfig] = None,
    ):
        self.root = Path(dales_root)
        if files is not None:
            # Caller explicitly provides split file list
            self.files = list(files)
        else:
            # Default: recursively find LAS/LAZ under root
            self.files = _find_dales_files(self.root)

        self.patch_cfg = patch_cfg
        self.feat_cfg = feat_cfg
        self.aug_cfg = aug_cfg
        self.ignore_index = ignore_index
        self.preproc = preproc or DalesPreprocConfig()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.use_cache = bool(use_cache)
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.cache_subdir = str(cache_subdir)
        self.cache_key_extra = cache_key_extra
        self.require_cache = bool(require_cache)
        self.write_cache = bool(write_cache)
        self.split_name = split_name
        self.label_lut = None
        self.label_map = None if label_map is None else {int(k): int(v) for k, v in label_map.items()}

        if label_map is not None:
            self.label_lut = _build_label_lut(label_map, ignore_index=self.ignore_index, max_label=255)
        else:
            # This pipeline always expects labels (train/val/test). Make it explicit to avoid silent failures.
            raise ValueError("DalesTiles requires label_map (native->train) to build label_lut.")

        if self.use_cache and self.cache_root is None:
            raise ValueError("DalesTiles: use_cache=True requires cache_root to be set.")

        self.is_train = is_train

        self.cache_kind = str(cache_kind).lower().strip()

        if self.use_cache:
            base = self.cache_root / self.cache_subdir
            if self.cache_kind == "raw":
                base = base / "raw"
            else:
                base = base / "voxel"
            if self.split_name:
                base = base / self.split_name
            base.mkdir(parents=True, exist_ok=True)
            self._cache_dir = base

        if self.cache_kind not in ("voxel", "raw"):
            raise ValueError(f"Unknown cache_kind={cache_kind}, expected 'voxel' or 'raw'")

        # ---- Safety gates (prevent silent training bugs) ----
        if self.is_train and self.write_cache:
            raise ValueError(
                "DalesTiles: write_cache=True during training is unsafe (multi-worker races). "
                "Precompute caches offline; use write_cache=false in training."
            )

        if self.is_train and bool(self.aug_cfg.enabled) and self.cache_kind != "raw":
            raise ValueError(
                "DalesTiles: aug.enabled=true requires cache_kind='raw' so we can do raw->augment->voxelize at runtime. "
                "Voxel-cache can bypass augmentation by returning cached tensors."
            )

        self.sampling_mode = str(sampling_mode).lower().strip()
        self.crop_cfg = crop_cfg
        self._epoch_shared = mp.Value("q", 0, lock=False)
        self.voxel_cfg = voxel_cfg or VoxelizationConfig()
        self.bev_cfg = bev_cfg
        self.mix3d_cfg = mix3d_cfg or ALSMix3DConfig(enabled=False)

        # Helpful safety: voxel-cache cannot apply per-epoch augmentation correctly.
        if self.is_train and self.sampling_mode == "crops":
            if self.crop_cfg is None:
                raise ValueError("sampling_mode='crops' requires crop_cfg.")
            if self.cache_kind != "raw":
                raise ValueError("sampling_mode='crops' requires cache_kind='raw' (raw point cache).")
            if self.label_lut is None:
                raise ValueError("sampling_mode='crops' requires label_map (label_lut).")

        if self.mix3d_cfg.enabled:
            if not self.is_train or self.sampling_mode != "crops":
                raise ValueError("DALES M1 requires the training split with sampling.mode='crops'.")
            if int(getattr(self.crop_cfg, "crops_per_item", 1)) != 1:
                raise ValueError("Initial DALES M1 requires sampling.crops_per_item=1.")
            if self.bev_cfg is not None and self.bev_cfg.enabled:
                raise ValueError("Initial M1 forbids BEV auxiliary supervision.")

    def _stable_seed(self, *, path: Path, tile_i: int, rep_i: int) -> int:
        h = zlib.crc32(path.name.encode("utf-8")) & 0xFFFFFFFF

        # val/test may not have repetition indexing; allow None safely.
        ti = int(tile_i) if tile_i is not None else 0
        ri = int(rep_i) if rep_i is not None else 0
        ep = int(self.epoch) if getattr(self, "epoch", None) is not None else 0

        s = (self.seed * 1000003 + ep * 9176 + ti * 101 + ri * 17 + h) & 0x7FFFFFFF
        return int(s)

    def set_epoch(self, epoch: int) -> None:
        """Call from training loop so crop RNG changes each epoch (important with DDP)."""
        self._epoch_shared.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch_shared.value)

    def __len__(self) -> int:
        if self.is_train and self.sampling_mode == "crops":
            k = max(1, int(self.crop_cfg.crops_per_tile_per_epoch))
            m = max(1, int(getattr(self.crop_cfg, "crops_per_item", 1)))
            groups = (k + m - 1) // m
            return len(self.files) * groups
        return len(self.files)

    def get_raw(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Return raw arrays for a tile (NO augmentation, NO preprocessing).
        Used by point-wise eval / window inference.
        """
        path = self.files[idx]

        raw_from_cache: Optional[Dict[str, np.ndarray]] = None
        if self.use_cache and self.cache_kind == "raw":
            key = _cache_key_for_dales_raw(path=path)
            cache_path = self._cache_dir / f"{key}.pt"
            if cache_path.exists():
                obj = torch.load(cache_path, map_location="cpu")
                if not isinstance(obj, dict) or "xyz" not in obj:
                    raise TypeError(f"Bad raw cache payload: {cache_path}")
                raw_from_cache = _ensure_raw_dtypes(obj)
            elif self.require_cache:
                raise RuntimeError(f"[DALES raw cache missing] {cache_path}")

        raw = raw_from_cache if raw_from_cache is not None else _ensure_raw_dtypes(_read_dales_las(path))

        xyz = raw["xyz"]

        # --- Self-policing invariant: raw xyz must remain float64 until localized ---
        if xyz.dtype != np.float64:
            msg = (
                f"[DALES get_raw] expected raw['xyz'] dtype=float64, got {xyz.dtype}. "
                f"cache_kind={self.cache_kind} use_cache={self.use_cache} require_cache={self.require_cache} "
                f"path={path}"
            )
            # log + raise (raise is deliberate to avoid silent scientific drift)
            try:
                import logging

                logging.getLogger(__name__).error(msg)
            except Exception:
                pass
            raise TypeError(msg)

        if self.patch_cfg.make_local_coords:
            xyz = xyz - xyz.min(axis=0, keepdims=True)

        out = dict(raw)
        out["xyz"] = xyz
        out["path"] = str(path)
        return out

    def _load_mix_tile(self, path: Path):
        """Load one DALES source file exactly through the canonical raw-cache path."""
        raw_from_cache = None
        if self.use_cache:
            if self.cache_kind != "raw":
                raise ValueError("DALES M1 requires cache_kind='raw'.")
            key = _cache_key_for_dales_raw(path=path)
            cache_path = self._cache_dir / f"{key}.pt"
            if cache_path.exists():
                obj = torch.load(cache_path, map_location="cpu")
                if not isinstance(obj, dict) or "xyz" not in obj:
                    raise TypeError(f"Bad raw cache payload: {cache_path}")
                raw_from_cache = _ensure_raw_dtypes(obj)
            elif self.require_cache:
                raise RuntimeError(f"[DALES raw cache missing] {cache_path}")

        raw = raw_from_cache if raw_from_cache is not None else _ensure_raw_dtypes(_read_dales_las(path))
        raw = _ensure_raw_dtypes(raw)
        xyz = raw["xyz"]
        if self.patch_cfg.make_local_coords:
            xyz = xyz - xyz.min(axis=0, keepdims=True)

        intensity_scaled = None
        if self.feat_cfg.use_intensity:
            intensity = raw.get("intensity", None)
            if intensity is None:
                raise RuntimeError("DALES: feat_cfg.use_intensity=True but raw intensity is missing/None.")
            intensity = intensity.astype(np.float32, copy=False)
            div = (
                float(self.preproc.intensity_divisor_override)
                if self.preproc.intensity_divisor_override
                else float(self.feat_cfg.intensity_divisor)
            )
            intensity_scaled = intensity if intensity.max() <= 1.0 + 1e-3 else intensity / div
            mode = (self.preproc.intensity_mode or "none").lower()
            if mode == "none":
                pass
            elif mode == "constant":
                intensity_scaled = np.full_like(
                    intensity_scaled,
                    float(self.preproc.intensity_constant),
                    dtype=np.float32,
                )
            elif mode == "robust_standardize":
                z = _robust_standardize(intensity_scaled)
                intensity_scaled = 1.0 / (1.0 + np.exp(-z))
            elif mode == "quantile_match":
                if (
                    self.preproc.ref_quantiles is None
                    or self.preproc.ref_probs is None
                    or self.preproc.tgt_quantiles is None
                    or self.preproc.tgt_probs is None
                ):
                    raise RuntimeError("quantile_match requires ref_quantiles/ref_probs and tgt_quantiles/tgt_probs")
                intensity_scaled = _quantile_match(
                    intensity_scaled,
                    self.preproc.ref_quantiles,
                    self.preproc.ref_probs,
                    self.preproc.tgt_quantiles,
                    self.preproc.tgt_probs,
                )
            else:
                raise ValueError(f"Unknown intensity_mode: {self.preproc.intensity_mode}")
            intensity_scaled = np.clip(intensity_scaled, 0.0, 1.0).astype(np.float32, copy=False)
        return raw, xyz, intensity_scaled

    def _sample_mix_crop(
        self,
        *,
        path: Path,
        tile_i: int,
        rep_i: int,
        raw: Dict[str, np.ndarray],
        xyz: np.ndarray,
        intensity_scaled: Optional[np.ndarray],
        sample_seed_override: Optional[int] = None,
    ):
        """Canonical DALES crop selection, returning accepted post-augmentation points."""
        y_native_pts = raw["native_labels"].astype(np.int64, copy=False)
        y_safe_pts = np.clip(y_native_pts, 0, 255).astype(np.int64, copy=False)
        y_train_pts = self.label_lut[y_safe_pts]

        crop_cfg = self.crop_cfg
        max_vox_soft = int(crop_cfg.max_voxels_per_crop_soft)
        max_vox_hard = int(getattr(crop_cfg, "max_voxels_per_crop_hard", max_vox_soft))
        if max_vox_hard < max_vox_soft:
            raise ValueError(f"max_voxels_per_crop_hard ({max_vox_hard}) must be >= soft ({max_vox_soft}).")
        hard_max_points = int(getattr(crop_cfg, "hard_max_points", 0) or 0)
        if hard_max_points and hard_max_points < int(crop_cfg.crop_min_points):
            raise ValueError("hard_max_points must be >= crop_min_points.")

        min_size = float(crop_cfg.min_crop_size_xy_m)
        base_size = float(crop_cfg.crop_size_xy_m)
        rare_ids = crop_cfg.rare_center_class_ids
        if rare_ids is None or len(rare_ids) == 0:
            rare_ids = tuple(self.preproc.rare_class_ids)
        rare_ids_arr = np.asarray(rare_ids, dtype=np.int64)

        sample_seed = (
            int(sample_seed_override)
            if sample_seed_override is not None
            else self._stable_seed(path=path, tile_i=tile_i, rep_i=rep_i)
        )
        rng = np.random.default_rng(sample_seed)
        out = None
        chosen_meta = None

        for attempt in range(int(crop_cfg.resample_tries)):
            c_idx, used_rare_center = _pick_center_index(
                y_train_pts=y_train_pts,
                ignore_index=int(self.ignore_index),
                crop_cfg=crop_cfg,
                fallback_rare_ids=tuple(self.preproc.rare_class_ids),
                rng=rng,
            )
            cx, cy = float(xyz[c_idx, 0]), float(xyz[c_idx, 1])
            tried_min_size = False
            size = base_size

            for shrink in range(int(crop_cfg.max_shrink_steps) + 1):
                pt_idx = _points_in_xy_square(xyz, cx=cx, cy=cy, half=0.5 * size)
                if pt_idx.size < int(crop_cfg.crop_min_points):
                    break

                if crop_cfg.max_points_per_crop_soft is not None and pt_idx.size > int(crop_cfg.max_points_per_crop_soft):
                    new_size = max(min_size, size * float(crop_cfg.shrink_ratio))
                    if new_size <= min_size + 1e-6:
                        if tried_min_size:
                            break
                        tried_min_size = True
                        size = min_size
                    else:
                        size = new_size
                    continue

                y_crop = y_train_pts[pt_idx]
                valid = y_crop != int(self.ignore_index)
                denom = int(valid.sum())
                rare_frac = float((valid & np.isin(y_crop, rare_ids_arr)).sum()) / float(denom) if denom > 0 else 0.0

                out_try = _points_to_sparse(
                    xyz_m=xyz,
                    raw={
                        "intensity": intensity_scaled,
                        "return_number": raw["return_number"],
                        "number_of_returns": raw["number_of_returns"],
                        "native_labels": raw["native_labels"],
                    },
                    point_idx=pt_idx,
                    patch_cfg=self.patch_cfg,
                    feat_cfg=self.feat_cfg,
                    ignore_index=int(self.ignore_index),
                    label_lut=self.label_lut,
                    voxel_cfg=self.voxel_cfg,
                    rng=rng,
                    crop_center_xy_m=(cx, cy),
                    do_aug=True,
                    aug_cfg=self.aug_cfg if self.aug_cfg.enabled else None,
                    return_prepared=True,
                )
                n_vox = int(out_try["coords"].shape[0])
                if n_vox <= max_vox_soft:
                    out = out_try
                    chosen_meta = {
                        "crop_size_xy_m": float(size),
                        "shrink_steps": int(shrink),
                        "resample_attempt": int(attempt),
                        "n_points": int(pt_idx.size),
                        "n_vox": n_vox,
                        "rare_frac": float(rare_frac),
                        "used_rare_center": int(used_rare_center),
                        "used_fallback": 0,
                        "accepted_over_soft": 0,
                    }
                    break

                out_of_runway = shrink >= int(crop_cfg.max_shrink_steps) or size <= min_size + 1e-6
                if out_of_runway and n_vox <= max_vox_hard:
                    out = out_try
                    chosen_meta = {
                        "crop_size_xy_m": float(size),
                        "shrink_steps": int(shrink),
                        "resample_attempt": int(attempt),
                        "n_points": int(pt_idx.size),
                        "n_vox": n_vox,
                        "rare_frac": float(rare_frac),
                        "used_rare_center": int(used_rare_center),
                        "used_fallback": 0,
                        "accepted_over_soft": 1,
                    }
                    break

                if out_of_runway and n_vox > max_vox_hard:
                    if hard_max_points > 0:
                        pt_idx2 = _subsample_points_keep_rare(
                            idx=pt_idx,
                            y_train_pts=y_train_pts,
                            ignore_index=int(self.ignore_index),
                            rare_ids=tuple(rare_ids),
                            max_points=min(hard_max_points, int(pt_idx.size)),
                            rng=rng,
                        )
                        y_crop2 = y_train_pts[pt_idx2]
                        valid2 = y_crop2 != int(self.ignore_index)
                        denom2 = int(valid2.sum())
                        rare_frac2 = float((valid2 & np.isin(y_crop2, rare_ids_arr)).sum()) / float(denom2) if denom2 > 0 else 0.0
                        out_try2 = _points_to_sparse(
                            xyz_m=xyz,
                            raw={
                                "intensity": intensity_scaled,
                                "return_number": raw["return_number"],
                                "number_of_returns": raw["number_of_returns"],
                                "native_labels": raw["native_labels"],
                            },
                            point_idx=pt_idx2,
                            patch_cfg=self.patch_cfg,
                            feat_cfg=self.feat_cfg,
                            ignore_index=int(self.ignore_index),
                            label_lut=self.label_lut,
                            voxel_cfg=self.voxel_cfg,
                            rng=rng,
                            crop_center_xy_m=(cx, cy),
                            do_aug=True,
                            aug_cfg=self.aug_cfg if self.aug_cfg.enabled else None,
                            return_prepared=True,
                        )
                        n_vox2 = int(out_try2["coords"].shape[0])
                        if n_vox2 <= max_vox_hard:
                            out = out_try2
                            chosen_meta = {
                                "crop_size_xy_m": float(size),
                                "shrink_steps": int(shrink),
                                "resample_attempt": int(attempt),
                                "n_points": int(pt_idx2.size),
                                "n_vox": n_vox2,
                                "rare_frac": float(rare_frac2),
                                "used_rare_center": int(used_rare_center),
                                "used_fallback": 1,
                                "accepted_over_soft": int(n_vox2 > max_vox_soft),
                            }
                    break

                new_size = max(min_size, size * float(crop_cfg.shrink_ratio))
                if new_size <= min_size + 1e-6:
                    if tried_min_size:
                        break
                    tried_min_size = True
                    size = min_size
                else:
                    size = new_size

            if out is not None:
                break

        if out is None or chosen_meta is None:
            raise RuntimeError(f"[DALES M1] No valid crop for file={path}, epoch={self.epoch}, rep={rep_i}.")
        prepared = out.pop("_prepared")
        prepared["source_id"] = str(path.resolve())
        prepared["context_side_xy_m"] = float(chosen_meta["crop_size_xy_m"])
        prepared["sample_seed"] = int(sample_seed)
        return out, prepared, chosen_meta

    def _prepared_to_sparse(
        self,
        payload: Mapping[str, Any],
        *,
        rng: np.random.Generator,
    ) -> Dict[str, torch.Tensor]:
        xyz = np.asarray(payload["xyz"])
        xyz_norm = (xyz / float(self.patch_cfg.coord_norm_factor)).astype(np.float32, copy=False)
        feats = build_features(
            xyz_local=xyz_norm,
            intensity=(payload.get("intensity", None) if self.feat_cfg.use_intensity else None),
            return_number=payload.get("return_number", None),
            number_of_returns=payload.get("number_of_returns", None),
            rgb=None,
            cfg=self.feat_cfg,
        ).astype(np.float32, copy=False)
        q = np.floor(xyz_norm / float(self.patch_cfg.voxel_size)).astype(np.int32, copy=False)
        vx = voxelize_from_q(
            q_int32=np.ascontiguousarray(q, dtype=np.int32),
            feats_p_f32=feats,
            labels_p_i64=np.asarray(payload["y_train"], dtype=np.int64),
            ignore_index=int(self.ignore_index),
            cfg=self.voxel_cfg,
            rng=rng,
            return_maps=False,
            num_classes_hint=8,
        )
        return {
            "coords": torch.from_numpy(np.ascontiguousarray(vx["coords_u"], dtype=np.int32)).int(),
            "feats": torch.from_numpy(np.ascontiguousarray(vx["feats_u"], dtype=np.float32)).float(),
            "labels": torch.from_numpy(np.ascontiguousarray(vx["labels_u"], dtype=np.int64)).long(),
        }

    @staticmethod
    def _attach_mix_meta(
        out: Dict[str, torch.Tensor],
        d: MixDiagnostics,
        *,
        donor_index: int,
    ) -> None:
        values = {
            "meta_mix_applied": (d.applied, torch.int64),
            "meta_mix_skip_code": (d.skip_code, torch.int64),
            "meta_mix_host_points": (d.host_points, torch.int64),
            "meta_mix_host_removed": (d.host_removed, torch.int64),
            "meta_mix_donor_inserted": (d.donor_inserted, torch.int64),
            "meta_mix_output_points": (d.output_points, torch.int64),
            "meta_mix_replacement_side_m": (d.replacement_side_m, torch.float32),
            "meta_mix_guard_band_m": (d.guard_band_m, torch.float32),
            "meta_mix_height_shift_m": (d.height_shift_m, torch.float32),
            "meta_mix_cross_provenance_voxels": (
                d.cross_provenance_voxels,
                torch.int64,
            ),
        }
        for key, (value, dtype) in values.items():
            out[key] = torch.tensor(value, dtype=dtype)
        out["meta_mix_removed_class_counts"] = torch.tensor(d.host_removed_class_counts, dtype=torch.int64)
        out["meta_mix_donor_class_counts"] = torch.tensor(d.donor_inserted_class_counts, dtype=torch.int64)
        out["meta_mix_output_class_counts"] = torch.tensor(d.output_class_counts, dtype=torch.int64)
        n_classes = len(d.output_class_counts)
        out["meta_mix_context_pairs"] = torch.tensor(d.donor_host_context_pairs, dtype=torch.int64).reshape(n_classes, n_classes)
        out["meta_mix_donor_index"] = torch.tensor(donor_index, dtype=torch.int64)

    @staticmethod
    def _attach_crop_meta(out: Dict[str, torch.Tensor], meta: Mapping[str, Any]) -> None:
        out["meta_crop_size_xy_m"] = torch.tensor(meta["crop_size_xy_m"], dtype=torch.float32)
        out["meta_shrink_steps"] = torch.tensor(meta["shrink_steps"], dtype=torch.int64)
        out["meta_resample_attempt"] = torch.tensor(meta["resample_attempt"], dtype=torch.int64)
        out["meta_n_points"] = torch.tensor(meta["n_points"], dtype=torch.int64)
        out["meta_n_vox"] = torch.tensor(meta["n_vox"], dtype=torch.int64)
        out["meta_rare_frac"] = torch.tensor(meta["rare_frac"], dtype=torch.float32)
        out["meta_used_rare_center"] = torch.tensor(meta["used_rare_center"], dtype=torch.int64)
        out["meta_used_fallback"] = torch.tensor(meta["used_fallback"], dtype=torch.int64)
        out["meta_accepted_over_soft"] = torch.tensor(meta.get("accepted_over_soft", 0), dtype=torch.int64)

    def _getitem_mix_crop(
        self,
        *,
        host_tile_i: int,
        host_rep_i: int,
        host_path: Path,
    ) -> Dict[str, torch.Tensor]:
        host_raw, host_xyz, host_intensity = self._load_mix_tile(host_path)
        host_out, host_payload, host_meta = self._sample_mix_crop(
            path=host_path,
            tile_i=host_tile_i,
            rep_i=host_rep_i,
            raw=host_raw,
            xyz=host_xyz,
            intensity_scaled=host_intensity,
        )
        voxel_edge_m = float(self.patch_cfg.voxel_size) * float(self.patch_cfg.coord_norm_factor)
        mix_rng = np.random.default_rng(
            stable_seed(
                self.seed,
                self.epoch,
                str(host_path.resolve()),
                host_rep_i,
                "mix3d",
            )
        )
        mix_donor_index = -1

        if not should_apply_mix(self.mix3d_cfg, mix_rng):
            result = unchanged_mix_result(
                host_payload,
                skip_code=MIX_SKIP_PROBABILITY,
                guard_band_m=float(self.mix3d_cfg.guard_band_voxels) * voxel_edge_m,
                num_classes=8,
            )
            out = host_out
            donor_meta = None
        else:
            donor_tile_i = choose_different_source_index(
                source_ids=[str(p.resolve()) for p in self.files],
                host_index=host_tile_i,
                rng=mix_rng,
                max_attempts=int(self.mix3d_cfg.max_donor_attempts),
            )
            if donor_tile_i is None:
                result = unchanged_mix_result(
                    host_payload,
                    skip_code=MIX_SKIP_NO_DONOR,
                    guard_band_m=float(self.mix3d_cfg.guard_band_voxels) * voxel_edge_m,
                    num_classes=8,
                )
                out = host_out
                donor_meta = None
            else:
                mix_donor_index = int(donor_tile_i)
                donor_path = self.files[int(donor_tile_i)]
                donor_raw, donor_xyz, donor_intensity = self._load_mix_tile(donor_path)
                donor_rep_i = int(mix_rng.integers(0, max(1, int(self.crop_cfg.crops_per_tile_per_epoch))))
                donor_seed = stable_seed(
                    self.seed,
                    self.epoch,
                    str(host_path.resolve()),
                    str(donor_path.resolve()),
                    host_rep_i,
                    donor_rep_i,
                    "donor_crop",
                )
                _donor_out, donor_payload, donor_meta = self._sample_mix_crop(
                    path=donor_path,
                    tile_i=int(donor_tile_i),
                    rep_i=donor_rep_i,
                    raw=donor_raw,
                    xyz=donor_xyz,
                    intensity_scaled=donor_intensity,
                    sample_seed_override=donor_seed,
                )
                out = host_out
                mixed_within_budget = False
                result = None
                hard_limit = int(self.crop_cfg.max_voxels_per_crop_hard)
                for budget_attempt in range(int(self.mix3d_cfg.max_region_attempts)):
                    candidate_result = compose_crop_replace(
                        host=host_payload,
                        donor=donor_payload,
                        cfg=self.mix3d_cfg,
                        num_classes=8,
                        voxel_edge_m=voxel_edge_m,
                        rng=mix_rng,
                    )
                    result = candidate_result
                    if not candidate_result.diagnostics.applied:
                        break
                    candidate_out = self._prepared_to_sparse(
                        candidate_result.sample,
                        rng=np.random.default_rng(
                            stable_seed(
                                self.seed,
                                self.epoch,
                                str(host_path.resolve()),
                                host_rep_i,
                                budget_attempt,
                                "mixed_voxel",
                            )
                        ),
                    )
                    if int(candidate_out["coords"].shape[0]) <= hard_limit:
                        out = candidate_out
                        mixed_within_budget = True
                        break

                if result is None:
                    raise RuntimeError("DALES M1 composition produced no result.")
                if result.diagnostics.applied and not mixed_within_budget:
                    result = unchanged_mix_result(
                        host_payload,
                        skip_code=MIX_SKIP_OUTPUT_BUDGET,
                        guard_band_m=float(self.mix3d_cfg.guard_band_voxels) * voxel_edge_m,
                        num_classes=8,
                    )
                    out = host_out

        final_meta = dict(host_meta)
        final_meta["n_points"] = int(result.diagnostics.output_points)
        final_meta["n_vox"] = int(out["coords"].shape[0])
        out["path"] = str(host_path)
        self._attach_crop_meta(out, final_meta)
        self._attach_mix_meta(
            out,
            result.diagnostics,
            donor_index=mix_donor_index,
        )
        out["meta_mix_donor_crop_size_xy_m"] = torch.tensor(
            float("nan") if donor_meta is None else donor_meta["crop_size_xy_m"],
            dtype=torch.float32,
        )
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # In crop-mode, __len__ already returns len(files) * groups where
        # groups = ceil(crops_per_tile_per_epoch / crops_per_item).
        # So __getitem__ must use the same mapping, otherwise you sample the wrong tiles
        # and you DO NOT get the speedup.
        if self.is_train and self.sampling_mode == "crops":
            k = max(1, int(self.crop_cfg.crops_per_tile_per_epoch))
            m = max(1, int(getattr(self.crop_cfg, "crops_per_item", 1)))
            groups = (k + m - 1) // m
            tile_i = int(idx // groups)
            group_i = int(idx % groups)
            path = self.files[tile_i]
            rep_start = group_i * m
            rep_count = min(m, k - rep_start)
        else:
            tile_i = int(idx)
            group_i = 0
            rep_start = 0
            rep_count = 1
            path = self.files[idx]

        if self.mix3d_cfg.enabled:
            if rep_count != 1:
                raise RuntimeError("Initial DALES M1 requires exactly one crop per item.")
            return self._getitem_mix_crop(
                host_tile_i=tile_i,
                host_rep_i=rep_start,
                host_path=path,
            )

        # Cache fast-path
        cache_path: Optional[Path] = None
        raw_from_cache: Optional[Dict[str, np.ndarray]] = None

        if self.use_cache:
            if self.cache_kind == "raw":
                key = _cache_key_for_dales_raw(path=path)
                cache_path = self._cache_dir / f"{key}.pt"
                if cache_path.exists():
                    obj = torch.load(cache_path, map_location="cpu")
                    if not isinstance(obj, dict) or "xyz" not in obj:
                        raise TypeError(f"Bad raw cache payload: {cache_path}")
                    raw_from_cache = _ensure_raw_dtypes(obj)
                elif self.require_cache:
                    raise RuntimeError(f"[DALES raw cache missing] {cache_path}")
            else:
                # old voxel-cache behavior (keep as-is)
                key = _cache_key_for_dales(
                    path=path,
                    patch_cfg=self.patch_cfg,
                    feat_cfg=self.feat_cfg,
                    preproc=self.preproc,
                    cache_key_extra=self.cache_key_extra,
                    ignore_index=self.ignore_index,
                    label_map=self.label_map,
                    voxel_cfg=self.voxel_cfg,
                )
                cache_path = self._cache_dir / f"{key}.pt"
                if cache_path.exists():
                    obj = torch.load(cache_path, map_location="cpu")
                    if isinstance(obj, dict) and "path" not in obj:
                        obj["path"] = str(path)
                    return obj
                if self.require_cache:
                    raise RuntimeError(f"[DALES voxel cache missing] {cache_path}")

        raw = raw_from_cache if raw_from_cache is not None else _read_dales_las(path)
        raw = _ensure_raw_dtypes(raw)

        xyz = raw["xyz"]
        if self.patch_cfg.make_local_coords:
            xyz = xyz - xyz.min(axis=0, keepdims=True)

        # Deterministic RNG per sample for reproducible augmentation (while still random across epochs).
        # We mix in global RNG state to vary each epoch naturally.
        # print(f"DALES: processing {path} (tile_i={tile_i}, rep_i={rep_i})")
        # IMPORTANT:
        # - In crop-mode, we do NOT augment the full tile (expensive).
        # - We augment only the crop points inside _points_to_sparse(do_aug=True).
        # - In tile-mode (non-crop), keep existing behavior.
        if not (self.is_train and self.sampling_mode == "crops"):
            sample_seed = self._stable_seed(path=path, tile_i=tile_i, rep_i=0)
            rng = np.random.default_rng(sample_seed)
            if self.is_train:
                xyz = augment_xyz(xyz, self.aug_cfg, rng)

        # intensity preprocessing happens in "scaled space" expected by FeatureConfig
        div = 1.0
        intensity_scaled = None
        if self.feat_cfg.use_intensity:
            # intensity preprocessing happens in "scaled space" expected by FeatureConfig
            intensity = raw.get("intensity", None)
            if intensity is None:
                intensity_scaled = None
            else:
                intensity = intensity.astype(np.float32, copy=False)

                div = (
                    float(self.preproc.intensity_divisor_override)
                    if self.preproc.intensity_divisor_override
                    else float(self.feat_cfg.intensity_divisor)
                )
            if intensity is None:
                raise RuntimeError("DALES: feat_cfg.use_intensity=True but raw intensity is missing/None.")
            if intensity.max() <= 1.0 + 1e-3:
                intensity_scaled = intensity  # already [0,1]
            else:
                intensity_scaled = intensity / div

            mode = (self.preproc.intensity_mode or "none").lower()
            if mode == "none":
                pass
            elif mode == "constant":
                intensity_scaled = np.full_like(
                    intensity_scaled,
                    float(self.preproc.intensity_constant),
                    dtype=np.float32,
                )
            elif mode == "robust_standardize":
                # standardize then squash to [0,1] with sigmoid-like clipping (optional)
                z = _robust_standardize(intensity_scaled)
                intensity_scaled = 1.0 / (1.0 + np.exp(-z))
            elif mode == "quantile_match":
                if (
                    self.preproc.ref_quantiles is None
                    or self.preproc.ref_probs is None
                    or self.preproc.tgt_quantiles is None
                    or self.preproc.tgt_probs is None
                ):
                    raise RuntimeError("quantile_match requires ref_quantiles/ref_probs and tgt_quantiles/tgt_probs")
                intensity_scaled = _quantile_match(
                    intensity_scaled,
                    self.preproc.ref_quantiles,
                    self.preproc.ref_probs,
                    self.preproc.tgt_quantiles,
                    self.preproc.tgt_probs,
                )
            else:
                raise ValueError(f"Unknown intensity_mode: {self.preproc.intensity_mode}")

            # clip to [0,1] range
            intensity_scaled = np.clip(intensity_scaled, 0.0, 1.0).astype(np.float32, copy=False)

        # ---- CROP MODE (TRAINING) ----
        if self.is_train and self.sampling_mode == "crops":
            # Map ALL POINT labels once (cheap; enables rare-center sampling)
            y_native_pts = raw["native_labels"].astype(np.int64, copy=False)
            y_safe_pts = np.clip(y_native_pts, 0, 255).astype(np.int64, copy=False)
            y_train_pts = self.label_lut[y_safe_pts]

            crop_cfg = self.crop_cfg
            max_vox_soft = int(crop_cfg.max_voxels_per_crop_soft)

            max_vox_hard = int(getattr(crop_cfg, "max_voxels_per_crop_hard", max_vox_soft))
            if max_vox_hard < max_vox_soft:
                raise ValueError(f"max_voxels_per_crop_hard ({max_vox_hard}) must be >= soft ({max_vox_soft}).")

            hard_max_points = int(getattr(crop_cfg, "hard_max_points", 0) or 0)
            if hard_max_points and hard_max_points < int(crop_cfg.crop_min_points):
                raise ValueError("hard_max_points must be >= crop_min_points.")

            min_size = float(crop_cfg.min_crop_size_xy_m)
            base_size = float(crop_cfg.crop_size_xy_m)

            rare_ids = crop_cfg.rare_center_class_ids
            if rare_ids is None or len(rare_ids) == 0:
                rare_ids = tuple(self.preproc.rare_class_ids)
            rare_ids_arr = np.asarray(rare_ids, dtype=np.int64)

            crops_out: List[Dict[str, torch.Tensor]] = []

            for rep_off in range(rep_count):
                rep_i = rep_start + rep_off
                sample_seed = self._stable_seed(path=path, tile_i=tile_i, rep_i=rep_i)
                rng = np.random.default_rng(sample_seed)

                out = None
                chosen_meta = None

                # Try resampling centers
                for _attempt in range(int(crop_cfg.resample_tries)):
                    c_idx, used_rare_center = _pick_center_index(
                        y_train_pts=y_train_pts,
                        ignore_index=int(self.ignore_index),
                        crop_cfg=crop_cfg,
                        fallback_rare_ids=tuple(self.preproc.rare_class_ids),
                        rng=rng,
                    )
                    cx, cy = float(xyz[c_idx, 0]), float(xyz[c_idx, 1])

                    tried_min_size = False  # reset per center
                    size = base_size

                    # Try shrinking crop
                    for _shrink in range(int(crop_cfg.max_shrink_steps) + 1):
                        # print(
                        #     base_size,
                        #     _shrink,
                        #     "n_vox:",
                        #     0,
                        #     "attempt:",
                        #     _attempt,
                        #     "center_idx:",
                        #     c_idx,
                        #     "used_rare_center:",
                        #     used_rare_center,
                        # )
                        half = 0.5 * size
                        pt_idx = _points_in_xy_square(xyz, cx=cx, cy=cy, half=half)

                        if pt_idx.size < int(crop_cfg.crop_min_points):
                            break

                        # Optional point soft-cap (pre-voxelization)
                        if crop_cfg.max_points_per_crop_soft is not None and pt_idx.size > int(crop_cfg.max_points_per_crop_soft):
                            new_size = max(min_size, size * float(crop_cfg.shrink_ratio))
                            if new_size <= min_size + 1e-6:
                                # allow exactly one "min_size attempt"
                                if tried_min_size:
                                    break
                                tried_min_size = True
                                size = min_size
                            else:
                                size = new_size
                            continue

                        # Rare fraction (point-space)
                        y_crop = y_train_pts[pt_idx]
                        valid = y_crop != int(self.ignore_index)
                        denom = int(valid.sum())

                        if denom > 0:
                            rare_mask = valid & np.isin(y_crop, rare_ids_arr)
                            rare_frac = float(rare_mask.sum()) / float(denom)
                        else:
                            rare_frac = 0.0

                        out_try = _points_to_sparse(
                            xyz_m=xyz,
                            raw={
                                "intensity": intensity_scaled if intensity_scaled is not None else None,
                                "return_number": raw["return_number"],
                                "number_of_returns": raw["number_of_returns"],
                                "native_labels": raw["native_labels"],
                            },
                            point_idx=pt_idx,
                            patch_cfg=self.patch_cfg,
                            feat_cfg=self.feat_cfg,
                            ignore_index=int(self.ignore_index),
                            label_lut=self.label_lut,
                            voxel_cfg=self.voxel_cfg,
                            rng=rng,
                            crop_center_xy_m=(cx, cy),
                            do_aug=True,
                            aug_cfg=self.aug_cfg if self.aug_cfg.enabled else None,
                        )

                        n_vox = int(out_try["coords"].shape[0])

                        if n_vox <= max_vox_soft:
                            out = out_try
                            chosen_meta = {
                                "crop_size_xy_m": float(size),
                                "shrink_steps": int(_shrink),
                                "resample_attempt": int(_attempt),
                                "n_points": int(pt_idx.size),
                                "n_vox": int(n_vox),
                                "rare_frac": float(rare_frac),
                                "used_rare_center": int(used_rare_center),
                                "used_fallback": 0,
                                "accepted_over_soft": 0,
                            }
                            break

                        # If we can't reach soft cap anymore (no shrink runway left),
                        # but we're within HARD cap, accept this crop instead of discarding it.
                        out_of_runway = (_shrink >= int(crop_cfg.max_shrink_steps)) or (size <= min_size + 1e-6)
                        if out_of_runway and (n_vox <= max_vox_hard):
                            out = out_try
                            chosen_meta = {
                                "crop_size_xy_m": float(size),
                                "shrink_steps": int(_shrink),
                                "resample_attempt": int(_attempt),
                                "n_points": int(pt_idx.size),
                                "n_vox": int(n_vox),
                                "rare_frac": float(rare_frac),
                                "used_rare_center": int(used_rare_center),
                                "used_fallback": 0,
                                "accepted_over_soft": 1,  # accepted above soft, within hard
                            }
                            break

                        # If crop is beyond HARD voxel cap and we are out of shrinking runway, do last-resort fallback.
                        if out_of_runway and (n_vox > max_vox_hard):
                            max_points = min(hard_max_points, int(pt_idx.size))

                            if hard_max_points > 0:
                                pt_idx2 = _subsample_points_keep_rare(
                                    idx=pt_idx,
                                    y_train_pts=y_train_pts,
                                    ignore_index=int(self.ignore_index),
                                    rare_ids=tuple(rare_ids),
                                    max_points=max_points,
                                    rng=rng,
                                )

                                # Rare fraction (point-space)
                                y_crop2 = y_train_pts[pt_idx2]
                                valid2 = y_crop2 != int(self.ignore_index)
                                denom2 = int(valid2.sum())
                                if denom2 > 0:
                                    rare_mask2 = valid2 & np.isin(y_crop2, rare_ids_arr)
                                    rare_frac2 = float(rare_mask2.sum()) / float(denom2)
                                else:
                                    rare_frac2 = 0.0

                                out_try2 = _points_to_sparse(
                                    xyz_m=xyz,
                                    raw={
                                        "intensity": intensity_scaled if intensity_scaled is not None else None,
                                        "return_number": raw["return_number"],
                                        "number_of_returns": raw["number_of_returns"],
                                        "native_labels": raw["native_labels"],
                                    },
                                    point_idx=pt_idx2,
                                    patch_cfg=self.patch_cfg,
                                    feat_cfg=self.feat_cfg,
                                    ignore_index=int(self.ignore_index),
                                    label_lut=self.label_lut,
                                    voxel_cfg=self.voxel_cfg,
                                    rng=rng,
                                    crop_center_xy_m=(cx, cy),
                                    do_aug=True,
                                    aug_cfg=self.aug_cfg if self.aug_cfg.enabled else None,
                                )

                                n_vox2 = int(out_try2["coords"].shape[0])
                                if n_vox2 <= max_vox_hard:
                                    out = out_try2
                                    chosen_meta = {
                                        "crop_size_xy_m": float(size),
                                        "shrink_steps": int(_shrink),
                                        "resample_attempt": int(_attempt),
                                        "n_points": int(pt_idx2.size),
                                        "n_vox": int(n_vox2),
                                        "rare_frac": float(rare_frac2),
                                        "used_rare_center": int(used_rare_center),
                                        "used_fallback": 1,  # <-- indicates point-sub fallback happened
                                        "accepted_over_soft": int(n_vox2 > max_vox_soft),
                                    }
                                    break

                            break

                        # Otherwise, continue shrinking (soft-cap behavior)
                        new_size = max(min_size, size * float(crop_cfg.shrink_ratio))
                        if new_size <= min_size + 1e-6:
                            if tried_min_size:
                                break
                            tried_min_size = True
                            size = min_size
                        else:
                            size = new_size
                        continue

                    if out is not None:
                        break  # break resample loop once we have a crop

                if out is None:
                    # resample a brand new center and try again (or force a min_size crop)
                    continue

                # Attach meta + BEV + path
                out["path"] = str(path)

                if chosen_meta is None:
                    chosen_meta = {
                        "crop_size_xy_m": float(base_size),
                        "shrink_steps": -1,
                        "resample_attempt": -1,
                        "n_points": -1,
                        "n_vox": int(out["coords"].shape[0]),
                        "rare_frac": float("nan"),
                        "used_rare_center": 0,
                        "used_fallback": 0,
                        "accepted_over_soft": 0,
                    }

                out["meta_crop_size_xy_m"] = torch.tensor(chosen_meta["crop_size_xy_m"], dtype=torch.float32)
                out["meta_shrink_steps"] = torch.tensor(chosen_meta["shrink_steps"], dtype=torch.int64)
                out["meta_resample_attempt"] = torch.tensor(chosen_meta["resample_attempt"], dtype=torch.int64)
                out["meta_n_points"] = torch.tensor(chosen_meta["n_points"], dtype=torch.int64)
                out["meta_n_vox"] = torch.tensor(chosen_meta["n_vox"], dtype=torch.int64)
                out["meta_rare_frac"] = torch.tensor(chosen_meta["rare_frac"], dtype=torch.float32)
                out["meta_used_rare_center"] = torch.tensor(chosen_meta["used_rare_center"], dtype=torch.int64)
                out["meta_used_fallback"] = torch.tensor(chosen_meta["used_fallback"], dtype=torch.int64)
                out["meta_accepted_over_soft"] = torch.tensor(int(chosen_meta.get("accepted_over_soft", 0)), dtype=torch.int64)

                if self.bev_cfg is not None and self.bev_cfg.enabled:
                    m_per_vox = float(self.patch_cfg.voxel_size) * float(self.patch_cfg.coord_norm_factor)
                    coords_np = out["coords"].cpu().numpy().astype(np.int32, copy=False)
                    labels_np = out["labels"].cpu().numpy().astype(np.int64, copy=False)
                    bev_labels = {}
                    bev_selected = {}
                    for lvl in self.bev_cfg.levels:
                        lbl, sel = build_bev_labels_and_selected_idx(
                            coords_vox_int32=coords_np,
                            labels_vox_i64=labels_np,
                            voxel_ignore_index=int(self.ignore_index),
                            bev_cfg=self.bev_cfg,
                            level=str(lvl),
                            meters_per_voxel=m_per_vox,
                            rng=rng,
                        )
                        bev_labels[str(lvl)] = torch.from_numpy(lbl).long()
                        bev_selected[str(lvl)] = torch.from_numpy(sel).long()
                    out["bev_labels"] = bev_labels
                    out["bev_selected_idx"] = bev_selected

                crops_out.append(out)

            if len(crops_out) == 0:
                raise RuntimeError(f"[DALES] No valid crops produced for tile={path} at epoch={self.epoch}.")

            if rep_count == 1:
                return crops_out[0]

            return crops_out

        # ---- TILE MODE (EVAL / DEBUG) ----
        # normalize coords for voxelization like ECLAIR pipeline
        xyz_norm = (xyz / float(self.patch_cfg.coord_norm_factor)).astype(np.float32, copy=False)

        feats = build_features(
            xyz_local=xyz_norm,
            intensity=intensity_scaled,
            return_number=raw["return_number"],
            number_of_returns=raw["number_of_returns"],
            rgb=None,
            cfg=self.feat_cfg,
        )

        # voxel quantize
        q = np.floor(xyz_norm / float(self.patch_cfg.voxel_size)).astype(np.int32)
        q = np.ascontiguousarray(q, dtype=np.int32)  # important for ME

        # Map ALL POINT labels (native -> train) in point space
        y_native_pts = raw["native_labels"].astype(np.int64, copy=False)
        y_safe_pts = np.clip(y_native_pts, 0, 255).astype(np.int64, copy=False)
        y_all_pts = self.label_lut[y_safe_pts].astype(np.int64, copy=False)
        rng_vox = rng if self.voxel_cfg.feat_pool == "random" else None
        need_maps = bool(self.preproc.use_height_xy_cap)
        vx = voxelize_from_q(
            q_int32=q,
            feats_p_f32=feats.astype(np.float32, copy=False),
            labels_p_i64=y_all_pts,
            ignore_index=int(self.ignore_index),
            cfg=self.voxel_cfg,
            rng=rng_vox,  # important if feat_pool='random'
            return_maps=need_maps,  # needed if you use height_xy_cap
            num_classes_hint=8,
        )

        q_u = vx["coords_u"]
        feats_u = vx["feats_u"]
        y_u = vx["labels_u"]
        unique_map = vx.get("unique_map", None)  # representative point index per voxel

        # Basic range sanity check (train ids must be in [0..7] except ignore)
        valid = y_u != self.ignore_index
        if np.any(valid):
            mn = int(y_u[valid].min())
            mx = int(y_u[valid].max())
            if mn < 0 or mx >= 8:
                raise RuntimeError(f"DALES label mapping out of range: min={mn}, max={mx}")

        # OPTIONAL: label-free thinning in XYxZ grid
        if self.preproc.use_height_xy_cap:
            # xyz in meters for voxel representatives (using unique_map)
            if unique_map is None:
                raise RuntimeError("height_xy_cap requires voxelize_from_q(return_maps=True) to provide unique_map.")
            xyz_u_m = (xyz_norm[unique_map] * float(self.patch_cfg.coord_norm_factor)).astype(np.float32, copy=False)

            xyz_u_m, feats_u, y_u = _height_xy_cap(
                xyz_u_m,
                feats_u,
                y_u,
                xy_cell_size_m=float(self.preproc.xy_cell_size_m),
                height_bins=int(self.preproc.height_bins),
                caps_per_bin=self.preproc.caps_per_bin,
                rng=rng_vox,
            )

            # recompute coords after thinning
            xyz_u_norm = (xyz_u_m / float(self.patch_cfg.coord_norm_factor)).astype(np.float32, copy=False)
            q_u = np.floor(xyz_u_norm / float(self.patch_cfg.voxel_size)).astype(np.int32)
            q_u = np.ascontiguousarray(q_u, dtype=np.int32)

        # Optional voxel cap (tile-mode only)
        if self.preproc.max_voxels is not None:
            max_v = int(self.preproc.max_voxels)
            coords_u = q_u
            feats_u = feats_u
            y_u = y_u
            coords_u, feats_u, y_u = _cap_voxels(
                coords_u,
                feats_u,
                y_u,
                max_voxels=max_v,
                ignore_index=int(self.ignore_index),
                class_aware=bool(self.preproc.class_aware_max_voxels),
                rare_ids=list(self.preproc.rare_class_ids),
                rng=rng_vox,  # stable per tile (already seeded above)
            )
            q_u, feats_u, y_u = coords_u, feats_u, y_u

        coords_t = torch.from_numpy(q_u).int()
        feats_t = torch.from_numpy(np.ascontiguousarray(feats_u)).float()
        labels_t = torch.from_numpy(np.ascontiguousarray(y_u)).long()

        # DEBUG: confirm actual voxel count after cap
        if idx == 0:
            print(
                f"[DALES] file={path.name} voxels_after_cap={coords_t.shape[0]}",
                flush=True,
            )
            u, c = np.unique(y_u, return_counts=True)
            print(
                "[DALES labels mapped] unique:",
                list(zip(u.tolist(), c.tolist()))[:20],
                flush=True,
            )

        out = {
            "coords": coords_t,
            "feats": feats_t,
            "labels": labels_t,  # train labels: ignore_index or [0..7]
            "path": str(path),
        }

        # if self.use_cache and (cache_path is not None) and self.write_cache:
        # Cache write: ONLY if enabled. Never write during multi-worker training unless you really intend to.
        if self.use_cache and (cache_path is not None) and self.write_cache:
            try:
                if self.cache_kind == "raw":
                    atomic_save_torch(raw, cache_path)
                else:
                    atomic_save_torch(out, cache_path)
            except Exception:
                pass

        # --- BEV supervision (Task 8) ---
        if self.bev_cfg is not None and self.bev_cfg.enabled and self.is_train:
            m_per_vox = float(self.patch_cfg.voxel_size) * float(self.patch_cfg.coord_norm_factor)

            coords_np = out["coords"].cpu().numpy().astype(np.int32, copy=False)
            labels_np = out["labels"].cpu().numpy().astype(np.int64, copy=False)

            bev_labels = {}
            bev_selected = {}
            for lvl in self.bev_cfg.levels:
                lbl, sel = build_bev_labels_and_selected_idx(
                    coords_vox_int32=coords_np,
                    labels_vox_i64=labels_np,
                    voxel_ignore_index=int(self.ignore_index),
                    bev_cfg=self.bev_cfg,
                    level=str(lvl),
                    meters_per_voxel=m_per_vox,
                    rng=rng,
                )
                bev_labels[str(lvl)] = torch.from_numpy(lbl).long()
                bev_selected[str(lvl)] = torch.from_numpy(sel).long()

            out["bev_labels"] = bev_labels
            out["bev_selected_idx"] = bev_selected

        return out


def minkowski_collate_dales(batch) -> Dict[str, torch.Tensor]:
    # In crop-mode, __getitem__ may return List[Dict] to bundle multiple crops per tile-load.
    flat = []
    for b in batch:
        if isinstance(b, list):
            flat.extend(b)
        else:
            flat.append(b)
    batch = flat

    coords_list = [b["coords"] for b in batch]
    feats_list = [b["feats"] for b in batch]
    labels_list = [b["labels"] for b in batch]

    coords, feats, labels = ME.utils.sparse_collate(coords_list, feats_list, labels_list)
    paths = [b["path"] for b in batch]
    out = {"coords": coords, "feats": feats, "labels": labels, "paths": paths}

    # --- NEW: carry DALES crop debug meta ---
    for k in batch[0].keys():
        if k.startswith("meta_"):
            out[k] = torch.stack([b[k] for b in batch], dim=0)

    if "bev_labels" in batch[0]:
        bev0 = batch[0]["bev_labels"]
        if isinstance(bev0, dict):
            out["bev_labels"] = {k: torch.stack([b["bev_labels"][k] for b in batch], dim=0) for k in bev0.keys()}
        else:
            out["bev_labels"] = torch.stack([b["bev_labels"] for b in batch], dim=0)

        if "bev_selected_idx" in batch[0]:
            sel0 = batch[0]["bev_selected_idx"]
            if isinstance(sel0, dict):
                out["bev_selected_idx"] = {k: torch.stack([b["bev_selected_idx"][k] for b in batch], dim=0) for k in sel0.keys()}
            else:
                out["bev_selected_idx"] = torch.stack([b["bev_selected_idx"] for b in batch], dim=0)

    return out
