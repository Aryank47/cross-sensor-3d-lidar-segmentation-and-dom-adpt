# src/data_dales.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import MinkowskiEngine as ME
import numpy as np
import torch

from .features import FeatureConfig, build_features
from .utils import read_las_arrays_robust


@dataclass
class DalesPatchConfig:
    make_local_coords: bool = True
    coord_norm_factor: float = 10.0
    voxel_size: float = 0.05  # in normalized space, like ECLAIR


@dataclass
class DalesPreprocConfig:
    # intensity preprocessing (label-free)
    intensity_mode: str = (
        "none"  # none | quantile_match | robust_standardize | constant
    )
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


def _find_dales_files(root: Union[str, Path]) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"DALES root not found: {root}")
    # recursively find LAS/LAZ
    files = sorted(list(root.rglob("*.las")) + list(root.rglob("*.laz")))
    if not files:
        raise RuntimeError(f"No .las/.laz files found under: {root}")
    return files


# def _read_dales_las(path: Path) -> Dict[str, np.ndarray]:
#     import laspy

#     las = laspy.read(str(path))

#     def _dim(name: str) -> Optional[np.ndarray]:
#         if name in set(las.point_format.dimension_names):
#             arr = las[name]
#             return getattr(arr, "array", arr)
#         return None

#     xyz = las.xyz.astype(np.float32, copy=True)

#     intensity = _dim("intensity")
#     return_number = _dim("return_number")
#     number_of_returns = _dim("number_of_returns")

#     # DALES labels are usually in "classification"
#     gt = _dim("classification")
#     if gt is None:
#         gt = _dim("raw_classification")
#     if gt is None:
#         raise RuntimeError(f"Missing classification labels in {path}")

#     return {
#         "xyz": xyz,
#         "intensity": intensity.astype(np.float32) if intensity is not None else None,
#         "return_number": (
#             return_number.astype(np.int64) if return_number is not None else None
#         ),
#         "number_of_returns": (
#             number_of_returns.astype(np.int64)
#             if number_of_returns is not None
#             else None
#         ),
#         "native_labels": gt.astype(np.int64),
#     }


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
        keys = (x_cell[idx].astype(np.int64) << 32) ^ (
            y_cell[idx].astype(np.int64) & 0xFFFFFFFF
        )
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
        ignore_index: int = -100,
        preproc: Optional[DalesPreprocConfig] = None,
        seed: int = 1234,
    ):
        self.root = Path(dales_root)
        self.files = _find_dales_files(self.root)
        self.patch_cfg = patch_cfg
        self.feat_cfg = feat_cfg
        self.ignore_index = ignore_index
        self.preproc = preproc or DalesPreprocConfig()
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        raw = _read_dales_las(path)

        xyz = raw["xyz"]
        if self.patch_cfg.make_local_coords:
            xyz = xyz - xyz.min(axis=0, keepdims=True)

        # normalize coords for voxelization like ECLAIR pipeline
        xyz_norm = (
            xyz.astype(np.float32) / float(self.patch_cfg.coord_norm_factor)
        ).astype(np.float32, copy=False)

        # intensity preprocessing happens in "scaled space" expected by FeatureConfig
        intensity = raw["intensity"].astype(np.float32, copy=False)
        div = (
            float(self.preproc.intensity_divisor_override)
            if self.preproc.intensity_divisor_override
            else float(self.feat_cfg.intensity_divisor)
        )

        if intensity is not None and self.feat_cfg.use_intensity:
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
                    raise RuntimeError(
                        "quantile_match requires ref_quantiles/ref_probs and tgt_quantiles/tgt_probs"
                    )
                intensity_scaled = _quantile_match(
                    intensity_scaled,
                    self.preproc.ref_quantiles,
                    self.preproc.ref_probs,
                    self.preproc.tgt_quantiles,
                    self.preproc.tgt_probs,
                )
            else:
                raise ValueError(
                    f"Unknown intensity_mode: {self.preproc.intensity_mode}"
                )

            # clip to [0,1] range
            intensity_scaled = np.clip(intensity_scaled, 0.0, 1.0).astype(
                np.float32, copy=False
            )
        else:
            intensity_scaled = None

        feats = build_features(
            xyz_local=xyz_norm,  # coords optionally included inside build_features via feat_cfg
            intensity=intensity_scaled,
            return_number=raw["return_number"],
            number_of_returns=raw["number_of_returns"],
            rgb=None,
            cfg=self.feat_cfg,
        )

        # voxel quantize
        q = np.floor(xyz_norm / float(self.patch_cfg.voxel_size)).astype(np.int32)
        q = np.ascontiguousarray(q, dtype=np.int32)  # important for ME
        _, unique_idx = ME.utils.sparse_quantize(q, return_index=True)

        q_u = q[unique_idx]
        feats_u = feats[unique_idx]
        y_u = raw["native_labels"][unique_idx].astype(np.int64, copy=False)

        # OPTIONAL: label-free thinning that tries to reduce low-height clutter
        if self.preproc.use_height_xy_cap:
            xyz_u_m = (
                xyz_norm[unique_idx] * float(self.patch_cfg.coord_norm_factor)
            ).astype(np.float32, copy=False)
            xyz_u_m, feats_u, y_u = _height_xy_cap(
                xyz_u_m,
                feats_u,
                y_u,
                xy_cell_size_m=float(self.preproc.xy_cell_size_m),
                height_bins=int(self.preproc.height_bins),
                caps_per_bin=self.preproc.caps_per_bin,
                rng=self.rng,
            )
            # recompute q_u consistent with thinned xyz_u_m
            xyz_u_norm = (xyz_u_m / float(self.patch_cfg.coord_norm_factor)).astype(
                np.float32, copy=False
            )
            q_u = np.floor(xyz_u_norm / float(self.patch_cfg.voxel_size)).astype(
                np.int32
            )
            q_u = np.ascontiguousarray(q_u, dtype=np.int32)

        coords_t = torch.from_numpy(q_u).int()
        feats_t = torch.from_numpy(np.ascontiguousarray(feats_u)).float()
        labels_t = torch.from_numpy(np.ascontiguousarray(y_u)).long()

        return {
            "coords": coords_t,
            "feats": feats_t,
            "labels": labels_t,  # DALES native labels (0..8)
            "path": str(path),
        }


def minkowski_collate_dales(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    coords_list = [b["coords"] for b in batch]
    feats_list = [b["feats"] for b in batch]
    labels_list = [b["labels"] for b in batch]

    coords, feats, labels = ME.utils.sparse_collate(
        coords_list, feats_list, labels_list
    )
    paths = [b["path"] for b in batch]
    return {"coords": coords, "feats": feats, "labels": labels, "paths": paths}
