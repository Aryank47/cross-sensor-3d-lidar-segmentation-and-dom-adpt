#!/usr/bin/env python3
"""
Domain EDA for ECLAIR (LAZ) vs DALES (LAS)

Outputs:
- out_dir/meta_files.jsonl                     (per-file LAS header meta)
- out_dir/raw_file_stats.csv                   (per-file raw stats)
- out_dir/raw_feature_hists.npz                (global histograms for apples-to-apples features)
- out_dir/harmonized_patch_stats.csv           (per 100m×100m patch stats + voxel occupancy stats)
- out_dir/domain_gap_ranked.csv                (ranked mismatch metrics across datasets)
- out_dir/plots/*.png                          (quick visual summaries)

Design goals:
- No assumptions about label IDs beyond what exists in the file (we log raw codes).
- Apple-to-apple focus: xyz + intensity + return_number + number_of_returns (+ classification).
- Two passes: RAW and HARMONIZED (tiling + ECLAIR-style voxelization config).

ECLAIR baseline preprocessing details used as DEFAULTS:
- tiles cropped to max 100m×100m
- normalization before quantization, voxel size 0.05, coord norm factor 10.0
- intensity scaled to [0,1], return_number and number_of_returns one-hot
(see ECLAIR paper technical details)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import laspy
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional distances (nice-to-have, not required)
try:
    from scipy.stats import wasserstein_distance
except Exception:
    wasserstein_distance = None

# ----------------------------
# Config
# ----------------------------


@dataclass
class EDAConfig:
    # Harmonization / tiling (ECLAIR uses max 100m×100m tiles)
    patch_size_m: float = 100.0
    patch_stride_m: float = 100.0

    # ECLAIR voxelization defaults (per paper)
    voxel_size: float = 0.05
    coord_norm_factor: float = (
        10.0  # applied before voxelization (we implement as division)
    )

    # Streaming / sampling
    chunk_size_points: int = 2_000_000
    sample_points_per_file: int = (
        1_000_000  # used for quantiles (histograms are exact via streaming)
    )
    seed: int = 0

    # Histogram settings (apple-to-apple)
    intensity_bins: int = 256
    # For LAS intensity (often 0..65535); we keep fixed range for comparability.
    intensity_range: Tuple[int, int] = (0, 65535)

    # Return hist ranges (DALES notes up to 4 returns)
    max_return_value: int = 8

    # Classification hist range (LAS classification stored in uint8 typically)
    class_bins: int = 256


# ----------------------------
# Utilities
# ----------------------------


def safe_has_dim(dim_names: List[str], dim: str) -> bool:
    return dim in set(dim_names)


def read_dim(points, dim: str) -> np.ndarray:
    # laspy point records support dict-like access
    return np.asarray(points[dim])


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)


def las_header_meta(path: Path) -> Dict:
    with laspy.open(str(path)) as f:
        h = f.header
        meta = {
            "path": str(path),
            "version": str(h.version),
            "point_format_id": int(h.point_format.id),
            "point_count": int(h.point_count),
            "scales": list(map(float, h.scales)),
            "offsets": list(map(float, h.offsets)),
            "mins": list(map(float, h.mins)),
            "maxs": list(map(float, h.maxs)),
            "dim_names": list(f.header.point_format.dimension_names),
        }
    return meta


def choose_chunk_sample_indices(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    if m <= 0:
        return np.empty((0,), dtype=np.int64)
    if m >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=m, replace=False)


def fixed_hist_intensity(vals: np.ndarray, bins: int, lo: int, hi: int) -> np.ndarray:
    # vals might be float or int; clamp to [lo, hi]
    v = np.asarray(vals)
    v = np.clip(v, lo, hi)
    # use np.histogram for stable bins
    hist, _ = np.histogram(v, bins=bins, range=(lo, hi))
    return hist.astype(np.int64)


def bincount_safe(vals: np.ndarray, minlength: int) -> np.ndarray:
    v = np.asarray(vals)
    v = v.astype(np.int64, copy=False)
    v = np.clip(v, 0, minlength - 1)
    return np.bincount(v, minlength=minlength).astype(np.int64)


def quantiles_from_sample(
    sample: np.ndarray, qs=(0.0, 0.01, 0.5, 0.99, 1.0)
) -> List[float]:
    if sample.size == 0:
        return [float("nan")] * len(qs)
    return list(map(float, np.quantile(sample, qs)))


def pack_voxel_ids(
    ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, bits_x: int, bits_y: int
) -> np.ndarray:
    """
    Collision-free packing into uint64 given known bit widths.
    """
    ixu = ix.astype(np.uint64, copy=False)
    iyu = iy.astype(np.uint64, copy=False)
    izu = iz.astype(np.uint64, copy=False)
    return ixu | (iyu << bits_x) | (izu << (bits_x + bits_y))


def needed_bits(max_val_inclusive: int) -> int:
    if max_val_inclusive <= 0:
        return 1
    return int(math.ceil(math.log2(max_val_inclusive + 1)))


# ----------------------------
# Per-file RAW stats
# ----------------------------


@dataclass
class RawFileStats:
    dataset: str
    path: str
    point_count: int

    # Bounding box (physical units)
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    span_x: float
    span_y: float
    span_z: float

    # Density proxy (points / (span_x * span_y))
    density_points_per_m2_proxy: float

    # Quantiles from sample (apple-to-apple)
    intensity_q: List[float]
    return_number_q: List[float]
    number_of_returns_q: List[float]

    # Exact histograms (streamed)
    intensity_hist_sum: str  # saved separately to NPZ; here store key
    return_number_hist_sum: str
    number_of_returns_hist_sum: str
    class_hist_sum: str


def compute_raw_stats(
    dataset_name: str,
    paths: List[Path],
    cfg: EDAConfig,
    out_dir: Path,
    global_hist_prefix: str,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Computes per-file RAW stats; returns dataframe + dict of hist arrays to be saved in NPZ.
    Hist arrays are also aggregated globally per dataset for domain-gap ranking.
    """
    rng = np.random.default_rng(cfg.seed)

    all_rows: List[RawFileStats] = []

    # Global aggregated histograms per dataset
    global_hists = {
        f"{global_hist_prefix}_{dataset_name}_intensity": np.zeros(
            cfg.intensity_bins, dtype=np.int64
        ),
        f"{global_hist_prefix}_{dataset_name}_return_number": np.zeros(
            cfg.max_return_value, dtype=np.int64
        ),
        f"{global_hist_prefix}_{dataset_name}_number_of_returns": np.zeros(
            cfg.max_return_value, dtype=np.int64
        ),
        f"{global_hist_prefix}_{dataset_name}_class": np.zeros(
            cfg.class_bins, dtype=np.int64
        ),
    }

    for p in tqdm(paths, desc=f"[RAW] {dataset_name}", leave=False):
        meta = las_header_meta(p)
        N = int(meta["point_count"])
        min_x, min_y, min_z = meta["mins"]
        max_x, max_y, max_z = meta["maxs"]
        span_x = max_x - min_x
        span_y = max_y - min_y
        span_z = max_z - min_z
        area = max(span_x * span_y, 1e-9)
        rho = float(N / area)

        dim_names = meta["dim_names"]

        # Exact histograms via streaming
        intensity_hist = np.zeros(cfg.intensity_bins, dtype=np.int64)
        rn_hist = np.zeros(cfg.max_return_value, dtype=np.int64)
        nor_hist = np.zeros(cfg.max_return_value, dtype=np.int64)
        cls_hist = np.zeros(cfg.class_bins, dtype=np.int64)

        # Sampling for quantiles
        # Use per-chunk uniform sampling with expected sample size ~= cfg.sample_points_per_file
        sample_int = []
        sample_rn = []
        sample_nor = []

        with laspy.open(str(p)) as f:
            total = f.header.point_count
            p_sample = min(1.0, cfg.sample_points_per_file / max(1, total))
            for points in f.chunk_iterator(cfg.chunk_size_points):
                n = len(points)

                if safe_has_dim(dim_names, "intensity"):
                    vals = read_dim(points, "intensity")
                    intensity_hist += fixed_hist_intensity(
                        vals, cfg.intensity_bins, *cfg.intensity_range
                    )
                if safe_has_dim(dim_names, "return_number"):
                    vals = read_dim(points, "return_number")
                    rn_hist += bincount_safe(vals, cfg.max_return_value)
                if safe_has_dim(dim_names, "number_of_returns"):
                    vals = read_dim(points, "number_of_returns")
                    nor_hist += bincount_safe(vals, cfg.max_return_value)
                if safe_has_dim(dim_names, "classification"):
                    vals = read_dim(points, "classification")
                    cls_hist += bincount_safe(vals, cfg.class_bins)

                # quantile samples (approx-uniform)
                m = int(round(p_sample * n))
                idx = choose_chunk_sample_indices(rng, n, m)
                if idx.size > 0:
                    if safe_has_dim(dim_names, "intensity"):
                        sample_int.append(read_dim(points, "intensity")[idx])
                    if safe_has_dim(dim_names, "return_number"):
                        sample_rn.append(read_dim(points, "return_number")[idx])
                    if safe_has_dim(dim_names, "number_of_returns"):
                        sample_nor.append(read_dim(points, "number_of_returns")[idx])

        s_int = (
            np.concatenate(sample_int)
            if sample_int
            else np.empty((0,), dtype=np.float32)
        )
        s_rn = (
            np.concatenate(sample_rn) if sample_rn else np.empty((0,), dtype=np.float32)
        )
        s_nor = (
            np.concatenate(sample_nor)
            if sample_nor
            else np.empty((0,), dtype=np.float32)
        )

        # Downsample to cap memory if needed
        def cap_sample(arr: np.ndarray, k: int) -> np.ndarray:
            if arr.size <= k:
                return arr
            idx = rng.choice(arr.size, size=k, replace=False)
            return arr[idx]

        s_int = cap_sample(s_int, cfg.sample_points_per_file)
        s_rn = cap_sample(s_rn, cfg.sample_points_per_file)
        s_nor = cap_sample(s_nor, cfg.sample_points_per_file)

        # Update global hist sums
        global_hists[f"{global_hist_prefix}_{dataset_name}_intensity"] += intensity_hist
        global_hists[f"{global_hist_prefix}_{dataset_name}_return_number"] += rn_hist
        global_hists[
            f"{global_hist_prefix}_{dataset_name}_number_of_returns"
        ] += nor_hist
        global_hists[f"{global_hist_prefix}_{dataset_name}_class"] += cls_hist

        row = RawFileStats(
            dataset=dataset_name,
            path=str(p),
            point_count=N,
            min_x=min_x,
            min_y=min_y,
            min_z=min_z,
            max_x=max_x,
            max_y=max_y,
            max_z=max_z,
            span_x=span_x,
            span_y=span_y,
            span_z=span_z,
            density_points_per_m2_proxy=rho,
            intensity_q=quantiles_from_sample(s_int),
            return_number_q=quantiles_from_sample(s_rn),
            number_of_returns_q=quantiles_from_sample(s_nor),
            intensity_hist_sum=f"{global_hist_prefix}_{dataset_name}_intensity",
            return_number_hist_sum=f"{global_hist_prefix}_{dataset_name}_return_number",
            number_of_returns_hist_sum=f"{global_hist_prefix}_{dataset_name}_number_of_returns",
            class_hist_sum=f"{global_hist_prefix}_{dataset_name}_class",
        )
        all_rows.append(row)

    df = pd.DataFrame([asdict(r) for r in all_rows])
    return df, global_hists


# ----------------------------
# Harmonized per-patch stats (100m tiling + voxelization occupancy)
# ----------------------------


@dataclass
class PatchStats:
    dataset: str
    src_path: str
    patch_ix: int
    patch_iy: int

    # point counts / class counts
    point_count: int
    class_hist_nonzero: str  # json mapping class_id->count for compactness

    # bounds within patch (physical)
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    span_z: float

    # density and voxel stats
    patch_area_m2: float
    points_per_m2_proxy: float
    voxel_size: float
    coord_norm_factor: float
    occupied_voxels: int
    points_per_voxel_mean: float
    points_per_voxel_q: List[float]

    # feature samples for reporting
    intensity_q: List[float]
    return_number_q: List[float]
    number_of_returns_q: List[float]


def compute_harmonized_patch_stats(
    dataset_name: str,
    paths: List[Path],
    cfg: EDAConfig,
) -> pd.DataFrame:
    """
    Re-tiles each file into (patch_size_m, patch_stride_m) XY patches and computes:
    - per-patch point counts, class counts
    - per-patch min/max z, etc.
    - voxel occupancy stats using ECLAIR-style normalization + voxelization params
    """
    rng = np.random.default_rng(cfg.seed)
    out_rows: List[PatchStats] = []

    patch_size = cfg.patch_size_m
    stride = cfg.patch_stride_m
    assert (
        abs(stride - patch_size) < 1e-6
    ), "This script assumes non-overlapping patches for simplicity."

    # Precompute bits for voxel packing (xy known from patch size)
    # Implement normalization as division by coord_norm_factor before voxelization.
    nx = int(math.ceil((patch_size / cfg.coord_norm_factor) / cfg.voxel_size))
    ny = int(math.ceil((patch_size / cfg.coord_norm_factor) / cfg.voxel_size))
    bits_x = needed_bits(nx)
    bits_y = needed_bits(ny)

    for p in tqdm(paths, desc=f"[HARM] {dataset_name}", leave=False):
        meta = las_header_meta(p)
        dim_names = meta["dim_names"]
        if not (
            safe_has_dim(dim_names, "classification")
            and safe_has_dim(dim_names, "intensity")
            and safe_has_dim(dim_names, "return_number")
            and safe_has_dim(dim_names, "number_of_returns")
        ):
            # Still run, but note: core features missing
            pass

        min_x, min_y, min_z = meta["mins"]
        max_x, max_y, max_z = meta["maxs"]

        # Anchor patches on a 100m grid aligned to file min corner (floor to grid)
        ax = math.floor(min_x / patch_size) * patch_size
        ay = math.floor(min_y / patch_size) * patch_size

        # nz bits from file z-span
        z_span = max_z - min_z
        nz = int(math.ceil((z_span / cfg.coord_norm_factor) / cfg.voxel_size)) + 1
        bits_z = needed_bits(nz)
        if bits_x + bits_y + bits_z > 63:
            raise RuntimeError(
                f"Voxel packing exceeds 63 bits for {p} (increase voxel_size or change packing)."
            )

        # Per-patch accumulators
        patch_point_count: Dict[Tuple[int, int], int] = {}
        patch_min = {}
        patch_max = {}
        patch_class_hist: Dict[Tuple[int, int], np.ndarray] = {}
        patch_int_samples: Dict[Tuple[int, int], List[np.ndarray]] = {}
        patch_rn_samples: Dict[Tuple[int, int], List[np.ndarray]] = {}
        patch_nor_samples: Dict[Tuple[int, int], List[np.ndarray]] = {}

        # Voxel counts per patch stored as dict packed_id -> count (exact, collision-free packing)
        patch_voxel_counts: Dict[Tuple[int, int], Dict[int, int]] = {}

        with laspy.open(str(p)) as f:
            total = f.header.point_count
            p_sample = min(1.0, cfg.sample_points_per_file / max(1, total))
            for points in f.chunk_iterator(cfg.chunk_size_points):
                x = np.asarray(points.x)
                y = np.asarray(points.y)
                z = np.asarray(points.z)

                # patch indices
                pix = np.floor((x - ax) / patch_size).astype(np.int32)
                piy = np.floor((y - ay) / patch_size).astype(np.int32)

                # filter only points inside computed bounds (safety)
                valid = (x >= ax) & (y >= ay) & (pix >= 0) & (piy >= 0)
                if not np.any(valid):
                    continue
                x = x[valid]
                y = y[valid]
                z = z[valid]
                pix = pix[valid]
                piy = piy[valid]

                intensity = (
                    read_dim(points, "intensity")[valid]
                    if safe_has_dim(dim_names, "intensity")
                    else None
                )
                rn = (
                    read_dim(points, "return_number")[valid]
                    if safe_has_dim(dim_names, "return_number")
                    else None
                )
                nor = (
                    read_dim(points, "number_of_returns")[valid]
                    if safe_has_dim(dim_names, "number_of_returns")
                    else None
                )
                cls = (
                    read_dim(points, "classification")[valid]
                    if safe_has_dim(dim_names, "classification")
                    else None
                )

                # group by patch within this chunk
                patch_keys = np.stack([pix, piy], axis=1)
                uniq, inv = np.unique(patch_keys, axis=0, return_inverse=True)

                for k_i, (ux, uy) in enumerate(uniq):
                    key = (int(ux), int(uy))
                    mask = inv == k_i
                    n_k = int(mask.sum())
                    if n_k == 0:
                        continue

                    xk = x[mask]
                    yk = y[mask]
                    zk = z[mask]

                    # point counts and bounds
                    patch_point_count[key] = patch_point_count.get(key, 0) + n_k

                    mn = np.array([xk.min(), yk.min(), zk.min()], dtype=np.float64)
                    mx = np.array([xk.max(), yk.max(), zk.max()], dtype=np.float64)
                    if key not in patch_min:
                        patch_min[key] = mn
                        patch_max[key] = mx
                    else:
                        patch_min[key] = np.minimum(patch_min[key], mn)
                        patch_max[key] = np.maximum(patch_max[key], mx)

                    # class hist
                    if cls is not None:
                        if key not in patch_class_hist:
                            patch_class_hist[key] = np.zeros(
                                cfg.class_bins, dtype=np.int64
                            )
                        patch_class_hist[key] += bincount_safe(
                            cls[mask], cfg.class_bins
                        )

                    # samples for quantiles
                    m = int(round(p_sample * n_k))
                    idx_local = choose_chunk_sample_indices(rng, n_k, m)
                    if idx_local.size > 0:
                        patch_int_samples.setdefault(key, []).append(
                            intensity[mask][idx_local]
                            if intensity is not None
                            else np.empty((0,))
                        )
                        patch_rn_samples.setdefault(key, []).append(
                            rn[mask][idx_local] if rn is not None else np.empty((0,))
                        )
                        patch_nor_samples.setdefault(key, []).append(
                            nor[mask][idx_local] if nor is not None else np.empty((0,))
                        )

                    # voxelization occupancy (ECLAIR-style: normalize coords before quantization)
                    # Use patch-local XY and file-min Z for nonnegative indices.
                    x0 = ax + key[0] * patch_size
                    y0 = ay + key[1] * patch_size
                    xr = (xk - x0) / cfg.coord_norm_factor
                    yr = (yk - y0) / cfg.coord_norm_factor
                    zr = (zk - min_z) / cfg.coord_norm_factor

                    ix = np.floor(xr / cfg.voxel_size).astype(np.int32)
                    iy = np.floor(yr / cfg.voxel_size).astype(np.int32)
                    iz = np.floor(zr / cfg.voxel_size).astype(np.int32)

                    packed = pack_voxel_ids(ix, iy, iz, bits_x, bits_y)
                    uvox, cvox = np.unique(packed, return_counts=True)

                    vc = patch_voxel_counts.setdefault(key, {})
                    # update dict counts
                    for vv, cc in zip(uvox.tolist(), cvox.tolist()):
                        vc[vv] = vc.get(vv, 0) + int(cc)

        # finalize per-patch rows
        for (ux, uy), Np in patch_point_count.items():
            mn = patch_min[(ux, uy)]
            mx = patch_max[(ux, uy)]
            span_z = float(mx[2] - mn[2])

            area = patch_size * patch_size
            rho = float(Np / area)

            vc = patch_voxel_counts.get((ux, uy), {})
            occ = int(len(vc))
            ppv_mean = float(Np / max(1, occ))
            ppv_q = quantiles_from_sample(
                np.array(list(vc.values()), dtype=np.float64),
                qs=(0.0, 0.25, 0.5, 0.75, 0.99, 1.0),
            )

            # samples quantiles
            def merge_cap(samples_dict):
                if (ux, uy) not in samples_dict:
                    return np.empty((0,), dtype=np.float32)
                arr = (
                    np.concatenate(samples_dict[(ux, uy)])
                    if samples_dict[(ux, uy)]
                    else np.empty((0,), dtype=np.float32)
                )
                if arr.size > cfg.sample_points_per_file:
                    idx = rng.choice(
                        arr.size, size=cfg.sample_points_per_file, replace=False
                    )
                    arr = arr[idx]
                return arr

            s_int = merge_cap(patch_int_samples)
            s_rn = merge_cap(patch_rn_samples)
            s_nor = merge_cap(patch_nor_samples)

            ch = patch_class_hist.get(
                (ux, uy), np.zeros(cfg.class_bins, dtype=np.int64)
            )
            nz_ids = np.nonzero(ch)[0]
            compact = {int(i): int(ch[i]) for i in nz_ids.tolist()}

            out_rows.append(
                PatchStats(
                    dataset=dataset_name,
                    src_path=str(p),
                    patch_ix=int(ux),
                    patch_iy=int(uy),
                    point_count=int(Np),
                    class_hist_nonzero=json.dumps(compact),
                    min_x=float(mn[0]),
                    min_y=float(mn[1]),
                    min_z=float(mn[2]),
                    max_x=float(mx[0]),
                    max_y=float(mx[1]),
                    max_z=float(mx[2]),
                    span_z=span_z,
                    patch_area_m2=float(area),
                    points_per_m2_proxy=rho,
                    voxel_size=float(cfg.voxel_size),
                    coord_norm_factor=float(cfg.coord_norm_factor),
                    occupied_voxels=occ,
                    points_per_voxel_mean=ppv_mean,
                    points_per_voxel_q=ppv_q,
                    intensity_q=quantiles_from_sample(s_int),
                    return_number_q=quantiles_from_sample(s_rn),
                    number_of_returns_q=quantiles_from_sample(s_nor),
                )
            )

    return pd.DataFrame([asdict(r) for r in out_rows])


# ----------------------------
# Domain-gap ranking (simple, reproducible)
# ----------------------------


def normalize_hist(h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float64)
    s = h.sum()
    if s <= 0:
        return h
    return h / s


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def compute_domain_gap_ranked(
    global_hists: Dict[str, np.ndarray], prefix: str
) -> pd.DataFrame:
    """
    Ranks mismatches using JS divergence of global histograms.
    (Optional) Wasserstein distance if scipy is available and the feature is ordinal.
    """
    # Expect keys like: f"{prefix}_ECLAIR_intensity" etc.
    rows = []

    def add(feature: str, ordinal: bool):
        he = normalize_hist(global_hists[f"{prefix}_ECLAIR_{feature}"])
        hd = normalize_hist(global_hists[f"{prefix}_DALES_{feature}"])
        js = js_divergence(he, hd)
        w = None
        if ordinal and wasserstein_distance is not None:
            # Convert hist to empirical positions
            xe = np.arange(len(he), dtype=np.float64)
            xd = np.arange(len(hd), dtype=np.float64)
            w = float(wasserstein_distance(xe, xd, u_weights=he, v_weights=hd))
        rows.append({"feature": feature, "js_divergence": js, "wasserstein": w})

    add("intensity", ordinal=True)
    add("return_number", ordinal=True)
    add("number_of_returns", ordinal=True)
    add("class", ordinal=False)

    df = (
        pd.DataFrame(rows)
        .sort_values("js_divergence", ascending=False)
        .reset_index(drop=True)
    )
    return df


# ----------------------------
# Plotting (minimal, robust)
# ----------------------------


def save_quick_plots(
    global_hists: Dict[str, np.ndarray], prefix: str, out_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    def plot_pair(feature: str, title: str):
        he = normalize_hist(global_hists[f"{prefix}_ECLAIR_{feature}"])
        hd = normalize_hist(global_hists[f"{prefix}_DALES_{feature}"])
        x = np.arange(len(he))
        plt.figure()
        plt.plot(x, he, label="ECLAIR")
        plt.plot(x, hd, label="DALES")
        plt.title(title)
        plt.xlabel(feature)
        plt.ylabel("probability")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "plots" / f"{feature}.png", dpi=150)
        plt.close()

    plot_pair("intensity", "Global intensity distribution (normalized histogram)")
    plot_pair("return_number", "Global return_number distribution")
    plot_pair("number_of_returns", "Global number_of_returns distribution")
    plot_pair("class", "Global classification-code distribution (raw codes)")


# ----------------------------
# Main
# ----------------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eclair_glob", type=str, required=True, help="Glob for ECLAIR .laz tiles"
    )
    ap.add_argument(
        "--dales_glob", type=str, required=True, help="Glob for DALES .las tiles"
    )
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--patch_size_m", type=float, default=100.0)
    ap.add_argument("--patch_stride_m", type=float, default=100.0)
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--coord_norm_factor", type=float, default=10.0)

    ap.add_argument("--chunk_size_points", type=int, default=2_000_000)
    ap.add_argument("--sample_points_per_file", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# def glob_paths(pattern: str) -> List[Path]:
#     # Accept both direct file path and glob
#     p = Path(pattern)
#     if p.exists():
#         return [p]
#     # glob from parent if needed
#     return sorted(Path().glob(pattern))


def glob_paths(pattern: str) -> List[Path]:
    """
    Supports:
      - direct file path
      - relative glob
      - absolute glob (e.g. /scratch/.../*.las)
    """
    p = Path(pattern)

    # If user passed an exact existing path (file or directory)
    if p.exists():
        if p.is_file():
            return [p]
        # If it's a directory, return all LAS/LAZ inside (optional convenience)
        return sorted(list(p.rglob("*.las")) + list(p.rglob("*.laz")))

    # Otherwise treat as a glob (works for absolute and relative)
    matches = glob.glob(pattern)
    paths = sorted(Path(m) for m in matches)

    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    return paths


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    cfg = EDAConfig(
        patch_size_m=args.patch_size_m,
        patch_stride_m=args.patch_stride_m,
        voxel_size=args.voxel_size,
        coord_norm_factor=args.coord_norm_factor,
        chunk_size_points=args.chunk_size_points,
        sample_points_per_file=args.sample_points_per_file,
        seed=args.seed,
    )

    eclair_paths = glob_paths(args.eclair_glob)
    dales_paths = glob_paths(args.dales_glob)
    if not eclair_paths:
        raise FileNotFoundError(f"No ECLAIR files matched: {args.eclair_glob}")
    if not dales_paths:
        raise FileNotFoundError(f"No DALES files matched: {args.dales_glob}")

    # Save per-file meta
    meta_path = out_dir / "meta_files.jsonl"
    with meta_path.open("w") as f:
        for p in eclair_paths + dales_paths:
            f.write(json.dumps(las_header_meta(p)) + "\n")

    # RAW stats + global hist
    raw_e, h_e = compute_raw_stats(
        "ECLAIR", eclair_paths, cfg, out_dir, global_hist_prefix="global"
    )
    raw_d, h_d = compute_raw_stats(
        "DALES", dales_paths, cfg, out_dir, global_hist_prefix="global"
    )

    raw_all = pd.concat([raw_e, raw_d], axis=0, ignore_index=True)
    raw_all.to_csv(out_dir / "raw_file_stats.csv", index=False)

    global_hists = {}
    global_hists.update(h_e)
    global_hists.update(h_d)
    np.savez_compressed(out_dir / "raw_feature_hists.npz", **global_hists)

    # Harmonized patch stats
    harm_e = compute_harmonized_patch_stats("ECLAIR", eclair_paths, cfg)
    harm_d = compute_harmonized_patch_stats("DALES", dales_paths, cfg)
    harm_all = pd.concat([harm_e, harm_d], axis=0, ignore_index=True)
    harm_all.to_csv(out_dir / "harmonized_patch_stats.csv", index=False)

    # Domain gap ranking
    gap = compute_domain_gap_ranked(global_hists, prefix="global")
    gap.to_csv(out_dir / "domain_gap_ranked.csv", index=False)

    # Plots
    save_quick_plots(global_hists, prefix="global", out_dir=out_dir)

    # Save config used
    with (out_dir / "eda_config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"[done] Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
