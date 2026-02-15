# src/data_dales.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# add near imports
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import MinkowskiEngine as ME
import numpy as np
import torch

from .features import FeatureConfig, build_features
from .utils import atomic_save_torch, read_las_arrays_robust


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
    max_voxels: Optional[int] = 200000

    # NEW: make voxel cap class-aware to protect rare classes.
    # When True, we keep all voxels belonging to rare_class_ids (if they fit),
    # and only downsample the rest.
    class_aware_max_voxels: bool = False
    # Train IDs of rare classes you want to protect.
    # For your mapping and class_names:
    #   0 ground, 1 vegetation, 2 cars, 3 trucks, 4 buildings,
    #   5 poles, 6 power_lines, 7 fences
    rare_class_ids: Tuple[int, ...] = (2, 3, 5, 6, 7)


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


def _stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


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
) -> str:
    """
    Stable cache key for a DALES file given feature/voxelization/preproc settings.
    NOTE: includes preproc array *digests* (not full arrays) to keep key compact.
    """
    key_obj = {
        "v": "dales_cache_v1",
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
        },
        "extra": cache_key_extra,
        "labels": {
            "ignore_index": int(ignore_index),
            "label_policy": "native_to_train_lut",
        },
        "label_map": (None if label_map is None else dict(sorted((int(k), int(v)) for k, v in label_map.items()))),
    }
    s = json.dumps(key_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha1_hex(s)[:24]


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
        use_cache: bool = True,
        cache_root: Optional[str | Path] = None,
        cache_subdir: str = "dales",
        cache_key_extra: Optional[str] = None,
        require_cache: bool = False,
        write_cache: bool = False,
        split_name: Optional[str] = None,
        files: Optional[List[Path]] = None,
        label_map: Optional[Dict[int, int]] = None,
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

        if self.use_cache and self.cache_root is None:
            raise ValueError("DalesTiles: use_cache=True requires cache_root to be set.")

        if self.use_cache:
            base = self.cache_root / self.cache_subdir
            if self.split_name:
                base = base / self.split_name
            base.mkdir(parents=True, exist_ok=True)
            self._cache_dir = base
        else:
            self._cache_dir = None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]

        # Cache fast-path
        cache_path: Optional[Path] = None
        if self.use_cache:
            key = _cache_key_for_dales(
                path=path,
                patch_cfg=self.patch_cfg,
                feat_cfg=self.feat_cfg,
                preproc=self.preproc,
                cache_key_extra=self.cache_key_extra,
                ignore_index=self.ignore_index,
                label_map=self.label_map,
            )
            cache_path = self._cache_dir / f"{key}.pt"
            if cache_path.exists():
                return torch.load(cache_path, map_location="cpu")
            if self.require_cache:
                raise RuntimeError(f"[DALES cache missing] {cache_path}")

        raw = _read_dales_las(path)

        xyz = raw["xyz"]
        if self.patch_cfg.make_local_coords:
            xyz = xyz - xyz.min(axis=0, keepdims=True)

        # normalize coords for voxelization like ECLAIR pipeline
        xyz_norm = (xyz.astype(np.float32) / float(self.patch_cfg.coord_norm_factor)).astype(np.float32, copy=False)

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
        else:
            intensity_scaled = None

        # feats = build_features(
        #     xyz_local=xyz_norm,  # coords optionally included inside build_features via feat_cfg
        #     intensity=intensity_scaled,
        #     return_number=raw["return_number"],
        #     number_of_returns=raw["number_of_returns"],
        #     rgb=None,
        #     cfg=self.feat_cfg,
        # )

        # # voxel quantize
        # q = np.floor(xyz_norm / float(self.patch_cfg.voxel_size)).astype(np.int32)
        # q = np.ascontiguousarray(q, dtype=np.int32)  # important for ME
        # _, unique_idx = ME.utils.sparse_quantize(q, return_index=True)

        # # ---- OOM guard: cap #voxels ----
        # max_vox = self.preproc.max_voxels
        # if max_vox is not None:
        #     max_vox = int(max_vox)
        #     if unique_idx.shape[0] > max_vox:
        #         # deterministic-ish per file: use dataset RNG
        #         rng = np.random.default_rng(self.seed + (hash(path.name) & 0xFFFFFFFF))
        #         sel = rng.choice(unique_idx.shape[0], size=max_vox, replace=False)
        #         sel.sort()  # stable order
        #         unique_idx = unique_idx[sel]

        # q_u = q[unique_idx]
        # feats_u = feats[unique_idx]
        # # y_u = raw["native_labels"][unique_idx].astype(np.int64, copy=False)
        # # # -------------------------
        # # # DALES label normalization
        # # # -------------------------
        # # # DALES commonly uses:
        # # #   0 = unlabeled / unclassified
        # # #   1..8 = the 8 semantic classes (ground..veg)
        # # #
        # # # We want training IDs:
        # # #   ignore_index for unlabeled
        # # #   0..7 for the 8 semantic classes
        # # #
        # # # Steps:
        # # #   (1) map 0 -> ignore_index
        # # #   (2) map 1..8 -> 0..7 (shift by -1)
        # # ignore = int(self.ignore_index)

        # # # safety: anything outside 0..8 -> ignore
        # # invalid = (y_u < 0) | (y_u > 8)
        # # if np.any(invalid):
        # #     y_u = y_u.copy()
        # #     y_u[invalid] = ignore

        # # # 0 -> ignore
        # # if ignore != 0:
        # #     y_u = y_u.copy()  # because we'll mutate
        # #     y_u[y_u == 0] = ignore

        # # # shift 1..8 -> 0..7 (only for non-ignored)
        # # mask = y_u != ignore
        # # y_u[mask] = y_u[mask] - 1

        # y_native = raw["native_labels"][unique_idx].astype(np.int64, copy=False)

        # if self.label_lut is not None:
        #     # Map native -> train_id (and ignore)
        #     y_safe = np.clip(y_native, 0, 255).astype(np.int64, copy=False)
        #     y_u = self.label_lut[y_safe]
        # else:
        #     # If no mapping provided, default to ignoring unknown and shifting 1..8 -> 0..7
        #     # NOTE: This default assumes train order matches native order, which is NOT your case
        #     # because buildings=8 is in the middle of your class order.
        #     # So: better to REQUIRE mapping for DALES runs.
        #     raise RuntimeError("DALES label_map is required for this 8-class setup.")

        # valid = y_u != self.ignore_index
        # if np.any(valid):
        #     mn = int(y_u[valid].min())
        #     mx = int(y_u[valid].max())
        #     if mn < 0 or mx >= 8:
        #         raise RuntimeError(
        #             f"DALES label mapping out of range: min={mn}, max={mx}"
        #         )

        # # OPTIONAL: label-free thinning that tries to reduce low-height clutter
        # if self.preproc.use_height_xy_cap:
        #     xyz_u_m = (
        #         xyz_norm[unique_idx] * float(self.patch_cfg.coord_norm_factor)
        #     ).astype(np.float32, copy=False)
        #     xyz_u_m, feats_u, y_u = _height_xy_cap(
        #         xyz_u_m,
        #         feats_u,
        #         y_u,
        #         xy_cell_size_m=float(self.preproc.xy_cell_size_m),
        #         height_bins=int(self.preproc.height_bins),
        #         caps_per_bin=self.preproc.caps_per_bin,
        #         rng=self.rng,
        #     )
        #     # recompute q_u consistent with thinned xyz_u_m
        #     xyz_u_norm = (xyz_u_m / float(self.patch_cfg.coord_norm_factor)).astype(
        #         np.float32, copy=False
        #     )
        #     q_u = np.floor(xyz_u_norm / float(self.patch_cfg.voxel_size)).astype(
        #         np.int32
        #     )
        #     q_u = np.ascontiguousarray(q_u, dtype=np.int32)

        # coords_t = torch.from_numpy(q_u).int()
        # feats_t = torch.from_numpy(np.ascontiguousarray(feats_u)).float()
        # labels_t = torch.from_numpy(np.ascontiguousarray(y_u)).long()
        # # DEBUG: confirm actual voxel count after cap
        # if idx == 0:
        #     print(
        #         f"[DALES] file={path.name} voxels_after_cap={unique_idx.shape[0]}",
        #         flush=True,
        #     )
        #     u, c = np.unique(y_u, return_counts=True)
        #     print(
        #         "[DALES labels mapped] unique:",
        #         list(zip(u.tolist(), c.tolist()))[:20],
        #         flush=True,
        #     )
        # out = {
        #     "coords": coords_t,
        #     "feats": feats_t,
        #     "labels": labels_t,  # train labels: ignore_index or [0..7]
        #     "path": str(path),
        # }

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
        _, unique_idx = ME.utils.sparse_quantize(q, return_index=True)
        n_vox = unique_idx.shape[0]

        # -------------------------
        # Label mapping for all voxels (before any cap)
        # -------------------------
        y_native_all = raw["native_labels"][unique_idx].astype(np.int64, copy=False)

        if self.label_lut is not None:
            # Map native -> train_id (and ignore) for all voxels
            y_safe_all = np.clip(y_native_all, 0, 255).astype(np.int64, copy=False)
            y_all = self.label_lut[y_safe_all]
        else:
            # For your 8-class DALES setup, we *expect* a mapping.
            raise RuntimeError("DALES label_map is required for this 8-class setup.")

        # -------------------------
        # OOM guard: class-aware voxel cap (optional)
        # -------------------------
        max_vox = self.preproc.max_voxels
        if (max_vox is not None) and (n_vox > int(max_vox)):
            max_vox = int(max_vox)

            # deterministic-ish per file: use dataset RNG + file name hash
            rng = np.random.default_rng(self.seed + (hash(path.name) & 0xFFFFFFFF))
            idx_all = np.arange(n_vox, dtype=np.int64)

            if self.preproc.class_aware_max_voxels:
                # Protect rare classes: cars, trucks, poles, power_lines, fences
                rare_ids = np.asarray(self.preproc.rare_class_ids, dtype=np.int64)
                is_rare = np.isin(y_all, rare_ids)

                rare_idx = idx_all[is_rare]
                common_idx = idx_all[~is_rare]

                if rare_idx.size >= max_vox:
                    # Even rare classes alone exceed capacity; sample among them.
                    keep_local = rng.choice(rare_idx, size=max_vox, replace=False)
                else:
                    # Keep all rare-class voxels, fill the rest with common ones.
                    remaining = max_vox - rare_idx.size
                    if common_idx.size <= remaining:
                        # everything fits, no extra downsampling needed
                        keep_local = idx_all
                    else:
                        common_sample = rng.choice(common_idx, size=remaining, replace=False)
                        keep_local = np.concatenate([rare_idx, common_sample])

                keep_local.sort()
            else:
                # Old behaviour: uniform downsampling over all voxels
                keep_local = rng.choice(n_vox, size=max_vox, replace=False)
                keep_local.sort()

            unique_idx = unique_idx[keep_local]
            y_all = y_all[keep_local]
        else:
            # no cap or no need to cap: use all voxels
            pass

        # Final ME-ready arrays after any cap
        q_u = q[unique_idx]
        feats_u = feats[unique_idx]
        y_u = y_all

        # Basic range sanity check (train ids must be in [0..7] except ignore)
        valid = y_u != self.ignore_index
        if np.any(valid):
            mn = int(y_u[valid].min())
            mx = int(y_u[valid].max())
            if mn < 0 or mx >= 8:
                raise RuntimeError(f"DALES label mapping out of range: min={mn}, max={mx}")

        # OPTIONAL: label-free thinning in XYxZ grid
        if self.preproc.use_height_xy_cap:
            xyz_u_m = (xyz_norm[unique_idx] * float(self.patch_cfg.coord_norm_factor)).astype(np.float32, copy=False)
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
            xyz_u_norm = (xyz_u_m / float(self.patch_cfg.coord_norm_factor)).astype(np.float32, copy=False)
            q_u = np.floor(xyz_u_norm / float(self.patch_cfg.voxel_size)).astype(np.int32)
            q_u = np.ascontiguousarray(q_u, dtype=np.int32)

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

        # Cache write (ONLY for precompute jobs; do not enable in multi-worker training)
        if self.use_cache and (cache_path is not None) and self.write_cache:
            # atomic write avoids partial files under multi-worker dataloaders
            try:
                atomic_save_torch(out, cache_path)
            except Exception:
                # best-effort: never break training due to cache write issues
                pass

        return out


def minkowski_collate_dales(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    coords_list = [b["coords"] for b in batch]
    feats_list = [b["feats"] for b in batch]
    labels_list = [b["labels"] for b in batch]

    coords, feats, labels = ME.utils.sparse_collate(coords_list, feats_list, labels_list)
    paths = [b["path"] for b in batch]
    return {"coords": coords, "feats": feats, "labels": labels, "paths": paths}
