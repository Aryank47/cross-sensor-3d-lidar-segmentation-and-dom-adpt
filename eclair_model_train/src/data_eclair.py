# src/data_eclair.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .augment import AugmentConfig, augment_xyz
from .features import FeatureConfig, build_features
from .label_maps import eclair_native_to_train_ids
from .utils import read_las_arrays_robust


@dataclass
class PatchConfig:
    make_local_coords: bool = True
    coord_norm_factor: float = 10.0
    voxel_size: float = 0.05


def _load_eclair_split_list(
    eclair_root: Union[str, Path],
    split: str,
    *,
    meta_filename: str = "labels.json",
    allowed_review_categories: Sequence[str] = ("approved",),
) -> List[str]:
    """
    Returns a list of tile filenames for the given split.

    Supports BOTH common metadata layouts:
      (A) dict-of-splits:
          {"train": ["pointcloud_1.laz", ...], "val": [...], "test": [...]}

      (B) list-of-records (your format):
          [
            {"tile_name": "pointcloud_1.laz", "split": "train", "review_category": "approved", ...},
            ...
          ]

    Parameters
    ----------
    allowed_review_categories:
        If provided, list-format metadata will be filtered by review_category.
        Use allowed_review_categories=None to disable filtering.
    """
    eclair_root = Path(eclair_root)
    meta_path = eclair_root / meta_filename
    if not meta_path.exists():
        raise FileNotFoundError(f"ECLAIR meta file not found: {meta_path}")

    with meta_path.open("r") as f:
        meta = json.load(f)

    split = str(split).lower().strip()
    valid_splits = {"train", "val", "test"}

    if split not in valid_splits:
        raise ValueError(f"Unknown split='{split}'. Expected one of {sorted(valid_splits)}")

    # ---- Case A: dict-of-splits ----
    if isinstance(meta, dict):
        # allow either lowercase or exact keys
        keys_lower = {str(k).lower(): k for k in meta.keys()}
        if split not in keys_lower:
            raise KeyError(f"{meta_path} does not contain split '{split}'. Keys: {list(meta.keys())}")
        names = meta[keys_lower[split]]
        if not isinstance(names, list):
            raise TypeError(f"{meta_path}[{keys_lower[split]}] must be a list, got: {type(names)}")
        return sorted([str(x) for x in names])

    # ---- Case B: list-of-records ----
    if isinstance(meta, list):
        out = []
        for rec in meta:
            if not isinstance(rec, dict):
                continue
            rec_split = str(rec.get("split", "")).lower().strip()
            if rec_split != split:
                continue

            if allowed_review_categories is not None:
                cat = str(rec.get("review_category", "")).lower().strip()
                if cat not in {c.lower() for c in allowed_review_categories}:
                    continue

            name = rec.get("tile_name", None)
            if name is None:
                continue

            out.append(str(name))

        if not out:
            # Helpful debugging context
            present_splits = sorted(
                {str(r.get("split", "")).lower().strip() for r in meta if isinstance(r, dict) and "split" in r}
            )
            raise RuntimeError(
                f"No tiles found for split='{split}' after filtering.\n"
                f"meta_path={meta_path}\n"
                f"present_splits={present_splits}\n"
                f"allowed_review_categories={allowed_review_categories}"
            )

        return sorted(list(dict.fromkeys(out)))  # stable unique

    raise TypeError(f"Unsupported JSON structure in {meta_path}: {type(meta)}")


def _resolve_pc_path(eclair_root: Path, fname: str) -> Path:
    pc_dir = eclair_root / "pointclouds"
    p = pc_dir / fname
    if p.exists():
        return p
    # fallback: maybe stored as .las
    alt = p.with_suffix(".las")
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not find pointcloud file for '{fname}' under {pc_dir}")


# def _read_las_arrays(path: Path) -> Dict[str, np.ndarray]:
#     import laspy

#     # laspy.read loads all points in memory; ok for ECLAIR tiles
#     las = laspy.read(str(path))

#     # xyz float64 -> float32
#     xyz = las.xyz.astype(np.float32, copy=True)

#     def _dim(name: str) -> Optional[np.ndarray]:
#         if name in set(las.point_format.dimension_names):
#             arr = las[name]
#             # laspy sometimes returns a SubFieldView
#             return getattr(arr, "array", arr)
#         return None

#     intensity = _dim("intensity")
#     return_number = _dim("return_number")
#     number_of_returns = _dim("number_of_returns")

#     # ECLAIR labels can be in 'classification' or 'raw_classification' depending on export
#     gt_key = (
#         "classification"
#         if "classification" in set(las.point_format.dimension_names)
#         else "raw_classification"
#     )
#     native_labels = _dim(gt_key)

#     rgb = None
#     if all(
#         k in set(las.point_format.dimension_names) for k in ("red", "green", "blue")
#     ):
#         r = _dim("red").astype(np.float32)
#         g = _dim("green").astype(np.float32)
#         b = _dim("blue").astype(np.float32)
#         # ECLAIR sometimes stores 16-bit colors; we will scale later if enabled.
#         rgb = np.stack([r, g, b], axis=1)

#     if native_labels is None:
#         raise RuntimeError(
#             f"Missing classification labels in {path} (looked for 'classification' or 'raw_classification')"
#         )

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
#         "rgb": rgb,
#         "native_labels": native_labels.astype(np.int64),
#     }


def _read_las_arrays(path: Path) -> Dict[str, np.ndarray]:
    """
    Robust reader wrapper for ECLAIR training.
    """
    # Matches old output: xyz, intensity, return_number, number_of_returns, rgb, native_labels
    return read_las_arrays_robust(path)


class EclairTiles(Dataset):
    """
    ECLAIR tile dataset (per-tile samples).
    Produces MinkowskiEngine-ready (coords, feats, labels) AFTER quantization (one label per voxel).

    This class supports optional .pt caching for faster I/O:
      cache_root/<split>/<tile_stem>.pt
    """

    def __init__(
        self,
        *,
        eclair_root: str | Path,
        split: str,
        is_train: bool,
        patch_cfg: PatchConfig,
        aug_cfg: AugmentConfig,
        feat_cfg: FeatureConfig,
        ignore_index: int = -100,
        undefined_id: int = 0,
        use_cache: bool = True,
        cache_root: Optional[str | Path] = None,
        seed: int = 1337,
        meta_filename: str = "labels.json",
        allowed_review_categories: Optional[Sequence[str]] = ("approved",),
    ):
        self.eclair_root = Path(eclair_root)
        self.split = split
        self.is_train = is_train
        self.patch_cfg = patch_cfg
        self.aug_cfg = aug_cfg
        self.feat_cfg = feat_cfg
        self.ignore_index = ignore_index
        self.undefined_id = undefined_id

        self.names = _load_eclair_split_list(
            self.eclair_root,
            split,
            meta_filename=meta_filename,
            allowed_review_categories=allowed_review_categories,
        )

        self.use_cache = use_cache and cache_root is not None
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self._rng = np.random.default_rng(seed + (0 if split == "train" else 10))

    def __len__(self) -> int:
        return len(self.names)

    def _cache_path(self, fname: str) -> Optional[Path]:
        if not self.use_cache or self.cache_root is None:
            return None
        stem = Path(fname).stem
        return self.cache_root / self.split / f"{stem}.pt"

    def _load_raw(self, fname: str):
        # 1) Try cache
        cpath = self._cache_path(fname)
        if cpath is not None and cpath.exists():
            obj = torch.load(cpath, map_location="cpu")

            # Case A: old cache stored a PyG Data object
            if isinstance(obj, Data):
                # Support both .xyz or .pos
                if hasattr(obj, "xyz"):
                    xyz = obj.xyz
                elif hasattr(obj, "pos"):
                    xyz = obj.pos
                else:
                    raise KeyError(f"{cpath} Data missing xyz/pos attributes")

                # Old cache often stored raw labels in .classification
                if hasattr(obj, "classification"):
                    native_labels = obj.classification
                elif hasattr(obj, "y"):
                    native_labels = obj.y
                else:
                    raise KeyError(f"{cpath} Data missing classification/y")

                def to_np(x):
                    if x is None:
                        return None
                    if isinstance(x, torch.Tensor):
                        return x.detach().cpu().numpy()
                    return np.asarray(x)

                out = {
                    "xyz": to_np(xyz).astype(np.float64, copy=False),
                    "native_labels": to_np(native_labels).astype(np.int64, copy=False),
                    "intensity": to_np(getattr(obj, "intensity", None)),
                    "return_number": to_np(getattr(obj, "return_number", None)),
                    "number_of_returns": to_np(getattr(obj, "number_of_returns", None)),
                    "rgb": to_np(getattr(obj, "rgb", None)),
                }

                if out["intensity"] is not None:
                    out["intensity"] = out["intensity"].astype(np.float32, copy=False)
                if out["return_number"] is not None:
                    out["return_number"] = out["return_number"].astype(np.int64, copy=False)
                if out["number_of_returns"] is not None:
                    out["number_of_returns"] = out["number_of_returns"].astype(np.int64, copy=False)
                if out["rgb"] is not None:
                    out["rgb"] = out["rgb"].astype(np.float32, copy=False)

                return out

            # Case B: new raw-cache dict (the script above writes this)
            if isinstance(obj, dict) and "xyz" in obj and "native_labels" in obj:
                # these are usually numpy arrays already, just ensure dtypes
                xyz = obj["xyz"].astype(np.float64, copy=False)
                y = obj["native_labels"].astype(np.int64, copy=False)
                intensity = obj.get("intensity", None)
                rn = obj.get("return_number", None)
                nor = obj.get("number_of_returns", None)
                rgb = obj.get("rgb", None)

                if intensity is not None:
                    intensity = intensity.astype(np.float32, copy=False)
                if rn is not None:
                    rn = rn.astype(np.int64, copy=False)
                if nor is not None:
                    nor = nor.astype(np.int64, copy=False)
                if rgb is not None:
                    rgb = rgb.astype(np.float32, copy=False)

                return {
                    "xyz": xyz,
                    "native_labels": y,
                    "intensity": intensity,
                    "return_number": rn,
                    "number_of_returns": nor,
                    "rgb": rgb,
                }

            raise TypeError(f"Unsupported cache object type in {cpath}: {type(obj)}")

        # 2) No cache -> read LAS/LAZ
        pc_path = _resolve_pc_path(self.eclair_root, fname)
        return _read_las_arrays(pc_path)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fname = self.names[idx]
        raw = self._load_raw(fname)

        xyz = raw["xyz"]
        if self.patch_cfg.make_local_coords:
            xyz = xyz - xyz.min(axis=0, keepdims=True)

        # Deterministic RNG per sample for reproducible augmentation (while still random across epochs).
        # We mix in global RNG state to vary each epoch naturally.
        sample_seed = int(self._rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sample_seed)

        if self.is_train:
            xyz = augment_xyz(xyz, self.aug_cfg, rng)

        # Normalize coordinates before quantization
        xyz_norm = (xyz / float(self.patch_cfg.coord_norm_factor)).astype(np.float32, copy=False)
        xyz_norm = np.ascontiguousarray(xyz_norm, dtype=np.float32)

        # Labels: native -> contiguous train ids (ignore undefined)
        y = eclair_native_to_train_ids(
            raw["native_labels"],
            undefined_id=self.undefined_id,
            ignore_index=self.ignore_index,
        )

        # # Quantize coords
        # q = np.floor(xyz_norm / float(self.patch_cfg.voxel_size)).astype(np.int32)

        # # Sparse quantize (deduplicate voxels)
        # # return_index gives indices of unique points
        # _, unique_idx = ME.utils.sparse_quantize(q, return_index=True)
        # q_u = q[unique_idx]
        # feats_u = feats[unique_idx]
        # y_u = y[unique_idx]

        # # Convert to torch
        # coords_t = torch.from_numpy(q_u).int()
        # feats_t = torch.from_numpy(feats_u).float()
        # labels_t = torch.from_numpy(y_u).long()

        feats = build_features(
            xyz_local=(xyz_norm if self.feat_cfg.include_coords else xyz_norm),  # coords included handled in build_features
            intensity=raw["intensity"],
            return_number=raw["return_number"],
            number_of_returns=raw["number_of_returns"],
            rgb=raw["rgb"],
            cfg=self.feat_cfg,
        )

        # Quantize coords
        q = np.floor(xyz_norm / float(self.patch_cfg.voxel_size)).astype(np.int32, copy=False)
        q = np.ascontiguousarray(q, dtype=np.int32)  # <<< CRITICAL

        # Sparse quantize (deduplicate voxels) — use torch + contiguous explicitly
        q_t = torch.from_numpy(q).int().contiguous()
        _, unique_idx_t = ME.utils.sparse_quantize(q_t, return_index=True)
        unique_idx = unique_idx_t.cpu().numpy().astype(np.int64, copy=False)

        # Now use numpy indexing safely
        q_u = q[unique_idx]
        feats_u = feats[unique_idx]
        y_u = y[unique_idx]

        # Ensure contiguous before converting to torch
        q_u = np.ascontiguousarray(q_u, dtype=np.int32)
        feats_u = np.ascontiguousarray(feats_u, dtype=np.float32)
        y_u = np.ascontiguousarray(y_u, dtype=np.int64)

        coords_t = torch.from_numpy(q_u).int()
        feats_t = torch.from_numpy(feats_u).float()
        labels_t = torch.from_numpy(y_u).long()

        return {
            "coords": coords_t,
            "feats": feats_t,
            "labels": labels_t,
            "fname": fname,
        }


def minkowski_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    coords_list = [b["coords"] for b in batch]
    feats_list = [b["feats"] for b in batch]
    labels_list = [b["labels"] for b in batch]

    coords, feats, labels = ME.utils.sparse_collate(coords_list, feats_list, labels_list)
    fnames = [b["fname"] for b in batch]
    return {"coords": coords, "feats": feats, "labels": labels, "fnames": fnames}
