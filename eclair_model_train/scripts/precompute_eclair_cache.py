# #!/usr/bin/env python3
# from __future__ import annotations

# import argparse
# import json
# from pathlib import Path
# from typing import Dict, List, Optional

# import numpy as np
# import torch


# def _load_eclair_split_list(eclair_root: Path, split: str, only_approved: bool = True):
#     meta_path = eclair_root / "labels.json"
#     meta = json.loads(meta_path.read_text())

#     # Dict format: {"train":[...], "val":[...], "test":[...]}
#     if isinstance(meta, dict):
#         if split not in meta:
#             raise ValueError(
#                 f"{meta_path} missing split='{split}'. keys={list(meta.keys())}"
#             )
#         return sorted(meta[split])

#     # List format: [{"tile_name":..., "split":..., "review_category":...}, ...]
#     if isinstance(meta, list):
#         rows = [r for r in meta if r.get("split") == split]
#         if only_approved:
#             rows = [
#                 r for r in rows if r.get("review_category", "approved") == "approved"
#             ]
#         names = [r["tile_name"] for r in rows if "tile_name" in r]
#         if not names:
#             raise ValueError(
#                 f"No tiles for split='{split}' in {meta_path} (only_approved={only_approved})"
#             )
#         return sorted(names)

#     raise TypeError(f"Unsupported labels.json type: {type(meta)} at {meta_path}")


# def _resolve_pc_path(eclair_root: Path, fname: str) -> Path:
#     pc_dir = eclair_root / "pointclouds"
#     p = pc_dir / fname
#     if p.exists():
#         return p
#     alt = p.with_suffix(".las")
#     if alt.exists():
#         return alt
#     raise FileNotFoundError(
#         f"Could not find pointcloud file for '{fname}' under {pc_dir}"
#     )


# def _read_las_arrays(path: Path) -> Dict[str, torch.Tensor]:
#     import laspy

#     las = laspy.read(str(path))

#     def _dim(name: str) -> Optional[np.ndarray]:
#         if name in set(las.point_format.dimension_names):
#             arr = las[name]
#             return getattr(arr, "array", arr)
#         return None

#     xyz = torch.from_numpy(las.xyz.astype(np.float32, copy=True))

#     intensity = _dim("intensity")
#     return_number = _dim("return_number")
#     number_of_returns = _dim("number_of_returns")

#     gt_key = (
#         "classification"
#         if "classification" in set(las.point_format.dimension_names)
#         else "raw_classification"
#     )
#     native_labels = _dim(gt_key)
#     if native_labels is None:
#         raise RuntimeError(f"Missing classification labels in {path}")

#     rgb = None
#     if all(
#         k in set(las.point_format.dimension_names) for k in ("red", "green", "blue")
#     ):
#         r = _dim("red").astype(np.float32)
#         g = _dim("green").astype(np.float32)
#         b = _dim("blue").astype(np.float32)
#         rgb = np.stack([r, g, b], axis=1)

#     out: Dict[str, torch.Tensor] = {
#         "xyz": xyz,
#         "native_labels": torch.from_numpy(native_labels.astype(np.int64, copy=True)),
#     }
#     out["intensity"] = (
#         torch.from_numpy(intensity.astype(np.float32, copy=True))
#         if intensity is not None
#         else None
#     )
#     out["return_number"] = (
#         torch.from_numpy(return_number.astype(np.int64, copy=True))
#         if return_number is not None
#         else None
#     )
#     out["number_of_returns"] = (
#         torch.from_numpy(number_of_returns.astype(np.int64, copy=True))
#         if number_of_returns is not None
#         else None
#     )
#     out["rgb"] = (
#         torch.from_numpy(rgb.astype(np.float32, copy=True)) if rgb is not None else None
#     )
#     return out


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument(
#         "--eclair_root",
#         required=True,
#         help="ECLAIR dataset root containing labels.json and pointclouds/",
#     )
#     ap.add_argument("--cache_root", required=True, help="Output cache directory")
#     ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
#     args = ap.parse_args()

#     eclair_root = Path(args.eclair_root)
#     cache_root = Path(args.cache_root)

#     for split in args.splits:
#         names = _load_eclair_split_list(eclair_root, split)
#         out_dir = cache_root / split
#         out_dir.mkdir(parents=True, exist_ok=True)

#         for i, fname in enumerate(names, start=1):
#             pc_path = _resolve_pc_path(eclair_root, fname)
#             sample = _read_las_arrays(pc_path)

#             out_path = out_dir / f"{Path(fname).stem}.pt"
#             torch.save(sample, out_path)

#             if i % 50 == 0 or i == len(names):
#                 print(f"[{split}] cached {i}/{len(names)} -> {out_path}")

#     print("Done.")


# if __name__ == "__main__":
#     main()

# scripts/precompute_eclair_cache.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import laspy
import numpy as np
import torch


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


def resolve_pc_path(eclair_root: Path, fname: str) -> Path:
    pc_dir = eclair_root / "pointclouds"
    p = pc_dir / fname
    if p.exists():
        return p
    # fallback: maybe stored as .las
    alt = p.with_suffix(".las")
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not find pointcloud file for '{fname}' under {pc_dir}")


def read_las_arrays_robust(path: Path) -> Dict[str, np.ndarray]:
    """
    Reads LAS/LAZ files using robust property-access to avoid bit-packing bugs.
    Standardizes output keys to: xyz, intensity, return_number, number_of_returns, rgb, native_labels.
    """
    try:
        las = laspy.read(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to read LAS file {path}: {e}")

    # Standardize XYZ to float32
    xyz = np.array(las.xyz, dtype=np.float64)

    def _get_dim(attr_name: str, fallback_names: List[str] = None) -> Optional[np.ndarray]:
        # Priority 1: Direct property access (handles bit-unpacking/scaling)
        if hasattr(las, attr_name):
            val = getattr(las, attr_name)
            return np.array(val)

        # Priority 2: Dictionary access (fallback for non-standard names)
        # Check standard dimension names case-insensitively
        dims_lower = set(d.lower() for d in las.point_format.dimension_names)

        if attr_name.lower() in dims_lower:
            return np.array(las[attr_name])

        if fallback_names:
            for name in fallback_names:
                if name.lower() in dims_lower:
                    return np.array(las[name])
        return None

    # Intensity
    intensity = _get_dim("intensity")
    if intensity is None:
        intensity = np.zeros((xyz.shape[0],), dtype=np.float32)
    else:
        intensity = intensity.astype(np.float32)

    # Returns (CRITICAL FIX: Use property access)
    rn = _get_dim("return_number")
    nor = _get_dim("number_of_returns")

    if rn is None:
        rn = np.ones((xyz.shape[0],), dtype=np.int64)
    else:
        rn = rn.astype(np.int64)

    if nor is None:
        nor = np.ones((xyz.shape[0],), dtype=np.int64)
    else:
        nor = nor.astype(np.int64)

    # Labels
    labels = _get_dim("classification", fallback_names=["raw_classification"])
    if labels is None:
        # Fallback for datasets that might be unlabeled
        # print(f" [WARN] No classification found in {path.name}, using zeros.") # Optional logging
        labels = np.zeros((xyz.shape[0],), dtype=np.int64)
    else:
        labels = labels.astype(np.int64)

    # RGB
    rgb = None
    red = _get_dim("red")
    green = _get_dim("green")
    blue = _get_dim("blue")

    if red is not None and green is not None and blue is not None:
        max_val = max(red.max(), green.max(), blue.max())
        scale = 1.0
        if max_val > 255:
            scale = 1.0 / 65535.0
        r = red.astype(np.float32) * scale
        g = green.astype(np.float32) * scale
        b = blue.astype(np.float32) * scale
        rgb = np.stack([r, g, b], axis=1)

    return {
        "xyz": xyz,
        "intensity": intensity,
        "return_number": rn,
        "number_of_returns": nor,
        "rgb": rgb,
        "native_labels": labels,
    }


def read_las_arrays(path: Path) -> Dict[str, np.ndarray]:
    """
    Robust reader wrapper for ECLAIR training.
    """
    # Matches old output: xyz, intensity, return_number, number_of_returns, rgb, native_labels
    return read_las_arrays_robust(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eclair_root",
        required=True,
        help="ECLAIR root containing labels.json and pointclouds/",
    )
    ap.add_argument("--cache_root", required=True, help="Output cache directory")
    ap.add_argument("--meta_filename", default="labels.json")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])

    # If you pass nothing, we default to the "paper-like" behavior below
    ap.add_argument("--train_review_categories", nargs="*", default=["approved", "rejected"])
    ap.add_argument("--eval_review_categories", nargs="*", default=["approved"])

    args = ap.parse_args()

    eclair_root = Path(args.eclair_root)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        if split == "train":
            cats: Optional[Sequence[str]] = args.train_review_categories
        else:
            cats = args.eval_review_categories
        names = _load_eclair_split_list(
            eclair_root,
            split,
            meta_filename=args.meta_filename,
            allowed_review_categories=cats,
        )
        out_dir = cache_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[precompute] split={split} tiles={len(names)} out_dir={out_dir}",
            flush=True,
        )

        for i, fname in enumerate(names, start=1):
            pc_path = resolve_pc_path(eclair_root, fname)
            sample = read_las_arrays(pc_path)

            out_path = out_dir / f"{Path(fname).stem}.pt"
            # store numpy arrays directly (fast + matches your __getitem__ expectations)
            torch.save(sample, out_path)

            if i % 50 == 0 or i == len(names):
                print(f"[{split}] cached {i}/{len(names)} -> {out_path}", flush=True)

    print("[precompute] Done.", flush=True)


if __name__ == "__main__":
    main()
