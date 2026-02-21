# scripts/precompute_eclair_cache.py
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import laspy
import numpy as np
from src.config_loader import load_yaml
from src.utils import atomic_save_torch


def _sha1_hex(x: bytes) -> str:
    return hashlib.sha1(x).hexdigest()


def _cache_meta_for_file(pc_path: Path) -> Dict[str, object]:
    st = pc_path.stat()
    return {
        "cache_version": "eclair_raw_v1",
        "source_path": str(pc_path.resolve()),
        "source_size": int(st.st_size),
        "source_mtime_ns": int(st.st_mtime_ns),
    }


def _load_eclair_split_list(
    eclair_root: Union[str, Path],
    split: str,
    *,
    meta_filename: str = "labels.json",
    allowed_review_categories: Sequence[str] = ("approved",),
) -> List[str]:
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
        allowed_set = None
        if allowed_review_categories is not None:
            allowed_set = {c.lower() for c in allowed_review_categories}

        for rec in meta:
            if not isinstance(rec, dict):
                continue
            rec_split = str(rec.get("split", "")).lower().strip()
            if rec_split != split:
                continue

            if allowed_set is not None:
                cat = str(rec.get("review_category", "")).lower().strip()
                if cat not in allowed_set:
                    continue

            name = rec.get("tile_name", None)
            if name is None:
                continue
            out.append(str(name))

        if not out:
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
    alt = p.with_suffix(".las")
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not find pointcloud file for '{fname}' under {pc_dir}")


def read_las_arrays_robust(path: Path) -> Dict[str, np.ndarray]:
    try:
        las = laspy.read(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to read LAS file {path}: {e}")

    xyz = np.array(las.xyz, dtype=np.float64)

    def _get_dim(attr_name: str, fallback_names: List[str] = None) -> Optional[np.ndarray]:
        if hasattr(las, attr_name):
            val = getattr(las, attr_name)
            return np.array(val)

        dims_lower = set(d.lower() for d in las.point_format.dimension_names)

        if attr_name.lower() in dims_lower:
            return np.array(las[attr_name])

        if fallback_names:
            for name in fallback_names:
                if name.lower() in dims_lower:
                    return np.array(las[name])
        return None

    intensity = _get_dim("intensity")
    if intensity is None:
        intensity = np.zeros((xyz.shape[0],), dtype=np.float32)
    else:
        intensity = intensity.astype(np.float32)

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

    labels = _get_dim("classification", fallback_names=["raw_classification"])
    if labels is None:
        labels = np.zeros((xyz.shape[0],), dtype=np.int64)
    else:
        labels = labels.astype(np.int64)

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


def main():
    ap = argparse.ArgumentParser()

    # NEW: config-driven mode (preferred)
    ap.add_argument("--config", default=None, help="Path to YAML config (e.g. configs/e0_eclair_dropI.yaml)")

    # Backward-compatible CLI mode
    ap.add_argument("--eclair_root", default=None, help="ECLAIR root containing labels.json and pointclouds/")
    ap.add_argument("--cache_root", default=None, help="Output cache directory")
    ap.add_argument("--meta_filename", default="labels.json")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])

    ap.add_argument("--train_review_categories", nargs="*", default=["approved", "rejected"])
    ap.add_argument("--eval_review_categories", nargs="*", default=["approved"])

    args = ap.parse_args()

    if args.config is not None:
        cfg = load_yaml(args.config)
        data = cfg["data"]
        eclair_cfg = cfg.get("eclair", {})

        eclair_root = Path(data["eclair_root"])
        cache_root = Path(data["cache_root"])
        meta_filename = str(eclair_cfg.get("meta_filename", "labels.json"))

        train_cats = eclair_cfg.get("train_review_categories", ["approved", "rejected"])
        val_cats = eclair_cfg.get("val_review_categories", ["approved"])
        test_cats = eclair_cfg.get("test_review_categories", ["approved"])
    else:
        if args.eclair_root is None or args.cache_root is None:
            raise ValueError("Either pass --config OR both --eclair_root and --cache_root.")
        eclair_root = Path(args.eclair_root)
        cache_root = Path(args.cache_root)
        meta_filename = str(args.meta_filename)

        train_cats = args.train_review_categories
        val_cats = args.eval_review_categories
        test_cats = args.eval_review_categories

    cache_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        if split == "train":
            cats: Optional[Sequence[str]] = train_cats
        elif split == "val":
            cats = val_cats
        else:
            cats = test_cats

        names = _load_eclair_split_list(
            eclair_root,
            split,
            meta_filename=meta_filename,
            allowed_review_categories=cats,
        )

        out_dir = cache_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[precompute] split={split} tiles={len(names)} out_dir={out_dir}", flush=True)

        for i, fname in enumerate(names, start=1):
            pc_path = resolve_pc_path(eclair_root, fname)
            sample = read_las_arrays_robust(pc_path)
            sample["_cache_meta"] = _cache_meta_for_file(pc_path)
            sample["_tile_name"] = str(fname)

            out_path = out_dir / f"{Path(fname).stem}.pt"
            atomic_save_torch(sample, out_path)

            if i % 50 == 0 or i == len(names):
                print(f"[{split}] cached {i}/{len(names)} -> {out_path.name}", flush=True)

    print("[precompute] Done.", flush=True)


if __name__ == "__main__":
    main()
