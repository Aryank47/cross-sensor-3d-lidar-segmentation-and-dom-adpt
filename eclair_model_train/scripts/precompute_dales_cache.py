# scripts/precompute_dales_cache.py
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

from src.augment import AugmentConfig
from src.config_loader import load_yaml
from src.data_dales import DalesPatchConfig, DalesPreprocConfig, DalesTiles, _find_dales_files
from src.features import FeatureConfig


def _split_train_val(
    *,
    train_root: Path,
    val_root: Optional[Path],
    seed: int,
    val_fraction_from_train: float,
) -> Tuple[List[Path], List[Path]]:
    if val_root is not None and val_root.exists():
        train_files = _find_dales_files(train_root)
        val_files = _find_dales_files(val_root)
        return train_files, val_files

    all_train = sorted(_find_dales_files(train_root))
    rng = random.Random(int(seed) + 777)
    rng.shuffle(all_train)
    n_val = max(1, int(round(len(all_train) * float(val_fraction_from_train))))
    val_files = all_train[:n_val]
    train_files = all_train[n_val:]
    return train_files, val_files


def _precompute_one(
    *,
    split_name: str,
    files: List[Path],
    dales_root: Path,
    patch_cfg: DalesPatchConfig,
    feat_cfg: FeatureConfig,
    aug_cfg: AugmentConfig,
    preproc_cfg: DalesPreprocConfig,
    ignore_index: int,
    seed: int,
    cache_root: Path,
    cache_subdir: str,
    cache_key_extra: Optional[str],
    num_workers: int,
) -> None:
    ds = DalesTiles(
        dales_root=dales_root,
        files=files,
        patch_cfg=patch_cfg,
        feat_cfg=feat_cfg,
        is_train=False,
        ignore_index=ignore_index,
        preproc=preproc_cfg,
        seed=seed,
        use_cache=True,
        cache_root=cache_root,
        cache_subdir=cache_subdir,
        cache_key_extra=cache_key_extra,
        require_cache=False,
        write_cache=True,
        split_name=split_name,
        aug_cfg=aug_cfg,
    )

    # Simple single-process loop is the most robust.
    # If you want parallelism, run multiple jobs or increase num_workers
    # in a DataLoader with a no-op collate; but this is safe everywhere.
    print(f"[precompute] split={split_name} n_files={len(ds)} cache_dir={ds._cache_dir}")
    for i in range(len(ds)):
        _ = ds[i]
        if (i + 1) % 50 == 0:
            print(f"  {split_name}: {i+1}/{len(ds)}")
    print(f"[precompute] done split={split_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. configs/e0_dales_dropI.yaml)",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data = cfg["data"]
    run = cfg["run"]

    if str(data.get("dataset", "dales")).lower() != "dales":
        raise ValueError("This precompute script expects data.dataset: dales")

    dales_train_root = Path(data["dales_train_root"])
    dales_test_root = Path(data["dales_test_root"])
    dales_val_root = Path(data["dales_val_root"]) if "dales_val_root" in data else None

    val_frac = float(data.get("val_fraction_from_train", 0.125))

    patch_cfg = DalesPatchConfig(**data["patch"])
    feat_cfg = FeatureConfig(**data["features"])
    preproc_cfg = DalesPreprocConfig(**data.get("preproc", {}))

    ls = data["label_space"]
    ignore_index = int(ls["ignore_index"])

    cache_root = Path(data["cache_root"])
    cache_subdir = str(data.get("cache_subdir", "dales_dropI"))
    cache_key_extra = data.get("cache_key_extra", None)

    seed = int(run.get("seed", 1337))
    num_workers = int(data.get("num_workers", 0))

    train_files, val_files = _split_train_val(
        train_root=dales_train_root,
        val_root=dales_val_root,
        seed=seed,
        val_fraction_from_train=val_frac,
    )
    test_files = _find_dales_files(dales_test_root)

    _precompute_one(
        split_name="train",
        files=train_files,
        dales_root=dales_train_root,
        patch_cfg=patch_cfg,
        feat_cfg=feat_cfg,
        aug_cfg=AugmentConfig(enabled=False),  # no augmentation for precompute
        preproc_cfg=preproc_cfg,
        ignore_index=ignore_index,
        seed=seed,
        cache_root=cache_root,
        cache_subdir=cache_subdir,
        cache_key_extra=cache_key_extra,
        num_workers=num_workers,
    )
    _precompute_one(
        split_name="val",
        files=val_files,
        dales_root=dales_train_root,
        patch_cfg=patch_cfg,
        feat_cfg=feat_cfg,
        aug_cfg=AugmentConfig(enabled=False),  # no augmentation for precompute
        preproc_cfg=preproc_cfg,
        ignore_index=ignore_index,
        seed=seed + 1,
        cache_root=cache_root,
        cache_subdir=cache_subdir,
        cache_key_extra=cache_key_extra,
        num_workers=num_workers,
    )
    _precompute_one(
        split_name="test",
        files=test_files,
        dales_root=dales_test_root,
        patch_cfg=patch_cfg,
        feat_cfg=feat_cfg,
        aug_cfg=AugmentConfig(enabled=False),  # no augmentation for precompute
        preproc_cfg=preproc_cfg,
        ignore_index=ignore_index,
        seed=seed + 2,
        cache_root=cache_root,
        cache_subdir=cache_subdir,
        cache_key_extra=cache_key_extra,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()
