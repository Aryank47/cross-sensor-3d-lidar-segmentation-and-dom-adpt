# scripts/precompute_dales_cache.py
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

from src.augment import AugmentConfig
from src.config_loader import load_yaml
from src.data_dales import (
    DalesPatchConfig,
    DalesPreprocConfig,
    DalesTiles,
    _cache_key_for_dales_raw,
    _ensure_raw_dtypes,
    _find_dales_files,
)
from src.features import FeatureConfig
from src.utils import atomic_save_torch, read_las_arrays_robust


def _read_manifest(p: Path) -> List[Path]:
    lines = p.read_text().splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        out.append(Path(ln))
    return out


def _split_train_val(
    *,
    train_root: Path,
    val_root: Optional[Path],
    seed: int,
    val_fraction_from_train: float,
    split_manifest_dir: Optional[Path] = None,
) -> Tuple[List[Path], List[Path]]:
    # 0) Preferred: use frozen manifests
    if split_manifest_dir is not None:
        tr = split_manifest_dir / "train.txt"
        va = split_manifest_dir / "val.txt"
        if tr.exists() and va.exists():
            train_files = _read_manifest(tr)
            val_files = _read_manifest(va)
            print(f"Using train/val split from manifests: {tr} ({len(train_files)} files), {va} ({len(val_files)} files)")
            return train_files, val_files

    # 1) If user provides explicit val folder
    if val_root is not None and val_root.exists():
        train_files = _find_dales_files(train_root)
        val_files = _find_dales_files(val_root)
        return train_files, val_files

    # 2) Fallback: deterministic random split
    all_train = sorted(_find_dales_files(train_root))
    rng = random.Random(int(seed) + 777)
    rng.shuffle(all_train)
    n_val = max(1, int(round(len(all_train) * float(val_fraction_from_train))))
    val_files = all_train[:n_val]
    train_files = all_train[n_val:]
    return train_files, val_files


def _precompute_raw(
    *,
    split_name: str,
    files: List[Path],
    cache_root: Path,
    cache_subdir: str,
    overwrite: bool,
) -> None:
    """
    Write raw (unaugmented) DALES caches:
      cache_root/cache_subdir/raw/<split>/<sha>.pt

    The filename key MUST match DalesTiles._cache_key_for_dales_raw, which is stat-based.
    """
    out_dir = cache_root / cache_subdir / "raw" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[precompute][raw] split={split_name} n_files={len(files)} out_dir={out_dir}", flush=True)

    written = 0
    skipped = 0
    for i, path in enumerate(files, start=1):
        key = _cache_key_for_dales_raw(path=path)
        out_path = out_dir / f"{key}.pt"

        if out_path.exists() and not overwrite:
            skipped += 1
        else:
            raw = read_las_arrays_robust(path)
            raw = _ensure_raw_dtypes(raw)
            atomic_save_torch(raw, out_path)
            written += 1

        if i % 50 == 0 or i == len(files):
            print(
                f"  [{split_name}] {i}/{len(files)} | written={written} skipped={skipped} last={out_path.name}",
                flush=True,
            )

    print(f"[precompute][raw] done split={split_name} written={written} skipped={skipped}", flush=True)


def _precompute_voxel(
    *,
    split_name: str,
    files: List[Path],
    dales_root: Path,
    patch_cfg: DalesPatchConfig,
    feat_cfg: FeatureConfig,
    preproc_cfg: DalesPreprocConfig,
    ignore_index: int,
    seed: int,
    cache_root: Path,
    cache_subdir: str,
    cache_key_extra: Optional[str],
    label_map: dict[int, int],
) -> None:
    """
    Legacy voxel-cache mode:
      cache_root/cache_subdir/voxel/<split>/<sha>.pt
    """
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
        aug_cfg=AugmentConfig(enabled=False),  # no augmentation for precompute
        label_map=label_map,
        cache_kind="voxel",
        sampling_mode="tiles",
    )

    print(f"[precompute][voxel] split={split_name} n_files={len(ds)} cache_dir={ds._cache_dir}", flush=True)
    for i in range(len(ds)):
        _ = ds[i]
        if (i + 1) % 50 == 0:
            print(f"  {split_name}: {i+1}/{len(ds)}", flush=True)
    print(f"[precompute][voxel] done split={split_name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/e0_dales_dropI.yaml)")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache files (default: skip if present).",
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
    cache_subdir = str(data.get("cache_subdir", "dales"))
    cache_key_extra = data.get("cache_key_extra", None)

    seed = int(run.get("seed", 1337))

    # Label map is required for voxel-cache path; parse it here once.
    label_map_cfg = data.get("dales_label_map_native_to_train", None)
    if label_map_cfg is None:
        raise ValueError("data.dales_label_map_native_to_train is required for DALES.")
    label_map = {int(k): int(v) for k, v in label_map_cfg.items()}

    split_manifest_dir = None
    if "split_manifest_dir" in data:
        print(f"Using split manifests from {data['split_manifest_dir']}", flush=True)
        split_manifest_dir = Path(str(data["split_manifest_dir"]))

    train_files, val_files = _split_train_val(
        train_root=dales_train_root,
        val_root=dales_val_root,
        seed=seed,
        val_fraction_from_train=val_frac,
        split_manifest_dir=split_manifest_dir,
    )

    test_manifest = (split_manifest_dir / "test.txt") if split_manifest_dir else None
    if test_manifest is not None and test_manifest.exists():
        test_files = _read_manifest(test_manifest)
    else:
        test_files = _find_dales_files(dales_test_root)

    cache_kind = str(data.get("cache_kind", "voxel")).lower().strip()
    if cache_kind not in ("raw", "voxel"):
        raise ValueError(f"Unknown data.cache_kind='{cache_kind}' (expected 'raw' or 'voxel').")

    print(f"[precompute] cache_kind={cache_kind} cache_root={cache_root} cache_subdir={cache_subdir}", flush=True)

    if cache_kind == "raw":
        print(f"computing raw caches with ignore_index={ignore_index}", flush=True)
        _precompute_raw(
            split_name="train",
            files=train_files,
            cache_root=cache_root,
            cache_subdir=cache_subdir,
            overwrite=bool(args.overwrite),
        )
        _precompute_raw(
            split_name="val",
            files=val_files,
            cache_root=cache_root,
            cache_subdir=cache_subdir,
            overwrite=bool(args.overwrite),
        )
        _precompute_raw(
            split_name="test",
            files=test_files,
            cache_root=cache_root,
            cache_subdir=cache_subdir,
            overwrite=bool(args.overwrite),
        )
    else:
        print(
            f"computing voxel caches with patch_cfg={patch_cfg} feat_cfg={feat_cfg} preproc_cfg={preproc_cfg} ignore_index={ignore_index} seed={seed}",
            flush=True,
        )
        _precompute_voxel(
            split_name="train",
            files=train_files,
            dales_root=dales_train_root,
            patch_cfg=patch_cfg,
            feat_cfg=feat_cfg,
            preproc_cfg=preproc_cfg,
            ignore_index=ignore_index,
            seed=seed,
            cache_root=cache_root,
            cache_subdir=cache_subdir,
            cache_key_extra=cache_key_extra,
            label_map=label_map,
        )
        _precompute_voxel(
            split_name="val",
            files=val_files,
            dales_root=dales_train_root,
            patch_cfg=patch_cfg,
            feat_cfg=feat_cfg,
            preproc_cfg=preproc_cfg,
            ignore_index=ignore_index,
            seed=seed + 1,
            cache_root=cache_root,
            cache_subdir=cache_subdir,
            cache_key_extra=cache_key_extra,
            label_map=label_map,
        )
        _precompute_voxel(
            split_name="test",
            files=test_files,
            dales_root=dales_test_root,
            patch_cfg=patch_cfg,
            feat_cfg=feat_cfg,
            preproc_cfg=preproc_cfg,
            ignore_index=ignore_index,
            seed=seed + 2,
            cache_root=cache_root,
            cache_subdir=cache_subdir,
            cache_key_extra=cache_key_extra,
            label_map=label_map,
        )


if __name__ == "__main__":
    main()
