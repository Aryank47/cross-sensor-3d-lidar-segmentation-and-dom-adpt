from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from .augment import AugmentConfig
from .config_loader import load_yaml
from .data_dales import DalesCropConfig, DalesPatchConfig, DalesPreprocConfig, DalesTiles, _find_dales_files
from .data_eclair import EclairTiles, PatchConfig, _resolve_pc_path
from .features import FeatureConfig
from .sampling_types import DatasetBundle, TileRecord
from .voxelization import VoxelizationConfig

ECLAIR_NATIVE_CLASS_NAMES = {
    0: "Undefined",
    1: "Unassigned",
    2: "Ground",
    3: "Vegetation",
    4: "Buildings",
    5: "Noise",
    6: "Transmission wires",
    7: "Distribution wires",
    8: "Poles",
    9: "Transmission towers",
    10: "Fence",
    11: "Vehicle",
}

DALES_NATIVE_CLASS_NAMES = {
    0: "Unknown",
    1: "Ground",
    2: "Vegetation",
    3: "Cars",
    4: "Trucks",
    5: "Power lines",
    6: "Fences",
    7: "Poles",
    8: "Buildings",
}


def _read_manifest(path: Path) -> list[Path]:
    paths: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(Path(line))
    return paths


def _mapping_to_lut(mapping: Mapping[int, int], *, minimum_size: int = 256) -> np.ndarray:
    if not mapping:
        raise ValueError("Mapping must not be empty")
    max_key = max(int(k) for k in mapping)
    lut = np.zeros(max(minimum_size, max_key + 1), dtype=np.int64)
    for key, value in mapping.items():
        key_i = int(key)
        value_i = int(value)
        if key_i < 0:
            raise ValueError(f"Negative mapping key is invalid: {key_i}")
        lut[key_i] = value_i
    return lut


def _load_int_mapping(path: Path) -> dict[int, int]:
    raw = load_yaml(path)
    return {int(k): int(v) for k, v in raw.items()}


def _resolve_dales_files(cfg: Dict[str, Any], split: str) -> tuple[Path, list[Path]]:
    data = cfg["data"]
    split = split.lower().strip()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported DALES split: {split}")

    train_root = Path(data["dales_train_root"])
    test_root = Path(data["dales_test_root"])
    val_root = Path(data["dales_val_root"]) if data.get("dales_val_root") else None
    manifest_dir = Path(data["split_manifest_dir"]) if data.get("split_manifest_dir") else None

    if manifest_dir is not None:
        manifest = manifest_dir / f"{split}.txt"
        if not manifest.exists():
            raise FileNotFoundError(f"DALES training config defines split_manifest_dir, but manifest is missing: {manifest}")
        root = test_root if split == "test" else train_root
        files = _read_manifest(manifest)
        return root, files

    if split == "test":
        return test_root, _find_dales_files(test_root)

    if val_root is not None and val_root.exists():
        if split == "train":
            return train_root, _find_dales_files(train_root)
        return val_root, _find_dales_files(val_root)

    all_train = sorted(_find_dales_files(train_root))
    val_fraction = float(data.get("val_fraction_from_train", 0.125))
    rng = random.Random(int(cfg["run"]["seed"]) + 777)
    rng.shuffle(all_train)
    n_val = max(1, int(round(len(all_train) * val_fraction)))
    val_files = all_train[:n_val]
    train_files = all_train[n_val:]
    return train_root, train_files if split == "train" else val_files


def _eclair_bundle(
    *,
    training_cfg_path: Path,
    native_mapping_path: Path,
    train_mapping_path: Path,
    split: str,
    use_project_cache: bool,
    seed: int,
) -> DatasetBundle:
    cfg = load_yaml(training_cfg_path)
    data = cfg["data"]
    eclair_cfg = cfg.get("eclair", {}) or {}
    split = split.lower().strip()

    review_key = {
        "train": "train_review_categories",
        "val": "val_review_categories",
        "test": "test_review_categories",
    }[split]
    default_reviews = None if split == "train" else ["approved"]
    review_categories = eclair_cfg.get(review_key, default_reviews)

    patch_cfg = PatchConfig(**data["patch"])
    feat_cfg = FeatureConfig(**data["features"])
    voxel_cfg = VoxelizationConfig.from_cfg(data)
    label_space = data["label_space"]

    ds = EclairTiles(
        eclair_root=data["eclair_root"],
        split=split,
        is_train=False,
        patch_cfg=patch_cfg,
        aug_cfg=AugmentConfig(enabled=False),
        feat_cfg=feat_cfg,
        num_classes=int(label_space["num_classes"]),
        ignore_index=int(label_space["ignore_index"]),
        undefined_id=int(label_space.get("eclair_undefined_id", 0)),
        use_cache=bool(use_project_cache and data.get("use_cache", True)),
        cache_root=data.get("cache_root"),
        seed=int(seed),
        meta_filename=eclair_cfg.get("meta_filename", "labels.json"),
        allowed_review_categories=review_categories,
        voxel_cfg=voxel_cfg,
        bev_cfg=None,
        run_cache_root=None,
        run_cache_precompute_returns_onehot=False,
    )

    root = Path(data["eclair_root"])
    records = [
        TileRecord(
            dataset="ECLAIR",
            tile_index=index,
            tile_id=Path(name).stem,
            path=_resolve_pc_path(root, name),
        )
        for index, name in enumerate(ds.names)
    ]

    native_map = _load_int_mapping(native_mapping_path)
    train_map = _load_int_mapping(train_mapping_path)
    return DatasetBundle(
        name="ECLAIR",
        training_config_path=training_cfg_path,
        training_config=cfg,
        dataset=ds,
        tile_records=records,
        native_to_common_lut=_mapping_to_lut(native_map),
        train_to_common_lut=_mapping_to_lut(train_map, minimum_size=int(label_space["num_classes"])),
        expected_native_ids=set(ECLAIR_NATIVE_CLASS_NAMES),
        native_class_names=ECLAIR_NATIVE_CLASS_NAMES,
        patch_cfg=patch_cfg,
        feat_cfg=feat_cfg,
        voxel_cfg=voxel_cfg,
        ignore_index=int(label_space["ignore_index"]),
        num_train_classes=int(label_space["num_classes"]),
        dataset_kind="eclair",
        source_split=split,
    )


def _dales_bundle(
    *,
    training_cfg_path: Path,
    native_mapping_path: Path,
    train_mapping_path: Path,
    split: str,
    use_project_cache: bool,
    require_project_cache: Optional[bool],
    seed: int,
) -> DatasetBundle:
    cfg = load_yaml(training_cfg_path)
    data = cfg["data"]
    root, files = _resolve_dales_files(cfg, split)

    patch_cfg = DalesPatchConfig(**data["patch"])
    feat_cfg = FeatureConfig(**data["features"])
    voxel_cfg = VoxelizationConfig.from_cfg(data)
    preproc_cfg = DalesPreprocConfig(**(data.get("preproc", {}) or {}))
    crop_cfg = DalesCropConfig(**{k: v for k, v in (data.get("sampling", {}) or {}).items() if k != "mode"})
    label_space = data["label_space"]
    label_map = {int(k): int(v) for k, v in data["dales_label_map_native_to_train"].items()}

    use_cache = bool(use_project_cache and data.get("use_cache", True))
    require_cache = bool(data.get("require_cache", False)) if require_project_cache is None else bool(require_project_cache)

    ds = DalesTiles(
        dales_root=root,
        files=files,
        patch_cfg=patch_cfg,
        feat_cfg=feat_cfg,
        is_train=False,
        aug_cfg=AugmentConfig(enabled=False),
        ignore_index=int(label_space["ignore_index"]),
        preproc=preproc_cfg,
        seed=int(seed),
        use_cache=use_cache,
        cache_root=data.get("cache_root"),
        cache_subdir=str(data.get("cache_subdir", "dales_dropI")),
        cache_key_extra=data.get("cache_key_extra"),
        require_cache=require_cache,
        write_cache=False,
        split_name=split,
        label_map=label_map,
        cache_kind=str(data.get("cache_kind", "raw")),
        sampling_mode="tiles",
        crop_cfg=crop_cfg,
        voxel_cfg=voxel_cfg,
        bev_cfg=None,
    )

    records = [
        TileRecord(
            dataset="DALES",
            tile_index=index,
            tile_id=path.stem,
            path=Path(path),
        )
        for index, path in enumerate(ds.files)
    ]

    native_map = _load_int_mapping(native_mapping_path)
    train_map = _load_int_mapping(train_mapping_path)
    return DatasetBundle(
        name="DALES",
        training_config_path=training_cfg_path,
        training_config=cfg,
        dataset=ds,
        tile_records=records,
        native_to_common_lut=_mapping_to_lut(native_map),
        train_to_common_lut=_mapping_to_lut(train_map, minimum_size=int(label_space["num_classes"])),
        expected_native_ids=set(DALES_NATIVE_CLASS_NAMES),
        native_class_names=DALES_NATIVE_CLASS_NAMES,
        patch_cfg=patch_cfg,
        feat_cfg=feat_cfg,
        voxel_cfg=voxel_cfg,
        ignore_index=int(label_space["ignore_index"]),
        num_train_classes=int(label_space["num_classes"]),
        dataset_kind="dales",
        source_split=split,
    )


def build_dataset_bundle(
    *,
    name: str,
    training_cfg_path: Path,
    native_mapping_path: Path,
    train_mapping_path: Path,
    split: str,
    use_project_cache: bool,
    require_project_cache: Optional[bool],
    seed: int,
) -> DatasetBundle:
    kind = name.lower().strip()
    if kind == "eclair":
        return _eclair_bundle(
            training_cfg_path=training_cfg_path,
            native_mapping_path=native_mapping_path,
            train_mapping_path=train_mapping_path,
            split=split,
            use_project_cache=use_project_cache,
            seed=seed,
        )
    if kind == "dales":
        return _dales_bundle(
            training_cfg_path=training_cfg_path,
            native_mapping_path=native_mapping_path,
            train_mapping_path=train_mapping_path,
            split=split,
            use_project_cache=use_project_cache,
            require_project_cache=require_project_cache,
            seed=seed,
        )
    raise ValueError(f"Unsupported dataset name: {name}")


def select_tile_records(
    records: Iterable[TileRecord],
    *,
    max_tiles: Optional[int],
    seed: int,
    dataset_name: str,
) -> list[TileRecord]:
    records = list(records)
    if max_tiles is None or max_tiles <= 0 or max_tiles >= len(records):
        return records
    payload = f"{seed}|{dataset_name}|tiles".encode("utf-8")
    derived = int.from_bytes(__import__("hashlib").sha1(payload).digest()[:8], "little")
    rng = np.random.default_rng(derived)
    indices = sorted(rng.choice(len(records), size=max_tiles, replace=False).tolist())
    return [records[index] for index in indices]
