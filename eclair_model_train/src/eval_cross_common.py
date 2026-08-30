from __future__ import annotations

# src/eval_cross_common.py
"""
Fresh cross-domain/common-space evaluator for ECLAIR/DALES checkpoints.

This script intentionally DOES NOT import or call the old eval script.

Scientific contract:
  - Source checkpoint config owns feature + voxelization semantics.
  - Target dataset owns raw points + native labels.
  - Predictions are source train IDs, mapped explicitly to common IDs.
  - Target labels are target native IDs, mapped explicitly to common IDs.
  - Metrics are computed pointwise in common label space.
  - Uses raw LAS/LAZ or raw cache, not pre-voxelized cache, for final eval.

Expected usage:
  python -m src.eval_cross_common ...
"""

import argparse
import csv
import gc
import json
import logging
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

try:
    import MinkowskiEngine as ME
except Exception as e:
    raise RuntimeError("MinkowskiEngine is required for this evaluator.") from e

from .augment import AugmentConfig
from .bev_head import BEVHeadConfig
from .config_loader import load_yaml
from .data_dales import DalesCropConfig, DalesPatchConfig, DalesPreprocConfig, DalesTiles, _find_dales_files
from .data_eclair import EclairTiles, PatchConfig
from .features import FeatureConfig, build_features, infer_in_channels
from .model import build_model
from .voxelization import VoxelizationConfig, voxelize_from_q

COMMON_CLASS_NAMES = [
    "ignore",
    "ground",
    "vegetation",
    "buildings",
    "wires",
    "poles",
    "fence",
    "vehicle",
]
COMMON_IGNORE_ID = 0


# -----------------------------------------------------------------------------
# Logging / config / mapping
# -----------------------------------------------------------------------------


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_cross_common")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(out_dir / "eval_cross_common.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config_any(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    return load_yaml(str(p))


def load_yaml_int_map(path: str | Path) -> Dict[int, int]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mapping file not found: {p}")
    obj = yaml.safe_load(p.read_text())
    if obj is None:
        raise RuntimeError(f"Empty YAML mapping file: {p}")
    return {int(k): int(v) for k, v in obj.items()}


def mapping_to_lut(
    mp: Mapping[int, int],
    *,
    size: int,
    default: int,
    name: str,
) -> np.ndarray:
    lut = np.full((int(size),), int(default), dtype=np.int64)
    for k, v in mp.items():
        kk = int(k)
        if 0 <= kk < int(size):
            lut[kk] = int(v)
    if int(lut.max()) >= len(COMMON_CLASS_NAMES):
        raise RuntimeError(
            f"{name} emits common id {int(lut.max())}, " f"but COMMON_CLASS_NAMES has only {len(COMMON_CLASS_NAMES)} classes."
        )
    return lut


def source_native_to_train_lut_from_config(
    *,
    source_cfg: Dict[str, Any],
    source_domain: str,
    logger: logging.Logger,
) -> Tuple[np.ndarray, int]:
    """
    Build source native-label -> source train-id LUT from the source training config.

    This is used only for validation of:
        source_train_to_common == compose(source_native_to_train, source_native_to_common)

    DALES:
      Uses data.dales_label_map_native_to_train from the training config.

    ECLAIR:
      Training convention is native labels 1..out_channels -> train IDs 0..out_channels-1,
      while native 0 is ignored.
    """
    source_domain = source_domain.lower().strip()
    data = source_cfg.get("data", {})
    label_space = data.get("label_space", {}) if isinstance(data.get("label_space", {}), dict) else {}
    ignore_index = int(label_space.get("ignore_index", -100))
    out_channels = int(source_cfg["model"]["out_channels"])

    lut = np.full((256,), ignore_index, dtype=np.int64)

    if source_domain == "dales":
        mp = data.get("dales_label_map_native_to_train", None)
        if mp is None:
            # Fallback names, in case a resolved config uses a generic key.
            mp = data.get("label_map_native_to_train", None)
        if mp is None:
            raise RuntimeError(
                "Could not validate DALES source mapping: " "source_cfg.data.dales_label_map_native_to_train is missing."
            )

        for k, v in mp.items():
            kk = int(k)
            if 0 <= kk < 256:
                lut[kk] = int(v)

        logger.info("[mapcheck] built DALES native->train LUT from source config.")
        return lut, ignore_index

    if source_domain == "eclair":
        # ECLAIR training convention:
        # native 0 is undefined/ignored.
        # native 1..out_channels are mapped to train 0..out_channels-1.
        undefined_id = int(label_space.get("eclair_undefined_id", 0))
        if 0 <= undefined_id < 256:
            lut[undefined_id] = ignore_index

        for native_id in range(1, out_channels + 1):
            if native_id < 256:
                lut[native_id] = native_id - 1

        logger.info(
            "[mapcheck] built ECLAIR native->train LUT by convention: " "native 1..out_channels -> train 0..out_channels-1."
        )
        return lut, ignore_index

    raise RuntimeError(f"Unknown source_domain={source_domain}")


def validate_source_train_to_common_mapping(
    *,
    source_cfg: Dict[str, Any],
    source_domain: str,
    source_train_to_common: np.ndarray,
    source_native_to_common_lut: np.ndarray,
    logger: logging.Logger,
) -> None:
    """
    Validate that the explicit source train-id -> common-id mapping agrees with:

        source native-id -> source train-id        from source training config
        source native-id -> common-id             from source native-to-common YAML

    This prevents silent mistakes such as:
      - using ECLAIR train_id_to_common for a DALES checkpoint
      - using DALES train_id_to_common for an ECLAIR checkpoint
      - mismatched class ordering after native->train remapping
    """
    native_to_train, ignore_index = source_native_to_train_lut_from_config(
        source_cfg=source_cfg,
        source_domain=source_domain,
        logger=logger,
    )

    out_channels = int(source_cfg["model"]["out_channels"])
    if source_train_to_common.shape[0] != out_channels:
        raise RuntimeError(
            "source_train_to_common size mismatch:\n"
            f"  LUT size      = {source_train_to_common.shape[0]}\n"
            f"  out_channels  = {out_channels}\n"
            "This usually means the wrong train_id_to_common YAML was passed."
        )

    seen_train_ids = set()
    mismatches = []

    for native_id in range(256):
        train_id = int(native_to_train[native_id])

        if train_id == ignore_index:
            continue

        if train_id < 0 or train_id >= out_channels:
            mismatches.append(
                {
                    "native_id": native_id,
                    "train_id": train_id,
                    "error": f"train_id outside [0, {out_channels - 1}]",
                }
            )
            continue

        seen_train_ids.add(train_id)

        expected_common = int(source_native_to_common_lut[native_id])
        actual_common = int(source_train_to_common[train_id])

        if actual_common != expected_common:
            mismatches.append(
                {
                    "native_id": native_id,
                    "train_id": train_id,
                    "expected_common_from_native_map": expected_common,
                    "actual_common_from_train_map": actual_common,
                }
            )

    missing_train_ids = sorted(set(range(out_channels)) - seen_train_ids)

    if mismatches:
        logger.error("[mapcheck] source train->common mapping does NOT match source config + native->common map.")
        logger.error(json.dumps(mismatches, indent=2))
        raise RuntimeError(
            "Source mapping validation failed. " "Check --source_train_to_common, --source_native_to_common, and --source_domain."
        )

    if missing_train_ids:
        raise RuntimeError(
            "Source native->train config does not cover all model output train IDs:\n"
            f"  missing_train_ids={missing_train_ids}\n"
            "This means the checkpoint output space is not fully represented in the source mapping."
        )

    logger.info(
        "[mapcheck] OK: source_train_to_common matches " "source_native_to_train(source config) + source_native_to_common."
    )
    logger.info(f"[mapcheck] source_train_to_common={source_train_to_common.tolist()}")


def read_manifest(path: str | Path) -> List[Path]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    out: List[Path] = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(Path(s))
    if not out:
        raise RuntimeError(f"Manifest has no files: {p}")
    return out


# -----------------------------------------------------------------------------
# Checkpoint/model
# -----------------------------------------------------------------------------


def strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model_state", "model_state_dict", "state_dict", "model", "net"):
            v = ckpt.get(key, None)
            if isinstance(v, dict) and len(v) > 0:
                return strip_module_prefix(v)
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return strip_module_prefix(ckpt)
    raise RuntimeError(
        "Could not find model state dict. "
        f"Checkpoint type={type(ckpt)}, keys={list(ckpt.keys()) if isinstance(ckpt, dict) else 'NA'}"
    )


def assert_source_feature_contract(source_cfg: Dict[str, Any]) -> int:
    feat_cfg = FeatureConfig(**source_cfg["data"]["features"])
    inferred = int(infer_in_channels(feat_cfg))
    declared = int(source_cfg["model"]["in_channels"])
    if inferred != declared:
        raise RuntimeError(
            "Source feature contract mismatch:\n"
            f"  infer_in_channels(FeatureConfig) = {inferred}\n"
            f"  source_cfg.model.in_channels = {declared}\n"
            "Refusing eval because checkpoint feature semantics are unsafe."
        )
    return inferred


def load_model_for_eval(
    *,
    source_cfg: Dict[str, Any],
    ckpt_path: str | Path,
    device: torch.device,
    logger: logging.Logger,
) -> torch.nn.Module:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    in_channels = assert_source_feature_contract(source_cfg)
    out_channels = int(source_cfg["model"]["out_channels"])
    D = int(source_cfg["model"].get("D", 3))

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = extract_state_dict(ckpt)

    # Important: pass cfg so BEV-wrapped checkpoints load correctly if BEV was enabled.
    try:
        model = build_model(
            in_channels=in_channels,
            out_channels=out_channels,
            D=D,
            cfg=source_cfg,
        ).to(device)
    except TypeError:
        # Backward-compatible fallback if build_model does not accept cfg in older code.
        model = build_model(
            in_channels=in_channels,
            out_channels=out_channels,
            D=D,
        ).to(device)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model state_dict mismatch.\n"
            f"Missing ({len(missing)}): {missing[:30]}\n"
            f"Unexpected ({len(unexpected)}): {unexpected[:30]}"
        )

    model.eval()
    logger.info(
        f"[model] loaded OK ckpt={ckpt_path} " f"in_channels={in_channels} out_channels={out_channels} D={D} device={device.type}"
    )
    return model


def sparse_output_from_model_output(out: Any) -> ME.SparseTensor:
    """
    Handles:
      - plain ME.SparseTensor
      - BEV wrapper returning (seg_out, bev_pred)
      - dict-style outputs if future model variants return dicts
    """
    if isinstance(out, ME.SparseTensor):
        return out

    if isinstance(out, (tuple, list)):
        for item in out:
            if isinstance(item, ME.SparseTensor):
                return item

    if isinstance(out, dict):
        for key in ("seg_out", "sparse", "sparse_logits", "seg", "seg_logits", "out"):
            item = out.get(key, None)
            if isinstance(item, ME.SparseTensor):
                return item

    raise RuntimeError(f"Could not extract ME.SparseTensor from model output type={type(out)}")


# -----------------------------------------------------------------------------
# Target dataset builders
# -----------------------------------------------------------------------------


def make_source_contract(source_cfg: Dict[str, Any]) -> Tuple[FeatureConfig, Dict[str, Any], VoxelizationConfig, int, int]:
    """
    Source checkpoint config owns preprocessing/feature/voxelization semantics.
    """
    data = source_cfg["data"]
    feat_cfg = FeatureConfig(**data["features"])
    patch_cfg_dict = dict(data["patch"])
    voxel_cfg = VoxelizationConfig.from_cfg(data)
    ignore_index = int(data.get("label_space", {}).get("ignore_index", -100))
    num_classes = int(source_cfg["model"]["out_channels"])
    return feat_cfg, patch_cfg_dict, voxel_cfg, ignore_index, num_classes


class RawDatasetConcat:
    """
    Minimal dataset wrapper used by eval_cross_common.

    The evaluator only needs:
      - len(dataset)
      - dataset.get_raw(i)

    This allows split='all' without changing EclairTiles or DalesTiles.
    """

    def __init__(self, parts: List[Any], names: Optional[List[str]] = None):
        self.parts = list(parts)
        self.names = names or [f"part_{i}" for i in range(len(self.parts))]

        self.offsets = []
        total = 0
        for ds in self.parts:
            self.offsets.append(total)
            total += len(ds)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def get_raw(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)

        # Small number of parts, so linear scan is fine.
        for part_i, ds in enumerate(self.parts):
            start = self.offsets[part_i]
            end = start + len(ds)
            if start <= idx < end:
                raw = ds.get_raw(idx - start)
                if isinstance(raw, dict):
                    raw = dict(raw)
                    raw.setdefault("_concat_part", self.names[part_i])
                    raw.setdefault("_concat_local_idx", idx - start)
                return raw

        raise IndexError(idx)


def build_target_dataset_one_split(
    *,
    target_domain: str,
    target_cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
    split: str,
    target_manifest: Optional[str],
    force_raw_cache: bool,
    logger: logging.Logger,
):
    """
    Build target dataset using target roots/splits, but force the target loader to use
    source checkpoint feature/patch/voxelization semantics.

    The dataset is used only for get_raw(idx), not for __getitem__ training/eval samples.
    """
    target_domain = target_domain.lower().strip()
    tdata = target_cfg["data"]

    feat_cfg, patch_cfg_dict, voxel_cfg, ignore_index, source_num_classes = make_source_contract(source_cfg)
    seed = int(source_cfg.get("run", {}).get("seed", 1337)) + 2027

    try:
        bev_cfg = BEVHeadConfig.from_cfg(source_cfg)
    except Exception:
        bev_cfg = None

    if target_domain == "eclair":
        eclair_cfg = target_cfg.get("eclair", {}) or {}
        meta_filename = eclair_cfg.get("meta_filename", "labels.json")
        review_key = f"{split}_review_categories"
        allowed_review_categories = eclair_cfg.get(review_key, ["approved"])
        undefined_id = int(tdata.get("label_space", {}).get("eclair_undefined_id", 0))

        ds = EclairTiles(
            eclair_root=tdata["eclair_root"],
            split=split,
            is_train=False,
            patch_cfg=PatchConfig(**patch_cfg_dict),
            aug_cfg=AugmentConfig(enabled=False),
            feat_cfg=feat_cfg,
            num_classes=source_num_classes,
            ignore_index=ignore_index,
            undefined_id=undefined_id,
            use_cache=bool(tdata.get("use_cache", True)),
            cache_root=tdata.get("cache_root", None),
            seed=seed,
            meta_filename=meta_filename,
            allowed_review_categories=allowed_review_categories,
            voxel_cfg=voxel_cfg,
            bev_cfg=bev_cfg,
        )
        logger.info(f"[target] ECLAIR split={split} n_tiles={len(ds)} " f"allowed_review_categories={allowed_review_categories}")
        return ds

    if target_domain == "dales":
        root_key = f"dales_{split}_root"
        if root_key not in tdata:
            if split == "test" and "dales_test_root" in tdata:
                root_key = "dales_test_root"
            elif split == "val" and "dales_val_root" in tdata:
                root_key = "dales_val_root"
            elif split in ("train", "val") and "dales_train_root" in tdata:
                root_key = "dales_train_root"
            else:
                raise RuntimeError(f"Could not find DALES root for split={split} in target_config.data")

        dales_root = Path(tdata[root_key])
        if target_manifest:
            files = read_manifest(target_manifest)
        else:
            files = _find_dales_files(dales_root)

        if not files:
            raise RuntimeError(f"No DALES files found for split={split}, root={dales_root}")

        # Official cross-domain eval should use raw/raw-cache, never pre-voxelized cache.
        cfg_cache_kind = str(tdata.get("cache_kind", "raw")).lower().strip()
        cache_kind = "raw" if force_raw_cache else cfg_cache_kind
        if cache_kind != "raw":
            raise RuntimeError(
                "Official eval must use DALES raw/raw-cache, not voxel cache. "
                "Set --force_raw_cache or set target config data.cache_kind='raw'."
            )

        preproc_cfg = DalesPreprocConfig(**tdata.get("preproc", {}))

        # DalesTiles requires a native->train label_map to build label_lut internally,
        # even though this evaluator computes final common-space GT explicitly from
        # target native labels using --target_native_to_common.
        #
        # Use the TARGET DALES config's native->train map here because this dataset
        # object represents the DALES target data.
        target_dales_label_map = tdata.get("dales_label_map_native_to_train", None)
        if target_dales_label_map is None:
            raise RuntimeError(
                "Target DALES config is missing data.dales_label_map_native_to_train. "
                "DalesTiles requires this to construct label_lut."
            )
        target_dales_label_map = {int(k): int(v) for k, v in target_dales_label_map.items()}

        ds = DalesTiles(
            dales_root=dales_root,
            files=files,
            patch_cfg=DalesPatchConfig(**patch_cfg_dict),
            feat_cfg=feat_cfg,
            aug_cfg=AugmentConfig(enabled=False),
            is_train=False,
            ignore_index=ignore_index,
            preproc=preproc_cfg,
            seed=seed,
            use_cache=bool(tdata.get("use_cache", True)),
            cache_root=tdata.get("cache_root", None),
            cache_subdir=str(tdata.get("cache_subdir", "dales_dropI")),
            cache_key_extra=tdata.get("cache_key_extra", None),
            require_cache=bool(tdata.get("require_cache", False)),
            write_cache=False,
            split_name=split,
            label_map=target_dales_label_map,
            cache_kind="raw",
            sampling_mode="tiles",
            crop_cfg=DalesCropConfig(),
            voxel_cfg=voxel_cfg,
            bev_cfg=bev_cfg,
        )
        logger.info(
            f"[target] DALES split={split} n_tiles={len(ds)} root={dales_root} " f"manifest={target_manifest} cache_kind=raw"
        )
        return ds

    raise RuntimeError(f"Unsupported target_domain={target_domain}")


def get_dales_manifest_for_split(target_cfg: Dict[str, Any], split: str) -> Optional[str]:
    """
    Return the DALES manifest path for a split if data.split_manifest_dir exists.

    This is required for DALES because train/val are split from the same
    physical dales_train_root directory, but their raw caches are stored under
    different cache namespaces:

        raw/train/
        raw/val/
        raw/test/

    Without the manifest, split='train' may include validation files and then
    incorrectly search for their cache under raw/train instead of raw/val.
    """
    data = target_cfg.get("data", {})
    split_manifest_dir = data.get("split_manifest_dir", None)
    if split_manifest_dir is None:
        return None

    mp = Path(str(split_manifest_dir)) / f"{split}.txt"
    if not mp.exists():
        raise RuntimeError(
            f"DALES split_manifest_dir is set, but manifest is missing: {mp}. "
            "Expected train.txt, val.txt, and test.txt for split='all'."
        )

    return str(mp)


def build_target_dataset(
    *,
    target_domain: str,
    target_cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
    split: str,
    target_manifest: Optional[str],
    force_raw_cache: bool,
    logger: logging.Logger,
):
    """
    Build target dataset for eval.

    split='test' / 'val' / 'train':
      normal split-aware evaluation.

    split='all':
      diagnostic full-labelled-target evaluation.

      DALES:
        combines dales_train_root + dales_test_root unless target_manifest is given.

      ECLAIR:
        concatenates train + val + test EclairTiles.
    """
    split = str(split).lower().strip()
    target_domain = str(target_domain).lower().strip()

    if split != "all":
        return build_target_dataset_one_split(
            target_domain=target_domain,
            target_cfg=target_cfg,
            source_cfg=source_cfg,
            split=split,
            target_manifest=target_manifest,
            force_raw_cache=force_raw_cache,
            logger=logger,
        )

    logger.info("[target] building split='all' diagnostic dataset")

    if target_domain == "dales":
        # DALES train/val/test must be reconstructed using split manifests.
        #
        # Reason:
        #   DALES train and val files both physically live under dales_train_root,
        #   but their raw caches are split-scoped:
        #
        #       raw/train/
        #       raw/val/
        #       raw/test/
        #
        # If we scan dales_train_root directly, split='train' will accidentally
        # include validation files and then look for val cache entries under raw/train.
        # That caused:
        #
        #   [DALES raw cache missing] .../raw/train/71f87a43d0952021a339f6c6.pt
        #
        # even though that hash exists under raw/val.
        if target_manifest:
            raise RuntimeError(
                "DALES split='all' should not use --target_manifest. "
                "Use data.split_manifest_dir with train.txt, val.txt, and test.txt instead."
            )

        parts = []
        names = []

        for sp in ("train", "val", "test"):
            manifest = get_dales_manifest_for_split(target_cfg, sp)

            logger.info(f"[target] DALES-all building split={sp} manifest={manifest}")

            ds = build_target_dataset_one_split(
                target_domain=target_domain,
                target_cfg=target_cfg,
                source_cfg=source_cfg,
                split=sp,
                target_manifest=manifest,
                force_raw_cache=force_raw_cache,
                logger=logger,
            )

            parts.append(ds)
            names.append(sp)

        out = RawDatasetConcat(parts, names=names)

        logger.info(f"[target] DALES-all n_tiles={len(out)} " f"parts={[(n, len(p)) for n, p in zip(names, parts)]}")

        return out

    if target_domain == "eclair":
        # For ECLAIR-all, concatenate train + val + test. This keeps EclairTiles split logic intact.
        parts = []
        names = []

        for sp in ("train", "val", "test"):
            ds = build_target_dataset_one_split(
                target_domain=target_domain,
                target_cfg=target_cfg,
                source_cfg=source_cfg,
                split=sp,
                target_manifest=None,
                force_raw_cache=force_raw_cache,
                logger=logger,
            )
            parts.append(ds)
            names.append(sp)

        out = RawDatasetConcat(parts, names=names)
        logger.info(f"[target] ECLAIR-all n_tiles={len(out)} " f"parts={[(n, len(p)) for n, p in zip(names, parts)]}")
        return out

    raise RuntimeError(f"Unsupported target_domain={target_domain} for split='all'")


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def confusion_from_labels(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    if gt.shape != pred.shape:
        raise RuntimeError(f"gt/pred shape mismatch: {gt.shape} vs {pred.shape}")

    mask = gt != int(ignore_index)
    gt_m = gt[mask].astype(np.int64, copy=False)
    pred_m = pred[mask].astype(np.int64, copy=False)

    if gt_m.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    bad = (gt_m < 0) | (gt_m >= num_classes) | (pred_m < 0) | (pred_m >= num_classes)
    if np.any(bad):
        raise RuntimeError(
            f"Common labels out of range. "
            f"gt min/max=({gt_m.min()}, {gt_m.max()}), "
            f"pred min/max=({pred_m.min()}, {pred_m.max()}), C={num_classes}"
        )

    idx = gt_m * num_classes + pred_m
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_cm(cm: np.ndarray, class_names: Sequence[str], ignore_index: int) -> Dict[str, Any]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    iou = tp / np.maximum(tp + fp + fn, 1e-12)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(tp + fn, 1e-12)
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-12)

    valid = np.ones((cm.shape[0],), dtype=bool)

    # For common-space metrics, ignore_index is usually 0 and exists in the confusion matrix.
    # For train-space metrics, ignore_index is usually -100 and is already masked out before
    # confusion-matrix construction, so it is not an actual class index.
    if ignore_index is not None and 0 <= int(ignore_index) < cm.shape[0]:
        valid[int(ignore_index)] = False

    per_class = []
    for i, name in enumerate(class_names):
        per_class.append(
            {
                "class_id": int(i),
                "name": str(name),
                "iou": float(iou[i]),
                "f1": float(f1[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "support": int(cm[i, :].sum()),
                "pred_count": int(cm[:, i].sum()),
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "ignored_for_mean": bool(i == ignore_index),
            }
        )

    return {
        "miou": float(np.mean(iou[valid])),
        "macro_f1": float(np.mean(f1[valid])),
        "iou_per_class": iou.tolist(),
        "f1_per_class": f1.tolist(),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def get_train_class_names(source_cfg: Dict[str, Any], num_classes: int) -> List[str]:
    """
    Return source train-space class names.

    DALES usually has 8:
      ground, vegetation, cars, trucks, buildings, poles, power_lines, fences

    ECLAIR usually has 11:
      Unassigned, Ground, Vegetation, Buildings, Noise, Transmission wires,
      Distribution wires, Poles, Transmission towers, Fence, Vehicles

    If class_names is missing in the config, fall back to class_0..class_N.
    """
    label_space = source_cfg.get("data", {}).get("label_space", {})
    names = label_space.get("class_names", None)

    if isinstance(names, list) and len(names) == int(num_classes):
        return [str(x) for x in names]

    return [f"class_{i}" for i in range(int(num_classes))]


def write_metric_files(metrics: Dict[str, Any], out_dir: Path) -> None:
    (out_dir / "metrics_common.json").write_text(json.dumps(metrics, indent=2))

    with (out_dir / "summary_common.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["miou", "macro_f1", "n_tiles", "n_points_total", "n_points_eval", "runtime_s"])
        w.writerow(
            [
                f"{metrics['miou']:.8f}",
                f"{metrics['macro_f1']:.8f}",
                int(metrics["n_tiles"]),
                int(metrics["n_points_total"]),
                int(metrics["n_points_eval"]),
                f"{metrics['runtime_s']:.3f}",
            ]
        )

    with (out_dir / "per_class_common.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "class_id",
                "name",
                "iou",
                "f1",
                "precision",
                "recall",
                "support",
                "pred_count",
                "tp",
                "fp",
                "fn",
                "ignored_for_mean",
            ]
        )
        for r in metrics["per_class"]:
            w.writerow(
                [
                    r["class_id"],
                    r["name"],
                    f"{r['iou']:.8f}",
                    f"{r['f1']:.8f}",
                    f"{r['precision']:.8f}",
                    f"{r['recall']:.8f}",
                    r["support"],
                    r["pred_count"],
                    r["tp"],
                    r["fp"],
                    r["fn"],
                    r["ignored_for_mean"],
                ]
            )


def write_train_space_metric_files(metrics: Dict[str, Any], out_dir: Path) -> None:
    """
    Write same-domain train/native-space metrics.

    These metrics are comparable to the training pipeline's native/train-space
    validation/test numbers, not to common-space cross-domain metrics.
    """
    (out_dir / "metrics_train_space.json").write_text(json.dumps(metrics, indent=2))

    with (out_dir / "summary_train_space.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["miou", "macro_f1", "n_tiles", "n_points_total", "n_points_eval", "runtime_s"])
        w.writerow(
            [
                f"{metrics['miou']:.8f}",
                f"{metrics['macro_f1']:.8f}",
                int(metrics["n_tiles"]),
                int(metrics["n_points_total"]),
                int(metrics["n_points_eval"]),
                f"{metrics['runtime_s']:.3f}",
            ]
        )

    with (out_dir / "per_class_train_space.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "class_id",
                "name",
                "iou",
                "f1",
                "precision",
                "recall",
                "support",
                "pred_count",
                "tp",
                "fp",
                "fn",
                "ignored_for_mean",
            ]
        )
        for r in metrics["per_class"]:
            w.writerow(
                [
                    r["class_id"],
                    r["name"],
                    f"{r['iou']:.8f}",
                    f"{r['f1']:.8f}",
                    f"{r['precision']:.8f}",
                    f"{r['recall']:.8f}",
                    r["support"],
                    r["pred_count"],
                    r["tp"],
                    r["fp"],
                    r["fn"],
                    r["ignored_for_mean"],
                ]
            )

    np.save(out_dir / "confusion_train_space.npy", np.asarray(metrics["confusion_matrix"], dtype=np.int64))


# -----------------------------------------------------------------------------
# Native labels and target mapping
# -----------------------------------------------------------------------------


def get_native_labels(raw: Dict[str, Any], *, target_domain: str) -> np.ndarray:
    y = raw.get("native_labels", None)
    if y is None:
        y = raw.get("labels", None)
    if y is None:
        raise KeyError(f"{target_domain} raw tile missing native_labels/labels")
    return np.asarray(y, dtype=np.int64)


def map_target_native_to_common(
    raw: Dict[str, Any],
    *,
    lut: np.ndarray,
    target_domain: str,
) -> np.ndarray:
    y_native = get_native_labels(raw, target_domain=target_domain)
    y_safe = np.clip(y_native, 0, lut.shape[0] - 1)
    return lut[y_safe].astype(np.int64, copy=False)


# -----------------------------------------------------------------------------
# Sparse coordinate alignment
# -----------------------------------------------------------------------------


def _row_keys_int32(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a.astype(np.int32, copy=False))
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).reshape(-1)


def align_sparse_output_to_input_coords(
    *,
    out: ME.SparseTensor,
    coords_in_t: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Return out.F reordered so each row aligns with coords_in_t row order.

    This avoids assuming that MinkowskiEngine output feature rows always preserve
    input coordinate order.
    """
    if hasattr(out, "features_at_coordinates"):
        try:
            got = out.features_at_coordinates(coords_in_t)
            if got.shape[0] == coords_in_t.shape[0]:
                return got
        except Exception:
            pass

    cin = coords_in_t.detach().cpu().to(torch.int32).contiguous().numpy()
    cout = out.C.detach().cpu().to(torch.int32).contiguous().numpy()

    if cin.shape[0] != cout.shape[0]:
        raise RuntimeError(f"Sparse output coordinate count mismatch: input={cin.shape[0]}, output={cout.shape[0]}")

    kin = _row_keys_int32(cin)
    kout = _row_keys_int32(cout)

    order = np.argsort(kout, kind="mergesort")
    kout_s = kout[order]
    pos = np.searchsorted(kout_s, kin)

    if pos.shape[0] != kin.shape[0] or np.any(kout_s[pos] != kin):
        raise RuntimeError("Failed to align sparse output coordinates to input coordinates.")

    idx = torch.from_numpy(order[pos].astype(np.int64, copy=False)).to(device)
    return out.F[idx]


# -----------------------------------------------------------------------------
# Windowing
# -----------------------------------------------------------------------------


def make_origins(max_exclusive: int, win: int, stride: int) -> List[int]:
    max_exclusive = int(max_exclusive)
    win = int(win)
    stride = int(stride)

    if max_exclusive <= 0:
        return [0]
    if max_exclusive <= win:
        return [0]

    origins = list(range(0, max_exclusive - win + 1, stride))
    last = max_exclusive - win
    if origins[-1] != last:
        origins.append(last)
    return origins


def is_oom_exception(e: BaseException) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cudaerrormemoryallocation" in msg) or ("std::bad_alloc" in msg)


def run_model_on_voxel_subset(
    *,
    model: torch.nn.Module,
    coords_u: np.ndarray,
    feats_u: np.ndarray,
    sel: np.ndarray,
    num_classes: int,
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    coords_sub = coords_u[sel].astype(np.int32, copy=False)
    coords_sub = coords_sub - coords_sub.min(axis=0, keepdims=True)

    coords_b = np.concatenate(
        [np.zeros((coords_sub.shape[0], 1), dtype=np.int32), coords_sub],
        axis=1,
    )
    coords_t = torch.from_numpy(np.ascontiguousarray(coords_b, dtype=np.int32)).int().to(device)
    feats_t = torch.from_numpy(np.ascontiguousarray(feats_u[sel], dtype=np.float32)).float().to(device)

    st = ME.SparseTensor(features=feats_t, coordinates=coords_t, device=device)

    use_amp = bool(amp and device.type == "cuda")
    if hasattr(torch.cuda, "amp"):
        ctx = torch.cuda.amp.autocast(enabled=use_amp)
    else:
        ctx = nullcontext()

    with torch.inference_mode():
        with ctx:
            out_any = model(st)
            out = sparse_output_from_model_output(out_any)
        logits_t = align_sparse_output_to_input_coords(out=out, coords_in_t=coords_t, device=device)

    logits = logits_t.detach().float().cpu().numpy()
    if logits.shape[0] != sel.shape[0]:
        raise RuntimeError(f"Window logits row mismatch: logits={logits.shape[0]}, selected={sel.shape[0]}")
    if logits.shape[1] != num_classes:
        raise RuntimeError(f"Window logits class mismatch: logits={logits.shape[1]}, expected={num_classes}")

    del st, out_any, out, logits_t, coords_t, feats_t
    return logits.astype(np.float32, copy=False)


def recursively_infer_subset(
    *,
    model: torch.nn.Module,
    coords_u: np.ndarray,
    feats_u: np.ndarray,
    sel: np.ndarray,
    num_classes: int,
    device: torch.device,
    amp: bool,
    max_voxels_per_forward: int,
    depth: int,
    max_depth: int,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Inference for a selected voxel set. If selection is too large or OOMs,
    split by XY median and recurse. Returns logits [len(sel), C] aligned to sel order.
    """
    if sel.size == 0:
        return np.zeros((0, num_classes), dtype=np.float32)

    if sel.size <= int(max_voxels_per_forward):
        try:
            return run_model_on_voxel_subset(
                model=model,
                coords_u=coords_u,
                feats_u=feats_u,
                sel=sel,
                num_classes=num_classes,
                device=device,
                amp=amp,
            )
        except (RuntimeError, MemoryError) as e:
            if not is_oom_exception(e) or depth >= max_depth:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            logger.warning(f"[oom] subset n_vox={sel.size} depth={depth}; splitting and retrying")

    if depth >= max_depth:
        raise RuntimeError(
            f"Reached max split depth={max_depth} with n_vox={sel.size}. "
            f"Try lowering --window_size_m or --max_voxels_per_forward."
        )

    xy = coords_u[sel, :2]
    x_span = int(xy[:, 0].max() - xy[:, 0].min()) if sel.size else 0
    y_span = int(xy[:, 1].max() - xy[:, 1].min()) if sel.size else 0
    axis = 0 if x_span >= y_span else 1
    med = float(np.median(xy[:, axis]))

    left_mask = xy[:, axis] <= med
    right_mask = ~left_mask

    if not left_mask.any() or not right_mask.any():
        # Fallback split by count.
        order = np.argsort(xy[:, axis], kind="mergesort")
        mid = max(1, sel.size // 2)
        left_sel = sel[order[:mid]]
        right_sel = sel[order[mid:]]
    else:
        left_sel = sel[left_mask]
        right_sel = sel[right_mask]

    out = np.empty((sel.size, num_classes), dtype=np.float32)

    left_logits = recursively_infer_subset(
        model=model,
        coords_u=coords_u,
        feats_u=feats_u,
        sel=left_sel,
        num_classes=num_classes,
        device=device,
        amp=amp,
        max_voxels_per_forward=max_voxels_per_forward,
        depth=depth + 1,
        max_depth=max_depth,
        logger=logger,
    )
    right_logits = recursively_infer_subset(
        model=model,
        coords_u=coords_u,
        feats_u=feats_u,
        sel=right_sel,
        num_classes=num_classes,
        device=device,
        amp=amp,
        max_voxels_per_forward=max_voxels_per_forward,
        depth=depth + 1,
        max_depth=max_depth,
        logger=logger,
    )

    # Fill output aligned to original sel order.
    pos = {int(v): i for i, v in enumerate(sel.tolist())}
    for s, lg in zip(left_sel.tolist(), left_logits):
        out[pos[int(s)]] = lg
    for s, lg in zip(right_sel.tolist(), right_logits):
        out[pos[int(s)]] = lg

    return out


# -----------------------------------------------------------------------------
# Tile inference
# -----------------------------------------------------------------------------


@torch.inference_mode()
def infer_tile_point_logits(
    *,
    model: torch.nn.Module,
    raw: Dict[str, Any],
    source_cfg: Dict[str, Any],
    device: torch.device,
    amp: bool,
    window_size_m: float,
    window_stride_m: float,
    aggregation: str,
    max_voxels_per_forward: int,
    max_split_depth: int,
    logger: logging.Logger,
    target_domain: str,
) -> np.ndarray:
    """
    Returns point logits [N_points, source_num_classes].

    Steps:
      1. Build point features using source checkpoint FeatureConfig.
      2. Quantize full target tile using source checkpoint patch + voxelization config.
      3. Run sparse model over deterministic XY voxel windows.
      4. Aggregate voxel logits.
      5. Project voxel logits back to all original points using inverse_map.
    """
    data = source_cfg["data"]
    feat_cfg = FeatureConfig(**data["features"])
    voxel_cfg = VoxelizationConfig.from_cfg(data)
    ignore_index = int(data.get("label_space", {}).get("ignore_index", -100))
    num_classes = int(source_cfg["model"]["out_channels"])
    patch = data["patch"]

    coord_norm_factor = float(patch["coord_norm_factor"])
    voxel_size = float(patch["voxel_size"])

    xyz64 = np.asarray(raw["xyz"], dtype=np.float64)

    if xyz64.ndim != 2 or xyz64.shape[1] != 3:
        raise RuntimeError(f"raw['xyz'] must be [N,3], got {xyz64.shape}")

    n_points = int(xyz64.shape[0])
    if n_points == 0:
        return np.zeros((0, num_classes), dtype=np.float32)

    if not np.isfinite(xyz64).all():
        raise RuntimeError(f"raw['xyz'] contains NaN/Inf for target_domain={target_domain}")

    if bool(patch.get("make_local_coords", True)):
        xyz_min = xyz64.min(axis=0)

        if np.max(np.abs(xyz_min)) > 1e-4:
            raise RuntimeError(
                "get_raw() violated local-coordinate contract: "
                f"target_domain={target_domain}, "
                f"xyz_min={xyz_min.tolist()}, "
                "while source patch.make_local_coords=True."
            )

    xyz = xyz64.astype(np.float32, copy=False)

    # get_raw() from your dataset classes already applies make_local_coords according to patch_cfg.
    xyz_norm = (xyz / coord_norm_factor).astype(np.float32, copy=False)

    feats_p = build_features(
        xyz_local=xyz_norm,
        intensity=raw.get("intensity", None),
        return_number=raw.get("return_number", None),
        number_of_returns=raw.get("number_of_returns", None),
        rgb=(raw.get("rgb", None) if bool(getattr(feat_cfg, "use_rgb", False)) else None),
        cfg=feat_cfg,
    )

    expected_c = int(infer_in_channels(feat_cfg))
    if feats_p.shape[1] != expected_c:
        raise RuntimeError(f"Feature dim mismatch: got {feats_p.shape[1]}, expected {expected_c}")

    q = np.floor(xyz_norm / voxel_size).astype(np.int32, copy=False)

    vx = voxelize_from_q(
        q_int32=q,
        feats_p_f32=feats_p.astype(np.float32, copy=False),
        labels_p_i64=None,
        ignore_index=ignore_index,
        cfg=voxel_cfg,
        rng=None,
        return_maps=True,
        num_classes_hint=num_classes,
    )

    coords_u = vx["coords_u"].astype(np.int32, copy=False)
    feats_u = vx["feats_u"].astype(np.float32, copy=False)
    inverse_map = vx["inverse_map"]
    if inverse_map is None:
        raise RuntimeError("voxelize_from_q returned inverse_map=None; return_maps=True is required.")
    inverse_map = inverse_map.astype(np.int64, copy=False)

    n_vox = int(coords_u.shape[0])
    if n_vox == 0:
        return np.zeros((n_points, num_classes), dtype=np.float32)

    meters_per_voxel = voxel_size * coord_norm_factor
    win_vox = max(1, int(math.ceil(float(window_size_m) / meters_per_voxel)))
    stride_vox = max(1, int(math.ceil(float(window_stride_m) / meters_per_voxel)))

    x_origins = make_origins(int(coords_u[:, 0].max()) + 1, win_vox, stride_vox)
    y_origins = make_origins(int(coords_u[:, 1].max()) + 1, win_vox, stride_vox)

    logits_sum = np.zeros((n_vox, num_classes), dtype=np.float32)
    counts = np.zeros((n_vox,), dtype=np.uint16)

    for x0 in x_origins:
        x1 = x0 + win_vox
        mx = (coords_u[:, 0] >= x0) & (coords_u[:, 0] < x1)
        if not bool(mx.any()):
            continue

        for y0 in y_origins:
            y1 = y0 + win_vox
            sel = np.where(mx & (coords_u[:, 1] >= y0) & (coords_u[:, 1] < y1))[0].astype(np.int64, copy=False)
            if sel.size == 0:
                continue

            lg = recursively_infer_subset(
                model=model,
                coords_u=coords_u,
                feats_u=feats_u,
                sel=sel,
                num_classes=num_classes,
                device=device,
                amp=amp,
                max_voxels_per_forward=max_voxels_per_forward,
                depth=0,
                max_depth=max_split_depth,
                logger=logger,
            )

            logits_sum[sel] += lg
            counts[sel] += 1

    missed = np.where(counts == 0)[0]
    if missed.size > 0:
        raise RuntimeError(
            f"Window scheduler missed {missed.size}/{n_vox} voxels. " "This would leave points without predictions."
        )

    if aggregation == "mean_logits":
        logits_u = logits_sum / np.maximum(counts[:, None].astype(np.float32), 1.0)
    elif aggregation == "sum_logits":
        logits_u = logits_sum
    else:
        raise RuntimeError(f"Unknown aggregation={aggregation}")

    return logits_u[inverse_map].astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Main eval loop
# -----------------------------------------------------------------------------


def evaluate_common(
    *,
    model: torch.nn.Module,
    dataset_obj: Any,
    source_cfg: Dict[str, Any],
    source_domain: str,
    target_domain: str,
    source_train_to_common: np.ndarray,
    target_native_to_common_lut: np.ndarray,
    device: torch.device,
    amp: bool,
    window_size_m: float,
    window_stride_m: float,
    aggregation: str,
    max_voxels_per_forward: int,
    max_split_depth: int,
    limit_tiles: int,
    logger: logging.Logger,
) -> Dict[str, Any]:
    n_tiles_total = len(dataset_obj)
    n_tiles = n_tiles_total if int(limit_tiles) <= 0 else min(n_tiles_total, int(limit_tiles))

    # -------------------------
    # Common-space accumulators
    # -------------------------
    cm = np.zeros((len(COMMON_CLASS_NAMES), len(COMMON_CLASS_NAMES)), dtype=np.int64)
    gt_hist = np.zeros((len(COMMON_CLASS_NAMES),), dtype=np.int64)
    pred_hist = np.zeros((len(COMMON_CLASS_NAMES),), dtype=np.int64)

    n_points_total = 0
    n_points_eval = 0

    # -------------------------
    # Optional same-domain train-space accumulators
    # -------------------------
    same_domain = str(source_domain).lower().strip() == str(target_domain).lower().strip()

    source_num_classes = int(source_cfg["model"]["out_channels"])
    source_label_space = source_cfg.get("data", {}).get("label_space", {})
    source_ignore_index = int(source_label_space.get("ignore_index", -100))
    source_train_class_names = get_train_class_names(source_cfg, source_num_classes)

    cm_train = None
    gt_hist_train = None
    pred_hist_train = None
    source_native_to_train_lut = None

    if same_domain:
        cm_train = np.zeros((source_num_classes, source_num_classes), dtype=np.int64)
        gt_hist_train = np.zeros((source_num_classes,), dtype=np.int64)
        pred_hist_train = np.zeros((source_num_classes,), dtype=np.int64)

        source_native_to_train_lut, source_ignore_index = source_native_to_train_lut_from_config(
            source_cfg=source_cfg,
            source_domain=source_domain,
            logger=logger,
        )

        logger.info(
            "[train_space] enabled because source_domain == target_domain. "
            f"num_classes={source_num_classes}, ignore_index={source_ignore_index}, "
            f"class_names={source_train_class_names}"
        )
    else:
        logger.info("[train_space] disabled because source_domain != target_domain.")

    t0 = time.time()

    for ti in range(n_tiles):
        raw = dataset_obj.get_raw(ti)

        tile_name = str(raw.get("tile_name", raw.get("path", f"tile_{ti}")))
        logger.info(f"[tile] {ti + 1}/{n_tiles} {tile_name}")

        logits_pts = infer_tile_point_logits(
            model=model,
            raw=raw,
            source_cfg=source_cfg,
            device=device,
            amp=amp,
            window_size_m=window_size_m,
            window_stride_m=window_stride_m,
            aggregation=aggregation,
            max_voxels_per_forward=max_voxels_per_forward,
            max_split_depth=max_split_depth,
            logger=logger,
            target_domain=target_domain,
        )

        pred_train = logits_pts.argmax(axis=1).astype(np.int64, copy=False)

        if pred_train.size and (pred_train.min() < 0 or pred_train.max() >= source_train_to_common.shape[0]):
            raise RuntimeError(
                f"pred_train out of source_train_to_common range: "
                f"min={pred_train.min()}, max={pred_train.max()}, lut_size={source_train_to_common.shape[0]}"
            )

        # ------------------------------------------------------------------
        # Optional same-domain train-space metrics.
        # This is comparable to the training pipeline's native/train-space eval.
        # ------------------------------------------------------------------
        if same_domain:
            assert cm_train is not None
            assert gt_hist_train is not None
            assert pred_hist_train is not None
            assert source_native_to_train_lut is not None

            y_native = get_native_labels(raw, target_domain=target_domain)
            y_native_safe = np.clip(y_native, 0, source_native_to_train_lut.shape[0] - 1)
            gt_train = source_native_to_train_lut[y_native_safe].astype(np.int64, copy=False)

            if gt_train.shape[0] != pred_train.shape[0]:
                raise RuntimeError(
                    f"Train-space point-count mismatch for {tile_name}: "
                    f"gt_train={gt_train.shape[0]}, pred_train={pred_train.shape[0]}"
                )

            valid_train = gt_train != int(source_ignore_index)

            if np.any(valid_train):
                gt_valid = gt_train[valid_train]
                pred_valid = pred_train[valid_train]

                bad_gt = (gt_valid < 0) | (gt_valid >= source_num_classes)
                bad_pred = (pred_valid < 0) | (pred_valid >= source_num_classes)

                if np.any(bad_gt):
                    raise RuntimeError(
                        f"Train-space GT out of range for {tile_name}: "
                        f"min={gt_valid.min()}, max={gt_valid.max()}, C={source_num_classes}"
                    )
                if np.any(bad_pred):
                    raise RuntimeError(
                        f"Train-space pred out of range for {tile_name}: "
                        f"min={pred_valid.min()}, max={pred_valid.max()}, C={source_num_classes}"
                    )

                cm_train += confusion_from_labels(
                    gt_train,
                    pred_train,
                    num_classes=source_num_classes,
                    ignore_index=source_ignore_index,
                )
                gt_hist_train += np.bincount(gt_valid.astype(np.int64), minlength=source_num_classes)
                pred_hist_train += np.bincount(pred_valid.astype(np.int64), minlength=source_num_classes)

        # ------------------------------------------------------------------
        # Common-space metrics.
        # This is the official cross-domain metric.
        # ------------------------------------------------------------------
        pred_common = source_train_to_common[pred_train].astype(np.int64, copy=False)
        gt_common = map_target_native_to_common(
            raw,
            lut=target_native_to_common_lut,
            target_domain=target_domain,
        )

        if gt_common.shape[0] != pred_common.shape[0]:
            raise RuntimeError(f"Point-count mismatch for {tile_name}: gt={gt_common.shape[0]}, pred={pred_common.shape[0]}")

        valid = gt_common != COMMON_IGNORE_ID
        cm += confusion_from_labels(
            gt_common,
            pred_common,
            num_classes=len(COMMON_CLASS_NAMES),
            ignore_index=COMMON_IGNORE_ID,
        )

        gt_hist += np.bincount(gt_common[valid].astype(np.int64), minlength=len(COMMON_CLASS_NAMES))
        pred_hist += np.bincount(pred_common[valid].astype(np.int64), minlength=len(COMMON_CLASS_NAMES))

        n_points_total += int(gt_common.shape[0])
        n_points_eval += int(valid.sum())

        partial = metrics_from_cm(cm, COMMON_CLASS_NAMES, ignore_index=COMMON_IGNORE_ID)
        logger.info(
            f"[tile_done] {ti + 1}/{n_tiles} "
            f"points={gt_common.shape[0]} eval_points_common={int(valid.sum())} "
            f"running_common_miou={partial['miou']:.6f} "
            f"running_common_macro_f1={partial['macro_f1']:.6f}"
        )
        logger.info(f"[hist_common] gt_common={gt_hist.tolist()}")
        logger.info(f"[hist_common] pred_common={pred_hist.tolist()}")

        if same_domain and cm_train is not None:
            partial_train = metrics_from_cm(
                cm_train,
                source_train_class_names,
                ignore_index=source_ignore_index,
            )
            logger.info(
                f"[train_space] running_train_miou={partial_train['miou']:.6f} "
                f"running_train_macro_f1={partial_train['macro_f1']:.6f}"
            )
            logger.info(f"[hist_train] gt_train={gt_hist_train.tolist() if gt_hist_train is not None else None}")
            logger.info(f"[hist_train] pred_train={pred_hist_train.tolist() if pred_hist_train is not None else None}")

        if device.type == "cuda":
            logger.info(
                f"[cuda] alloc={torch.cuda.memory_allocated() / 1e9:.3f}GB "
                f"reserved={torch.cuda.memory_reserved() / 1e9:.3f}GB"
            )
            torch.cuda.empty_cache()
        gc.collect()

    runtime_s = float(time.time() - t0)

    metrics = metrics_from_cm(cm, COMMON_CLASS_NAMES, ignore_index=COMMON_IGNORE_ID)
    metrics.update(
        {
            "runtime_s": runtime_s,
            "n_tiles": int(n_tiles),
            "n_tiles_total_available": int(n_tiles_total),
            "n_points_total": int(n_points_total),
            "n_points_eval": int(n_points_eval),
            "gt_hist_common": gt_hist.tolist(),
            "pred_hist_common": pred_hist.tolist(),
            "source_train_to_common": source_train_to_common.tolist(),
            "target_native_to_common_lut_nonzero": {
                str(i): int(v) for i, v in enumerate(target_native_to_common_lut.tolist()) if int(v) != COMMON_IGNORE_ID
            },
            "common_class_names": COMMON_CLASS_NAMES,
        }
    )

    if same_domain:
        assert cm_train is not None
        assert gt_hist_train is not None
        assert pred_hist_train is not None

        train_metrics = metrics_from_cm(
            cm_train,
            source_train_class_names,
            ignore_index=source_ignore_index,
        )
        train_metrics.update(
            {
                "runtime_s": runtime_s,
                "n_tiles": int(n_tiles),
                "n_tiles_total_available": int(n_tiles_total),
                "n_points_total": int(n_points_total),
                "n_points_eval": int(gt_hist_train.sum()),
                "gt_hist_train": gt_hist_train.tolist(),
                "pred_hist_train": pred_hist_train.tolist(),
                "train_class_names": source_train_class_names,
                "source_domain": str(source_domain),
                "target_domain": str(target_domain),
                "source_ignore_index": int(source_ignore_index),
                "note": (
                    "Same-domain source train-space metrics. "
                    "Comparable to native/train-space training-pipeline eval, not to common-space cross-domain metrics."
                ),
            }
        )
        metrics["train_space"] = train_metrics

    return metrics


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source_config", required=True, help="Exact resolved YAML/JSON config for source checkpoint.")
    ap.add_argument("--target_config", required=True, help="YAML/JSON config containing target dataset paths/cache settings.")

    ap.add_argument("--source_domain", required=True, choices=["eclair", "dales"])
    ap.add_argument("--target_domain", required=True, choices=["eclair", "dales"])
    ap.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "all"],
        help=(
            "Evaluation split. Use 'test' for official reporting. "
            "Use 'all' only for full-labelled-target diagnostic cross-eval."
        ),
    )

    ap.add_argument("--source_train_to_common", required=True, help="YAML map: source train_id -> common_id.")
    ap.add_argument(
        "--source_native_to_common",
        required=True,
        help=(
            "YAML map: source native label_id -> common_id. "
            "Used only for validating that source_train_to_common agrees with the training config."
        ),
    )
    ap.add_argument("--target_native_to_common", required=True, help="YAML map: target native label_id -> common_id.")

    ap.add_argument("--target_manifest", default=None, help="Optional exact DALES file manifest.")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--window_size_m", type=float, default=120.0)
    ap.add_argument("--window_stride_m", type=float, default=120.0)
    ap.add_argument("--aggregation", default="mean_logits", choices=["mean_logits", "sum_logits"])

    ap.add_argument("--max_voxels_per_forward", type=int, default=180000)
    ap.add_argument("--max_split_depth", type=int, default=8)

    ap.add_argument("--amp", action="store_true", help="Use CUDA AMP. Recommended only after non-AMP sanity passes.")
    ap.add_argument("--limit_tiles", type=int, default=0, help="Debug only: evaluate first N target tiles.")
    ap.add_argument("--force_raw_cache", action="store_true", help="Force DALES cache_kind='raw'. Recommended for official eval.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    logger = setup_logger(out_dir)
    set_all_seeds(int(args.seed))

    source_cfg = load_config_any(args.source_config)
    target_cfg = load_config_any(args.target_config)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        logger.warning("CUDA requested but unavailable; falling back to CPU.")

    logger.info("[start] eval_cross_common")
    logger.info(f"[args] {json.dumps(vars(args), indent=2, default=str)}")

    model = load_model_for_eval(
        source_cfg=source_cfg,
        ckpt_path=args.ckpt,
        device=device,
        logger=logger,
    )

    source_out_channels = int(source_cfg["model"]["out_channels"])
    source_train_to_common = mapping_to_lut(
        load_yaml_int_map(args.source_train_to_common),
        size=source_out_channels,
        default=COMMON_IGNORE_ID,
        name="source_train_to_common",
    )

    source_native_to_common_lut = mapping_to_lut(
        load_yaml_int_map(args.source_native_to_common),
        size=256,
        default=COMMON_IGNORE_ID,
        name="source_native_to_common",
    )

    target_native_to_common_lut = mapping_to_lut(
        load_yaml_int_map(args.target_native_to_common),
        size=256,
        default=COMMON_IGNORE_ID,
        name="target_native_to_common",
    )

    validate_source_train_to_common_mapping(
        source_cfg=source_cfg,
        source_domain=args.source_domain,
        source_train_to_common=source_train_to_common,
        source_native_to_common_lut=source_native_to_common_lut,
        logger=logger,
    )

    dataset_obj = build_target_dataset(
        target_domain=args.target_domain,
        target_cfg=target_cfg,
        source_cfg=source_cfg,
        split=args.split,
        target_manifest=args.target_manifest,
        force_raw_cache=bool(args.force_raw_cache),
        logger=logger,
    )

    manifest = {
        "ckpt": str(args.ckpt),
        "source_config": str(args.source_config),
        "target_config": str(args.target_config),
        "source_domain": str(args.source_domain),
        "target_domain": str(args.target_domain),
        "split": str(args.split),
        "source_train_to_common": str(args.source_train_to_common),
        "source_native_to_common": str(args.source_native_to_common),
        "target_native_to_common": str(args.target_native_to_common),
        "target_manifest": str(args.target_manifest) if args.target_manifest else None,
        "window_size_m": float(args.window_size_m),
        "window_stride_m": float(args.window_stride_m),
        "aggregation": str(args.aggregation),
        "max_voxels_per_forward": int(args.max_voxels_per_forward),
        "max_split_depth": int(args.max_split_depth),
        "amp": bool(args.amp and device.type == "cuda"),
        "device": device.type,
        "source_model": source_cfg.get("model", {}),
        "source_features": source_cfg.get("data", {}).get("features", {}),
        "source_patch": source_cfg.get("data", {}).get("patch", {}),
        "source_voxelization": source_cfg.get("data", {}).get("voxelization", {}),
        "common_class_names": COMMON_CLASS_NAMES,
    }
    (out_dir / "eval_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(f"[saved] {out_dir / 'eval_manifest.json'}")

    metrics = evaluate_common(
        model=model,
        dataset_obj=dataset_obj,
        source_cfg=source_cfg,
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        source_train_to_common=source_train_to_common,
        target_native_to_common_lut=target_native_to_common_lut,
        device=device,
        amp=bool(args.amp and device.type == "cuda"),
        window_size_m=float(args.window_size_m),
        window_stride_m=float(args.window_stride_m),
        aggregation=str(args.aggregation),
        max_voxels_per_forward=int(args.max_voxels_per_forward),
        max_split_depth=int(args.max_split_depth),
        limit_tiles=int(args.limit_tiles),
        logger=logger,
    )

    train_space_metrics = metrics.pop("train_space", None)

    write_metric_files(metrics, out_dir)
    np.save(out_dir / "confusion_common.npy", np.asarray(metrics["confusion_matrix"], dtype=np.int64))

    if train_space_metrics is not None:
        write_train_space_metric_files(train_space_metrics, out_dir)
        logger.info(
            f"[done_train_space] train_mIoU={train_space_metrics['miou']:.6f} "
            f"train_macroF1={train_space_metrics['macro_f1']:.6f}"
        )
        logger.info(f"[saved] {out_dir / 'metrics_train_space.json'}")
        logger.info(f"[saved] {out_dir / 'summary_train_space.csv'}")
        logger.info(f"[saved] {out_dir / 'per_class_train_space.csv'}")
        logger.info(f"[saved] {out_dir / 'confusion_train_space.npy'}")

    logger.info(f"[done_common] common_mIoU={metrics['miou']:.6f} common_macroF1={metrics['macro_f1']:.6f}")
    logger.info(f"[saved] {out_dir / 'metrics_common.json'}")
    logger.info(f"[saved] {out_dir / 'summary_common.csv'}")
    logger.info(f"[saved] {out_dir / 'per_class_common.csv'}")
    logger.info(f"[saved] {out_dir / 'confusion_common.npy'}")


if __name__ == "__main__":
    main()
