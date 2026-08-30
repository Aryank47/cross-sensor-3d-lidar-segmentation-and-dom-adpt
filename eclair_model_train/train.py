# /eclair_model_train/train.py
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn.functional as F
from src.augment import AugmentConfig
from src.bev_als_config import ALSBEVConfig, bev_als_weight_for_epoch
from src.bev_als_head import als_bev_multilabel_loss
from src.bev_head import BEVHeadConfig
from src.bev_labels import build_bev_labels_and_selected_idx
from src.config_loader import load_yaml
from src.data_dales import (
    DalesCropConfig,
    DalesPatchConfig,
    DalesPreprocConfig,
    DalesTiles,
    _find_dales_files,
    minkowski_collate_dales,
)
from src.data_eclair import EclairSamplingConfig, EclairTiles, PatchConfig, minkowski_collate_fn
from src.dist import DistEnv, all_reduce_sum, init_distributed, is_main_process
from src.features import FeatureConfig, build_features, infer_in_channels
from src.label_maps import ECLAIR_CLASS_NAMES_11
from src.losses import FocalLoss, FocalLossConfig, FocalLovaszLoss, LovaszSoftmaxLoss, LovaszWarmupConfig
from src.metrics import ConfusionMatrix
from src.method_config import MethodContract, validate_method_contract
from src.mix3d import ALSMix3DConfig
from src.model import build_model
from src.ocons import (
    OConsConfig,
    make_masked_view,
    match_sparse_coordinates,
    ocons_consistency_loss,
    ocons_weight_for_epoch,
    stable_ocons_seed,
)
from src.utils import CSVLogger, atomic_save_torch, format_seconds, save_json, set_seed, unwrap_model
from src.voxelization import VoxelizationConfig, voxelize_from_q
from torch.utils.data import DataLoader


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += int(p.stat().st_size)
        except FileNotFoundError:
            continue
    return total


def _prune_epoch_snapshots(checkpoint_dir: Path, keep_last_k: int) -> List[str]:
    snapshots = sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: p.name)
    keep = max(0, int(keep_last_k))
    doomed = snapshots if keep == 0 else snapshots[:-keep]
    removed = []
    for path in doomed:
        try:
            path.unlink()
            removed.append(path.name)
        except FileNotFoundError:
            pass
    return removed


def _cleanup_checkpoint_temp_files(checkpoint_dir: Path) -> List[str]:
    removed = []
    for path in checkpoint_dir.glob("*.tmp"):
        try:
            path.unlink()
            removed.append(path.name)
        except FileNotFoundError:
            pass
    return removed


def _eclair_run_cache_key(cfg: Dict[str, Any]) -> str:
    """
    Hash only the parts that affect ECLAIR invariants we cache.
    If any of these change, we must regenerate the run-cache.
    """
    data = cfg.get("data", {}) or {}
    ls = data.get("label_space", {}) or {}
    patch = data.get("patch", {}) or {}
    feats = data.get("features", {}) or {}

    obj = {
        "patch": {
            "make_local_coords": bool(patch.get("make_local_coords", True)),
            "coord_norm_factor": float(patch.get("coord_norm_factor", 10.0)),
        },
        "label_space": {
            "ignore_index": int(ls.get("ignore_index", -100)),
            "eclair_undefined_id": int(ls.get("eclair_undefined_id", 0)),
            "num_classes": int(ls.get("num_classes", 11)),
        },
        "features": {
            "use_intensity": bool(feats.get("use_intensity", False)),
            "intensity_divisor": float(feats.get("intensity_divisor", 65535.0)),
            "use_return_number": bool(feats.get("use_return_number", True)),
            "use_number_of_returns": bool(feats.get("use_number_of_returns", True)),
            "returns_onehot_k": int(feats.get("returns_onehot_k", 5)),
            "use_rgb": bool(feats.get("use_rgb", False)),
            "include_coords": bool(feats.get("include_coords", False)),
        },
    }
    s = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:12]


def _model_forward_safe(
    model: torch.nn.Module,
    st: ME.SparseTensor,
    *,
    is_train: bool,
    bev_selected_idx=None,
    compute_bev: Optional[bool] = None,
    bev_als_frames=None,
    compute_bev_als: Optional[bool] = None,
):
    """
    Call model.forward() with only supported kwargs.
    Works with DDP wrappers, explicit params, and **kwargs forwards.
    """
    m = model.module if hasattr(model, "module") else model
    sig = inspect.signature(m.forward)
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    kwargs = {}
    if has_varkw or ("is_train" in params):
        # Only include if explicitly supported OR forward has **kwargs.
        if "is_train" in params or has_varkw:
            kwargs["is_train"] = is_train
    if bev_selected_idx is not None and (has_varkw or ("bev_selected_idx" in params)):
        kwargs["bev_selected_idx"] = bev_selected_idx
    if compute_bev is not None and (has_varkw or ("compute_bev" in params)):
        kwargs["compute_bev"] = bool(compute_bev)
    if bev_als_frames is not None and (has_varkw or ("bev_als_frames" in params)):
        kwargs["bev_als_frames"] = bev_als_frames
    if compute_bev_als is not None and (has_varkw or ("compute_bev_als" in params)):
        kwargs["compute_bev_als"] = bool(compute_bev_als)

    return model(st, **kwargs)


def _read_manifest(p: Path) -> List[Path]:
    lines = p.read_text().splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        out.append(Path(ln))
    return out


def build_dataloaders(cfg: Dict[str, Any], dist_env: DistEnv, *, eclair_run_cache_dir: Optional[Path] = None):
    data = cfg["data"]
    dataset = str(data.get("dataset", "eclair")).lower()

    feat_cfg = FeatureConfig(**data["features"])
    voxel_cfg = VoxelizationConfig.from_cfg(data)

    bev_cfg = BEVHeadConfig.from_cfg(cfg)
    bev_als_cfg = ALSBEVConfig.from_cfg(cfg)
    mix3d_cfg = ALSMix3DConfig.from_cfg(data.get("mix3d", {}))
    if mix3d_cfg.enabled and (bev_cfg.enabled or bev_als_cfg.enabled):
        raise ValueError("Initial M1 forbids model.aux_heads.bev.enabled=true.")

    # -------------------------
    # Feature-stack sanity check (prevents silent channel bugs)
    # -------------------------
    expected_c = infer_in_channels(feat_cfg)
    model_c = int(cfg["model"]["in_channels"])
    if model_c != expected_c:
        raise ValueError(
            f"model.in_channels={model_c} but inferred={expected_c} "
            f"from data.features.(For Option-A returns-only with k=5 "
            f"=> expected 10.)"
        )

    ls = data["label_space"]
    ignore_index = int(ls["ignore_index"])

    if dataset == "eclair":
        eclair_aug_cfg = AugmentConfig(**data["aug"])

        eclair_sampling_cfg = EclairSamplingConfig.from_cfg(data.get("sampling", {}))

        eclair_cfg = cfg.get("eclair", {})
        meta_filename = eclair_cfg.get("meta_filename", "labels.json")
        train_cats = eclair_cfg.get("train_review_categories", None)  # None = no filtering
        val_cats = eclair_cfg.get("val_review_categories", ["approved"])
        test_cats = eclair_cfg.get("test_review_categories", ["approved"])

        patch_cfg = PatchConfig(**data["patch"])

        undefined_id = int(ls["eclair_undefined_id"])

        train_ds = EclairTiles(
            eclair_root=data["eclair_root"],
            split="train",
            is_train=True,
            patch_cfg=patch_cfg,
            aug_cfg=eclair_aug_cfg,
            feat_cfg=feat_cfg,
            ignore_index=ignore_index,
            undefined_id=undefined_id,
            use_cache=bool(data.get("use_cache", True)),
            cache_root=data.get("cache_root", None),
            seed=int(cfg["run"]["seed"]),
            meta_filename=meta_filename,
            allowed_review_categories=train_cats,
            voxel_cfg=voxel_cfg,
            bev_cfg=bev_cfg,
            bev_als_cfg=bev_als_cfg,
            num_classes=int(ls["num_classes"]),
            run_cache_root=str(eclair_run_cache_dir) if eclair_run_cache_dir is not None else None,
            run_cache_precompute_returns_onehot=(
                eclair_run_cache_dir is not None
                and bool(data.get("run_cache_precompute_returns_onehot", False))
                and (feat_cfg.use_return_number or feat_cfg.use_number_of_returns)
            ),
            sampling_cfg=eclair_sampling_cfg,
            mix3d_cfg=mix3d_cfg,
        )
        val_ds = EclairTiles(
            eclair_root=data["eclair_root"],
            split="val",
            is_train=False,
            patch_cfg=patch_cfg,
            aug_cfg=eclair_aug_cfg,  # aug_cfg ignored when is_train=False
            feat_cfg=feat_cfg,
            ignore_index=ignore_index,
            undefined_id=undefined_id,
            use_cache=bool(data.get("use_cache", True)),
            cache_root=data.get("cache_root", None),
            seed=int(cfg["run"]["seed"]) + 1,
            meta_filename=meta_filename,
            allowed_review_categories=val_cats,
            voxel_cfg=voxel_cfg,
            bev_cfg=bev_cfg,
            bev_als_cfg=ALSBEVConfig(enabled=False),
            num_classes=int(ls["num_classes"]),
            run_cache_root=str(eclair_run_cache_dir) if eclair_run_cache_dir is not None else None,
            run_cache_precompute_returns_onehot=(
                eclair_run_cache_dir is not None
                and bool(data.get("run_cache_precompute_returns_onehot", False))
                and (feat_cfg.use_return_number or feat_cfg.use_number_of_returns)
            ),
            sampling_cfg=EclairSamplingConfig(mode="tiles"),
            mix3d_cfg=ALSMix3DConfig(enabled=False),
        )
        test_ds = EclairTiles(
            eclair_root=data["eclair_root"],
            split="test",
            is_train=False,
            patch_cfg=patch_cfg,
            aug_cfg=eclair_aug_cfg,
            feat_cfg=feat_cfg,
            ignore_index=ignore_index,
            undefined_id=undefined_id,
            use_cache=bool(data.get("use_cache", True)),
            cache_root=data.get("cache_root", None),
            seed=int(cfg["run"]["seed"]) + 2,
            meta_filename=meta_filename,
            allowed_review_categories=test_cats,
            voxel_cfg=voxel_cfg,
            bev_cfg=bev_cfg,
            bev_als_cfg=ALSBEVConfig(enabled=False),
            num_classes=int(ls["num_classes"]),
            run_cache_root=str(eclair_run_cache_dir) if eclair_run_cache_dir is not None else None,
            run_cache_precompute_returns_onehot=(
                eclair_run_cache_dir is not None
                and bool(data.get("run_cache_precompute_returns_onehot", False))
                and (feat_cfg.use_return_number or feat_cfg.use_number_of_returns)
            ),
            sampling_cfg=EclairSamplingConfig(mode="tiles"),
            mix3d_cfg=ALSMix3DConfig(enabled=False),
        )

        collate = minkowski_collate_fn

    elif dataset == "dales":
        # DALES has train/ and test/ only in your setup.
        # We create val by splitting train deterministically unless a val root is provided.
        dales_aug_cfg = AugmentConfig(**data["aug"]) if data.get("aug", None) else AugmentConfig(enabled=False)
        label_map = data.get("dales_label_map_native_to_train", None)
        if label_map is not None:
            label_map = {int(k): int(v) for k, v in label_map.items()}

        dales_train_root = Path(data["dales_train_root"])
        dales_test_root = Path(data["dales_test_root"])
        dales_val_root = Path(data["dales_val_root"]) if "dales_val_root" in data else None

        patch_cfg = DalesPatchConfig(**data["patch"])
        preproc_cfg = DalesPreprocConfig(**data.get("preproc", {}))

        # cache controls
        use_cache = bool(data.get("use_cache", True))
        cache_root = data.get("cache_root", None)
        cache_subdir = data.get("cache_subdir", "dales_dropI")
        cache_key_extra = data.get("cache_key_extra", None)

        sampling = data.get("sampling", {}) or {}
        sampling_mode = str(sampling.get("mode", "tiles")).lower()
        crop_kwargs = {k: v for k, v in sampling.items() if k != "mode"}
        crop_cfg = DalesCropConfig(**crop_kwargs)

        cache_kind = str(data.get("cache_kind", "voxel")).lower()
        require_cache = bool(data.get("require_cache", False))
        write_cache = bool(data.get("write_cache", False))

        # -------------------------
        # IMPORTANT: runtime augmentation requires RAW cache.
        # If cache_kind='voxel', __getitem__ can bypass aug by returning cached tensors.
        # Make this a hard error to avoid silent training bugs.
        # -------------------------
        if bool(dales_aug_cfg.enabled) and cache_kind != "raw":
            raise ValueError(
                "DALES: data.aug.enabled=true requires data.cache_kind='raw'. "
                "With cache_kind='voxel', cached samples can bypass runtime augmentations."
            )

        split_manifest_dir = data.get("split_manifest_dir", None)
        if split_manifest_dir is not None:
            smd = Path(str(split_manifest_dir))
            tr = smd / "train.txt"
            va = smd / "val.txt"
            if tr.exists() and va.exists():
                train_files = _read_manifest(tr)
                val_files = _read_manifest(va)
            else:
                raise RuntimeError(f"split_manifest_dir set but train.txt/val.txt missing: {smd}")
        else:
            if dales_val_root is not None and dales_val_root.exists():
                train_files = _find_dales_files(dales_train_root)
                val_files = _find_dales_files(dales_val_root)
            else:
                all_train = _find_dales_files(dales_train_root)
                val_frac = float(data.get("val_fraction_from_train", 0.125))
                rng = random.Random(int(cfg["run"]["seed"]) + 777)
                all_train = sorted(all_train)
                rng.shuffle(all_train)
                n_val = max(1, int(round(len(all_train) * val_frac)))
                val_files = all_train[:n_val]
                train_files = all_train[n_val:]

        test_manifest = None
        if split_manifest_dir is not None:
            test_manifest = Path(str(split_manifest_dir)) / "test.txt"

        if test_manifest is not None and test_manifest.exists():
            test_files = _read_manifest(test_manifest)
        else:
            test_files = _find_dales_files(dales_test_root)

        train_ds = DalesTiles(
            dales_root=dales_train_root,
            files=train_files,
            patch_cfg=patch_cfg,
            feat_cfg=feat_cfg,
            is_train=True,
            aug_cfg=dales_aug_cfg,
            ignore_index=ignore_index,
            preproc=preproc_cfg,
            seed=int(cfg["run"]["seed"]),
            use_cache=use_cache,
            cache_root=cache_root,
            cache_subdir=str(cache_subdir),
            cache_key_extra=cache_key_extra,
            split_name="train",
            label_map=label_map,
            cache_kind=cache_kind,
            require_cache=require_cache,
            write_cache=write_cache,
            sampling_mode=sampling_mode,
            crop_cfg=crop_cfg,
            voxel_cfg=voxel_cfg,
            bev_cfg=bev_cfg,
            bev_als_cfg=bev_als_cfg,
            num_classes=int(ls["num_classes"]),
            mix3d_cfg=mix3d_cfg,
        )
        val_ds = DalesTiles(
            dales_root=dales_train_root,
            files=val_files,
            patch_cfg=patch_cfg,
            feat_cfg=feat_cfg,
            is_train=False,
            aug_cfg=dales_aug_cfg,
            ignore_index=ignore_index,
            preproc=preproc_cfg,
            seed=int(cfg["run"]["seed"]) + 1,
            use_cache=use_cache,
            cache_root=cache_root,
            cache_subdir=str(cache_subdir),
            cache_key_extra=cache_key_extra,
            split_name="val",
            label_map=label_map,
            cache_kind=cache_kind,
            require_cache=require_cache,
            write_cache=write_cache,
            sampling_mode="tiles",  # recommended: deterministic eval (see note below)
            crop_cfg=crop_cfg,
            voxel_cfg=voxel_cfg,
            bev_cfg=bev_cfg,
            bev_als_cfg=ALSBEVConfig(enabled=False),
            num_classes=int(ls["num_classes"]),
            mix3d_cfg=ALSMix3DConfig(enabled=False),
        )
        test_ds = DalesTiles(
            dales_root=dales_test_root,
            files=test_files,
            patch_cfg=patch_cfg,
            feat_cfg=feat_cfg,
            is_train=False,
            aug_cfg=dales_aug_cfg,
            ignore_index=ignore_index,
            preproc=preproc_cfg,
            seed=int(cfg["run"]["seed"]) + 2,
            use_cache=use_cache,
            cache_root=cache_root,
            cache_subdir=str(cache_subdir),
            cache_key_extra=cache_key_extra,
            split_name="test",
            label_map=label_map,
            cache_kind=cache_kind,
            require_cache=require_cache,
            write_cache=write_cache,
            sampling_mode="tiles",
            crop_cfg=crop_cfg,
            voxel_cfg=voxel_cfg,
            bev_cfg=bev_cfg,
            bev_als_cfg=ALSBEVConfig(enabled=False),
            num_classes=int(ls["num_classes"]),
            mix3d_cfg=ALSMix3DConfig(enabled=False),
        )

        # NOTE: caching is handled inside DalesTiles only if you add the
        # cache params as in my earlier patch; if you want that,
        # tell me and I’ll diff it cleanly against your current
        # src/data_dales.py version.

        collate = minkowski_collate_dales

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Distributed samplers (optional)
    if dist_env.enabled:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    dl_kwargs = dict(
        batch_size=int(data["batch_size"]),
        num_workers=int(data["num_workers"]),
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=int(data["num_workers"]) > 0,
    )
    # Speed: prefetch more batches per worker (PyTorch default is 2).
    if int(data["num_workers"]) > 0:
        dl_kwargs["prefetch_factor"] = int(data.get("prefetch_factor", 4))

    train_loader = DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        **dl_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, sampler=val_sampler, drop_last=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, sampler=test_sampler, drop_last=False, **dl_kwargs)
    return train_loader, val_loader, test_loader


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module):
    optim_cfg = cfg["optim"]
    name = optim_cfg["name"].lower()
    lr = float(optim_cfg["lr"])
    wd = float(optim_cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
    sched = cfg.get("sched", {"name": "none"})
    name = str(sched.get("name", "none")).lower()
    if name in ("none", "", None):
        return None
    if name == "step":
        step_size = int(sched.get("step_size_epochs", 10))
        gamma = float(sched.get("gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f"Unknown scheduler: {name}")


def _bev_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(((cfg.get("model", {}) or {}).get("aux_heads", {}) or {}).get("bev", {}).get("enabled", False))


def _bev_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return ((cfg.get("model", {}) or {}).get("aux_heads", {}) or {}).get("bev", {}) or {}


def _bev_weight_for_epoch(bev_cfg: Dict[str, Any], epoch: int) -> float:
    # simple constant weight for now (you can add ramps later)
    return float(bev_cfg.get("weight", 0.5))


def soft_dice_loss_2d(
    logits: torch.Tensor,  # [B,C,H,W]
    target: torch.Tensor,  # [B,H,W] int64
    *,
    ignore_index: int,
    smooth: float = 1.0,
    classes: str = "present",  # "present" | "all"
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0

    B, C, H, W = logits.shape
    target = target.to(torch.int64)

    valid = target != ignore_index
    if valid.sum() == 0:
        return logits.sum() * 0.0

    probs = F.softmax(logits, dim=1)  # [B,C,H,W]

    # one-hot target (clamp invalid to 0 just for scatter, then mask out)
    t_clamped = target.clamp_min(0)
    tgt_1h = torch.zeros((B, C, H, W), device=logits.device, dtype=probs.dtype)
    tgt_1h.scatter_(1, t_clamped.unsqueeze(1), 1.0)

    m = valid.unsqueeze(1).to(probs.dtype)
    probs = probs * m
    tgt_1h = tgt_1h * m

    # per-class dice over batch+spatial
    dims = (0, 2, 3)
    inter = (probs * tgt_1h).sum(dim=dims)  # [C]
    denom = probs.sum(dim=dims) + tgt_1h.sum(dim=dims)  # [C]
    dice = (2.0 * inter + smooth) / (denom + smooth)  # [C]
    loss_c = 1.0 - dice  # [C]

    if classes == "present":
        present = tgt_1h.sum(dim=dims) > 0  # [C]
        if present.sum() == 0:
            return logits.sum() * 0.0
        return loss_c[present].mean()

    return loss_c.mean()


def _compute_bev_loss(
    bev_pred,  # dict[level] -> [B,C,H,W]
    bev_targets,  # dict[level] -> [B,H,W]
    *,
    loss_type: str,
    ignore_index: int,
    dice_smooth: float,
    dice_classes: str,
    ce_criterion: torch.nn.Module,
) -> torch.Tensor:
    # average across levels (LiDOG-style dict outputs)
    levels = list(bev_pred.keys())
    loss = None
    for lvl in levels:
        logits = bev_pred[lvl]
        tgt = bev_targets[lvl].to(torch.int64)

        if loss_type == "dice":
            l = soft_dice_loss_2d(
                logits,
                tgt,
                ignore_index=ignore_index,
                smooth=float(dice_smooth),
                classes=str(dice_classes),
            )
        elif loss_type == "ce":
            l = ce_criterion(logits, tgt)
        else:
            raise ValueError(f"Unknown BEV loss_type='{loss_type}' (expected 'dice' or 'ce').")

        loss = l if loss is None else (loss + l)

    if loss is None:
        # no levels present
        return torch.tensor(0.0, device=next(iter(bev_pred.values())).device)

    return loss / float(len(levels))


def _colorize_label_map(lbl: np.ndarray, *, num_classes: int, ignore_index: int) -> np.ndarray:
    # lbl: [H,W] int
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colors[0] = np.array([0, 0, 0], dtype=np.uint8)

    out = np.zeros((lbl.shape[0], lbl.shape[1], 3), dtype=np.uint8)
    valid = (lbl != ignore_index) & (lbl >= 0) & (lbl < num_classes)
    out[valid] = colors[lbl[valid]]
    # ignored pixels in gray
    out[~valid] = np.array([120, 120, 120], dtype=np.uint8)
    return out


@torch.no_grad()
def eval_and_visualize_bev(
    *,
    model: torch.nn.Module,
    dist_env,
    loader,
    device: torch.device,
    amp: bool,
    out_dir: Path,
    epoch: int,
    num_classes: int,
    cfg: Dict[str, Any],
):
    model.eval()

    eval_bev = (cfg.get("eval", {}) or {}).get("bev", {}) or {}
    if not bool(eval_bev.get("enabled", False)):
        return None

    every = int(eval_bev.get("every_epochs", 5))
    if (epoch % every) != 0:
        return None

    max_batches = int(eval_bev.get("max_batches", 2))
    save_npz = bool(eval_bev.get("save_npz", True))
    save_png = bool(eval_bev.get("save_png", True))

    # Use the structured config object for levels/img sizes/ignore
    bev_head_cfg = BEVHeadConfig.from_cfg(cfg)
    levels_req = eval_bev.get("levels", None)
    if levels_req is None:
        levels_req = list(bev_head_cfg.levels)

    bev_ignore = int(bev_head_cfg.ignore_index)
    voxel_ignore = int(cfg["data"]["label_space"]["ignore_index"])

    # meters per voxel in YOUR coordinate convention
    patch_cfg = loader.dataset.patch_cfg
    m_per_vox = float(patch_cfg.voxel_size) * float(patch_cfg.coord_norm_factor)

    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=bev_ignore)

    vis_dir = out_dir / "bev_vis" / f"epoch_{epoch:03d}"
    if is_main_process(dist_env):
        vis_dir.mkdir(parents=True, exist_ok=True)

    for batch_i, batch in enumerate(loader):
        if batch_i >= max_batches:
            break

        # ---------- Build sparse input ----------
        coords_cpu = batch["coords"]  # CPU int tensor [N,1+3]
        feats_cpu = batch["feats"]
        labels_cpu = batch["labels"]  # CPU long [N]

        # Create BEV GT on CPU using (coords, labels)
        coords_np = coords_cpu.numpy().astype(np.int32, copy=False)
        labels_np = labels_cpu.numpy().astype(np.int64, copy=False)

        if coords_np.size == 0:
            continue

        B = int(coords_np[:, 0].max()) + 1  # batch size inside this sparse batch

        bev_targets_cpu: Dict[str, List[np.ndarray]] = {lvl: [] for lvl in levels_req}
        bev_sel_cpu: Dict[str, List[np.ndarray]] = {lvl: [] for lvl in levels_req}

        for bi in range(B):
            m = coords_np[:, 0] == bi
            c_b = coords_np[m, 1:4]  # [Nv_b,3] voxel coords
            y_b = labels_np[m]  # [Nv_b] voxel labels

            for lvl in levels_req:
                lbl, sel = build_bev_labels_and_selected_idx(
                    coords_vox_int32=c_b,
                    labels_vox_i64=y_b,
                    voxel_ignore_index=voxel_ignore,
                    bev_cfg=bev_head_cfg,
                    level=str(lvl),
                    meters_per_voxel=m_per_vox,
                    rng=None,  # deterministic eval
                )
                bev_targets_cpu[lvl].append(lbl)
                bev_sel_cpu[lvl].append(sel)

        bev_targets = {lvl: torch.from_numpy(np.stack(bev_targets_cpu[lvl], axis=0)).long() for lvl in levels_req}
        bev_sel = {lvl: torch.from_numpy(np.stack(bev_sel_cpu[lvl], axis=0)).long() for lvl in levels_req}

        # Move sparse input to GPU
        coords = coords_cpu.to(device, non_blocking=True)
        feats = feats_cpu.to(device, non_blocking=True)
        st = ME.SparseTensor(feats, coordinates=coords, device=device)

        # Move selected_idx to GPU (model/projector may use it)
        bev_sel_dev = {k: v.to(device, non_blocking=True) for k, v in bev_sel.items()}

        # ---------- Forward (force BEV in eval) ----------
        with torch.autocast(device_type="cuda", enabled=amp):
            out = _model_forward_safe(model, st, is_train=False, bev_selected_idx=bev_sel_dev, compute_bev=True)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                out_st, bev_pred = out
            else:
                # If the model doesn't return BEV even when requested, skip
                out_st, bev_pred = out, None

        if bev_pred is None:
            continue

        # ---------- Metrics + visuals ----------
        for lvl, logits in bev_pred.items():
            if lvl not in bev_targets:
                continue
            tgt = bev_targets[lvl]  # [B,H,W] CPU

            pred = logits.argmax(dim=1).detach().cpu()  # [B,H,W]

            cm.update(pred.reshape(-1), tgt.reshape(-1))

            if is_main_process(dist_env):
                p0 = pred[0].numpy().astype(np.int32)
                t0 = tgt[0].numpy().astype(np.int32)

                if save_npz:
                    np.savez_compressed(
                        vis_dir / f"bev_{lvl}_b{batch_i:02d}.npz",
                        pred=p0,
                        target=t0,
                        ignore_index=np.int32(bev_ignore),
                    )
                if save_png:
                    try:
                        from PIL import Image

                        pimg = _colorize_label_map(p0, num_classes=num_classes, ignore_index=bev_ignore)
                        timg = _colorize_label_map(t0, num_classes=num_classes, ignore_index=bev_ignore)
                        Image.fromarray(pimg).save(vis_dir / f"bev_pred_{lvl}_b{batch_i:02d}.png")
                        Image.fromarray(timg).save(vis_dir / f"bev_gt_{lvl}_b{batch_i:02d}.png")
                    except Exception:
                        pass

    # DDP reduce
    if dist_env.enabled:
        cm_mat = cm.mat.to(device=device)
        cm_mat = all_reduce_sum(dist_env, cm_mat)
        cm.mat = cm_mat.cpu()

    res = cm.compute()
    bev_metrics = {
        "bev_miou": res.miou,
        "bev_macro_f1": res.macro_f1,
        "bev_per_class_iou": res.per_class_iou,
        "bev_per_class_f1": res.per_class_f1,
    }

    if is_main_process(dist_env):
        save_json(vis_dir / "bev_metrics.json", bev_metrics)

    return bev_metrics


def build_loss(cfg: Dict[str, Any]):
    loss_cfg = cfg["loss"]
    name = loss_cfg["name"].lower()
    ignore_index = int(cfg["data"]["label_space"]["ignore_index"])

    if name != "focal":
        raise ValueError(f"Unknown loss: {name}")

    # --- focal (existing) ---
    alpha_raw = loss_cfg.get("alpha", None)
    alpha_tensor = None
    if alpha_raw is not None:
        if isinstance(alpha_raw, (list, tuple)):
            num_classes = int(cfg["data"]["label_space"]["num_classes"])
            if len(alpha_raw) != num_classes:
                raise ValueError(f"loss.alpha length {len(alpha_raw)} != num_classes={num_classes}")
            alpha_tensor = torch.tensor(alpha_raw, dtype=torch.float32)
        else:
            raise TypeError(f"loss.alpha must be a list/tuple of floats or null; got {type(alpha_raw)}")

    fl_cfg = FocalLossConfig(
        gamma=float(loss_cfg.get("gamma", 2.0)),
        alpha=alpha_tensor,
        ignore_index=ignore_index,
    )
    focal = FocalLoss(fl_cfg)

    # --- optional lovasz warmup/ramp ---
    lovasz_block = loss_cfg.get("lovasz", None) or {}
    enabled = bool(lovasz_block.get("enabled", False))
    if not enabled:
        return focal

    lw_cfg = LovaszWarmupConfig(
        enabled=True,
        weight=float(lovasz_block.get("weight", 0.5)),
        warmup_epochs=int(lovasz_block.get("warmup_epochs", 10)),
        ramp_epochs=int(lovasz_block.get("ramp_epochs", 10)),
        classes=str(lovasz_block.get("classes", "present")),
    )
    lovasz = LovaszSoftmaxLoss(ignore_index=ignore_index, classes=lw_cfg.classes)

    return FocalLovaszLoss(focal=focal, lovasz=lovasz, cfg=lw_cfg)


def _forward_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    is_train: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    coords = batch["coords"].to(device, non_blocking=True)
    feats = batch["feats"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)

    st = ME.SparseTensor(feats, coordinates=coords, device=device)

    bev_sel = batch.get("bev_selected_idx", None)
    if isinstance(bev_sel, dict):
        bev_sel = {k: v.to(device, non_blocking=True) for k, v in bev_sel.items()}
    elif torch.is_tensor(bev_sel):
        bev_sel = bev_sel.to(device, non_blocking=True)

    bev_als_frames = None
    if "bev_als_center_xy_m" in batch:
        bev_als_frames = {
            "center_xy_m": batch["bev_als_center_xy_m"].to(device, non_blocking=True),
            "height_edges_m": batch["bev_als_height_edges_m"].to(device, non_blocking=True),
        }

    out = _model_forward_safe(
        model,
        st,
        is_train=is_train,
        bev_selected_idx=bev_sel,
        compute_bev=None,
        bev_als_frames=bev_als_frames,
        compute_bev_als=None,
    )

    bev_pred = None
    if isinstance(out, (tuple, list)) and len(out) == 2:
        out, bev_pred = out  # out is SparseTensor
    logits = out.F  # [N, C]
    return logits, labels, bev_pred


def _labels_in_output_order(
    input_coords: torch.Tensor,
    input_labels: torch.Tensor,
    output_coords: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    indices, coverage = match_sparse_coordinates(input_coords, output_coords, require_all=True)
    return input_labels.index_select(0, indices), coverage


def _gradient_l2_norms(model: torch.nn.Module) -> Tuple[float, float]:
    """Return unscaled backbone/BEV-ALS gradient norms for application diagnostics."""

    backbone_sq = 0.0
    aux_sq = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().norm(2).item())
        if "bev_als_head" in name:
            aux_sq += value * value
        else:
            backbone_sq += value * value
    return math.sqrt(backbone_sq), math.sqrt(aux_sq)


def _ocons_train_microstep(
    *,
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    criterion: torch.nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    grad_accum: int,
    cfg: OConsConfig,
    num_classes: int,
    ignore_index: int,
    base_seed: int,
    epoch: int,
    global_step: int,
    rank: int,
    amp: bool,
) -> Dict[str, Any]:
    seed = stable_ocons_seed(base_seed, epoch, global_step, rank, cfg.seed_offset)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    view = make_masked_view(
        batch["coords"],
        batch["feats"],
        batch["labels"],
        cfg,
        num_classes=num_classes,
        generator=generator,
    )

    clean_coords = batch["coords"].to(device, non_blocking=True)
    clean_feats = batch["feats"].to(device, non_blocking=True)
    clean_labels_input = batch["labels"].to(device, non_blocking=True)
    clean_st = ME.SparseTensor(clean_feats, coordinates=clean_coords, device=device)

    with torch.autocast(device_type="cuda", enabled=amp):
        clean_raw = _model_forward_safe(model, clean_st, is_train=True, compute_bev_als=False)
        clean_out = clean_raw[0] if isinstance(clean_raw, (tuple, list)) else clean_raw
        clean_labels, clean_coverage = _labels_in_output_order(clean_coords, clean_labels_input, clean_out.C)
        clean_loss = criterion(clean_out.F, clean_labels)
        clean_term = 0.5 * clean_loss
    scaler.scale(clean_term / float(grad_accum)).backward()
    clean_logits_detached = clean_out.F.detach()
    clean_output_coords = clean_out.C.detach()

    pert_coords = view.coordinates.to(device, non_blocking=True)
    pert_feats = view.features.to(device, non_blocking=True)
    pert_labels_input = view.labels.to(device, non_blocking=True)
    pert_st = ME.SparseTensor(pert_feats, coordinates=pert_coords, device=device)

    with torch.autocast(device_type="cuda", enabled=amp):
        pert_raw = _model_forward_safe(model, pert_st, is_train=True, compute_bev_als=False)
        pert_out = pert_raw[0] if isinstance(pert_raw, (tuple, list)) else pert_raw
        pert_labels, pert_coverage = _labels_in_output_order(pert_coords, pert_labels_input, pert_out.C)
        clean_for_pert_idx, cross_coverage = match_sparse_coordinates(clean_output_coords, pert_out.C, require_all=True)
        clean_aligned = clean_logits_detached.index_select(0, clean_for_pert_idx)
        pert_loss = criterion(pert_out.F, pert_labels)
        cons_loss, cons_diag = ocons_consistency_loss(
            clean_aligned,
            pert_out.F,
            pert_labels,
            cfg,
            num_classes=num_classes,
        )
        cons_weight = ocons_weight_for_epoch(cfg, epoch)
        pert_term = 0.5 * pert_loss + cons_weight * cons_loss
    scaler.scale(pert_term / float(grad_accum)).backward()

    total = clean_term.detach() + 0.5 * pert_loss.detach() + cons_weight * cons_loss.detach()
    valid = clean_labels != int(ignore_index)
    return {
        "total": total,
        "clean_loss": clean_loss.detach(),
        "perturbed_loss": pert_loss.detach(),
        "consistency_loss": cons_loss.detach(),
        "consistency_weight": float(cons_weight),
        "clean_logits": clean_out.F.detach(),
        "clean_labels": clean_labels.detach(),
        "valid_count": int(valid.sum().item()),
        "view_diagnostics": view.diagnostics,
        "consistency_diagnostics": cons_diag,
        "coordinate_match_coverage": min(clean_coverage, pert_coverage, cross_coverage),
    }


def _rotate_xy_np(xyz: np.ndarray, deg: float) -> np.ndarray:
    """Rotate XYZ around +Z axis by deg, returning a new array."""
    if deg % 360 == 0:
        return xyz
    theta = np.deg2rad(deg)
    c, s = np.cos(theta), np.sin(theta)
    xy = xyz[:, :2]
    R = np.array([[c, -s], [s, c]], dtype=xyz.dtype)
    xy_r = xy @ R.T
    out = xyz.copy()
    out[:, 0] = xy_r[:, 0]
    out[:, 1] = xy_r[:, 1]
    return out


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    dist_env: DistEnv,
    loader: DataLoader,
    criterion: torch.nn.Module,
    num_classes: int,
    ignore_index: int,
    device: torch.device,
    amp: bool,
) -> Dict[str, Any]:
    model.eval()
    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)

    total_loss = 0.0
    total_n = 0

    for batch in loader:
        with torch.autocast(device_type="cuda", enabled=amp):
            logits, labels, _bev = _forward_batch(model, batch, device, is_train=False)
            loss = criterion(logits, labels)

        # metrics
        preds = logits.argmax(dim=1)
        cm.update(preds, labels)

        # approximate loss weighting by number of valid voxels
        valid = labels != ignore_index
        n = int(valid.sum().item())
        total_loss += float(loss.item()) * max(1, n)
        total_n += max(1, n)

    # Aggregate across ranks
    if dist_env.enabled:
        cm_mat = cm.mat.to(device=device)
        cm_mat = all_reduce_sum(dist_env, cm_mat)
        cm.mat = cm_mat.cpu()

        tl = torch.tensor([total_loss], dtype=torch.float64, device=device)
        tn = torch.tensor([total_n], dtype=torch.float64, device=device)
        tl = all_reduce_sum(dist_env, tl)
        tn = all_reduce_sum(dist_env, tn)
        total_loss = float(tl.item())
        total_n = int(tn.item())

    res = cm.compute()
    return {
        "loss": total_loss / max(1, total_n),
        "miou": res.miou,
        "macro_f1": res.macro_f1,
        "per_class_iou": res.per_class_iou,
        "per_class_f1": res.per_class_f1,
        "per_class_precision": res.per_class_precision,
        "per_class_recall": res.per_class_recall,
        "miou_valid": res.miou_valid,
        "confusion_matrix": cm.mat.tolist(),
    }


def _make_sliding_origins(max_xy_vox: int, win_vox: int, stride_vox: int) -> List[int]:
    if max_xy_vox <= win_vox:
        return [0]
    xs = list(range(0, max_xy_vox - win_vox + 1, stride_vox))
    last = max_xy_vox - win_vox
    if xs[-1] != last:
        xs.append(last)
    return xs


@torch.no_grad()
def evaluate_voxel_windowed(
    *,
    model: torch.nn.Module,
    dist_env: DistEnv,
    dataset_obj,
    criterion_cpu: torch.nn.Module,
    num_classes: int,
    ignore_index: int,
    device: torch.device,
    amp: bool,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    OOM-safe voxel evaluation for huge DALES tiles:
      - load raw points
      - voxelize full tile ONCE (cpu), INCLUDING voxel labels (label_pool)
      - run sparse UNet in sliding windows over voxels (gpu-safe)
      - aggregate voxel logits over overlapping windows
      - compute voxel IoU/F1 directly on voxels (no inverse_map projection)
    """
    model.eval()
    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)

    eval_cfg = cfg.get("eval", {}) or {}
    win_cfg = eval_cfg.get("window", {}) or {}

    win_m = float(win_cfg.get("size_xy_m", 50.0))
    stride_m = float(win_cfg.get("stride_xy_m", win_m))
    accum_dtype = str(win_cfg.get("accum_dtype", "float32")).lower()
    if accum_dtype not in ("float32", "float16"):
        raise ValueError("eval.window.accum_dtype must be float32 or float16")

    agg = str(win_cfg.get("aggregation", "mean_logits")).lower()
    if agg not in ("mean_logits", "sum_logits"):
        raise ValueError("eval.window.aggregation must be mean_logits or sum_logits")

    # reuse the same flag for convenience (loss over voxels here)
    compute_voxel_loss = bool(win_cfg.get("compute_point_loss", False))

    total_loss = 0.0
    total_n = 0

    n_tiles = len(dataset_obj)
    for ti in range(n_tiles):
        if dist_env.enabled and (ti % dist_env.world_size) != dist_env.rank:
            continue

        raw = dataset_obj.get_raw(ti)
        xyz = np.asarray(raw["xyz"], dtype=np.float64)

        patch_cfg = dataset_obj.patch_cfg

        if bool(patch_cfg.make_local_coords):
            xyz = xyz - xyz.min(axis=0, keepdims=True)

        # --- labels per POINT in train-id space ---
        if hasattr(dataset_obj, "label_lut") and dataset_obj.label_lut is not None:
            # DALES path (native -> train LUT)
            y_native = raw.get("native_labels", None)
            if y_native is None:
                raise KeyError("DALES raw must include native_labels")
            y_safe = np.clip(y_native.astype(np.int64, copy=False), 0, 255)
            y_pts = dataset_obj.label_lut[y_safe].astype(np.int64, copy=False)
        else:
            # ECLAIR path
            y_native = raw.get("native_labels", None)
            if y_native is None:
                y_native = raw.get("labels", None)
            if y_native is None:
                raise KeyError("ECLAIR raw must include native_labels or labels")
            undefined_id = int(getattr(dataset_obj, "undefined_id", 0))
            y_native = y_native.astype(np.int64, copy=False)
            y_pts = y_native - 1
            y_pts[y_native == undefined_id] = ignore_index

        # --- features per POINT ---
        feat_cfg = dataset_obj.feat_cfg

        xyz_norm = (xyz / float(patch_cfg.coord_norm_factor)).astype(np.float32, copy=False)
        feats_p = build_features(
            xyz_local=xyz_norm,
            intensity=(raw.get("intensity", None)),
            return_number=raw.get("return_number", None),
            number_of_returns=raw.get("number_of_returns", None),
            rgb=(raw.get("rgb", None) if feat_cfg.use_rgb else None),
            cfg=feat_cfg,
        )  # [Np, Cin]

        # --- full-tile quantization (CPU) INCLUDING voxel labels ---
        q = np.floor(xyz_norm / float(patch_cfg.voxel_size)).astype(np.int32, copy=False)
        voxel_cfg = VoxelizationConfig.from_cfg(cfg["data"])
        vx = voxelize_from_q(
            q_int32=q,
            feats_p_f32=feats_p.astype(np.float32, copy=False),
            labels_p_i64=y_pts.astype(np.int64, copy=False),
            ignore_index=int(ignore_index),
            cfg=voxel_cfg,
            rng=None,
            return_maps=False,
            num_classes_hint=num_classes,
        )

        coords_u = vx["coords_u"]  # [Nv,3] int32
        feats_v = vx["feats_u"]  # [Nv,Cin] float32
        labels_v = vx["labels_u"]  # [Nv] int64
        if labels_v is None:
            raise RuntimeError("voxelize_from_q did not return labels_u; labels_p_i64 was provided but labels_u is None.")
        nv = int(coords_u.shape[0])
        if nv == 0:
            continue

        # --- window schedule in VOXEL units ---
        m_per_vox = float(patch_cfg.voxel_size) * float(patch_cfg.coord_norm_factor)
        win_vox = max(1, int(np.ceil(win_m / m_per_vox)))
        stride_vox = max(1, int(np.ceil(stride_m / m_per_vox)))

        x = coords_u[:, 0]
        y = coords_u[:, 1]
        xmax = int(x.max())
        ymax = int(y.max())
        xs = _make_sliding_origins(xmax + 1, win_vox, stride_vox)
        ys = _make_sliding_origins(ymax + 1, win_vox, stride_vox)

        # --- accumulate voxel logits across windows ---
        sum_dtype = np.float16 if accum_dtype == "float16" else np.float32
        voxel_logits_sum = np.zeros((nv, num_classes), dtype=sum_dtype)
        voxel_counts = np.zeros((nv,), dtype=np.uint16)

        for x0 in xs:
            x1 = x0 + win_vox
            mx = (x >= x0) & (x < x1)
            if not bool(mx.any()):
                continue
            for y0 in ys:
                y1 = y0 + win_vox
                sel = np.where(mx & (y >= y0) & (y < y1))[0]
                if sel.size == 0:
                    continue

                coords_sub = coords_u[sel].astype(np.int32, copy=False)
                coords_sub = coords_sub - coords_sub.min(axis=0, keepdims=True)

                coords_sub_t = torch.from_numpy(np.ascontiguousarray(coords_sub, dtype=np.int32)).int()
                feats_sub_t = torch.from_numpy(np.ascontiguousarray(feats_v[sel], dtype=np.float32)).float()

                coords_b = ME.utils.batched_coordinates([coords_sub_t], dtype=torch.int32)
                st = ME.SparseTensor(
                    feats_sub_t.to(device, non_blocking=True),
                    coordinates=coords_b.to(device, non_blocking=True),
                    device=device,
                )

                with torch.autocast(device_type="cuda", enabled=amp):
                    try:
                        out = model(st, is_train=False, compute_bev=False)
                    except TypeError:
                        out = model(st)

                if isinstance(out, (tuple, list)) and len(out) == 2:
                    out_st = out[0]
                else:
                    out_st = out

                logits_sub_np = out_st.F.detach().float().cpu().numpy()
                voxel_logits_sum[sel] += logits_sub_np.astype(sum_dtype, copy=False)
                voxel_counts[sel] += 1

        # finalize voxel logits
        missed = np.where(voxel_counts == 0)[0]

        if missed.size > 0:
            raise RuntimeError(f"Evaluation window scheduler missed " f"{missed.size}/{nv} voxels.")
        counts = voxel_counts.astype(np.float32)
        counts[counts == 0] = 1.0
        if agg == "mean_logits":
            voxel_logits = (voxel_logits_sum.astype(np.float32) / counts[:, None]).astype(np.float32, copy=False)
        else:
            voxel_logits = voxel_logits_sum.astype(np.float32, copy=False)

        # metrics (voxel-wise)
        voxel_pred = voxel_logits.argmax(axis=1).astype(np.int64, copy=False)
        preds_t = torch.from_numpy(voxel_pred)
        labels_t = torch.from_numpy(labels_v.astype(np.int64, copy=False))
        cm.update(preds_t, labels_t)

        if compute_voxel_loss:
            # compute loss on CPU to avoid GPU memory blowups
            logits_t = torch.from_numpy(voxel_logits)
            loss = criterion_cpu(logits_t, labels_t)
            valid = labels_t != ignore_index
            n = int(valid.sum().item())
            total_loss += float(loss.item()) * max(1, n)
            total_n += max(1, n)
        else:
            total_n += max(1, int((labels_t != ignore_index).sum().item()))

    # DDP reduce
    if dist_env.enabled:
        cm_mat = cm.mat.to(device=device)
        cm_mat = all_reduce_sum(dist_env, cm_mat)
        cm.mat = cm_mat.cpu()

        tl = torch.tensor([total_loss], dtype=torch.float64, device=device)
        tn = torch.tensor([total_n], dtype=torch.float64, device=device)
        tl = all_reduce_sum(dist_env, tl)
        tn = all_reduce_sum(dist_env, tn)
        total_loss = float(tl.item())
        total_n = int(tn.item())

    res = cm.compute()
    loss_out = (total_loss / max(1, total_n)) if compute_voxel_loss else float("nan")
    return {
        "loss": loss_out,
        "miou": res.miou,
        "macro_f1": res.macro_f1,
        "per_class_iou": res.per_class_iou,
        "per_class_f1": res.per_class_f1,
        "per_class_precision": res.per_class_precision,
        "per_class_recall": res.per_class_recall,
        "miou_valid": res.miou_valid,
        "confusion_matrix": cm.mat.tolist(),
    }


@torch.no_grad()
def evaluate_pointwise(
    *,
    model: torch.nn.Module,
    dist_env: DistEnv,
    dataset_obj,
    criterion_cpu: torch.nn.Module,
    num_classes: int,
    ignore_index: int,
    device: torch.device,
    amp: bool,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Strict point-wise evaluation with optional TTA rotations:
      - load raw points
      - (optional) rotate XY + shift to non-negative local frame
      - voxelize full tile (cpu)
      - run sparse UNet in sliding windows over voxels (gpu-safe)
      - aggregate voxel logits over overlapping windows
      - project voxel logits -> points via inverse_map
      - average point logits across TTA rotations
      - compute point IoU/F1
    """
    model.eval()
    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)

    eval_cfg = cfg.get("eval", {}) or {}
    win_cfg = eval_cfg.get("window", {}) or {}

    # --- TTA config (dict-safe) ---
    tta_cfg = eval_cfg.get("tta", {}) or {}
    tta_enabled = bool(tta_cfg.get("enabled", False))
    rotations_deg = [0.0]
    if tta_enabled:
        rotations_deg = tta_cfg.get("rotations_deg", [0.0])
        if isinstance(rotations_deg, (int, float)):
            rotations_deg = [float(rotations_deg)]
        else:
            rotations_deg = [float(x) for x in list(rotations_deg)]
        if len(rotations_deg) == 0:
            rotations_deg = [0.0]

    num_tta = len(rotations_deg)

    win_m = float(win_cfg.get("size_xy_m", 50.0))
    stride_m = float(win_cfg.get("stride_xy_m", win_m))
    # NOTE: do NOT drop small windows in eval; only skip empty
    # (otherwise some points might never get predicted).
    accum_dtype = str(win_cfg.get("accum_dtype", "float32")).lower()
    if accum_dtype not in ("float32", "float16"):
        raise ValueError("eval.window.accum_dtype must be float32 or float16")

    agg = str(win_cfg.get("aggregation", "mean_logits")).lower()
    if agg not in ("mean_logits", "sum_logits"):
        raise ValueError("eval.window.aggregation must be mean_logits or sum_logits")

    compute_point_loss = bool(win_cfg.get("compute_point_loss", False))

    total_loss = 0.0
    total_n = 0

    voxel_cfg = VoxelizationConfig.from_cfg(cfg["data"])

    n_tiles = len(dataset_obj)
    for ti in range(n_tiles):
        if dist_env.enabled and (ti % dist_env.world_size) != dist_env.rank:
            continue

        raw = dataset_obj.get_raw(ti)
        xyz_base = raw["xyz"].astype(np.float64, copy=False)
        Np = int(xyz_base.shape[0])

        # --- labels per POINT in train-id space (computed once) ---
        if hasattr(dataset_obj, "label_lut") and dataset_obj.label_lut is not None:
            y_native = raw.get("native_labels", None)
            if y_native is None:
                raise KeyError("DALES raw must include native_labels")
            y_safe = np.clip(y_native.astype(np.int64, copy=False), 0, 255)
            y_pts = dataset_obj.label_lut[y_safe].astype(np.int64, copy=False)
        else:
            # ECLAIR path (undefined -> ignore, else 1..K -> 0..K-1)
            y_native = raw.get("native_labels", None)
            if y_native is None:
                # some readers call it "labels"
                y_native = raw.get("labels", None)
            if y_native is None:
                raise KeyError("ECLAIR raw must include native_labels or labels")

            undefined_id = int(getattr(dataset_obj, "undefined_id", 0))
            y_native = y_native.astype(np.int64, copy=False)
            y_pts = y_native - 1
            y_pts[y_native == undefined_id] = ignore_index

        # --- build per-point features (no aug in eval) ---
        patch_cfg = dataset_obj.patch_cfg
        feat_cfg = dataset_obj.feat_cfg

        intensity = raw.get("intensity", None)
        return_number = raw.get("return_number", None)
        number_of_returns = raw.get("number_of_returns", None)
        rgb = raw.get("rgb", None) if getattr(feat_cfg, "use_rgb", False) else None

        # --- TTA accumulation on POINT logits ---
        pt_logits_sum = np.zeros((Np, num_classes), dtype=np.float32)

        for ri, rot_deg in enumerate(rotations_deg):
            xyz_r = xyz_base

            # rotate about XY centroid (pure rigid transform)
            if rot_deg % 360 != 0:
                ctr = xyz_r[:, :2].mean(axis=0, keepdims=True)
                tmp = xyz_r.copy()
                tmp[:, :2] -= ctr
                tmp = _rotate_xy_np(tmp, rot_deg)
                tmp[:, :2] += ctr
                xyz_r = tmp

            # shift to non-negative local frame so window masks don't drop negatives
            xyz_r = xyz_r - xyz_r.min(axis=0, keepdims=True)

            # --- build per-point features (no aug in eval) ---
            xyz_norm = (xyz_r / float(patch_cfg.coord_norm_factor)).astype(np.float32, copy=False)
            feats_p = build_features(
                xyz_local=xyz_norm,
                intensity=intensity,
                return_number=return_number,
                number_of_returns=number_of_returns,
                rgb=rgb,
                cfg=feat_cfg,
            )

            # --- full-tile quantization + inverse_map (CPU) ---
            q = np.floor(xyz_norm / float(patch_cfg.voxel_size)).astype(np.int32, copy=False)

            # deterministic rng (only matters if voxelization uses random pooling)
            rng = np.random.default_rng(seed=(ti * 1009 + ri * 9176 + 1337) & 0x7FFFFFFF)

            vx = voxelize_from_q(
                q_int32=q.astype(np.int32, copy=False),
                feats_p_f32=feats_p.astype(np.float32, copy=False),
                labels_p_i64=None,
                ignore_index=int(ignore_index),
                cfg=voxel_cfg,
                rng=rng,
                return_maps=True,
                num_classes_hint=None,
            )

            coords_u = vx["coords_u"]
            feats_v = vx["feats_u"]
            inverse_map = vx["inverse_map"]
            nv = int(coords_u.shape[0])
            if nv == 0:
                continue

            # --- window schedule in VOXEL units ---
            m_per_vox = float(patch_cfg.voxel_size) * float(patch_cfg.coord_norm_factor)
            win_vox = max(1, int(np.ceil(win_m / m_per_vox)))
            stride_vox = max(1, int(np.ceil(stride_m / m_per_vox)))

            x = coords_u[:, 0]
            y = coords_u[:, 1]
            xmax = int(x.max()) if nv > 0 else 0
            ymax = int(y.max()) if nv > 0 else 0
            xs = _make_sliding_origins(xmax + 1, win_vox, stride_vox)
            ys = _make_sliding_origins(ymax + 1, win_vox, stride_vox)

            # --- accumulate voxel logits across windows ---
            sum_dtype = np.float16 if accum_dtype == "float16" else np.float32
            voxel_logits_sum = np.zeros((nv, num_classes), dtype=sum_dtype)
            voxel_counts = np.zeros((nv,), dtype=np.uint16)

            for x0 in xs:
                x1 = x0 + win_vox
                mx = (x >= x0) & (x < x1)
                if not bool(mx.any()):
                    continue
                for y0 in ys:
                    y1 = y0 + win_vox
                    sel = np.where(mx & (y >= y0) & (y < y1))[0]
                    if sel.size == 0:
                        continue

                    coords_sub = coords_u[sel].astype(np.int32, copy=False)
                    coords_sub = coords_sub - coords_sub.min(axis=0, keepdims=True)

                    coords_sub_t = torch.from_numpy(np.ascontiguousarray(coords_sub, dtype=np.int32)).int()
                    feats_sub_t = torch.from_numpy(np.ascontiguousarray(feats_v[sel], dtype=np.float32)).float()

                    coords_b = ME.utils.batched_coordinates([coords_sub_t], dtype=torch.int32)
                    st = ME.SparseTensor(
                        feats_sub_t.to(device, non_blocking=True),
                        coordinates=coords_b.to(device, non_blocking=True),
                        device=device,
                    )

                    with torch.autocast(device_type="cuda", enabled=amp):
                        try:
                            out = model(st, is_train=False, compute_bev=False)
                        except TypeError:
                            out = model(st)

                    out_st = out[0] if isinstance(out, (tuple, list)) and len(out) == 2 else out
                    cin = coords_b.cpu().numpy()
                    cout = out_st.C.detach().cpu().numpy()

                    if not np.array_equal(cin, cout):
                        raise RuntimeError("Minkowski output coordinate order differs from input.")
                    logits_sub_np = out_st.F.detach().float().cpu().numpy()

                    voxel_logits_sum[sel] += logits_sub_np.astype(sum_dtype, copy=False)
                    voxel_counts[sel] += 1

            missed = np.where(voxel_counts == 0)[0]

            if missed.size > 0:
                raise RuntimeError(f"Evaluation window scheduler missed " f"{missed.size}/{nv} voxels.")
            counts = voxel_counts.astype(np.float32)
            counts[counts == 0] = 1.0

            if agg == "mean_logits":
                voxel_logits = (voxel_logits_sum.astype(np.float32) / counts[:, None]).astype(np.float32, copy=False)
            else:
                voxel_logits = voxel_logits_sum.astype(np.float32, copy=False)

            pt_logits = voxel_logits[inverse_map]  # [Np, C]
            pt_logits_sum += pt_logits.astype(np.float32, copy=False)

        # --- average across TTA rotations ---
        pt_logits_avg = pt_logits_sum / float(num_tta)
        pt_pred = pt_logits_avg.argmax(axis=1).astype(np.int64, copy=False)

        preds_t = torch.from_numpy(pt_pred)
        labels_t = torch.from_numpy(y_pts.astype(np.int64, copy=False))
        cm.update(preds_t, labels_t)

        # optional pointwise loss on averaged logits
        valid = labels_t != ignore_index
        n = int(valid.sum().item())
        total_n += max(1, n)

        if compute_point_loss:
            loss = criterion_cpu(torch.from_numpy(pt_logits_avg), labels_t)
            total_loss += float(loss.item()) * max(1, n)

    # DDP reduce
    if dist_env.enabled:
        cm_mat = cm.mat.to(device=device)
        cm_mat = all_reduce_sum(dist_env, cm_mat)
        cm.mat = cm_mat.cpu()

        tl = torch.tensor([total_loss], dtype=torch.float64, device=device)
        tn = torch.tensor([total_n], dtype=torch.float64, device=device)
        tl = all_reduce_sum(dist_env, tl)
        tn = all_reduce_sum(dist_env, tn)
        total_loss = float(tl.item())
        total_n = int(tn.item())

    res = cm.compute()
    loss_out = (total_loss / max(1, total_n)) if compute_point_loss else float("nan")
    return {
        "loss": loss_out,
        "miou": res.miou,
        "macro_f1": res.macro_f1,
        "per_class_iou": res.per_class_iou,
        "per_class_f1": res.per_class_f1,
        "per_class_precision": res.per_class_precision,
        "per_class_recall": res.per_class_recall,
        "miou_valid": res.miou_valid,
        "confusion_matrix": cm.mat.tolist(),
    }


def train(cfg_path: str):
    cfg = load_yaml(cfg_path)
    method_contract: MethodContract = validate_method_contract(cfg)
    ocons_cfg = OConsConfig.from_cfg(cfg)
    bev_als_cfg = ALSBEVConfig.from_cfg(cfg)

    run = cfg["run"]
    out_dir = Path(run["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dist_env = init_distributed()  # <-- move this UP before any writes

    dataset = str(cfg.get("data", {}).get("dataset", "eclair")).lower()
    eclair_run_cache_dir: Optional[Path] = None
    eclair_run_cache_base: Optional[Path] = None

    run_cache_enabled = bool(cfg.get("data", {}).get("run_cache_enabled", True))
    if dataset == "eclair" and run_cache_enabled:
        key = _eclair_run_cache_key(cfg)
        eclair_run_cache_base = out_dir / "_run_cache" / f"eclair_{key}"
        rank_tag = f"rank{dist_env.rank}" if dist_env.enabled else "single"
        eclair_run_cache_dir = eclair_run_cache_base / rank_tag
        if is_main_process(dist_env):
            eclair_run_cache_dir.mkdir(parents=True, exist_ok=True)
        # each rank creates its own dir (avoid shared-writer issues)
        eclair_run_cache_dir.mkdir(parents=True, exist_ok=True)
        if dist_env.enabled:
            torch.distributed.barrier()

    # Only rank0 writes files
    if is_main_process(dist_env):
        save_json(out_dir / "config_resolved.json", cfg)
        save_json(out_dir / "method_contract.json", method_contract.to_dict())
        checkpoint_dir = out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        removed_tmp = _cleanup_checkpoint_temp_files(checkpoint_dir)
        keep_snapshots = bool(run.get("keep_epoch_snapshots", False))
        keep_last_k = int(run.get("keep_last_k_epoch_snapshots", 0))
        removed_epochs = _prune_epoch_snapshots(
            checkpoint_dir,
            keep_last_k if keep_snapshots else 0,
        )
        if removed_tmp or removed_epochs:
            print(
                f"[storage] startup cleanup temp={removed_tmp} epoch_snapshots={removed_epochs}",
                flush=True,
            )

    if dist_env.enabled:
        torch.distributed.barrier()  # ensure dirs exist before others proceed

    set_seed(int(run["seed"]) + (dist_env.rank if dist_env.enabled else 0))
    device = torch.device(f"cuda:{dist_env.local_rank}" if torch.cuda.is_available() else "cpu")

    amp = bool(run.get("amp", True))

    train_loader, val_loader, test_loader = build_dataloaders(cfg, dist_env, eclair_run_cache_dir=eclair_run_cache_dir)

    if is_main_process(dist_env):
        ds = train_loader.dataset
        # Under DistributedSampler, dataset may be the original EclairTiles.
        summary = getattr(ds, "sampling_summary", None)
        if summary is not None:
            save_json(out_dir / "eclair_sampling_summary.json", summary)
            print(f"[ECLAIR sampling summary] {json.dumps(summary, indent=2)}", flush=True)

    model_cfg = cfg["model"]
    model = build_model(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        D=int(model_cfg.get("D", 3)),
        cfg=cfg,
    ).to(device)

    if dist_env.enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_env.local_rank],
            output_device=dist_env.local_rank,
            find_unused_parameters=False,
        )

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    criterion = build_loss(cfg).to(device)
    criterion_cpu = build_loss(cfg).cpu()

    # ensure same schedule behavior for both
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(0)
    if hasattr(criterion_cpu, "set_epoch"):
        criterion_cpu.set_epoch(0)

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    num_classes = int(cfg["data"]["label_space"]["num_classes"])
    ignore_index = int(cfg["data"]["label_space"]["ignore_index"])
    grad_accum = int(cfg["data"].get("grad_accum_steps", 1))
    epochs = int(cfg.get("epochs", 80)) if "epochs" in cfg else 80

    # Logger fields
    dataset = str(cfg["data"].get("dataset", "eclair")).lower()
    ls = cfg["data"]["label_space"]
    num_classes = int(ls["num_classes"])

    # Prefer explicit class_names from config; else use ECLAIR defaults;
    # else fallback generic.
    class_names: List[str]
    if "class_names" in ls and ls["class_names"] is not None:
        class_names = list(ls["class_names"])
    elif dataset == "eclair":
        class_names = list(ECLAIR_CLASS_NAMES_11)
    else:
        class_names = [f"class_{i}" for i in range(num_classes)]

    if len(class_names) != num_classes:
        raise ValueError(
            f"len(class_names)={len(class_names)} != num_classes={num_classes}. "
            f"Fix data.label_space.class_names or num_classes."
        )

    fields = [
        "epoch",
        "method",
        "debug_limited_run",
        "lr",
        "train_loss",
        "train_seg_clean_loss",
        "train_seg_perturbed_loss",
        "train_ocons_consistency_loss",
        "train_bev_als_loss",
        "aux_weight",
        "val_loss",
        "val_miou",
        "val_macro_f1",
        "time_epoch_s",
        "eval_time_s",
        "epoch_wall_time_s",
        "world_size",
        "train_batches_per_rank",
        "best_val_miou",
        "checkpoint_dir_gb",
        "run_dir_gb",
        "run_cache_gb",
        "gpu_peak_allocated_mib",
        "gpu_peak_reserved_mib",
        "backbone_grad_norm",
        "method_aux_grad_norm",
    ]

    # BEV scalar metric (row currently writes it; add column so it actually appears)
    fields.append("bev_miou")

    if ocons_cfg.enabled:
        fields += [
            "ocons_actual_mask_fraction",
            "ocons_active_jaccard",
            "ocons_coordinate_match_coverage",
            "ocons_protection_constrained_rate",
            "ocons_protected_disappearances",
            "ocons_micro_kl",
            "ocons_macro_kl",
            "ocons_agreement",
            "ocons_clean_confidence",
            "ocons_perturbed_confidence",
        ]
    if bev_als_cfg.enabled:
        fields += [
            "bev_als_bce",
            "bev_als_dice_loss",
            "bev_als_precision",
            "bev_als_recall",
            "bev_als_f1",
            "bev_als_in_bounds_fraction",
            "bev_als_multi_label_fraction",
            "bev_als_empty_slice_rate",
            "bev_als_feature_in_bounds_fraction",
        ]

    # DALES crop debug metrics (blank for non-DALES / non-crops)
    is_dales = dataset == "dales"
    sampling_mode = str(cfg.get("data", {}).get("sampling", {}).get("mode", "tiles")).lower()
    is_dales_crops = is_dales and (sampling_mode == "crops")
    if is_dales_crops:
        fields += [
            "train_vox_mean",
            "train_rare_frac_mean",
            "train_crop_size_mean",
            "train_shrink_steps_mean",
            "train_fallback_rate",
            "train_rare_center_rate",
            "train_resample_attempt_mean",
        ]

    mix_enabled = bool((cfg.get("data", {}).get("mix3d", {}) or {}).get("enabled", False))
    if mix_enabled:
        fields += [
            "mix_rate",
            "mix_skip_rate",
            "mix_host_points_mean",
            "mix_host_removed_mean",
            "mix_donor_inserted_mean",
            "mix_output_points_mean",
            "mix_replacement_side_m_mean",
            "mix_guard_band_m_mean",
            "mix_height_shift_m_mean",
            "mix_cross_provenance_voxels",
            "mix_output_budget_skip_rate",
            "mix_probability_skip_rate",
            "mix_no_donor_skip_rate",
            "mix_invalid_region_skip_rate",
            "mix_collision_skip_rate",
            "mix_replacement_side_m_std",
            "mix_height_shift_m_std",
            "mix_abs_height_shift_m_mean",
            "mix_output_to_host_points_ratio_mean",
            "mix_output_to_host_points_ratio_std",
            "mix_donor_unique_count",
            "mix_donor_selection_entropy",
            "mix_donor_max_share",
        ]

    # per-class columns
    for name in class_names:
        fields.append(f"val_iou_{name.replace(' ', '_').replace('.', '')}")
    for name in class_names:
        fields.append(f"val_f1_{name.replace(' ', '_').replace('.', '')}")
    for name in class_names:
        fields.append(f"val_precision_{name.replace(' ', '_').replace('.', '')}")
    for name in class_names:
        fields.append(f"val_recall_{name.replace(' ', '_').replace('.', '')}")

    if is_main_process(dist_env):
        logger = CSVLogger(out_dir / "metrics.csv", fields)
    else:

        class _NoOp:
            def log(self, *_args, **_kwargs):
                return

        logger = _NoOp()

    if mix_enabled and is_main_process(dist_env):
        mix_class_logger = CSVLogger(
            out_dir / "mix_class_diagnostics.csv",
            [
                "epoch",
                "class_id",
                "class_name",
                "host_removed_points",
                "donor_inserted_points",
                "mixed_output_points",
                "donor_presence_samples",
            ],
        )
    else:
        mix_class_logger = None
    mix_context_history: List[Dict[str, Any]] = []

    if ocons_cfg.enabled and is_main_process(dist_env):
        ocons_class_logger = CSVLogger(
            out_dir / "ocons_class_diagnostics.csv",
            [
                "epoch", "class_id", "class_name", "clean_voxels",
                "perturbed_voxels", "retention", "kl", "kl_support",
            ],
        )
    else:
        ocons_class_logger = None

    if bev_als_cfg.enabled and is_main_process(dist_env):
        bev_als_class_logger = CSVLogger(
            out_dir / "bev_als_slice_class_diagnostics.csv",
            [
                "epoch", "height_slice", "class_id", "class_name",
                "input_voxels", "in_bounds_voxels", "bounds_retention",
                "positive_support", "tp", "fp", "fn", "precision", "recall", "f1",
            ],
        )
    else:
        bev_als_class_logger = None

    best_val_miou = -1.0
    global_step = 0
    start_epoch = 1

    resume_from = run.get("resume_from", None)
    if resume_from:
        resume_path = Path(str(resume_from)).expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")

        checkpoint = torch.load(resume_path, map_location="cpu")
        required = {"epoch", "model_state", "optimizer_state"}
        missing = sorted(required.difference(checkpoint.keys()))
        if missing:
            raise KeyError(f"Resume checkpoint is missing required keys {missing}: {resume_path}")
        checkpoint_cfg = checkpoint.get("cfg", {}) or {}
        checkpoint_contract = validate_method_contract(checkpoint_cfg)
        if checkpoint_contract.name != method_contract.name:
            raise ValueError(
                "Refusing to resume across methods: "
                f"checkpoint={checkpoint_contract.name!r}, current={method_contract.name!r}."
            )

        unwrap_model(model).load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        # Optimizer checkpoints were loaded on CPU; move tensor states to the
        # current rank's device before the first optimizer step.
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device=device, non_blocking=True)

        scheduler_state = checkpoint.get("scheduler_state", None)
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        scaler_state = checkpoint.get("scaler_state", None)
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)

        completed_epoch = int(checkpoint["epoch"])
        start_epoch = completed_epoch + 1
        best_val_miou = float(checkpoint.get("best_val_miou", -1.0))
        global_step = int(checkpoint.get("global_step", 0))

        if start_epoch > epochs:
            raise ValueError(
                f"Checkpoint already completed epoch {completed_epoch}, but epochs={epochs}. "
                "Increase epochs or remove run.resume_from."
            )
        if is_main_process(dist_env):
            print(
                f"[resume] loaded={resume_path} completed_epoch={completed_epoch} "
                f"start_epoch={start_epoch} best_val_miou={best_val_miou:.6f}",
                flush=True,
            )
        if dist_env.enabled:
            torch.distributed.barrier()

    track_train_cm = bool(run.get("track_train_cm", False))
    ds_tr, ds_va, ds_te = train_loader.dataset, val_loader.dataset, test_loader.dataset

    def _ds_items(ds):
        # DALES exposes `files: List[Path]`; ECLAIR exposes `names: List[str]`
        if hasattr(ds, "files"):
            return list(ds.files)
        if hasattr(ds, "names"):
            return list(ds.names)
        return None

    tr_items = _ds_items(ds_tr)
    va_items = _ds_items(ds_va)
    te_items = _ds_items(ds_te)

    print("train tiles:", None if tr_items is None else len(tr_items))
    print("val tiles:", None if va_items is None else len(va_items))
    print("test tiles:", None if te_items is None else len(te_items))

    if va_items is not None:
        head = va_items[:10]
        # Paths -> show .name; strings -> show directly
        print("val files head:", [getattr(p, "name", str(p)) for p in head])

    for epoch in range(start_epoch, epochs + 1):

        epoch_wall_start = time.perf_counter()

        model.train()

        if dist_env.enabled:
            # ensures shuffling differs each epoch
            train_loader.sampler.set_epoch(epoch)

        # Ensure crop RNG changes per epoch (and is stable)
        ds = train_loader.dataset
        if hasattr(ds, "set_epoch"):
            # Optionally mix in rank so different ranks don't accidentally correlate
            ds.set_epoch(epoch + (dist_env.rank * 100000 if dist_env.enabled else 0))

        # Update loss warmup schedule
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)
        if hasattr(criterion_cpu, "set_epoch"):
            criterion_cpu.set_epoch(epoch)

        # Optional: log lovasz weight
        if is_main_process(dist_env) and hasattr(criterion, "current_weight"):
            print(f"[loss] epoch={epoch:03d} lovasz_w={criterion.current_weight():.4f}", flush=True)

        if torch.cuda.is_available():
            if is_main_process(dist_env):
                peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
                peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
                print(f"[gpu] previous_peak_alloc={peak_alloc:.2f} GB previous_peak_reserved={peak_reserved:.2f} GB", flush=True)
            torch.cuda.reset_peak_memory_stats(device)

        cm_train = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index) if track_train_cm else None
        train_loss_sum = 0.0
        train_n_sum = 0
        processed_steps = 0
        component_steps = 0.0
        seg_clean_sum = 0.0
        seg_pert_sum = 0.0
        ocons_loss_sum = 0.0
        bev_als_loss_sum = 0.0
        aux_weight_sum = 0.0
        grad_measurements = 0.0
        backbone_grad_norm_sum = 0.0
        aux_grad_norm_sum = 0.0
        ocons_scalar_sum = torch.zeros(11, dtype=torch.float64)
        ocons_class_before = torch.zeros(num_classes, dtype=torch.float64)
        ocons_class_after = torch.zeros(num_classes, dtype=torch.float64)
        ocons_class_kl_sum = torch.zeros(num_classes, dtype=torch.float64)
        ocons_class_kl_support = torch.zeros(num_classes, dtype=torch.float64)
        bev_als_scalar_sum = torch.zeros(9, dtype=torch.float64)
        bev_als_tp = torch.zeros((bev_als_cfg.height_slices, num_classes), dtype=torch.float64)
        bev_als_fp = torch.zeros_like(bev_als_tp)
        bev_als_fn = torch.zeros_like(bev_als_tp)
        bev_als_positive = torch.zeros_like(bev_als_tp)
        bev_als_class_before = torch.zeros(num_classes, dtype=torch.float64)
        bev_als_class_in_bounds = torch.zeros(num_classes, dtype=torch.float64)
        # --- NEW: crop debug accumulators (DALES crop-mode only) ---
        dbg_n = 0
        dbg_vox_sum = 0.0
        dbg_rare_sum = 0.0
        dbg_crop_size_sum = 0.0
        dbg_shrink_sum = 0.0
        dbg_fallback_sum = 0.0
        dbg_rare_center_sum = 0.0
        dbg_resample_sum = 0.0
        mix_count = 0.0
        mix_applied_sum = 0.0
        mix_host_points_sum = 0.0
        mix_host_removed_sum = 0.0
        mix_donor_inserted_sum = 0.0
        mix_output_points_sum = 0.0
        mix_side_sum = 0.0
        mix_side_sq_sum = 0.0
        mix_guard_sum = 0.0
        mix_height_shift_sum = 0.0
        mix_height_shift_sq_sum = 0.0
        mix_abs_height_shift_sum = 0.0
        mix_output_ratio_sum = 0.0
        mix_output_ratio_sq_sum = 0.0
        mix_collision_sum = 0.0
        mix_skip_code_counts = np.zeros((6,), dtype=np.float64)
        mix_removed_class_counts = torch.zeros(num_classes, dtype=torch.float64)
        mix_donor_class_counts = torch.zeros(num_classes, dtype=torch.float64)
        mix_output_class_counts = torch.zeros(num_classes, dtype=torch.float64)
        mix_donor_presence_counts = torch.zeros(num_classes, dtype=torch.float64)
        mix_context_pair_counts = torch.zeros((num_classes, num_classes), dtype=torch.float64)
        mix_donor_counter: Counter[int] = Counter()
        optimizer.zero_grad(set_to_none=True)

        use_cuda_timer = torch.cuda.is_available()
        if use_cuda_timer:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
        else:
            t0_wall = time.perf_counter()

        bev_cfg = _bev_cfg(cfg)
        bev_enabled = _bev_enabled(cfg)
        max_train_batches = int(run.get("max_train_batches_per_epoch", 0) or 0)
        bev_loss_type = "ce"
        bev_ignore_index = -1
        bev_dice_smooth = 1.0
        bev_dice_classes = "present"
        bev_ce = None

        if bev_enabled:
            bev_loss_type = str(bev_cfg.get("loss", "ce")).lower()
            bev_ignore_index = int(bev_cfg.get("ignore_index", -1))
            bev_dice_smooth = float(bev_cfg.get("dice_smooth", 1.0))
            bev_dice_classes = str(bev_cfg.get("dice_classes", "present")).lower()
            bev_ce = torch.nn.CrossEntropyLoss(ignore_index=bev_ignore_index)

        for step, batch in enumerate(train_loader, start=1):
            if max_train_batches > 0 and step > max_train_batches:
                break
            processed_steps += 1
            component_steps += 1.0
            bev_loss_step = None
            bev_als_diag = None
            ocons_result = None

            if ocons_cfg.enabled:
                ocons_result = _ocons_train_microstep(
                    model=model,
                    batch=batch,
                    device=device,
                    criterion=criterion,
                    scaler=scaler,
                    grad_accum=grad_accum,
                    cfg=ocons_cfg,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    base_seed=int(run["seed"]),
                    epoch=epoch,
                    global_step=global_step,
                    rank=dist_env.rank,
                    amp=amp,
                )
                total = ocons_result["total"]
                seg_loss = ocons_result["clean_loss"]
                logits = ocons_result["clean_logits"]
                labels = ocons_result["clean_labels"]
            else:
                with torch.autocast(device_type="cuda", enabled=amp):
                    logits, labels, bev_pred = _forward_batch(model, batch, device, is_train=True)
                    seg_loss = criterion(logits, labels)
                    total = seg_loss

                    if bev_enabled and (bev_pred is not None) and ("bev_labels" in batch):
                        bev_targets = batch["bev_labels"]
                        if isinstance(bev_targets, dict):
                            bev_targets = {k: v.to(device, non_blocking=True) for k, v in bev_targets.items()}
                        else:
                            raise TypeError("batch['bev_labels'] must be a dict[level]->Tensor[B,H,W].")
                        bev_loss_step = _compute_bev_loss(
                            bev_pred,
                            bev_targets,
                            loss_type=bev_loss_type,
                            ignore_index=bev_ignore_index,
                            dice_smooth=bev_dice_smooth,
                            dice_classes=bev_dice_classes,
                            ce_criterion=bev_ce,
                        )
                        if bev_loss_step is not None:
                            bev_w = _bev_weight_for_epoch(bev_cfg, epoch)
                            if bool(bev_cfg.get("warmup_only_bev", False)) and (
                                epoch <= int(bev_cfg.get("warmup_epochs", 0))
                            ):
                                total = bev_w * bev_loss_step
                            else:
                                total = seg_loss + bev_w * bev_loss_step

                    if bev_als_cfg.enabled:
                        if bev_pred is None or "bev_als_logits" not in bev_pred:
                            raise RuntimeError("BEV-ALS is enabled but the model did not return bev_als_logits.")
                        required = {"bev_als_target", "bev_als_occupied"}
                        missing = sorted(required.difference(batch))
                        if missing:
                            raise RuntimeError(f"BEV-ALS batch is missing {missing}.")
                        target = batch["bev_als_target"].to(device, non_blocking=True)
                        occupied = batch["bev_als_occupied"].to(device, non_blocking=True)
                        bev_loss_step, bev_als_diag = als_bev_multilabel_loss(
                            bev_pred["bev_als_logits"], target, occupied, bev_als_cfg
                        )
                        bev_w = bev_als_weight_for_epoch(bev_als_cfg, epoch)
                        total = seg_loss + bev_w * bev_loss_step
                        bev_als_diag = {
                            **bev_als_diag,
                            **(bev_pred.get("bev_als_diagnostics", {}) or {}),
                        }

                scaler.scale(total / float(grad_accum)).backward()

            if cm_train is not None:
                preds = logits.argmax(dim=1)
                cm_train.update(preds, labels)

            valid = labels != ignore_index
            n = int(valid.sum().item())
            train_loss_sum += float(total.item()) * max(1, n)
            train_n_sum += max(1, n)

            seg_clean_sum += float(seg_loss.item())
            if ocons_result is not None:
                seg_pert_sum += float(ocons_result["perturbed_loss"].item())
                ocons_loss_sum += float(ocons_result["consistency_loss"].item())
                aux_weight_sum += float(ocons_result["consistency_weight"])
                vd = ocons_result["view_diagnostics"]
                cd = ocons_result["consistency_diagnostics"]
                ocons_scalar_sum += torch.tensor(
                    [
                        vd.actual_mask_fraction,
                        vd.jaccard,
                        ocons_result["coordinate_match_coverage"],
                        float(vd.constrained_samples),
                        float(vd.samples),
                        float(vd.protected_disappearances.sum().item()),
                        float(cd["micro_kl"].item()),
                        float(cd["macro_kl"].item()),
                        float(cd["agreement"].item()),
                        float(cd["clean_confidence"].item()),
                        float(cd["perturbed_confidence"].item()),
                    ],
                    dtype=torch.float64,
                )
                ocons_class_before += vd.class_before.to(torch.float64)
                ocons_class_after += vd.class_after.to(torch.float64)
                support = cd["per_class_support"].detach().cpu().to(torch.float64)
                ocons_class_kl_sum += cd["per_class_kl"].detach().cpu().to(torch.float64) * support
                ocons_class_kl_support += support
            elif bev_als_cfg.enabled:
                assert bev_loss_step is not None and bev_als_diag is not None
                bev_als_loss_sum += float(bev_loss_step.item())
                aux_weight_sum += float(bev_w)
                in_bounds = float(batch["meta_bev_als_in_bounds_fraction"].float().mean().item())
                multi = float(batch["meta_bev_als_multi_label_fraction"].float().mean().item())
                feature_in_bounds = float(bev_als_diag["feature_in_bounds_fraction"].item())
                if in_bounds < bev_als_cfg.min_in_bounds_fraction or feature_in_bounds < bev_als_cfg.min_in_bounds_fraction:
                    raise RuntimeError(
                        "BEV-ALS crop/frame coverage fell below the configured correctness floor: "
                        f"target={in_bounds:.6f}, block8={feature_in_bounds:.6f}, "
                        f"minimum={bev_als_cfg.min_in_bounds_fraction:.6f}."
                    )
                empty_rate = float(batch["meta_bev_als_empty_slices"].float().mean().item()) / float(
                    bev_als_cfg.height_slices
                )
                bev_als_scalar_sum += torch.tensor(
                    [
                        float(bev_als_diag["bce"].item()),
                        float(bev_als_diag["dice_loss"].item()),
                        float(bev_als_diag["tp"].item()),
                        float(bev_als_diag["fp"].item()),
                        float(bev_als_diag["fn"].item()),
                        in_bounds,
                        multi,
                        empty_rate,
                        feature_in_bounds,
                    ],
                    dtype=torch.float64,
                )
                bev_als_tp += bev_als_diag["tp_by_slice_class"].detach().cpu().to(torch.float64)
                bev_als_fp += bev_als_diag["fp_by_slice_class"].detach().cpu().to(torch.float64)
                bev_als_fn += bev_als_diag["fn_by_slice_class"].detach().cpu().to(torch.float64)
                bev_als_positive += bev_als_diag["positive_support"].detach().cpu().to(torch.float64)
                bev_als_class_before += batch["meta_bev_als_class_before"].sum(dim=0).to(torch.float64)
                bev_als_class_in_bounds += batch["meta_bev_als_class_in_bounds"].sum(dim=0).to(torch.float64)

            # --- NEW: consume DALES crop meta if present ---
            if "meta_n_vox" in batch:
                # batch values are [B], but B might be 1 for DALES; take mean for safety
                dbg_n += 1
                dbg_vox_sum += float(batch["meta_n_vox"].float().mean().item())
                dbg_rare_sum += float(batch["meta_rare_frac"].float().mean().item())
                dbg_crop_size_sum += float(batch["meta_crop_size_xy_m"].float().mean().item())
                dbg_shrink_sum += float(batch["meta_shrink_steps"].float().mean().item())
                dbg_fallback_sum += float(batch["meta_used_fallback"].float().mean().item())
                dbg_rare_center_sum += float(batch["meta_used_rare_center"].float().mean().item())
                dbg_resample_sum += float(batch["meta_resample_attempt"].float().mean().item())

            if "meta_mix_applied" in batch:
                applied = batch["meta_mix_applied"].to(torch.bool).reshape(-1)
                n_mix = float(applied.numel())
                mix_count += n_mix
                mix_applied_sum += float(applied.double().sum().item())
                mix_host_points_sum += float(batch["meta_mix_host_points"].double().sum().item())
                mix_host_removed_sum += float(batch["meta_mix_host_removed"].double().sum().item())
                mix_donor_inserted_sum += float(batch["meta_mix_donor_inserted"].double().sum().item())
                mix_output_points_sum += float(batch["meta_mix_output_points"].double().sum().item())
                mix_guard_sum += float(batch["meta_mix_guard_band_m"].double().sum().item())
                mix_collision_sum += float(batch["meta_mix_cross_provenance_voxels"].double().sum().item())

                skip_codes = batch["meta_mix_skip_code"].to(torch.int64).reshape(-1)
                skip_bc = torch.bincount(skip_codes.clamp(0, 5), minlength=6)
                mix_skip_code_counts += skip_bc.cpu().numpy().astype(np.float64)
                host_points = batch["meta_mix_host_points"].double().reshape(-1).clamp_min(1.0)
                output_points = batch["meta_mix_output_points"].double().reshape(-1)
                ratios = output_points / host_points
                mix_output_ratio_sum += float(ratios.sum().item())
                mix_output_ratio_sq_sum += float((ratios * ratios).sum().item())

                if bool(applied.any()):
                    side = batch["meta_mix_replacement_side_m"].double().reshape(-1)[applied]
                    shift = batch["meta_mix_height_shift_m"].double().reshape(-1)[applied]
                    mix_side_sum += float(side.sum().item())
                    mix_side_sq_sum += float((side * side).sum().item())
                    mix_height_shift_sum += float(shift.sum().item())
                    mix_height_shift_sq_sum += float((shift * shift).sum().item())
                    mix_abs_height_shift_sum += float(shift.abs().sum().item())

                removed = batch["meta_mix_removed_class_counts"].double()
                donor = batch["meta_mix_donor_class_counts"].double()
                output_cls = batch["meta_mix_output_class_counts"].double()
                context_pairs = batch["meta_mix_context_pairs"].double()
                mix_removed_class_counts += removed.sum(dim=0)
                mix_donor_class_counts += donor.sum(dim=0)
                mix_output_class_counts += output_cls.sum(dim=0)
                mix_donor_presence_counts += (donor > 0).double().sum(dim=0)
                mix_context_pair_counts += context_pairs.sum(dim=0)

                donor_indices = batch["meta_mix_donor_index"].to(torch.int64).reshape(-1)
                for was_applied, donor_idx in zip(applied.tolist(), donor_indices.tolist()):
                    if was_applied and int(donor_idx) >= 0:
                        mix_donor_counter[int(donor_idx)] += 1

            if step % grad_accum == 0:
                scaler.unscale_(optimizer)
                backbone_norm, aux_norm = _gradient_l2_norms(model)
                grad_measurements += 1.0
                backbone_grad_norm_sum += backbone_norm
                aux_grad_norm_sum += aux_norm
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % int(run.get("log_every_steps", 50)) == 0 and is_main_process(dist_env):
                lr = optimizer.param_groups[0]["lr"]
                if ocons_result is not None:
                    msg = (
                        f"[epoch {epoch:03d} step {step:05d}] total={total.item():.4f} "
                        f"clean={seg_loss.item():.4f} pert={ocons_result['perturbed_loss'].item():.4f} "
                        f"cons={ocons_result['consistency_loss'].item():.4f} "
                        f"w={ocons_result['consistency_weight']:.4f} "
                        f"mask={ocons_result['view_diagnostics'].actual_mask_fraction:.4f} "
                        f"match={ocons_result['coordinate_match_coverage']:.6f} lr={lr:.2e}"
                    )
                elif bev_als_cfg.enabled:
                    bev_loss_val = float(bev_loss_step.item()) if bev_loss_step is not None else 0.0
                    msg = (
                        f"[epoch {epoch:03d} step {step:05d}] total={total.item():.4f} "
                        f"seg={seg_loss.item():.4f} bev_als={bev_loss_val:.4f} "
                        f"w={bev_w:.4f} bounds={bev_als_diag['feature_in_bounds_fraction'].item():.4f} "
                        f"lr={lr:.2e}"
                    )
                elif bev_enabled:
                    bev_loss_val = float(bev_loss_step.item()) if (bev_loss_step is not None) else 0.0
                    msg = f"[epoch {epoch:03d} step {step:05d}] total={total.item():.4f} seg={seg_loss.item():.4f} bev={bev_loss_val:.4f} lr={lr:.2e}"
                else:
                    msg = f"[epoch {epoch:03d} step {step:05d}] total={total.item():.4f} seg={seg_loss.item():.4f} lr={lr:.2e}"

                # append crop debug if present
                if "meta_n_vox" in batch:
                    msg += (
                        f" | n_vox={int(batch['meta_n_vox'].float().mean().item())}"
                        f" crop_m={batch['meta_crop_size_xy_m'].float().mean().item():.1f}"
                        f" shrink_min={int(batch['meta_shrink_steps'].float().min().item())}"
                        f" shrink_mean={int(batch['meta_shrink_steps'].float().mean().item())}"
                        f" shrink_max={int(batch['meta_shrink_steps'].float().max().item())}"
                        f" rare_frac={batch['meta_rare_frac'].float().mean().item():.3f}"
                        f" rare_center={int(batch['meta_used_rare_center'].float().mean().item())}"
                        f" fallback={int(batch['meta_used_fallback'].float().mean().item())}"
                        f" resample={int(batch['meta_resample_attempt'].float().mean().item())}"
                    )
                print(msg, flush=True)

        # flush leftover grads if dataloader size not divisible by grad_accum
        if processed_steps > 0 and (processed_steps % grad_accum) != 0:
            scaler.unscale_(optimizer)
            backbone_norm, aux_norm = _gradient_l2_norms(model)
            grad_measurements += 1.0
            backbone_grad_norm_sum += backbone_norm
            aux_grad_norm_sum += aux_norm
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

        if use_cuda_timer:
            t1.record()
            torch.cuda.synchronize()
            epoch_ms = t0.elapsed_time(t1)
            epoch_s = epoch_ms / 1000.0
        else:
            epoch_s = time.perf_counter() - t0_wall

        train_time_max = torch.tensor(float(epoch_s), dtype=torch.float64, device=device)
        if dist_env.enabled:
            torch.distributed.all_reduce(train_time_max, op=torch.distributed.ReduceOp.MAX)
        epoch_s = float(train_time_max.item())

        if processed_steps == 0:
            raise RuntimeError("Training DataLoader produced no batches.")

        base_totals = torch.tensor(
            [
                train_loss_sum,
                float(train_n_sum),
                component_steps,
                seg_clean_sum,
                seg_pert_sum,
                ocons_loss_sum,
                bev_als_loss_sum,
                aux_weight_sum,
                grad_measurements,
                backbone_grad_norm_sum,
                aux_grad_norm_sum,
            ],
            dtype=torch.float64,
            device=device,
        )
        base_totals = all_reduce_sum(dist_env, base_totals).detach().cpu()
        component_denom = max(1.0, float(base_totals[2].item()))
        grad_denom = max(1.0, float(base_totals[8].item()))
        backbone_grad_mean = float(base_totals[9].item()) / grad_denom
        aux_grad_mean = float(base_totals[10].item()) / grad_denom
        strict_diagnostic = bool(
            max_train_batches > 0 or run.get("skip_validation", False) or run.get("skip_final_test", False)
        )
        if strict_diagnostic and (not math.isfinite(backbone_grad_mean) or backbone_grad_mean <= 0.0):
            raise RuntimeError(f"Backbone gradient diagnostic is invalid: {backbone_grad_mean}")
        if strict_diagnostic and bev_als_cfg.enabled and (
            not math.isfinite(aux_grad_mean) or aux_grad_mean <= 0.0
        ):
            raise RuntimeError(f"BEV-ALS head gradient diagnostic is invalid: {aux_grad_mean}")

        ocons_global = all_reduce_sum(dist_env, ocons_scalar_sum.to(device)).detach().cpu()
        ocons_before_global = all_reduce_sum(dist_env, ocons_class_before.to(device)).detach().cpu()
        ocons_after_global = all_reduce_sum(dist_env, ocons_class_after.to(device)).detach().cpu()
        ocons_kl_sum_global = all_reduce_sum(dist_env, ocons_class_kl_sum.to(device)).detach().cpu()
        ocons_kl_support_global = all_reduce_sum(dist_env, ocons_class_kl_support.to(device)).detach().cpu()

        bev_als_global = all_reduce_sum(dist_env, bev_als_scalar_sum.to(device)).detach().cpu()
        bev_als_tp_global = all_reduce_sum(dist_env, bev_als_tp.to(device)).detach().cpu()
        bev_als_fp_global = all_reduce_sum(dist_env, bev_als_fp.to(device)).detach().cpu()
        bev_als_fn_global = all_reduce_sum(dist_env, bev_als_fn.to(device)).detach().cpu()
        bev_als_positive_global = all_reduce_sum(dist_env, bev_als_positive.to(device)).detach().cpu()
        bev_als_class_before_global = all_reduce_sum(dist_env, bev_als_class_before.to(device)).detach().cpu()
        bev_als_class_in_bounds_global = all_reduce_sum(
            dist_env, bev_als_class_in_bounds.to(device)
        ).detach().cpu()

        peak = torch.zeros(2, dtype=torch.float64, device=device)
        if torch.cuda.is_available():
            peak[0] = float(torch.cuda.max_memory_allocated(device)) / float(1024**2)
            peak[1] = float(torch.cuda.max_memory_reserved(device)) / float(1024**2)
        if dist_env.enabled:
            torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)

        bev_metrics = None
        train_loss = float(base_totals[0].item()) / max(1.0, float(base_totals[1].item()))

        # ---- Eval ----
        do_eval = (not bool(run.get("skip_validation", False))) and (
            (epoch % int(run.get("eval_every_epochs", 1)) == 0) or (epoch == epochs)
        )
        eval_start = time.perf_counter()
        if do_eval:
            eval_mode = str(cfg.get("eval", {}).get("mode", "voxel")).lower()
            if eval_mode == "point":
                val_metrics = evaluate_pointwise(
                    model=model,
                    dist_env=dist_env,
                    dataset_obj=val_loader.dataset,
                    criterion_cpu=criterion_cpu,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    device=device,
                    amp=amp,
                    cfg=cfg,
                )
            elif eval_mode in ("voxel_windowed", "voxel_window"):
                val_metrics = evaluate_voxel_windowed(
                    model=model,
                    dist_env=dist_env,
                    dataset_obj=val_loader.dataset,
                    criterion_cpu=criterion_cpu,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    device=device,
                    amp=amp,
                    cfg=cfg,
                )
            else:
                val_metrics = evaluate(
                    model=model,
                    dist_env=dist_env,
                    loader=val_loader,
                    criterion=criterion,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    device=device,
                    amp=amp,
                )

            if bev_enabled and do_eval:
                bev_metrics = eval_and_visualize_bev(
                    model=model,
                    dist_env=dist_env,
                    loader=val_loader,
                    device=device,
                    amp=amp,
                    out_dir=out_dir,
                    epoch=epoch,
                    num_classes=num_classes,
                    cfg=cfg,
                )
        else:
            val_metrics = {
                "loss": float("nan"),
                "miou": float("nan"),
                "miou_valid": float("nan"),  # <-- add this
                "macro_f1": float("nan"),
                "per_class_iou": [float("nan")] * num_classes,
                "per_class_f1": [float("nan")] * num_classes,
                "per_class_precision": [float("nan")] * num_classes,
                "per_class_recall": [float("nan")] * num_classes,
                "confusion_matrix": None,
            }
        eval_time_s = time.perf_counter() - eval_start if do_eval else 0.0
        eval_time_max = torch.tensor(float(eval_time_s), dtype=torch.float64, device=device)
        if dist_env.enabled:
            torch.distributed.all_reduce(eval_time_max, op=torch.distributed.ReduceOp.MAX)
        eval_time_s = float(eval_time_max.item())

        # ---- Checkpointing ----
        is_best = do_eval and (val_metrics["miou"] > best_val_miou)
        if is_best:
            best_val_miou = float(val_metrics["miou"])

        if is_main_process(dist_env) and do_eval:
            confusion_payload = {
                "epoch": epoch,
                "method": method_contract.name,
                "class_names": class_names,
                "rows": "ground_truth",
                "columns": "prediction",
                "matrix": val_metrics["confusion_matrix"],
            }
            # These files are overwritten, not accumulated, so observability does
            # not grow the run directory over a long or resumed training.
            save_json(out_dir / "val_confusion_latest.json", confusion_payload)
            if is_best:
                save_json(out_dir / "val_confusion_best.json", confusion_payload)

        save_every = int(run.get("save_every_epochs", 5))

        if is_main_process(dist_env) and (epoch % save_every == 0 or epoch == epochs or is_best):
            common_checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "method_contract": method_contract.to_dict(),
                "model_state": unwrap_model(model).state_dict(),
                "cfg": cfg,
                "best_val_miou": best_val_miou,
            }
            resumable_state = {
                **common_checkpoint,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": (None if scheduler is None else scheduler.state_dict()),
                "scaler_state": scaler.state_dict(),
            }
            ckpt_dir = out_dir / "checkpoints"
            # last.pt is resumable and is atomically overwritten; it never accumulates.
            atomic_save_torch(resumable_state, ckpt_dir / "last.pt")
            if is_best:
                best_model_only = bool(run.get("best_checkpoint_model_only", False))
                atomic_save_torch(
                    common_checkpoint if best_model_only else resumable_state,
                    ckpt_dir / "best.pt",
                )

            keep_epoch_snapshots = bool(run.get("keep_epoch_snapshots", False))
            if keep_epoch_snapshots and (epoch % save_every == 0 or epoch == epochs):
                snapshot_model_only = bool(run.get("epoch_snapshots_model_only", True))
                atomic_save_torch(
                    common_checkpoint if snapshot_model_only else resumable_state,
                    ckpt_dir / f"epoch_{epoch:03d}.pt",
                )

            keep_last_k = int(run.get("keep_last_k_epoch_snapshots", 0))
            removed = _prune_epoch_snapshots(
                ckpt_dir,
                keep_last_k if keep_epoch_snapshots else 0,
            )
            if removed:
                print(f"[storage] pruned epoch checkpoints: {removed}", flush=True)

        if dist_env.enabled:
            torch.distributed.barrier()  # optional but nice: sync after saving

        storage_gb = torch.zeros(3, dtype=torch.float64, device=device)
        if is_main_process(dist_env):
            checkpoint_bytes = _directory_size_bytes(out_dir / "checkpoints")
            run_cache_bytes = _directory_size_bytes(out_dir / "_run_cache")
            run_bytes = _directory_size_bytes(out_dir)
            storage_gb[:] = torch.tensor(
                [checkpoint_bytes, run_bytes, run_cache_bytes],
                dtype=torch.float64,
                device=device,
            ) / float(1024**3)
        if dist_env.enabled:
            torch.distributed.broadcast(storage_gb, src=0)
        checkpoint_dir_gb, run_dir_gb, run_cache_gb = [float(x) for x in storage_gb.detach().cpu().tolist()]
        max_run_dir_gb = float(run.get("max_run_dir_gb", 0.0) or 0.0)
        if max_run_dir_gb > 0.0 and run_dir_gb > max_run_dir_gb:
            raise RuntimeError(
                f"Run directory exceeded run.max_run_dir_gb: "
                f"{run_dir_gb:.2f} GiB > {max_run_dir_gb:.2f} GiB. "
                "Training stopped before exhausting quota."
            )

        lr = optimizer.param_groups[0]["lr"]

        epoch_wall_time_s = time.perf_counter() - epoch_wall_start
        epoch_wall_time_max = torch.tensor(
            float(epoch_wall_time_s), dtype=torch.float64, device=device
        )
        if dist_env.enabled:
            torch.distributed.all_reduce(epoch_wall_time_max, op=torch.distributed.ReduceOp.MAX)
        epoch_wall_time_s = float(epoch_wall_time_max.item())

        row = {
            "epoch": epoch,
            "method": method_contract.name,
            "debug_limited_run": int(strict_diagnostic),
            "lr": lr,
            "train_loss": train_loss,
            "train_seg_clean_loss": float(base_totals[3].item()) / component_denom,
            "train_seg_perturbed_loss": (
                float(base_totals[4].item()) / component_denom if ocons_cfg.enabled else float("nan")
            ),
            "train_ocons_consistency_loss": (
                float(base_totals[5].item()) / component_denom if ocons_cfg.enabled else float("nan")
            ),
            "train_bev_als_loss": (
                float(base_totals[6].item()) / component_denom if bev_als_cfg.enabled else float("nan")
            ),
            "aux_weight": (
                float(base_totals[7].item()) / component_denom
                if (ocons_cfg.enabled or bev_als_cfg.enabled)
                else 0.0
            ),
            "val_loss": val_metrics["loss"],
            "val_miou": val_metrics["miou"],
            "val_macro_f1": val_metrics["macro_f1"],
            "time_epoch_s": epoch_s,
            "eval_time_s": eval_time_s,
            "epoch_wall_time_s": epoch_wall_time_s,
            "world_size": int(dist_env.world_size),
            "train_batches_per_rank": int(processed_steps),
            "best_val_miou": best_val_miou,
            "checkpoint_dir_gb": checkpoint_dir_gb,
            "run_dir_gb": run_dir_gb,
            "run_cache_gb": run_cache_gb,
            "gpu_peak_allocated_mib": float(peak[0].item()),
            "gpu_peak_reserved_mib": float(peak[1].item()),
            "backbone_grad_norm": backbone_grad_mean,
            "method_aux_grad_norm": (
                aux_grad_mean
                if bev_als_cfg.enabled
                else float("nan")
            ),
            "bev_miou": bev_metrics["bev_miou"] if bev_metrics is not None else float("nan"),
        }

        if ocons_cfg.enabled:
            row.update(
                {
                    "ocons_actual_mask_fraction": float(ocons_global[0].item()) / component_denom,
                    "ocons_active_jaccard": float(ocons_global[1].item()) / component_denom,
                    "ocons_coordinate_match_coverage": float(ocons_global[2].item()) / component_denom,
                    "ocons_protection_constrained_rate": float(ocons_global[3].item())
                    / max(1.0, float(ocons_global[4].item())),
                    "ocons_protected_disappearances": int(ocons_global[5].item()),
                    "ocons_micro_kl": float(ocons_global[6].item()) / component_denom,
                    "ocons_macro_kl": float(ocons_global[7].item()) / component_denom,
                    "ocons_agreement": float(ocons_global[8].item()) / component_denom,
                    "ocons_clean_confidence": float(ocons_global[9].item()) / component_denom,
                    "ocons_perturbed_confidence": float(ocons_global[10].item()) / component_denom,
                }
            )
            if is_main_process(dist_env):
                assert ocons_class_logger is not None
                for class_id, class_name in enumerate(class_names):
                    before = float(ocons_before_global[class_id].item())
                    after = float(ocons_after_global[class_id].item())
                    kl_support = float(ocons_kl_support_global[class_id].item())
                    ocons_class_logger.log(
                        {
                            "epoch": epoch,
                            "class_id": class_id,
                            "class_name": class_name,
                            "clean_voxels": int(before),
                            "perturbed_voxels": int(after),
                            "retention": after / max(1.0, before),
                            "kl": float(ocons_kl_sum_global[class_id].item()) / max(1.0, kl_support),
                            "kl_support": int(kl_support),
                        }
                    )

        if bev_als_cfg.enabled:
            tp_all = float(bev_als_global[2].item())
            fp_all = float(bev_als_global[3].item())
            fn_all = float(bev_als_global[4].item())
            precision = tp_all / max(1.0, tp_all + fp_all)
            recall = tp_all / max(1.0, tp_all + fn_all)
            f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
            row.update(
                {
                    "bev_als_bce": float(bev_als_global[0].item()) / component_denom,
                    "bev_als_dice_loss": float(bev_als_global[1].item()) / component_denom,
                    "bev_als_precision": precision,
                    "bev_als_recall": recall,
                    "bev_als_f1": f1,
                    "bev_als_in_bounds_fraction": float(bev_als_global[5].item()) / component_denom,
                    "bev_als_multi_label_fraction": float(bev_als_global[6].item()) / component_denom,
                    "bev_als_empty_slice_rate": float(bev_als_global[7].item()) / component_denom,
                    "bev_als_feature_in_bounds_fraction": float(bev_als_global[8].item()) / component_denom,
                }
            )
            if is_main_process(dist_env):
                assert bev_als_class_logger is not None
                for slice_id in range(bev_als_cfg.height_slices):
                    for class_id, class_name in enumerate(class_names):
                        tp = float(bev_als_tp_global[slice_id, class_id].item())
                        fp = float(bev_als_fp_global[slice_id, class_id].item())
                        fn = float(bev_als_fn_global[slice_id, class_id].item())
                        p = tp / max(1.0, tp + fp)
                        r = tp / max(1.0, tp + fn)
                        class_before = float(bev_als_class_before_global[class_id].item())
                        class_in_bounds = float(bev_als_class_in_bounds_global[class_id].item())
                        bev_als_class_logger.log(
                            {
                                "epoch": epoch,
                                "height_slice": slice_id,
                                "class_id": class_id,
                                "class_name": class_name,
                                "input_voxels": int(class_before),
                                "in_bounds_voxels": int(class_in_bounds),
                                "bounds_retention": class_in_bounds / max(1.0, class_before),
                                "positive_support": int(bev_als_positive_global[slice_id, class_id].item()),
                                "tp": int(tp),
                                "fp": int(fp),
                                "fn": int(fn),
                                "precision": p,
                                "recall": r,
                                "f1": 2.0 * p * r / max(1e-8, p + r),
                            }
                        )

        if mix_enabled:
            mix_totals = torch.tensor(
                [
                    mix_count,
                    mix_applied_sum,
                    mix_host_points_sum,
                    mix_host_removed_sum,
                    mix_donor_inserted_sum,
                    mix_output_points_sum,
                    mix_side_sum,
                    mix_side_sq_sum,
                    mix_guard_sum,
                    mix_height_shift_sum,
                    mix_height_shift_sq_sum,
                    mix_abs_height_shift_sum,
                    mix_output_ratio_sum,
                    mix_output_ratio_sq_sum,
                    mix_collision_sum,
                    *mix_skip_code_counts.tolist(),
                ],
                device=device,
                dtype=torch.float64,
            )
            mix_totals = all_reduce_sum(dist_env, mix_totals)
            mt = mix_totals.detach().cpu().numpy()
            denom = max(1.0, float(mt[0]))
            applied_denom = max(1.0, float(mt[1]))
            side_mean = float(mt[6] / applied_denom)
            side_std = math.sqrt(max(0.0, float(mt[7] / applied_denom) - side_mean**2))
            shift_mean = float(mt[9] / applied_denom)
            shift_std = math.sqrt(max(0.0, float(mt[10] / applied_denom) - shift_mean**2))
            ratio_mean = float(mt[12] / denom)
            ratio_std = math.sqrt(max(0.0, float(mt[13] / denom) - ratio_mean**2))
            skip_counts = mt[15:21]

            class_totals = torch.cat(
                [
                    mix_removed_class_counts,
                    mix_donor_class_counts,
                    mix_output_class_counts,
                    mix_donor_presence_counts,
                    mix_context_pair_counts.reshape(-1),
                ]
            ).to(device=device, dtype=torch.float64)
            class_totals = all_reduce_sum(dist_env, class_totals).cpu()
            pos = 0
            removed_global = class_totals[pos : pos + num_classes]
            pos += num_classes
            donor_global = class_totals[pos : pos + num_classes]
            pos += num_classes
            output_global = class_totals[pos : pos + num_classes]
            pos += num_classes
            donor_presence_global = class_totals[pos : pos + num_classes]
            pos += num_classes
            context_global = class_totals[pos:].reshape(num_classes, num_classes)

            if dist_env.enabled:
                gathered_counters = [None for _ in range(dist_env.world_size)]
                torch.distributed.all_gather_object(
                    gathered_counters,
                    dict(mix_donor_counter),
                )
                donor_counter_global: Counter[int] = Counter()
                for counter_dict in gathered_counters:
                    donor_counter_global.update(counter_dict or {})
            else:
                donor_counter_global = mix_donor_counter

            donor_total = float(sum(donor_counter_global.values()))
            donor_unique = len(donor_counter_global)
            if donor_total > 0.0:
                probs = [float(v) / donor_total for v in donor_counter_global.values()]
                entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
                entropy_norm = entropy / math.log(donor_unique) if donor_unique > 1 else 1.0
                donor_max_share = max(probs)
            else:
                entropy_norm = 0.0
                donor_max_share = 0.0

            row.update(
                {
                    "mix_rate": float(mt[1] / denom),
                    "mix_skip_rate": float(1.0 - mt[1] / denom),
                    "mix_host_points_mean": float(mt[2] / denom),
                    "mix_host_removed_mean": float(mt[3] / denom),
                    "mix_donor_inserted_mean": float(mt[4] / denom),
                    "mix_output_points_mean": float(mt[5] / denom),
                    "mix_replacement_side_m_mean": side_mean,
                    "mix_guard_band_m_mean": float(mt[8] / denom),
                    "mix_height_shift_m_mean": shift_mean,
                    "mix_cross_provenance_voxels": float(mt[14]),
                    "mix_output_budget_skip_rate": float(skip_counts[5] / denom),
                    "mix_probability_skip_rate": float(skip_counts[1] / denom),
                    "mix_no_donor_skip_rate": float(skip_counts[2] / denom),
                    "mix_invalid_region_skip_rate": float(skip_counts[3] / denom),
                    "mix_collision_skip_rate": float(skip_counts[4] / denom),
                    "mix_replacement_side_m_std": side_std,
                    "mix_height_shift_m_std": shift_std,
                    "mix_abs_height_shift_m_mean": float(mt[11] / applied_denom),
                    "mix_output_to_host_points_ratio_mean": ratio_mean,
                    "mix_output_to_host_points_ratio_std": ratio_std,
                    "mix_donor_unique_count": int(donor_unique),
                    "mix_donor_selection_entropy": float(entropy_norm),
                    "mix_donor_max_share": float(donor_max_share),
                }
            )

            if is_main_process(dist_env):
                assert mix_class_logger is not None
                for class_id, class_name in enumerate(class_names):
                    mix_class_logger.log(
                        {
                            "epoch": epoch,
                            "class_id": class_id,
                            "class_name": class_name,
                            "host_removed_points": int(removed_global[class_id].item()),
                            "donor_inserted_points": int(donor_global[class_id].item()),
                            "mixed_output_points": int(output_global[class_id].item()),
                            "donor_presence_samples": int(donor_presence_global[class_id].item()),
                        }
                    )
                mix_context_history.append(
                    {
                        "epoch": int(epoch),
                        "donor_class_by_host_context_class": context_global.to(torch.int64).tolist(),
                        "donor_usage_counts": {str(k): int(v) for k, v in sorted(donor_counter_global.items())},
                    }
                )
                save_json(
                    out_dir / "mix_context_diagnostics.json",
                    {
                        "class_names": class_names,
                        "epochs": mix_context_history,
                    },
                )

        # --- NEW: DALES crop debug epoch aggregates ---
        if dbg_n > 0:
            row["train_vox_mean"] = dbg_vox_sum / dbg_n
            row["train_rare_frac_mean"] = dbg_rare_sum / dbg_n
            row["train_crop_size_mean"] = dbg_crop_size_sum / dbg_n
            row["train_shrink_steps_mean"] = dbg_shrink_sum / dbg_n
            row["train_fallback_rate"] = dbg_fallback_sum / dbg_n
            row["train_rare_center_rate"] = dbg_rare_center_sum / dbg_n
            row["train_resample_attempt_mean"] = dbg_resample_sum / dbg_n

        for name, v in zip(class_names, val_metrics["per_class_iou"]):
            row[f"val_iou_{name.replace(' ', '_').replace('.', '')}"] = v
        for name, v in zip(class_names, val_metrics["per_class_f1"]):
            row[f"val_f1_{name.replace(' ', '_').replace('.', '')}"] = v
        for name, v in zip(class_names, val_metrics["per_class_precision"]):
            row[f"val_precision_{name.replace(' ', '_').replace('.', '')}"] = v
        for name, v in zip(class_names, val_metrics["per_class_recall"]):
            row[f"val_recall_{name.replace(' ', '_').replace('.', '')}"] = v

        logger.log(row)

        if is_main_process(dist_env):
            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
                f"val_miou={val_metrics['miou']:.4f} "
                f"val_miou_valid={float(val_metrics.get('miou_valid', float('nan'))):.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"best={best_val_miou:.4f} train_time={format_seconds(epoch_s)} "
                f"eval_time={format_seconds(eval_time_s)} wall_time={format_seconds(epoch_wall_time_s)}"
            )

    if bool(run.get("skip_final_test", False)):
        if is_main_process(dist_env):
            save_json(
                out_dir / "diagnostic_complete.json",
                {
                    "status": "ok",
                    "method": method_contract.name,
                    "epochs": epochs,
                    "max_train_batches_per_epoch": int(run.get("max_train_batches_per_epoch", 0) or 0),
                    "skip_validation": bool(run.get("skip_validation", False)),
                    "skip_final_test": True,
                    "warning": "This is an integration diagnostic, not a scientific result.",
                },
            )
            print("[diagnostic] completed successfully; final test intentionally skipped.", flush=True)
        if dataset == "eclair" and eclair_run_cache_base is not None:
            if dist_env.enabled:
                torch.distributed.barrier()
            if is_main_process(dist_env):
                shutil.rmtree(eclair_run_cache_base, ignore_errors=True)
            if dist_env.enabled:
                torch.distributed.barrier()
        return

    # Final test evaluation on best model (respect eval.mode)
    best_ckpt = torch.load(out_dir / "checkpoints" / "best.pt", map_location="cpu")
    unwrap_model(model).load_state_dict(best_ckpt["model_state"])

    eval_mode = str(cfg.get("eval", {}).get("mode", "voxel")).lower()
    if eval_mode == "point":
        test_metrics = evaluate_pointwise(
            model=model,
            dist_env=dist_env,
            dataset_obj=test_loader.dataset,
            criterion_cpu=criterion_cpu,  # keep parity with val
            num_classes=num_classes,
            ignore_index=ignore_index,
            device=device,
            amp=amp,
            cfg=cfg,
        )
    elif eval_mode in ("voxel_windowed", "voxel_window"):
        test_metrics = evaluate_voxel_windowed(
            model=model,
            dist_env=dist_env,
            dataset_obj=test_loader.dataset,
            criterion_cpu=criterion_cpu,
            num_classes=num_classes,
            ignore_index=ignore_index,
            device=device,
            amp=amp,
            cfg=cfg,
        )
    else:
        test_metrics = evaluate(
            model=model,
            dist_env=dist_env,
            loader=test_loader,
            criterion=criterion,
            num_classes=num_classes,
            ignore_index=ignore_index,
            device=device,
            amp=amp,
        )

    if is_main_process(dist_env):
        save_json(out_dir / "test_metrics.json", test_metrics)
        print(
            f"[TEST] loss={test_metrics['loss']:.4f} mIoU={test_metrics['miou']:.4f} "
            f"macroF1={test_metrics['macro_f1']:.4f} miou_valid={test_metrics['miou_valid']:.4f}"
        )

    # ---- Final checkpoint retention enforcement ----
    if is_main_process(dist_env):
        keep_epoch_snapshots = bool(run.get("keep_epoch_snapshots", False))
        keep_last_k = int(run.get("keep_last_k_epoch_snapshots", 0))
        _prune_epoch_snapshots(
            out_dir / "checkpoints",
            keep_last_k if keep_epoch_snapshots else 0,
        )

    # ---- Cleanup run-scoped ECLAIR cache (prevents staleness across runs) ----
    if dataset == "eclair" and eclair_run_cache_base is not None:
        if dist_env.enabled:
            torch.distributed.barrier()
        if is_main_process(dist_env):
            shutil.rmtree(eclair_run_cache_base, ignore_errors=True)
        if dist_env.enabled:
            torch.distributed.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g., configs/e0_eclair.yaml)",
    )
    args = ap.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
