# /eclair_model_train/train.py
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn.functional as F
from src.augment import AugmentConfig
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
from src.data_eclair import EclairTiles, PatchConfig, minkowski_collate_fn
from src.dist import DistEnv, all_reduce_sum, init_distributed, is_main_process
from src.features import FeatureConfig, build_features, infer_in_channels
from src.label_maps import ECLAIR_CLASS_NAMES_11
from src.losses import FocalLoss, FocalLossConfig, FocalLovaszLoss, LovaszSoftmaxLoss, LovaszWarmupConfig
from src.metrics import ConfusionMatrix
from src.model import build_model
from src.utils import CSVLogger, atomic_save_torch, format_seconds, save_json, set_seed, unwrap_model
from src.voxelization import VoxelizationConfig, voxelize_from_q
from torch.utils.data import DataLoader


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
            num_classes=int(ls["num_classes"]),
            run_cache_root=str(eclair_run_cache_dir) if eclair_run_cache_dir is not None else None,
            run_cache_precompute_returns_onehot=(
                eclair_run_cache_dir is not None and (feat_cfg.use_return_number or feat_cfg.use_number_of_returns)
            ),
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
            num_classes=int(ls["num_classes"]),
            run_cache_root=str(eclair_run_cache_dir) if eclair_run_cache_dir is not None else None,
            run_cache_precompute_returns_onehot=(
                eclair_run_cache_dir is not None and (feat_cfg.use_return_number or feat_cfg.use_number_of_returns)
            ),
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
            num_classes=int(ls["num_classes"]),
            run_cache_root=str(eclair_run_cache_dir) if eclair_run_cache_dir is not None else None,
            run_cache_precompute_returns_onehot=(
                eclair_run_cache_dir is not None and (feat_cfg.use_return_number or feat_cfg.use_number_of_returns)
            ),
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

    out = _model_forward_safe(model, st, is_train=is_train, bev_selected_idx=bev_sel, compute_bev=None)

    bev_pred = None
    if isinstance(out, (tuple, list)) and len(out) == 2:
        out, bev_pred = out  # out is SparseTensor
    logits = out.F  # [N, C]
    return logits, labels, bev_pred


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
        "miou_valid": res.miou_valid,
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
        xyz = raw["xyz"].astype(np.float64, copy=False)

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
        patch_cfg = dataset_obj.patch_cfg
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
        "miou_valid": res.miou_valid,
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
                    logits_sub_np = out_st.F.detach().float().cpu().numpy()

                    voxel_logits_sum[sel] += logits_sub_np.astype(sum_dtype, copy=False)
                    voxel_counts[sel] += 1

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
        "miou_valid": res.miou_valid,
    }


def train(cfg_path: str):
    cfg = load_yaml(cfg_path)

    run = cfg["run"]
    out_dir = Path(run["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dist_env = init_distributed()  # <-- move this UP before any writes

    dataset = str(cfg.get("data", {}).get("dataset", "eclair")).lower()
    eclair_run_cache_dir: Optional[Path] = None
    eclair_run_cache_base: Optional[Path] = None

    if dataset == "eclair":
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
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if dist_env.enabled:
        torch.distributed.barrier()  # ensure dirs exist before others proceed

    set_seed(int(run["seed"]) + (dist_env.rank if dist_env.enabled else 0))
    device = torch.device(f"cuda:{dist_env.local_rank}" if torch.cuda.is_available() else "cpu")

    amp = bool(run.get("amp", True))

    train_loader, val_loader, test_loader = build_dataloaders(cfg, dist_env, eclair_run_cache_dir=eclair_run_cache_dir)

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
        "lr",
        "train_loss",
        "val_loss",
        "val_miou",
        "val_macro_f1",
        "time_epoch_s",
        "best_val_miou",
    ]

    # BEV scalar metric (row currently writes it; add column so it actually appears)
    fields.append("bev_miou")

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

    # per-class columns
    for name in class_names:
        fields.append(f"val_iou_{name.replace(' ', '_').replace('.', '')}")
    for name in class_names:
        fields.append(f"val_f1_{name.replace(' ', '_').replace('.', '')}")

    if is_main_process(dist_env):
        logger = CSVLogger(out_dir / "metrics.csv", fields)
    else:

        class _NoOp:
            def log(self, *_args, **_kwargs):
                return

        logger = _NoOp()

    best_val_miou = -1.0
    global_step = 0
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

    for epoch in range(1, epochs + 1):

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

        if torch.cuda.is_available() and is_main_process(dist_env):
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
            print(f"[gpu] peak_alloc={peak_alloc:.2f} GB peak_reserved={peak_reserved:.2f} GB", flush=True)
            torch.cuda.reset_peak_memory_stats()

        cm_train = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index) if track_train_cm else None
        train_loss_sum = 0.0
        train_n_sum = 0
        # --- NEW: crop debug accumulators (DALES crop-mode only) ---
        dbg_n = 0
        dbg_vox_sum = 0.0
        dbg_rare_sum = 0.0
        dbg_crop_size_sum = 0.0
        dbg_shrink_sum = 0.0
        dbg_fallback_sum = 0.0
        dbg_rare_center_sum = 0.0
        dbg_resample_sum = 0.0
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
            with torch.autocast(device_type="cuda", enabled=amp):
                if is_main_process(dist_env) and global_step % 20 == 0:
                    print("batch meta_n_vox shape:", batch.get("meta_n_vox", None).shape if "meta_n_vox" in batch else None)

                logits, labels, bev_pred = _forward_batch(model, batch, device, is_train=True)
                seg_loss = criterion(logits, labels)
                # Always defined, even when BEV disabled
                total = seg_loss
                # Step-local BEV loss (prevents carry-over)
                bev_loss_step = None

                if bev_enabled and (bev_pred is not None) and ("bev_labels" in batch):
                    bev_targets = batch["bev_labels"]
                    # move to GPU
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
                    if bev_enabled and (bev_loss_step is not None):
                        # keep your existing warmup/weight logic (already correct)
                        bev_w = _bev_weight_for_epoch(bev_cfg, epoch)

                        if bool(bev_cfg.get("warmup_only_bev", False)) and (epoch <= int(bev_cfg.get("warmup_epochs", 0))):
                            total = bev_w * bev_loss_step
                        else:
                            total = seg_loss + bev_w * bev_loss_step

                loss_scaled = total / float(grad_accum)

            scaler.scale(loss_scaled).backward()

            if cm_train is not None:
                preds = logits.argmax(dim=1)
                cm_train.update(preds, labels)

            valid = labels != ignore_index
            n = int(valid.sum().item())
            train_loss_sum += float(total.item()) * max(1, n)
            train_n_sum += max(1, n)

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

            if step % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % int(run.get("log_every_steps", 50)) == 0 and is_main_process(dist_env):
                lr = optimizer.param_groups[0]["lr"]
                if bev_enabled:
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
        if (len(train_loader) % grad_accum) != 0:
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

        bev_metrics = None
        train_loss = train_loss_sum / max(1, train_n_sum)

        # ---- Eval ----
        do_eval = (epoch % int(run.get("eval_every_epochs", 1)) == 0) or (epoch == epochs)
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
            }

        # ---- Checkpointing ----
        is_best = do_eval and (val_metrics["miou"] > best_val_miou)
        if is_best:
            best_val_miou = float(val_metrics["miou"])

        save_every = int(run.get("save_every_epochs", 5))

        if is_main_process(dist_env) and (epoch % save_every == 0 or epoch == epochs or is_best):
            state = {
                "epoch": epoch,
                "model_state": unwrap_model(model).state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": (None if scheduler is None else scheduler.state_dict()),
                "scaler_state": scaler.state_dict(),
                "cfg": cfg,
                "best_val_miou": best_val_miou,
            }
            ckpt_dir = out_dir / "checkpoints"
            # atomic_save_torch(state, ckpt_dir / f"epoch_{epoch:03d}.pt")
            atomic_save_torch(state, ckpt_dir / "last.pt")
            if is_best:
                atomic_save_torch(state, ckpt_dir / "best.pt")

            # Optional periodic "epoch snapshots" (off by default)
            keep_epoch_snapshots = bool(run.get("keep_epoch_snapshots", False))
            if keep_epoch_snapshots and (epoch % save_every == 0 or epoch == epochs):
                atomic_save_torch(state, ckpt_dir / f"epoch_{epoch:03d}.pt")

        if dist_env.enabled:
            torch.distributed.barrier()  # optional but nice: sync after saving

        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_miou": val_metrics["miou"],
            "val_macro_f1": val_metrics["macro_f1"],
            "time_epoch_s": epoch_s,
            "best_val_miou": best_val_miou,
            "bev_miou": bev_metrics["bev_miou"] if bev_metrics is not None else float("nan"),
        }

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

        logger.log(row)

        if is_main_process(dist_env):
            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
                f"val_miou={val_metrics['miou']:.4f}"
                f"val_miou_valid={float(val_metrics.get('miou_valid', float('nan'))):.4f}"
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"best={best_val_miou:.4f} time={format_seconds(epoch_s)}"
            )

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

    # ---- Optional: cleanup epoch checkpoints to save disk ----
    if is_main_process(dist_env):
        import glob

        for p in glob.glob(str(out_dir / "checkpoints" / "epoch_*.pt")):
            try:
                Path(p).unlink()
            except FileNotFoundError:
                pass

    # ---- Cleanup run-scoped ECLAIR cache (prevents staleness across runs) ----
    if dataset == "eclair":
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
