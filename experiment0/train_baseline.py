import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import MinkowskiEngine as ME
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import EclairTiles, GenericLasFolder
from functional import compose_transforms_from_list
from losses import make_loss
from metrics import ConfusionMatrix, compute_scores, map_labels_tensor
from MinkowskiEngine import (MinkowskiAlgorithm, SparseTensorQuantizationMode,
                             TensorField)
from model import MinkUNet14C
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm


def set_seed(seed: int):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    voxel_size: float,
    num_classes_native: int,
    ignore_ids_native: List[int],
    common_mapping: Dict[int, int],
    num_common: int,
    important_native: List[int],
):
    """
    Returns a dict with native and common metrics and per-class IoUs.
    """
    model.eval()

    conf_native = ConfusionMatrix(num_classes_native)
    conf_common = ConfusionMatrix(num_common)

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        coords, feats, batch_idx = batch.pos / voxel_size, batch.x, batch.batch
        coords = torch.cat([batch_idx.unsqueeze(1), coords], dim=1)

        in_field = TensorField(
            features=feats.to(device),
            coordinates=coords.to(device),
            quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
        )
        sinput = in_field.sparse()
        soutput = model(sinput)
        logits = soutput.slice(in_field).F  # [N, C]

        # y = batch.classification.long().to(device)
        # mask = torch.ones_like(y, dtype=torch.bool)
        # for ig in ignore_ids_native:
        #     mask &= (y != ig)

        # if mask.sum() == 0:
        #     continue

        # logits = logits[mask]
        # y = y[mask]
        # pred_native = logits.argmax(dim=1)
        # print('DALES eval ranges:',
        # f'y_true[{int(y.min())},{int(y.max())}]',
        # f'pred[{int(pred_native.min())},{int(pred_native.max())}]')

        # pred_native = pred_native.clamp_(min=0, max=8)  # DALES has 9 classes
        # # native confusion
        # conf_native.update(pred_native.to('cpu'), y.to('cpu'))
        y = batch.classification.long().to(device)

        # For DALES: model outputs 12 classes (ECLAIR), but y is DALES native (0-8)
        # We need to ignore DALES class 0 (unknown)
        mask = torch.ones_like(y, dtype=torch.bool)
        for ig in ignore_ids_native:  # This will be [0] for DALES
            mask &= y != ig

        if mask.sum() == 0:
            continue

        logits = logits[mask]
        y = y[mask]
        pred_native = logits.argmax(
            dim=1
        )  # [N] with values in [0, 11] (ECLAIR classes)

        # IMPORTANT: Don't clamp predictions! Instead, we'll only use common-space metrics
        # The model was trained on ECLAIR (12 classes), so pred_native can be 0-11.
        # We'll map both pred and gt to common space for fair comparison.

        # Native confusion: This is NOT meaningful for DALES because model uses ECLAIR taxonomy.
        # We keep it for debugging but it will show mismatches.
        # Clamp ONLY for this native confusion matrix (not used in metrics):
        pred_native_clamped = pred_native.clamp(min=0, max=num_classes_native - 1)
        conf_native.update(pred_native_clamped.to("cpu"), y.to("cpu"))

        # map to common - USE UNCLAMPED PREDICTIONS
        pred_common = map_labels_tensor(pred_native, common_mapping)  # Map 12→8
        y_common = map_labels_tensor(y, common_mapping)  # Map 9→8

        # ignore id for common assumed 0
        cmask = y_common != 0
        if cmask.sum() > 0:
            conf_common.update(pred_common[cmask], y_common[cmask])

    native_scores = compute_scores(
        conf_native, ignore_ids=ignore_ids_native, label_set=important_native
    )
    common_scores = compute_scores(conf_common, ignore_ids=[0])

    return {
        "native": native_scores,
        "common": common_scores,
    }


def save_metrics_csv(
    path: Path, epoch: int, tag: str, results: Dict, metadata: Optional[Dict] = None
):
    """
    Save metrics to CSV with experimental metadata.

    Args:
        metadata: dict with keys like loss_name, voxel_size, focal_gamma, etc.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Expanded header with metadata
    header = [
        "epoch",
        "tag",
        "loss_name",
        "voxel_size",
        "focal_gamma",
        "dice_smooth",  # NEW
        "mIoU_native",
        "macroF1_native",
        "mIoU_common",
        "macroF1_common",
        # NEW: Per-class metrics for key utility classes
        "wires_IoU_common",
        "wires_recall_common",
        "wires_precision_common",
        "poles_IoU_common",
        "poles_recall_common",
        "poles_precision_common",
    ]

    # Extract utility class metrics (common space: 4=Wires, 5=Poles)
    wires_iou = (
        results["common"]["IoU_per_class"][4]
        if len(results["common"]["IoU_per_class"]) > 4
        else 0.0
    )
    wires_rec = (
        results["common"]["Recall_per_class"][4]
        if len(results["common"]["Recall_per_class"]) > 4
        else 0.0
    )
    wires_prec = (
        results["common"]["Precision_per_class"][4]
        if len(results["common"]["Precision_per_class"]) > 4
        else 0.0
    )

    poles_iou = (
        results["common"]["IoU_per_class"][5]
        if len(results["common"]["IoU_per_class"]) > 5
        else 0.0
    )
    poles_rec = (
        results["common"]["Recall_per_class"][5]
        if len(results["common"]["Recall_per_class"]) > 5
        else 0.0
    )
    poles_prec = (
        results["common"]["Precision_per_class"][5]
        if len(results["common"]["Precision_per_class"]) > 5
        else 0.0
    )

    row = [
        epoch,
        tag,
        metadata.get("loss_name", "") if metadata else "",
        metadata.get("voxel_size", "") if metadata else "",
        metadata.get("focal_gamma", "") if metadata else "",
        metadata.get("dice_smooth", "") if metadata else "",
        results["native"]["mIoU"],
        results["native"]["macroF1"],
        results["common"]["mIoU"],
        results["common"]["macroF1"],
        wires_iou,
        wires_rec,
        wires_prec,
        poles_iou,
        poles_rec,
        poles_prec,
    ]

    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)


def validate_config_with_data(cfg, dataset, config_file):
    """
    Pre-flight validation: ensure config matches actual data dimensions.
    Catches mismatches BEFORE wasting GPU hours.
    """
    print("Running pre-flight config validation...")

    # Get one sample
    sample = dataset[0]
    if cfg.train_transforms:
        transforms = compose_transforms_from_list(cfg.train_transforms)
        sample = transforms(sample)

    expected_dim = cfg.num_features
    actual_dim = sample.x.shape[1]

    if actual_dim != expected_dim:
        # Provide detailed breakdown
        feature_breakdown = []
        expected_breakdown = []

        if "intensity" in cfg.feature_names:
            expected_breakdown.append("intensity: 1")
        if "return_number" in cfg.feature_names:
            expected_breakdown.append("return_number: 5 (one-hot)")
        if "number_of_returns" in cfg.feature_names:
            expected_breakdown.append("number_of_returns: 5 (one-hot)")
        if "coordinates" in cfg.feature_names:
            expected_breakdown.append("coordinates: 3")
        if "colors" in cfg.feature_names:
            expected_breakdown.append("colors: 3")

        expected_sum = sum(
            [
                1 if "intensity" in cfg.feature_names else 0,
                5 if "return_number" in cfg.feature_names else 0,
                5 if "number_of_returns" in cfg.feature_names else 0,
                3 if "coordinates" in cfg.feature_names else 0,
                3 if "colors" in cfg.feature_names else 0,
            ]
        )

        raise ValueError(
            f"\n{'='*70}\n"
            f"FATAL: Feature dimension mismatch!\n"
            f"{'='*70}\n"
            f"  Config expects: {expected_dim} features (num_features in {config_file})\n"
            f"  Data produces:  {actual_dim} features\n"
            f"\n"
            f"  feature_names in config: {cfg.feature_names}\n"
            f"  Expected breakdown:\n"
            f"    {chr(10).join('    ' + x for x in expected_breakdown)}\n"
            f"  Expected sum: {expected_sum}\n"
            f"\n"
            f"  ACTION REQUIRED:\n"
            f"    Update 'num_features: {actual_dim}' in {config_file}\n"
            f"{'='*70}\n"
        )

    print(f"✓ Config validation passed: {actual_dim} features")
    print(f"  Features: {cfg.feature_names}")
    print(f"  Voxel size: {cfg.voxel_size}")
    print()


def train(
    eclair_dir: str,
    dales_dir: str,
    output_dir: str = "runs/E0",
    config_file: str = "./configs/train_e0.yaml",
    loss_name: str = "ce",
    dice_smooth: float = 1.0,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[List[float]] = None,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    amp: bool = True,
    seed: int = 1984,
    eval_every: int = 5,
    num_workers: int = 4,
    voxel_size: Optional[float] = None,
):
    cfg = OmegaConf.load(config_file)
    set_seed(seed)
    # Allow voxel size override from CLI
    if voxel_size is not None:
        cfg.voxel_size = voxel_size
        print(f"✓ Overriding voxel_size: {cfg.voxel_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------- datasets & loaders ----------------------
    # ECLAIR splits from labels.json
    eclair = EclairTiles(
        root=eclair_dir,
        split="train",
        transforms=compose_transforms_from_list(cfg.train_transforms),
    )
    validate_config_with_data(cfg, eclair, config_file)
    eclair_val = EclairTiles(
        root=eclair_dir,
        split="val",
        transforms=compose_transforms_from_list(cfg.eval_transforms),
    )

    # DALES test (zero-shot). We simply glob LAS/LAZ files inside provided folder.
    dales_test = GenericLasFolder(
        root=dales_dir, transforms=compose_transforms_from_list(cfg.eval_transforms)
    )

    train_loader = PyGDataLoader(
        eclair,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = PyGDataLoader(
        eclair_val,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    dales_loader = PyGDataLoader(
        dales_test,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ---------------------- model & loss ----------------------------
    model = MinkUNet14C(cfg.num_features, cfg.num_classes_native).to(device)

    # Load class statistics for Class-Balanced Loss (if needed)
    samples_per_cls = None
    if loss_name.lower() in ["class_balanced", "cb"]:
        class_stats_path = Path("./configs/eclair_class_counts.yaml")
        if not class_stats_path.exists():
            raise FileNotFoundError(
                f"Class statistics not found at {class_stats_path}. "
                f"Run: python compute_class_stats.py --eclair_dir {eclair_dir}"
            )
        class_stats = OmegaConf.load(class_stats_path)
        samples_per_cls = class_stats.samples_per_cls
        print(f"✓ Loaded class statistics: {len(samples_per_cls)} classes")

    loss_fn = make_loss(
        name=loss_name,
        num_classes=cfg.num_classes_native,
        ignore_index=0,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        dice_smooth=dice_smooth,
        samples_per_cls=samples_per_cls,  # NEW
        cb_beta=0.9999,  # NEW - can be made a CLI arg if needed
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # convenience shorthands
    voxel_size = cfg.voxel_size
    ignore_ids_native = cfg.ignore_ids_native
    important_native = cfg.important_native

    # common mapping dicts
    eclair2common: Dict[int, int] = OmegaConf.to_container(
        OmegaConf.load(cfg.mapping_eclair_to_common), resolve=True
    )
    dales2common: Dict[int, int] = OmegaConf.to_container(
        OmegaConf.load(cfg.mapping_dales_to_common), resolve=True
    )
    num_common = cfg.num_classes_common

    out_dir = Path(output_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"

    best_val_mIoU_common = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_points = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            coords, feats, batch_idx = batch.pos / voxel_size, batch.x, batch.batch
            if feats.shape[1] != cfg.num_features:
                raise ValueError(
                    f"Feature dim mismatch: got {feats.shape[1]} but cfg.num_features={cfg.num_features}. "
                    f"Check feature_names in {config_file} and NormalizeFeatures."
                )
            coords = torch.cat([batch_idx.unsqueeze(1), coords], dim=1)

            in_field = TensorField(
                features=feats.to(device),
                coordinates=coords.to(device),
                quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
            )
            sinput = in_field.sparse()

            with torch.cuda.amp.autocast(enabled=amp):
                logits_sparse = model(sinput)  # [M_sparse, C]
                logits = logits_sparse.slice(in_field).F  # [N_points, C]
                y = batch.classification.long().to(device)
                # mask out ignored ids (0 == Undefined) for loss
                mask = torch.ones_like(y, dtype=torch.bool)
                for ig in ignore_ids_native:
                    mask &= y != ig
                if mask.sum() == 0:
                    continue
                loss = loss_fn(logits[mask], y[mask])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * int(mask.sum())
            n_points += int(mask.sum())
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_points)

        # periodic evaluation
        if epoch % eval_every == 0 or epoch == epochs:
            # ECLAIR-val (native + common via eclair2common)
            val_results = evaluate_split(
                model,
                val_loader,
                device,
                voxel_size,
                cfg.num_classes_native,
                ignore_ids_native,
                common_mapping=eclair2common,
                num_common=num_common,
                important_native=important_native,
            )
            # DALES-test (common via dales2common)
            dales_results = evaluate_split(
                model,
                dales_loader,
                device,
                voxel_size,
                cfg.num_classes_dales_native,
                ignore_ids_native=cfg.ignore_ids_dales,
                common_mapping=dales2common,
                num_common=num_common,
                important_native=list(range(cfg.num_classes_dales_native)),
            )

            # compute domain gap on common set
            delta_miou = val_results["common"]["mIoU"] - dales_results["common"]["mIoU"]

            # logging
            exp_metadata = {
                "loss_name": loss_name,
                "voxel_size": cfg.voxel_size,
                "focal_gamma": focal_gamma,
                "dice_smooth": dice_smooth,
            }
            save_metrics_csv(
                metrics_csv, epoch, "ECLAIR_val", val_results, exp_metadata
            )
            save_metrics_csv(
                metrics_csv, epoch, "DALES_test", dales_results, exp_metadata
            )
            # Enhanced logging with per-class metrics for rare utility classes
            with open(out_dir / "last_eval.txt", "w") as f:
                f.write(f"Epoch {epoch}: loss={avg_loss:.6f}\n")
                f.write("=" * 70 + "\n")

                # ECLAIR validation (native taxonomy)
                f.write("ECLAIR Validation (Native Taxonomy):\n")
                f.write(
                    f"  Overall: mIoU={val_results['native']['mIoU']:.4f} | macroF1={val_results['native']['macroF1']:.4f}\n"
                )
                f.write("  Utility Classes (ECLAIR-specific):\n")

                # Indices: 6=Trans Wires, 7=Dist Wires, 8=Poles, 9=Towers
                for idx, name in [
                    (6, "Trans Wires"),
                    (7, "Dist Wires"),
                    (8, "Poles"),
                    (9, "Towers"),
                ]:
                    iou = val_results["native"]["IoU_per_class"][idx]
                    rec = val_results["native"]["Recall_per_class"][idx]
                    prec = val_results["native"]["Precision_per_class"][idx]
                    f1 = val_results["native"]["F1_per_class"][idx]
                    f.write(
                        f"    {name:15s}: IoU={iou:.3f} | Rec={rec:.3f} | Prec={prec:.3f} | F1={f1:.3f}\n"
                    )

                f.write("\n")

                # ECLAIR validation (common taxonomy)
                f.write("ECLAIR Validation (Common Taxonomy):\n")
                f.write(
                    f"  Overall: mIoU={val_results['common']['mIoU']:.4f} | macroF1={val_results['common']['macroF1']:.4f}\n"
                )
                f.write("  Common Classes:\n")

                # Common space: 4=Wires, 5=Poles (Towers merged into Poles)
                common_names = [
                    "Ignore",
                    "Ground",
                    "Vegetation",
                    "Buildings",
                    "Wires",
                    "Poles",
                    "Fence",
                    "Vehicle",
                ]
                for idx in [4, 5]:  # Wires, Poles
                    name = common_names[idx]
                    iou = val_results["common"]["IoU_per_class"][idx]
                    rec = val_results["common"]["Recall_per_class"][idx]
                    prec = val_results["common"]["Precision_per_class"][idx]
                    f1 = val_results["common"]["F1_per_class"][idx]
                    f.write(
                        f"    {name:15s}: IoU={iou:.3f} | Rec={rec:.3f} | Prec={prec:.3f} | F1={f1:.3f}\n"
                    )

                f.write("\n")
                f.write("=" * 70 + "\n")

                # DALES test (common taxonomy only)
                f.write("DALES Test (Common Taxonomy):\n")
                f.write(
                    f"  Overall: mIoU={dales_results['common']['mIoU']:.4f} | macroF1={dales_results['common']['macroF1']:.4f}\n"
                )
                f.write("  Domain Gap:\n")
                f.write(f"    ΔmIoU (ECLAIR→DALES) = {delta_miou:.4f}\n")
                f.write("\n")

                # Per-class domain gap for utility classes
                f.write("  Per-Class Domain Gap (Common Space):\n")
                for idx in [4, 5]:  # Wires, Poles
                    name = common_names[idx]
                    eclair_iou = val_results["common"]["IoU_per_class"][idx]
                    dales_iou = dales_results["common"]["IoU_per_class"][idx]
                    gap = eclair_iou - dales_iou
                    f.write(
                        f"    {name:15s}: ECLAIR={eclair_iou:.3f} | DALES={dales_iou:.3f} | Δ={gap:+.3f}\n"
                    )

                f.write("=" * 70 + "\n")

            # checkpoint by best common mIoU on ECLAIR-val
            if val_results["common"]["mIoU"] > best_val_mIoU_common:
                best_val_mIoU_common = val_results["common"]["mIoU"]
                torch.save(model.state_dict(), out_dir / "checkpoints/best.pth")

            # always save last
            torch.save(model.state_dict(), out_dir / "checkpoints/last.pth")

    print("Training finished. Metrics saved to:", metrics_csv)


if __name__ == "__main__":
    import fire

    fire.Fire(train)
