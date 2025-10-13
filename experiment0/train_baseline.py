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

        y = batch.classification.long().to(device)
        mask = torch.ones_like(y, dtype=torch.bool)
        for ig in ignore_ids_native:
            mask &= (y != ig)

        if mask.sum() == 0:
            continue

        logits = logits[mask]
        y = y[mask]
        pred_native = logits.argmax(dim=1)
        print('DALES eval ranges:',
        f'y_true[{int(y.min())},{int(y.max())}]',
        f'pred[{int(pred_native.min())},{int(pred_native.max())}]')

        pred_native = pred_native.clamp_(min=0, max=8)  # DALES has 9 classes
        # native confusion
        conf_native.update(pred_native.to('cpu'), y.to('cpu'))

        # map to common
        pred_common = map_labels_tensor(pred_native, common_mapping)
        y_common = map_labels_tensor(y, common_mapping)
        # ignore id for common assumed 0
        cmask = y_common != 0
        if cmask.sum() > 0:
            conf_common.update(pred_common[cmask], y_common[cmask])

    native_scores = compute_scores(conf_native, ignore_ids=ignore_ids_native, label_set=important_native)
    common_scores = compute_scores(conf_common, ignore_ids=[0])

    return {
        "native": native_scores,
        "common": common_scores,
    }


def save_metrics_csv(path: Path, epoch: int, tag: str, results: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "epoch",
        "tag",
        "mIoU_native",
        "macroF1_native",
        "mIoU_common",
        "macroF1_common",
    ]
    row = [
        epoch,
        tag,
        results["native"]["mIoU"],
        results["native"]["macroF1"],
        results["common"]["mIoU"],
        results["common"]["macroF1"],
    ]
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)


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
):
    cfg = OmegaConf.load(config_file)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------- datasets & loaders ----------------------
    # ECLAIR splits from labels.json
    eclair = EclairTiles(root=eclair_dir, split="train", transforms=compose_transforms_from_list(cfg.train_transforms))
    eclair_val = EclairTiles(root=eclair_dir, split="val", transforms=compose_transforms_from_list(cfg.eval_transforms))

    # DALES test (zero-shot). We simply glob LAS/LAZ files inside provided folder.
    dales_test = GenericLasFolder(root=dales_dir, transforms=compose_transforms_from_list(cfg.eval_transforms))

    train_loader = PyGDataLoader(eclair, batch_size=cfg.train_batch_size, shuffle=True, num_workers=num_workers,
                                 pin_memory=True)
    val_loader = PyGDataLoader(eclair_val, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=num_workers,
                               pin_memory=True)
    dales_loader = PyGDataLoader(dales_test, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=True)

    # ---------------------- model & loss ----------------------------
    model = MinkUNet14C(cfg.num_features, cfg.num_classes_native).to(device)

    loss_fn = make_loss(
        name=loss_name,
        num_classes=cfg.num_classes_native,
        ignore_index=0,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        dice_smooth=dice_smooth,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # convenience shorthands
    voxel_size = cfg.voxel_size
    ignore_ids_native = cfg.ignore_ids_native
    important_native = cfg.important_native

    # common mapping dicts
    eclair2common: Dict[int, int] = OmegaConf.to_container(OmegaConf.load(cfg.mapping_eclair_to_common), resolve=True)
    dales2common: Dict[int, int] = OmegaConf.to_container(OmegaConf.load(cfg.mapping_dales_to_common), resolve=True)
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
                    mask &= (y != ig)
                if mask.sum() == 0:
                    continue
                loss = loss_fn(logits[mask], y[mask])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * int(mask.sum())
            n_points += int(mask.sum())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

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
            save_metrics_csv(metrics_csv, epoch, "ECLAIR_val", val_results)
            save_metrics_csv(metrics_csv, epoch, "DALES_test", dales_results)
            with open(out_dir / "last_eval.txt", "w") as f:
                f.write(
                    f"Epoch {epoch}: loss={avg_loss:.6f}\n"
                    f"ECLAIR-val mIoU(native)={val_results['native']['mIoU']:.4f} macroF1(native)={val_results['native']['macroF1']:.4f}\n"
                    f"ECLAIR-val mIoU(common)={val_results['common']['mIoU']:.4f} macroF1(common)={val_results['common']['macroF1']:.4f}\n"
                    f"DALES-test mIoU(common)={dales_results['common']['mIoU']:.4f} macroF1(common)={dales_results['common']['macroF1']:.4f}\n"
                    f"Delta mIoU (ECLAIR val common - DALES test common) = {delta_miou:.4f}\n"
                )

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



