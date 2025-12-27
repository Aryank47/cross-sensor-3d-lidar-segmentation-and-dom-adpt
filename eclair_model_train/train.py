#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
from src.augment import AugmentConfig
from src.config_loader import load_yaml
from src.data_eclair import EclairTiles, PatchConfig, minkowski_collate_fn
from src.dist import DistEnv, all_reduce_sum, init_distributed, is_main_process
from src.features import FeatureConfig
from src.label_maps import ECLAIR_CLASS_NAMES_11
from src.losses import FocalLoss, FocalLossConfig
from src.metrics import ConfusionMatrix
from src.model import build_model
from src.utils import (
    CSVLogger,
    atomic_save_torch,
    format_seconds,
    save_json,
    set_seed,
    unwrap_model,
)
from torch.utils.data import DataLoader


def build_dataloaders(cfg: Dict[str, Any], dist_env: DistEnv):
    data = cfg["data"]
    eclair_cfg = cfg.get("eclair", {})
    meta_filename = eclair_cfg.get("meta_filename", "labels.json")
    train_cats = eclair_cfg.get("train_review_categories", None)  # None = no filtering
    val_cats = eclair_cfg.get("val_review_categories", ["approved"])
    test_cats = eclair_cfg.get("test_review_categories", ["approved"])
    patch_cfg = PatchConfig(**data["patch"])
    aug_cfg = AugmentConfig(**data["aug"])
    feat_cfg = FeatureConfig(**data["features"])

    ls = data["label_space"]
    ignore_index = int(ls["ignore_index"])
    undefined_id = int(ls["eclair_undefined_id"])

    train_ds = EclairTiles(
        eclair_root=data["eclair_root"],
        split="train",
        is_train=True,
        patch_cfg=patch_cfg,
        aug_cfg=aug_cfg,
        feat_cfg=feat_cfg,
        ignore_index=ignore_index,
        undefined_id=undefined_id,
        use_cache=bool(data.get("use_cache", True)),
        cache_root=data.get("cache_root", None),
        seed=int(cfg["run"]["seed"]),
        meta_filename=meta_filename,
        allowed_review_categories=train_cats,
    )
    val_ds = EclairTiles(
        eclair_root=data["eclair_root"],
        split="val",
        is_train=False,
        patch_cfg=patch_cfg,
        aug_cfg=aug_cfg,  # aug_cfg ignored when is_train=False
        feat_cfg=feat_cfg,
        ignore_index=ignore_index,
        undefined_id=undefined_id,
        use_cache=bool(data.get("use_cache", True)),
        cache_root=data.get("cache_root", None),
        seed=int(cfg["run"]["seed"]) + 1,
        meta_filename=meta_filename,
        allowed_review_categories=val_cats,
    )
    test_ds = EclairTiles(
        eclair_root=data["eclair_root"],
        split="test",
        is_train=False,
        patch_cfg=patch_cfg,
        aug_cfg=aug_cfg,
        feat_cfg=feat_cfg,
        ignore_index=ignore_index,
        undefined_id=undefined_id,
        use_cache=bool(data.get("use_cache", True)),
        cache_root=data.get("cache_root", None),
        seed=int(cfg["run"]["seed"]) + 2,
        meta_filename=meta_filename,
        allowed_review_categories=test_cats,
    )

    # Distributed samplers (optional)
    if dist_env.enabled:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_ds, shuffle=False
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_ds, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    dl_kwargs = dict(
        batch_size=int(data["batch_size"]),
        num_workers=int(data["num_workers"]),
        pin_memory=True,
        collate_fn=minkowski_collate_fn,
        persistent_workers=int(data["num_workers"]) > 0,
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, sampler=val_sampler, drop_last=False, **dl_kwargs
    )
    test_loader = DataLoader(
        test_ds, shuffle=False, sampler=test_sampler, drop_last=False, **dl_kwargs
    )
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
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    raise ValueError(f"Unknown scheduler: {name}")


def build_loss(cfg: Dict[str, Any]):
    loss_cfg = cfg["loss"]
    name = loss_cfg["name"].lower()
    ignore_index = int(cfg["data"]["label_space"]["ignore_index"])
    if name == "focal":
        fl_cfg = FocalLossConfig(
            gamma=float(loss_cfg.get("gamma", 2.0)),
            alpha=None,
            ignore_index=ignore_index,
        )
        return FocalLoss(fl_cfg)
    raise ValueError(f"Unknown loss: {name}")


def _forward_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = batch["coords"].to(device, non_blocking=True)
    feats = batch["feats"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    st = ME.SparseTensor(feats, coordinates=coords, device=device)
    out = model(st)  # SparseTensor
    logits = out.F  # [N, C]
    return logits, labels


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
            logits, labels = _forward_batch(model, batch, device)
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
    }


def train(cfg_path: str):
    cfg = load_yaml(cfg_path)

    run = cfg["run"]
    out_dir = Path(run["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dist_env = init_distributed()  # <-- move this UP before any writes

    # Only rank0 writes files
    if is_main_process(dist_env):
        save_json(out_dir / "config_resolved.json", cfg)
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if dist_env.enabled:
        torch.distributed.barrier()  # ensure dirs exist before others proceed

    set_seed(int(run["seed"]) + (dist_env.rank if dist_env.enabled else 0))
    device = torch.device(
        f"cuda:{dist_env.local_rank}" if torch.cuda.is_available() else "cpu"
    )

    amp = bool(run.get("amp", True))

    train_loader, val_loader, test_loader = build_dataloaders(cfg, dist_env)

    model_cfg = cfg["model"]
    model = build_model(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        D=int(model_cfg.get("D", 3)),
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

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    num_classes = int(cfg["data"]["label_space"]["num_classes"])
    ignore_index = int(cfg["data"]["label_space"]["ignore_index"])
    grad_accum = int(cfg["data"].get("grad_accum_steps", 1))
    epochs = int(cfg.get("epochs", 80)) if "epochs" in cfg else 80

    # Logger fields
    class_names = ECLAIR_CLASS_NAMES_11
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

    for epoch in range(1, epochs + 1):
        model.train()
        if dist_env.enabled:
            # ensures shuffling differs each epoch
            train_loader.sampler.set_epoch(epoch)
        cm_train = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
        train_loss_sum = 0.0
        train_n_sum = 0
        optimizer.zero_grad(set_to_none=True)

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

        for step, batch in enumerate(train_loader, start=1):
            with torch.autocast(device_type="cuda", enabled=amp):
                logits, labels = _forward_batch(model, batch, device)
                loss = criterion(logits, labels)
                loss_scaled = loss / float(grad_accum)

            scaler.scale(loss_scaled).backward()

            preds = logits.argmax(dim=1)
            cm_train.update(preds, labels)

            valid = labels != ignore_index
            n = int(valid.sum().item())
            train_loss_sum += float(loss.item()) * max(1, n)
            train_n_sum += max(1, n)

            if step % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % int(run.get("log_every_steps", 50)) == 0:
                lr = optimizer.param_groups[0]["lr"]
                if is_main_process(dist_env):
                    print(
                        f"[epoch {epoch:03d} step {step:05d}] loss={loss.item():.4f} lr={lr:.2e}"
                    )

        # flush leftover grads if dataloader size not divisible by grad_accum
        if (len(train_loader) % grad_accum) != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

        t1.record()
        torch.cuda.synchronize()
        epoch_ms = t0.elapsed_time(t1)
        epoch_s = epoch_ms / 1000.0

        train_loss = train_loss_sum / max(1, train_n_sum)

        # ---- Eval ----
        do_eval = (epoch % int(run.get("eval_every_epochs", 1)) == 0) or (
            epoch == epochs
        )
        if do_eval:
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
        else:
            val_metrics = {
                "loss": float("nan"),
                "miou": float("nan"),
                "macro_f1": float("nan"),
                "per_class_iou": [float("nan")] * num_classes,
                "per_class_f1": [float("nan")] * num_classes,
            }

        # ---- Checkpointing ----
        is_best = do_eval and (val_metrics["miou"] > best_val_miou)
        if is_best:
            best_val_miou = float(val_metrics["miou"])

        save_every = int(run.get("save_every_epochs", 5))

        if is_main_process(dist_env) and (
            epoch % save_every == 0 or epoch == epochs or is_best
        ):
            state = {
                "epoch": epoch,
                "model_state": unwrap_model(model).state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": (
                    None if scheduler is None else scheduler.state_dict()
                ),
                "scaler_state": scaler.state_dict(),
                "cfg": cfg,
                "best_val_miou": best_val_miou,
            }
            ckpt_dir = out_dir / "checkpoints"
            atomic_save_torch(state, ckpt_dir / f"epoch_{epoch:03d}.pt")
            if is_best:
                atomic_save_torch(state, ckpt_dir / "best.pt")
            atomic_save_torch(state, ckpt_dir / "last.pt")

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
        }
        for name, v in zip(class_names, val_metrics["per_class_iou"]):
            row[f"val_iou_{name.replace(' ', '_').replace('.', '')}"] = v
        for name, v in zip(class_names, val_metrics["per_class_f1"]):
            row[f"val_f1_{name.replace(' ', '_').replace('.', '')}"] = v

        logger.log(row)

        if is_main_process(dist_env):
            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
                f"val_miou={val_metrics['miou']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"best={best_val_miou:.4f} time={format_seconds(epoch_s)}"
            )

    # Final test evaluation on best model
    best_ckpt = torch.load(out_dir / "checkpoints" / "best.pt", map_location="cpu")
    unwrap_model(model).load_state_dict(best_ckpt["model_state"])
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
            f"[TEST] loss={test_metrics['loss']:.4f} mIoU={test_metrics['miou']:.4f} macroF1={test_metrics['macro_f1']:.4f}"
        )


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
