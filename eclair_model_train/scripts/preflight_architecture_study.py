#!/usr/bin/env python3
"""Validate the frozen recipe and exercise a real GPU batch before long training."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config_loader import load_yaml
from src.study_architecture import ARMS, block_counts
from src.study_runtime import atomic_json, storage_guard


def validate(cfg, frozen):
    if not cfg.get("study", {}).get("enabled"):
        raise ValueError("Expected an enabled study")
    if cfg["study"].get("version") != 1:
        raise ValueError("Unsupported study contract version")
    arm = cfg["study"]["arm"]
    if arm not in ARMS or block_counts(cfg) != ARMS[arm]:
        raise ValueError("Arm and layer counts disagree")
    for key in ("data", "loss", "optim", "sched", "eval", "eclair", "epochs", "method", "ocons"):
        if cfg.get(key) != frozen.get(key):
            raise ValueError(f"Frozen baseline differs at {key}")
    model = copy.deepcopy(cfg["model"])
    model.pop("layers", None)
    if model != frozen["model"]:
        raise ValueError("Only model.layers may change")
    for key in ("seed", "amp", "eval_every_epochs"):
        if cfg["run"].get(key) != frozen["run"].get(key):
            raise ValueError(f"Frozen run setting differs: {key}")
    if cfg["data"]["dataset"] != "dales" or cfg["data"]["sampling"]["crop_size_xy_m"] != 120:
        raise ValueError("This study requires the canonical 120 m DALES recipe")
    for key in ("max_train_batches_per_epoch", "skip_validation", "skip_final_test"):
        if cfg["run"].get(key):
            raise ValueError(f"Diagnostic override forbidden: {key}")
    if cfg["run"].get("keep_epoch_snapshots"):
        raise ValueError("Study forbids accumulating checkpoint snapshots")
    return {"arm": arm, "layers": list(block_counts(cfg)), "frozen_contract": "matched"}


def gpu_smoke(cfg):
    import torch
    from src.dist import DistEnv
    from src.model import build_model
    from src.utils import atomic_save_torch
    from train import _forward_batch, build_dataloaders, build_loss, build_optimizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; static validation is not a GPU preflight")
    device = torch.device("cuda:0")
    before_rng = torch.get_rng_state().clone()
    before_cuda_rng = torch.cuda.get_rng_state(device).clone()
    model = build_model(int(cfg["model"]["in_channels"]), int(cfg["model"]["out_channels"]), cfg=cfg)
    if not torch.equal(before_rng, torch.get_rng_state()):
        raise RuntimeError("Study model initialization advanced the data RNG")
    if not torch.equal(before_cuda_rng, torch.cuda.get_rng_state(device)):
        raise RuntimeError("Study CPU initialization advanced the CUDA RNG")
    state_hashes = {}
    import hashlib

    for name, value in model.state_dict().items():
        state_hashes[name] = hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
    # All shared weights must match across all arms, and extra block tensors must
    # match each other. This CPU-only part runs inside the GPU allocation.
    extras = {}
    for arm in ARMS:
        variant_cfg = copy.deepcopy(cfg)
        variant_cfg["model"]["layers"] = list(ARMS[arm])
        variant = build_model(
            int(cfg["model"]["in_channels"]), int(cfg["model"]["out_channels"]), cfg=variant_cfg
        )
        for key, value in variant.state_dict().items():
            digest = hashlib.sha256(value.numpy().tobytes()).hexdigest()
            if (
                key in state_hashes
                and not key.startswith(("block3.1", "block8.1"))
                and state_hashes[key] != digest
            ):
                raise RuntimeError(f"Unmatched shared initialization: {arm}/{key}")
            if key.startswith(("block3.1", "block8.1")):
                canonical = key.split(".", 1)[1]
                if canonical in extras and extras[canonical] != digest:
                    raise RuntimeError(f"Extra-block initialization mismatch at {key}")
                extras[canonical] = digest
        del variant
    model = model.to(device).train()
    loader, _, _ = build_dataloaders(cfg, DistEnv(False))
    loader.dataset.set_epoch(1)
    loader.generator.manual_seed(int(cfg["run"]["seed"]) + 1009)
    batch = next(iter(loader))
    optimizer = build_optimizer(cfg, model)
    criterion = build_loss(cfg).to(device)
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(1)
    amp = bool(cfg["run"]["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    with torch.autocast(device_type="cuda", enabled=amp):
        model._study_capture = True
        logits, labels, _ = _forward_batch(model, batch, device, is_train=True)
        loss = criterion(logits, labels)
    # Exercise the actual telemetry path without polluting the real run's logs.
    from src.study_runtime import StudyRuntime

    with tempfile.TemporaryDirectory(prefix="study_audit_", dir=cfg["run"]["out_dir"]) as tmp:
        runtime = StudyRuntime(cfg, DistEnv(False), device, tmp)
        runtime.begin_epoch(1, loader)
        runtime.audit_loss(logits, labels, criterion, model, 1)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Diagnostic audit changed training .grad buffers")
        rng_state = runtime.state()
        expected = torch.rand(4, device=device)
        runtime.restore([rng_state], 0)
        if not torch.equal(expected, torch.rand(4, device=device)):
            raise RuntimeError("CUDA RNG restoration failed")
    model._study_capture = False
    model._study_classifier_features = None
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite real-batch smoke loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"Nonfinite smoke gradient: {name}")
    scaler.step(optimizer)
    scaler.update()
    with tempfile.TemporaryDirectory(prefix="study_smoke_", dir=cfg["run"]["out_dir"]) as tmp:
        path = Path(tmp) / "roundtrip.pt"
        estimate = sum(v.numel() * v.element_size() for v in model.state_dict().values()) * 4
        storage_guard(Path(tmp), estimate, cfg["study"]["storage"])
        atomic_save_torch(
            {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, path
        )
        saved_bytes = path.stat().st_size
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        restored = build_model(int(cfg["model"]["in_channels"]), int(cfg["model"]["out_channels"]), cfg=cfg)
        restored.load_state_dict(checkpoint["model_state"], strict=True)
        restored_optimizer = build_optimizer(cfg, restored)
        restored_optimizer.load_state_dict(checkpoint["optimizer_state"])
        for key, value in model.state_dict().items():
            if not torch.equal(value.cpu(), restored.state_dict()[key]):
                raise RuntimeError(f"Checkpoint roundtrip mismatch at {key}")
    return {
        "status": "passed",
        "batch_voxels": len(labels),
        "loss": float(loss.detach()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_tensor_hashes": state_hashes,
        "resume_checkpoint_bytes": saved_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    result = validate(cfg, load_yaml("configs/m0_dales_frozen_29364.yaml"))
    if args.gpu_smoke:
        result["gpu_smoke"] = gpu_smoke(cfg)
    result["status"] = "passed_gpu_smoke" if args.gpu_smoke else "static_contract_only"
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
