#!/usr/bin/env python3
"""Source-only cache/feature parity and a real GPU step before full training."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.submit_feature_contract_study import load_frozen
from src.config_loader import load_yaml
from src.feature_contract_study import ARMS, make_config, model_digest, validate_config
from src.features import assert_checkpoint_feature_config
from src.study_runtime import atomic_json, storage_guard


def gpu_preflight(cfg, frozen):
    import numpy as np
    import torch
    from src.dist import DistEnv
    from src.model import build_model
    from src.study_runtime import StudyRuntime
    from src.utils import atomic_save_torch
    from train import _forward_batch, build_dataloaders, build_loss, build_optimizer

    if not torch.cuda.is_available():
        raise RuntimeError("A compute-node CUDA preflight is required")
    initial = {}
    before = torch.get_rng_state().clone()
    for arm in ARMS:
        other = make_config(frozen, arm, cfg["run"]["out_dir"])
        model = build_model(10, 11, cfg=other)
        initial[arm] = model_digest(model.state_dict())
        del model
    if len(set(initial.values())) != 1 or not torch.equal(
        before, torch.get_rng_state()
    ):
        raise RuntimeError(
            "Unmatched initialization or model setup consumed the data RNG"
        )
    # Worker count is an explicitly recorded smoke-only override, not a training change.
    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg["data"]["num_workers"] = 0
    loader, _, _ = build_dataloaders(
        smoke_cfg, DistEnv(False), eclair_run_cache_dir=None
    )
    ds = loader.dataset
    ds.set_epoch(1)
    parity = []
    for index in sorted({0, len(ds) // 2, len(ds) - 1}):
        name = ds.names[index]
        raw = ds._load_raw_base(name)
        with patch.object(ds, "run_cache_precompute_returns_onehot", True):
            invariant = ds._materialize_invariant(raw)
        versions = {}
        original_feature_cfg = ds.feat_cfg
        try:
            for mode in ("observed", "constant_pair"):
                ds.feat_cfg = replace(original_feature_cfg, return_input_mode=mode)
                with patch.object(ds, "_load_raw", return_value=raw):
                    direct = ds[index]
                with patch.object(ds, "_load_raw", return_value=invariant):
                    cached = ds[index]
                for key in ("coords", "labels", "feats"):
                    if not torch.equal(direct[key], cached[key]):
                        raise RuntimeError(
                            f"Disabling duplicate run caches changed {name}/{mode}/{key}"
                        )
                versions[mode] = direct
            for key in ("coords", "labels"):
                if not torch.equal(
                    versions["observed"][key], versions["constant_pair"][key]
                ):
                    raise RuntimeError(
                        f"Input ablation changed geometry or labels: {name}/{key}"
                    )
            constant = versions["constant_pair"]["feats"]
            expected = torch.tensor(
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=constant.dtype
            )
            if not torch.equal(constant, expected.expand_as(constant)):
                raise RuntimeError("Constant features changed during voxel pooling")
            parity.append(
                {"tile": name, "voxels": len(constant), "status": "bitwise_equal"}
            )
        finally:
            ds.feat_cfg = original_feature_cfg
        del raw, invariant, versions, direct, cached, constant
    loader.generator.manual_seed(int(cfg["run"]["seed"]) + 1009)
    batch = next(iter(loader))
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(10, 11, cfg=cfg).to(device).train()
    optimizer = build_optimizer(cfg, model)
    criterion = build_loss(cfg).to(device)
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(21)  # exercise the fully active Lovasz memory/gradient path
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["run"]["amp"]))
    tick = time.monotonic()
    with torch.autocast(device_type="cuda", enabled=bool(cfg["run"]["amp"])):
        logits, labels, _ = _forward_batch(model, batch, device, is_train=True)
        loss = criterion(logits, labels)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite preflight loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad).all()
        ):
            raise RuntimeError(f"Nonfinite preflight gradient: {name}")
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    with tempfile.TemporaryDirectory(
        prefix="feature_smoke_", dir=cfg["run"]["out_dir"]
    ) as tmp:
        runtime = StudyRuntime(cfg, DistEnv(False), device, tmp)
        runtime.begin_epoch(1, loader)
        runtime.observe_batch(batch)
        runtime.end_epoch(
            {"miou": 1.0, "confusion_matrix": np.eye(11, dtype=np.int64).tolist()},
            time.monotonic() - tick,
            model,
        )
        # This synthetic identity matrix only exercises 11-class telemetry, not model accuracy.
        path = Path(tmp) / "roundtrip.pt"
        estimate = (
            sum(v.numel() * v.element_size() for v in model.state_dict().values()) * 4
        )
        storage_guard(path.parent, estimate, cfg["study"]["storage"])
        atomic_save_torch(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": cfg,
            },
            path,
        )
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert_checkpoint_feature_config(saved, cfg)
        wrong = copy.deepcopy(cfg)
        wrong["data"]["features"]["return_input_mode"] = (
            "constant_pair" if cfg["study"]["arm"] == "observed" else "observed"
        )
        try:
            assert_checkpoint_feature_config(saved, wrong)
        except ValueError:
            pass
        else:
            raise RuntimeError("Checkpoint accepted an incompatible feature mode")
        restored = build_model(10, 11, cfg=cfg)
        restored.load_state_dict(saved["model_state"], strict=True)
        restored_optimizer = build_optimizer(cfg, restored)
        restored_optimizer.load_state_dict(saved["optimizer_state"])
        if model_digest(restored.state_dict()) != model_digest(model.state_dict()):
            raise RuntimeError("Checkpoint roundtrip changed model tensors")
        checkpoint_bytes = path.stat().st_size
    return {
        "status": "passed_gpu_smoke",
        "initial_model_sha256": initial["observed"],
        "cache_input_parity": parity,
        "batch_voxels": len(labels),
        "loss": float(loss.detach()),
        "step_seconds": time.monotonic() - tick,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "checkpoint_roundtrip_bytes": checkpoint_bytes,
        "gpu": torch.cuda.get_device_name(0),
        "smoke_worker_count": 0,
        "smoke_loss_epoch": 21,
        "timing_is_not_full_run_eta": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(args.config)
    frozen = load_frozen(root)
    validate_config(cfg, frozen)
    result = (
        {"status": "static_contract_only"}
        if args.static_only
        else gpu_preflight(cfg, frozen)
    )
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
