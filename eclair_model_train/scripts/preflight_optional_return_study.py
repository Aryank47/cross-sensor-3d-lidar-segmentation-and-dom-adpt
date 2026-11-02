#!/usr/bin/env python3
"""Source-only actual masking, full-loss GPU steps and paired inference preflight."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config_loader import load_yaml
from src.feature_contract_study import make_config, model_digest
from src.optional_return_study import constant_diagnostic, validate_optional_config
from src.study_runtime import atomic_json

from scripts.preflight_feature_contract_study import gpu_preflight
from scripts.submit_feature_contract_study import load_frozen


def optional_smoke(cfg, frozen):
    import numpy as np
    import torch
    from src import eval_cross_common as ev
    from src.dist import DistEnv
    from src.model import build_model
    from src.optional_returns import CONSTANT, mask_batch
    from src.study_runtime import StudyRuntime
    from train import _forward_batch, build_dataloaders, build_loss, build_optimizer

    base = gpu_preflight(make_config(frozen, "observed", cfg["run"]["out_dir"]), frozen)
    smoke = copy.deepcopy(cfg)
    smoke["data"]["num_workers"] = 0
    smoke["study"]["reference_epochs_path"] = None
    loader, _, _ = build_dataloaders(smoke, DistEnv(False), eclair_run_cache_dir=None)
    loader.dataset.set_epoch(1)
    loader.generator.manual_seed(cfg["run"]["seed"] + 1009)
    original = next(iter(loader))
    coords = original["coords"].numpy().copy()
    labels = original["labels"].numpy().copy()
    feats = original["feats"].numpy().copy()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = build_model(10, 11, cfg=cfg)
    if model_digest(model.state_dict()) != base["initial_model_sha256"]:
        raise RuntimeError("Optional model initialization differs")
    model = model.to(device).train()
    optimizer = build_optimizer(cfg, model)
    criterion = build_loss(cfg).to(device)
    criterion.set_epoch(21)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["run"]["amp"])
    steps = []
    with tempfile.TemporaryDirectory(
        prefix="optional_smoke_", dir=cfg["run"]["out_dir"]
    ) as temporary:
        runtime = StudyRuntime(smoke, DistEnv(False), device, temporary)
        runtime.begin_epoch(1, loader)
        prepared = {**original}
        runtime.prepare_batch(prepared)
        runtime.observe_batch(prepared)
        if not torch.equal(prepared["coords"], original["coords"]) or not torch.equal(
            prepared["labels"], original["labels"]
        ):
            raise RuntimeError("Masking changed coordinates or labels")
        for probability in (0.0, 1.0, 0.5):
            expected, audit = mask_batch(
                coords,
                labels,
                feats,
                original["fnames"],
                seed=cfg["run"]["seed"],
                epoch=1,
                probability=probability,
            )
            if probability == 0.5 and not np.array_equal(
                prepared["feats"].numpy(), expected
            ):
                raise RuntimeError(
                    "Training hook differs from approved mask assignment"
                )
            if probability == 0 and not np.array_equal(expected, feats):
                raise RuntimeError("Observed input changed")
            if probability == 1 and not np.array_equal(
                expected, np.broadcast_to(CONSTANT, expected.shape)
            ):
                raise RuntimeError("Constant features incorrect")
            batch = {**original, "feats": torch.from_numpy(expected)}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=cfg["run"]["amp"]):
                logits, y, _ = _forward_batch(model, batch, device, is_train=True)
                loss = criterion(logits, y)
            runtime.check_objective(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if any(
                p.grad is not None and not bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
            ):
                raise RuntimeError("Nonfinite optional-return smoke gradient")
            runtime.optimizer_step(scaler, optimizer)
            steps.append(
                {
                    "probability_override": probability,
                    "loss": float(loss.detach()),
                    "masked_samples": audit["masked_samples"],
                }
            )
        runtime.end_epoch(
            {"miou": 1.0, "confusion_matrix": np.eye(11, dtype=np.int64).tolist()},
            1.0,
            model,
        )
    if not np.array_equal(original["feats"].numpy(), feats):
        raise RuntimeError("Original source features were mutated")
    raw = loader.dataset.get_raw(0)
    model.eval()
    logger = logging.getLogger("optional_smoke")
    observed = ev.infer_tile_point_logits(
        model=model,
        raw=raw,
        source_cfg=cfg,
        device=device,
        amp=False,
        window_size_m=120.0,
        window_stride_m=120.0,
        aggregation="mean_logits",
        max_voxels_per_forward=180000,
        max_split_depth=8,
        logger=logger,
        target_domain="eclair",
    )
    if not np.isfinite(observed).all():
        raise RuntimeError("Nonfinite observed-view preflight logits")
    prediction = observed.argmax(1)
    del observed
    native = ev.get_native_labels(raw, target_domain="eclair").astype(np.int64)
    paired = constant_diagnostic(model, raw, cfg, "eclair", prediction, native, logger)
    if sum(row[3] for row in paired["native_gt_observed_constant"]) != len(native):
        raise RuntimeError("Paired preflight point coverage mismatch")
    return {
        **base,
        "status": "passed_optional_gpu_smoke",
        "train_only_mask_hook": "passed",
        "forced_and_actual_mask_forward_backward": steps,
        "paired_canonical_source_train_points": len(native),
        "training_config_overrides": False,
        "smoke_overrides": {
            "num_workers": 0,
            "reference_epoch_check": False,
            "loss_epoch": 21,
        },
        "target_data_used_for_preflight": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    frozen = load_frozen(Path(__file__).resolve().parents[1])
    validate_optional_config(cfg, frozen)
    result = optional_smoke(cfg, frozen)
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
