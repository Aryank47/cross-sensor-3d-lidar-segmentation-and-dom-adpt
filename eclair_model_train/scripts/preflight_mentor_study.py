#!/usr/bin/env python3
"""Source-only real GPU loss/alignment/frozen-mentor/checkpoint checks."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config_loader import load_yaml
from src.feature_contract_study import make_config, model_digest
from src.mentor_study import validate_mentor_config
from src.study_runtime import atomic_json

from scripts.preflight_feature_contract_study import gpu_preflight
from scripts.submit_feature_contract_study import load_frozen


def smoke(cfg, frozen):
    import numpy as np
    import torch
    from src import eval_cross_common as ev
    from src.dist import DistEnv
    from src.features import assert_checkpoint_feature_config
    from src.mentor_guidance import FrozenMentor, gradient_diagnostics, guidance_weight
    from src.model import build_model
    from src.optional_return_study import constant_diagnostic
    from src.study_runtime import StudyRuntime
    from train import _forward_batch, build_dataloaders, build_loss, build_optimizer

    base = gpu_preflight(make_config(frozen, "observed", cfg["run"]["out_dir"]), frozen)
    torch.cuda.empty_cache()
    altered = copy.deepcopy(cfg)
    altered["data"]["num_workers"] = 0
    altered["study"]["reference_epochs_path"] = None
    loader, _, _ = build_dataloaders(altered, DistEnv(False), eclair_run_cache_dir=None)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    runtime = StudyRuntime(altered, DistEnv(False), device, cfg["run"]["out_dir"])
    runtime.begin_epoch(1, loader)
    loader.dataset.set_epoch(1)
    batch = next(iter(loader))
    runtime.observe_batch(batch)
    model = build_model(10, 11, cfg=cfg)
    if model_digest(model.state_dict()) != base["initial_model_sha256"]:
        raise RuntimeError("Student initialization differs")
    model = model.to(device).train()
    model._mentor_output_coords = None
    optimizer = build_optimizer(cfg, model)
    criterion = build_loss(cfg).to(device)
    criterion.set_epoch(21)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["run"]["amp"])
    mentor = FrozenMentor(cfg, device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    tick = time.monotonic()
    prediction = mentor.predict(batch)
    with torch.autocast(device_type="cuda", enabled=cfg["run"]["amp"]):
        logits, labels, _ = _forward_batch(model, batch, device, is_train=True)
        supervised = criterion(logits, labels)
    guide, stats = mentor.loss(prediction, logits, labels, model._mentor_output_coords)
    if not sum(stats["eligible_counts"]) or stats["loss"] <= 0:
        raise RuntimeError("Mentor guidance has no source signal")
    weight = guidance_weight(21)
    stats.update(gradient_diagnostics(guide, supervised, logits, weight))
    total = supervised + weight * guide
    runtime.check_objective(total)
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    if any(
        p.grad is not None and not bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
    ):
        raise RuntimeError("Nonfinite guided preflight gradient")
    runtime.optimizer_step(scaler, optimizer)
    mentor.verify_immutable()
    torch.cuda.synchronize(device)
    guided_seconds = time.monotonic() - tick
    guided_peak = torch.cuda.max_memory_allocated(device)
    guided_reserved = torch.cuda.max_memory_reserved(device)
    with tempfile.TemporaryDirectory(
        prefix="mentor_checkpoint_smoke_", dir=cfg["run"]["out_dir"]
    ) as temporary:
        path = Path(temporary) / "roundtrip.pt"
        torch.save({"model_state": model.state_dict(), "cfg": cfg}, path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert_checkpoint_feature_config(saved, cfg)
        validate_mentor_config(saved["cfg"], frozen)
        if model_digest(saved["model_state"]) != model_digest(model.state_dict()):
            raise RuntimeError("Guided checkpoint roundtrip differs")
    del (
        prediction,
        logits,
        labels,
        supervised,
        guide,
        total,
        optimizer,
        criterion,
        scaler,
        mentor,
    )
    torch.cuda.empty_cache()
    model.eval()
    raw = loader.dataset.get_raw(0)
    logger = logging.getLogger("mentor_smoke")
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
        raise RuntimeError("Nonfinite paired inference")
    predicted = observed.argmax(1)
    del observed
    native = ev.get_native_labels(raw, target_domain="eclair").astype(np.int64)
    paired = constant_diagnostic(model, raw, cfg, "eclair", predicted, native, logger)
    if sum(r[3] for r in paired["native_gt_observed_constant"]) != len(native):
        raise RuntimeError("Paired source coverage differs")
    return {
        **base,
        "baseline_preflight_peak_allocated_bytes": base["peak_allocated_bytes"],
        "peak_allocated_bytes": guided_peak,
        "guided_peak_reserved_bytes": guided_reserved,
        "guided_full_step_seconds": guided_seconds,
        "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "status": "passed_mentor_gpu_smoke",
        "guidance": stats,
        "mentor": cfg["study"]["mentor"],
        "mentor_frozen_check": "passed",
        "checkpoint_roundtrip": "passed",
        "target_data_used": False,
        "paired_source_train_points": len(native),
        "training_config_overrides": False,
        "smoke_only": {
            "num_workers": 0,
            "reference_epoch_check": False,
            "full_lovasz_epoch": 21,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    frozen = load_frozen(Path(__file__).resolve().parents[1])
    validate_mentor_config(cfg, frozen)
    result = smoke(cfg, frozen)
    atomic_json(args.out, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
