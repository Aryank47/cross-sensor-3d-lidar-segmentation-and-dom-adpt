#!/usr/bin/env python3
"""Fail-fast validation for DALES/ECLAIR O-CONS and BEV-ALS configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bev_als_config import ALSBEVConfig
from src.config_loader import load_yaml
from src.dg_dataset import get_dg_dataset_contract
from src.method_config import validate_method_contract
from src.ocons import OConsConfig


_MISSING = object()


def _at(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return _MISSING
        cur = cur[key]
    return cur


def _stable_hash(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parity_paths(dataset: str) -> list[str]:
    common = [
        "run.seed", "run.amp", "eval", "data.dataset", "data.voxelization",
        "data.patch", "data.sampling", "data.preproc", "data.aug", "data.features",
        "data.label_space", "data.batch_size", "data.grad_accum_steps", "data.use_cache",
        "data.cache_root", "data.require_cache", "data.require_cache_metadata", "optim",
        "sched", "loss", "model.in_channels", "model.out_channels", "model.D",
    ]
    if dataset == "dales":
        common += [
            "data.dales_train_root", "data.dales_test_root", "data.split_manifest_dir",
            "data.dales_label_map_native_to_train", "data.cache_subdir",
            "data.cache_key_extra", "data.cache_kind", "data.write_cache",
        ]
    else:
        common += [
            "data.eclair_root", "data.run_cache_enabled",
            "data.run_cache_precompute_returns_onehot", "eclair",
        ]
    return common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frozen-base", required=True)
    ap.add_argument("--dataset", choices=("dales", "eclair"), required=True)
    ap.add_argument("--expect-method", choices=("ocons", "bev_als"), required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--diagnostic", action="store_true")
    ap.add_argument("--build-model", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    base = load_yaml(args.frozen_base)
    dataset_contract = get_dg_dataset_contract(args.dataset)
    contract = validate_method_contract(cfg)
    if contract.name != args.expect_method:
        raise ValueError(f"Expected method={args.expect_method!r}; got {contract.name!r}")
    if str(cfg.get("data", {}).get("dataset", "")).lower() != args.dataset:
        raise ValueError("Config dataset does not match --dataset.")
    if str(base.get("data", {}).get("dataset", "")).lower() != args.dataset:
        raise ValueError("Frozen-base dataset does not match --dataset.")

    mismatches = [path for path in _parity_paths(args.dataset) if _at(cfg, path) != _at(base, path)]
    if mismatches:
        raise ValueError(f"Method config changed frozen comparator fields: {mismatches}")
    if bool((cfg.get("data", {}).get("mix3d", {}) or {}).get("enabled", False)):
        raise ValueError("Discovery configs must keep Mix3D disabled.")

    expected_epochs = None
    if args.full:
        expected_epochs = dataset_contract.full_epochs
    elif args.pilot:
        expected_epochs = dataset_contract.pilot_epochs
    if expected_epochs is not None and int(cfg.get("epochs", -1)) != expected_epochs:
        mode_name = "full" if args.full else "pilot"
        raise ValueError(
            f"{args.dataset} {mode_name} config must use "
            f"epochs={expected_epochs}; got {cfg.get('epochs')}."
        )
    run_cfg = cfg.get("run", {}) or {}
    if args.full:
        forbidden = ["max_train_batches_per_epoch", "skip_validation", "skip_final_test"]
        present = [key for key in forbidden if bool(run_cfg.get(key, False))]
        if present:
            raise ValueError(f"Full config contains pilot/diagnostic controls: {present}")
    elif args.pilot:
        if bool(run_cfg.get("skip_validation", True)):
            raise ValueError("Pilot must run validation.")
        if not bool(run_cfg.get("skip_final_test", False)):
            raise ValueError("Pilot must skip the final test set.")
        if int(run_cfg.get("resume_split_epoch", -1)) != dataset_contract.pilot_resume_epoch:
            raise ValueError(
                f"Pilot must exercise resume after epoch {dataset_contract.pilot_resume_epoch}."
            )
    else:
        if int(run_cfg.get("max_train_batches_per_epoch", 0)) <= 0:
            raise ValueError("Diagnostic config must cap training batches per epoch.")
        if not bool(run_cfg.get("skip_final_test", False)):
            raise ValueError("Diagnostic config must skip the final test set.")

    ocons = OConsConfig.from_cfg(cfg)
    bev = ALSBEVConfig.from_cfg(cfg)
    if args.expect_method == "ocons":
        if not ocons.enabled or bev.enabled:
            raise ValueError("O-CONS isolation check failed.")
        if tuple(ocons.protected_class_ids) != dataset_contract.utility_class_ids:
            raise ValueError("O-CONS utility-class protection does not match the dataset contract.")
    else:
        if not bev.enabled or ocons.enabled:
            raise ValueError("BEV-ALS isolation check failed.")
        if bev.grid_size != 360:
            raise ValueError(f"Expected 360x360 BEV grid, got {bev.grid_size}.")
        if bev.half_extent_m != 90.0 or bev.resolution_m != 0.5:
            raise ValueError("Final BEV-ALS protocol requires ±90 m at 0.5 m resolution.")

    data = cfg.get("data", {}) or {}
    if args.dataset == "dales":
        if str(data.get("cache_kind", "")).lower() != "raw" or not bool(data.get("require_cache", False)):
            raise ValueError("DALES DG runs require the validated raw cache path.")
    else:
        if not bool(data.get("require_cache", False)) or not bool(data.get("require_cache_metadata", False)):
            raise ValueError("ECLAIR DG runs require cache entries with source metadata.")

    params = None
    if args.build_model:
        from src.model import build_model

        model = build_model(
            in_channels=int(cfg["model"]["in_channels"]),
            out_channels=int(cfg["model"]["out_channels"]),
            D=int(cfg["model"].get("D", 3)),
            cfg=cfg,
        )
        params = sum(int(parameter.numel()) for parameter in model.parameters())

    environment_keys = (
        "DALES_TRAIN_ROOT", "DALES_TEST_ROOT", "DALES_CACHE_ROOT_DROP_I",
        "ECLAIR_ROOT", "ECLAIR_CACHE_ROOT",
    )
    roots = {
        key: {"set": bool(os.environ.get(key)), "value": os.environ.get(key, "")}
        for key in environment_keys
    }
    print(
        json.dumps(
            {
                "status": "ok",
                "dataset": args.dataset,
                "mode": "full" if args.full else ("pilot" if args.pilot else "diagnostic"),
                "method": contract.to_dict(),
                "config": str(Path(args.config).resolve()),
                "frozen_base": str(Path(args.frozen_base).resolve()),
                "resolved_sha256": _stable_hash(cfg),
                "model_parameters": params,
                "pilot_epochs": dataset_contract.pilot_epochs,
                "resume_split_epoch": dataset_contract.pilot_resume_epoch,
                "environment": roots,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
