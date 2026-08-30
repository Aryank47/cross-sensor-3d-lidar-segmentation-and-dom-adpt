#!/usr/bin/env python3
"""Fail-fast validation for the isolated O-CONS and BEV-ALS run configs."""

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
from src.method_config import validate_method_contract
from src.ocons import OConsConfig


def _at(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        cur = cur[key]
    return cur


def _stable_hash(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frozen-base", default="configs/m0_dales_frozen_29364.yaml")
    ap.add_argument("--expect-method", choices=["ocons", "bev_als"], required=True)
    ap.add_argument("--full", action="store_true", help="Reject diagnostic-only run limits.")
    ap.add_argument("--build-model", action="store_true", help="Instantiate the Minkowski model and count parameters.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    base = load_yaml(args.frozen_base)
    contract = validate_method_contract(cfg)
    if contract.name != args.expect_method:
        raise ValueError(f"Expected method={args.expect_method!r}; got {contract.name!r}")
    ocons = OConsConfig.from_cfg(cfg)
    bev = ALSBEVConfig.from_cfg(cfg)

    parity_paths = [
        "run.seed",
        "eval",
        "data.dataset",
        "data.voxelization",
        "data.dales_label_map_native_to_train",
        "data.patch",
        "data.sampling",
        "data.preproc",
        "data.aug",
        "data.features",
        "data.label_space",
        "data.batch_size",
        "data.grad_accum_steps",
        "optim",
        "sched",
        "loss",
        "model.in_channels",
        "model.out_channels",
        "model.D",
    ]
    mismatches = [p for p in parity_paths if _at(cfg, p) != _at(base, p)]
    if mismatches:
        raise ValueError(f"Method config changed frozen comparator fields: {mismatches}")
    if bool(cfg["data"]["mix3d"]["enabled"]):
        raise ValueError("Discovery configs must keep Mix3D disabled.")
    if args.full:
        if int(cfg["epochs"]) != int(base["epochs"]):
            raise ValueError("Full method config changed the frozen epoch count.")
        forbidden = ["max_train_batches_per_epoch", "skip_validation", "skip_final_test"]
        present = [k for k in forbidden if bool(cfg["run"].get(k, False))]
        if present:
            raise ValueError(f"Full config contains diagnostic controls: {present}")

    if args.expect_method == "ocons":
        if not ocons.enabled or bev.enabled:
            raise ValueError("O-CONS isolation check failed.")
        if tuple(ocons.protected_class_ids) != (5, 6):
            raise ValueError("O-CONS must protect DALES train IDs 5=poles and 6=power_lines.")
    else:
        if not bev.enabled or ocons.enabled:
            raise ValueError("BEV-ALS isolation check failed.")
        if bev.grid_size != 360:
            raise ValueError(f"Expected 360x360 BEV grid, got {bev.grid_size}.")

    params = None
    if args.build_model:
        from src.model import build_model

        model = build_model(
            in_channels=int(cfg["model"]["in_channels"]),
            out_channels=int(cfg["model"]["out_channels"]),
            D=int(cfg["model"].get("D", 3)),
            cfg=cfg,
        )
        params = sum(int(p.numel()) for p in model.parameters())

    roots = {
        key: {"set": bool(os.environ.get(key)), "value": os.environ.get(key, "")}
        for key in ("DALES_TRAIN_ROOT", "DALES_TEST_ROOT", "DALES_CACHE_ROOT_DROP_I")
    }
    print(
        json.dumps(
            {
                "status": "ok",
                "method": contract.to_dict(),
                "config": str(Path(args.config).resolve()),
                "resolved_sha256": _stable_hash(cfg),
                "model_parameters": params,
                "environment": roots,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
