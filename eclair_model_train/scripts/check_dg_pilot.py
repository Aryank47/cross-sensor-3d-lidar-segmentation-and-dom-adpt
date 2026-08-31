#!/usr/bin/env python3
"""Validate a scheduler-crossing, resume-exercising DG pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dg_dataset import get_dg_dataset_contract


def _finite(row: Dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or non-numeric metrics field {key!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"metrics field {key!r} is not finite: {value}")
    return value


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=("dales", "eclair"))
    parser.add_argument("--method", required=True, choices=("ocons", "bev_als"))
    parser.add_argument("--expected-world-size", type=int, default=2)
    parser.add_argument("--minimum-batches-per-rank", type=int, required=True)
    parser.add_argument("--wall-limit-hours", type=float, default=36.0)
    parser.add_argument("--storage-budget-gb", type=float, default=10.0)
    parser.add_argument("--gpu-memory-mib", type=float, default=0.0)
    parser.add_argument("--time-safety-factor", type=float, default=1.15)
    args = parser.parse_args()

    dataset_contract = get_dg_dataset_contract(args.dataset)
    run_dir = Path(args.run_dir).resolve()
    errors: List[str] = []
    warnings: List[str] = []

    try:
        rows = _read_csv(run_dir / "metrics.csv")
        contract = _read_json(run_dir / "method_contract.json")
        complete = _read_json(run_dir / "diagnostic_complete.json")
        resolved = _read_json(run_dir / "config_resolved.json")
        resume = _read_json(run_dir / "resume_state.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"pilot artifacts are incomplete or unreadable: {exc}")

    expected_epochs = dataset_contract.pilot_epochs
    if contract.get("name") != args.method:
        errors.append(f"method contract is {contract.get('name')!r}, expected {args.method!r}")
    expected_flags = {
        "mix3d_enabled": False,
        "ocons_enabled": args.method == "ocons",
        "bev_als_enabled": args.method == "bev_als",
        "legacy_bev_enabled": False,
    }
    for key, expected in expected_flags.items():
        if bool(contract.get(key, False)) != expected:
            errors.append(f"method contract {key}={contract.get(key)!r}, expected {expected}")
    if str((resolved.get("data", {}) or {}).get("dataset", "")).lower() != args.dataset:
        errors.append("resolved dataset does not match the requested dataset")
    if complete.get("status") != "ok" or complete.get("method") != args.method:
        errors.append("diagnostic_complete.json does not certify the requested method")
    if not bool(complete.get("resume_exercised", False)):
        errors.append("diagnostic_complete.json does not certify that resume was exercised")
    if len(rows) != expected_epochs:
        errors.append(f"metrics.csv has {len(rows)} rows, expected {expected_epochs}")

    expected_epoch_ids = list(range(1, expected_epochs + 1))
    try:
        actual_epoch_ids = [int(row["epoch"]) for row in rows]
    except (KeyError, ValueError):
        actual_epoch_ids = []
        errors.append("metrics.csv contains an invalid epoch value")
    if actual_epoch_ids and actual_epoch_ids != expected_epoch_ids:
        errors.append(f"metrics epochs are {actual_epoch_ids}, expected {expected_epoch_ids}")

    if int(resume.get("completed_epoch", -1)) != dataset_contract.pilot_resume_epoch:
        errors.append("resume_state.json has the wrong completed epoch")
    if int(resume.get("start_epoch", -1)) != dataset_contract.pilot_resume_epoch + 1:
        errors.append("resume_state.json has the wrong resumed start epoch")

    common_fields = (
        "lr_start",
        "lr",
        "train_loss",
        "train_objective_batch_mean",
        "train_aux_weighted_loss",
        "train_objective_reconstruction_error",
        "train_seg_clean_loss",
        "time_epoch_s",
        "eval_time_s",
        "epoch_wall_time_s",
        "gpu_peak_allocated_mib",
        "gpu_peak_reserved_mib",
        "backbone_grad_norm",
        "checkpoint_dir_gb",
        "run_dir_gb",
    )
    for index, row in enumerate(rows, start=1):
        if row.get("method") != args.method:
            errors.append(f"epoch {index}: method column is {row.get('method')!r}")
        for field in common_fields:
            try:
                _finite(row, field)
            except ValueError as exc:
                errors.append(f"epoch {index}: {exc}")
        try:
            if int(float(row.get("world_size", "0"))) != args.expected_world_size:
                errors.append(f"epoch {index}: unexpected world_size={row.get('world_size')}")
            batches = int(float(row.get("train_batches_per_rank", "0")))
            if batches < args.minimum_batches_per_rank:
                errors.append(f"epoch {index}: only {batches} training batches per rank")
            if _finite(row, "backbone_grad_norm") <= 0.0:
                errors.append(f"epoch {index}: backbone gradient norm is not positive")
            if _finite(row, "train_objective_reconstruction_error") > 1e-5:
                errors.append(f"epoch {index}: optimized/logged loss components do not reconcile")
        except ValueError as exc:
            errors.append(f"epoch {index}: {exc}")

    eval_every = int((resolved.get("run", {}) or {}).get("eval_every_epochs", 1))
    eval_rows = [row for row in rows if int(row["epoch"]) % max(1, eval_every) == 0 or int(row["epoch"]) == expected_epochs]
    for row in eval_rows:
        for field in ("val_loss", "val_miou", "val_macro_f1"):
            try:
                _finite(row, field)
            except ValueError as exc:
                errors.append(f"validation epoch {row.get('epoch')}: {exc}")
        for class_name in dataset_contract.class_names:
            suffix = class_name.replace(" ", "_").replace(".", "")
            try:
                _finite(row, f"val_iou_{suffix}")
            except ValueError as exc:
                errors.append(f"validation epoch {row.get('epoch')}: {exc}")
    if not eval_rows:
        errors.append("pilot did not execute validation")

    try:
        initial_lr = _finite(rows[0], "lr_start")
        post_step_rows = [row for row in rows if int(row["epoch"]) > dataset_contract.scheduler_step_epoch]
        if not post_step_rows or not _finite(post_step_rows[0], "lr_start") < initial_lr:
            errors.append("pilot did not demonstrate the configured StepLR transition")
    except (IndexError, ValueError) as exc:
        errors.append(f"cannot validate learning-rate transition: {exc}")

    data_cfg = resolved.get("data", {}) or {}
    run_cfg = resolved.get("run", {}) or {}
    if int(data_cfg.get("num_workers", -1)) != 4:
        errors.append(f"resolved data.num_workers={data_cfg.get('num_workers')}, expected 4")
    if bool(run_cfg.get("skip_validation", True)):
        errors.append("resolved run.skip_validation must be false")
    if not bool(run_cfg.get("skip_final_test", False)):
        errors.append("pilot must skip the final test set")
    if int(run_cfg.get("save_every_epochs", 0)) != 1:
        errors.append("pilot must atomically refresh last.pt every epoch")
    if bool(run_cfg.get("keep_epoch_snapshots", True)):
        errors.append("pilot must not retain accumulating epoch snapshots")

    if args.method == "ocons":
        ocons_cfg = resolved.get("ocons", {}) or {}
        if float(ocons_cfg.get("application_probability", -1.0)) != 1.0:
            errors.append("resolved O-CONS application_probability must be exactly 1.0")
        if tuple(int(x) for x in ocons_cfg.get("protected_class_ids", [])) != dataset_contract.utility_class_ids:
            errors.append("resolved O-CONS protected IDs do not match the dataset contract")
        required = (
            "train_seg_perturbed_loss",
            "train_ocons_consistency_loss",
            "ocons_actual_mask_fraction",
            "ocons_active_jaccard",
            "ocons_coordinate_match_coverage",
            "ocons_protected_disappearances",
            "ocons_application_rate",
            "ocons_clean_voxels",
            "ocons_perturbed_voxels",
            "ocons_protected_min_retention",
        )
        for index, row in enumerate(rows, start=1):
            try:
                for field in required:
                    _finite(row, field)
                mask = _finite(row, "ocons_actual_mask_fraction")
                jaccard = _finite(row, "ocons_active_jaccard")
                if not 0.095 <= mask <= 0.105:
                    errors.append(f"epoch {index}: O-CONS mask fraction is {mask:.6f}")
                if abs(jaccard - (1.0 - mask)) > 1e-6:
                    errors.append(f"epoch {index}: O-CONS subset Jaccard is inconsistent")
                if _finite(row, "ocons_coordinate_match_coverage") < 0.999999:
                    errors.append(f"epoch {index}: O-CONS coordinate matching is incomplete")
                if int(round(_finite(row, "ocons_protected_disappearances"))) != 0:
                    errors.append(f"epoch {index}: protected class disappearance occurred")
                if abs(_finite(row, "ocons_application_rate") - 1.0) > 1e-9:
                    errors.append(f"epoch {index}: O-CONS was not applied to every batch")
                if _finite(row, "ocons_protected_min_retention") + 1e-8 < 0.70:
                    errors.append(f"epoch {index}: per-sample protected retention fell below 70%")
            except ValueError as exc:
                errors.append(f"epoch {index}: {exc}")

        class_rows = _read_csv(run_dir / "ocons_class_diagnostics.csv")
        final_rows = [row for row in class_rows if int(row["epoch"]) == expected_epochs]
        for class_id in dataset_contract.utility_class_ids:
            selected = [row for row in final_rows if int(row["class_id"]) == class_id]
            if not selected or int(selected[0]["clean_voxels"]) <= 0:
                errors.append(f"final O-CONS epoch has no utility class {class_id} support")
            elif float(selected[0]["retention"]) < 0.70:
                errors.append(f"final O-CONS class {class_id} retention is below 70%")
    else:
        bev_cfg = ((resolved.get("model", {}) or {}).get("aux_heads", {}) or {}).get("bev_als", {}) or {}
        expected_bev = {
            "feature_level": "block8",
            "xy_frame": "bbox_centered",
            "half_extent_m": 90.0,
            "resolution_m": 0.5,
            "height_mode": "quantile",
            "height_slices": 4,
            "target_mode": "height_sliced_multilabel",
        }
        for key, expected in expected_bev.items():
            if bev_cfg.get(key) != expected:
                errors.append(f"resolved BEV-ALS {key}={bev_cfg.get(key)!r}, expected {expected!r}")
        required = (
            "train_bev_als_loss",
            "method_aux_grad_norm",
            "bev_als_bce",
            "bev_als_dice_loss",
            "bev_als_in_bounds_fraction",
            "bev_als_in_bounds_fraction_min",
            "bev_als_feature_in_bounds_fraction",
            "bev_als_feature_in_bounds_fraction_min",
            "bev_als_multi_label_fraction",
            "bev_als_multi_class_xy_column_fraction",
            "bev_als_empty_slice_rate",
            "bev_als_height_edge_min_gap_m",
            "bev_als_degenerate_height_edge_rate",
            "bev_als_macro_f1_supported",
            "bev_als_utility_f1_supported",
        )
        for index, row in enumerate(rows, start=1):
            try:
                for field in required:
                    _finite(row, field)
                if _finite(row, "method_aux_grad_norm") <= 0.0:
                    errors.append(f"epoch {index}: BEV-ALS head gradient norm is not positive")
                if _finite(row, "bev_als_in_bounds_fraction_min") < 0.995:
                    errors.append(f"epoch {index}: a BEV target frame falls below 99.5% coverage")
                if _finite(row, "bev_als_feature_in_bounds_fraction_min") < 0.995:
                    errors.append(f"epoch {index}: a block8 frame falls below 99.5% coverage")
                degenerate = _finite(row, "bev_als_degenerate_height_edge_rate")
                empty_slice_rate = _finite(row, "bev_als_empty_slice_rate")

                # Quantile ties are expected for discretized ALS heights. Treat them as
                # informational unless they become widespread or materially remove
                # height-slice supervision.
                if degenerate > 0.15:
                    errors.append(f"epoch {index}: degenerate height-edge rate is {degenerate:.2%}")
                elif degenerate > 0.0:
                    warnings.append(f"epoch {index}: degenerate height-edge rate is {degenerate:.2%}")

                if empty_slice_rate > 0.05:
                    errors.append(f"epoch {index}: empty BEV height-slice rate is {empty_slice_rate:.2%}")
                elif empty_slice_rate > 0.025:
                    warnings.append(f"epoch {index}: empty BEV height-slice rate is {empty_slice_rate:.2%}")

                # After both auxiliary and Lovasz ramps are complete, the nominally
                # auxiliary BEV term should not remain larger than the total objective's
                # segmentation contribution.
                if index > expected_epochs - 2:
                    objective = _finite(row, "train_objective_batch_mean")
                    weighted_aux = _finite(row, "train_aux_weighted_loss")
                    if objective <= 0.0:
                        errors.append(f"epoch {index}: non-positive optimized objective prevents " "BEV contribution analysis")
                    else:
                        aux_share = weighted_aux / objective
                        if aux_share > 0.50:
                            errors.append(f"epoch {index}: weighted BEV loss is {aux_share:.2%} " "of the optimized objective")
                        elif aux_share > 0.40:
                            warnings.append(f"epoch {index}: weighted BEV loss is {aux_share:.2%} " "of the optimized objective")
            except ValueError as exc:
                errors.append(f"epoch {index}: {exc}")

        class_rows = _read_csv(run_dir / "bev_als_slice_class_diagnostics.csv")
        final_rows = [row for row in class_rows if int(row["epoch"]) == expected_epochs]
        for class_id in dataset_contract.utility_class_ids:
            support = sum(int(float(row["positive_support"])) for row in final_rows if int(row["class_id"]) == class_id)
            if support <= 0:
                errors.append(f"final BEV-ALS epoch has no utility class {class_id} target cells")

    checkpoint_dir = run_dir / "checkpoints"
    try:
        last = torch.load(checkpoint_dir / "last.pt", map_location="cpu")
        best = torch.load(checkpoint_dir / "best.pt", map_location="cpu")
        if int(last.get("epoch", -1)) != expected_epochs:
            errors.append("last.pt does not represent the final pilot epoch")
        if "optimizer_state" not in last or "scheduler_state" not in last:
            errors.append("last.pt is not resumable")
        best_eval_row = max(eval_rows, key=lambda row: float(row["val_miou"])) if eval_rows else None
        if best_eval_row is not None:
            expected_best_epoch = int(best_eval_row["epoch"])
            if int(best.get("epoch", -1)) != expected_best_epoch:
                errors.append("best.pt epoch does not match the best logged validation epoch")
            if int(last.get("best_epoch", -1)) != expected_best_epoch:
                errors.append("last.pt best_epoch does not match logged checkpoint selection")
            if int(complete.get("best_epoch", -1)) != expected_best_epoch:
                errors.append("diagnostic_complete.json best_epoch does not match logged selection")
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        errors.append(f"cannot validate checkpoints: {exc}")

    num_classes = dataset_contract.num_classes
    for name in ("val_confusion_latest.json", "val_confusion_best.json"):
        try:
            payload = _read_json(run_dir / name)
            matrix = payload.get("matrix")
            if payload.get("rows") != "ground_truth" or payload.get("columns") != "prediction":
                errors.append(f"{name} does not declare matrix orientation")
            if not isinstance(matrix, list) or len(matrix) != num_classes:
                errors.append(f"{name} does not contain a {num_classes}x{num_classes} matrix")
            elif any(not isinstance(row, list) or len(row) != num_classes for row in matrix):
                errors.append(f"{name} does not contain a {num_classes}x{num_classes} matrix")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"cannot validate {name}: {exc}")

    for name in ("source_manifest.sha256", "source_snapshot.tar.gz", "git_commit.txt", "git_status.txt"):
        if not (run_dir / name).is_file():
            errors.append(f"missing provenance artifact {name}")

    peak_reserved = max((_finite(row, "gpu_peak_reserved_mib") for row in rows), default=0.0)
    gpu_fraction = None
    if args.gpu_memory_mib > 0.0:
        gpu_fraction = peak_reserved / args.gpu_memory_mib
        if gpu_fraction >= 0.90:
            errors.append(f"peak reserved GPU memory is {gpu_fraction:.1%} of device capacity")
        elif gpu_fraction >= 0.80:
            warnings.append(f"peak reserved GPU memory is {gpu_fraction:.1%} of device capacity")

    run_dir_gb = max((_finite(row, "run_dir_gb") for row in rows), default=0.0)
    if run_dir_gb > args.storage_budget_gb:
        errors.append(f"run directory used {run_dir_gb:.3f} GiB, above budget")

    train_epoch_s = max((_finite(row, "time_epoch_s") for row in rows), default=0.0)
    eval_s = max((_finite(row, "eval_time_s") for row in eval_rows), default=0.0)
    overhead_s = max(
        (
            max(0.0, _finite(row, "epoch_wall_time_s") - _finite(row, "time_epoch_s") - _finite(row, "eval_time_s"))
            for row in rows
        ),
        default=0.0,
    )
    full_eval_count = math.ceil(dataset_contract.full_epochs / max(1, eval_every))
    raw_full_seconds = (
        dataset_contract.full_epochs * (train_epoch_s + overhead_s)
        + full_eval_count * eval_s
        + dataset_contract.final_test_eval_factor * eval_s
    )
    projected_hours = raw_full_seconds * args.time_safety_factor / 3600.0
    single_job_time_pass = projected_hours <= args.wall_limit_hours
    if not single_job_time_pass:
        warnings.append("full-run projection exceeds one wall-time allocation; resumable jobs are required")

    report = {
        "status": "pass" if not errors else "fail",
        "dataset": args.dataset,
        "method": args.method,
        "implementation_correctness_pass": not errors,
        "research_performance_evaluated": False,
        "ready_for_full_single_job": not errors and single_job_time_pass,
        "ready_for_resumable_full_run": not errors,
        "observed": {
            "epochs": len(rows),
            "resume_split_epoch": dataset_contract.pilot_resume_epoch,
            "scheduler_step_epoch": dataset_contract.scheduler_step_epoch,
            "conservative_train_epoch_seconds": train_epoch_s,
            "validation_seconds": eval_s,
            "peak_reserved_gpu_mib": peak_reserved,
            "gpu_capacity_fraction": gpu_fraction,
            "run_dir_gib": run_dir_gb,
        },
        "projection": {
            "full_epochs": dataset_contract.full_epochs,
            "buffered_full_run_hours": projected_hours,
            "wall_limit_hours": args.wall_limit_hours,
        },
        "errors": errors,
        "warnings": warnings,
    }
    (run_dir / "pilot_acceptance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
