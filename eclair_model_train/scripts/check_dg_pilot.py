#!/usr/bin/env python3
"""Validate a two-epoch DG pilot and estimate full-run resource requirements."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


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
    parser.add_argument("--method", required=True, choices=("ocons", "bev_als"))
    parser.add_argument("--expected-epochs", type=int, default=2)
    parser.add_argument("--expected-world-size", type=int, default=2)
    parser.add_argument("--minimum-batches-per-rank", type=int, default=150)
    parser.add_argument("--full-epochs", type=int, default=200)
    parser.add_argument("--eval-every-epochs", type=int, default=2)
    parser.add_argument("--wall-limit-hours", type=float, default=36.0)
    parser.add_argument("--storage-budget-gb", type=float, default=10.0)
    parser.add_argument("--gpu-memory-mib", type=float, default=0.0)
    parser.add_argument("--time-safety-factor", type=float, default=1.15)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    errors: List[str] = []
    warnings: List[str] = []

    try:
        rows = _read_csv(run_dir / "metrics.csv")
        contract = _read_json(run_dir / "method_contract.json")
        complete = _read_json(run_dir / "diagnostic_complete.json")
        resolved = _read_json(run_dir / "config_resolved.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"pilot artifacts are incomplete or unreadable: {exc}")

    if contract.get("name") != args.method:
        errors.append(f"method contract is {contract.get('name')!r}, expected {args.method!r}")
    if complete.get("status") != "ok" or complete.get("method") != args.method:
        errors.append("diagnostic_complete.json does not certify the requested method")
    if len(rows) != args.expected_epochs:
        errors.append(f"metrics.csv has {len(rows)} rows, expected {args.expected_epochs}")

    expected_epoch_ids = list(range(1, args.expected_epochs + 1))
    actual_epoch_ids: List[int] = []
    for row in rows:
        try:
            actual_epoch_ids.append(int(row["epoch"]))
        except (KeyError, ValueError):
            errors.append("metrics.csv contains an invalid epoch value")
            break
    if actual_epoch_ids and actual_epoch_ids != expected_epoch_ids:
        errors.append(f"metrics epochs are {actual_epoch_ids}, expected {expected_epoch_ids}")

    common_fields = (
        "train_loss",
        "train_seg_clean_loss",
        "time_epoch_s",
        "eval_time_s",
        "epoch_wall_time_s",
        "gpu_peak_allocated_mib",
        "gpu_peak_reserved_mib",
        "backbone_grad_norm",
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
                errors.append(
                    f"epoch {index}: world_size={row.get('world_size')}, "
                    f"expected {args.expected_world_size}"
                )
            batches = int(float(row.get("train_batches_per_rank", "0")))
            if batches < args.minimum_batches_per_rank:
                errors.append(
                    f"epoch {index}: only {batches} training batches per rank; "
                    f"expected at least {args.minimum_batches_per_rank}"
                )
            if _finite(row, "backbone_grad_norm") <= 0.0:
                errors.append(f"epoch {index}: backbone gradient norm is not positive")
        except ValueError as exc:
            errors.append(f"epoch {index}: {exc}")

    final = rows[-1] if rows else {}
    for field in ("val_loss", "val_miou", "val_macro_f1"):
        try:
            _finite(final, field)
        except ValueError as exc:
            errors.append(f"final validation: {exc}")

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
        required = (
            "train_seg_perturbed_loss",
            "train_ocons_consistency_loss",
            "ocons_actual_mask_fraction",
            "ocons_active_jaccard",
            "ocons_coordinate_match_coverage",
            "ocons_protected_disappearances",
        )
        for index, row in enumerate(rows, start=1):
            try:
                for field in required:
                    _finite(row, field)
                mask = _finite(row, "ocons_actual_mask_fraction")
                jaccard = _finite(row, "ocons_active_jaccard")
                match = _finite(row, "ocons_coordinate_match_coverage")
                disappear = int(round(_finite(row, "ocons_protected_disappearances")))
                if not 0.095 <= mask <= 0.105:
                    errors.append(f"epoch {index}: O-CONS mask fraction is {mask:.6f}")
                if abs(jaccard - (1.0 - mask)) > 1e-4:
                    errors.append(f"epoch {index}: O-CONS Jaccard is inconsistent with subset masking")
                if match < 0.999999:
                    errors.append(f"epoch {index}: coordinate match coverage is {match:.9f}")
                if disappear != 0:
                    errors.append(f"epoch {index}: protected class disappearances={disappear}")
            except ValueError as exc:
                errors.append(f"epoch {index}: {exc}")

        class_rows = _read_csv(run_dir / "ocons_class_diagnostics.csv")
        final_class_rows = [r for r in class_rows if int(r["epoch"]) == args.expected_epochs]
        for class_id, name in ((5, "poles"), (6, "power_lines")):
            selected = [r for r in final_class_rows if int(r["class_id"]) == class_id]
            if not selected or int(selected[0]["clean_voxels"]) <= 0:
                errors.append(f"final O-CONS epoch has no {name} support")
            elif float(selected[0]["retention"]) < 0.70:
                errors.append(f"final O-CONS {name} retention is below 70%")

    else:
        required = (
            "train_bev_als_loss",
            "method_aux_grad_norm",
            "bev_als_bce",
            "bev_als_dice_loss",
            "bev_als_in_bounds_fraction",
            "bev_als_feature_in_bounds_fraction",
            "bev_als_multi_label_fraction",
        )
        for index, row in enumerate(rows, start=1):
            try:
                for field in required:
                    _finite(row, field)
                if _finite(row, "method_aux_grad_norm") <= 0.0:
                    errors.append(f"epoch {index}: BEV-ALS head gradient norm is not positive")
                target_bounds = _finite(row, "bev_als_in_bounds_fraction")
                feature_bounds = _finite(row, "bev_als_feature_in_bounds_fraction")
                if target_bounds < 0.995 or feature_bounds < 0.995:
                    errors.append(
                        f"epoch {index}: BEV-ALS bounds coverage is "
                        f"target={target_bounds:.6f}, feature={feature_bounds:.6f}"
                    )
            except ValueError as exc:
                errors.append(f"epoch {index}: {exc}")

        class_rows = _read_csv(run_dir / "bev_als_slice_class_diagnostics.csv")
        final_class_rows = [r for r in class_rows if int(r["epoch"]) == args.expected_epochs]
        for class_id, name in ((5, "poles"), (6, "power_lines")):
            support = sum(
                int(r["positive_support"])
                for r in final_class_rows
                if int(r["class_id"]) == class_id
            )
            if support <= 0:
                errors.append(f"final BEV-ALS epoch has no {name} target cells")
        truck_support = sum(
            int(r["positive_support"])
            for r in final_class_rows
            if int(r["class_id"]) == 3
        )
        if truck_support <= 0:
            warnings.append("final BEV-ALS epoch has no truck target cells")

    for name in ("last.pt", "best.pt"):
        if not (run_dir / "checkpoints" / name).is_file():
            errors.append(f"missing checkpoints/{name}")

    num_classes = int(((resolved.get("data", {}) or {}).get("label_space", {}) or {}).get("num_classes", 0))
    for name in ("val_confusion_latest.json", "val_confusion_best.json"):
        path = run_dir / name
        try:
            payload = _read_json(path)
            matrix = payload.get("matrix")
            if payload.get("rows") != "ground_truth" or payload.get("columns") != "prediction":
                errors.append(f"{name} does not declare matrix orientation")
            if not isinstance(matrix, list) or len(matrix) != num_classes:
                errors.append(f"{name} does not contain a {num_classes}x{num_classes} matrix")
                continue
            if any(not isinstance(row, list) or len(row) != num_classes for row in matrix):
                errors.append(f"{name} does not contain a {num_classes}x{num_classes} matrix")
                continue
            flat = [int(value) for row in matrix for value in row]
            if any(value < 0 for value in flat) or sum(flat) <= 0:
                errors.append(f"{name} contains an empty or invalid confusion matrix")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"cannot validate {name}: {exc}")

    peak_reserved = max((_finite(row, "gpu_peak_reserved_mib") for row in rows), default=0.0)
    gpu_fraction = None
    if args.gpu_memory_mib > 0.0:
        gpu_fraction = peak_reserved / args.gpu_memory_mib
        if gpu_fraction >= 0.90:
            errors.append(f"peak reserved GPU memory is {gpu_fraction:.1%} of device capacity")
        elif gpu_fraction >= 0.80:
            warnings.append(f"peak reserved GPU memory is {gpu_fraction:.1%} of device capacity")

    run_dir_gb = max((_finite(row, "run_dir_gb") for row in rows), default=0.0)
    storage_pass = run_dir_gb <= args.storage_budget_gb
    if not storage_pass:
        errors.append(
            f"run directory used {run_dir_gb:.3f} GiB, above {args.storage_budget_gb:.3f} GiB"
        )

    train_epoch_s = max((_finite(row, "time_epoch_s") for row in rows), default=0.0)
    eval_s = max((_finite(row, "eval_time_s") for row in rows), default=0.0)
    overhead_s = max(
        (
            max(
                0.0,
                _finite(row, "epoch_wall_time_s")
                - _finite(row, "time_epoch_s")
                - _finite(row, "eval_time_s"),
            )
            for row in rows
        ),
        default=0.0,
    )
    eval_count = math.ceil(args.full_epochs / max(1, args.eval_every_epochs))
    # DALES has 11 test tiles versus 3 validation tiles; use 4x validation as a
    # conservative final-test allowance.
    raw_full_seconds = (
        args.full_epochs * (train_epoch_s + overhead_s)
        + eval_count * eval_s
        + 4.0 * eval_s
    )
    projected_hours = raw_full_seconds * args.time_safety_factor / 3600.0
    single_job_time_pass = projected_hours <= args.wall_limit_hours
    if not single_job_time_pass:
        warnings.append(
            f"buffered full-run projection is {projected_hours:.2f} h, above the "
            f"{args.wall_limit_hours:.2f} h job limit; use resumable jobs"
        )

    report = {
        "status": "pass" if not errors else "fail",
        "method": args.method,
        "correctness_pass": not errors,
        "storage_pass": storage_pass,
        "single_job_time_pass": single_job_time_pass,
        "ready_for_full_single_job": not errors and storage_pass and single_job_time_pass,
        "ready_for_resumable_full_run": not errors and storage_pass,
        "observed": {
            "epochs": len(rows),
            "world_size": int(float(final.get("world_size", 0) or 0)),
            "train_batches_per_rank": int(float(final.get("train_batches_per_rank", 0) or 0)),
            "conservative_train_epoch_seconds": train_epoch_s,
            "validation_seconds": eval_s,
            "checkpoint_and_logging_overhead_seconds": overhead_s,
            "peak_reserved_gpu_mib": peak_reserved,
            "gpu_capacity_fraction": gpu_fraction,
            "run_dir_gib": run_dir_gb,
        },
        "projection": {
            "full_epochs": args.full_epochs,
            "evaluation_count": eval_count,
            "time_safety_factor": args.time_safety_factor,
            "buffered_full_run_hours": projected_hours,
            "wall_limit_hours": args.wall_limit_hours,
        },
        "errors": errors,
        "warnings": warnings,
    }
    output_path = run_dir / "pilot_acceptance.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
