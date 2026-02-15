#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -------------------------
# Data structures
# -------------------------


@dataclass
class DirectionSummary:
    name: str
    baseline_id: str
    baseline: Dict[str, Any]
    rows: List[Dict[str, Any]]
    best_miou: Dict[str, Any]
    best_wires: Dict[str, Any]
    best_poles: Dict[str, Any]


# -------------------------
# Core helpers
# -------------------------


def load_results(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Load a results.json produced by eval_dales_preproc_sweep_wo_intensity.

    Expected structure:
    {
      "baseline_id": "...",
      "rows": [
         {
           "run_id": "...",
           "full_miou": ...,
           "full_wires": ...,
           "full_poles": ...,
           ...
         },
         ...
      ]
    }
    """
    data = json.loads(path.read_text())
    baseline_id = data["baseline_id"]
    rows: List[Dict[str, Any]] = data["rows"]
    return baseline_id, rows


def get_baseline(baseline_id: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    for r in rows:
        if r["run_id"] == baseline_id:
            return r
    raise RuntimeError(f"Baseline run_id={baseline_id} not found in rows.")


def find_best(
    rows: List[Dict[str, Any]],
    metric_key: str,
) -> Dict[str, Any]:
    """
    Return the row with the highest value of metric_key.
    Assumes all rows have that key and are successful runs.
    """
    best_row: Optional[Dict[str, Any]] = None
    best_val: float = float("-inf")

    for r in rows:
        v = float(r.get(metric_key, float("nan")))
        if v > best_val:
            best_val = v
            best_row = r

    if best_row is None:
        raise RuntimeError(f"No rows found when searching for best {metric_key}.")
    return best_row


def summarize_direction(name: str, path: Path) -> DirectionSummary:
    baseline_id, rows = load_results(path)
    baseline = get_baseline(baseline_id, rows)

    best_miou = find_best(rows, "full_miou")
    best_wires = find_best(rows, "full_wires")
    best_poles = find_best(rows, "full_poles")

    return DirectionSummary(
        name=name,
        baseline_id=baseline_id,
        baseline=baseline,
        rows=rows,
        best_miou=best_miou,
        best_wires=best_wires,
        best_poles=best_poles,
    )


def fmt(x: float, ndigits: int = 4) -> str:
    return f"{x:.{ndigits}f}"


def extract_metric(row: Dict[str, Any], key: str) -> float:
    return float(row.get(key, float("nan")))


# -------------------------
# Printing helpers
# -------------------------


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    """Simple ASCII/markdown-ish table printer."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    sep = "-+-".join("-" * w for w in col_widths)

    print(fmt_row(headers))
    print(sep)
    for r in rows:
        print(fmt_row(r))


# -------------------------
# Main comparison logic
# -------------------------


def build_json_summary(
    e2d: DirectionSummary,
    d2e: DirectionSummary,
) -> Dict[str, Any]:
    def core(dir_sum: DirectionSummary) -> Dict[str, Any]:
        return {
            "baseline_id": dir_sum.baseline_id,
            "baseline_metrics": {
                "full_miou": extract_metric(dir_sum.baseline, "full_miou"),
                "full_wires": extract_metric(dir_sum.baseline, "full_wires"),
                "full_poles": extract_metric(dir_sum.baseline, "full_poles"),
                "full_fence": extract_metric(dir_sum.baseline, "full_fence"),
                "full_vehicle": extract_metric(dir_sum.baseline, "full_vehicle"),
            },
            "best_by_miou": {
                "run_id": dir_sum.best_miou["run_id"],
                "metrics": {
                    "full_miou": extract_metric(dir_sum.best_miou, "full_miou"),
                    "full_wires": extract_metric(dir_sum.best_miou, "full_wires"),
                    "full_poles": extract_metric(dir_sum.best_miou, "full_poles"),
                    "full_fence": extract_metric(dir_sum.best_miou, "full_fence"),
                    "full_vehicle": extract_metric(dir_sum.best_miou, "full_vehicle"),
                },
                "spec": dir_sum.best_miou.get("spec", {}),
            },
            "best_by_wires": {
                "run_id": dir_sum.best_wires["run_id"],
                "metrics": {
                    "full_miou": extract_metric(dir_sum.best_wires, "full_miou"),
                    "full_wires": extract_metric(dir_sum.best_wires, "full_wires"),
                    "full_poles": extract_metric(dir_sum.best_wires, "full_poles"),
                },
                "spec": dir_sum.best_wires.get("spec", {}),
            },
            "best_by_poles": {
                "run_id": dir_sum.best_poles["run_id"],
                "metrics": {
                    "full_miou": extract_metric(dir_sum.best_poles, "full_miou"),
                    "full_wires": extract_metric(dir_sum.best_poles, "full_wires"),
                    "full_poles": extract_metric(dir_sum.best_poles, "full_poles"),
                },
                "spec": dir_sum.best_poles.get("spec", {}),
            },
        }

    # per-metric comparison table for JSON
    comparison_rows = []
    for metric_key, pretty_name in [
        ("full_miou", "mIoU"),
        ("full_wires", "IoU_wires"),
        ("full_poles", "IoU_poles"),
    ]:
        e_best = find_best(e2d.rows, metric_key)
        d_best = find_best(d2e.rows, metric_key)
        comparison_rows.append(
            {
                "metric": pretty_name,
                "eclair_to_dales": {
                    "run_id": e_best["run_id"],
                    "value": extract_metric(e_best, metric_key),
                },
                "dales_to_eclair": {
                    "run_id": d_best["run_id"],
                    "value": extract_metric(d_best, metric_key),
                },
            }
        )

    return {
        "eclair_to_dales": core(e2d),
        "dales_to_eclair": core(d2e),
        "metric_best_runs_comparison": comparison_rows,
    }


def print_human_summary(e2d: DirectionSummary, d2e: DirectionSummary) -> None:
    print()
    print("=== Cross-Domain Returns-Only Sweep Summary ===")
    print()

    # 1) Baseline vs best mIoU per direction
    baseline_rows = [
        [
            e2d.name,
            e2d.baseline_id,
            fmt(extract_metric(e2d.baseline, "full_miou")),
            fmt(extract_metric(e2d.baseline, "full_wires")),
            fmt(extract_metric(e2d.baseline, "full_poles")),
        ],
        [
            d2e.name,
            d2e.baseline_id,
            fmt(extract_metric(d2e.baseline, "full_miou")),
            fmt(extract_metric(d2e.baseline, "full_wires")),
            fmt(extract_metric(d2e.baseline, "full_poles")),
        ],
    ]
    print("Baseline metrics (per direction):")
    print_table(
        headers=["Direction", "Baseline run_id", "mIoU", "IoU_wires", "IoU_poles"],
        rows=baseline_rows,
    )
    print()

    best_miou_rows = [
        [
            e2d.name,
            e2d.best_miou["run_id"],
            fmt(extract_metric(e2d.best_miou, "full_miou")),
            fmt(extract_metric(e2d.best_miou, "full_wires")),
            fmt(extract_metric(e2d.best_miou, "full_poles")),
        ],
        [
            d2e.name,
            d2e.best_miou["run_id"],
            fmt(extract_metric(d2e.best_miou, "full_miou")),
            fmt(extract_metric(d2e.best_miou, "full_wires")),
            fmt(extract_metric(d2e.best_miou, "full_poles")),
        ],
    ]
    print("Best mIoU runs (per direction):")
    print_table(
        headers=["Direction", "Best-mIoU run_id", "mIoU", "IoU_wires", "IoU_poles"],
        rows=best_miou_rows,
    )
    print()

    # 2) Single table: per-metric best run in each direction (this is what you asked for)
    comparison_rows: List[List[str]] = []
    for metric_key, pretty_name in [
        ("full_miou", "mIoU"),
        ("full_wires", "IoU_wires"),
        ("full_poles", "IoU_poles"),
    ]:
        e_best = find_best(e2d.rows, metric_key)
        d_best = find_best(d2e.rows, metric_key)
        comparison_rows.append(
            [
                pretty_name,
                fmt(extract_metric(e_best, metric_key)),
                e_best["run_id"],
                fmt(extract_metric(d_best, metric_key)),
                d_best["run_id"],
            ]
        )

    print("Per-metric best runs (side-by-side, both directions):")
    print_table(
        headers=[
            "Metric",
            "E→D value",
            "E→D run_id",
            "D→E value",
            "D→E run_id",
        ],
        rows=comparison_rows,
    )
    print()


# -------------------------
# CLI
# -------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize cross-domain returns-only sweeps.\n"
            "Takes two results.json files (ECLAIR→DALES and DALES→ECLAIR) and "
            "prints side-by-side tables plus optional JSON summary."
        )
    )
    ap.add_argument(
        "--e2d",
        required=True,
        type=Path,
        help="Path to results.json for ECLAIR→DALES sweep.",
    )
    ap.add_argument(
        "--d2e",
        required=True,
        type=Path,
        help="Path to results.json for DALES→ECLAIR sweep.",
    )
    ap.add_argument(
        "--label-e2d",
        default="ECLAIR → DALES",
        help="Human-readable label for the E→D direction (for tables).",
    )
    ap.add_argument(
        "--label-d2e",
        default="DALES → ECLAIR",
        help="Human-readable label for the D→E direction (for tables).",
    )
    ap.add_argument(
        "--out-json",
        default=None,
        type=Path,
        help="Optional path to write a machine-readable JSON summary.",
    )

    args = ap.parse_args()

    e2d = summarize_direction(args.label_e2d, args.e2d)
    d2e = summarize_direction(args.label_d2e, args.d2e)

    print_human_summary(e2d, d2e)

    if args.out_json is not None:
        summary = build_json_summary(e2d, d2e)
        args.out_json.write_text(json.dumps(summary, indent=2))
        print(f"[info] JSON summary written to: {args.out_json}")


if __name__ == "__main__":
    main()
