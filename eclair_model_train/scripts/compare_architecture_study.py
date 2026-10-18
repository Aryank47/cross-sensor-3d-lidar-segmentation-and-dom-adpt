#!/usr/bin/env python3
"""Compare every completed primary checkpoint; never cherry-pick target epochs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.study_metrics import balanced_gate, summary
from src.study_runtime import atomic_json


def read_arm(root, arm):
    run = root / arm
    if json.loads((run / "STATUS.json").read_text())["status"] != "complete":
        raise RuntimeError(f"{arm} is not complete")
    manifest = json.loads((run / "evaluation_manifest.json").read_text())
    entry = manifest["checkpoints"]["best"]
    folder = run / "evaluation" / entry["model_sha256"]
    return {
        domain: json.loads((folder / domain / "metrics.json").read_text()) for domain in ("dales", "eclair")
    }


def paired_tiles(reference, candidate):
    a = np.asarray([r["confusion_matrix"] for r in reference["tiles"]], dtype=np.int64)
    b = np.asarray([r["confusion_matrix"] for r in candidate["tiles"]], dtype=np.int64)
    if a.shape != b.shape or reference["contract"]["dataset"] != candidate["contract"]["dataset"]:
        raise ValueError("Paired evaluation support mismatch")
    rng = np.random.default_rng(1337)
    differences = []
    for _ in range(1000):
        ids = rng.integers(0, len(a), len(a))
        ma, mb = summary(a[ids].sum(0)), summary(b[ids].sum(0))
        differences.append([mb[k] - ma[k] for k in ("all_seven_miou", "utility_miou")])
    return {
        "all_and_utility_delta_percentiles_2_5_97_5": np.quantile(
            differences, [0.025, 0.975], axis=0
        ).tolist(),
        "scope": "descriptive paired tile resampling; not seed or independent-acquisition uncertainty",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", required=True)
    args = parser.parse_args()
    root = Path(args.study_dir)
    reference = read_arm(root, "reference")
    result = {
        "reference": reference,
        "candidates": {},
        "selection": "fixed source-selected best checkpoints",
        "target_informed_exploratory_comparison": True,
        "replicated": False,
    }
    for arm in ("context", "fine"):
        if not (root / arm).exists():
            continue
        candidate = read_arm(root, arm)
        result["candidates"][arm] = {
            "metrics": candidate,
            "gate": balanced_gate(
                reference["dales"], candidate["dales"], reference["eclair"], candidate["eclair"]
            ),
            "paired_tiles": paired_tiles(reference["eclair"], candidate["eclair"]),
        }
    atomic_json(root / "comparison.json", result)
    print(json.dumps({arm: v["gate"] for arm, v in result["candidates"].items()}, indent=2))


if __name__ == "__main__":
    main()
