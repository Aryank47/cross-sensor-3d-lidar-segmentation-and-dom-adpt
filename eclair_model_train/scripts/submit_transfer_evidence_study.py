#!/usr/bin/env python3
"""Quota-gated submission of one frozen-weight evidence job; always exclude cn07."""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_architecture_study import quota_headroom, resolve_excluded  # noqa: E402
from src.transfer_evidence import (  # noqa: E402
    ASSETS,
    read_json,
    safe_output,
    validate_selection,
    verify_checkpoints,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="transfer_evidence_v1")
    parser.add_argument("--hours", type=int, choices=range(1, 37), default=36)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = read_json(ASSETS / "study.json")
    out = safe_output(f"/scratch/{getpass.getuser()}/e0_results/transfer_evidence/{args.study}")
    parent = next(p for p in (out, *out.parents) if p.is_dir())
    headroom = quota_headroom(parent)
    required = cfg["minimum_quota_headroom_bytes"]
    if min(headroom, shutil.disk_usage(parent).free) < required:
        parser.error("Need 10 GiB of USER QUOTA and filesystem headroom; no automatic cleanup or submission")
    if args.resume:
        if not (out / "study_manifest.json").is_file():
            parser.error("No existing study manifest to resume")
        if (out / "STATUS.json").is_file() and read_json(out / "STATUS.json").get("status") == "complete":
            parser.error("Study already complete; no repeat job needed")
    elif out.exists() and any(out.iterdir()):
        parser.error("Study directory already contains files; choose another --study name or --resume")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "scripts", "configs"], cwd=ROOT, text=True
    ).strip():
        parser.error("Evaluation source/config files must be committed before submission")
    tasks = verify_checkpoints(cfg)
    import torch

    for task in tasks:
        checkpoint = torch.load(task["checkpoint"], map_location="cpu")
        with (Path(task["checkpoint"]).parents[1] / "metrics.csv").open() as f:
            validate_selection(checkpoint, task, list(csv.DictReader(f)))
        del checkpoint
    nodes = subprocess.check_output(["sinfo", "-N", "-h", "-o", "%N"], text=True).splitlines()
    excluded = resolve_excluded(nodes)
    queued = subprocess.check_output(["squeue", "-u", getpass.getuser(), "-h", "-o", "%i"], text=True).splitlines()
    if len(queued) >= 3:
        parser.error("Already three queued/running jobs; no new job submitted")
    print(
        json.dumps(
            {
                "checkpoint_count": len(tasks),
                "canonical_pairs": 20,
                "cn07_excluded": excluded,
                "quota_headroom_gib": headroom / 1024**3,
                "output": str(out),
                "selection_warnings": [t["selection_warning"] for t in tasks if t.get("selection_warning")],
            },
            indent=2,
        ),
        flush=True,
    )
    if args.check_only:
        return
    receipt = {
        "version": 1,
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "checked_at_unix": time.time(),
        "output_dir": str(out),
        "headroom_bytes": headroom,
        "required_bytes": required,
        "budget_bytes": cfg["output_budget_bytes"],
        "quota_reservation": False,
    }
    (out / "logs").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        TRANSFER_CODE_ROOT=str(ROOT),
        TRANSFER_OUTPUT=str(out),
        TRANSFER_RESUME="1" if args.resume else "0",
        TRANSFER_QUOTA_ADMISSION=json.dumps(receipt),
    )
    job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            f"--exclude={excluded}",
            f"--time={args.hours}:00:00",
            f"--output={out}/logs/job_%j.out",
            f"--error={out}/logs/job_%j.err",
            "--export=ALL",
            str(ROOT / "scripts/slurm_transfer_evidence_study.sh"),
        ],
        cwd=ROOT,
        env=env,
        text=True,
    ).strip()
    print(f"Submitted evidence job {job}; no training; cn07 excluded. Output: {out}", flush=True)


if __name__ == "__main__":
    main()
