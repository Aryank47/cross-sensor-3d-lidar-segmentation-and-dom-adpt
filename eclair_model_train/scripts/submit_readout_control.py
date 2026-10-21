#!/usr/bin/env python3
"""One GPU job for a matched classifier control; never refit on target labels."""

from __future__ import annotations

import argparse
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
from src.readout_control import OUTPUT_BUDGET, RESERVE, bind_plan, make_plan, safe_output  # noqa: E402
from src.study_runtime import atomic_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--output-dir", help="Default: STUDY_DIR/readout_control_v1")
    parser.add_argument("--arms", nargs="+", choices=["reference", "context", "fine"], default=["reference", "context"])
    parser.add_argument("--endpoints", nargs="+", choices=["best", "last"], default=["best", "last"])
    parser.add_argument("--hours", type=int, choices=range(1, 37), default=12, metavar="1..36")
    parser.add_argument("--max-queued", type=int, choices=[2, 3], default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate artifacts and quota without submitting or writing")
    args = parser.parse_args()
    study = Path(args.study_dir).expanduser().resolve()
    out = safe_output(study, args.output_dir or study / "readout_control_v1", args.arms)
    plan = make_plan(study, args.arms, args.endpoints, ROOT)
    bind_plan(out, plan, args.resume)
    if out.exists() and any(out.iterdir()) and not args.resume:
        parser.error("Output is not empty; choose a new directory or use --resume")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src", "scripts", "configs"], cwd=ROOT).returncode:
        parser.error("Commit local source changes before submission")
    nodes = subprocess.check_output(["sinfo", "-N", "-h", "-o", "%N"], text=True).splitlines()
    excluded = resolve_excluded(nodes)
    user = getpass.getuser()
    queued = subprocess.check_output(["squeue", "-u", user, "-h", "-o", "%i"], text=True).splitlines()
    if len(queued) + 1 > args.max_queued:
        parser.error("Adding one job would exceed the selected queue limit")
    # Find an existing parent for quota/free-space checks; don't create output on check-only.
    parent = next(p for p in (out, *out.parents) if p.is_dir())
    headroom = quota_headroom(parent)
    if min(headroom, shutil.disk_usage(parent).free) < OUTPUT_BUDGET + RESERVE:
        parser.error("Need 640 MiB of user quota and filesystem headroom")
    print(json.dumps({"unique_models": len(plan["tasks"]), "tasks": [
        {k: t[k] for k in ("arm", "epoch", "endpoints")} for t in plan["tasks"]],
        "output_dir": str(out), "excluded_nodes": excluded, "headroom_bytes": headroom}, indent=2), flush=True)
    if args.check_only:
        return
    receipt = {"user": user, "hostname": socket.gethostname(), "checked_at_unix": time.time(),
               "output_dir": str(out), "headroom_bytes": headroom, "budget_bytes": OUTPUT_BUDGET,
               "reserve_bytes": RESERVE}
    # A manifest is created only by the allocated runner. Admission is passed in the job environment.
    env = os.environ.copy()
    env.update(READOUT_CODE_ROOT=str(ROOT), READOUT_STUDY_DIR=str(study), READOUT_OUTPUT_DIR=str(out),
               READOUT_ARMS=",".join(args.arms), READOUT_ENDPOINTS=",".join(args.endpoints),
               READOUT_RESUME="1" if args.resume else "0", READOUT_QUOTA_ADMISSION=json.dumps(receipt))
    logs = Path(f"/scratch/{user}/e0_logs")
    logs.mkdir(parents=True, exist_ok=True)
    job = subprocess.check_output([
        "sbatch", "--parsable", f"--exclude={excluded}", f"--time={args.hours}:00:00", "--export=ALL",
        f"--output={logs}/readout_control_%j.out", f"--error={logs}/readout_control_%j.err",
        str(ROOT / "scripts/slurm_readout_control.sh"),
    ], cwd=ROOT, env=env, text=True).strip()
    # Avoid a post-submit filesystem failure obscuring a successfully submitted job.
    print(f"Submitted readout control job {job}; cn07 excluded. No training or head fitting.", flush=True)
    if args.resume:
        atomic_json(out / "resubmission.json", {"job_id": job, "admission": receipt})


if __name__ == "__main__":
    main()
