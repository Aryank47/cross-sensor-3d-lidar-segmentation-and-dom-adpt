#!/usr/bin/env python3
"""Submit at most two study jobs by default, always excluding the cn07 node."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.study_architecture import ARMS

RUN_BUDGET_BYTES = 2 * 1024**3
ADMISSION_RESERVE_BYTES = 2 * 1024**3
ADMISSION_MAX_AGE_SECONDS = 48 * 3600


def resolve_excluded(nodes):
    matches = sorted({n.strip() for n in nodes if n.strip().split(".")[0].lower() == "cn07"})
    if not matches:
        raise ValueError("Cannot resolve cn07 in sinfo; refuse submission without verified exclusion")
    return ",".join(matches)


def quota_headroom(path):
    """Quota's default units are 1 KiB blocks; -w prevents line wrapping."""
    device = subprocess.check_output(["df", "-P", str(path)], text=True).splitlines()[-1].split()[0]
    output = subprocess.run(
        ["quota", "-w", "-u", getpass.getuser()], capture_output=True, text=True, check=False
    )
    for line in output.stdout.splitlines():
        cells = line.split()
        if cells and cells[0] == device and len(cells) >= 4:
            try:
                used, soft, hard = [int(v.rstrip("*")) for v in cells[1:4]]
            except ValueError:
                continue
            limits = [v for v in (soft, hard) if v > 0]
            if limits:
                return max(0, min(limits) - used) * 1024
    detail = getattr(output, "stderr", "").strip()[-400:]
    raise RuntimeError(
        f"Cannot determine user quota headroom on {socket.gethostname()} for {device}"
        + (f": {detail}" if detail else "; no matching numeric quota row")
    )


def quota_admission(study_dir, arms, headroom):
    """Record a required live login-node check for this exact submission."""
    return {
        "version": 1,
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "checked_at_unix": time.time(),
        "study_dir": str(Path(study_dir).resolve()),
        "arms": list(arms),
        "quota_headroom_bytes": headroom,
        "run_budget_bytes": RUN_BUDGET_BYTES,
        "reserved_total_bytes": len(arms) * RUN_BUDGET_BYTES + ADMISSION_RESERVE_BYTES,
    }


def allocation_quota_admission(out, required_bytes):
    """Prefer live quota; otherwise use the explicitly recorded submission check.

    This receipt is an observation, not a filesystem quota reservation. Other
    account activity can consume headroom after submission. Never substitute
    filesystem-wide free space for an unknown user quota.
    """
    try:
        receipt = json.loads(os.environ["STUDY_QUOTA_ADMISSION"])
        arms = receipt["arms"]
        age = time.time() - float(receipt["checked_at_unix"])
        out = Path(out).resolve()
        valid = (
            receipt["version"] == 1
            and receipt["user"] == getpass.getuser()
            and Path(receipt["study_dir"]).resolve() == out.parent
            and isinstance(arms, list)
            and 1 <= len(arms) <= 3
            and len(set(arms)) == len(arms)
            and all(arm in ARMS for arm in arms)
            and out.name in arms
            and -300 <= age <= ADMISSION_MAX_AGE_SECONDS
            and receipt["run_budget_bytes"] == RUN_BUDGET_BYTES
            and required_bytes <= RUN_BUDGET_BYTES + ADMISSION_RESERVE_BYTES
            and receipt["reserved_total_bytes"]
            == len(arms) * RUN_BUDGET_BYTES + ADMISSION_RESERVE_BYTES
            and receipt["quota_headroom_bytes"] >= receipt["reserved_total_bytes"]
        )
        if not valid:
            raise ValueError("receipt identity, age or capacity does not match this allocation")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "Missing, invalid or older-than-48h submission quota receipt; "
            "submit again from the login node for a fresh quota check"
        ) from exc
    try:
        headroom = quota_headroom(out)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        result = {
            "quota_check_mode": "submission_snapshot_compute_query_unavailable",
            "submission_admission": receipt,
            "live_quota_error": str(exc),
            "quota_headroom_bytes": None,
            "quota_reservation": False,
        }
        print(
            "Storage admission: compute-node quota query unavailable; using the "
            f"login-node quota check from {age / 3600:.2f} hours ago. "
            "Per-run output caps and live filesystem checks remain active. "
            "This is not a live quota measurement or reservation.",
            flush=True,
        )
        return result
    if headroom < required_bytes:
        raise RuntimeError("Live user quota headroom fell below the admitted run budget")
    return {
        "quota_check_mode": "live_compute_query",
        "submission_admission": receipt,
        "quota_headroom_bytes": headroom,
        "quota_reservation": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, help="Reusable name for this complete study")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=["reference", "context"])
    parser.add_argument("--max-queued", type=int, choices=[2, 3], default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hours", type=int, choices=[36, 48], default=36)
    parser.add_argument("--out-root", default=f"/scratch/{getpass.getuser()}/e0_results/architecture_studies")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.study):
        parser.error("Study name must be a short safe directory name")
    if len(set(args.arms)) != len(args.arms):
        parser.error("Repeated arms would duplicate experiments")
    root = Path(__file__).resolve().parents[1]
    nodes = subprocess.check_output(["sinfo", "-N", "-h", "-o", "%N"], text=True).splitlines()
    excluded = resolve_excluded(nodes)
    user = getpass.getuser()
    queued = subprocess.check_output(["squeue", "-u", user, "-h", "-o", "%i"], text=True).splitlines()
    if len(queued) + len(args.arms) > args.max_queued:
        parser.error(f"Already {len(queued)} jobs; adding {len(args.arms)} exceeds limit {args.max_queued}")
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    budget = len(args.arms) * RUN_BUDGET_BYTES + ADMISSION_RESERVE_BYTES
    headroom = quota_headroom(out_root)
    if headroom < budget or shutil.disk_usage(out_root).free < budget:
        parser.error(f"Need {budget / 1024**3:.1f} GiB of filesystem AND quota headroom")
    study_dir = out_root / args.study
    logs = Path(f"/scratch/{user}/e0_logs")
    logs.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        run_dir = study_dir / arm
        if run_dir.exists() and not args.resume:
            parser.error(f"{run_dir} already exists; use --resume to continue, or a new study name")
        if args.resume and not (run_dir / "config_resolved.yaml").is_file():
            parser.error(f"No study config to resume at {run_dir}")
        if subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "train.py", "src", "scripts", "configs"],
            cwd=root,
            check=False,
        ).returncode:
            parser.error("Study requires committed code; commit changes before submission")
    for arm in args.arms:
        env = os.environ.copy()
        env.update(
            STUDY_CODE_ROOT=str(root),
            STUDY_DIR=str(study_dir),
            STUDY_ARM=arm,
            STUDY_RESUME="1" if args.resume else "0",
            STUDY_QUOTA_ADMISSION=json.dumps(quota_admission(study_dir, args.arms, headroom)),
        )
        command = [
            "sbatch",
            "--parsable",
            f"--exclude={excluded}",
            f"--time={args.hours}:00:00",
            f"--job-name=arch_{arm}",
            f"--output={logs}/arch_{arm}_%j.out",
            f"--error={logs}/arch_{arm}_%j.err",
            "--export=ALL",
            "--chdir",
            str(root),
            str(root / "scripts/slurm_architecture_study.sh"),
        ]
        job_id = subprocess.check_output(command, cwd=root, env=env, text=True).strip()
        print(
            json.dumps(
                {
                    "arm": arm,
                    "job_id": job_id,
                    "run_dir": str(study_dir / arm),
                    "excluded": excluded,
                    "resume": args.resume,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
