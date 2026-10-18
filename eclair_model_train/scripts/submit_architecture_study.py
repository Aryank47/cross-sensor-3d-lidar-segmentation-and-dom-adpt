#!/usr/bin/env python3
"""Submit at most two study jobs by default, always excluding the cn07 node."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.study_architecture import ARMS


def resolve_excluded(nodes):
    matches = sorted({n.strip() for n in nodes if n.strip().split(".")[0].lower() == "cn07"})
    if not matches:
        raise ValueError("Cannot resolve cn07 in sinfo; refuse submission without verified exclusion")
    return ",".join(matches)


def quota_headroom(path):
    """Quota's default units are 1 KiB blocks; -w prevents line wrapping."""
    device = subprocess.check_output(["df", "-P", str(path)], text=True).splitlines()[-1].split()[0]
    output = subprocess.run(["quota", "-w", "-u", getpass.getuser()], capture_output=True, text=True, check=False)
    for line in output.stdout.splitlines():
        cells = line.split()
        if cells and cells[0] == device and len(cells) >= 4:
            used, soft, hard = [int(v.rstrip("*")) for v in cells[1:4]]
            limits = [v for v in (soft, hard) if v > 0]
            if limits:
                return max(0, min(limits) - used) * 1024
    raise RuntimeError("Cannot determine user quota headroom; inspect quota before submitting")


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
    budget = len(args.arms) * 2 * 1024**3 + 2 * 1024**3
    if quota_headroom(out_root) < budget or shutil.disk_usage(out_root).free < budget:
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
