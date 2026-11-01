#!/usr/bin/env python3
"""Submit the two-arm ECLAIR feature study with verified user-quota admission."""

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
from scripts.submit_architecture_study import quota_headroom, resolve_excluded
from src.config_loader import load_yaml
from src.feature_contract_study import ARMS, make_config, validate_config

MIN_HEADROOM = 10 * 1024**3
RUN_BUDGET = 2 * 1024**3


def load_frozen(root):
    wrapped = load_yaml(root / "configs/feature_contract/frozen_eclair_29306.yaml")
    if wrapped.get("status") != "reference_only_not_a_training_config":
        raise ValueError("Invalid frozen reference wrapper")
    return wrapped["historical_run_config"]


def admission_receipt(study_dir, arms, headroom):
    return {
        "version": 1,
        "kind": "feature_contract",
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "checked_at_unix": time.time(),
        "study_dir": str(Path(study_dir).resolve()),
        "arms": list(arms),
        "quota_headroom_bytes": int(headroom),
        "run_budget_bytes": RUN_BUDGET,
        "minimum_headroom_bytes": MIN_HEADROOM,
        "quota_reservation": False,
    }


def allocation_admission(out):
    """Compute NFS quota may be unavailable; accept only this submission's receipt."""
    out = Path(out).resolve()
    try:
        receipt = json.loads(os.environ["FEATURE_QUOTA_ADMISSION"])
        age = time.time() - float(receipt["checked_at_unix"])
        arms = receipt["arms"]
        valid = (
            receipt["version"] == 1
            and receipt["kind"] == "feature_contract"
            and receipt["user"] == getpass.getuser()
            and Path(receipt["study_dir"]).resolve() == out.parent
            and isinstance(arms, list)
            and 1 <= len(arms) <= 2
            and len(set(arms)) == len(arms)
            and all(a in ARMS for a in arms)
            and out.name in arms
            and -300 <= age <= 48 * 3600
            and receipt["run_budget_bytes"] == RUN_BUDGET
            and receipt["minimum_headroom_bytes"] == MIN_HEADROOM
            and receipt["quota_headroom_bytes"] >= MIN_HEADROOM
        )
        if not valid:
            raise ValueError("receipt identity, capacity or age mismatch")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "Missing/invalid quota admission; resubmit from the login node"
        ) from exc
    try:
        headroom = quota_headroom(out)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return {
            "mode": "validated_login_snapshot_compute_query_unavailable",
            "receipt": receipt,
            "snapshot_age_seconds": age,
            "live_query_error": str(exc),
            "quota_reservation": False,
        }
    if headroom < RUN_BUDGET + 512 * 1024**2:
        raise RuntimeError(
            "Live compute-node quota fell below this arm's admitted budget"
        )
    return {
        "mode": "live_compute_query",
        "headroom_bytes": headroom,
        "receipt": receipt,
        "quota_reservation": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--hours", type=int, choices=[36, 48], default=36)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out-root",
        default=f"/scratch/{getpass.getuser()}/e0_results/feature_contract_studies",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.study):
        parser.error("Use a short safe study name, without path separators")
    if len(set(args.arms)) != len(args.arms):
        parser.error("Repeated arms are forbidden")
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src",
            "scripts",
            "configs",
            "train.py",
        ],
        cwd=root,
        text=True,
    )
    if dirty.strip():
        parser.error("Commit the study code/configuration before submission")
    out_root = Path(args.out_root).expanduser().resolve()
    if not out_root.exists():
        if args.dry_run:
            quota_path = next(p for p in out_root.parents if p.exists())
        else:
            out_root.mkdir(parents=True)
            quota_path = out_root
    else:
        quota_path = out_root
    study_dir = out_root / args.study
    arms = list(args.arms)
    if args.resume:
        for arm in arms:
            if not (study_dir / arm / "config_resolved.yaml").is_file():
                parser.error(f"No study config to resume: {study_dir / arm}")
        arms = [
            a
            for a in arms
            if not (study_dir / a / "STATUS.json").is_file()
            or json.loads((study_dir / a / "STATUS.json").read_text()).get("status")
            != "complete"
        ]
        if not arms:
            print("Both requested arms are already complete; no jobs submitted.")
            return
    elif study_dir.exists():
        parser.error(f"{study_dir} already exists; use --resume or a new study name")
    frozen = load_frozen(root)
    for arm in arms:
        validate_config(make_config(frozen, arm, study_dir / arm), frozen)
    nodes = subprocess.check_output(
        ["sinfo", "-N", "-h", "-o", "%N"], text=True
    ).splitlines()
    excluded = resolve_excluded(nodes)
    queued = subprocess.check_output(
        ["squeue", "-u", getpass.getuser(), "-h", "-o", "%i"], text=True
    ).splitlines()
    if len(queued) + len(arms) > 2:
        parser.error(
            f"Already {len(queued)} jobs queued; adding {len(arms)} exceeds this study's two-job limit"
        )
    submission_path = study_dir / "submission.json"
    previous = (
        json.loads(submission_path.read_text())
        if submission_path.is_file()
        else {"submissions": []}
    )
    if any(r["arm"] in arms and r["job_id"] in queued for r in previous["submissions"]):
        parser.error("A requested arm already has a queued/running allocation")
    headroom = quota_headroom(quota_path)
    if min(headroom, shutil.disk_usage(quota_path).free) < MIN_HEADROOM:
        parser.error(
            "Need at least 10 GiB of verified user-quota AND filesystem headroom"
        )
    logs = Path(f"/scratch/{getpass.getuser()}/e0_logs")
    receipt = admission_receipt(study_dir, arms, headroom)
    if not args.dry_run:
        if not args.resume:
            study_dir.mkdir(exist_ok=False)
        logs.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        command = [
            "sbatch",
            "--parsable",
            f"--exclude={excluded}",
            f"--time={args.hours}:00:00",
            f"--job-name=returns_{arm}",
            f"--output={logs}/returns_{arm}_%j.out",
            f"--error={logs}/returns_{arm}_%j.err",
            "--export=ALL",
            "--chdir",
            str(root),
            str(root / "scripts/slurm_feature_contract_study.sh"),
        ]
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "command": command,
                        "run_dir": str(study_dir / arm),
                        "quota_headroom_bytes": headroom,
                        "submitted": False,
                    }
                )
            )
            continue
        env = os.environ.copy()
        env.update(
            FEATURE_CODE_ROOT=str(root),
            FEATURE_STUDY_DIR=str(study_dir),
            FEATURE_STUDY_ARM=arm,
            FEATURE_RESUME="1" if args.resume else "0",
            FEATURE_QUOTA_ADMISSION=json.dumps(receipt),
        )
        try:
            job = (
                subprocess.check_output(command, cwd=root, env=env, text=True)
                .strip()
                .split(";")[0]
            )
        except Exception:
            print(
                "Submission interrupted. Earlier printed job IDs may already be queued; inspect squeue before retrying.",
                file=sys.stderr,
            )
            raise
        previous["submissions"].append(
            {
                "arm": arm,
                "job_id": job,
                "submitted_at": time.time(),
                "resume": args.resume,
            }
        )
        submission_path.write_text(json.dumps(previous, indent=2))
        print(
            json.dumps(
                {
                    "arm": arm,
                    "job_id": job,
                    "run_dir": str(study_dir / arm),
                    "excluded": excluded,
                    "resume": args.resume,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
