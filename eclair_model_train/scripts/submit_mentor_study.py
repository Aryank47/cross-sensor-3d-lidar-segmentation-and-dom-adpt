#!/usr/bin/env python3
"""Admit and submit exactly two serialized mentor arms, or one explicit resume."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mentor_study import (
    ARMS,
    MENTORS,
    REFERENCE_ROOT,
    make_mentor_config,
    validate_mentor_config,
)
from src.optional_return_study import verify_reference
from src.study_runtime import atomic_json

from scripts.submit_architecture_study import quota_headroom, resolve_excluded
from scripts.submit_feature_contract_study import load_frozen
from scripts.submit_optional_return_study import export_code

MIN_HEADROOM = 10 * 1024**3


def receipt(study, commit, headroom):
    return {
        "version": 1,
        "kind": "mentor_guidance",
        "user": getpass.getuser(),
        "study_dir": str(Path(study).resolve()),
        "commit": commit,
        "checked_at_unix": time.time(),
        "headroom_bytes": int(headroom),
        "quota_reservation": False,
    }


def allocation_admission(out):
    parent = Path(out).resolve().parent
    value = json.loads((parent / "quota_admission.json").read_text())
    code = json.loads((parent / "code_provenance.json").read_text())
    age = time.time() - value["checked_at_unix"]
    if (
        value["version"] != 1
        or value["kind"] != "mentor_guidance"
        or value["user"] != getpass.getuser()
        or Path(value["study_dir"]).resolve() != parent
        or value["commit"] != code["commit"]
        or value["headroom_bytes"] < MIN_HEADROOM
        or age < -300
    ):
        raise RuntimeError("Quota admission identity/capacity mismatch")
    try:
        headroom = quota_headroom(Path(out))
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        if age > 48 * 3600:
            raise RuntimeError("Admission expired; refresh on login node") from exc
        return {
            "mode": "validated_login_snapshot_compute_query_unavailable",
            "receipt": value,
            "snapshot_age_seconds": age,
            "live_error": str(exc),
            "quota_reservation": False,
        }
    if headroom < 5 * 1024**3:
        raise RuntimeError("Live quota below two-arm budget and reserve")
    return {
        "mode": "live_compute_quota",
        "headroom_bytes": headroom,
        "receipt": value,
        "quota_reservation": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True)
    p.add_argument("--hours", type=int, choices=(36, 48), default=36)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--resume-arm", choices=ARMS)
    modes.add_argument("--refresh-admission", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--out-root", default=f"/scratch/{getpass.getuser()}/e0_results/mentor_studies"
    )
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.study):
        p.error("Unsafe study name")
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
        p.error("Scientific source must be committed")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    study = Path(args.out_root).expanduser().resolve() / args.study
    existing = next(x for x in (study.parent, *study.parents) if x.exists())
    headroom = quota_headroom(existing)
    if min(headroom, shutil.disk_usage(existing).free) < MIN_HEADROOM:
        p.error("Need10GiB quota and filesystem headroom")
    if args.resume_arm or args.refresh_admission:
        if not (study / "code_provenance.json").is_file():
            p.error("No admitted study")
        commit = json.loads((study / "code_provenance.json").read_text())["commit"]
    elif study.exists():
        p.error("Study exists; use explicit resume or a new name")
    if args.refresh_admission:
        if not args.dry_run:
            atomic_json(
                study / "quota_admission.json", receipt(study, commit, headroom)
            )
        print(
            json.dumps(
                {
                    "refreshed": not args.dry_run,
                    "headroom_bytes": headroom,
                    "submitted": False,
                }
            )
        )
        return
    queued = subprocess.check_output(
        ["squeue", "-r", "-u", getpass.getuser(), "-h", "-o", "%i"], text=True
    ).splitlines()
    needed = 1 if args.resume_arm else 2
    if len(queued) + needed > 2:
        p.error("Two-job limit would be exceeded")
    if args.resume_arm:
        out = study / args.resume_arm
        state = json.loads((out / "STATUS.json").read_text())
        if state["status"] == "complete":
            print("Arm already complete; no submission")
            return
        if (out / "numerical_failure.json").exists():
            p.error("Numerical screen failed; no automatic recipe restart")
        if not (out / "config_resolved.yaml").is_file():
            p.error("No candidate state to resume")
        old = json.loads((study / "submission.json").read_text())
        if any(
            q.split("_")[0] == r["job_id"].split("_")[0]
            for q in queued
            for r in old["submissions"]
        ):
            p.error("Study already active")
    else:
        spec = json.loads(
            (root / "configs/mentor_study/reference_manifest.json").read_text()
        )
        verify_reference(Path(spec["remote_root"]), spec)
        frozen = load_frozen(root)
        for arm in ARMS:
            cfg = make_mentor_config(frozen, arm, study / arm)
            validate_mentor_config(cfg, frozen)
            mentor = Path(cfg["study"]["mentor"]["checkpoint"])
            if not mentor.is_file() or mentor.stat().st_size < 1024**2:
                p.error("Frozen mentor checkpoint missing")
            old = json.loads(
                (
                    Path(REFERENCE_ROOT)
                    / MENTORS[arm]["reference_arm"]
                    / "evaluation_manifest.json"
                ).read_text()
            )["checkpoints"]["best"]
            if (
                old["epoch"] != MENTORS[arm]["epoch"]
                or old["model_sha256"] != MENTORS[arm]["model_sha256"]
            ):
                p.error("Mentor source-selected identity changed")
    excluded = resolve_excluded(
        subprocess.check_output(
            ["sinfo", "-N", "-h", "-o", "%N"], text=True
        ).splitlines()
    )
    logs = Path(f"/scratch/{getpass.getuser()}/e0_logs")
    suffix = f"mentor_{args.resume_arm}_%j" if args.resume_arm else "mentor_%A_%a"
    command = [
        "sbatch",
        "--parsable",
        f"--exclude={excluded}",
        f"--time={args.hours}:00:00",
        "--no-requeue",
        "--job-name=mentor_study",
        f"--output={logs}/{suffix}.out",
        f"--error={logs}/{suffix}.err",
        "--export=ALL",
        "--chdir",
        str(study / "code"),
    ]
    if not args.resume_arm:
        command += ["--array=0-1%1"]
    command += [str(study / "code/scripts/slurm_mentor_study.sh")]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "submitted": False,
                    "command": command,
                    "commit": commit,
                    "headroom_bytes": headroom,
                    "trainings": needed,
                    "automatic_reverse_jobs": 0,
                    "study": str(study),
                }
            )
        )
        return
    if not args.resume_arm:
        study.mkdir(parents=True, exist_ok=False)
        code = export_code(root, study / "code", commit)
        atomic_json(study / "code_provenance.json", code)
        for name in spec["files"]:
            target = study / "reference" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(spec["remote_root"]) / name, target)
        verify_reference(study / "reference", spec)
        atomic_json(study / "reference_manifest.json", spec)
        shutil.copyfile(
            root / "configs/mentor_study/decision_rules.json",
            study / "decision_rules.json",
        )
    atomic_json(study / "quota_admission.json", receipt(study, commit, headroom))
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        MENTOR_STUDY_DIR=str(study),
        MENTOR_CODE_ROOT=str(study / "code"),
        MENTOR_ARM=args.resume_arm or "",
        MENTOR_RESUME="1" if args.resume_arm else "0",
    )
    job = (
        subprocess.check_output(command, cwd=root, env=env, text=True)
        .strip()
        .split(";")[0]
    )
    path = study / "submission.json"
    saved = json.loads(path.read_text()) if path.is_file() else {"submissions": []}
    saved["submissions"].append(
        {
            "job_id": job,
            "commit": commit,
            "resume_arm": args.resume_arm,
            "arms": [args.resume_arm] if args.resume_arm else list(ARMS),
            "time_unix": time.time(),
        }
    )
    atomic_json(path, saved)
    print(
        json.dumps(
            {
                "job_id": job,
                "array_tasks": {str(i): a for i, a in enumerate(ARMS)}
                if not args.resume_arm
                else None,
                "excluded": excluded,
                "commit": commit,
                "study": str(study),
                "logs": str(logs / suffix),
                "automatic_reverse_jobs": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
