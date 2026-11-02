#!/usr/bin/env python3
"""Submit one approved candidate, with immutable code and copied reference evidence."""

from __future__ import annotations

import argparse
import getpass
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.optional_return_study import (
    make_optional_config,
    validate_optional_config,
    verify_reference,
)
from src.study_runtime import atomic_json, sha256

from scripts.submit_architecture_study import quota_headroom, resolve_excluded
from scripts.submit_feature_contract_study import load_frozen

MIN_HEADROOM = 10 * 1024**3


def receipt(study, commit, headroom):
    return {
        "version": 1,
        "kind": "optional_returns",
        "user": getpass.getuser(),
        "study_dir": str(Path(study).resolve()),
        "commit": commit,
        "checked_at_unix": time.time(),
        "headroom_bytes": int(headroom),
        "quota_reservation": False,
    }


def allocation_admission(out):
    out = Path(out).resolve()
    parent = out.parent
    value = json.loads((parent / "quota_admission.json").read_text())
    code = json.loads((parent / "code_provenance.json").read_text())
    age = time.time() - float(value["checked_at_unix"])
    if (
        value["version"] != 1
        or value["kind"] != "optional_returns"
        or value["user"] != getpass.getuser()
        or Path(value["study_dir"]).resolve() != parent
        or value["commit"] != code["commit"]
        or value["headroom_bytes"] < MIN_HEADROOM
        or age < -300
    ):
        raise RuntimeError("Quota admission identity/capacity mismatch")
    try:
        headroom = quota_headroom(out)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        if age > 48 * 3600:
            raise RuntimeError(
                "Quota snapshot expired; refresh admission on the login node before resuming"
            ) from exc
        return {
            "mode": "validated_login_snapshot_compute_query_unavailable",
            "receipt": value,
            "snapshot_age_seconds": age,
            "live_error": str(exc),
            "quota_reservation": False,
        }
    if headroom < 2560 * 1024**2:
        raise RuntimeError("Live quota below candidate budget and reserve")
    return {
        "mode": "live_compute_quota",
        "headroom_bytes": headroom,
        "receipt": value,
        "quota_reservation": False,
    }


def export_code(root, destination, commit):
    prefix = (
        subprocess.check_output(
            ["git", "rev-parse", "--show-prefix"], cwd=root, text=True
        )
        .strip()
        .rstrip("/")
    )
    blob = subprocess.check_output(
        ["git", "archive", "--format=tar", f"{commit}:{prefix}"], cwd=root
    )
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        members = archive.getmembers()
        if sum(m.size for m in members) > 128 * 1024**2:
            raise OSError("Code export exceeds128MiB")
        for member in members:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("Unsafe code archive member")
        destination.mkdir(exist_ok=False)
        for member in members:
            path = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
    return {
        "commit": commit,
        "files": {
            p.relative_to(destination).as_posix(): sha256(p)
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        },
        "execution": "immutable_export_no_working_repository_dependency",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--hours", type=int, choices=(36, 48), default=36)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--refresh-admission", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out-root",
        default=f"/scratch/{getpass.getuser()}/e0_results/optional_return_studies",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.study):
        parser.error("Unsafe study name")
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
        parser.error("Study source must be committed")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    base = Path(args.out_root).expanduser().resolve()
    study = base / args.study
    out = study / "candidate"
    if args.resume or args.refresh_admission:
        if not (study / "code_provenance.json").is_file():
            parser.error("No admitted study to resume/refresh")
        commit = json.loads((study / "code_provenance.json").read_text())["commit"]
    elif study.exists():
        parser.error("Study already exists; use a new name or explicit resume")
    existing = next(p for p in (base, *base.parents) if p.exists())
    headroom = quota_headroom(existing)
    if min(headroom, shutil.disk_usage(existing).free) < MIN_HEADROOM:
        parser.error("Need10GiB user-quota and filesystem headroom")
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
                    "job_resubmitted": False,
                }
            )
        )
        return
    queued = subprocess.check_output(
        ["squeue", "-u", getpass.getuser(), "-h", "-o", "%i"], text=True
    ).splitlines()
    if len(queued) >= 2:
        parser.error("Two jobs already queued/running; no additional job submitted")
    if args.resume:
        status = json.loads((out / "STATUS.json").read_text())
        if status["status"] == "complete":
            print("Already complete; no job submitted")
            return
        if (out / "numerical_failure.json").exists():
            parser.error(
                "Numerical screen failed; no automatic recipe change or restart"
            )
        old = json.loads((study / "submission.json").read_text())
        if any(r["job_id"] in queued for r in old["submissions"]):
            parser.error("Candidate already active")
        if not (out / "config_resolved.yaml").is_file():
            parser.error("No candidate config to resume")
    else:
        spec = json.loads(
            (root / "configs/optional_returns/reference_manifest.json").read_text()
        )
        verify_reference(Path(spec["remote_root"]), spec)
        validate_optional_config(
            make_optional_config(load_frozen(root), out), load_frozen(root)
        )
    nodes = subprocess.check_output(
        ["sinfo", "-N", "-h", "-o", "%N"], text=True
    ).splitlines()
    excluded = resolve_excluded(nodes)
    logs = Path(f"/scratch/{getpass.getuser()}/e0_logs")
    command = [
        "sbatch",
        "--parsable",
        f"--exclude={excluded}",
        f"--time={args.hours}:00:00",
        "--job-name=optional_returns",
        f"--output={logs}/optional_returns_%j.out",
        f"--error={logs}/optional_returns_%j.err",
        "--export=ALL",
        "--chdir",
        str(study / "code"),
        str(study / "code/scripts/slurm_optional_return_study.sh"),
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "submitted": False,
                    "command": command,
                    "commit": commit,
                    "headroom_bytes": headroom,
                    "candidate_trainings": 1,
                    "automatic_reverse_jobs": 0,
                    "run_dir": str(out),
                }
            )
        )
        return
    if not args.resume:
        study.mkdir(parents=True, exist_ok=False)
        code = export_code(root, study / "code", commit)
        atomic_json(study / "code_provenance.json", code)
        reference = study / "reference"
        for name in spec["files"]:
            target = reference / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(spec["remote_root"]) / name, target)
        verify_reference(reference, spec)
        atomic_json(study / "reference_manifest.json", spec)
        shutil.copyfile(
            root / "configs/optional_returns/decision_rules.json",
            study / "decision_rules.json",
        )
    atomic_json(study / "quota_admission.json", receipt(study, commit, headroom))
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        OPTIONAL_STUDY_DIR=str(study),
        OPTIONAL_CODE_ROOT=str(study / "code"),
        OPTIONAL_RESUME="1" if args.resume else "0",
    )
    job = (
        subprocess.check_output(command, cwd=root, env=env, text=True)
        .strip()
        .split(";")[0]
    )
    path = study / "submission.json"
    submission = json.loads(path.read_text()) if path.is_file() else {"submissions": []}
    submission["submissions"].append(
        {
            "job_id": job,
            "commit": commit,
            "submitted_at_unix": time.time(),
            "resume": args.resume,
        }
    )
    atomic_json(path, submission)
    print(
        json.dumps(
            {
                "job_id": job,
                "excluded": excluded,
                "commit": commit,
                "run_dir": str(out),
                "logs": str(logs / f"optional_returns_{job}.out"),
                "automatic_reverse_jobs": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
