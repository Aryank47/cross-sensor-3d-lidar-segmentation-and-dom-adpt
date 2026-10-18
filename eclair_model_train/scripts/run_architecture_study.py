#!/usr/bin/env python3
"""One arm per allocation; continue training or evaluation using the same command."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config_loader import load_yaml
from src.study_runtime import atomic_json, sha256, storage_guard


def provenance(root, out, resume):
    tracked = subprocess.check_output(["git", "ls-files", "-z", "."], cwd=root).decode().split("\0")
    names = sorted(n for n in tracked if n and Path(n).suffix in (".py", ".yaml", ".sh", ".md"))
    manifest = "".join(f"{sha256(root / name)}  {name}\n" for name in names)
    destination = out / "source_manifest.sha256"
    if resume:
        if not destination.exists() or destination.read_text() != manifest:
            raise RuntimeError("Code changed since the original allocation; refusing mixed-code resume")
    else:
        destination.write_text(manifest)
        with tarfile.open(out / "source_snapshot.tar.gz", "w:gz") as archive:
            for name in names:
                archive.add(root / name, arcname=name)
        (out / "git_commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True))
    return sha256(destination)


def model_digest(state):
    import hashlib

    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        result.update(key.encode())
        result.update(str(value.shape).encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def run(args):
    import yaml

    root = Path(__file__).resolve().parents[1]
    if socket.gethostname().split(".")[0].lower() == "cn07":
        raise RuntimeError("cn07 is excluded from all study computation")
    os.chdir(root)
    out = Path(args.study_dir).resolve() / args.arm
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".allocation.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("This arm already has an active allocation; its status was not modified") from exc
    resume = args.resume_flag == "1"
    source_id = provenance(root, out, resume)
    (out / "STOP_REQUESTED").unlink(missing_ok=True)
    cfg = load_yaml(root / f"configs/study_dales_{args.arm}120.yaml")
    cfg["run"]["out_dir"] = str(out)
    cp = out / "checkpoints/last.pt"
    if resume and cp.exists():
        cfg["run"]["resume_from"] = str(cp)
    cfg_path = out / "config_resolved.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    status_path = out / "STATUS.json"

    def status(value, **fields):
        atomic_json(
            status_path,
            {
                "status": value,
                "arm": args.arm,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "source_manifest_sha256": source_id,
                **fields,
            },
        )

    status("preflight")
    storage_guard(out, 512 * 1024**2, cfg["study"]["storage"])
    import shutil

    from scripts.submit_architecture_study import allocation_quota_admission

    admission = allocation_quota_admission(
        out, cfg["study"]["storage"]["run_budget_bytes"] + 512 * 1024**2
    )
    atomic_json(
        out / "storage_admission.json",
        {
            **admission,
            "filesystem_free_bytes": shutil.disk_usage(out).free,
            "checked_at_unix": time.time(),
            "run_budget_bytes": cfg["study"]["storage"]["run_budget_bytes"],
        },
    )
    import MinkowskiEngine as ME
    import numpy as np
    import torch

    runtime = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "minkowski_engine": str(getattr(ME, "__version__", "unknown")),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "hostname": socket.gethostname(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    runtime_path = out / "runtime_identity.json"
    if resume and runtime_path.exists():
        old_runtime = json.loads(runtime_path.read_text())
        for key in ("torch", "cuda", "minkowski_engine", "numpy", "python"):
            if old_runtime[key] != runtime[key]:
                raise RuntimeError(f"Runtime version changed across resume: {key}")
    atomic_json(runtime_path, runtime)
    # Validate all data/cache inventories BEFORE spending a long training allocation.
    from src.study_analysis import load_dataset

    inventories = []
    for domain, split in (("dales", "train"), ("dales", "val"), ("dales", "test"), ("eclair", "test")):
        _, _, identity = load_dataset(domain, split, cfg, logging.getLogger("study_preflight"))
        inventories.append(identity)
    inventory_path = out / "dataset_inventory.json"
    if resume and inventory_path.exists() and json.loads(inventory_path.read_text()) != {"datasets": inventories}:
        raise RuntimeError("Dataset/cache identities changed; original inventory preserved")
    atomic_json(inventory_path, {"datasets": inventories})
    inventory_id = sha256(out / "dataset_inventory.json")
    original_inventory = out / "dataset_inventory_sha256.txt"
    if resume and original_inventory.exists() and original_inventory.read_text().strip() != inventory_id:
        raise RuntimeError("Dataset/cache identities changed since the original allocation")
    original_inventory.write_text(inventory_id + "\n")
    atomic_json(
        out / "study_specification.json",
        {
            "config": cfg,
            "source_manifest_sha256": source_id,
            "source_only_selection": True,
            "guardrail": "both target means improve; no target class/source mean regression",
            "single_seed_screen": True,
            "excluded_node": "cn07 (scheduler name resolved at submission)",
        },
    )
    completed = (
        json.loads((out / "training_status.json").read_text()).get("status") == "training_complete"
        if (out / "training_status.json").exists()
        else False
    )
    if not completed:
        subprocess.run(
            [
                sys.executable,
                "scripts/preflight_architecture_study.py",
                "--config",
                str(cfg_path),
                "--gpu-smoke",
                "--out",
                str(out / "preflight.json"),
            ],
            check=True,
        )
        status("training")
        port = 10000 + int(os.environ.get("SLURM_JOB_ID", "1")) % 50000
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--nproc_per_node=2",
                "--rdzv_backend=c10d",
                f"--rdzv_endpoint=127.0.0.1:{port}",
                "train.py",
                "--config",
                str(cfg_path),
            ],
            check=True,
        )
        training = json.loads((out / "training_status.json").read_text())
        if training["status"] != "training_complete":
            status("needs_resume", phase="training", last_completed_epoch=training["last_completed_epoch"])
            return
    status("evaluating", phase="canonical_and_bounded_analysis")
    from src.model import build_model
    from src.study_analysis import bounded_analysis, evaluate_tiles, fingerprint

    device = torch.device("cuda:0")
    deadline = float(os.environ.get("STUDY_DEADLINE_UNIX", "inf"))
    logger = logging.getLogger("architecture_study")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    references = {}
    for name in ("best", "last"):
        if time.time() > deadline - 900:
            status("needs_resume", phase="evaluation")
            return
        checkpoint = torch.load(out / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=False)
        model_id = model_digest(checkpoint["model_state"])
        references[name] = {"model_sha256": model_id, "epoch": int(checkpoint["epoch"])}
        stage = out / "evaluation" / model_id
        stage.mkdir(parents=True, exist_ok=True)
        model = build_model(int(cfg["model"]["in_channels"]), int(cfg["model"]["out_channels"]), cfg=cfg).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        del checkpoint
        identity = fingerprint({"model_sha256": model_id, "source_manifest": source_id, "datasets": inventory_id})
        for domain in ("dales", "eclair"):
            if not evaluate_tiles(model, cfg, identity, domain, stage, deadline, logger):
                status("needs_resume", phase="evaluation", checkpoint=name, domain=domain)
                return
        # Both unique primary/final weights get the same bounded analysis.
        marker = stage / "bounded_complete.json"
        valid_marker = marker.is_file() and all(
            (stage / filename).is_file() and sha256(stage / filename) == digest
            for filename, digest in json.loads(marker.read_text()).items()
        )
        if not valid_marker and not bounded_analysis(model, cfg, stage, deadline, logger):
            status("needs_resume", phase="bounded_analysis", checkpoint=name)
            return
        del model
        torch.cuda.empty_cache()
        atomic_json(
            out / "evaluation_manifest.json",
            {
                "checkpoints": references,
                "source_manifest_sha256": source_id,
                "dataset_inventory_sha256": inventory_id,
                "target_checkpoint_selection": False,
            },
        )
    atomic_json(
        out / "resource_summary.json",
        {
            "checkpoint_bytes": {p.name: p.stat().st_size for p in (out / "checkpoints").glob("*.pt")},
            "run_bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
            "storage_budget_bytes": cfg["study"]["storage"]["run_budget_bytes"],
            "training_resource_curves": "metrics.csv",
            "gpu_preflight": "preflight.json",
        },
    )
    (out / "MEASUREMENTS_AND_LIMITS.md").write_text(
        "# Measurements and limits\n\n"
        "Full common-space test matrices: evaluation/*/{dales,eclair}/metrics.json. "
        "Native GT rows retain ECLAIR wire/pole/tower subtypes. Source validation selected best.pt. "
        "last.pt is a prespecified secondary endpoint, not a target-selected replacement.\n\n"
        "Source exposure and class-presence counts include all ranks. Per-class classifier gradients "
        "are bounded focal-only diagnostic objectives; Lovasz is logged separately. "
        "Frozen ridge heads use only source train/validation cohorts. "
        "Perturbations use the same retained GT points and fixed quantization origin; "
        "they are not physical acquisition simulators. Geometry uses a 1m radius on capped cohorts.\n\n"
        "One seed is screening only. Target datasets already informed design. "
        "No intermediate checkpoints, per-point outputs, or feature banks are retained.\n"
    )
    atomic_json(
        out / "bundle_manifest.json",
        {
            str(p.relative_to(out)): sha256(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name not in ("bundle_manifest.json", "STATUS.json", ".allocation.lock")
        },
    )
    status(
        "complete",
        training_epochs=int(cfg["epochs"]),
        evaluation_manifest="evaluation_manifest.json",
        peak_run_bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
        limitations=[
            "one-seed screen",
            "target domains already informed design",
            "no intermediate weights retained",
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--arm", choices=["reference", "context", "fine"], required=True)
    parser.add_argument("--resume-flag", choices=["0", "1"], default="0")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        out = Path(args.study_dir).resolve() / args.arm
        try:
            atomic_json(
                out / "STATUS.json",
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "resume_uses_previous_atomic_checkpoint": True,
                },
            )
        except OSError:
            pass  # stderr remains available when even a tiny status write fails
        raise


if __name__ == "__main__":
    main()
