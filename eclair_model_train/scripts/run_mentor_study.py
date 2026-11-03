#!/usr/bin/env python3
"""One mentor arm: frozen source guidance, training, evaluations and comparison."""

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
from src.feature_contract_study import (
    STORAGE,
    evaluate_tiles,
    load_inventory,
    model_digest,
)
from src.features import assert_checkpoint_feature_config
from src.mentor_study import (
    ARMS,
    arm_report,
    collect_bundle,
    compare_if_ready,
    make_mentor_config,
    original_student_identity,
    validate_mentor_config,
)
from src.optional_return_study import constant_diagnostic, verify_reference
from src.study_runtime import atomic_json, sha256, storage_guard

from scripts.submit_feature_contract_study import load_frozen
from scripts.submit_mentor_study import allocation_admission


def verify_code(root, study, out, resume):
    source = json.loads((study / "code_provenance.json").read_text())
    if root.resolve() != (study / "code").resolve():
        raise RuntimeError("Allocation must execute the immutable export")
    for name, digest in source["files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Exported code was changed: {name}")
    old = (study / "reference/observed/source_manifest.sha256").read_text().splitlines()
    allowed = {"train.py", "src/study_runtime.py", "src/feature_contract_study.py"}
    for line in old:
        digest, name = line.split("  ", 1)
        if name not in allowed and sha256(root / name) != digest:
            raise RuntimeError(
                f"Reused-control source primitive changed unexpectedly: {name}"
            )
    identity = sha256(study / "code_provenance.json")
    path = out / "source_manifest.sha256"
    text = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(source["files"].items())
    )
    if resume and (not path.is_file() or path.read_text() != text):
        raise RuntimeError("Mixed-code resume refused")
    if not resume:
        path.write_text(text)
        (out / "git_commit.txt").write_text(source["commit"] + "\n")
        with tarfile.open(out / "source_snapshot.tar.gz", "w:gz") as archive:
            for name in source["files"]:
                archive.add(root / name, arcname=name)
    return identity


def run(args):
    import yaml

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if socket.gethostname().split(".")[0].lower() == "cn07":
        raise RuntimeError("cn07 is forbidden")
    study = Path(args.study_dir).resolve()
    out = study / args.arm
    resume = args.resume_flag == "1"
    if not resume and out.exists():
        raise FileExistsError("Candidate output exists; refusing overwrite")
    if resume and not (out / "config_resolved.yaml").is_file():
        raise FileNotFoundError("No candidate state to resume")
    out.mkdir(exist_ok=resume)
    lock = (out / ".allocation.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("Another allocation owns the candidate") from exc
    phase = "admission"

    def status(value, **fields):
        atomic_json(
            out / "STATUS.json",
            {
                "status": value,
                "kind": "mentor_guidance",
                "arm": args.arm,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "phase": phase,
                **fields,
            },
        )

    try:
        spec = json.loads((study / "reference_manifest.json").read_text())
        verify_reference(study / "reference", spec)
        frozen = load_frozen(root)
        cfg = make_mentor_config(frozen, args.arm, out)
        validate_mentor_config(cfg, frozen)
        if original_student_identity(cfg) != original_student_identity(
            json.loads((study / "reference/observed/config_resolved.json").read_text())
        ):
            raise RuntimeError(
                "Recipe changed beyond approved optional-return intervention"
            )
        if resume:
            validate_mentor_config(load_yaml(out / "config_resolved.yaml"), frozen)
        code_id = verify_code(root, study, out, resume)
        atomic_json(out / "storage_admission.json", allocation_admission(out))
        if resume:
            (out / "STOP_REQUESTED").unlink(missing_ok=True)
        path = out / "config_resolved.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        atomic_json(out / "config_resolved.json", cfg)
        status("preflight")
        storage_guard(out, 512 * 1024**2, STORAGE)
        import MinkowskiEngine as ME
        import numpy as np
        import torch

        if torch.cuda.device_count() != 2:
            raise RuntimeError("Frozen global batch requires two allocated GPUs")
        runtime = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "minkowski_engine": str(ME.__version__),
            "numpy": np.__version__,
            "python": platform.python_version(),
            "world_size": 2,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)],
            "hostname": socket.gethostname(),
        }
        reference = json.loads(
            (study / "reference/observed/runtime_identity.json").read_text()
        )
        for key in (
            "torch",
            "cuda",
            "minkowski_engine",
            "numpy",
            "python",
            "world_size",
        ):
            if runtime[key] != reference[key]:
                raise RuntimeError(f"Completed-control runtime differs: {key}")
        if resume and (out / "runtime_identity.json").is_file():
            previous = json.loads((out / "runtime_identity.json").read_text())
            if any(
                runtime[k] != previous[k]
                for k in (
                    "torch",
                    "cuda",
                    "minkowski_engine",
                    "numpy",
                    "python",
                    "world_size",
                )
            ):
                raise RuntimeError("Runtime changed across resume")
        atomic_json(out / "runtime_identity.json", runtime)
        logger = logging.getLogger("mentor_study")
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())
        phase = "dataset_inventory"
        inventory = load_inventory(cfg, logger)
        if inventory != json.loads(
            (study / "reference/observed/dataset_inventory.json").read_text()
        ):
            raise RuntimeError(
                "Source/cache/split identities differ from completed control"
            )
        atomic_json(out / "dataset_inventory.json", inventory)
        atomic_json(
            out / "study_specification.json",
            {
                "kind": "mentor_guidance",
                "arm": args.arm,
                "mentor": cfg["study"]["mentor"],
                "guidance_weight": 0.1,
                "guidance_ramp_epochs": 10,
                "guidance": "native_correct_full_distribution_macro_common_classes",
                "primary_input": "observed",
                "diagnostic_input": "constant_pair_best_only",
                "source_only_selection": True,
                "automatic_reverse_jobs": 0,
                "code_identity": code_id,
                "fixed_rules_sha256": sha256(study / "decision_rules.json"),
            },
        )
        training_file = out / "training_status.json"
        trained = (
            training_file.is_file()
            and json.loads(training_file.read_text())["status"] == "training_complete"
        )
        if resume and (out / "numerical_failure.json").is_file():
            raise RuntimeError("Numerical health failed; no automatic restart")
        if (
            resume
            and not trained
            and not (out / "checkpoints/last.pt").is_file()
            and any((out / n).is_file() for n in ("metrics.csv", "study_epochs.jsonl"))
        ):
            raise RuntimeError(
                "Training history exists without a resumable checkpoint; use a new study after review"
            )
        if not trained:
            phase = "gpu_preflight"
            status("preflight")
            if not resume or not (out / "preflight.json").is_file():
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/preflight_mentor_study.py",
                        "--config",
                        str(path),
                        "--out",
                        str(out / "preflight.json"),
                    ],
                    check=True,
                )
            preflight = json.loads((out / "preflight.json").read_text())
            if preflight["status"] != "passed_mentor_gpu_smoke":
                raise RuntimeError("Mentor GPU preflight failed")
            if (
                preflight["initial_model_sha256"]
                != json.loads(
                    (study / "reference/observed/preflight.json").read_text()
                )["initial_model_sha256"]
            ):
                raise RuntimeError(
                    "Candidate initialization differs from completed control"
                )
            if resume and (out / "checkpoints/last.pt").is_file():
                cfg["run"]["resume_from"] = str(out / "checkpoints/last.pt")
                path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                atomic_json(out / "config_resolved.json", cfg)
            phase = "training"
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
                    str(path),
                ],
                check=True,
            )
            training = json.loads(training_file.read_text())
            if training["status"] != "training_complete":
                status(
                    "needs_resume",
                    last_completed_epoch=training.get("last_completed_epoch"),
                )
                return
        phase = "canonical_evaluation"
        status("evaluating")
        from src.model import build_model

        deadline = float(os.environ.get("STUDY_DEADLINE_UNIX", "inf"))
        checkpoints = {}
        for endpoint in ("best", "last"):
            if time.time() > deadline - 900:
                status("needs_resume", checkpoint=endpoint)
                return
            checkpoint = torch.load(
                out / "checkpoints" / f"{endpoint}.pt",
                map_location="cpu",
                weights_only=False,
            )
            assert_checkpoint_feature_config(checkpoint, cfg)
            validate_mentor_config(checkpoint["cfg"], frozen)
            model_id = model_digest(checkpoint["model_state"])
            relative = f"evaluation/{endpoint}/{model_id}"
            checkpoints[endpoint] = {
                "model_sha256": model_id,
                "epoch": int(checkpoint["epoch"]),
                "relative_dir": relative,
                "selection": "source_native_validation_miou"
                if endpoint == "best"
                else "final",
            }
            model = build_model(10, 11, cfg=cfg).to("cuda:0")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            del checkpoint
            identity = {
                "model_sha256": model_id,
                "source_manifest": code_id,
                "dataset_inventory_sha256": sha256(out / "dataset_inventory.json"),
            }
            for domain in ("eclair", "dales"):
                if not evaluate_tiles(
                    model,
                    cfg,
                    identity,
                    domain,
                    out / relative,
                    deadline,
                    logger,
                    paired_diagnostic=constant_diagnostic
                    if endpoint == "best"
                    else None,
                ):
                    status("needs_resume", checkpoint=endpoint, domain=domain)
                    return
            del model
            torch.cuda.empty_cache()
            atomic_json(
                out / "evaluation_manifest.json",
                {
                    "checkpoints": checkpoints,
                    "primary_feature_mode": "observed",
                    "target_checkpoint_selection": False,
                    "constant_diagnostic": "best_only_not_deployment",
                },
            )
        phase = "analysis"
        status("analyzing")
        report = arm_report(study, args.arm)
        atomic_json(
            out / "resource_summary.json",
            {
                "run_bytes": sum(
                    p.stat().st_size for p in out.rglob("*") if p.is_file()
                ),
                "run_budget_bytes": STORAGE["run_budget_bytes"],
                "world_size": 2,
                "training_epochs": 100,
            },
        )
        phase = "complete"
        status(
            "complete",
            classification=report["classification_vs_original"],
            automatic_reverse_jobs=0,
        )
        print(json.dumps(collect_bundle(study, args.arm)), flush=True)
        with (study / ".comparison.lock").open("a") as comparison_lock:
            fcntl.flock(comparison_lock, fcntl.LOCK_EX)
            comparison = compare_if_ready(study)
            if comparison is not None:
                print(
                    json.dumps(
                        {
                            "comparison": comparison["classification"],
                            "bundle": collect_bundle(study),
                        }
                    ),
                    flush=True,
                )
    except Exception as exc:
        status(
            "numerical_screen_failed"
            if (out / "numerical_failure.json").is_file()
            else "failed",
            error=f"{type(exc).__name__}: {exc}",
            completed_work_retained=True,
            automatic_requeue=False,
        )
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--resume-flag", choices=("0", "1"), default="0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
