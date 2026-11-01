#!/usr/bin/env python3
"""One allocation per arm; completed-epoch training and tile-level evaluation resume."""

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
from scripts.submit_feature_contract_study import allocation_admission, load_frozen
from src.config_loader import load_yaml
from src.feature_contract_study import (
    ARMS,
    STORAGE,
    evaluate_tiles,
    load_inventory,
    make_config,
    model_digest,
    pair_identity,
    validate_config,
)
from src.features import assert_checkpoint_feature_config
from src.study_runtime import atomic_json, sha256, storage_guard


def provenance(root, out, resume):
    names = (
        subprocess.check_output(["git", "ls-files", "-z", "."], cwd=root)
        .decode()
        .split("\0")
    )
    names = sorted(
        n for n in names if n and Path(n).suffix in (".py", ".yaml", ".json", ".sh")
    )
    manifest = "".join(f"{sha256(root / n)}  {n}\n" for n in names)
    path = out / "source_manifest.sha256"
    if resume:
        if not path.is_file() or path.read_text() != manifest:
            raise RuntimeError(
                "Scientific code/configuration changed; mixed-code resume refused"
            )
    else:
        path.write_text(manifest)
        with tarfile.open(out / "source_snapshot.tar.gz", "w:gz") as archive:
            for name in names:
                archive.add(root / name, arcname=name)
        (out / "git_commit.txt").write_text(
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True)
        )
    return sha256(path)


def run(args):
    import yaml

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if socket.gethostname().split(".")[0].lower() == "cn07":
        raise RuntimeError("cn07 is forbidden for this study")
    out = Path(args.study_dir).resolve() / args.arm
    resume = args.resume_flag == "1"
    if not resume and out.exists():
        raise FileExistsError(
            f"Refusing to overwrite {out}; use --resume through the submit script"
        )
    if resume and not (out / "config_resolved.yaml").is_file():
        raise FileNotFoundError("No arm state to resume")
    out.mkdir(parents=True, exist_ok=resume)
    lock = (out / ".allocation.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(
            "Another allocation owns this arm; its status was not changed"
        ) from exc

    def status(value, **fields):
        atomic_json(
            out / "STATUS.json",
            {
                "status": value,
                "kind": "feature_contract",
                "arm": args.arm,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                **fields,
            },
        )

    phase = "admission"
    try:
        frozen = load_frozen(root)
        cfg = make_config(frozen, args.arm, out)
        if resume:
            old = load_yaml(out / "config_resolved.yaml")
            validate_config(old, frozen)
            if pair_identity(old) != pair_identity(cfg):
                raise RuntimeError("Frozen scientific configuration changed")
        validate_config(cfg, frozen)
        code_id = provenance(root, out, resume)
        if resume:
            (out / "STOP_REQUESTED").unlink(missing_ok=True)
        config_path = out / "config_resolved.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        atomic_json(out / "config_resolved.json", cfg)
        status("preflight", phase=phase, source_manifest_sha256=code_id)
        atomic_json(out / "storage_admission.json", allocation_admission(out))
        storage_guard(out, 512 * 1024**2, STORAGE)
        import MinkowskiEngine as ME
        import numpy as np
        import torch

        if torch.cuda.device_count() != 2:
            raise RuntimeError(
                "This paired recipe requires the same two allocated GPUs in each arm"
            )
        runtime = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "minkowski_engine": str(getattr(ME, "__version__", "unknown")),
            "numpy": np.__version__,
            "python": platform.python_version(),
            "world_size": 2,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)],
            "hostname": socket.gethostname(),
        }
        if resume and (out / "runtime_identity.json").is_file():
            prior = json.loads((out / "runtime_identity.json").read_text())
            for key in (
                "torch",
                "cuda",
                "minkowski_engine",
                "numpy",
                "python",
                "world_size",
            ):
                if prior[key] != runtime[key]:
                    raise RuntimeError(f"Runtime changed across resume: {key}")
        atomic_json(out / "runtime_identity.json", runtime)
        logger = logging.getLogger("feature_contract")
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())
        phase = "dataset_inventory"
        inventory = load_inventory(cfg, logger)
        path = out / "dataset_inventory.json"
        if resume and path.is_file() and json.loads(path.read_text()) != inventory:
            raise RuntimeError("Raw/cache/split identities changed across resume")
        atomic_json(path, inventory)
        atomic_json(
            out / "study_specification.json",
            {
                "kind": "feature_contract",
                "arm": args.arm,
                "pair_identity": pair_identity(cfg),
                "source_manifest_sha256": code_id,
                "target_checkpoint_selection": False,
                "source_domain": "eclair",
                "third_training": "not_submitted",
                "source_train_to_common": [0, 1, 2, 3, 0, 4, 4, 5, 5, 6, 7],
            },
        )
        training_path = out / "training_status.json"
        trained = (
            training_path.is_file()
            and json.loads(training_path.read_text()).get("status")
            == "training_complete"
        )
        if (
            resume
            and not trained
            and not (out / "checkpoints/last.pt").is_file()
            and any(
                (out / name).is_file() for name in ("metrics.csv", "study_epochs.jsonl")
            )
        ):
            raise RuntimeError(
                "Training telemetry exists without a resumable checkpoint; fresh restart requires a new study identity"
            )
        if not trained:
            phase = "gpu_preflight"
            if not resume or not (out / "preflight.json").is_file():
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/preflight_feature_contract_study.py",
                        "--config",
                        str(config_path),
                        "--out",
                        str(out / "preflight.json"),
                    ],
                    check=True,
                )
            smoke = json.loads((out / "preflight.json").read_text())
            if smoke.get("status") != "passed_gpu_smoke":
                raise RuntimeError("GPU/input/cache preflight did not pass")
            if resume and (out / "checkpoints/last.pt").is_file():
                cfg["run"]["resume_from"] = str(out / "checkpoints/last.pt")
                config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                atomic_json(out / "config_resolved.json", cfg)
            phase = "training"
            status("training", phase=phase)
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
                    str(config_path),
                ],
                check=True,
            )
            training = json.loads(training_path.read_text())
            if training.get("status") != "training_complete":
                status(
                    "needs_resume",
                    phase=phase,
                    last_completed_epoch=training.get("last_completed_epoch"),
                )
                return
        phase = "canonical_evaluation"
        status("evaluating", phase=phase)
        from src.model import build_model

        deadline = float(os.environ.get("STUDY_DEADLINE_UNIX", "inf"))
        checkpoints = {}
        for endpoint in ("best", "last"):
            if time.time() > deadline - 900:
                status("needs_resume", phase=phase, checkpoint=endpoint)
                return
            checkpoint = torch.load(
                out / "checkpoints" / f"{endpoint}.pt",
                map_location="cpu",
                weights_only=False,
            )
            assert_checkpoint_feature_config(checkpoint, cfg)
            validate_config(checkpoint["cfg"], frozen)
            model_id = model_digest(checkpoint["model_state"])
            checkpoints[endpoint] = {
                "model_sha256": model_id,
                "epoch": int(checkpoint["epoch"]),
                "selection": "source_native_validation_miou"
                if endpoint == "best"
                else "final",
            }
            stage = out / "evaluation" / model_id
            model = build_model(10, 11, cfg=cfg).to("cuda:0")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            del checkpoint
            identity = {
                "model_sha256": model_id,
                "source_manifest": code_id,
                "dataset_inventory_sha256": sha256(path),
            }
            for domain in ("eclair", "dales"):
                if not evaluate_tiles(
                    model, cfg, identity, domain, stage, deadline, logger
                ):
                    status(
                        "needs_resume", phase=phase, checkpoint=endpoint, domain=domain
                    )
                    return
            del model
            torch.cuda.empty_cache()
            atomic_json(
                out / "evaluation_manifest.json",
                {
                    "checkpoints": checkpoints,
                    "source_manifest_sha256": code_id,
                    "target_checkpoint_selection": False,
                    "feature_mode": cfg["data"]["features"]["return_input_mode"],
                },
            )
        atomic_json(
            out / "resource_summary.json",
            {
                "run_bytes": sum(
                    p.stat().st_size for p in out.rglob("*") if p.is_file()
                ),
                "storage_budget_bytes": STORAGE["run_budget_bytes"],
                "world_size": 2,
                "checkpoint_bytes": {
                    p.name: p.stat().st_size for p in (out / "checkpoints").glob("*.pt")
                },
                "training_history": "metrics.csv",
                "source_exposure": "study_epochs.jsonl",
            },
        )
        status(
            "complete",
            phase="complete",
            training_epochs=100,
            source_manifest_sha256=code_id,
            feature_mode=cfg["data"]["features"]["return_input_mode"],
        )
        phase = "paired_report"
        subprocess.run(
            [
                sys.executable,
                "scripts/compare_feature_contract_study.py",
                "--study-dir",
                str(out.parent),
            ],
            check=True,
        )
    except Exception as exc:
        status(
            "failed",
            phase=phase,
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
    parser.add_argument("--resume-flag", choices=["0", "1"], default="0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
