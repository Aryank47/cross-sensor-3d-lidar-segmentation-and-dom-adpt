#!/usr/bin/env python3
"""Evaluate saved source-only readouts on the original full canonical test sets."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import signal
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.readout_control import (  # noqa: E402
    OUTPUT_BUDGET,
    RESERVE,
    bind_plan,
    comparison,
    head_arrays,
    make_plan,
    project_features,
    read_json,
    safe_output,
    validate_admission,
)
from src.study_analysis import cohorts, evaluate_tiles, fingerprint, infer_patch, load_dataset  # noqa: E402
from src.study_runtime import atomic_json, sha256, storage_guard  # noqa: E402


class ResumeRequested(Exception):
    pass


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arms", nargs="+", choices=["reference", "context", "fine"], default=["reference", "context"])
    parser.add_argument("--endpoints", nargs="+", choices=["best", "last"], default=["best", "last"])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def install_head(model, head):
    import MinkowskiEngine as ME
    import torch

    mean, std, coefficients = head_arrays(head)

    class FrozenReadout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name, value in (("mean", mean), ("std", std), ("coefficients", coefficients)):
                self.register_buffer(name, torch.from_numpy(value.copy()).to("cuda:0"))

        def forward(self, sparse):
            features = (sparse.F.double() - self.mean) / self.std
            logits = features @ self.coefficients[:-1] + self.coefficients[-1]
            return ME.SparseTensor(
                features=logits.float(), coordinate_map_key=sparse.coordinate_map_key,
                coordinate_manager=sparse.coordinate_manager,
            )

    model.final = FrozenReadout()
    model.requires_grad_(False)
    model.eval()


def backbone_digest(model):
    from scripts.run_architecture_study import model_digest

    return model_digest({k: v for k, v in model.state_dict().items() if not k.startswith("final.")})


def load_frozen_model(task, cfg):
    import torch
    from scripts.run_architecture_study import model_digest
    from src.model import build_model

    checkpoint = torch.load(task["checkpoint"], map_location="cpu", weights_only=False)
    if model_digest(checkpoint["model_state"]) != task["model_id"] or int(checkpoint["epoch"]) != task["epoch"]:
        raise ValueError("Checkpoint tensors or epoch differ from the original evaluation manifest")
    model = build_model(int(cfg["model"]["in_channels"]), int(cfg["model"]["out_channels"]), cfg=cfg)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to("cuda:0").eval().requires_grad_(False)
    del checkpoint
    return model


def smoke(model, cfg, head, logger):
    import numpy as np

    before = backbone_digest(model)
    cohort = next(cohorts("dales", "val", cfg, logger))
    _, original_features, _ = infer_patch(model, cohort["raw"], cfg, features=True)
    install_head(model, head)
    actual, features, _ = infer_patch(model, cohort["raw"], cfg, features=True)
    np.testing.assert_allclose(features, original_features, rtol=1e-4, atol=1e-4)
    expected = project_features(features, head).astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
    if backbone_digest(model) != before:
        raise ValueError("Frozen backbone parameters or normalization buffers changed in smoke check")
    return {"passed": True, "domain": "dales", "split": "val", "points": len(features),
            "maximum_feature_error": float(np.max(np.abs(features - original_features))),
            "maximum_logit_error": float(np.max(np.abs(actual - expected))), "backbone_unchanged": True,
            "original_classifier_not_evaluated_on_this_cohort_for_comparison": True}


def run_task(task, out, deadline, logger):
    import torch

    cfg = copy.deepcopy(read_json(task["config"]))
    cfg["study"]["storage"] = {
        "run_budget_bytes": OUTPUT_BUDGET, "diagnostics_budget_bytes": OUTPUT_BUDGET, "reserve_bytes": RESERVE,
    }
    original = {domain: read_json(path) for domain, path in task["baselines"].items()}
    # Fail before inference if caches, membership or labels differ from the original tests.
    for domain in original:
        dataset, _, identity = load_dataset(domain, "test", cfg, logger)
        del dataset
        contract = original[domain]["contract"]
        if identity != contract["dataset"]:
            raise ValueError(f"{domain} raw/cache/split identity changed since original evaluation")
        if sha256(f"configs/mapping_{domain}_to_common.yaml") != contract["target_mapping_sha256"]:
            raise ValueError("Target mapping changed")
        if sha256("configs/train_id_to_common_dales.yaml") != contract["source_mapping_sha256"]:
            raise ValueError("Source mapping changed")
    model = load_frozen_model(task, cfg)
    frozen_id = backbone_digest(model)
    stage = out / task["arm"] / task["model_id"]
    stage.mkdir(parents=True, exist_ok=True)
    atomic_json(stage / "gpu_smoke.json", smoke(model, cfg, read_json(task["head"]), logger))
    replacement_id = "source-readout:" + fingerprint({"model": task["model_id"], "head": sha256(task["head"])})
    try:
        with torch.inference_mode():
            for domain in ("dales", "eclair"):
                atomic_json(out / "progress.json", {"arm": task["arm"], "endpoints": task["endpoints"],
                            "model_id": task["model_id"], "domain": domain, "updated_at_unix": time.time()})
                if not evaluate_tiles(model, cfg, replacement_id, domain, stage, deadline, logger):
                    raise ResumeRequested("Allocation deadline: completed tiles are available for --resume")
        if backbone_digest(model) != frozen_id:
            raise ValueError("Frozen backbone changed during canonical evaluation")
        replacement = {domain: read_json(stage / domain / "metrics.json") for domain in original}
        result = comparison(original, replacement)
        result.update(arm=task["arm"], epoch=task["epoch"], endpoints=task["endpoints"],
                      model_id=task["model_id"], backbone_unchanged=True, target_fitting_or_selection=False)
        storage_guard(out, 2 * 1024**2, cfg["study"]["storage"])
        atomic_json(stage / "comparison.json", result)
        return {"arm": task["arm"], "model_id": task["model_id"], "endpoints": task["endpoints"],
                "comparison": str(stage / "comparison.json"), "balanced_screen": result["balanced_screen"]}
    finally:
        del model
        torch.cuda.empty_cache()


def main():
    args = arguments()
    import fcntl

    from scripts.submit_architecture_study import quota_headroom

    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().split(".")[0].lower() == "cn07":
        raise RuntimeError("Run inside a Slurm GPU allocation with cn07 excluded")
    os.chdir(ROOT)
    out = safe_output(args.study_dir, args.output_dir, args.arms)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError("Fresh output must be empty; choose a new directory or use --resume")
    plan = make_plan(args.study_dir, args.arms, args.endpoints, ROOT)
    identity = bind_plan(out, plan, args.resume)
    receipt = json.loads(os.environ["READOUT_QUOTA_ADMISSION"])
    validate_admission(receipt, out)
    deadline = float(os.environ["READOUT_DEADLINE_UNIX"])
    if deadline <= time.time() + 900:
        raise RuntimeError("Insufficient remaining allocation time")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".allocation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # A tiny marker lets the existing canonical evaluator enforce this entire output's cap.
    marker = out / "config_resolved.yaml"
    if not marker.exists():
        marker.write_text("readout_control: 1\n")
    atomic_json(out / "experiment_manifest.json", {"identity": identity, "plan": plan})
    status_path = out / "STATUS.json"

    def status(value, **fields):
        atomic_json(status_path, {"status": value, "job_id": os.environ["SLURM_JOB_ID"],
                    "updated_at_unix": time.time(), "experiment_identity": identity, **fields})

    def interrupted(_signum, _frame):
        raise ResumeRequested("Slurm time-limit signal; rerun submission with --resume")

    signal.signal(signal.SIGUSR1, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("readout_control")
    status("preflight")
    try:
        try:
            headroom = quota_headroom(out)
        except (OSError, RuntimeError) as exc:
            admission = {"mode": "login_snapshot_compute_query_unavailable", "live_error": str(exc)}
        else:
            if headroom < OUTPUT_BUDGET + RESERVE:
                raise OSError("Live quota no longer has the admitted headroom")
            admission = {"mode": "live_compute_quota", "headroom_bytes": headroom}
        atomic_json(out / "storage_admission.json", {**admission, "receipt": receipt, "quota_reservation": False})
        storage_guard(out, 2 * 1024**2, {"run_budget_bytes": OUTPUT_BUDGET, "reserve_bytes": RESERVE})
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU in allocation")
        # The saved readout was fitted with NumPy float64. Preserve that projection on GPU.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        results = []
        status("evaluating", unique_models=len(plan["tasks"]))
        for task in plan["tasks"]:
            if time.time() > deadline - 900:
                raise ResumeRequested("Allocation deadline before next model")
            results.append(run_task(task, out, deadline, logger))
        # Recheck all source artifacts and code after inference; never combine changed inputs.
        if make_plan(args.study_dir, args.arms, args.endpoints, ROOT) != plan:
            raise ValueError("Input artifacts or evaluation code changed during evaluation")
        atomic_json(out / "comparison.json", {"primary": "source-best", "secondary": "last",
                    "target_fitting_or_selection": False, "results": results,
                    "interpretation": "Observed classifier control; no causal isolation of a loss or backbone family"})
        status("complete", unique_models=len(results), comparison=str(out / "comparison.json"))
    except ResumeRequested as exc:
        status("needs_resume", reason=str(exc))
        logger.warning("%s", exc)
    except Exception as exc:
        status("failed", reason=str(exc))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
