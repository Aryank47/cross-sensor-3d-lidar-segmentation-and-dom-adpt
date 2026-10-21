"""Contracts for evaluating an existing source-fitted readout on canonical points."""

from __future__ import annotations

import copy
import getpass
import json
import re
import time
from pathlib import Path

import numpy as np

from .study_analysis import fingerprint
from .study_metrics import balanced_gate, summary
from .study_runtime import sha256

OUTPUT_BUDGET = 128 * 1024**2
RESERVE = 512 * 1024**2
VERSION = 1


def read_json(path):
    return json.loads(Path(path).read_text())


def head_arrays(head, channels=128, classes=8):
    if head.get("target_labels_used_for_fit_or_selection") is not False:
        raise ValueError("Readout must certify source-only fitting and selection")
    if head.get("inverse_native_frequency_weighted_ridge") is not True:
        raise ValueError("Unsupported saved classifier fitting procedure")
    mean = np.asarray(head["standardization_mean"], dtype=np.float64)
    std = np.asarray(head["standardization_std"], dtype=np.float64)
    coefficients = np.asarray(head["coefficients"], dtype=np.float64)
    if mean.shape != (channels,) or std.shape != (channels,) or coefficients.shape != (channels + 1, classes):
        raise ValueError("Readout feature/native-class dimensions differ from the frozen classifier contract")
    if not all(np.isfinite(x).all() for x in (mean, std, coefficients)) or np.any(std <= 0):
        raise ValueError("Nonfinite readout parameters or invalid source standardization")
    candidates = head.get("source_val_candidates", [])
    if not candidates or not all(
        np.isfinite(float(c["miou"])) and 0 <= float(c["miou"]) <= 1
        and np.isfinite(float(c["penalty"])) and float(c["penalty"]) > 0 for c in candidates
    ):
        raise ValueError("Missing finite source-validation selection record")
    selected = max(candidates, key=lambda c: (float(c["miou"]), -float(c["penalty"])))
    if float(head["penalty"]) != float(selected["penalty"]) or not np.isclose(
        float(head["selected_source_val_miou"]), float(selected["miou"]), rtol=0, atol=1e-12
    ):
        raise ValueError("Saved readout does not match its source-validation selection")
    return mean, std, coefficients


def project_features(features, head):
    mean, std, coefficients = head_arrays(head, channels=np.asarray(features).shape[1])
    return ((np.asarray(features, dtype=np.float64) - mean) / std) @ coefficients[:-1] + coefficients[-1]


def safe_output(study, output, arms):
    study, output = Path(study).resolve(), Path(output).resolve()
    if output == study:
        raise ValueError("Output must be separate from the existing study root")
    for arm in set(arms) | {"reference", "context", "fine"}:
        protected = study / arm
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Refuse output that contains or changes an existing arm")
    return output


def verify_original_sources(arm, code_root):
    manifest = arm / "source_manifest.sha256"
    if sha256(manifest) != read_json(arm / "STATUS.json")["source_manifest_sha256"]:
        raise ValueError("Original source manifest identity mismatch")
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", 1)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe original source-manifest member")
        if relative.suffix in (".py", ".yaml", ".sh") and sha256(code_root / relative) != digest:
            raise ValueError(f"Original execution source changed: {name}; use compatible code")


def validate_metrics(record):
    calculated = summary(record["confusion_matrix"])
    for key in ("all_seven_miou", "utility_miou"):
        if not np.isclose(calculated[key], record[key], rtol=0, atol=1e-12):
            raise ValueError("Recorded score does not match its confusion matrix")
    for key in ("support", "predicted", "per_class_iou", "precision", "recall"):
        if calculated[key] != record[key]:
            raise ValueError(f"Recorded {key} differs from its confusion matrix")
    tiles = record.get("tiles", [])
    if not tiles or not np.array_equal(
        np.asarray([t["confusion_matrix"] for t in tiles]).sum(0), record["confusion_matrix"]
    ):
        raise ValueError("Missing or inconsistent canonical tile matrices")
    if len({t["tile"] for t in tiles}) != len(tiles):
        raise ValueError("Repeated canonical tile")
    for tile in tiles:
        if summary(tile["confusion_matrix"])["support"] != tile["support"]:
            raise ValueError("Incorrect tile GT support")
    native = np.asarray(record["native_gt_by_common_prediction"])
    if native.shape != (256, 8) or np.any(native < 0):
        raise ValueError("Invalid native subtype matrix")


def validate_config_contract(cfg, baseline):
    contract = baseline["contract"]
    expected = {"model": cfg["model"], "features": cfg["data"]["features"], "patch": cfg["data"]["patch"],
                "voxelization": cfg["data"].get("voxelization"), "window_m": 120, "stride_m": 120,
                "aggregation": "mean_logits", "amp": False}
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("Source config or evaluation settings differ from the original canonical result")
    validate_metrics(baseline)


def make_plan(study, arms, endpoints, code_root):
    study, code_root = Path(study).resolve(), Path(code_root).resolve()
    if not arms or len(set(arms)) != len(arms) or not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("Arms and endpoints must be nonempty and unique")
    tasks = []
    for name in arms:
        if name not in ("reference", "context", "fine"):
            raise ValueError("Unknown architecture arm")
        arm = study / name
        if read_json(arm / "STATUS.json")["status"] != "complete":
            raise ValueError(f"{name} study is not complete")
        selection = read_json(arm / "checkpoint_selection.json")
        if selection.get("target_labels_used") is not False or selection.get("primary") != "source_all_native_validation_miou":
            raise ValueError("Unsupported or target-selected checkpoint rule")
        verify_original_sources(arm, code_root)
        cfg = read_json(arm / "config_resolved.json")
        if int(cfg["model"]["out_channels"]) != 8:
            raise ValueError("This readout contract supports DALES-native eight-class heads only")
        manifest = read_json(arm / "evaluation_manifest.json")
        if manifest.get("target_checkpoint_selection") is not False:
            raise ValueError("Original evaluation manifest is not source-selected")
        groups = {}
        for endpoint in endpoints:
            if endpoint not in ("best", "last"):
                raise ValueError("Only predeclared source-best and final endpoints are supported")
            entry = manifest["checkpoints"][endpoint]
            model_id = entry["model_sha256"]
            if not re.fullmatch(r"[a-f0-9]{64}", model_id):
                raise ValueError("Unsafe model identity")
            expected_epoch = selection["best_epoch" if endpoint == "best" else "last_epoch"]
            if int(entry["epoch"]) != int(expected_epoch):
                raise ValueError("Checkpoint epoch differs from the source selection record")
            if model_id in groups:
                groups[model_id]["endpoints"].append(endpoint)
                continue
            stage = arm / "evaluation" / model_id
            head_path = stage / "frozen_head_results.json"
            if read_json(stage / "bounded_complete.json").get(head_path.name) != sha256(head_path):
                raise ValueError("Frozen head fails the original bounded-analysis completion hash")
            head_arrays(read_json(head_path))
            baselines = {domain: stage / domain / "metrics.json" for domain in ("dales", "eclair")}
            for path in baselines.values():
                validate_config_contract(cfg, read_json(path))
            checkpoint = arm / "checkpoints" / f"{endpoint}.pt"
            inputs = [arm / "STATUS.json", arm / "checkpoint_selection.json", arm / "evaluation_manifest.json",
                      arm / "config_resolved.json", arm / "source_manifest.sha256", stage / "bounded_complete.json",
                      head_path, checkpoint, *baselines.values()]
            task = {"arm": name, "model_id": model_id, "epoch": int(entry["epoch"]), "endpoints": [endpoint],
                    "checkpoint": str(checkpoint), "head": str(head_path), "config": str(arm / "config_resolved.json"),
                    "baselines": {k: str(v) for k, v in baselines.items()},
                    "input_sha256": {str(p): sha256(p) for p in inputs}}
            groups[model_id] = task
            tasks.append(task)
    # Finish both primary source-best comparisons before spending time on the final-only sensitivity check.
    tasks.sort(key=lambda task: "best" not in task["endpoints"])
    sources = sorted((code_root / "src").glob("*.py")) + [
        code_root / "scripts/run_readout_control.py", code_root / "scripts/submit_readout_control.py",
        code_root / "scripts/slurm_readout_control.sh"]
    return {"version": VERSION, "study_dir": str(study), "arms": list(arms), "endpoints": list(endpoints),
            "selection": "source-best primary; final secondary; identical weights deduplicated",
            "target_fitting_or_selection": False, "tasks": tasks,
            "evaluation_code_sha256": {str(p.relative_to(code_root)): sha256(p) for p in sources}}


def verify_paired(original, replacement):
    validate_metrics(original)
    validate_metrics(replacement)
    a, b = copy.deepcopy(original["contract"]), copy.deepcopy(replacement["contract"])
    a.pop("checkpoint", None)
    b.pop("checkpoint", None)
    if a != b or original["support"] != replacement["support"]:
        raise ValueError("Original and replacement do not evaluate the same canonical protocol and GT support")
    for x, y in zip(original["tiles"], replacement["tiles"]):
        if x["tile"] != y["tile"] or x["support"] != y["support"]:
            raise ValueError("Original and replacement tile-level GT support differs")
    if len(original["tiles"]) != len(replacement["tiles"]) or not np.array_equal(
        np.asarray(original["native_gt_by_common_prediction"]).sum(1),
        np.asarray(replacement["native_gt_by_common_prediction"]).sum(1)
    ):
        raise ValueError("Original and replacement native subtype support differs")


def comparison(original, replacement):
    for domain in ("dales", "eclair"):
        verify_paired(original[domain], replacement[domain])
    return {"balanced_screen": balanced_gate(original["dales"], replacement["dales"],
                                              original["eclair"], replacement["eclair"]),
            "domains": {domain: {"original": original[domain], "replacement": replacement[domain],
                                  "per_class_delta_pp": {
                                      name: 100 * (replacement[domain]["per_class_iou"][name] - value)
                                      for name, value in original[domain]["per_class_iou"].items()},
                                  "all_delta_pp": 100 * (replacement[domain]["all_seven_miou"]
                                                         - original[domain]["all_seven_miou"]),
                                  "utility_delta_pp": 100 * (replacement[domain]["utility_miou"]
                                                             - original[domain]["utility_miou"])}
                        for domain in ("dales", "eclair")}}


def validate_admission(receipt, output, now=None):
    age = (time.time() if now is None else now) - float(receipt["checked_at_unix"])
    valid = (receipt["user"] == getpass.getuser() and Path(receipt["output_dir"]).resolve() == Path(output).resolve()
             and -300 <= age <= 48 * 3600 and receipt["budget_bytes"] == OUTPUT_BUDGET
             and receipt["reserve_bytes"] == RESERVE and receipt["headroom_bytes"] >= OUTPUT_BUDGET + RESERVE)
    if not valid:
        raise ValueError("Quota admission is stale, insufficient, or for another user/output")


def bind_plan(output, plan, resume):
    path = Path(output) / "experiment_manifest.json"
    identity = fingerprint(plan)
    if path.exists():
        if not resume or read_json(path)["identity"] != identity:
            raise ValueError("Existing output requires --resume with identical inputs and evaluator code")
    elif resume:
        raise ValueError("Cannot resume without an existing evaluation manifest")
    return identity
