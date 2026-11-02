"""Fixed optional-return development study, paired-input diagnostic and gates."""

from __future__ import annotations

import copy
import json
import os
import tarfile
import time
from pathlib import Path

import numpy as np

from .feature_contract_study import (
    EXPECTED_TILES,
    bootstrap,
    fingerprint,
    make_config,
    metric_vector,
    summarize,
)
from .study_metrics import COMMON_NAMES, ECLAIR_TRAIN_TO_COMMON
from .study_runtime import atomic_json, sha256


def make_optional_config(frozen, out):
    cfg = make_config(frozen, "observed", out)
    cfg["run"]["name"] = "eclair_optional_returns_s1337"
    cfg["study"].update(
        kind="optional_returns",
        arm="masked",
        return_mask_probability=0.5,
        maximum_amp_skips=13,
        reference_epochs_path=str(
            Path(out).resolve().parent / "reference/observed/study_epochs.jsonl"
        ),
    )
    return cfg


def validate_optional_config(cfg, frozen):
    actual = copy.deepcopy(cfg)
    resume = actual["run"].pop("resume_from", None)
    if (
        resume is not None
        and Path(resume).resolve()
        != Path(cfg["run"]["out_dir"]).resolve() / "checkpoints/last.pt"
    ):
        raise ValueError("Resume must use the candidate's own last checkpoint")
    if actual != make_optional_config(frozen, cfg["run"]["out_dir"]):
        raise ValueError("Optional-return recipe differs from its frozen configuration")
    if cfg["data"]["voxelization"]["feat_pool"] != "mean_all":
        raise ValueError(
            "Post-voxel constant substitution requires frozen mean pooling"
        )
    return actual


def scientific_identity(cfg):
    value = copy.deepcopy(cfg)
    for key in ("name", "out_dir", "resume_from"):
        value["run"].pop(key, None)
    for key in (
        "arm",
        "kind",
        "return_mask_probability",
        "maximum_amp_skips",
        "reference_epochs_path",
    ):
        value["study"].pop(key, None)
    return fingerprint(value)


def verify_reference(root, spec):
    root = Path(root)
    for name, digest in spec["files"].items():
        path = root / name
        if not path.is_file() or path.is_symlink() or sha256(path) != digest:
            raise ValueError(f"Frozen reference evidence missing or changed: {name}")
    for arm in ("observed", "constant"):
        if json.loads((root / arm / "STATUS.json").read_text())["status"] != "complete":
            raise ValueError("Reference study is not complete")


def load_result(base, endpoint, domain):
    base = Path(base)
    manifest = json.loads((base / "evaluation_manifest.json").read_text())
    record = manifest["checkpoints"][endpoint]
    folder = (
        base
        / record.get("relative_dir", "evaluation/" + record["model_sha256"])
        / domain
    )
    marker = json.loads((folder / "complete.json").read_text())
    if marker["metrics_sha256"] != sha256(folder / "metrics.json"):
        raise ValueError("Canonical completion hash mismatch")
    result = json.loads((folder / "metrics.json").read_text())
    if (
        marker["tiles"] != EXPECTED_TILES[domain]
        or len(result["tiles"]) != EXPECTED_TILES[domain]
    ):
        raise ValueError("Canonical test coverage incomplete")
    matrices = np.asarray([t["matrix"] for t in result["tiles"]], np.int64)
    if not np.array_equal(matrices.sum(0), result["confusion_matrix"]):
        raise ValueError("Canonical tile totals differ from aggregate matrix")
    return result, folder


def paired_counts(native, observed, constant):
    gt, a, b = [np.asarray(x, dtype=np.int64) for x in (native, observed, constant)]
    if (
        gt.shape != a.shape
        or gt.shape != b.shape
        or np.any((gt < 0) | (gt > 255))
        or np.any((a < 0) | (a > 10))
        or np.any((b < 0) | (b > 10))
    ):
        raise ValueError("Paired native predictions are not aligned/range-valid")
    counts = np.bincount(gt * 121 + a * 11 + b, minlength=256 * 121).reshape(
        256, 11, 11
    )
    indices = np.argwhere(counts)
    return [[int(g), int(p), int(q), int(counts[g, p, q])] for g, p, q in indices]


def constant_diagnostic(model, raw, cfg, domain, observed, native, logger):
    import torch

    from . import eval_cross_common as ev

    config = copy.deepcopy(cfg)
    config["data"]["features"]["return_input_mode"] = "constant_pair"
    tick = time.monotonic()
    logits = ev.infer_tile_point_logits(
        model=model,
        raw=raw,
        source_cfg=config,
        device=torch.device("cuda:0"),
        amp=False,
        window_size_m=120.0,
        window_stride_m=120.0,
        aggregation="mean_logits",
        max_voxels_per_forward=180000,
        max_split_depth=8,
        logger=logger,
        target_domain=domain,
    )
    for start in range(0, len(logits), 1000000):
        if not np.isfinite(logits[start : start + 1000000]).all():
            raise RuntimeError("Nonfinite constant-view diagnostic logits")
    if logits.shape != (len(observed), 11):
        raise ValueError("Paired diagnostic does not cover the same points")
    other = logits.argmax(1)
    del logits
    return {
        "input_mode": "constant_pair",
        "deployment_selection": False,
        "seconds": time.monotonic() - tick,
        "native_gt_observed_constant": paired_counts(native, observed, other),
    }


def diagnostic_summary(folder, primary, gt_lut):
    aggregate = np.zeros((256, 11, 11), np.int64)
    tile_rows = []
    pred_lut = np.asarray(ECLAIR_TRAIN_TO_COMMON)
    observed_native = np.zeros((256, 11), np.int64)
    for tile in primary["tiles"]:
        payload = json.loads(
            (Path(folder) / f"tile_{tile['tile']:04d}.json").read_text()
        )
        joint = np.zeros_like(aggregate)
        for gt, normal, other, count in payload["paired_diagnostic"][
            "native_gt_observed_constant"
        ]:
            if not (
                0 <= gt < 256 and 0 <= normal < 11 and 0 <= other < 11 and count > 0
            ):
                raise ValueError("Invalid sparse paired counts")
            joint[gt, normal, other] += count
        normal_native, other_native = joint.sum(2), joint.sum(1)
        normal_cm, other_cm = np.zeros((8, 8), np.int64), np.zeros((8, 8), np.int64)
        np.add.at(normal_cm, (gt_lut[:, None], pred_lut[None, :]), normal_native)
        np.add.at(other_cm, (gt_lut[:, None], pred_lut[None, :]), other_native)
        normal_cm[0] = 0
        other_cm[0] = 0
        if not np.array_equal(normal_cm, tile["matrix"]):
            raise ValueError(
                "Paired diagnostic changed primary point alignment/scoring"
            )
        aggregate += joint
        observed_native += normal_native
        tile_rows.append(
            {"tile": tile["tile"], "path": tile["path"], "matrix": other_cm.tolist()}
        )
    if not np.array_equal(observed_native, primary["native_gt_by_source_prediction"]):
        raise ValueError("Paired diagnostic native coverage differs from primary")
    matrix = np.sum([t["matrix"] for t in tile_rows], axis=0)
    transitions = {}
    for cls, name in enumerate(COMMON_NAMES[1:], 1):
        values = {
            "both_correct": 0,
            "observed_only_correct": 0,
            "constant_only_correct": 0,
            "both_wrong": 0,
        }
        for g, a, b in np.argwhere(aggregate):
            if gt_lut[g] != cls:
                continue
            normal_correct, other_correct = pred_lut[a] == cls, pred_lut[b] == cls
            key = (
                "both_correct"
                if normal_correct and other_correct
                else "observed_only_correct"
                if normal_correct
                else "constant_only_correct"
                if other_correct
                else "both_wrong"
            )
            values[key] += int(aggregate[g, a, b])
        transitions[name] = values
    result = {
        **summarize(matrix),
        "tiles": tile_rows,
        "class_correctness_transitions": transitions,
        "native_gt_observed_constant": [
            [int(g), int(a), int(b), int(aggregate[g, a, b])]
            for g, a, b in np.argwhere(aggregate)
        ],
        "native_gt_by_source_prediction": aggregate.sum(1).tolist(),
        "input_mode": "constant_pair",
        "deployment_selection": False,
        "primary_metrics_sha256": sha256(Path(folder) / "metrics.json"),
    }
    atomic_json(Path(folder) / "constant_diagnostic.json", result)
    return result


def deltas(reference, candidate):
    return (
        (
            metric_vector(candidate["confusion_matrix"])
            - metric_vector(reference["confusion_matrix"])
        )
        * 100
    ).tolist()


def source_pass(delta):
    return min(delta[:7]) >= -2 and min(delta[7:]) >= -1


def advancement(
    best_source, best_target, intervals, last_source, last_target, execution_valid
):
    if not execution_valid:
        return "invalid_development_comparison"
    if not source_pass(best_source):
        return "source_preservation_failed_stop_recipe"
    point_pass = (
        best_target[7] >= 1 and best_target[8] >= 1 and min(best_target[:7]) >= -2
    )
    if not point_pass:
        return "target_worthwhile_gain_screen_failed_stop_recipe"
    if (
        min(intervals[name]["low_pp"] for name in ("all_seven_miou", "utility_miou"))
        <= 0
    ):
        return "uncertain_gain_no_automatic_expansion"
    if (
        not source_pass(last_source)
        or min(last_target[7:]) <= 0
        or min(last_target[:7]) < -2
    ):
        return "final_weight_instability_no_automatic_expansion"
    return "clear_development_pass_reverse_pair_eligible"


def generate_report(study):
    root = Path(study)
    out = root / "candidate"
    ref = root / "reference"
    spec = json.loads((root / "reference_manifest.json").read_text())
    verify_reference(ref, spec)
    configs = [
        json.loads((p / "config_resolved.json").read_text())
        for p in (ref / "observed", out)
    ]
    errors = []
    if scientific_identity(configs[0]) != scientific_identity(configs[1]):
        errors.append("scientific_configuration")
    if json.loads((ref / "observed/dataset_inventory.json").read_text()) != json.loads(
        (out / "dataset_inventory.json").read_text()
    ):
        errors.append("dataset_inventory")
    if (
        json.loads((ref / "observed/preflight.json").read_text())[
            "initial_model_sha256"
        ]
        != json.loads((out / "preflight.json").read_text())["initial_model_sha256"]
    ):
        errors.append("model_initialization")
    runtime = [
        json.loads((p / "runtime_identity.json").read_text())
        for p in (ref / "observed", out)
    ]
    for key in ("torch", "cuda", "minkowski_engine", "numpy", "python", "world_size"):
        if runtime[0][key] != runtime[1][key]:
            errors.append("runtime_" + key)
    records = [
        [json.loads(s) for s in (p / "study_epochs.jsonl").read_text().splitlines()]
        for p in (ref / "observed", out)
    ]
    if any([r["epoch"] for r in rows] != list(range(1, 101)) for rows in records):
        errors.append("epoch_coverage")
    for a, b in zip(*records):
        if len(a["ranks"]) != len(b["ranks"]):
            errors.append("world_size")
            continue
        if a["accepted_native_counts"] != b["accepted_native_counts"]:
            errors.append("class_exposure")
        for l, r in zip(a["ranks"], b["ranks"]):
            for key in (
                "native_counts",
                "presence",
                "samples",
                "microbatches",
                "metadata",
                "geometry_label_exposure_sha256",
                "coordinate_reordered_batches",
            ):
                if l[key] != r[key]:
                    errors.append("exposure_" + key)
            view = r.get("return_exposure", {})
            if sum(view.get(k, 0) for k in ("masked_samples", "observed_samples")) != r[
                "samples"
            ] or not np.array_equal(
                np.asarray(view.get("masked_native_counts", []))
                + np.asarray(view.get("observed_native_counts", [])),
                r["native_counts"],
            ):
                errors.append("mask_exposure_partition")
    last = records[1][-1]["ranks"]
    if any(
        r["optimizer_updates_cumulative"] + r["amp_skips_cumulative"] != 13300
        for r in last
    ):
        errors.append("attempted_budget")
    from .optional_returns import validate_update_counters

    try:
        validate_update_counters(
            [
                [r["optimizer_updates_cumulative"], r["amp_skips_cumulative"]]
                for r in last
            ]
        )
    except RuntimeError:
        errors.append("numerical_accounting")
    endpoints = {}
    for endpoint in ("best", "last"):
        domains = {}
        for domain in ("eclair", "dales"):
            control, _ = load_result(ref / "observed", endpoint, domain)
            constant, _ = load_result(ref / "constant", endpoint, domain)
            candidate, folder = load_result(out, endpoint, domain)
            paths = lambda r: [t["path"] for t in r["tiles"]]
            if paths(control) != paths(candidate) or paths(constant) != paths(
                candidate
            ):
                raise ValueError("Test tile identities differ from frozen comparisons")
            value = {
                "control": control,
                "trained_constant": constant,
                "candidate": candidate,
                "delta_pp": deltas(control, candidate),
                "paired_tile_intervals": bootstrap(
                    [t["matrix"] for t in control["tiles"]],
                    [t["matrix"] for t in candidate["tiles"]],
                ),
            }
            if endpoint == "best":
                import yaml

                lut_dict = yaml.safe_load(
                    (
                        Path(__file__).resolve().parents[1]
                        / "configs"
                        / f"mapping_{domain}_to_common.yaml"
                    ).read_text()
                )
                lut = np.array([lut_dict.get(i, 0) for i in range(256)], np.int64)
                diagnostic = diagnostic_summary(folder, candidate, lut)
                value["constant_input_diagnostic"] = {
                    "scores": diagnostic,
                    "delta_vs_candidate_observed_pp": deltas(candidate, diagnostic),
                    "delta_vs_trained_constant_pp": deltas(constant, diagnostic),
                }
            domains[domain] = value
        endpoints[endpoint] = domains
    classification = advancement(
        endpoints["best"]["eclair"]["delta_pp"],
        endpoints["best"]["dales"]["delta_pp"],
        endpoints["best"]["dales"]["paired_tile_intervals"],
        endpoints["last"]["eclair"]["delta_pp"],
        endpoints["last"]["dales"]["delta_pp"],
        not errors,
    )
    result = {
        "status": "complete",
        "classification": classification,
        "development_comparison_valid": not errors,
        "comparison_errors": sorted(set(errors)),
        "primary_endpoint": "best_observed_inputs",
        "advance_reverse_pair_eligible": classification
        == "clear_development_pass_reverse_pair_eligible",
        "automatic_reverse_submission": False,
        "old_exact_update_classification": "invalid_execution_unchanged",
        "numerical_accounting": {
            "candidate_ranks": [
                {
                    "updates": r["optimizer_updates_cumulative"],
                    "skips": r["amp_skips_cumulative"],
                }
                for r in last
            ]
        },
        "hardware_warning": runtime[0]["gpu_names"] != runtime[1]["gpu_names"],
        "endpoints": endpoints,
        "historical_benchmarks": spec["historical_benchmarks"],
        "limitations": [
            "one candidate training seed",
            "reused development control",
            "one source direction",
            "AMP timing differs and applied counts are reported",
            "tile intervals do not measure seed variation",
            "diagnostic input mode is not deployment selection",
        ],
    }
    atomic_json(root / "report.json", result)
    lines = [
        "# Optional-return development result",
        "",
        f"Decision: **{classification}**",
        "",
        "Primary inputs: observed returns. Source-only best checkpoint. No automatic reverse job.",
        "",
        "| Domain/class | Control IoU | Candidate IoU | Change, pp |",
        "|---|---:|---:|---:|",
    ]
    for domain, value in endpoints["best"].items():
        for i, name in enumerate(COMMON_NAMES[1:]):
            lines.append(
                f"| {domain}/{name} | {100 * value['control']['per_class_iou'][name]:.2f} | {100 * value['candidate']['per_class_iou'][name]:.2f} | {value['delta_pp'][i]:+.2f} |"
            )
        for i, name in ((7, "all_seven_miou"), (8, "utility_miou")):
            lines.append(
                f"| {domain}/{name} | {100 * value['control'][name]:.2f} | {100 * value['candidate'][name]:.2f} | {value['delta_pp'][i]:+.2f} |"
            )
    lines += [
        "",
        f"Execution checks: {not errors}; errors: {sorted(set(errors))}.",
        "The constant-input view is diagnostic; best/final and historical class tradeoffs are in report.json.",
        "This is a development screen, not bidirectional or independent-seed confirmation.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return result


def collect_results(study):
    root = Path(study).resolve()
    if (
        json.loads((root / "candidate/STATUS.json").read_text())["status"] != "complete"
        or json.loads((root / "report.json").read_text())["status"] != "complete"
    ):
        raise ValueError("Study/report not complete")
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and not p.is_symlink()
        and not ({"code", "checkpoints"} & set(p.relative_to(root).parts))
        and p.name != "collection_manifest.json"
        and p.suffix in (".json", ".jsonl", ".csv", ".yaml", ".md", ".txt", ".sha256")
    ]
    if sum(p.stat().st_size for p in files) > 256 * 1024**2:
        raise OSError("Evidence collection exceeds 256MiB")
    manifest = root / "collection_manifest.json"
    atomic_json(
        manifest,
        {
            "files": {str(p.relative_to(root)): sha256(p) for p in files},
            "checkpoints_included": False,
            "raw_point_clouds_included": False,
        },
    )
    temp = root / "results_bundle.tar.gz.tmp"
    with tarfile.open(temp, "w:gz") as archive:
        for p in [*files, manifest]:
            archive.add(p, arcname=str(p.relative_to(root)), recursive=False)
    os.replace(temp, root / "results_bundle.tar.gz")
    return {
        "bundle": str(root / "results_bundle.tar.gz"),
        "sha256": sha256(root / "results_bundle.tar.gz"),
    }
