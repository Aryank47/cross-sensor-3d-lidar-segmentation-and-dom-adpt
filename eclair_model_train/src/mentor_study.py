"""Frozen protocol, execution audit and two-mentor development decision."""

from __future__ import annotations

import copy
import json
import tarfile
from pathlib import Path

import numpy as np

from .feature_contract_study import bootstrap, fingerprint, make_config, metric_vector
from .optional_return_study import (
    advancement,
    deltas,
    diagnostic_summary,
    load_result,
    source_pass,
)
from .study_metrics import COMMON_NAMES
from .study_runtime import atomic_json, sha256

ARMS = ("observed_mentor", "geometry_mentor")
REFERENCE_ROOT = (
    "/scratch/m23csa510/e0_results/feature_contract_studies/eclair_returns_s1337_v2"
)
MENTORS = {
    "observed_mentor": {
        "reference_arm": "observed",
        "input_mode": "observed",
        "epoch": 72,
        "model_sha256": "fb025ad8f83a910b4569b12c9550f0a786bb87d3b18c4fce321557636f29e6fb",
    },
    "geometry_mentor": {
        "reference_arm": "constant",
        "input_mode": "constant_pair",
        "epoch": 68,
        "model_sha256": "3adb11d2c25a6a392f97261bfdd34420cc45c4d8c4703a082cc94e8de810144b",
    },
}


def make_mentor_config(frozen, arm, out):
    if arm not in ARMS:
        raise ValueError("Unknown mentor arm")
    cfg = make_config(frozen, "observed", out)
    cfg["run"]["name"] = f"eclair_{arm}_s1337"
    mentor = copy.deepcopy(MENTORS[arm])
    mentor["checkpoint"] = (
        f"{REFERENCE_ROOT}/{mentor['reference_arm']}/checkpoints/best.pt"
    )
    cfg["study"].update(
        kind="mentor_guidance",
        arm=arm,
        mentor=mentor,
        guidance_max_weight=0.1,
        guidance_ramp_epochs=10,
        maximum_amp_skips=13,
        reference_epochs_path=str(
            Path(out).resolve().parent / "reference/observed/study_epochs.jsonl"
        ),
    )
    return cfg


def validate_mentor_config(cfg, frozen):
    actual = copy.deepcopy(cfg)
    resume = actual["run"].pop("resume_from", None)
    if (
        resume is not None
        and Path(resume).resolve()
        != Path(cfg["run"]["out_dir"]).resolve() / "checkpoints/last.pt"
    ):
        raise ValueError("Resume must use own last checkpoint")
    expected = make_mentor_config(frozen, cfg["study"]["arm"], cfg["run"]["out_dir"])
    if actual != expected:
        raise ValueError("Mentor recipe differs from frozen specification")
    return cfg["study"]["arm"]


def student_identity(cfg):
    value = copy.deepcopy(cfg)
    for key in ("name", "out_dir", "resume_from"):
        value["run"].pop(key, None)
    for key in ("arm", "mentor", "reference_epochs_path"):
        value["study"].pop(key, None)
    return fingerprint(value)


def original_student_identity(cfg):
    value = copy.deepcopy(cfg)
    for key in ("name", "out_dir", "resume_from"):
        value["run"].pop(key, None)
    for key in (
        "arm",
        "mentor",
        "reference_epochs_path",
        "kind",
        "guidance_max_weight",
        "guidance_ramp_epochs",
        "maximum_amp_skips",
    ):
        value["study"].pop(key, None)
    return fingerprint(value)


def execution_audit(study, arm):
    root = Path(study)
    out = root / arm
    errors = []
    reference = root / "reference/observed"
    cfg = json.loads((out / "config_resolved.json").read_text())
    from scripts.submit_feature_contract_study import load_frozen

    validate_mentor_config(cfg, load_frozen(Path(__file__).resolve().parents[1]))
    old = json.loads((reference / "config_resolved.json").read_text())
    if original_student_identity(cfg) != original_student_identity(old):
        errors.append("student_recipe")
    for name in ("dataset_inventory.json",):
        if json.loads((out / name).read_text()) != json.loads(
            (reference / name).read_text()
        ):
            errors.append(name)
    initial = [
        json.loads((p / "preflight.json").read_text())["initial_model_sha256"]
        for p in (reference, out)
    ]
    if initial[0] != initial[1]:
        errors.append("initialization")
    runtime = [
        json.loads((p / "runtime_identity.json").read_text()) for p in (reference, out)
    ]
    for key in ("torch", "cuda", "minkowski_engine", "numpy", "python", "world_size"):
        if runtime[0][key] != runtime[1][key]:
            errors.append("runtime_" + key)
    records = [
        [json.loads(s) for s in (p / "study_epochs.jsonl").read_text().splitlines()]
        for p in (reference, out)
    ]
    if any([r["epoch"] for r in rows] != list(range(1, 101)) for rows in records):
        errors.append("epoch_coverage")
    totals = {
        k: np.zeros(7, float)
        for k in (
            "valid_counts",
            "eligible_counts",
            "teacher_entropy_sum",
            "teacher_nontarget_mass_sum",
            "teacher_true_probability_sum",
            "teacher_ignore_mass_sum",
            "kl_sum",
        )
    }
    gradients = []
    for left, right in zip(*records):
        if len(left["ranks"]) != len(right["ranks"]):
            errors.append("world_size")
            continue
        for a, b in zip(left["ranks"], right["ranks"]):
            for key in (
                "native_counts",
                "presence",
                "samples",
                "microbatches",
                "metadata",
                "geometry_label_exposure_sha256",
                "coordinate_reordered_batches",
            ):
                if a[key] != b[key]:
                    errors.append("exposure_" + key)
            if b.get("mentor_model_sha256") != MENTORS[arm]["model_sha256"]:
                errors.append("mentor_changed")
            guide = b["guidance"]
            from .study_metrics import common_histogram

            if not np.array_equal(
                np.asarray(guide["valid_counts"]),
                common_histogram(b["native_counts"], [0, 1, 2, 3, 0, 4, 4, 5, 5, 6, 7])[
                    1:
                ],
            ):
                errors.append("guidance_class_exposure")
            if np.any(
                np.asarray(guide["eligible_counts"]) > np.asarray(guide["valid_counts"])
            ):
                errors.append("guidance_eligibility_counts")
            if guide["batches"] != b["microbatches"]:
                errors.append("guidance_batch_coverage")
            for key in totals:
                values = np.asarray(guide[key], float)
                if values.shape != (7,) or not np.isfinite(values).all():
                    errors.append("guidance_" + key)
                else:
                    totals[key] += values
            gradients += guide["gradient_diagnostics"]
    counters = [
        [r["optimizer_updates_cumulative"], r["amp_skips_cumulative"]]
        for r in records[1][-1]["ranks"]
    ]
    from .optional_returns import validate_update_counters

    try:
        validate_update_counters(counters)
    except RuntimeError:
        errors.append("numerical_counters")
    if any(sum(r) != 13300 for r in counters):
        errors.append("attempted_budget")
    if np.any(totals["eligible_counts"] <= 0) or np.any(totals["kl_sum"] <= 0):
        errors.append("guidance_no_signal_for_class")
    if len(gradients) != 200 or not any(
        g["weighted_logit_gradient_ratio"] > 0 for g in gradients
    ):
        errors.append("guidance_gradient_signal")
    identity = json.loads((out / "mentor_identity.json").read_text())
    if (
        identity["model_sha256"] != MENTORS[arm]["model_sha256"]
        or not identity["frozen"]
    ):
        errors.append("mentor_identity")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "counters": counters,
        "hardware_warning": runtime[0]["gpu_names"] != runtime[1]["gpu_names"],
        "guidance_totals": {k: v.tolist() for k, v in totals.items()},
        "gradient_diagnostics": gradients,
        "mentor_identity": identity,
        "different_skip_timings_are_not_hidden": True,
    }


def arm_report(study, arm):
    import yaml

    root = Path(study)
    out = root / arm
    audit = execution_audit(root, arm)
    endpoints = {}
    for endpoint in ("best", "last"):
        endpoints[endpoint] = {}
        for domain in ("eclair", "dales"):
            reference, _ = load_result(root / "reference/observed", endpoint, domain)
            candidate, folder = load_result(out, endpoint, domain)
            if [x["path"] for x in reference["tiles"]] != [
                x["path"] for x in candidate["tiles"]
            ]:
                raise ValueError("Tile identity mismatch")
            value = {
                "reference": reference,
                "student": candidate,
                "delta_vs_original_pp": deltas(reference, candidate),
                "intervals_vs_original": bootstrap(
                    [t["matrix"] for t in reference["tiles"]],
                    [t["matrix"] for t in candidate["tiles"]],
                ),
            }
            if endpoint == "best":
                mapping = yaml.safe_load(
                    (
                        Path(__file__).resolve().parents[1]
                        / f"configs/mapping_{domain}_to_common.yaml"
                    ).read_text()
                )
                lut = np.array([mapping.get(i, 0) for i in range(256)], np.int64)
                diagnostic = diagnostic_summary(folder, candidate, lut)
                value["constant_diagnostic"] = {
                    "scores": diagnostic,
                    "delta_vs_observed_pp": deltas(candidate, diagnostic),
                }
            endpoints[endpoint][domain] = value
    classification = advancement(
        endpoints["best"]["eclair"]["delta_vs_original_pp"],
        endpoints["best"]["dales"]["delta_vs_original_pp"],
        endpoints["best"]["dales"]["intervals_vs_original"],
        endpoints["last"]["eclair"]["delta_vs_original_pp"],
        endpoints["last"]["dales"]["delta_vs_original_pp"],
        audit["valid"],
    )
    result = {
        "status": "complete",
        "arm": arm,
        "execution": audit,
        "endpoints": endpoints,
        "classification_vs_original": classification,
        "source_only_selection": True,
        "automatic_reverse_jobs": 0,
    }
    atomic_json(out / "arm_report.json", result)
    return result


def decide_comparison(geometry, control, comparisons):
    if not geometry["execution"]["valid"] or not control["execution"]["valid"]:
        return "invalid_development_comparison"
    for ep in ("best", "last"):
        if not source_pass(control["endpoints"][ep]["eclair"]["delta_vs_original_pp"]):
            return "fresh_control_source_preservation_failed_stop"
    if (
        geometry["classification_vs_original"]
        != "clear_development_pass_reverse_pair_eligible"
    ):
        return "geometry_mentor_" + geometry["classification_vs_original"]
    result = advancement(
        comparisons["best"]["eclair"]["delta_pp"],
        comparisons["best"]["dales"]["delta_pp"],
        comparisons["best"]["dales"]["intervals"],
        comparisons["last"]["eclair"]["delta_pp"],
        comparisons["last"]["dales"]["delta_pp"],
        True,
    )
    if result == "clear_development_pass_reverse_pair_eligible":
        return "geometry_mentor_development_pass_reverse_pair_eligible"
    if (
        control["classification_vs_original"]
        == "clear_development_pass_reverse_pair_eligible"
    ):
        return "generic_guidance_gain_no_geometry_specific_advancement"
    return "no_geometry_specific_advancement_" + result


def compare_if_ready(study):
    root = Path(study)
    for arm in ARMS:
        path = root / arm / "STATUS.json"
        if not path.is_file() or json.loads(path.read_text())["status"] != "complete":
            return None
    reports = {a: json.loads((root / a / "arm_report.json").read_text()) for a in ARMS}
    configs = [
        json.loads((root / a / "config_resolved.json").read_text()) for a in ARMS
    ]
    if student_identity(configs[0]) != student_identity(configs[1]):
        raise ValueError("Student configurations differ beyond mentor")
    comparisons = {}
    for ep in ("best", "last"):
        comparisons[ep] = {}
        for domain in ("eclair", "dales"):
            control = reports[ARMS[0]]["endpoints"][ep][domain]["student"]
            candidate = reports[ARMS[1]]["endpoints"][ep][domain]["student"]
            if [t["path"] for t in control["tiles"]] != [
                t["path"] for t in candidate["tiles"]
            ]:
                raise ValueError("Fresh comparison tiles differ")
            comparisons[ep][domain] = {
                "delta_pp": deltas(control, candidate),
                "intervals": bootstrap(
                    [t["matrix"] for t in control["tiles"]],
                    [t["matrix"] for t in candidate["tiles"]],
                ),
            }
    classification = decide_comparison(reports[ARMS[1]], reports[ARMS[0]], comparisons)
    result = {
        "status": "complete",
        "classification": classification,
        "arms": reports,
        "geometry_minus_observed_mentor": comparisons,
        "advance_reverse_pair_eligible": classification
        == "geometry_mentor_development_pass_reverse_pair_eligible",
        "automatic_reverse_jobs": 0,
        "old_exact_update_classification": "invalid_execution_unchanged",
        "limitations": [
            "one training seed",
            "one source direction",
            "mentor-specific value, not isolated input-feature causality",
            "mentors differ in learned weights, selected epochs, competence and histories",
            "tile intervals do not measure training-seed variation",
            "source correctness does not guarantee target reliability",
        ],
    }
    atomic_json(root / "comparison.json", result)
    lines = [
        "# Frozen mentor comparison",
        "",
        f"Decision: **{classification}**",
        "",
        "Normal observed student inputs; source-native best selection; final secondary. No automatic reverse jobs.",
        "",
        "| Domain/metric | Observed mentor | Geometry mentor | Change, pp |",
        "|---|---:|---:|---:|",
    ]
    for domain in ("eclair", "dales"):
        for i, name in enumerate([*COMMON_NAMES[1:], "all_seven_miou", "utility_miou"]):
            values = [
                metric_vector(
                    reports[a]["endpoints"]["best"][domain]["student"][
                        "confusion_matrix"
                    ]
                )[i]
                * 100
                for a in ARMS
            ]
            lines.append(
                f"| {domain}/{name} | {values[0]:.2f} | {values[1]:.2f} | {values[1] - values[0]:+.2f} |"
            )
    lines += [
        "",
        "Both the fresh-control comparison and original-reference preservation/gain gates are required.",
        "Full class tradeoffs, mentor information/coverage and final endpoints are in comparison.json.",
    ]
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    return result


def collect_bundle(study, arm=None):
    root = Path(study).resolve()
    name = f"{arm}_results" if arm else "comparison_results"
    if arm is None and not (root / "comparison.json").is_file():
        return None
    files = []
    for path in sorted(root.rglob("*")):
        parts = path.relative_to(root).parts
        if (
            not path.is_file()
            or path.is_symlink()
            or {"code", "checkpoints"} & set(parts)
        ):
            continue
        if path.suffix not in (
            ".json",
            ".jsonl",
            ".csv",
            ".yaml",
            ".md",
            ".txt",
            ".sha256",
        ):
            continue
        if path.name.endswith("_collection_manifest.json"):
            continue
        if arm and parts[0] in ARMS and parts[0] != arm:
            continue
        files.append(path)
    if sum(p.stat().st_size for p in files) > 256 * 1024**2:
        raise OSError("Compact collection exceeds256MiB")
    manifest = root / f"{name}_collection_manifest.json"
    atomic_json(
        manifest,
        {
            "files": {p.relative_to(root).as_posix(): sha256(p) for p in files},
            "checkpoints_included": False,
        },
    )
    target = root / f"{name}.tar.gz"
    temp = target.with_suffix(".gz.tmp")
    with tarfile.open(temp, "w:gz") as archive:
        for path in [*files, manifest]:
            archive.add(
                path, arcname=path.relative_to(root).as_posix(), recursive=False
            )
    temp.replace(target)
    return {"bundle": str(target), "sha256": sha256(target)}
