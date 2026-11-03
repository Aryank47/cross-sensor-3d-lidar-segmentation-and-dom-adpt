"""Synthetic full-study reporting regression; no model or target-data fitting."""

import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_feature_contract_study import load_frozen
from src.feature_contract_study import summarize
from src.mentor_study import (
    ARMS,
    MENTORS,
    arm_report,
    collect_bundle,
    compare_if_ready,
    make_mentor_config,
)
from src.study_metrics import ECLAIR_TRAIN_TO_COMMON, common_histogram
from src.study_runtime import atomic_json, sha256
from test_optional_returns import make_complete_study

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path):
    root = make_complete_study(tmp_path)
    frozen = load_frozen(ROOT)
    for arm in ARMS:
        out = root / arm
        shutil.copytree(root / "candidate", out)
        cfg = make_mentor_config(frozen, arm, out)
        atomic_json(out / "config_resolved.json", cfg)
        atomic_json(
            out / "mentor_identity.json",
            {"model_sha256": MENTORS[arm]["model_sha256"], "frozen": True},
        )
        rows = [
            json.loads(s) for s in (out / "study_epochs.jsonl").read_text().splitlines()
        ]
        for row in rows:
            for r in row["ranks"]:
                r.pop("return_exposure")
                r["mentor_model_sha256"] = MENTORS[arm]["model_sha256"]
                counts = common_histogram(r["native_counts"], ECLAIR_TRAIN_TO_COMMON)[
                    1:
                ].tolist()
                r["guidance"] = {
                    k: counts[:]
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
                r["guidance"].update(
                    batches=r["microbatches"],
                    gradient_diagnostics=[{"weighted_logit_gradient_ratio": 0.1}],
                )
        (out / "study_epochs.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    out = root / "observed_mentor"
    mapping = yaml.safe_load(
        (ROOT / "configs/mapping_dales_to_common.yaml").read_text()
    )
    for ep in ("best", "last"):
        folder = out / f"evaluation/{ep}/dales"
        value = json.loads((folder / "metrics.json").read_text())
        matrix = np.zeros((8, 8), np.int64)
        native = np.zeros((256, 11), np.int64)
        joint = []
        for cls in range(1, 8):
            gt = next(k for k, v in mapping.items() if v == cls)
            pred = ECLAIR_TRAIN_TO_COMMON.index(cls)
            matrix[cls, cls] = 85
            matrix[cls, 0] = 15
            native[gt, pred] = 85
            native[gt, 0] = 15
            joint += [[gt, pred, pred, 85], [gt, 0, 0, 15]]
        tiles = [{**t, "matrix": matrix.tolist()} for t in value["tiles"]]
        atomic_json(
            folder / "metrics.json",
            {
                **summarize(matrix * 11),
                "tiles": tiles,
                "native_gt_by_source_prediction": (native * 11).tolist(),
            },
        )
        atomic_json(
            folder / "complete.json",
            {"tiles": 11, "metrics_sha256": sha256(folder / "metrics.json")},
        )
        if ep == "best":
            for tile in tiles:
                atomic_json(
                    folder / f"tile_{tile['tile']:04d}.json",
                    {"paired_diagnostic": {"native_gt_observed_constant": joint}},
                )
    return root


def test_full_comparison_and_collection(tmp_path):
    root = fixture(tmp_path)
    for arm in ARMS:
        result = arm_report(root, arm)
        assert result["execution"]["valid"]
    result = compare_if_ready(root)
    assert (
        result["classification"]
        == "geometry_mentor_development_pass_reverse_pair_eligible"
    )
    assert result["automatic_reverse_jobs"] == 0
    (root / "code").mkdir()
    (root / "code/file.py").write_text("not in results")
    (root / "geometry_mentor/checkpoints").mkdir()
    (root / "geometry_mentor/checkpoints/last.pt").write_bytes(b"not in results")
    bundle = collect_bundle(root)
    with tarfile.open(bundle["bundle"]) as archive:
        assert "comparison.json" in archive.getnames()
        assert not any(
            {"code", "checkpoints"} & set(Path(n).parts) for n in archive.getnames()
        )


def test_changed_exposure_or_mentor_invalidates_comparison(tmp_path):
    root = fixture(tmp_path)
    for arm in ARMS:
        arm_report(root, arm)
    path = root / "geometry_mentor/study_epochs.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows[1]["ranks"][0]["geometry_label_exposure_sha256"] = "poison"
    rows[2]["ranks"][1]["mentor_model_sha256"] = "changed"
    rows[3]["ranks"][0]["guidance"]["valid_counts"][0] += 1
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = arm_report(root, "geometry_mentor")
    assert not report["execution"]["valid"]
    assert "mentor_changed" in report["execution"]["errors"]
    assert "guidance_class_exposure" in report["execution"]["errors"]
    assert compare_if_ready(root)["classification"] == "invalid_development_comparison"


def test_comparison_waits_for_both_complete(tmp_path):
    assert compare_if_ready(tmp_path) is None
