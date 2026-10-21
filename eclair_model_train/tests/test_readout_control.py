"""CPU contract tests; GPU projection smoke runs inside the allocated evaluation job."""

import copy
import getpass
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

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
    safe_output,
    validate_admission,
    validate_metrics,
    verify_original_sources,
    verify_paired,
)
from src.study_metrics import summary  # noqa: E402
from src.study_runtime import sha256  # noqa: E402


def head(channels=2):
    return {"target_labels_used_for_fit_or_selection": False,
            "inverse_native_frequency_weighted_ridge": True,
            "standardization_mean": [1.] * channels, "standardization_std": [2.] * channels,
            "coefficients": np.arange((channels + 1) * 8).reshape(channels + 1, 8).tolist(),
            "penalty": 1., "selected_source_val_miou": .5,
            "source_val_candidates": [{"penalty": 10., "miou": .5}, {"penalty": 1., "miou": .5}]}


def record():
    cm = np.diag([0] + [10] * 7)
    native = np.zeros((256, 8), dtype=int)
    native[1:8] = cm[1:8]
    return {**summary(cm), "tiles": [{"tile": 0, **summary(cm)}],
            "native_gt_by_common_prediction": native.tolist(),
            "contract": {"checkpoint": "old", "dataset": {"split": "test"}}}


class ReadoutControlTests(unittest.TestCase):
    def test_projection_standardization_bias_native_classes(self):
        value = head()
        features = np.array([[3., 5.], [1., 1.]])
        coefficients = np.asarray(value["coefficients"])
        expected = np.vstack((coefficients[0] + 2 * coefficients[1] + coefficients[2], coefficients[2]))
        np.testing.assert_array_equal(project_features(features, value), expected)
        self.assertEqual(expected.shape, (2, 8))  # do not prematurely merge the two vehicle outputs

    def test_invalid_head_and_selection_rejected(self):
        mutations = [
            lambda h: h.update(target_labels_used_for_fit_or_selection=True),
            lambda h: h.update(target_labels_used_for_fit_or_selection=None),
            lambda h: h.update(standardization_std=[0, 1]),
            lambda h: h.update(standardization_mean=[float("nan"), 1]),
            lambda h: h.update(coefficients=[[1] * 7] * 3),
            lambda h: h.update(penalty=10),  # tie must keep source-selected smaller penalty
            lambda h: h.update(source_val_candidates=[{"penalty": -1, "miou": .5}]),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                value = head()
                mutation(value)
                with self.assertRaises(ValueError):
                    head_arrays(value, channels=2)

    def test_output_protects_selected_and_unselected_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp) / "study"
            for destination in (study, study.parent, study / "context", study / "fine" / "new"):
                with self.assertRaises(ValueError):
                    safe_output(study, destination, ["reference"])
            self.assertEqual(safe_output(study, study / "readout", ["reference"]), study / "readout")

    def test_point_protocol_and_subtype_mismatches_rejected(self):
        original = record()
        replacement = copy.deepcopy(original)
        replacement["contract"]["checkpoint"] = "readout"
        verify_paired(original, replacement)
        for mutate in (
            lambda r: r["contract"].update(dataset={"split": "val"}),
            lambda r: r["native_gt_by_common_prediction"][1].__setitem__(1, 9),
            lambda r: r["tiles"][0].update(tile=1),
        ):
            value = copy.deepcopy(replacement)
            mutate(value)
            with self.assertRaises(ValueError):
                verify_paired(original, value)

    def test_corrupt_aggregate_or_tile_support_rejected(self):
        for mutate in (
            lambda r: r.update(all_seven_miou=.2),
            lambda r: r["support"].__setitem__(1, 9),
            lambda r: r["tiles"][0]["confusion_matrix"][1].__setitem__(1, 9),
            lambda r: r["per_class_iou"].update(poles=0),
        ):
            value = record()
            mutate(value)
            with self.assertRaises(ValueError):
                validate_metrics(value)

    def test_utility_gain_does_not_hide_class_regression(self):
        original = record()
        cm = np.asarray(original["confusion_matrix"])
        cm[4, 4], cm[4, 5] = 5, 5
        original = {**original, **summary(cm), "tiles": [{"tile": 0, **summary(cm)}]}
        replacement = record()
        cm = np.asarray(replacement["confusion_matrix"])
        cm[3, 3], cm[3, 2] = 5, 5
        replacement = {**replacement, **summary(cm), "tiles": [{"tile": 0, **summary(cm)}]}
        result = comparison({"dales": original, "eclair": original}, {"dales": replacement, "eclair": replacement})
        self.assertGreater(result["domains"]["eclair"]["utility_delta_pp"], 0)
        self.assertFalse(result["balanced_screen"]["passes_observed_screen"])

    def test_quota_identity_age_and_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = {"checked_at_unix": 1000., "user": getpass.getuser(), "output_dir": tmp,
                       "budget_bytes": OUTPUT_BUDGET, "reserve_bytes": RESERVE,
                       "headroom_bytes": OUTPUT_BUDGET + RESERVE}
            validate_admission(receipt, tmp, now=1100)
            for change in ({"checked_at_unix": -200000}, {"user": "someone-else"}, {"headroom_bytes": 0}):
                with self.assertRaises(ValueError):
                    validate_admission({**receipt, **change}, tmp, now=1100)

    def test_resume_rejects_changed_evaluator_or_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = {"code": "first", "inputs": "first"}
            identity = bind_plan(tmp, plan, False)
            path = Path(tmp) / "experiment_manifest.json"
            path.write_text(json.dumps({"identity": identity}))
            self.assertEqual(identity, bind_plan(tmp, plan, True))
            with self.assertRaises(ValueError):
                bind_plan(tmp, plan, False)
            with self.assertRaises(ValueError):
                bind_plan(tmp, {**plan, "code": "changed"}, True)

    def test_original_source_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            arm, code = Path(tmp) / "arm", Path(tmp) / "code"
            arm.mkdir()
            (code / "src").mkdir(parents=True)
            source = code / "src/old.py"
            source.write_text("original\n")
            manifest = arm / "source_manifest.sha256"
            manifest.write_text(f"{sha256(source)}  src/old.py\n")
            (arm / "STATUS.json").write_text(json.dumps({"source_manifest_sha256": sha256(manifest)}))
            verify_original_sources(arm, code)
            source.write_text("changed\n")
            with self.assertRaises(ValueError):
                verify_original_sources(arm, code)

    def test_plan_deduplicates_identical_endpoints_and_validates_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            study, code = Path(tmp) / "study", Path(tmp) / "code"
            arm = study / "reference"
            (arm / "checkpoints").mkdir(parents=True)
            (code / "src").mkdir(parents=True)
            (code / "scripts").mkdir()
            for filename in ("run_readout_control.py", "submit_readout_control.py", "slurm_readout_control.sh"):
                (code / "scripts" / filename).write_text("test\n")

            def save(path, payload):
                path.write_text(json.dumps(payload))

            manifest = arm / "source_manifest.sha256"
            manifest.write_text("")
            save(arm / "STATUS.json", {"status": "complete", "source_manifest_sha256": sha256(manifest)})
            save(arm / "checkpoint_selection.json", {"primary": "source_all_native_validation_miou",
                 "target_labels_used": False, "best_epoch": 200, "last_epoch": 200})
            model_id = "a" * 64
            save(arm / "evaluation_manifest.json", {"target_checkpoint_selection": False, "checkpoints": {
                endpoint: {"model_sha256": model_id, "epoch": 200} for endpoint in ("best", "last")}})
            cfg = {"model": {"out_channels": 8}, "data": {"features": {}, "patch": {}}}
            save(arm / "config_resolved.json", cfg)
            (arm / "checkpoints/best.pt").write_bytes(b"dummy checkpoint identity; no torch needed")
            stage = arm / "evaluation" / model_id
            stage.mkdir(parents=True)
            save(stage / "frozen_head_results.json", head(channels=128))
            save(stage / "bounded_complete.json", {"frozen_head_results.json": sha256(stage / "frozen_head_results.json")})
            baseline = record()
            baseline["contract"].update(model=cfg["model"], features={}, patch={}, voxelization=None,
                                        window_m=120, stride_m=120, aggregation="mean_logits", amp=False)
            for domain in ("dales", "eclair"):
                (stage / domain).mkdir()
                save(stage / domain / "metrics.json", baseline)
            plan = make_plan(study, ["reference"], ["best", "last"], code)
            self.assertEqual(len(plan["tasks"]), 1)
            self.assertEqual(plan["tasks"][0]["endpoints"], ["best", "last"])
            self.assertFalse(plan["target_fitting_or_selection"])
            with self.assertRaises(ValueError):
                make_plan(study, ["reference", "reference"], ["best"], code)
            changed = json.loads((stage / "frozen_head_results.json").read_text())
            changed["penalty"] = 10
            save(stage / "frozen_head_results.json", changed)
            with self.assertRaises(ValueError):
                make_plan(study, ["reference"], ["best"], code)


if __name__ == "__main__":
    unittest.main()
