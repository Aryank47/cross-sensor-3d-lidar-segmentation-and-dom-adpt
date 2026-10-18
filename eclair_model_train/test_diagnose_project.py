"""Local scientific/packaging tests. Does not simulate a GPU reproduction."""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import itertools
import json
import logging
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("project_diagnostic_tests", HERE / "diagnose_project.py")
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


@contextlib.contextmanager
def workspace_temp():
    root = HERE / "test_tmp"
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as folder:
        assert Path(folder).resolve().is_relative_to(HERE)
        try:
            yield folder
        finally:
            logger = logging.getLogger("project_diagnosis")
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)


class ScientificTests(unittest.TestCase):
    def test_all_sixteen_frozen_matrices(self):
        result = d.offline_audit(d.frozen())
        self.assertEqual(result["canonical_evaluations"], 16)
        self.assertTrue(result["all_miou_match"])
        support = result["known_test_class_composition"]
        self.assertEqual(support["eclair"]["support"][7], 5775)
        self.assertEqual(support["dales"]["support"][7], 1224696)

    def test_non_common_prediction_is_false_negative(self):
        m = d.metrics(d.cm_from(np.array([7, 7, 0]), np.array([7, 0, 7])))
        self.assertEqual(m["iou"][7], .5)
        self.assertEqual(m["recall"][7], .5)
        self.assertEqual(m["precision"][7], 1.)
        self.assertEqual(m["n_valid"], 2)

    def test_majority_matches_reference_for_random_voxels(self):
        rng = np.random.default_rng(5)
        for _ in range(100):
            inv = np.r_[np.arange(5), rng.integers(0, 5, size=40)]
            labels = rng.integers(-1, 6, size=len(inv))
            labels[labels == -1] = -100
            actual = d.grouped_majority(inv, labels, 6)[0]
            expected = []
            for v in range(5):
                group = labels[(inv == v) & (labels != -100)]
                expected.append(np.bincount(group, minlength=6).argmax() if len(group) else -100)
            np.testing.assert_array_equal(actual, expected)

    def test_zero_native_train_label_is_valid(self):
        majority = d.grouped_majority(np.array([0, 0, 0]), np.array([0, 0, 1]), 2)[0]
        self.assertEqual(majority[0], 0)

    def test_ignore_does_not_win_majority(self):
        actual = d.grouped_majority(np.array([0, 0, 0, 1]), np.array([-100, -100, 3, -100]), 4)[0]
        np.testing.assert_array_equal(actual, [3, -100])

    def test_oracle_accuracy_is_bruteforce_optimal(self):
        rng = np.random.default_rng(4)
        for _ in range(30):
            cells = np.r_[np.arange(3), rng.integers(0, 3, size=7)]
            xyz = np.column_stack((cells + .1, np.zeros(len(cells)), np.zeros(len(cells))))
            labels = rng.integers(1, 4, size=len(cells))
            result = d.voxel_diagnostic(xyz, labels, labels, np.arange(4), 1.)
            best = max(np.mean(np.array(choice)[cells] == labels) for choice in itertools.product(range(1, 4), repeat=3))
            self.assertAlmostEqual(result["constant_voxel_accuracy_ceiling"], best)

    def test_majority_miou_is_not_a_ceiling(self):
        xyz = np.array([[.1, 0., 0.]]*10 + [[1.1, 0., 0.]]*10)
        gt = np.array([1]*16 + [2]*4)
        result = d.voxel_diagnostic(xyz, gt, gt, np.arange(3), 1.)
        better_macro_pred = np.array([1]*10 + [2]*10)
        self.assertGreater(d.metrics(d.cm_from(gt, better_macro_pred))["legacy_miou_all_seven"],
                           result["common_majority_backprojection"]["legacy_miou_all_seven"])

    def test_merge_and_majority_do_not_commute(self):
        xyz = np.array([[.1, 0., 0.]]*3)
        result = d.voxel_diagnostic(xyz, np.array([4, 4, 1]), np.array([1, 2, 0]), np.array([1, 4, 4]), 1.)
        self.assertEqual(result["class_common_majority_voxels"][4], 1)
        self.assertEqual(result["class_native_majority_then_map_voxels"][4], 0)

    def test_quantization_matches_evaluator_float_order(self):
        rng = np.random.default_rng(7)
        xyz = rng.uniform(0, 400, (1000, 3))
        for edge in (.1, .2, .4, .5):
            expected = np.floor((xyz.astype(np.float32)/10.)/(edge/10.)).astype(np.int32)
            np.testing.assert_array_equal(d.quantize_like_evaluator(xyz, edge), expected)

    def test_paired_thinning_preserves_anchor_and_identity(self):
        xyz = np.array([[0., 1., 3.], [2., 0., 4.], [4., 5., 0.], [2., 3., 1.]])
        raw = {"xyz": xyz, "native_labels": np.arange(4), "path": "tile"}
        keep = np.array([False, False, False, True])
        keep[np.argmin(xyz, axis=0)] = True
        subset = d.subset_raw(raw, keep)
        np.testing.assert_array_equal(subset["xyz"].min(0), [0., 0., 0.])
        np.testing.assert_array_equal(subset["native_labels"], np.arange(4)[keep])

    def test_bootstrap_same_model_has_zero_delta(self):
        a = np.zeros((5, 8, 8), np.int64)
        a[:, 1, 1] = 5
        result = d.bootstrap_delta(a, a, 3, repetitions=40)
        np.testing.assert_array_equal(result["miou_delta_95pct"], [0., 0.])
        self.assertEqual(result["fraction_replicates_without_gt_class"][7], 1.)

    def test_invalid_labels_fail(self):
        with self.assertRaises(ValueError):
            d.cm_from(np.array([8]), np.array([1]))
        with self.assertRaises(ValueError):
            d.grouped_majority(np.array([0]), np.array([8]), 8)

    def test_missing_values_are_json_null(self):
        obj = d.clean_json({"a": np.array([1., np.nan, np.inf])})
        self.assertEqual(obj, {"a": [1., None, None]})
        json.dumps(obj, allow_nan=False)

    def test_prefix_rewrites_respect_boundaries(self):
        maps = [("/data", "/new"), ("/data/deep", "/special")]
        self.assertEqual(d.remap_paths("/data/deep/file", maps), "/special/file")
        self.assertEqual(d.remap_paths("/database/file", maps), "/database/file")

    def test_payload_hash_catches_data_change(self):
        raw = {"xyz": np.zeros((2, 3)), "native_labels": np.array([1, 2]),
               "return_number": np.array([1, 1]), "number_of_returns": np.array([1, 1])}
        before = d.payload_digest(raw)
        raw["native_labels"][0] = 3
        self.assertNotEqual(before, d.payload_digest(raw))

    def test_summary_checks_paired_support(self):
        a, b = np.zeros((8, 8), np.int64), np.zeros((8, 8), np.int64)
        a[1, 1], b[1, 1] = 2, 3
        probe = {"run": "x", "domain": "eclair", "split": "test", "available": True,
                 "variants": {"v": {"metrics": d.metrics(a), "paired_clean_same_retained_points": d.metrics(b)}}}
        with self.assertRaises(RuntimeError):
            d.summarize_diagnostics([], [probe], [], {})

    def test_summary_paired_delta_and_feature_moments(self):
        cm = np.zeros((8, 8), np.int64)
        cm[1, 1] = 5
        probe = {"run": "x", "domain": "eclair", "split": "test", "available": True,
                 "variants": {"v": {"metrics": d.metrics(cm), "paired_clean_same_retained_points": d.metrics(cm)}}}
        feature = {str(c): {"n": 2, "sum": [2., 4.], "sum_sq": [2., 8.]} for c in range(1, 8)}
        result = d.summarize_diagnostics([], [probe], [], {("x", "eclair"): [{"feature_moments": feature}],
                                                          ("x", "dales"): [{"feature_moments": feature}]})
        np.testing.assert_array_equal(result["perturbation_summaries"][0]["iou_delta"], np.zeros(8))
        self.assertEqual(result["feature_domain_comparison"][0]["standardized_centroid_shift_rms"], 0.)

    def test_geometry_summary_weights_accuracy_by_points(self):
        xyz = np.array([[0., 0., 0.], [.1, 0., 0.]])
        gt = np.array([1, 2])
        vd = d.voxel_diagnostic(xyz, gt, gt, np.arange(3), .5)
        scan = {"domain": "eclair", "split": "test", "common_histogram": np.bincount(gt, minlength=8),
                "voxel_diagnostics": [vd], "returns": {"return_number": {}, "number_of_returns": {}}}
        result = d.summarize_diagnostics([scan], [], [], {})
        self.assertEqual(result["dataset_geometry"][0]["point_weighted_accuracy_ceiling"], .5)

    def test_frozen_runs_match_canonical_sources(self):
        f = d.frozen()
        self.assertEqual(len(f["runs"]), 8)
        for run in f["runs"]:
            self.assertEqual(run["epochs"], 100 if run["source"] == "eclair" else 200)
            self.assertEqual(sum(e["run"] == run["run"] for e in f["evaluations"]), 2)


class WorkflowTests(unittest.TestCase):
    def make_runner(self, temp):
        args = argparse.Namespace(out=temp, audit_only=True, preflight_only=False)
        r = d.Runner(args)
        r.signature = "test"
        return r

    def test_resume_and_forced_outer_retry(self):
        with workspace_temp() as temp:
            r = self.make_runner(temp)
            calls = []
            def work():
                calls.append(1)
                return {"value": len(calls)}
            r.task("a", work)
            r.task("a", work)
            self.assertEqual(len(calls), 1)
            r.task("a", work, resume=False)
            self.assertEqual(len(calls), 2)

    def test_failed_stage_does_not_block_independent_work(self):
        with workspace_temp() as temp:
            r = self.make_runner(temp)
            def bad():
                raise RuntimeError("test failure")
            self.assertIsNone(r.task("bad", bad))
            self.assertEqual(r.task("good", lambda: {"ok": True}), {"ok": True})
            self.assertEqual(len(r.failures), 1)

    def test_artifact_hash_invalidates_resume(self):
        with workspace_temp() as temp:
            r = self.make_runner(temp)
            artifact = Path(temp)/"payload"
            calls = []
            def work():
                calls.append(1)
                artifact.write_text("correct")
                return {"artifact_path": str(artifact), "artifact_sha256": d.digest_file(artifact)}
            r.task("artifact", work)
            artifact.write_text("changed")
            r.task("artifact", work)
            self.assertEqual(len(calls), 2)

    def test_manifest_hashes_and_archive_excludes_patches(self):
        with workspace_temp() as parent:
            temp = str(Path(parent)/"result")
            r = self.make_runner(temp)
            (Path(temp)/"working_patches").mkdir()
            (Path(temp)/"working_patches"/"private.npz").write_bytes(b"private")
            r.task("audit", lambda: {"ok": True})
            r.workflow_completed = True
            r.finish()
            manifest = json.loads((Path(temp)/"bundle_manifest.json").read_text())
            with tarfile.open(Path(parent)/"result_results.tar.gz") as tar:
                self.assertFalse(any("working_patches" in m.name for m in tar.getmembers()))
                for entry in manifest:
                    actual = hashlib.sha256(tar.extractfile(entry["path"]).read()).hexdigest()
                    self.assertEqual(actual, entry["sha256"])

    def test_project_api_signatures_statically(self):
        code = HERE.parent/"evidence"/"context_repair"/"code"
        tree = ast.parse((code/"src"/"eval_cross_common.py").read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        expected = {
            "infer_tile_point_logits": {"model", "raw", "source_cfg", "device", "amp", "window_size_m", "window_stride_m", "aggregation", "max_voxels_per_forward", "max_split_depth", "logger", "target_domain"},
            "run_model_on_voxel_subset": {"model", "coords_u", "feats_u", "sel", "num_classes", "device", "amp"},
            "align_sparse_output_to_input_coords": {"out", "coords_in_t", "device"},
        }
        for name, keys in expected.items():
            fn = functions[name]
            accepted = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            self.assertTrue(keys <= accepted)


if __name__ == "__main__":
    unittest.main()
