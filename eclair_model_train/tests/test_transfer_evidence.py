"""CPU scientific-contract tests; actual Minkowski/CUDA parity is also gated in-job."""

from __future__ import annotations

import ast
import copy
import json
import logging
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from src.features import FeatureConfig, build_features, infer_in_channels
from src.study_metrics import confusion
from src.transfer_evidence import (
    ASSETS,
    ModelPool,
    Observer,
    Store,
    bind_manifest,
    config_contract,
    validate_selection,
)
from src.transfer_evidence_report import (
    bootstrap_delta,
    create_report,
    novelty,
    source_feature_bank,
    verify_canonical,
)
from src.transfer_evidence_sampling import (
    GEOMETRY_NAMES,
    allocate,
    choose_windows,
    conditional_inputs,
    confidence_arrays,
    geometry,
    geometry_descriptor,
    perturb,
    sample_tile,
    sampling_plan,
)


class ScientificContracts(unittest.TestCase):
    def test_only_one_gpu_model_is_resident(self):
        class Model:
            base = True

            def __init__(self):
                self.calls = []
                self._feature = object()
                self._feat_cache = {"old": object()}

            def to(self, device):
                self.calls.append(device)
                return self

        a, b = Model(), Model()
        models = ModelPool({"a": a, "b": b})
        models["a"]
        models["b"]
        models["b"]
        self.assertEqual(a.calls, ["cuda:0", "cpu"])
        self.assertEqual(b.calls, ["cuda:0"])
        self.assertIsNone(a._feature)
        self.assertEqual(a._feat_cache, {})

    def test_feature_observer_unsorted_rows_duplicates_and_overlap(self):
        import sys
        from unittest.mock import patch

        class Tensor:
            def __init__(self, x):
                self.x = np.asarray(x)

            def __getitem__(self, key):
                return Tensor(self.x[key])

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.x

        fake = SimpleNamespace(align_sparse_output_to_input_coords=lambda **kw: kw["out"])
        cfg = {"data": {"patch": {"voxel_size": 0.02, "coord_norm_factor": 10}}}
        tap = Observer([0, 1, 2], cfg)
        tap.prepared(np.zeros((10, 3), np.int32), np.zeros((10, 10)), np.array([4, 9, 4]))
        with patch.dict(sys.modules, {"src.eval_cross_common": fake}):
            tap.captured = Tensor([[90.0, 9.0], [40.0, 4.0]])
            tap.forwarded(np.array([9, 4]), None, None)
            tap.captured = Tensor([[60.0, 6.0]])
            tap.forwarded(np.array([4]), None, None)
        np.testing.assert_array_equal(tap.counts, [2, 1, 2])
        np.testing.assert_array_equal(tap.features(), [[50.0, 5.0], [90.0, 9.0], [50.0, 5.0]])

    def test_probability_allocation_and_supplement_weights(self):
        counts = np.array([10, 30, 60])
        draws = np.asarray([allocate(counts, 20, i) for i in range(1000)])
        np.testing.assert_array_equal(draws.sum(1), 20)
        np.testing.assert_allclose(draws.mean(0), counts / 5, atol=0.2)
        plan = {
            "class_quotas": [[0, 3, 0, 0, 0, 0, 0, 0]],
            "native_quotas": [[0, 2]],
            "class_population": [0, 10, 0, 0, 0, 0, 0, 0],
        }
        gt, native = np.ones(10, int), np.ones(10, int)
        ids, flags, p = sample_tile(gt, native, plan, 0, 1337)
        self.assertEqual(int(((flags & 1) > 0).sum()), 3)
        np.testing.assert_array_equal(p[(flags & 1) == 0], 0)
        np.testing.assert_allclose(p[(flags & 1) > 0], 0.3)
        self.assertEqual(len(np.unique(ids)), len(ids))

    def test_frozen_assets_and_all_sampling_caps(self):
        cfg, catalog = (
            json.loads((ASSETS / "study.json").read_text()),
            json.loads((ASSETS / "scan_catalog.json").read_text()),
        )
        self.assertEqual(len(cfg["models"]), 10)
        self.assertEqual({x["epoch"] for x in cfg["models"] if x["id"] == "dales_reference"}, {162})
        self.assertEqual([x["id"] for x in cfg["models"] if not x["source_validation_verified"]], ["dales_38163"])
        for d in ("eclair", "dales"):
            for s in ("train", "test"):
                plan = sampling_plan(catalog, d, s, cfg)
                self.assertLessEqual(plan["maximum_anchors"], cfg["anchor_cap"])
                for cls in range(1, 8):
                    self.assertEqual(
                        sum(r[cls] for r in plan["class_quotas"]),
                        min(cfg["class_points"], plan["class_population"][cls]),
                    )
        for model in cfg["models"]:
            self.assertEqual(len(model["checkpoint_sha256"]), 64)
            for domain in ("dales", "eclair"):
                matrix = np.asarray(model["canonical"][domain]["matrix"])
                self.assertEqual(matrix.shape, (8, 8))
                self.assertTrue((matrix >= 0).all())
        for record in catalog.values():
            joint = np.asarray(record["returns"]["joint_common_class_by_rn_nor"])
            np.testing.assert_array_equal(joint.sum((1, 2)), record["common_histogram"])

    def test_native_argmax_is_not_probability_merge(self):
        result = confidence_arrays(np.array([[0.8, 0.7, 0.7]], np.float32), np.array([1, 7, 7]))
        self.assertEqual(int(result["pred"][0]), 1)
        self.assertEqual(int(result["common_probability_argmax"][0]), 7)
        self.assertGreater(float(result["native_logit_margin"][0]), 0)

    def test_joint_counts_and_invalid_native_records(self):
        gt, pred = np.array([0, 1, 1, 5, 11]), np.array([1, 1, 0, 5, 7])
        rn, nor = np.array([1, 1, 0, 3, 20]), np.array([1, 2, 1, 2, 20])
        joint, bad = conditional_inputs(gt, pred, rn, nor)
        self.assertEqual(int(joint.sum()), 3)
        self.assertEqual(int(bad.sum()), 2)
        native, _ = conditional_inputs(gt, pred, rn, nor, classes=256, ignore_zero=False)
        self.assertEqual(int(native.sum()), 5)
        self.assertEqual(int(native[11, 7, 16, 16]), 1)

    def test_probes_preserve_joint_validity_and_raw_id_alignment(self):
        raw = {
            "xyz": np.arange(90).reshape(30, 3),
            "return_number": np.tile([1, 2, 3], 10),
            "number_of_returns": np.full(30, 3),
            "native_labels": np.arange(30),
            "path": "tile",
        }
        before = copy.deepcopy(raw)
        shuffled, ids = perturb(raw, "return_shuffle", 1337)
        np.testing.assert_array_equal(shuffled["xyz"], raw["xyz"])
        np.testing.assert_array_equal(np.sort(shuffled["return_number"]), np.sort(raw["return_number"]))
        self.assertTrue((shuffled["return_number"] <= shuffled["number_of_returns"]).all())
        thin, retained = perturb(raw, "thin50", 1337)
        self.assertEqual(len(retained), 15)
        np.testing.assert_array_equal(thin["native_labels"], raw["native_labels"][retained])
        for k in ("xyz", "return_number", "number_of_returns"):
            np.testing.assert_array_equal(raw[k], before[k])
        constant = {**raw, "return_number": np.ones(30), "number_of_returns": np.ones(30)}
        changed, _ = perturb(constant, "return_shuffle", 1)
        np.testing.assert_array_equal(changed["return_number"], 1)

    def test_full_raw_neighbourhood_shape_and_clipped_support(self):
        # A line with 101 raw returns, deliberately more than any putative local point cap.
        xyz = np.column_stack((np.linspace(-5, 5, 101), np.zeros(101), np.zeros(101)))
        gt = np.ones(101, int)
        values = geometry(xyz, gt, np.array([50]), [0.5, 1.0, 2.0, 4.0])
        self.assertEqual(values[0, 2, 0], 41)
        self.assertAlmostEqual(float(values[0, 2, 4]), 1.0, places=6)
        self.assertEqual(values[0, 2, 10], 1.0)
        # Degenerate tile Y extent means truncation; never call it poor source support.
        self.assertTrue(np.isnan(geometry_descriptor(values)).all())
        small = geometry(xyz[:4], gt[:4], np.array([0]), [0.5, 1.0, 2.0, 4.0])
        self.assertTrue(np.isnan(small[0, :, 4]).all())

    def test_source_only_support_and_missing_overlap(self):
        source = np.column_stack((np.linspace(0, 1, 40), np.zeros(40)))
        tiles = np.arange(40) % 2
        dist, threshold = novelty(source, tiles, np.array([[0.5, 0], [20, 0], [np.nan, 0]]))
        self.assertLess(dist[0], threshold)
        self.assertGreater(dist[1], threshold)
        self.assertTrue(np.isnan(dist[2]))
        missing, boundary = novelty(source[:5], tiles[:5], source)
        self.assertIsNone(boundary)
        self.assertTrue(np.isnan(missing).all())

    def test_window_selection_ignores_model_predictions(self):
        groups = [(0, np.arange(7), np.column_stack((np.arange(7) * 140, np.zeros(7), np.zeros(7))), np.arange(1, 8))]
        a, b = choose_windows(groups, 7, 4), choose_windows(groups, 7, 4)
        self.assertEqual(a, b)
        self.assertEqual({r["class"] for r in a}, set(range(1, 8)))

    def test_selection_validation_normalizes_integer_label_keys(self):
        cfg = {
            "model": {"out_channels": 8},
            "data": {"features": {}, "patch": {}, "dales_label_map_native_to_train": {1: 0}},
        }
        other = json.loads(json.dumps(cfg))
        self.assertEqual(config_contract(cfg), config_contract(other))
        task = {"id": "dales_x", "epoch": 4, "best_val_miou": 0.8, "cfg": other, "source_validation_verified": True}
        obj = {"epoch": 4, "best_val_miou": 0.8, "cfg": cfg}
        validate_selection(obj, task, [{"epoch": "2", "val_miou": ".7"}, {"epoch": "4", "val_miou": ".8"}])
        with self.assertRaises(ValueError):
            validate_selection(obj, task, [{"epoch": "4", "val_miou": ".7"}])
        obj["cfg"]["data"]["features"] = {"include_coords": True}
        with self.assertRaises(ValueError):
            validate_selection(obj, task, [{"epoch": "4", "val_miou": ".8"}])

    def test_quota_receipt_freshness_and_capacity(self):
        import getpass

        from scripts.run_transfer_evidence_study import validate_admission

        settings = json.loads((ASSETS / "study.json").read_text())
        with tempfile.TemporaryDirectory() as d:
            receipt = {
                "version": 1,
                "user": getpass.getuser(),
                "output_dir": d,
                "checked_at_unix": 1000,
                "required_bytes": settings["minimum_quota_headroom_bytes"],
                "headroom_bytes": settings["minimum_quota_headroom_bytes"],
                "budget_bytes": settings["output_budget_bytes"],
            }
            validate_admission(receipt, d, now=1100)
            with self.assertRaises(ValueError):
                validate_admission(receipt, d, now=200000)
            with self.assertRaises(ValueError):
                validate_admission({**receipt, "headroom_bytes": 1}, d, now=1100)

    def test_atomic_stage_tampering_and_resume_contract(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(d, {"output_budget_bytes": 1000000}, float("inf"))
            store.identity = "test"
            artifact = store.npz("a.npz", ids=np.arange(5))
            store.json("done.json", {"identity": "test", "artifacts": [artifact]})
            self.assertIsNotNone(store.complete("done.json"))
            (Path(d) / "a.npz").write_bytes(b"bad")
            with self.assertRaises(ValueError):
                store.complete("done.json")
        with tempfile.TemporaryDirectory() as d:
            cfg = json.loads((ASSETS / "study.json").read_text())
            identity, manifest = bind_manifest(d, cfg, [], {}, False)
            (Path(d) / "study_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(identity, bind_manifest(d, cfg, [], {}, True)[0])
            with self.assertRaises(ValueError):
                bind_manifest(d, {**cfg, "seed": 42}, [], {}, True)

    def test_bootstrap_pairs_and_ignore_predictions(self):
        a = confusion(np.array([1, 1, 2]), np.array([1, 0, 2]))
        self.assertEqual(int(a[1, 0]), 1)
        # All seven classes are represented so no bootstrap replicate silently drops a class.
        a = np.eye(8, dtype=np.int64) * 10
        a[0] = 0
        result = bootstrap_delta(np.stack([a, a]), np.stack([a, a]), 1337)
        self.assertEqual(result["valid_resamples"], 500)
        np.testing.assert_array_equal(result["lower_pp"], 0)

    def test_feature_bank_deduplicates_overlap_by_raw_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "source").mkdir()
            for i, value in enumerate((1.0, 3.0)):
                p = root / f"source/eclair_{i:03d}_model.npz"
                np.savez(p, point_ids=[8], features=[[value, value]], gt=[1], geometry=np.zeros((1, 4, 14)))
                (root / f"source/eclair_{i:03d}.json").write_text(
                    json.dumps({"spec": {"tile": 2}, "models": {"model": {"artifact": str(p.relative_to(root))}}})
                )
            bank = source_feature_bank(root, {"id": "model", "source": "eclair"}, {"eclair/train": [0, 1]})
            self.assertEqual(len(bank["gt"]), 1)
            np.testing.assert_array_equal(bank["features"], [[2.0, 2.0]])


def numerical_namespace(source, revised=False):
    """Run the real old/new window/projection Python bodies against a controlled CPU backend."""
    tree = ast.parse(source)
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("recursively_infer_subset", "infer_tile_point_logits")
    ]
    for n in nodes:
        n.decorator_list = []
    namespace = {
        "np": np,
        "math": math,
        "FeatureConfig": FeatureConfig,
        "build_features": build_features,
        "infer_in_channels": infer_in_channels,
        "VoxelizationConfig": SimpleNamespace(from_cfg=lambda c: c),
        "is_oom_exception": lambda e: False,
    }

    def voxelize(**kwargs):
        coords, inv = np.unique(kwargs["q_int32"], axis=0, return_inverse=True)
        values = np.zeros((len(coords), kwargs["feats_p_f32"].shape[1]), np.float32)
        np.add.at(values, inv, kwargs["feats_p_f32"])
        values /= np.bincount(inv)[:, None]
        return {"coords_u": coords, "inverse_map": inv, "feats_u": values}

    def forward(**kw):
        # Deliberately depends on subset context, exposing accidental window/split changes.
        s = kw["sel"]
        f = kw["feats_u"][s]
        z = f.sum(1) + kw["coords_u"][s].sum(1) + len(s) / 10
        return np.column_stack((z, -z)).astype(np.float32)

    namespace.update(voxelize_from_q=voxelize, run_model_on_voxel_subset=forward)
    compiled = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(compiled), "<numerical-contract>", "exec"), namespace)
    return namespace


class CanonicalParity(unittest.TestCase):
    def test_original_and_observed_windows_recursion_overlap_projection(self):
        root = Path(__file__).resolve().parents[1]
        old = numerical_namespace((root / "tests/fixtures/transfer_canonical_primitives.txt").read_text())
        new = numerical_namespace((root / "src/eval_cross_common.py").read_text(), revised=True)
        rng = np.random.default_rng(4)
        xyz = rng.uniform(0, 8, (600, 3))
        xyz -= xyz.min(0)
        raw = {"xyz": xyz, "return_number": rng.integers(1, 4, 600), "number_of_returns": np.full(600, 3)}
        cfg = {
            "model": {"out_channels": 2},
            "data": {
                "features": {"use_intensity": False, "use_rgb": False},
                "patch": {"coord_norm_factor": 10.0, "voxel_size": 0.02, "make_local_coords": True},
            },
        }
        tap = SimpleNamespace(prepared=lambda *a: None, window=lambda *a: None, split=lambda *a: None)
        for ns in (old, new):
            # Same canonical origin scheduler as the committed evaluator.
            origins = ast.parse((root / "src/eval_cross_common.py").read_text())
            node = next(n for n in origins.body if isinstance(n, ast.FunctionDef) and n.name == "make_origins")
            exec(
                compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "<origins>", "exec"),
                {"List": list},
                result := {},
            )
            ns["make_origins"] = result["make_origins"]
        args = dict(
            model=None,
            raw=raw,
            source_cfg=cfg,
            device=None,
            amp=False,
            window_size_m=4.0,
            window_stride_m=2.0,
            aggregation="mean_logits",
            max_voxels_per_forward=35,
            max_split_depth=8,
            logger=logging.getLogger("parity"),
            target_domain="dales",
        )
        baseline = old["infer_tile_point_logits"](**args)
        np.testing.assert_array_equal(baseline, new["infer_tile_point_logits"](**args))
        np.testing.assert_array_equal(baseline, new["infer_tile_point_logits"](**args, observer=tap))
        self.assertEqual(baseline.shape, (600, 2))


class EndToEndReport(unittest.TestCase):
    def test_complete_small_study_and_corrupted_canonical_rejection(self):
        """Exercise the real complete reporter, including intentionally unsupported source banks."""
        import xml.etree.ElementTree as ET

        with tempfile.TemporaryDirectory() as d:
            cfg = {
                "output_budget_bytes": 2 * 1024**3,
                "test_tiles": {"dales": 2, "eclair": 2},
                "train_tiles": {"dales": 2, "eclair": 2},
                "source_windows": 2,
                "probe_windows": 2,
                "radii_m": [0.5, 1.0, 2.0, 4.0],
                "historical_reference": {"dales": "d", "eclair": "e"},
                "controlled_candidate": "other",
                "controlled_reference": "d",
            }
            store = Store(d, cfg, float("inf"))
            store.identity = "fixture"
            store.json(
                "mapping_manifest.json",
                {"native_to_common": {d: [*range(8), *([0] * 248)] for d in ("dales", "eclair")}},
            )
            gt = np.arange(1, 8, dtype=np.int64)
            ids = np.arange(7)
            cm = confusion(gt, gt)
            score = __import__("src.study_metrics", fromlist=["summary"]).summary(cm)
            tasks = [
                {
                    "id": k,
                    "source": domain,
                    "canonical": {a: {"matrix": cm * 2, "tiles": 2} for a in ("dales", "eclair")},
                }
                for k, domain in [("d", "dales"), ("e", "eclair")]
            ]
            # JSON compatibility for the same task objects used by the actual reporter.
            for t in tasks:
                for v in t["canonical"].values():
                    v["matrix"] = v["matrix"].tolist()
            banks, windows = {}, {}
            geo = np.zeros((7, 4, len(GEOMETRY_NAMES)), np.float32)
            geo[:, :, 0] = 20
            geo[:, :, 4] = 0.8
            geo[:, :, 10] = 1
            features = np.column_stack((gt, gt, gt, gt)).astype(np.float32)
            xyz = np.column_stack((ids, ids, ids)).astype(float)
            for domain in ("dales", "eclair"):
                for split in ("train", "test"):
                    windows[domain + "/" + split] = [{"tile": 0}, {"tile": 1}]
                    for i in range(2):
                        path = f"geometry/{domain}_{split}_{i:04d}.npz"
                        store.npz(
                            path,
                            ids=ids,
                            flags=np.ones(7, np.uint8),
                            probability=np.ones(7),
                            xyz=xyz,
                            gt=gt,
                            native=gt,
                            geometry=geo,
                            rn=np.ones(7),
                            nor=np.ones(7),
                        )
                        banks[(domain, split, i)] = path
                for i in range(2):
                    row = {"identity": store.identity, "artifacts": [], "models": {}, "agreement": {}, "gallery": []}
                    for task in tasks:
                        native_cm = np.zeros((256, 8), np.int64)
                        native_cm[gt, gt] = 1
                        joint, invalid = conditional_inputs(gt, gt, np.ones(7), np.ones(7))
                        njoint, ninvalid = conditional_inputs(
                            gt, gt, np.ones(7), np.ones(7), classes=256, ignore_zero=False
                        )
                        conf = np.zeros((8, 2, 10), np.int64)
                        conf[gt, 0, 9] = 1
                        p = store.npz(
                            f"test/{domain}_{i:04d}_{task['id']}.npz",
                            point_ids=ids,
                            features=features,
                            gt=gt,
                            native=gt,
                            pred=gt,
                            native_confidence=np.ones(7),
                            mapped_class_probability=np.ones(7),
                            native_logit_margin=np.ones(7),
                            pooled_inputs=np.zeros((7, 10)),
                            window_edge_m=np.ones(7),
                            split_depth=np.zeros(7),
                            joint_returns=joint,
                            invalid_returns=invalid,
                            pooled_sum=np.zeros((64, 10)),
                            pooled_square=np.zeros((64, 10)),
                            confidence_hist=conf,
                            confidence_sums=conf.astype(float),
                            common_confidence_hist=conf,
                            native_logit_margin_hist=np.concatenate(
                                (cm[:, :, None], np.zeros((8, 8, 8), np.int64)), axis=2
                            ),
                            window_edge_confusions=np.stack([cm, cm * 0, cm * 0, cm * 0]),
                            actual_subset_edge_confusions=np.stack([cm, cm * 0, cm * 0, cm * 0]),
                            split_depth_confusions=np.stack([cm, *[cm * 0 for _ in range(8)]]),
                            native_gt_by_native_prediction=native_cm,
                            native_gt_by_common_prediction=native_cm,
                            native_return_joint=njoint,
                            native_invalid_returns=ninvalid,
                        )
                        row["models"][task["id"]] = {
                            **score,
                            "artifact": p["path"],
                            "restricted_common_outputs": score,
                            "probability_merge_rule": score,
                        }
                        row["artifacts"].append(p)
                    if i == 0:
                        p = store.npz(f"gallery/{domain}_example.npz", xyz=xyz, gt=gt)
                        row["gallery"] = [
                            {
                                "group": [domain, 1, "persistent"],
                                "tile": 0,
                                "anchor_point": 0,
                                "native_label": 1,
                                "saved_points": 7,
                                "raw_neighbourhood_points": 7,
                                "artifact": p,
                            }
                        ]
                        row["artifacts"].append(p)
                    store.json(f"test/{domain}_{i:04d}.json", row)
                    source = {"identity": store.identity, "spec": {"tile": i}, "models": {}, "artifacts": []}
                    task = next(t for t in tasks if t["source"] == domain)
                    p = store.npz(
                        f"source/{domain}_{i:03d}_{task['id']}.npz",
                        point_ids=ids,
                        gt=gt,
                        features=features,
                        geometry=geo,
                    )
                    source["models"][task["id"]] = {"artifact": p["path"]}
                    source["artifacts"].append(p)
                    store.json(f"source/{domain}_{i:03d}.json", source)
                    probe = {"identity": store.identity, "models": {}, "artifacts": []}
                    for task in tasks:
                        probe["models"][task["id"]] = {
                            "identity_control_passed": True,
                            "state_unchanged": True,
                            "probes": [
                                {
                                    "mode": mode,
                                    "seed": seed,
                                    "effective_change": {"lost_voxel_fraction": 0.5},
                                    "clean_on_retained": score,
                                    "perturbed": score,
                                }
                                for mode in ("thin50", "return_shuffle")
                                for seed in (1, 2)
                            ],
                        }
                    store.json(f"probes/{domain}_{i:03d}.json", probe)
            coverage = create_report(store, tasks, banks, windows, cfg)
            self.assertEqual(coverage["canonical_pairs"], 4)
            self.assertTrue((Path(d) / "Research_Decision_Report.md").is_file())
            for svg in (Path(d) / "gallery").glob("*.svg"):
                ET.parse(svg)
            text = (Path(d) / "feature_support.csv").read_text()
            self.assertIn("False", text)  # insufficient reference coverage stays unavailable
            tasks[0]["canonical"]["dales"]["matrix"][1][1] += 1
            with self.assertRaises(ValueError):
                verify_canonical(d, tasks, cfg)


if __name__ == "__main__":
    unittest.main()
