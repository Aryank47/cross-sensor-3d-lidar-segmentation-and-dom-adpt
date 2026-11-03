"""CPU contracts, full-distribution autograd and experimental decision checks.

Works with unittest on the cluster; PyTorch tests skip on lightweight laptops.
"""

from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_feature_contract_study import load_frozen
from scripts.submit_mentor_study import allocation_admission, receipt
from scripts.submit_optional_return_study import export_code
from src.feature_contract_study import make_config
from src.mentor_guidance import guidance_weight, validate_mentor_contract
from src.mentor_study import (
    ARMS,
    decide_comparison,
    make_mentor_config,
    original_student_identity,
    student_identity,
    validate_mentor_config,
)
from src.study_runtime import atomic_json

HAS_TORCH = importlib.util.find_spec("torch") is not None


class Contracts(unittest.TestCase):
    def test_mentor_metadata_cannot_change_with_identical_weights(self):
        frozen = load_frozen(ROOT)
        student = make_mentor_config(
            frozen, "geometry_mentor", "/tmp/study/geometry_mentor"
        )
        reference = make_config(frozen, "constant", "/tmp/old/constant")
        validate_mentor_contract(reference, reference, student, "constant_pair")
        for section, key, value in [
            ("features", "return_input_mode", "observed"),
            ("patch", "voxel_size", 0.04),
            ("label_space", "ignore_index", -1),
        ]:
            changed = copy.deepcopy(reference)
            changed["data"][section][key] = value
            with self.assertRaises(ValueError):
                validate_mentor_contract(changed, reference, student, "constant_pair")
        changed = copy.deepcopy(student)
        changed["data"]["features"]["include_coords"] = True
        with self.assertRaises(ValueError):
            validate_mentor_contract(reference, reference, changed, "constant_pair")

    def test_fixed_student_config_and_mentor_identity(self):
        frozen = load_frozen(ROOT)
        configs = [make_mentor_config(frozen, a, Path("/tmp/study") / a) for a in ARMS]
        for cfg in configs:
            self.assertEqual(validate_mentor_config(cfg, frozen), cfg["study"]["arm"])
            self.assertEqual(
                original_student_identity(cfg),
                original_student_identity(make_config(frozen, "observed", "/tmp/old")),
            )
            self.assertEqual(cfg["data"]["features"]["return_input_mode"], "observed")
        self.assertEqual(student_identity(configs[0]), student_identity(configs[1]))
        self.assertEqual(configs[0]["study"]["mentor"]["epoch"], 72)
        self.assertEqual(configs[1]["study"]["mentor"]["epoch"], 68)
        for path, value in [
            (("loss", "alpha"), [1] * 11),
            (("study", "guidance_max_weight"), 0.2),
            (("study", "mentor"), {}),
        ]:
            cfg = copy.deepcopy(configs[1])
            cfg[path[0]][path[1]] = value
            with self.assertRaises((ValueError, KeyError)):
                validate_mentor_config(cfg, frozen)
        cfg = copy.deepcopy(configs[0])
        cfg["run"]["resume_from"] = str(
            Path(cfg["run"]["out_dir"]) / "checkpoints/last.pt"
        )
        validate_mentor_config(cfg, frozen)
        cfg["run"]["resume_from"] = "/tmp/other.pt"
        with self.assertRaises(ValueError):
            validate_mentor_config(cfg, frozen)

    def test_fixed_schedule(self):
        self.assertAlmostEqual(guidance_weight(1), 0.01)
        self.assertAlmostEqual(guidance_weight(10), 0.1)
        self.assertAlmostEqual(guidance_weight(100), 0.1)
        for args in [(0,), (1, 0.2), (1, 0.1, 20)]:
            with self.assertRaises(ValueError):
                guidance_weight(*args)

    def test_real_export_from_repository_subdirectory(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            root = repo / "eclair_model_train"
            root.mkdir(parents=True)
            (root / "train.py").write_text("print(1)\n")
            (repo / "outside").write_text("not exported")

            def git(*args):
                return subprocess.check_output(
                    ["git", *args], cwd=repo, text=True, stderr=subprocess.PIPE
                ).strip()

            git("init")
            git("add", ".")
            git(
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-m",
                "fixture",
            )
            output = Path(temporary) / "export"
            result = export_code(root, output, git("rev-parse", "HEAD"))
            self.assertEqual(set(result["files"]), {"train.py"})
            self.assertFalse((output / "outside").exists())

    def test_quota_fallback_and_expiry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = receipt(root, "commit", 11 * 1024**3)
            atomic_json(root / "quota_admission.json", value)
            atomic_json(root / "code_provenance.json", {"commit": "commit"})
            with patch(
                "scripts.submit_mentor_study.quota_headroom",
                side_effect=RuntimeError("mount unavailable"),
            ):
                self.assertIn(
                    "snapshot", allocation_admission(root / "geometry_mentor")["mode"]
                )
                value["checked_at_unix"] -= 49 * 3600
                atomic_json(root / "quota_admission.json", value)
                with self.assertRaises(RuntimeError):
                    allocation_admission(root / "geometry_mentor")


def fake_report(
    source_loss=0.0,
    target_gain=2.0,
    valid=True,
    classification="clear_development_pass_reverse_pair_eligible",
):
    values = [0.0] * 7 + [target_gain, target_gain]
    source = [0.0] * 9
    source[6] = source_loss
    return {
        "execution": {"valid": valid},
        "classification_vs_original": classification,
        "endpoints": {
            ep: {
                "eclair": {"delta_vs_original_pp": source},
                "dales": {"delta_vs_original_pp": values},
            }
            for ep in ("best", "last")
        },
    }


def comparison(target_gain=2.0, interval_low=1.0, last_loss=0.0):
    result = {}
    for ep in ("best", "last"):
        result[ep] = {
            "eclair": {"delta_pp": [0.0] * 9},
            "dales": {
                "delta_pp": [0.0] * 7 + [target_gain, target_gain],
                "intervals": {
                    n: {"low_pp": interval_low, "high_pp": 3.0}
                    for n in ("all_seven_miou", "utility_miou")
                },
            },
        }
    result["last"]["eclair"]["delta_pp"][6] = last_loss
    return result


class Decisions(unittest.TestCase):
    def test_pass_requires_both_references(self):
        self.assertEqual(
            decide_comparison(fake_report(), fake_report(), comparison()),
            "geometry_mentor_development_pass_reverse_pair_eligible",
        )
        bad = fake_report(classification="source_preservation_failed_stop_recipe")
        self.assertIn(
            "source_preservation_failed",
            decide_comparison(bad, fake_report(), comparison()),
        )

    def test_invalid_and_weak_control(self):
        self.assertEqual(
            decide_comparison(fake_report(valid=False), fake_report(), comparison()),
            "invalid_development_comparison",
        )
        self.assertEqual(
            decide_comparison(fake_report(), fake_report(source_loss=-3), comparison()),
            "fresh_control_source_preservation_failed_stop",
        )

    def test_no_geometry_credit_for_generic_guidance(self):
        for changes in (
            comparison(target_gain=0.5),
            comparison(interval_low=-0.1),
            comparison(last_loss=-3),
        ):
            self.assertEqual(
                decide_comparison(fake_report(), fake_report(), changes),
                "generic_guidance_gain_no_geometry_specific_advancement",
            )

    def test_target_class_regression_veto(self):
        change = comparison()
        change["best"]["dales"]["delta_pp"][2] = -3
        self.assertNotIn(
            "development_pass", decide_comparison(fake_report(), fake_report(), change)
        )


@unittest.skipUnless(HAS_TORCH, "PyTorch is exercised on the cluster and GPU preflight")
class GuidanceAutograd(unittest.TestCase):
    def test_nonfinite_mentor_cannot_hide_in_ineligible_mask(self):
        t = self.t
        from src.mentor_guidance import check_mentor_finite

        q = t.ones(1, 11) / 11
        q[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.loss(t.zeros(1, 11, requires_grad=True), q, t.tensor([10]))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                check_mentor_finite(q, directory)
            self.assertTrue((Path(directory) / "numerical_failure.json").is_file())

    def setUp(self):
        import torch

        self.t = torch
        from src.mentor_guidance import distribution_guidance

        self.loss = distribution_guidance

    def test_competing_class_structure_not_binary_reweighting(self):
        t = self.t
        p = t.tensor([[0.1, 0.1, 0.2, 0, 0, 0, 0, 0, 0, 0, 0.6]]).clamp_min(1e-7)
        p = p / p.sum(1, keepdim=True)
        q = t.tensor([[0.2, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.6]]).clamp_min(1e-7)
        q = q / q.sum(1, keepdim=True)
        z = p.log().requires_grad_()
        loss, stats = self.loss(z, q, t.tensor([10]))
        loss.backward()
        self.assertGreater(float(loss), 0)
        self.assertLess(abs(float(z.grad[0, 10])), 1e-5)
        self.assertGreater(float(z.grad[0, :3].norm()), 0.01)
        self.assertEqual(stats["eligible_counts"][-1], 1)

    def test_wrong_native_subtype_is_not_correct_guidance(self):
        t = self.t
        z = t.zeros(2, 11, requires_grad=True)
        q = t.zeros(2, 11)
        q[0, 6] = 1
        q[1, 10] = 1
        loss, stats = self.loss(z, q, t.tensor([5, 10]))
        loss.backward()
        self.assertEqual(stats["eligible_counts"][3], 0)
        self.assertEqual(stats["eligible_counts"][6], 1)
        self.assertEqual(float(z.grad[0].abs().sum()), 0)

    def test_denominator_includes_ineligible_class_points(self):
        t = self.t
        q = t.zeros(1, 11)
        q[0, 10] = 1
        a, _ = self.loss(t.zeros(1, 11, requires_grad=True), q, t.tensor([10]))
        q2 = t.cat([q, t.eye(11)[1:2]], 0)
        b, stats = self.loss(t.zeros(2, 11, requires_grad=True), q2, t.tensor([10, 10]))
        self.assertAlmostEqual(float(b), float(a) / 2, places=6)
        self.assertEqual(stats["valid_counts"][-1], 2)

    def test_ignore_source_categories_and_empty_guidance(self):
        t = self.t
        z = t.zeros(3, 11, requires_grad=True)
        q = t.eye(11)[[0, 4, 1]]
        loss, stats = self.loss(z, q, t.tensor([0, 4, -100]))
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertEqual(float(z.grad.abs().sum()), 0)
        self.assertEqual(sum(stats["valid_counts"]), 0)

    def test_teacher_is_detached_and_shapes_validated(self):
        t = self.t
        z = t.zeros(1, 11, requires_grad=True)
        with self.assertRaises(ValueError):
            self.loss(z, t.ones(1, 11, requires_grad=True), t.tensor([1]))
        with self.assertRaises(ValueError):
            self.loss(z, t.ones(1, 10), t.tensor([1]))
        with self.assertRaises(ValueError):
            self.loss(z, t.ones(1, 11) / 11, t.tensor([11]))

    def test_backward_moves_towards_full_teacher_distribution(self):
        t = self.t
        z = t.zeros(1, 11, requires_grad=True)
        q = t.ones(1, 11) * 0.02
        q[0, 10] = 0.8
        before, _ = self.loss(z, q, t.tensor([10]))
        before.backward()
        after, _ = self.loss(
            (z - 0.2 * z.grad).detach().requires_grad_(), q, t.tensor([10])
        )
        self.assertLess(float(after), float(before))


if __name__ == "__main__":
    unittest.main()
