"""CPU tests for scientific accounting, frozen contracts and storage behavior."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config_loader import load_yaml
from src.study_architecture import ARMS, block_counts
from src.study_metrics import balanced_gate, common_confusion, common_histogram, confusion, summary
from src.study_runtime import atomic_json, storage_guard, trim_jsonl_after


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_false_positives_from_nonutility_classes_remain():
    cm = np.zeros((8, 8), dtype=int)
    cm[5, 5], cm[3, 5], cm[4, 5] = 10, 20, 10
    result = summary(cm)
    assert result["per_class_iou"]["poles"] == 0.25
    assert result["precision"]["poles"] == 0.25
    assert result["recall"]["poles"] == 1.0


def test_native_vehicle_classes_merge_before_iou():
    cm = np.zeros((8, 8), dtype=int)
    cm[2, 3], cm[3, 2] = 5, 7
    merged = common_confusion(cm)
    assert merged[7, 7] == 12
    assert summary(merged)["per_class_iou"]["vehicle"] == 1.0


def test_accepted_native_histogram_merges_vehicle_counts():
    result = common_histogram([0, 0, 2, 3, 0, 5, 7, 0])
    assert result[7] == 5
    assert result[4] == 7
    assert result[5] == 5
    assert result.sum() == 17


def test_ignore_prediction_is_false_negative_and_ignore_gt_is_excluded():
    result = summary(confusion([4, 4, 0], [4, 0, 4]))
    assert result["per_class_iou"]["wires"] == 0.5
    assert result["support"][4] == 2


def test_balanced_gate_rejects_one_class_tradeoff():
    reference = summary(np.eye(8, dtype=int) * 10)
    candidate = copy.deepcopy(reference)
    candidate["all_seven_miou"] += 0.01
    candidate["utility_miou"] += 0.01
    candidate["per_class_iou"]["buildings"] -= 0.1
    result = balanced_gate(reference, reference, reference, candidate)
    assert not result["passes_observed_screen"]
    assert not result["checks"]["no_target_class_regression"]


@pytest.mark.parametrize("layers", [[1] * 7, [True] * 8, [0] * 8, [1.0] * 8, "reference"])
def test_invalid_depth_is_rejected(layers):
    with pytest.raises(ValueError):
        block_counts({"model": {"layers": layers}})


def test_default_and_config_contracts(monkeypatch):
    for key in (
        "DALES_TRAIN_ROOT",
        "DALES_TEST_ROOT",
        "DALES_CACHE_ROOT_DROP_I",
        "ECLAIR_ROOT",
        "ECLAIR_CACHE_ROOT",
    ):
        monkeypatch.setenv(key, "/not-used-in-static-tests")
    baseline = load_yaml(ROOT / "configs/m0_dales_frozen_29364.yaml")
    validate = script("preflight_architecture_study").validate
    assert block_counts({}) == ARMS["reference"]
    for arm in ARMS:
        cfg = load_yaml(ROOT / f"configs/study_dales_{arm}120.yaml")
        assert validate(cfg, baseline)["arm"] == arm
        altered = copy.deepcopy(cfg)
        altered["data"]["sampling"]["crop_size_xy_m"] = 140.0
        with pytest.raises(ValueError):
            validate(altered, baseline)
        assert not cfg["run"]["keep_epoch_snapshots"]
        assert cfg["run"]["best_checkpoint_model_only"]


def test_storage_reserves_temporary_write_and_keeps_old_file(tmp_path, monkeypatch):
    old = tmp_path / "last.pt"
    old.write_bytes(b"previous-good-state")
    monkeypatch.setattr("src.study_runtime.shutil.disk_usage", lambda _p: SimpleNamespace(free=99))
    with pytest.raises(OSError):
        storage_guard(tmp_path, 100, {"reserve_bytes": 0, "run_budget_bytes": 1000})
    assert old.read_bytes() == b"previous-good-state"


def test_storage_cap_covers_entire_run_not_only_evaluation_subdir(tmp_path, monkeypatch):
    (tmp_path / "config_resolved.yaml").write_text("run: {}")
    (tmp_path / "last.pt").write_bytes(b"x" * 100)
    nested = tmp_path / "evaluation/a/dales"
    nested.mkdir(parents=True)
    monkeypatch.setattr("src.study_runtime.shutil.disk_usage", lambda _p: SimpleNamespace(free=100000))
    with pytest.raises(OSError):
        storage_guard(nested, 50, {"reserve_bytes": 0, "run_budget_bytes": 140})


def test_failed_atomic_json_preserves_previous(tmp_path):
    path = tmp_path / "status.json"
    atomic_json(path, {"valid": True})
    with pytest.raises(ValueError):
        atomic_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"valid": True}


def test_resume_trims_only_uncommitted_epochs_and_partial_tail(tmp_path):
    path = tmp_path / "epochs.jsonl"
    path.write_text('{"epoch":1}\n{"epoch":2}\n{"epoch":3}\n{"epoch":')
    trim_jsonl_after(path, 2)
    assert [json.loads(row)["epoch"] for row in path.read_text().splitlines()] == [1, 2]
    path.write_text('not valid json\n{"epoch":2}\n')
    with pytest.raises(ValueError):
        trim_jsonl_after(path, 2)


def test_cn07_is_resolved_case_insensitively_and_unknown_refused():
    resolve = script("submit_architecture_study").resolve_excluded
    assert resolve(["C01", "cn07", "C08"]) == "cn07"
    assert resolve(["cn07.cluster.example", "C09"]) == "cn07.cluster.example"
    with pytest.raises(ValueError):
        resolve(["C01", "C08"])


def test_quota_admission_uses_user_soft_limit_not_filesystem_free(monkeypatch, tmp_path):
    module = script("submit_architecture_study")
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *a, **k: ("Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/sdb3 999999 1 999998 1% /scratch\n"),
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout="Disk quotas for user test:\nFilesystem blocks quota limit grace files quota limit grace\n/dev/sdb3 108* 200 205 7553 0 0\n"
        ),
    )
    assert module.quota_headroom(tmp_path) == 92 * 1024


def test_unknown_quota_is_not_treated_as_unlimited(monkeypatch, tmp_path):
    module = script("submit_architecture_study")
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *a, **k: ("Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/sdb3 999999 1 999998 1% /scratch\n"),
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    with pytest.raises(RuntimeError):
        module.quota_headroom(tmp_path)


def test_actual_optimizer_update_count_excludes_amp_skips():
    from src.study_runtime import StudyRuntime

    runtime = StudyRuntime.__new__(StudyRuntime)
    runtime.updates = runtime.skips = 0

    class Scaler:
        def __init__(self, scales):
            self.scales = iter(scales)
            self.scale = next(self.scales)

        def get_scale(self):
            return self.scale

        def step(self, _optimizer):
            pass

        def update(self):
            self.scale = next(self.scales)

    runtime.optimizer_step(Scaler([128.0, 64.0]), None)
    runtime.optimizer_step(Scaler([64.0, 64.0]), None)
    runtime.optimizer_step(Scaler([64.0, 128.0]), None)
    assert runtime.updates == 2
    assert runtime.skips == 1
