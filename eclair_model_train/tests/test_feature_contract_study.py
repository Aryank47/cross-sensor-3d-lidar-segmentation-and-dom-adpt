"""CPU checks for the paired input experiment's failure and scoring contracts."""

from __future__ import annotations

import copy
import json
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_feature_contract_results import collect
from scripts.submit_feature_contract_study import (
    MIN_HEADROOM,
    admission_receipt,
    allocation_admission,
    load_frozen,
)
from src.feature_contract_study import (
    ARMS,
    EXPECTED_TILES,
    bootstrap,
    compare_study,
    fingerprint,
    make_config,
    metric_vector,
    outcome,
    pair_identity,
    summarize,
    validate_config,
)
from src.features import FeatureConfig, assert_checkpoint_feature_config, build_features
from src.study_metrics import (
    DALES_TRAIN_TO_COMMON,
    ECLAIR_TRAIN_TO_COMMON,
    common_confusion,
    common_histogram,
    source_train_to_common,
)
from src.study_runtime import atomic_json, sha256


@pytest.fixture
def frozen():
    return load_frozen(ROOT)


def features(mode="observed", n=12, cached=False):
    rn = np.arange(n) % 5 + 1
    nr = np.arange(n)[::-1] % 5 + 1
    values = {
        "xyz_local": np.zeros((n, 3)),
        "intensity": None,
        "return_number": rn,
        "number_of_returns": nr,
        "rgb": None,
        "cfg": FeatureConfig(use_intensity=False, return_input_mode=mode),
    }
    if cached:
        values.update(
            return_number_1h=np.eye(5, dtype=np.uint8)[rn - 1],
            number_of_returns_1h=np.eye(5, dtype=np.uint8)[nr - 1],
        )
    return values


def test_observed_matches_legacy_and_cached_without_randomness():
    raw, cached = features(), features(cached=True)
    before = np.random.get_state()
    result = build_features(**raw)
    expected = np.concatenate(
        [
            np.eye(5, dtype=np.float32)[raw[k] - 1]
            for k in ("return_number", "number_of_returns")
        ],
        axis=1,
    )
    assert np.array_equal(result, expected)
    assert np.array_equal(result, build_features(**cached))
    after = np.random.get_state()
    assert (
        before[0] == after[0]
        and np.array_equal(before[1], after[1])
        and before[2:] == after[2:]
    )


@pytest.mark.parametrize("n", [0, 1, 13])
def test_constant_ignores_raw_and_cached_fields_without_mutation(n):
    values = features("constant_pair", n, cached=True)
    backup = copy.deepcopy(values)
    values["return_number"][:] = 99
    values["number_of_returns_1h"][:] = 0
    expected = np.tile(np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0], np.float32), (n, 1))
    assert np.array_equal(build_features(**values), expected)
    assert np.array_equal(values["xyz_local"], backup["xyz_local"])
    values.update(
        return_number=None,
        number_of_returns=None,
        return_number_1h=None,
        number_of_returns_1h=None,
    )
    assert np.array_equal(build_features(**values), expected)


@pytest.mark.parametrize(
    "extra",
    [
        {"use_intensity": True},
        {"include_coords": True},
        {"returns_onehot_k": 4},
        {"use_number_of_returns": False},
        {"use_rgb": True},
    ],
)
def test_invalid_constant_contract_fails(extra):
    cfg = {"use_intensity": False, "return_input_mode": "constant_pair", **extra}
    with pytest.raises(ValueError):
        FeatureConfig(**cfg)


def test_unknown_feature_mode_fails():
    with pytest.raises(ValueError):
        replace(FeatureConfig(), return_input_mode="typo")


def test_checkpoint_mode_mismatch_even_with_same_channels(frozen, tmp_path):
    observed = make_config(frozen, "observed", tmp_path / "observed")
    constant = make_config(frozen, "constant", tmp_path / "constant")
    assert_checkpoint_feature_config({"cfg": observed}, observed)
    assert_checkpoint_feature_config({}, observed)
    for checkpoint in ({}, {"cfg": observed}):
        with pytest.raises(ValueError):
            assert_checkpoint_feature_config(checkpoint, constant)
    with pytest.raises(ValueError):
        assert_checkpoint_feature_config({"cfg": constant}, observed)


def test_frozen_source_and_only_allowed_differences(frozen, tmp_path):
    a, b = [make_config(frozen, arm, tmp_path / arm) for arm in ARMS]
    assert validate_config(a, frozen) == "observed"
    assert validate_config(b, frozen) == "constant"
    assert a["loss"]["alpha"] is None
    assert pair_identity(a) == pair_identity(b)
    assert "return_input_mode" not in frozen["data"]["features"]
    a["run"]["resume_from"] = str(tmp_path / "observed/checkpoints/last.pt")
    validate_config(a, frozen)
    a["run"]["resume_from"] = str(tmp_path / "constant/checkpoints/last.pt")
    with pytest.raises(ValueError):
        validate_config(a, frozen)
    b["loss"]["alpha"] = [1] * 11
    with pytest.raises(ValueError):
        validate_config(b, frozen)


def test_eclair_mapping_merges_subtypes_and_preserves_ignore_predictions(
    frozen, tmp_path
):
    cfg = make_config(frozen, "observed", tmp_path)
    assert source_train_to_common(cfg) == ECLAIR_TRAIN_TO_COMMON
    cm = np.zeros((11, 11), np.int64)
    cm[5, 6] = 3  # wire subtype -> other wire subtype is correct after merging
    cm[7, 0] = 2  # pole -> ignored output remains a false negative
    cm[0, 7] = 9  # ignored ground truth contributes no score
    result = common_confusion(cm, source_train_to_common(cfg))
    assert result[4, 4] == 3 and result[5, 0] == 2 and result.sum() == 5
    assert common_histogram(np.ones(11, np.int64), ECLAIR_TRAIN_TO_COMMON).tolist() == [
        2,
        1,
        1,
        1,
        2,
        2,
        1,
        1,
    ]
    assert (
        source_train_to_common(
            {"data": {"dataset": "dales"}, "model": {"out_channels": 8}}
        )
        == DALES_TRAIN_TO_COMMON
    )
    with pytest.raises(ValueError):
        source_train_to_common({"model": {"out_channels": 11}})


def test_all_class_metrics_and_ignore_column():
    cm = np.eye(8, dtype=np.int64) * 20
    cm[0] = 0
    cm[5, 0] = 20
    result = summarize(cm)
    assert result["per_class_iou"]["poles"] == 0.5
    assert result["per_class_f1"]["poles"] == pytest.approx(2 / 3)
    vector = metric_vector(cm)
    assert vector[4] == 0.5 and vector[8] == 0.75
    intervals = bootstrap([cm] * 4, [cm] * 4, samples=128)
    assert all(v == {"low_pp": 0.0, "high_pp": 0.0} for v in intervals.values())


@pytest.mark.parametrize(
    "source,target,valid,label",
    [
        ([0] * 9, [0] * 9, False, "invalid_execution"),
        ([-3] + [0] * 8, [0] * 9, True, "source_competence_failed"),
        ([0] * 9, [0] * 6 + [-3, 0, 0], True, "observed_material_target_regression"),
        ([0] * 9, [1] * 9, True, "strict_observed_positive_confirmation_candidate"),
        ([0] * 9, [0] * 9, True, "practical_preservation_confirmation_candidate"),
    ],
)
def test_ordered_decision_rules(source, target, valid, label):
    intervals = bootstrap(
        [np.eye(8, dtype=np.int64)], [np.eye(8, dtype=np.int64)], samples=10
    )
    assert outcome(source, target, intervals, valid) == label
    for value in intervals.values():
        value["low_pp"] = -10
    assert outcome([0] * 9, [0] * 9, intervals) == "unresolved_practical_comparison"


def test_quota_snapshot_fallback_and_identity_fail_closed(tmp_path, monkeypatch):
    import scripts.submit_feature_contract_study as submit

    out = tmp_path / "study/constant"
    receipt = admission_receipt(out.parent, ARMS, MIN_HEADROOM)
    monkeypatch.setenv("FEATURE_QUOTA_ADMISSION", json.dumps(receipt))

    def unavailable(_):
        raise RuntimeError("NFS mount not in quota response")

    monkeypatch.setattr(submit, "quota_headroom", unavailable)
    result = allocation_admission(out)
    assert result["mode"] == "validated_login_snapshot_compute_query_unavailable"
    assert result["quota_reservation"] is False
    for key, value in (
        ("checked_at_unix", 0),
        ("arms", ["observed"]),
        ("user", "another_user"),
        ("quota_headroom_bytes", 1),
    ):
        bad = {**receipt, key: value}
        monkeypatch.setenv("FEATURE_QUOTA_ADMISSION", json.dumps(bad))
        with pytest.raises(RuntimeError):
            allocation_admission(out)
    monkeypatch.setenv("FEATURE_QUOTA_ADMISSION", json.dumps(receipt))
    monkeypatch.setattr(submit, "quota_headroom", lambda _: 1)
    with pytest.raises(RuntimeError):
        allocation_admission(out)


def completed_fixture(root, frozen):
    matrix = np.eye(8, dtype=np.int64) * 10
    matrix[0] = 0
    for arm in ARMS:
        out = root / arm
        atomic_json(out / "STATUS.json", {"status": "complete"})
        atomic_json(out / "config_resolved.json", make_config(frozen, arm, out))
        atomic_json(out / "dataset_inventory.json", {"identical": "inventory"})
        atomic_json(out / "preflight.json", {"initial_model_sha256": "same"})
        atomic_json(
            out / "runtime_identity.json",
            {
                "torch": "2",
                "cuda": "11",
                "minkowski_engine": "0.5",
                "numpy": "1",
                "python": "3",
                "world_size": 2,
                "gpu_names": ["same", "same"],
            },
        )
        (out / "source_manifest.sha256").write_text("same code")
        rows = [
            {
                "epoch": e,
                "accepted_native_counts": [10] * 11,
                "ranks": [
                    {
                        "geometry_label_exposure_sha256": fingerprint(e),
                        "samples": 2,
                        "microbatches": 1,
                        "optimizer_updates_cumulative": e,
                        "amp_skips_cumulative": 0,
                    }
                ],
            }
            for e in range(1, 101)
        ]
        (out / "study_epochs.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        manifest = {"checkpoints": {k: {"model_sha256": k} for k in ("best", "last")}}
        atomic_json(out / "evaluation_manifest.json", manifest)
        for endpoint in ("best", "last"):
            for domain, n in EXPECTED_TILES.items():
                folder = out / "evaluation" / endpoint / domain
                tiles = [
                    {"tile": i, "path": f"{domain}_{i}", "matrix": matrix.tolist()}
                    for i in range(n)
                ]
                atomic_json(
                    folder / "metrics.json", {**summarize(matrix * n), "tiles": tiles}
                )
                atomic_json(
                    folder / "complete.json",
                    {"tiles": n, "metrics_sha256": sha256(folder / "metrics.json")},
                )
    return root


def test_complete_pair_report_bundle_and_tamper_checks(frozen, tmp_path):
    root = completed_fixture(tmp_path, frozen)
    assert (
        compare_study(root)["classification"]
        == "practical_preservation_confirmation_candidate"
    )
    (root / "constant/checkpoints").mkdir()
    (root / "constant/checkpoints/last.pt").write_bytes(b"excluded")
    bundled = collect(root)
    with tarfile.open(bundled["bundle"]) as archive:
        names = archive.getnames()
    assert "comparison.json" in names and "collection_manifest.json" in names
    assert not any("checkpoints" in n for n in names)
    audit = root / "constant/study_epochs.jsonl"
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    rows[2]["ranks"][0].pop("geometry_label_exposure_sha256")
    audit.write_text("\n".join(json.dumps(r) for r in rows))
    assert compare_study(root)["classification"] == "invalid_execution"
    marker = root / "constant/evaluation/best/dales/metrics.json"
    marker.write_text("{}")
    with pytest.raises(ValueError, match="completion hash"):
        compare_study(root)


def test_partial_pair_is_not_a_result(tmp_path):
    assert compare_study(tmp_path)["status"] == "waiting_for_complete_arms"
    for arm in ARMS:
        atomic_json(tmp_path / arm / "STATUS.json", {"status": "needs_resume"})
    with pytest.raises(ValueError):
        collect(tmp_path)


def test_submit_records_two_jobs_and_excludes_cn07(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import scripts.submit_feature_contract_study as submit

    calls = []

    def command(args, **kwargs):
        if args[0] == "git":
            return ""
        if args[0] == "sinfo":
            return "cn02\ncn07\n"
        if args[0] == "squeue":
            return ""
        assert args[0] == "sbatch"
        calls.append((args, kwargs["env"]))
        return str(45000 + len(calls))

    real_path = Path

    def local_path(value):
        return (
            real_path(tmp_path / "logs")
            if str(value).endswith("/e0_logs")
            else real_path(value)
        )

    monkeypatch.setattr(submit, "Path", local_path)
    monkeypatch.setattr(submit.subprocess, "check_output", command)
    monkeypatch.setattr(submit, "quota_headroom", lambda _: 20 * 1024**3)
    monkeypatch.setattr(
        submit.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3)
    )
    monkeypatch.setattr(
        sys, "argv", ["submit", "--study", "test", "--out-root", str(tmp_path)]
    )
    submit.main()
    receipt = json.loads((tmp_path / "test/submission.json").read_text())
    assert len(receipt["submissions"]) == len(calls) == 2
    for (args, env), arm in zip(calls, ARMS):
        assert "--exclude=cn07" in args and "--time=36:00:00" in args
        assert env["FEATURE_STUDY_ARM"] == arm and env["FEATURE_RESUME"] == "0"
        admission = json.loads(env["FEATURE_QUOTA_ADMISSION"])
        assert admission["arms"] == list(ARMS) and not admission["quota_reservation"]


def test_duplicate_epoch_is_invalid_execution(frozen, tmp_path):
    root = completed_fixture(tmp_path, frozen)
    path = root / "constant/study_epochs.jsonl"
    lines = path.read_text().splitlines()
    lines[1] = lines[0]
    path.write_text("\n".join(lines))
    assert compare_study(root)["classification"] == "invalid_execution"


@pytest.mark.parametrize(
    "field", ["eclair_root", "cache_root", "label_space", "review_filters"]
)
def test_inventory_cannot_certify_a_different_training_recipe(
    frozen, tmp_path, monkeypatch, field
):
    from src.feature_contract_study import load_inventory

    archived = copy.deepcopy(frozen)
    if field == "review_filters":
        archived["eclair"]["train_review_categories"] = ["approved"]
    else:
        archived["data"][field] = "changed"
    monkeypatch.setattr(Path, "read_text", lambda self: json.dumps(archived))
    with pytest.raises(ValueError, match="Archived inventory"):
        load_inventory(make_config(frozen, "observed", tmp_path), None)
