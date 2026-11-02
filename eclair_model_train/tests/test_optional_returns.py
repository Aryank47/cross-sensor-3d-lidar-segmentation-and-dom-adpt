"""CPU regression and synthetic end-to-end checks for the approved study."""

from __future__ import annotations

import copy
import io
import json
import random
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_feature_contract_study import load_frozen
from scripts.submit_optional_return_study import export_code
from src.feature_contract_study import make_config, summarize
from src.optional_return_study import (
    advancement,
    collect_results,
    generate_report,
    make_optional_config,
    paired_counts,
    scientific_identity,
    validate_optional_config,
)
from src.optional_returns import (
    CONSTANT,
    mask_batch,
    sample_is_masked,
    validate_update_counters,
)
from src.study_metrics import ECLAIR_TRAIN_TO_COMMON
from src.study_runtime import StudyRuntime, atomic_json, sha256


def batch_arrays():
    coords = np.array(
        [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 2, 0, 0]], np.int32
    )
    labels = np.array([3, 10, -100, 1], np.int64)
    features = np.tile(np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0], np.float32), (4, 1))
    return coords, labels, features, ["first.laz", "second.laz"]


def test_mask_reproducible_without_rng_or_input_mutation():
    values = batch_arrays()
    before = [v.copy() if isinstance(v, np.ndarray) else v[:] for v in values]
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    a, audit = mask_batch(*values, seed=1337, epoch=4, probability=0.5)
    b, _ = mask_batch(*values, seed=1337, epoch=4, probability=0.5)
    assert np.array_equal(a, b) and python_state == random.getstate()
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    for x, y in zip(values[:3], before[:3]):
        assert np.array_equal(x, y)
    assert audit["masked_samples"] + audit["observed_samples"] == 2
    assert (audit["masked_native_counts"] + audit["observed_native_counts"]).sum() == 3
    decisions = [sample_is_masked(1337, 1, str(i), 0.5) for i in range(100)]
    assert any(decisions) and not all(decisions)


def test_mask_extremes_preserve_both_group_semantics():
    values = batch_arrays()
    a, normal = mask_batch(*values, seed=1, epoch=1, probability=0)
    b, masked = mask_batch(*values, seed=1, epoch=1, probability=1)
    assert np.array_equal(a, values[2])
    assert np.array_equal(b, np.broadcast_to(CONSTANT, b.shape))
    assert normal["observed_samples"] == masked["masked_samples"] == 2


@pytest.mark.parametrize("p", [-1, 1.1, float("nan"), float("inf")])
def test_bad_probability_fails(p):
    with pytest.raises(ValueError):
        sample_is_masked(1, 1, "file", p)


def test_collated_identity_and_feature_shape_fail_closed():
    c, y, f, n = batch_arrays()
    with pytest.raises(ValueError):
        mask_batch(c, y, f, n[:1], seed=1, epoch=1)
    with pytest.raises(ValueError):
        mask_batch(c, y, f[:, :9], n, seed=1, epoch=1)


def test_constant_substitution_commutes_with_mean_pooling():
    raw = np.eye(5, dtype=np.float32)[[0, 1, 4, 3]]
    raw = np.concatenate([raw, raw], axis=1)
    groups = np.array([0, 0, 1, 1])
    mean = np.stack([raw[groups == i].mean(0) for i in (0, 1)])
    before = np.stack([np.tile(CONSTANT, (2, 1)).mean(0) for _ in (0, 1)])
    coords = np.array([[0, 0, 0, 0], [0, 1, 0, 0]], np.int32)
    after, _ = mask_batch(
        coords, np.array([1, 2]), mean, ["tile"], seed=1, epoch=1, probability=1
    )
    assert np.array_equal(before, after)


@pytest.mark.parametrize("values", [[[100, 13], [100, 13]], [[13300, 0]]])
def test_valid_numerical_counters(values):
    validate_update_counters(values)


@pytest.mark.parametrize(
    "values", [[[100, 14], [100, 14]], [[100, 3], [99, 4]], [[-1, 0]], [[1, 2, 3]]]
)
def test_invalid_numerical_counters(values):
    with pytest.raises(RuntimeError):
        validate_update_counters(values)


class ArrayTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value

    def cpu(self):
        return self

    def detach(self):
        return self

    def is_pinned(self):
        return False

    def all(self):
        return ArrayTensor(self.value.all())

    def to(self, **kwargs):
        return ArrayTensor(self.value.astype(kwargs.get("dtype", self.value.dtype)))

    def item(self):
        return self.value.item()

    def tolist(self):
        return self.value.tolist()


FAKE_TORCH = SimpleNamespace(
    from_numpy=ArrayTensor,
    int32=np.int32,
    int64=np.int64,
    isfinite=lambda x: ArrayTensor(np.isfinite(x.value)),
    tensor=lambda x, **kwargs: ArrayTensor(x),
)


def runtime(tmp_path):
    cfg = make_optional_config(load_frozen(ROOT), tmp_path)
    value = StudyRuntime.__new__(StudyRuntime)
    value.cfg = cfg
    value.options = cfg["study"]
    value.reference_epochs = None
    value.torch = FAKE_TORCH
    value.dist = SimpleNamespace(enabled=False, rank=0, world_size=1)
    value.device = None
    value.out = tmp_path
    value.native_counts = np.zeros(11, np.int64)
    value.train_to_common = ECLAIR_TRAIN_TO_COMMON
    value.best_utility = None
    value.epoch_times = []
    value.updates = value.skips = 0
    value.begin_epoch(
        1, SimpleNamespace(generator=SimpleNamespace(manual_seed=lambda _: None))
    )
    return value


def test_real_training_hook_partitions_exposure_and_preserves_data(tmp_path):
    value = runtime(tmp_path)
    c, y, f, n = batch_arrays()
    batch = {
        "coords": ArrayTensor(c),
        "labels": ArrayTensor(y),
        "feats": ArrayTensor(f),
        "fnames": n,
    }
    value.prepare_batch(batch)
    value.observe_batch(batch)
    assert np.array_equal(batch["coords"].numpy(), c) and np.array_equal(
        batch["labels"].numpy(), y
    )
    expected, _ = mask_batch(c, y, f, n, seed=1337, epoch=1)
    assert np.array_equal(batch["feats"].numpy(), expected)
    value.end_epoch(
        {"miou": 1.0, "confusion_matrix": np.eye(11, dtype=np.int64)},
        1.0,
        SimpleNamespace(parameters=list),
    )
    row = json.loads((tmp_path / "study_epochs.jsonl").read_text())
    r = row["ranks"][0]
    exposure = r["return_exposure"]
    assert np.array_equal(
        np.array(exposure["masked_native_counts"]) + exposure["observed_native_counts"],
        r["native_counts"],
    )
    with pytest.raises(RuntimeError):
        value.prepare_batch(batch)
    value.reference_epochs = {1: {"ranks": [{**r, "samples": 999}]}}
    with pytest.raises(RuntimeError, match="exposure differs"):
        value.end_epoch(
            {"miou": 1.0, "confusion_matrix": np.eye(11, dtype=np.int64)},
            1.0,
            SimpleNamespace(parameters=list),
        )


def test_nonoptional_training_hook_no_op(tmp_path):
    value = runtime(tmp_path)
    value.options = {"kind": "feature_contract"}
    batch = {"unchanged": object()}
    before = dict(batch)
    value.prepare_batch(batch)
    value.check_objective(None)
    assert batch == before


def test_nonfinite_objective_aborts_and_records(tmp_path):
    value = runtime(tmp_path)
    value.check_objective(ArrayTensor(1.0))
    with pytest.raises(RuntimeError, match="Nonfinite"):
        value.check_objective(ArrayTensor(float("nan")))
    assert (
        json.loads((tmp_path / "numerical_failure.json").read_text())[
            "automatic_restart"
        ]
        is False
    )


def test_fourteenth_skip_stops_actual_runtime(tmp_path):
    value = runtime(tmp_path)
    value.skips = 13

    class Scaler:
        scale = 64

        def get_scale(self):
            return self.scale

        def step(self, _):
            pass

        def update(self):
            self.scale = 32

    with pytest.raises(RuntimeError, match="limit"):
        value.optimizer_step(Scaler(), None)
    assert json.loads((tmp_path / "numerical_failure.json").read_text())[
        "rank_counter_extrema"
    ] == [[0, 14]]


def test_configuration_and_source_identity_are_fixed(tmp_path):
    frozen = load_frozen(ROOT)
    cfg = make_optional_config(frozen, tmp_path)
    validate_optional_config(cfg, frozen)
    assert scientific_identity(cfg) == scientific_identity(
        make_config(frozen, "observed", tmp_path / "other")
    )
    for parent, key, new in [
        ("study", "return_mask_probability", 0.2),
        ("loss", "alpha", [1] * 11),
        ("epochs", None, 99),
    ]:
        other = copy.deepcopy(cfg)
        if key is None:
            other[parent] = new
        else:
            other[parent][key] = new
        with pytest.raises(ValueError):
            validate_optional_config(other, frozen)
    assert cfg["data"]["features"]["return_input_mode"] == "observed"


def test_paired_counts_preserve_native_ignored_and_subtype_predictions():
    counts = paired_counts([0, 8, 9, 9], [0, 7, 8, 0], [3, 0, 8, 7])
    assert sum(r[3] for r in counts) == 4
    assert [9, 0, 7, 1] in counts
    with pytest.raises(ValueError):
        paired_counts([1], [12], [0])


@pytest.mark.parametrize(
    "source,target,last_source,last_target,valid,ci_low,label",
    [
        (
            [0] * 9,
            [2] * 9,
            [0] * 9,
            [2] * 9,
            True,
            0.1,
            "clear_development_pass_reverse_pair_eligible",
        ),
        (
            [0] * 9,
            [2] * 9,
            [0] * 9,
            [2] * 9,
            False,
            0.1,
            "invalid_development_comparison",
        ),
        (
            [-3] + [0] * 8,
            [2] * 9,
            [0] * 9,
            [2] * 9,
            True,
            0.1,
            "source_preservation_failed_stop_recipe",
        ),
        (
            [0] * 9,
            [0.5] * 9,
            [0] * 9,
            [2] * 9,
            True,
            0.1,
            "target_worthwhile_gain_screen_failed_stop_recipe",
        ),
        (
            [0] * 9,
            [2] * 9,
            [0] * 9,
            [2] * 9,
            True,
            -0.1,
            "uncertain_gain_no_automatic_expansion",
        ),
        (
            [0] * 9,
            [2] * 9,
            [-3] + [0] * 8,
            [2] * 9,
            True,
            0.1,
            "final_weight_instability_no_automatic_expansion",
        ),
        (
            [0] * 9,
            [2] * 9,
            [0] * 9,
            [0] * 9,
            True,
            0.1,
            "final_weight_instability_no_automatic_expansion",
        ),
    ],
)
def test_predeclared_decision_branches(
    source, target, last_source, last_target, valid, ci_low, label
):
    ci = {name: {"low_pp": ci_low} for name in ("all_seven_miou", "utility_miou")}
    assert advancement(source, target, ci, last_source, last_target, valid) == label


def make_complete_study(root):
    frozen = load_frozen(ROOT)
    reference = root / "reference"
    candidate = root / "candidate"
    for arm in ("observed", "constant", "candidate"):
        out = candidate if arm == "candidate" else reference / arm
        cfg = (
            make_optional_config(frozen, out)
            if arm == "candidate"
            else make_config(frozen, arm, out)
        )
        atomic_json(out / "config_resolved.json", cfg)
        atomic_json(out / "STATUS.json", {"status": "complete"})
        atomic_json(out / "preflight.json", {"initial_model_sha256": "same"})
        atomic_json(out / "dataset_inventory.json", {"same": "inventory"})
        atomic_json(
            out / "runtime_identity.json",
            {
                "torch": "2",
                "cuda": "11",
                "minkowski_engine": ".5",
                "numpy": "1",
                "python": "3",
                "world_size": 2,
                "gpu_names": ["same"],
            },
        )
        (out / "source_manifest.sha256").write_text("same source")
        audits = []
        for epoch in range(1, 101):
            rank = {
                "native_counts": [10] * 11,
                "presence": [1] * 11,
                "samples": 530,
                "microbatches": 265,
                "metadata": {},
                "geometry_label_exposure_sha256": str(epoch),
                "coordinate_reordered_batches": 0,
                "optimizer_updates_cumulative": 133 * epoch,
                "amp_skips_cumulative": 0,
            }
            ranks = [copy.deepcopy(rank), copy.deepcopy(rank)]
            if arm == "candidate":
                for r in ranks:
                    r["return_exposure"] = {
                        "masked_samples": 265,
                        "observed_samples": 265,
                        "masked_native_counts": [5] * 11,
                        "observed_native_counts": [5] * 11,
                    }
            audits.append(
                {"epoch": epoch, "accepted_native_counts": [20] * 11, "ranks": ranks}
            )
        (out / "study_epochs.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in audits)
        )
        manifest = {
            "checkpoints": {ep: {"model_sha256": ep} for ep in ("best", "last")}
        }
        atomic_json(out / "evaluation_manifest.json", manifest)
        for ep in ("best", "last"):
            for domain, n in (("eclair", 125), ("dales", 11)):
                gtmap = yaml.safe_load(
                    (ROOT / f"configs/mapping_{domain}_to_common.yaml").read_text()
                )
                gt = {
                    common: next(k for k, v in gtmap.items() if v == common)
                    for common in range(1, 8)
                }
                pred = {
                    common: ECLAIR_TRAIN_TO_COMMON.index(common)
                    for common in range(1, 8)
                }
                amount = 90 if arm == "candidate" else 80
                native = np.zeros((256, 11), np.int64)
                matrix = np.zeros((8, 8), np.int64)
                joint = []
                for common in range(1, 8):
                    native[gt[common], pred[common]] = amount
                    native[gt[common], 0] = 100 - amount
                    matrix[common, common] = amount
                    matrix[common, 0] = 100 - amount
                    joint += [
                        [gt[common], pred[common], pred[common], amount],
                        [gt[common], 0, 0, 100 - amount],
                    ]
                tiles = [
                    {"tile": i, "path": f"{domain}_{i}", "matrix": matrix.tolist()}
                    for i in range(n)
                ]
                folder = out / "evaluation" / ep / domain
                atomic_json(
                    folder / "metrics.json",
                    {
                        **summarize(matrix * n),
                        "tiles": tiles,
                        "native_gt_by_source_prediction": (native * n).tolist(),
                    },
                )
                atomic_json(
                    folder / "complete.json",
                    {"tiles": n, "metrics_sha256": sha256(folder / "metrics.json")},
                )
                if arm == "candidate" and ep == "best":
                    for tile in tiles:
                        atomic_json(
                            folder / f"tile_{tile['tile']:04d}.json",
                            {
                                "paired_diagnostic": {
                                    "native_gt_observed_constant": joint
                                }
                            },
                        )
    files = {
        str(p.relative_to(reference)): sha256(p)
        for p in reference.rglob("*")
        if p.is_file()
    }
    atomic_json(
        root / "reference_manifest.json", {"files": files, "historical_benchmarks": []}
    )
    return root


def test_complete_report_diagnostic_and_bounded_bundle(tmp_path):
    root = make_complete_study(tmp_path)
    result = generate_report(root)
    assert result["classification"] == "clear_development_pass_reverse_pair_eligible"
    assert not result["automatic_reverse_submission"]
    paired = result["endpoints"]["best"]["dales"]["constant_input_diagnostic"]["scores"]
    assert paired["class_correctness_transitions"]["vehicle"]["both_correct"] == 990
    (root / "candidate/checkpoints").mkdir()
    (root / "candidate/checkpoints/last.pt").write_bytes(b"never package")
    (root / "code").mkdir()
    (root / "code/file.py").write_text("never package")
    bundle = collect_results(root)
    with tarfile.open(bundle["bundle"]) as archive:
        names = archive.getnames()
    assert "report.json" in names and not any(
        "checkpoints" in n or n.startswith("code/") for n in names
    )
    rows = [
        json.loads(s)
        for s in (root / "candidate/study_epochs.jsonl").read_text().splitlines()
    ]
    rows[1]["ranks"][0]["geometry_label_exposure_sha256"] = "changed"
    (root / "candidate/study_epochs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    assert generate_report(root)["classification"] == "invalid_development_comparison"


@pytest.mark.parametrize("unsafe", [False, True])
def test_immutable_export_checks_archive_members(tmp_path, monkeypatch, unsafe):
    import scripts.submit_optional_return_study as submit

    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        payload = b"print('code')\n"
        info = tarfile.TarInfo("../escape.py" if unsafe else "src/file.py")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    monkeypatch.setattr(
        submit.subprocess,
        "check_output",
        lambda args, **kwargs: (
            "eclair_model_train/" if args[1] == "rev-parse" else stream.getvalue()
        ),
    )
    if unsafe:
        with pytest.raises(ValueError):
            export_code(tmp_path, tmp_path / "export", "commit")
        assert not (tmp_path / "export").exists()
    else:
        result = export_code(tmp_path, tmp_path / "export", "commit")
        assert result["files"]["src/file.py"] == sha256(tmp_path / "export/src/file.py")


def test_nonfinite_other_rank_aborts_before_backward(tmp_path):
    value = runtime(tmp_path)
    value.dist.enabled = True

    def reduce_flag(flag, **kwargs):
        flag.value[...] = 0  # Other rank reports a nonfinite objective.

    value.torch = SimpleNamespace(
        **vars(FAKE_TORCH),
        distributed=SimpleNamespace(
            all_reduce=reduce_flag, ReduceOp=SimpleNamespace(MIN="minimum")
        ),
    )
    with pytest.raises(RuntimeError, match="at least one rank"):
        value.check_objective(ArrayTensor(1.0))
    assert (tmp_path / "numerical_failure.json").is_file()


@pytest.mark.parametrize("age_hours,expected", [(1, "fallback"), (49, "expired")])
def test_quota_fallback_requires_fresh_matching_admission(
    tmp_path, monkeypatch, age_hours, expected
):
    import scripts.submit_optional_return_study as submit

    now = 200000.0
    monkeypatch.setattr(submit.time, "time", lambda: now)
    admission = submit.receipt(tmp_path, "commit", 11 * 1024**3)
    admission["checked_at_unix"] = now - age_hours * 3600
    atomic_json(tmp_path / "quota_admission.json", admission)
    atomic_json(tmp_path / "code_provenance.json", {"commit": "commit"})

    def unavailable(_):
        raise RuntimeError("Compute mount cannot be matched to login quota")

    monkeypatch.setattr(submit, "quota_headroom", unavailable)
    if expected == "expired":
        with pytest.raises(RuntimeError, match="expired"):
            submit.allocation_admission(tmp_path / "candidate")
    else:
        result = submit.allocation_admission(tmp_path / "candidate")
        assert result["mode"] == "validated_login_snapshot_compute_query_unavailable"
        assert not result["quota_reservation"]
        admission["commit"] = "different"
        atomic_json(tmp_path / "quota_admission.json", admission)
        with pytest.raises(RuntimeError, match="identity/capacity"):
            submit.allocation_admission(tmp_path / "candidate")
