"""Frozen ECLAIR input ablation: contracts, canonical evaluation and paired report."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .features import FeatureConfig
from .study_metrics import COMMON_NAMES, ECLAIR_TRAIN_TO_COMMON, confusion, summary
from .study_runtime import atomic_json, sha256, storage_guard

ARMS = ("observed", "constant")
MODES = {"observed": "observed", "constant": "constant_pair"}
EXPECTED_TILES = {"eclair": 125, "dales": 11}
STORAGE = {
    "run_budget_bytes": 2 * 1024**3,
    "diagnostics_budget_bytes": 128 * 1024**2,
    "reserve_bytes": 512 * 1024**2,
}


def fingerprint(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def make_config(frozen, arm, out_dir):
    if arm not in ARMS:
        raise ValueError(f"Unknown feature-study arm {arm!r}")
    cfg = copy.deepcopy(frozen)
    cfg["data"]["features"]["return_input_mode"] = MODES[arm]
    cfg["data"].update(run_cache_enabled=False, require_cache=True)
    cfg["run"].update(
        name=f"eclair_returns_{arm}_s1337",
        out_dir=str(Path(out_dir).resolve()),
        keep_epoch_snapshots=False,
        keep_last_k_epoch_snapshots=0,
        best_checkpoint_model_only=True,
        epoch_snapshots_model_only=True,
        save_every_epochs=1,
        max_run_dir_gb=2.0,
    )
    cfg["method"] = {"name": "baseline"}
    cfg["ocons"] = {"enabled": False}
    cfg["model"]["layers"] = [1] * 8
    cfg["study"] = {
        "enabled": True,
        "version": 1,
        "kind": "feature_contract",
        "arm": arm,
        "gradient_batches_per_epoch": 0,
        "storage": dict(STORAGE),
    }
    return cfg


def validate_config(cfg, frozen):
    arm = cfg.get("study", {}).get("arm")
    expected = make_config(frozen, arm, cfg["run"]["out_dir"])
    actual = copy.deepcopy(cfg)
    resume = actual["run"].pop("resume_from", None)
    if (
        resume is not None
        and Path(resume).resolve()
        != Path(cfg["run"]["out_dir"]).resolve() / "checkpoints/last.pt"
    ):
        raise ValueError("Resume must use this arm's own last checkpoint")
    if actual != expected:
        raise ValueError(
            "Feature study differs from its frozen recipe/allowed execution overrides"
        )
    fc = FeatureConfig(**cfg["data"]["features"])
    if (
        cfg["data"]["dataset"] != "eclair"
        or cfg["epochs"] != 100
        or cfg["model"]["in_channels"] != 10
        or cfg["model"]["out_channels"] != 11
        or cfg["loss"]["alpha"] is not None
        or cfg["run"]["seed"] != 1337
        or fc.return_input_mode != MODES[arm]
    ):
        raise ValueError("Invalid source/feature/label/seed/loss contract")
    patch = cfg["data"]["patch"]
    if not np.isclose(patch["voxel_size"] * patch["coord_norm_factor"], 0.5):
        raise ValueError("ECLAIR source physical grid must remain 0.5m")
    return arm


def pair_identity(cfg):
    value = copy.deepcopy(cfg)
    value["data"]["features"].pop("return_input_mode", None)
    value["run"].pop("out_dir", None)
    value["run"].pop("name", None)
    value["run"].pop("resume_from", None)
    value["study"].pop("arm", None)
    return fingerprint(value)


def model_digest(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def summarize(cm):
    result = summary(cm)
    matrix = np.asarray(cm, dtype=np.int64)
    support, predicted = matrix.sum(1), matrix.sum(0)
    f1 = np.divide(
        2 * matrix.diagonal(),
        support + predicted,
        out=np.zeros(8, dtype=float),
        where=(support + predicted) > 0,
    )
    result["per_class_f1"] = dict(zip(COMMON_NAMES[1:], f1[1:].tolist()))
    result["macro_f1"] = float(f1[1:].mean())
    return result


def load_inventory(cfg, logger):
    from .study_analysis import load_dataset

    archived = json.loads(
        Path(
            "/scratch/m23csa510/e0_results/e0_eclair_repro_29306/config_resolved.json"
        ).read_text()
    )
    for key in ("eclair_root", "cache_root", "label_space"):
        if archived["data"][key] != cfg["data"][key]:
            raise ValueError(
                f"Archived inventory and actual frozen training data differ: {key}"
            )
    if archived.get("eclair", {}) != cfg.get("eclair", {}):
        raise ValueError(
            "Archived inventory and actual frozen training review filters differ"
        )
    inventories = []
    for domain, split in (
        ("eclair", "train"),
        ("eclair", "val"),
        ("eclair", "test"),
        ("dales", "test"),
    ):
        _, _, identity = load_dataset(domain, split, cfg, logger)
        inventories.append(identity)
    sets = [{m["path"] for m in inv["members"]} for inv in inventories[:3]]
    if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Source train, validation and test file identities overlap")
    for inv in inventories:
        if (
            inv["split"] == "test"
            and len(inv["members"]) != EXPECTED_TILES[inv["domain"]]
        ):
            raise ValueError("Canonical test inventory changed; review before running")
    return {"datasets": inventories}


def evaluate_tiles(model, cfg, checkpoint_id, domain, out, deadline, logger):
    import torch

    from . import eval_cross_common as ev
    from .study_analysis import load_dataset

    ds, gt_lut, identity = load_dataset(domain, "test", cfg, logger)
    if len(ds) != EXPECTED_TILES[domain]:
        raise ValueError("Incomplete canonical test inventory")
    pred_lut = np.asarray(ECLAIR_TRAIN_TO_COMMON, dtype=np.int64)
    ev.validate_source_train_to_common_mapping(
        source_cfg=cfg,
        source_domain="eclair",
        source_train_to_common=pred_lut,
        source_native_to_common_lut=ev.mapping_to_lut(
            ev.load_yaml_int_map("configs/mapping_eclair_to_common.yaml"),
            size=256,
            default=0,
            name="source_native",
        ),
        logger=logger,
    )
    contract = {
        "checkpoint": checkpoint_id,
        "dataset": identity,
        "source_domain": "eclair",
        "features": cfg["data"]["features"],
        "patch": cfg["data"]["patch"],
        "voxelization": cfg["data"].get("voxelization"),
        "window_m": 120,
        "stride_m": 120,
        "aggregation": "mean_logits",
        "amp": False,
        "source_mapping_sha256": sha256("configs/train_id_to_common_id.yaml"),
        "target_mapping_sha256": sha256(f"configs/mapping_{domain}_to_common.yaml"),
    }
    signature = fingerprint(contract)
    stage = Path(out) / domain
    stage.mkdir(parents=True, exist_ok=True)
    common_total = np.zeros((8, 8), dtype=np.int64)
    native_total = np.zeros((256, 11), dtype=np.int64)
    subtype_total = np.zeros((256, 8), dtype=np.int64)
    tiles = []
    for index in range(len(ds)):
        path = stage / f"tile_{index:04d}.json"
        result = json.loads(path.read_text()) if path.is_file() else None
        if result is None or result.get("signature") != signature:
            if time.time() > deadline - 900:
                return False
            raw = ds.get_raw(index)
            tick = time.monotonic()
            logits = ev.infer_tile_point_logits(
                model=model,
                raw=raw,
                source_cfg=cfg,
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
                    raise RuntimeError("Nonfinite canonical point logits")
            native = ev.get_native_labels(raw, target_domain=domain).astype(np.int64)
            if np.any((native < 0) | (native > 255)) or logits.shape != (
                len(native),
                11,
            ):
                raise ValueError("Invalid native labels or ECLAIR output shape")
            prediction = logits.argmax(1)
            common_pred = pred_lut[prediction]
            result = {
                "signature": signature,
                "tile": index,
                "path": identity["members"][index]["path"],
                "matrix": confusion(gt_lut[native], common_pred).tolist(),
                "native_gt_by_source_prediction": np.bincount(
                    native * 11 + prediction, minlength=256 * 11
                )
                .reshape(256, 11)
                .tolist(),
                "native_gt_by_common_prediction": np.bincount(
                    native * 8 + common_pred, minlength=256 * 8
                )
                .reshape(256, 8)
                .tolist(),
                "seconds": time.monotonic() - tick,
            }
            storage_guard(stage, 128 * 1024, cfg["study"]["storage"])
            atomic_json(path, result)
            del logits, raw
        common_total += np.asarray(result["matrix"], np.int64)
        native_total += np.asarray(result["native_gt_by_source_prediction"], np.int64)
        subtype_total += np.asarray(result["native_gt_by_common_prediction"], np.int64)
        tiles.append(
            {"tile": index, "path": result["path"], "matrix": result["matrix"]}
        )
        logger.info(
            "canonical %s %s: %d/%d tiles",
            cfg["study"]["arm"],
            domain,
            index + 1,
            len(ds),
        )
    payload = {
        **summarize(common_total),
        "tiles": tiles,
        "contract": contract,
        "native_gt_by_source_prediction": native_total.tolist(),
        "native_gt_by_common_prediction": subtype_total.tolist(),
        "subtype_recall": {
            str(i): float(subtype_total[i, gt_lut[i]] / subtype_total[i].sum())
            for i in range(256)
            if subtype_total[i].sum() and gt_lut[i] in (4, 5)
        },
    }
    if domain == "eclair":
        native_cm = native_total[1:12]
        union = native_cm.sum(0) + native_cm.sum(1) - native_cm.diagonal()
        native_iou = np.divide(
            native_cm.diagonal(), union, out=np.zeros(11), where=union > 0
        )
        payload["native_source_test"] = {
            "matrix": native_cm.tolist(),
            "per_class_iou": native_iou.tolist(),
            "miou_all_eleven": float(native_iou.mean()),
            "undefined_union_ids": np.flatnonzero(union == 0).tolist(),
        }
    atomic_json(stage / "metrics.json", payload)
    atomic_json(
        stage / "complete.json",
        {
            "signature": signature,
            "tiles": len(ds),
            "metrics_sha256": sha256(stage / "metrics.json"),
        },
    )
    return True


def metric_vector(matrices):
    matrices = np.asarray(matrices, dtype=np.int64)
    tp = np.diagonal(matrices, axis1=-2, axis2=-1)
    union = matrices.sum(-1) + matrices.sum(-2) - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp, dtype=float), where=union > 0)[
        ..., 1:
    ]
    return np.concatenate(
        (iou, iou.mean(-1, keepdims=True), iou[..., [3, 4]].mean(-1, keepdims=True)),
        axis=-1,
    )


def bootstrap(reference, candidate, samples=4000):
    a, b = np.asarray(reference, np.int64), np.asarray(candidate, np.int64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (8, 8):
        raise ValueError("Paired tile matrices must have equal N x 8 x 8 shapes")
    rng = np.random.default_rng(2337)
    changes = []
    for start in range(0, samples, 64):
        ids = rng.integers(0, len(a), size=(min(64, samples - start), len(a)))
        changes.append(
            (metric_vector(b[ids].sum(1)) - metric_vector(a[ids].sum(1))) * 100
        )
    interval = np.quantile(np.concatenate(changes), [0.025, 0.975], axis=0)
    names = (*COMMON_NAMES[1:], "all_seven_miou", "utility_miou")
    return {
        name: {"low_pp": float(interval[0, i]), "high_pp": float(interval[1, i])}
        for i, name in enumerate(names)
    }


def outcome(source_delta, target_delta, intervals, execution_valid=True):
    if not execution_valid:
        return "invalid_execution"
    if min(source_delta[:7]) < -2 or min(source_delta[7:]) < -1:
        return "source_competence_failed"
    if min(target_delta[:7]) < -2 or min(target_delta[7:]) < -1:
        return "observed_material_target_regression"
    if min(target_delta[:7]) >= 0 and min(target_delta[7:]) > 0:
        return "strict_observed_positive_confirmation_candidate"
    names = (*COMMON_NAMES[1:], "all_seven_miou", "utility_miou")
    if all(intervals[n]["low_pp"] > (-2 if i < 7 else -1) for i, n in enumerate(names)):
        return "practical_preservation_confirmation_candidate"
    return "unresolved_practical_comparison"


def compare_study(study_dir):
    root = Path(study_dir)
    for arm in ARMS:
        status = root / arm / "STATUS.json"
        if (
            not status.is_file()
            or json.loads(status.read_text()).get("status") != "complete"
        ):
            return {"status": "waiting_for_complete_arms", "waiting_for": arm}
    cfgs = [json.loads((root / a / "config_resolved.json").read_text()) for a in ARMS]
    if pair_identity(cfgs[0]) != pair_identity(cfgs[1]):
        raise ValueError(
            "The two arms have scientific configuration differences beyond the input mode"
        )
    inventories = [
        json.loads((root / a / "dataset_inventory.json").read_text()) for a in ARMS
    ]
    if inventories[0] != inventories[1]:
        raise ValueError("Dataset/cache identities differ between arms")
    preflights = [json.loads((root / a / "preflight.json").read_text()) for a in ARMS]
    if preflights[0]["initial_model_sha256"] != preflights[1]["initial_model_sha256"]:
        raise ValueError("Model initializations were not matched")
    audits = [
        [
            json.loads(line)
            for line in (root / a / "study_epochs.jsonl").read_text().splitlines()
            if line
        ]
        for a in ARMS
    ]
    expected_epochs = list(range(1, 101))
    execution_valid = all(
        [r["epoch"] for r in rows] == expected_epochs for rows in audits
    )
    mismatches = []
    if not execution_valid:
        mismatches.append({"kind": "missing_duplicate_or_out_of_order_epochs"})
    runtime = [
        json.loads((root / a / "runtime_identity.json").read_text()) for a in ARMS
    ]
    for key in ("torch", "cuda", "minkowski_engine", "numpy", "python", "world_size"):
        if runtime[0][key] != runtime[1][key]:
            mismatches.append({"kind": "runtime_mismatch", "field": key})
    code = [(root / a / "source_manifest.sha256").read_text() for a in ARMS]
    if code[0] != code[1]:
        mismatches.append({"kind": "scientific_code_mismatch"})
    for left, right in zip(*audits):
        keys = ("epoch", "accepted_native_counts")
        if any(left[k] != right[k] for k in keys) or len(left["ranks"]) != len(
            right["ranks"]
        ):
            mismatches.append(
                {"epoch": left["epoch"], "kind": "exposure_counts_or_world_size"}
            )
            continue
        for rank, (lr, rr) in enumerate(zip(left["ranks"], right["ranks"])):
            for key in (
                "geometry_label_exposure_sha256",
                "samples",
                "microbatches",
                "optimizer_updates_cumulative",
                "amp_skips_cumulative",
            ):
                if key not in lr or key not in rr or lr[key] != rr[key]:
                    mismatches.append(
                        {"epoch": left["epoch"], "rank": rank, "field": key}
                    )
    execution_valid = execution_valid and not mismatches
    manifests = [
        json.loads((root / a / "evaluation_manifest.json").read_text()) for a in ARMS
    ]
    endpoints = {}
    for endpoint in ("best", "last"):
        domains = {}
        for domain in ("eclair", "dales"):
            results = []
            for arm, manifest in zip(ARMS, manifests):
                folder = (
                    root
                    / arm
                    / "evaluation"
                    / manifest["checkpoints"][endpoint]["model_sha256"]
                    / domain
                )
                marker = json.loads((folder / "complete.json").read_text())
                if marker["metrics_sha256"] != sha256(folder / "metrics.json"):
                    raise ValueError("Canonical evaluation completion hash mismatch")
                result = json.loads((folder / "metrics.json").read_text())
                if len(result["tiles"]) != EXPECTED_TILES[domain]:
                    raise ValueError("Canonical tile coverage is incomplete")
                results.append(result)
            a, b = results
            if [t["path"] for t in a["tiles"]] != [t["path"] for t in b["tiles"]]:
                raise ValueError("Paired tile identities differ")
            delta = (
                metric_vector(b["confusion_matrix"])
                - metric_vector(a["confusion_matrix"])
            ) * 100
            domains[domain] = {
                "observed": a,
                "constant": b,
                "delta_pp": delta.tolist(),
                "paired_tile_intervals": bootstrap(
                    [t["matrix"] for t in a["tiles"]], [t["matrix"] for t in b["tiles"]]
                ),
            }
        endpoints[endpoint] = {
            "domains": domains,
            "classification": outcome(
                domains["eclair"]["delta_pp"],
                domains["dales"]["delta_pp"],
                domains["dales"]["paired_tile_intervals"],
                execution_valid,
            ),
        }
    result = {
        "status": "complete",
        "primary_endpoint": "best",
        "endpoints": endpoints,
        "execution_valid": execution_valid,
        "exposure_mismatches": mismatches,
        "hardware_warning": runtime[0]["gpu_names"] != runtime[1]["gpu_names"],
        "limitations": [
            "one paired training seed",
            "one source direction",
            "development datasets",
            "marginal tile intervals do not measure seed repeatability or joint confidence",
        ],
        "third_training": "not_submitted",
    }
    atomic_json(root / "comparison.json", result)
    lines = [
        "# ECLAIR return-input comparison",
        "",
        f"Primary outcome: **{endpoints['best']['classification']}**",
        "",
        f"Matched execution: {execution_valid}. Third training: not submitted.",
        "",
        "| Domain/class | Observed IoU | Constant IoU | Change, pp |",
        "|---|---:|---:|---:|",
    ]
    for domain, values in endpoints["best"]["domains"].items():
        for i, name in enumerate(COMMON_NAMES[1:]):
            lines.append(
                f"| {domain}/{name} | {100 * values['observed']['per_class_iou'][name]:.2f} | "
                f"{100 * values['constant']['per_class_iou'][name]:.2f} | {values['delta_pp'][i]:+.2f} |"
            )
        for i, name in ((7, "all_seven_miou"), (8, "utility_miou")):
            lines.append(
                f"| {domain}/{name} | {100 * values['observed'][name]:.2f} | "
                f"{100 * values['constant'][name]:.2f} | {values['delta_pp'][i]:+.2f} |"
            )
    lines += [
        "",
        "Best was selected on source-native validation only; final weights are secondary.",
        "This is an observed feature-contract screen, not a unique root-cause finding or a replicated winner.",
    ]
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    return {"status": "complete", "classification": endpoints["best"]["classification"]}
