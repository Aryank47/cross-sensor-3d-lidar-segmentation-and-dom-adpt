"""Frozen source-only crop coverage intervention and auditable exposure counts."""

import copy
import json
from pathlib import Path

import numpy as np

from .study_runtime import sha256

ARMS = ("coverage_control", "coverage_candidate")
CLASSES = (3, 4, 5, 6, 7)
BINS = 4
CONTENTS_POLICY = "contents_v1"
CONTENTS_LEAST = {"3": 2, "4": 0, "5": 0, "6": 0, "7": 0}
QUALIFIED_REPLAY_SHA = (
    "bb7cf1612b779b93edc9ff8b755aaa73bbb8fdc9f8020fdef1343ca0d3c25881"
)
QUALIFIED_CATALOG_SHA = (
    "c75ac34ff7d62c24fd34fa1e60bae8dd60a4238e82d5879f43412c1c85f5ce37"
)


def qualify_contents(path, catalog_digest):
    """Reuse the exact completed 47041 source replay and its catalogue."""
    if sha256(path) != QUALIFIED_REPLAY_SHA or catalog_digest != QUALIFIED_CATALOG_SHA:
        raise ValueError(
            "Contents training requires the qualified replay and catalogue bytes"
        )
    report = json.loads(Path(path).read_text())
    if report["status"] != "passed" or not all(report["checks"].values()):
        raise ValueError("Contents replay did not clear admission")
    return {
        "status": "admitted",
        "policy": CONTENTS_POLICY,
        "catalog_sha256": catalog_digest,
        "qualification_sha256": QUALIFIED_REPLAY_SHA,
        "least_represented_bins": CONTENTS_LEAST,
        "qualification_delta": report["delta"],
        "qualification_replacement_rate": report["accepted_replacement_rate"],
        "qualification_workload_ratio": report["voxel_workload_ratio"],
    }


def verified_anchors(root, manifest, domain, split, index):
    path = Path(root) / f"geometry/{domain}_{split}_{index:04d}.npz"
    marker = json.loads(path.with_suffix(".json").read_text())
    expected = [{"path": str(path.relative_to(root)), "sha256": sha256(path)}]
    if marker["identity"] != manifest["identity"] or marker["artifacts"] != expected:
        raise ValueError(f"Geometry provenance mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("ids", "xyz", "gt", "native", "geometry")}


def strata_inputs(data):
    geometry = data["geometry"]
    local = geometry[:, 2, 4]  # linearity at 2 m
    broad = geometry[:, 5, 9]  # height span at 16 m
    valid = (
        np.isfinite(local)
        & np.isfinite(broad)
        & (geometry[:, 2, 12] == 0)
        & (geometry[:, 5, 12] == 0)
    )
    return np.stack((local, broad), axis=1), valid


def geometry_bins(data, medians):
    values, valid = strata_inputs(data)
    bins = np.full(len(values), -1, dtype=int)
    for common in CLASSES:
        take = valid & (data["gt"] == common)
        high = values[take] > np.asarray(medians[str(common)])
        bins[take] = 2 * high[:, 0] + high[:, 1]
    return bins


def build_catalog(root):
    root = Path(root).resolve()
    manifest_path = root / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    schema = json.loads((root / "measurement_schema.json").read_text())
    assert schema["radii_m"][2] == 2 and schema["radii_m"][5] == 16
    assert [schema["geometry_variables"][i] for i in (4, 9, 12)] == [
        "linearity",
        "z_extent",
        "edge_truncated",
    ]
    inventory = manifest["datasets"]["dales/train"]
    members = inventory["members"]
    if len(members) != 26:
        raise ValueError("Expected all 26 frozen DALES source training tiles")
    arrays = [
        verified_anchors(root, manifest, "dales", "train", i)
        for i in range(len(members))
    ]
    medians = {}
    for common in CLASSES:
        parts = []
        for data in arrays:
            values, valid = strata_inputs(data)
            parts.append(values[valid & (data["gt"] == common)])
        values = np.concatenate(parts)
        if len(values) == 0:
            raise ValueError(f"No source anchors for class {common}")
        medians[str(common)] = np.median(values, axis=0).tolist()
    tiles = {}
    supports = np.zeros((8, BINS), dtype=int)
    for member, data in zip(members, arrays):
        name = Path(member["path"]).name
        if name in tiles:
            raise ValueError(f"Ambiguous basename: {name}")
        bins = geometry_bins(data, medians)
        take = bins >= 0
        rows = np.column_stack(
            (data["ids"], data["native"], data["gt"], bins, data["xyz"])
        )
        tiles[name] = rows[take].tolist()
        hit = np.zeros_like(supports)
        hit[data["gt"][take], bins[take]] = 1
        supports += hit
    admitted = all(np.sum(supports[c] >= 4) >= 3 for c in CLASSES)
    return {
        "status": "source_catalog_admitted"
        if admitted
        else "insufficient_source_strata",
        "source_study": str(root),
        "source_manifest_sha256": sha256(manifest_path),
        "inventory": inventory,
        "medians": medians,
        "tiles": tiles,
        "distinct_source_tiles_per_class_bin": supports.tolist(),
    }


class TileCoverage:
    def __init__(self, rows, xyz, native):
        self.rows = np.asarray(rows, dtype=np.float64).reshape(-1, 7)
        self.ids = self.rows[:, 0].astype(int)
        if (
            np.any(self.ids < 0)
            or np.any(self.ids >= len(xyz))
            or not np.array_equal(native[self.ids], self.rows[:, 1])
            or not np.allclose(xyz[self.ids], self.rows[:, 4:7], atol=1e-3, rtol=0)
        ):
            raise ValueError("Source catalogue anchors do not match the raw cache")
        self.groups = {}
        # One representative per 60 m cell in each same-label geometry bin.
        for label in np.unique(self.rows[:, 1]):
            groups = []
            for band in range(BINS):
                select = (self.rows[:, 1] == label) & (self.rows[:, 3] == band)
                if np.any(select):
                    cells = np.floor(self.rows[select, 4:6] / 60).astype(int)
                    _, indices = np.unique(cells, axis=0, return_index=True)
                    groups.append(self.ids[select][indices])
            self.groups[int(label)] = groups

    def choose(self, original, native_label, rng):
        groups = self.groups.get(int(native_label), ())
        if len(groups) < 2 or rng.random() >= 0.5:
            return original
        return int(rng.choice(groups[int(rng.integers(len(groups)))]))

    def crop_hits(self, points, retained_ids, centre, size):
        inside = np.all(
            np.abs(self.rows[:, 4:6] - points[centre, :2]) <= size / 2, axis=1
        )
        inside &= np.isin(self.ids, retained_ids)
        hits = np.zeros((8, BINS), dtype=np.int64)
        hits[self.rows[inside, 2].astype(int), self.rows[inside, 3].astype(int)] = 1
        return hits


def select_center(xyz, original, alternatives, rare_xy, rare_class):
    """Replace a center only when the predicted crop adds a priority group."""
    if len(alternatives) == 0 or len(rare_xy) == 0:
        return original
    original_hits = np.any(
        (np.all(np.abs(rare_xy - xyz[original]) <= 60, axis=1))[:, None] & rare_class,
        axis=0,
    )
    keep = alternatives != original
    candidates = alternatives[keep]
    if len(candidates) == 0:
        return original
    inside = np.all(
        np.abs(xyz[candidates, None, :] - rare_xy[None, :, :]) <= 54, axis=2
    )
    hits = (inside.astype(np.int32) @ rare_class.astype(np.int32)) > 0
    gains = hits & ~original_hits
    primary = gains[:, :3].sum(1)
    eligible = np.all(hits | ~original_hits, axis=1) & (primary > 0)
    if not eligible.any():
        return original
    choices = np.flatnonzero(eligible)
    distance = np.sum((xyz[candidates[choices]] - xyz[original]) ** 2, axis=1)
    order = np.lexsort(
        (candidates[choices], distance, -gains[choices].sum(1), -primary[choices])
    )
    return int(candidates[choices[order[0]]])


class ContentsCoverage(TileCoverage):
    """Exact tile-local policy exercised in the fresh source replay."""

    def __init__(self, rows, xyz, native):
        super().__init__(rows, xyz, native)
        self.xy = xyz[:, :2]
        classes = np.arange(3, 8)
        least = np.array([CONTENTS_LEAST[str(c)] for c in classes])
        common = self.rows[:, 2].astype(int)
        take = self.rows[:, 3] == least[common - 3]
        self.rare_xy = self.rows[take, 4:6]
        self.rare_class = common[take, None] == classes[None, :]

    def choose(self, original, native_label, rng):
        alternatives = self.ids[self.rows[:, 1] == native_label]
        return select_center(
            self.xy, original, alternatives, self.rare_xy, self.rare_class
        )


def check_eval_inventory(catalog, identity, domain):
    root = Path(catalog["source_study"])
    path = root / "study_manifest.json"
    if sha256(path) != catalog["source_manifest_sha256"]:
        raise ValueError("Frozen geometry manifest changed")
    manifest = json.loads(path.read_text())
    if manifest["datasets"][f"{domain}/test"] != identity:
        raise ValueError(
            f"Geometry and canonical {domain} evaluation inventories differ"
        )
    return root, manifest


def geometry_eval_rows(data, raw, native, gt, pred, medians):
    ids = data["ids"]
    if (
        np.any(ids < 0)
        or np.any(ids >= len(native))
        or not np.array_equal(native[ids], data["native"])
        or not np.array_equal(gt[ids], data["gt"])
        or not np.allclose(raw["xyz"][ids], data["xyz"], atol=1e-3, rtol=0)
    ):
        raise ValueError("Evaluation anchors do not align with canonical points")
    bins = geometry_bins(data, medians)
    keep = bins >= 0
    counts = np.zeros((8, BINS, 8), dtype=np.int64)
    np.add.at(counts, (gt[ids][keep], bins[keep], pred[ids][keep]), 1)
    return counts.tolist(), {
        "ids": ids.tolist(),
        "native_gt": native[ids].tolist(),
        "gt": gt[ids].tolist(),
        "prediction": pred[ids].tolist(),
    }


def replay_exposure(cfg, out):
    """Compare accepted crops before either arm trains or reads target outcomes."""
    from train import build_dataloaders

    from .dist import DistEnv
    from .study_runtime import atomic_json

    result = {}
    for arm, changed in zip(ARMS, (False, True), strict=True):
        settings = copy.deepcopy(cfg)
        settings["data"]["sampling"]["coverage_resample"] = changed
        loader, _, _ = build_dataloaders(settings, DistEnv(False))
        dataset = loader.dataset
        dataset.set_epoch(1)
        cfg_crop = dataset.crop_cfg
        if cfg_crop.crops_per_item != 1 or cfg_crop.crops_per_tile_per_epoch != 12:
            raise ValueError("Replay requires the frozen 12 crops per tile")
        rows = []
        for tile in range(len(dataset.files)):
            for rep in (0, 6):
                crops = dataset[tile * 12 + rep]
                crops = crops if isinstance(crops, list) else [crops]
                if len(crops) != 1:
                    raise ValueError(f"Replay crop missing: {arm}/{tile}/{rep}")
                crop = crops[0]
                rows.append(
                    {
                        "tile": tile,
                        "rep": rep,
                        "hits": crop["meta_coverage_hits"].tolist(),
                        "size_m": float(crop["meta_crop_size_xy_m"]),
                        "voxels": len(crop["labels"]),
                        "class_voxels": np.bincount(
                            crop["labels"].numpy()[crop["labels"].numpy() >= 0],
                            minlength=8,
                        ).tolist(),
                        "replaced": bool(crop["meta_coverage_replaced"]),
                        "centre_xy": crop["meta_coverage_center_xy"].tolist(),
                        "centre_native": int(crop["meta_coverage_center_native"]),
                    }
                )
        result[arm] = rows
    left, right = (result[a] for a in ARMS)
    a, b = (np.sum([row["hits"] for row in rows], axis=0) for rows in (left, right))
    # Compare each class's least represented source geometry bin, defined by control.
    least = {c: int(np.argmin(a[c])) for c in CLASSES}
    changes = {c: int(b[c, k] - a[c, k]) for c, k in least.items()}
    replacement = sum(r["replaced"] for r in right) / len(right)
    ratio = sum(r["voxels"] for r in right) / sum(r["voxels"] for r in left)
    passed = (
        replacement >= 0.10
        and sum(v >= 2 for v in changes.values()) >= 2
        and 0.8 <= ratio <= 1.2
    )
    value = {
        "status": "admitted" if passed else "no_meaningful_controlled_exposure_change",
        "catalog_sha256": sha256(cfg["data"]["sampling"]["coverage_catalog"]),
        "least_represented_bins": least,
        "extra_crops_with_anchors": changes,
        "accepted_replacement_rate": replacement,
        "voxel_workload_ratio": ratio,
        "crops": result,
    }
    atomic_json(out, value)
    return value


def exposure_record(rows):
    ranks = [rank for row in rows for rank in row["ranks"]]
    centres = [entry for rank in ranks for entry in rank["coverage_centres"]]
    return {
        "epochs": len(rows),
        "crops": len(centres),
        "crop_hits": np.sum([rank["coverage_hits"] for rank in ranks], axis=0).tolist(),
        "accepted_native_counts": np.sum(
            [row["accepted_native_counts"] for row in rows], axis=0
        ).tolist(),
        "replaced": sum(entry[3] for entry in centres),
        "distinct_60m_cells_by_native_center": {
            str(label): len(
                {
                    (p, int(x // 60), int(y // 60))
                    for p, (x, y), c, _ in centres
                    if c == label
                }
            )
            for label in sorted({c for _, _, c, _ in centres})
        },
        "counters_per_rank": [
            [rank["optimizer_updates_cumulative"], rank["amp_skips_cumulative"]]
            for rank in rows[-1]["ranks"]
        ],
    }


def compare_if_complete(study):
    import fcntl

    from .feature_contract_study import bootstrap, metric_vector
    from .study_metrics import COMMON_NAMES, summary
    from .study_runtime import atomic_json

    study = Path(study)
    paths = [study / arm for arm in ARMS]
    with (study / ".coverage_comparison.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not all(
            (p / "STATUS.json").is_file()
            and json.loads((p / "STATUS.json").read_text())["status"] == "complete"
            for p in paths
        ):
            return
        manifests = [
            json.loads((p / "evaluation_manifest.json").read_text()) for p in paths
        ]
        import yaml

        configs = [
            yaml.safe_load((p / "config_resolved.yaml").read_text()) for p in paths
        ]
        for cfg in configs:
            cfg["run"].pop("out_dir")
            cfg["run"].pop("resume_from", None)
            cfg["study"].pop("arm")
            cfg["data"]["sampling"].pop("coverage_resample")
        if configs[0] != configs[1]:
            raise ValueError("Resolved settings differ beyond the intervention")
        if any(m["target_checkpoint_selection"] for m in manifests):
            raise ValueError("Target-selected checkpoint")
        if any(
            manifests[0][k] != manifests[1][k]
            for k in ("source_manifest_sha256", "dataset_inventory_sha256")
        ):
            raise ValueError("Code or datasets differ")
        init = [
            json.loads((p / "preflight.json").read_text())["gpu_smoke"][
                "initial_tensor_hashes"
            ]
            for p in paths
        ]
        if init[0] != init[1]:
            raise ValueError("Initial weights differ")
        exposure = {}
        for arm, path in zip(ARMS, paths, strict=True):
            rows = [
                json.loads(line)
                for line in (path / "study_epochs.jsonl").read_text().splitlines()
            ]
            if [r["epoch"] for r in rows] != list(range(1, 201)):
                raise ValueError(f"Incomplete training audit for {arm}")
            exposure[arm] = exposure_record(rows)
        replay = json.loads((study / "coverage_replay.json").read_text())
        left, right = [exposure[a] for a in ARMS]
        changes = {
            c: right["crop_hits"][int(c)][k] / right["crops"]
            - left["crop_hits"][int(c)][k] / left["crops"]
            for c, k in replay["least_represented_bins"].items()
        }
        counts = [exposure[a]["counters_per_rank"] for a in ARMS]
        execution = (
            left["crops"] == right["crops"]
            and all(len(c) == 2 and c[0] == c[1] and c[0][1] <= 13 for c in counts)
            and sum(counts[0][0]) == sum(counts[1][0])
        )
        delivered = (
            right["replaced"] / right["crops"] >= 0.10
            and sum(d >= 0.02 for d in changes.values()) >= 2
        )
        if replay.get("policy") == CONTENTS_POLICY:
            expected = qualify_contents(
                study / "coverage_qualification.json",
                sha256(study / "coverage_catalog.json"),
            )
            if replay != expected:
                raise ValueError("Contents admission record changed")
            delivered = (
                delivered
                and changes["3"] >= 0.02
                and max(changes["4"], changes["5"]) >= 0.02
                and min(changes.values()) >= 0.0
            )
        names = [*COMMON_NAMES[1:], "all_seven_miou", "utility_miou"]
        result = {}
        for ep in ("best", "last"):
            result[ep] = {}
            for domain, expected in (("dales", 11), ("eclair", 125)):
                folders = [
                    p / "evaluation" / m["checkpoints"][ep]["model_sha256"] / domain
                    for p, m in zip(paths, manifests, strict=True)
                ]
                metrics = [
                    json.loads((f / "metrics.json").read_text()) for f in folders
                ]
                contracts = [
                    {k: v for k, v in m["contract"].items() if k != "checkpoint"}
                    for m in metrics
                ]
                if contracts[0] != contracts[1]:
                    raise ValueError("Canonical evaluation contracts differ")
                rows = [
                    [json.loads(p.read_text()) for p in sorted(f.glob("tile_*.json"))]
                    for f in folders
                ]
                if len(rows[0]) != expected or [r["path"] for r in rows[0]] != [
                    r["path"] for r in rows[1]
                ]:
                    raise ValueError("Canonical tile lists differ")
                matrices = [np.asarray([r["matrix"] for r in arm]) for arm in rows]
                pooled = [a.sum(0) for a in matrices]
                for cm, m in zip(pooled, metrics, strict=True):
                    if not np.array_equal(cm, m["confusion_matrix"]):
                        raise ValueError("Tile and pooled matrices differ")
                strata = [
                    np.sum([r["geometry_anchor_counts"] for r in arm], axis=0)
                    for arm in rows
                ]
                if not np.array_equal(strata[0].sum(2), strata[1].sum(2)):
                    raise ValueError("Geometry anchor support differs")
                transitions = np.zeros((8, 8, 8), dtype=np.int64)
                for a, b in zip(*rows, strict=True):
                    x, y = a["anchor_predictions"], b["anchor_predictions"]
                    if any(x[k] != y[k] for k in ("ids", "native_gt", "gt")):
                        raise ValueError("Paired anchor identity differs")
                    np.add.at(
                        transitions, (x["gt"], x["prediction"], y["prediction"]), 1
                    )
                result[ep][domain] = {
                    "delta_pp": dict(
                        zip(
                            names,
                            (
                                100
                                * (metric_vector(pooled[1]) - metric_vector(pooled[0]))
                            ).tolist(),
                        )
                    ),
                    "arms": [
                        {
                            **summary(cm),
                            "subtype_recall": m["subtype_recall"],
                            "native_gt_by_common_prediction": m[
                                "native_gt_by_common_prediction"
                            ],
                        }
                        for cm, m in zip(pooled, metrics, strict=True)
                    ],
                    "intervals": bootstrap(*matrices),
                    "geometry_anchor_counts": [s.tolist() for s in strata],
                    "anchor_gt_by_control_by_candidate": transitions.tolist(),
                }
        source_ok = all(
            all(
                v >= (-2 if k in COMMON_NAMES else -1)
                for k, v in result[ep]["dales"]["delta_pp"].items()
            )
            for ep in ("best", "last")
        )
        best = result["best"]["eclair"]
        last = result["last"]["eclair"]
        aggregates = ("all_seven_miou", "utility_miou")
        target_ok = (
            min(best["delta_pp"].values()) >= -2
            and min(last["delta_pp"].values()) >= -2
            and all(
                best["delta_pp"][k] >= 1
                and best["intervals"][k]["low_pp"] > 0
                and last["delta_pp"][k] > 0
                for k in aggregates
            )
        )
        if not execution:
            decision = "execution_mismatch"
        elif not delivered:
            decision = "exposure_manipulation_not_delivered"
        elif not source_ok:
            decision = "source_preservation_failed"
        elif target_ok:
            decision = "balanced_development_pass_confirm"
        else:
            decision = "no_balanced_gain_close_tested_policy"
        atomic_json(
            study / "coverage_comparison.json",
            {
                "decision": decision,
                "source_guard_passed": source_ok,
                "target_guard_passed": target_ok,
                "execution_matched": execution,
                "exposure_delivered": delivered,
                "least_bin_crop_rate_changes": changes,
                "exposure": exposure,
                "endpoints": result,
                "seed_scope": "one development seed",
            },
        )
