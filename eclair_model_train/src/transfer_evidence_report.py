"""Complete measurement ledger and descriptive research triage; no causal claims."""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np

from .study_metrics import COMMON_NAMES, summary
from .transfer_evidence_sampling import geometry_descriptor, seed_for


def read_json(path):
    import json

    return json.loads(Path(path).read_text())


def write_csv(store, name, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    store.text(name, stream.getvalue())


def finite_mean(x, weights=None):
    x = np.asarray(x, float)
    valid = np.isfinite(x)
    if weights is not None:
        weights = np.asarray(weights)
        valid &= weights > 0
    if not valid.any():
        return None
    return float(np.average(x[valid], weights=weights[valid] if weights is not None else None))


def verify_canonical(out, tasks, cfg):
    rows = {}
    for domain in ("dales", "eclair"):
        tiles = [read_json(Path(out) / f"test/{domain}_{i:04d}.json") for i in range(cfg["test_tiles"][domain])]
        for task in tasks:
            matrices = np.asarray([t["models"][task["id"]]["confusion_matrix"] for t in tiles], np.int64)
            expected = task["canonical"][domain]
            if len(tiles) != expected["tiles"] or not np.array_equal(matrices.sum(0), expected["matrix"]):
                raise ValueError(f"Canonical integer matrix not reproduced: {task['id']} -> {domain}")
            rows[(task["id"], domain)] = {"tiles": matrices, "summary": summary(matrices.sum(0))}
    return rows


def bootstrap_delta(reference, candidate, seed):
    if reference.shape != candidate.shape:
        raise ValueError("Paired bootstrap requires matching raw tiles")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(500):
        ix = rng.integers(len(reference), size=len(reference))
        a, b = summary(reference[ix].sum(0)), summary(candidate[ix].sum(0))
        if a["undefined_union_ids"] or b["undefined_union_ids"]:
            continue
        estimates.append(
            [
                b["all_seven_miou"] - a["all_seven_miou"],
                b["utility_miou"] - a["utility_miou"],
                *[b["per_class_iou"][c] - a["per_class_iou"][c] for c in COMMON_NAMES[1:]],
            ]
        )
    if not estimates:
        return {"valid_resamples": 0, "lower_pp": None, "upper_pp": None}
    lo, hi = np.quantile(estimates, [0.025, 0.975], axis=0) * 100
    return {"valid_resamples": len(estimates), "lower_pp": lo.tolist(), "upper_pp": hi.tolist()}


def load_anchors(out, banks, domain, split):
    parts = defaultdict(list)
    for (d, s, tile), path in sorted(banks.items()):
        if (d, s) != (domain, split):
            continue
        with np.load(Path(out) / path, allow_pickle=False) as a:
            for key in a.files:
                parts[key].append(a[key])
            parts["tile"].append(np.full(len(a["ids"]), tile, np.int32))
    return {k: np.concatenate(v) for k, v in parts.items()}


def load_test_features(out, task, domain, cfg):
    parts = defaultdict(list)
    for i in range(cfg["test_tiles"][domain]):
        with np.load(Path(out) / f"test/{domain}_{i:04d}_{task['id']}.npz", allow_pickle=False) as a:
            for key in (
                "point_ids",
                "features",
                "gt",
                "native",
                "pred",
                "native_confidence",
                "mapped_class_probability",
                "native_logit_margin",
                "pooled_inputs",
                "window_edge_m",
                "split_depth",
            ):
                parts[key].append(a[key])
    return {k: np.concatenate(v) for k, v in parts.items()}


def novelty(source, source_tiles, query):
    """Source-only scaling and leave-one-source-tile-out distance reference."""
    from scipy.spatial import cKDTree

    result = np.full(len(query), np.nan)
    valid = np.isfinite(source).all(1)
    source, source_tiles = source[valid], source_tiles[valid]
    if len(source) < 16 or len(np.unique(source_tiles)) < 2:
        return result, None
    center = np.median(source, axis=0)
    lo, hi = np.quantile(source, [0.25, 0.75], axis=0)
    scale = np.maximum(hi - lo, 1e-4)
    z = (source - center) / scale
    reference = []
    for tile in np.unique(source_tiles):
        keep = source_tiles != tile
        distances = cKDTree(z[keep]).query(z[~keep], k=1)[0]
        reference.extend(np.atleast_1d(distances).tolist())
    threshold = float(np.quantile(reference, 0.95))
    valid_query = np.isfinite(query).all(1)
    result[valid_query] = cKDTree(z).query((query[valid_query] - center) / scale, k=1)[0]
    return result, threshold


def source_feature_bank(out, task, windows):
    parts = defaultdict(list)
    for wi in range(len(windows[task["source"] + "/train"])):
        row = read_json(Path(out) / f"source/{task['source']}_{wi:03d}.json")
        with np.load(Path(out) / row["models"][task["id"]]["artifact"], allow_pickle=False) as a:
            for key in ("point_ids", "features", "gt", "geometry"):
                parts[key].append(a[key])
            parts["tile"].append(np.full(len(a["gt"]), row["spec"]["tile"], np.int32))
    data = {k: np.concatenate(v) for k, v in parts.items()}
    _, inverse = np.unique(np.column_stack((data["tile"], data["point_ids"])), axis=0, return_inverse=True)
    first = np.unique(inverse, return_index=True)[1]
    # A point observed in overlapping reference windows counts once; average its features.
    sums = np.zeros((len(first), data["features"].shape[1]), np.float64)
    np.add.at(sums, inverse, data["features"])
    means = sums / np.bincount(inverse)[:, None]
    return {**{k: v[first] for k, v in data.items()}, "features": means}


def create_report(store, tasks, banks, windows, cfg):
    out = store.root
    canonical = verify_canonical(out, tasks, cfg)
    tile_cache = {
        (d, i): store.complete(f"test/{d}_{i:04d}.json")
        for d in ("dales", "eclair")
        for i in range(cfg["test_tiles"][d])
    }
    measures = {
        "canonical_pairs": len(canonical),
        "geometry_tiles": len(banks),
        "source_windows": {},
        "probe_windows": {},
        "sampled_classes": {},
        "limitations": [],
    }
    canonical_rows, delta_rows, uncertainty = [], [], []
    for task in tasks:
        key = task["id"]
        reference = (
            cfg["controlled_reference"]
            if key == cfg["controlled_candidate"]
            else cfg["historical_reference"][task["source"]]
        )
        for domain in ("dales", "eclair"):
            score = canonical[(key, domain)]["summary"]
            base = canonical[(reference, domain)]["summary"]
            for cls in COMMON_NAMES[1:]:
                p, r = score["precision"][cls], score["recall"][cls]
                canonical_rows.append(
                    {
                        "model": key,
                        "source": task["source"],
                        "evaluated_domain": domain,
                        "class": cls,
                        "iou": score["per_class_iou"][cls],
                        "precision": p,
                        "recall": r,
                        "f1": 2 * p * r / (p + r) if p + r else 0.0,
                        "all_seven_miou": score["all_seven_miou"],
                        "utility_miou": score["utility_miou"],
                    }
                )
                delta_rows.append(
                    {
                        "model": key,
                        "reference": reference,
                        "domain": domain,
                        "class": cls,
                        "delta_iou_pp": 100 * (score["per_class_iou"][cls] - base["per_class_iou"][cls]),
                    }
                )
            if key != reference:
                uncertainty.append(
                    {
                        "model": key,
                        "reference": reference,
                        "domain": domain,
                        **bootstrap_delta(
                            canonical[(reference, domain)]["tiles"],
                            canonical[(key, domain)]["tiles"],
                            seed_for(key, domain, "bootstrap"),
                        ),
                    }
                )
    write_csv(store, "canonical_metrics.csv", canonical_rows)
    write_csv(store, "class_deltas.csv", delta_rows)
    store.json(
        "tile_bootstrap.json",
        {
            "unit": "tile",
            "descriptive_not_seed_replication": True,
            "metric_order": ["all_seven", "utility", *COMMON_NAMES[1:]],
            "comparisons": uncertainty,
        },
    )
    store.json("canonical_matrices.json", {k + "/" + d: v["summary"] for (k, d), v in canonical.items()})

    input_rows, pooled_rows, calibration, edge_rows, agreement = [], [], [], [], {}
    native_tables = {}
    for domain in ("dales", "eclair"):
        for task in tasks:
            key = task["id"]
            totals = {}
            restricted, merged = np.zeros((8, 8), np.int64), np.zeros((8, 8), np.int64)
            for i in range(cfg["test_tiles"][domain]):
                tile = tile_cache[(domain, i)]
                record = tile["models"][key]
                restricted += np.asarray(record["restricted_common_outputs"]["confusion_matrix"])
                merged += np.asarray(record["probability_merge_rule"]["confusion_matrix"])
                with np.load(out / record["artifact"], allow_pickle=False) as a:
                    for name in (
                        "joint_returns",
                        "invalid_returns",
                        "pooled_sum",
                        "pooled_square",
                        "confidence_hist",
                        "confidence_sums",
                        "common_confidence_hist",
                        "window_edge_confusions",
                        "actual_subset_edge_confusions",
                        "split_depth_confusions",
                        "native_gt_by_native_prediction",
                        "native_gt_by_common_prediction",
                        "native_return_joint",
                        "native_invalid_returns",
                        "native_logit_margin_hist",
                    ):
                        totals[name] = totals.get(name, 0) + a[name]
            cm = np.asarray(canonical[(key, domain)]["summary"]["confusion_matrix"])
            if totals["joint_returns"].sum() != cm.sum():
                raise ValueError("Error-linked return counts do not cover all canonical points")
            for cls, pred, rn, nor in np.argwhere(totals["joint_returns"] > 0):
                input_rows.append(
                    {
                        "model": key,
                        "domain": domain,
                        "gt": int(cls),
                        "pred": int(pred),
                        "rn_bin": int(rn),
                        "nor_bin": int(nor),
                        "points": int(totals["joint_returns"][cls, pred, rn, nor]),
                    }
                )
            for pair in range(64):
                n = int(cm.ravel()[pair])
                if n:
                    for channel in range(totals["pooled_sum"].shape[1]):
                        mean = totals["pooled_sum"][pair, channel] / n
                        pooled_rows.append(
                            {
                                "model": key,
                                "domain": domain,
                                "gt": pair // 8,
                                "pred": pair % 8,
                                "channel": channel,
                                "points": n,
                                "mean": float(mean),
                                "variance": float(max(0, totals["pooled_square"][pair, channel] / n - mean**2)),
                            }
                        )
            hist, sums = totals["confidence_hist"], totals["confidence_sums"]
            if int(hist.sum()) != int(cm.sum()):
                raise ValueError("Confidence counter omits canonical points")
            if int(totals["native_logit_margin_hist"].sum()) != int(cm.sum()):
                raise ValueError("Margin counter omits canonical points")
            for cls in range(1, 8):
                n = hist[cls].sum()
                ece = 0.0
                for b in range(10):
                    size = hist[cls, :, b].sum()
                    if size:
                        ece += abs(hist[cls, 0, b] - sums[cls, :, b].sum()) / max(1, n)
                calibration.append(
                    {
                        "model": key,
                        "domain": domain,
                        "class": COMMON_NAMES[cls],
                        "native_confidence_ece_fixed_bins": float(ece),
                        "points": int(n),
                    }
                )
                for definition, array_key in (
                    ("scheduled_window", "window_edge_confusions"),
                    ("actual_forward_subset_bbox", "actual_subset_edge_confusions"),
                ):
                    for b in range(4):
                        s = summary(totals[array_key][b])
                        edge_rows.append(
                            {
                                "model": key,
                                "domain": domain,
                                "class": COMMON_NAMES[cls],
                                "edge_definition": definition,
                                "edge_bin": b,
                                "support": s["support"][cls],
                                "recall": s["recall"][COMMON_NAMES[cls]],
                            }
                        )
            native_tables[key + "/" + domain] = {
                "native_gt_by_native_prediction": totals["native_gt_by_native_prediction"].tolist(),
                "native_gt_by_common_prediction": totals["native_gt_by_common_prediction"].tolist(),
                "invalid_return_pairs": totals["invalid_returns"].tolist(),
                "confidence_hist": hist.tolist(),
                "confidence_sums": sums.tolist(),
                "common_confidence_hist": totals["common_confidence_hist"].tolist(),
                "native_logit_margin_hist": totals["native_logit_margin_hist"].tolist(),
                "window_edge_confusions": totals["window_edge_confusions"].tolist(),
                "actual_subset_edge_confusions": totals["actual_subset_edge_confusions"].tolist(),
                "split_depth_confusions": totals["split_depth_confusions"].tolist(),
                "restricted_outputs": summary(restricted),
                "probability_merge_rule": summary(merged),
            }
            artifact = store.npz(
                f"native_inputs/{key}_{domain}.npz",
                native_return_joint=totals["native_return_joint"],
                native_invalid_returns=totals["native_invalid_returns"],
            )
            native_tables[key + "/" + domain]["native_input_artifact"] = artifact
            # Use the runtime's exact mapping, preserved as part of the output bundle.
            runtime_mapping = read_json(out / "mapping_manifest.json")
            if runtime_mapping:
                lut = np.asarray(runtime_mapping["native_to_common"][domain])
                table = totals["native_gt_by_common_prediction"]
                native_tables[key + "/" + domain]["native_subtype_common_recall"] = {
                    str(n): {
                        "support": int(table[n].sum()),
                        "common_class": int(lut[n]),
                        "recall": float(table[n, lut[n]] / table[n].sum()),
                    }
                    for n in range(256)
                    if table[n].sum() and lut[n] > 0
                }
        for i in range(cfg["test_tiles"][domain]):
            tile = tile_cache[(domain, i)]
            for pair, values in tile["agreement"].items():
                k = domain + "/" + pair
                agreement[k] = np.asarray(agreement.get(k, 0)) + np.asarray(values)
    write_csv(store, "error_conditioned_inputs.csv", input_rows)
    write_csv(store, "pooled_input_features.csv", pooled_rows)
    write_csv(store, "calibration.csv", calibration)
    write_csv(store, "inference_boundaries.csv", edge_rows)
    store.json("native_and_decision_metrics.json", native_tables)
    store.json("model_error_agreement.json", {k: v.tolist() for k, v in agreement.items()})
    catalog = read_json(Path(__file__).resolve().parents[1] / "configs/transfer_evidence/scan_catalog.json")
    raw_joints = {}
    for group in ("dales/train", "dales/test", "eclair/train", "eclair/test"):
        records = [v for k, v in catalog.items() if k.startswith(group + "/")]
        raw_joints[group] = np.sum([r["returns"]["joint_common_class_by_rn_nor"] for r in records], axis=0).tolist()
    store.json(
        "reused_raw_return_distributions.json",
        {
            "encoding": "canonical five clipped bins per attribute",
            "source": "payload-verified earlier full scans",
            "groups": raw_joints,
        },
    )

    raw_banks = {
        (domain, split): load_anchors(out, banks, domain, split)
        for domain in ("dales", "eclair")
        for split in ("train", "test")
    }
    source_geometry_rows = []
    for domain in ("dales", "eclair"):
        raw_source = raw_banks[(domain, "train")]
        for cls in range(1, 8):
            keep = (raw_source["gt"] == cls) & ((raw_source["flags"] & 1) != 0)
            for ri, radius in enumerate(cfg["radii_m"]):
                for var in range(raw_source["geometry"].shape[2]):
                    values = raw_source["geometry"][keep, ri, var]
                    source_geometry_rows.append(
                        {
                            "domain": domain,
                            "class": COMMON_NAMES[cls],
                            "radius_m": radius,
                            "variable": var,
                            "valid_points": int(np.isfinite(values).sum()),
                            "mean": finite_mean(values),
                        }
                    )
    write_csv(store, "source_training_geometry.csv", source_geometry_rows)
    geometry_rows, conditional_rows, support_rows, feature_rows, subtype_rows = [], [], [], [], []
    for task in tasks:
        source = raw_banks[(task["source"], "train")]
        bank = source_feature_bank(out, task, windows)
        measures["sampled_classes"][task["id"]] = {}
        for domain in ("dales", "eclair"):
            store.heartbeat()
            raw = raw_banks[(domain, "test")]
            test = load_test_features(out, task, domain, cfg)
            if not np.array_equal(raw["ids"], test["point_ids"]) or not np.array_equal(raw["gt"], test["gt"]):
                raise ValueError("Geometry/error/feature point identities do not align")
            primary = (raw["flags"] & 1) != 0
            wrong = test["pred"] != raw["gt"]
            for cls in range(1, 8):
                s = (source["gt"] == cls) & ((source["flags"] & 1) != 0)
                q = (raw["gt"] == cls) & primary
                if not q.any() or not s.any():
                    raise ValueError("Required common class has no probability sample")
                measures["sampled_classes"][task["id"]][domain + "/" + str(cls)] = int(q.sum())
                for radius in range(len(cfg["radii_m"])):
                    for failed in (False, True):
                        keep = q & (wrong == failed)
                        for variable in range(raw["geometry"].shape[2]):
                            values = raw["geometry"][keep, radius, variable]
                            geometry_rows.append(
                                {
                                    "model": task["id"],
                                    "domain": domain,
                                    "class": COMMON_NAMES[cls],
                                    "wrong": failed,
                                    "radius_m": cfg["radii_m"][radius],
                                    "variable": variable,
                                    "sampled_points": int(keep.sum()),
                                    "valid_values": int(np.isfinite(values).sum()),
                                    "mean": finite_mean(values, 1 / raw["probability"][keep]),
                                }
                            )
                failed = wrong[q]
                local_support = None
                for ri in dict.fromkeys((2, len(cfg["radii_m"]) - 1)):
                    source_desc = geometry_descriptor(source["geometry"][s], ri)
                    query_desc = geometry_descriptor(raw["geometry"][q], ri)
                    distances, threshold = novelty(source_desc, source["tile"][s], query_desc)
                    covered = distances <= threshold if threshold is not None else np.zeros(len(distances), bool)
                    known = np.isfinite(distances) if threshold is not None else np.zeros(len(distances), bool)
                    if ri == 2:
                        local_support = covered, known
                    support_rows.append(
                        {
                            "model": task["id"],
                            "source": task["source"],
                            "domain": domain,
                            "class": COMMON_NAMES[cls],
                            "radius_m": cfg["radii_m"][ri],
                            "source_probability_points": int(s.sum()),
                            "query_points": int(q.sum()),
                            "valid_support_comparisons": int(known.sum()),
                            "source_only_distance_threshold": threshold,
                            "inside_support_points": int((covered & known).sum()),
                            "outside_support_points": int(((~covered) & known).sum()),
                            "outside_support_fraction": finite_mean((~covered)[known]),
                            "error_inside_support": finite_mean(failed[covered & known]),
                            "error_outside_support": finite_mean(failed[(~covered) & known]),
                        }
                    )
                covered, known = local_support
                # Source-only bin edges. Comparisons remain observational; empty cells stay explicit.
                density_edges = np.unique(np.quantile(np.log1p(source["geometry"][s, 2, 0]), [0.25, 0.5, 0.75]))
                shape_values = source["geometry"][s, 2, 4]
                shape_values = shape_values[np.isfinite(shape_values)]
                shape_edges = (
                    np.unique(np.quantile(shape_values, [0.25, 0.5, 0.75])) if len(shape_values) else np.array([])
                )
                density = np.digitize(np.log1p(raw["geometry"][q, 2, 0]), density_edges)
                shape = np.digitize(raw["geometry"][q, 2, 4], shape_edges)
                shape[~np.isfinite(raw["geometry"][q, 2, 4])] = -1
                rn = np.clip(raw["rn"][q], 1, 5).astype(int)
                nor = np.clip(raw["nor"][q], 1, 5).astype(int)
                return_pair = rn * 10 + nor
                for name, left, right in (
                    ("returns_density", return_pair, density),
                    ("shape_density", shape, density),
                    ("native_returns", raw["native"][q], return_pair),
                    ("label_mixture_density", (raw["geometry"][q, 2, 10] < 0.95).astype(int), density),
                    ("broad_label_mixture_density", (raw["geometry"][q, -1, 10] < 0.95).astype(int), density),
                ):
                    for a, b in np.unique(np.column_stack((left, right)), axis=0):
                        keep = (left == a) & (right == b)
                        conditional_rows.append(
                            {
                                "model": task["id"],
                                "domain": domain,
                                "class": COMMON_NAMES[cls],
                                "comparison": name,
                                "left": int(a),
                                "right": int(b),
                                "sampled_points": int(keep.sum()),
                                "wrong_points": int(failed[keep].sum()),
                                "source_density_edges": repr(density_edges.tolist()),
                                "source_shape_edges": repr(shape_edges.tolist()),
                            }
                        )
                bs = bank["gt"] == cls
                bvalues = bank["features"][bs].astype(float)
                valid_features = len(bvalues) >= 16 and len(np.unique(bank["tile"][bs])) >= 2
                if valid_features:
                    scale = np.maximum(bank["features"].astype(float).std(0), 1e-4)
                    center = bvalues.mean(0)
                    distance = np.sqrt(np.mean(((test["features"][q] - center) / scale) ** 2, axis=1))
                else:
                    distance = np.full(int(q.sum()), np.nan)
                feature_rows.append(
                    {
                        "model": task["id"],
                        "domain": domain,
                        "class": COMMON_NAMES[cls],
                        "source_bank_points": int(bs.sum()),
                        "raw_support_radius_m": 2.0,
                        "source_bank_tiles": int(len(np.unique(bank["tile"][bs]))),
                        "source_bank_adequate_for_descriptive_comparison": bool(valid_features),
                        "distance_correct": finite_mean(distance[~failed]),
                        "distance_wrong": finite_mean(distance[failed]),
                        "distance_supported_correct": finite_mean(distance[(~failed) & covered & known]),
                        "distance_supported_wrong": finite_mean(distance[failed & covered & known]),
                        "supported_correct_points": int(((~failed) & covered & known).sum()),
                        "supported_wrong_points": int((failed & covered & known).sum()),
                    }
                )
            for native in np.unique(raw["native"]):
                # Native supplements are intentionally not population-weighted. Label their role explicitly.
                keep = raw["native"] == native
                subtype_rows.append(
                    {
                        "model": task["id"],
                        "domain": domain,
                        "native_label": int(native),
                        "sampled_points": int(keep.sum()),
                        "sampling": "union of probability/native/tile samples; descriptive only",
                        "sample_error_fraction": float(wrong[keep].mean()),
                        "rn_mean": float(raw["rn"][keep].mean()),
                        "nor_mean": float(raw["nor"][keep].mean()),
                    }
                )
    write_csv(store, "error_conditioned_geometry.csv", geometry_rows)
    write_csv(store, "conditional_error_strata.csv", conditional_rows)
    write_csv(store, "source_support.csv", support_rows)
    write_csv(store, "feature_support.csv", feature_rows)
    write_csv(store, "sampled_subtypes.csv", subtype_rows)

    probe_rows, probe_validity = [], []
    for domain in ("dales", "eclair"):
        measures["source_windows"][domain] = len(windows[domain + "/train"])
        measures["probe_windows"][domain] = len(windows[domain + "/test"])
        for wi in range(len(windows[domain + "/test"])):
            row = store.complete(f"probes/{domain}_{wi:03d}.json")
            for task in tasks:
                item = row["models"][task["id"]]
                if not item["identity_control_passed"] or not item["state_unchanged"] or len(item["probes"]) != 4:
                    raise ValueError("Missing paired probe controls or seeds")
                for p in item["probes"]:
                    probe_validity.append(
                        {
                            "domain": domain,
                            "model": task["id"],
                            "window": wi,
                            "mode": p["mode"],
                            "seed": p["seed"],
                            **p["effective_change"],
                        }
                    )
                    for cls in COMMON_NAMES[1:]:
                        probe_rows.append(
                            {
                                "domain": domain,
                                "model": task["id"],
                                "window": wi,
                                "mode": p["mode"],
                                "seed": p["seed"],
                                "class": cls,
                                "clean_iou": p["clean_on_retained"]["per_class_iou"][cls],
                                "perturbed_iou": p["perturbed"]["per_class_iou"][cls],
                                "support": p["perturbed"]["support"][COMMON_NAMES.index(cls)],
                            }
                        )
    write_csv(store, "paired_probe_results.csv", probe_rows)
    write_csv(store, "probe_validity.csv", probe_validity)
    render_gallery(store, cfg)
    build_decision_report(store, tasks, canonical, support_rows, feature_rows, probe_rows, measures, cfg)
    expected_geometry = sum(cfg[k][d] for k in ("train_tiles", "test_tiles") for d in ("dales", "eclair"))
    if measures["canonical_pairs"] != 2 * len(tasks) or measures["geometry_tiles"] != expected_geometry:
        raise ValueError("Required checkpoint/data matrix incomplete")
    for d in ("dales", "eclair"):
        if (
            measures["source_windows"][d] != cfg["source_windows"]
            or measures["probe_windows"][d] != cfg["probe_windows"]
        ):
            raise ValueError("Frozen source/probe window count not completed")
    store.json("measurement_coverage.json", measures)
    return measures


def render_gallery(store, cfg):
    colors = ["#999999", "#b88646", "#238b45", "#477bcc", "#e3a008", "#dc3939", "#a253a4", "#1b9393"]
    items = []
    for domain in ("dales", "eclair"):
        for i in range(cfg["test_tiles"][domain]):
            items.extend(read_json(store.root / f"test/{domain}_{i:04d}.json").get("gallery", []))
    html = [
        "<!doctype html><meta charset='utf-8'><title>Transfer error review</title>",
        "<style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{width:100%}article{margin:35px 0}</style>",
        "<h1>Persistent failures and rescued examples</h1><p>Sampled context, not complete assets. "
        "Selection is illustrative. Native labels need human review; these views cannot establish missing source subtypes.</p>",
    ]
    for i, item in enumerate(items):
        with np.load(store.root / item["artifact"]["path"], allow_pickle=False) as a:
            xyz, gt = a["xyz"], a["gt"]
        title = (
            f"{item['group'][0]} · {COMMON_NAMES[item['group'][1]]} · {item['group'][2]} · "
            f"tile {item['tile']} · point {item['anchor_point']}"
        )
        svg = [
            "<svg xmlns='http://www.w3.org/2000/svg' width='1000' height='560' viewBox='0 0 1000 560'>",
            "<rect width='1000' height='560' fill='white'/>",
            f"<text x='24' y='28' font-family='sans-serif' font-size='17'>{escape(title)}</text>",
        ]
        for panel, axis in enumerate((1, 2)):
            ox = 250 + panel * 500
            svg.append(
                f"<text x='{ox - 100}' y='60' font-family='sans-serif'>X / {'Y' if axis == 1 else 'Z'} · ±16 m</text>"
            )
            for p, label in zip(xyz, gt):
                svg.append(
                    f"<circle cx='{ox + p[0] * 13:.2f}' cy='{295 - p[axis] * 13:.2f}' r='1.4' fill='{colors[int(label)]}'/>"
                )
        svg.append(
            "<text x='24' y='545' font-family='sans-serif' font-size='12'>Ground brown · vegetation green · building blue · "
            "wire yellow · pole red · fence purple · vehicle teal</text></svg>"
        )
        name = f"gallery/review_{i:02d}.svg"
        store.text(name, "".join(svg))
        html.append(
            f"<article><h2>{escape(title)}</h2><img src='review_{i:02d}.svg'><p>Native label {item['native_label']}; "
            f"{item['saved_points']} / {item['raw_neighbourhood_points']} neighbours shown. "
            "Raw point identity is retained in the index.</p></article>"
        )
        if item.get("anchor_predictions"):
            labels = "; ".join(f"{escape(k)}: {COMMON_NAMES[v]}" for k, v in item["anchor_predictions"].items())
            html.append(f"<p>Anchor predictions: {labels}. Views show ground-truth context.</p>")
    store.text("gallery/review.html", "\n".join(html))
    store.json("gallery/index.json", items)


def evidence_cards(store, tasks, canonical, support, features, probes):
    sources = {t["id"]: t["source"] for t in tasks}
    coverage = [
        r
        for r in support
        if r["domain"] != r["source"]
        and r["inside_support_points"] >= 32
        and r["outside_support_points"] >= 32
        and r["error_inside_support"] is not None
        and r["error_outside_support"] is not None
        and r["error_outside_support"] - r["error_inside_support"] >= 0.05
    ]
    feature = [
        r
        for r in features
        if r["domain"] != sources[r["model"]]
        and r["source_bank_adequate_for_descriptive_comparison"]
        and r["supported_correct_points"] >= 32
        and r["supported_wrong_points"] >= 32
        and r["distance_supported_wrong"] is not None
        and r["distance_supported_correct"] is not None
        and r["distance_supported_wrong"] > r["distance_supported_correct"]
    ]
    grouped = defaultdict(list)
    for p in probes:
        if p["support"] >= 32:
            grouped[(p["model"], p["domain"], p["mode"], p["seed"], p["class"])].append(
                p["perturbed_iou"] - p["clean_iou"]
            )
    sensitivities = []
    for (model, domain, mode, seed, cls), values in grouped.items():
        if domain == sources[model]:
            continue
        control = grouped.get((model, sources[model], mode, seed, cls), [])
        delta = float(np.median(values))
        own = float(np.median(control)) if control else None
        if own is not None and own > -0.20 and delta - own <= -0.05:
            sensitivities.append(
                {
                    "model": model,
                    "domain": domain,
                    "mode": mode,
                    "seed": seed,
                    "class": cls,
                    "median_target_window_delta": delta,
                    "median_source_window_delta": own,
                    "requires_effective_change_and_real_input_association_review": True,
                }
            )
    native = read_json(store.root / "native_and_decision_metrics.json")
    decisions = []
    for task in tasks:
        model, source = task["id"], task["source"]
        target = "dales" if source == "eclair" else "eclair"
        original_source = canonical[(model, source)]["summary"]
        restricted_source = native[model + "/" + source]["restricted_outputs"]
        guard = (
            restricted_source["all_seven_miou"] >= original_source["all_seven_miou"] - 0.005
            and min(
                restricted_source["per_class_iou"][c] - original_source["per_class_iou"][c] for c in COMMON_NAMES[1:]
            )
            >= -0.02
        )
        if guard:
            original, restricted = (
                canonical[(model, target)]["summary"],
                native[model + "/" + target]["restricted_outputs"],
            )
            for cls in COMMON_NAMES[1:]:
                delta = restricted["per_class_iou"][cls] - original["per_class_iou"][cls]
                if delta >= 0.02:
                    decisions.append({"model": model, "target": target, "class": cls, "delta_iou_pp": delta * 100})
    boundary = []
    with (store.root / "inference_boundaries.csv").open(newline="") as stream:
        edge_rows = list(csv.DictReader(stream))
    edge_groups = defaultdict(dict)
    for row in edge_rows:
        edge_groups[(row["model"], row["domain"], row["class"], row["edge_definition"])][int(row["edge_bin"])] = row
    for (model, domain, cls, definition), values in edge_groups.items():
        if domain == sources[model] or not all(b in values and int(values[b]["support"]) >= 32 for b in (0, 3)):
            continue
        gap = float(values[3]["recall"]) - float(values[0]["recall"])
        if gap >= 0.10:
            boundary.append(
                {
                    "model": model,
                    "target": domain,
                    "class": cls,
                    "edge_definition": definition,
                    "interior_minus_edge_recall_pp": 100 * gap,
                }
            )
    cards = {
        "coverage_or_changed_geometry": {
            "signatures": coverage,
            "limitation": "Descriptor mismatch combines shape and acquisition; cannot establish absent semantic subtypes.",
        },
        "acquisition_cue_sensitivity": {
            "signatures": sensitivities,
            "limitation": "Window medians are descriptive; overlap is possible. "
            "Destructive probes are not sensor simulators.",
        },
        "supported_input_feature_learning": {
            "signatures": feature,
            "limitation": "Class-centre distance cannot distinguish architecture, objective, normalization or decision learning.",
        },
        "label_and_output_competition": {
            "signatures": decisions,
            "limitation": "A changed decision rule is not a retrained loss or label-contract control.",
        },
        "inference_boundary_processing": {
            "signatures": boundary,
            "limitation": "Read boundary/split strata; error concentration alone cannot establish causal context loss.",
        },
    }
    for item in cards.values():
        item["status"] = (
            "measured lead requiring counterevidence review"
            if item["signatures"]
            else "unresolved; no qualifying screen"
        )
    store.json(
        "direction_evidence.json",
        {
            "cards": cards,
            "is_causal_ranking": False,
            "screen_thresholds": "coverage error difference 5pp with32 samples per group; "
            "probe target-source median difference5pp "
            "and source drop<20pp; output-rule class gain2pp with source all-class guards",
            "next_step": "Review exact class signatures and counterevidence, then shortlist up to three directions. "
            "No automatic training.",
        },
    )
    return cards


def build_decision_report(store, tasks, canonical, support, features, probes, measures, cfg):
    lines = [
        "# Transfer evidence and research decisions",
        "",
        "## Scope and validity",
        "",
        "- Ten existing source-selected checkpoints; both test domains; no training or target fitting.",
        "- All 20 canonical integer confusion matrices reproduced exactly. All-class results remain primary.",
        "- Historical references are tuned project models, not untouched ECLAIR-paper baselines.",
        "- Sampling is point-based. Geometry comparisons and frozen probes are descriptive, not causal attribution.",
    ]
    for task in tasks:
        if task.get("selection_warning"):
            lines.append(f"- Provenance warning for {task['id']}: {task['selection_warning']}")
    lines += [
        "",
        "## Full results",
        "",
        "| Model | Source test mIoU | Target test mIoU | Target wire/pole mean |",
        "|---|---:|---:|---:|",
    ]
    for task in tasks:
        source, target = task["source"], "eclair" if task["source"] == "dales" else "dales"
        a, b = canonical[(task["id"], source)]["summary"], canonical[(task["id"], target)]["summary"]
        lines.append(
            f"| {task['id']} | {100 * a['all_seven_miou']:.2f} | {100 * b['all_seven_miou']:.2f} | "
            f"{100 * b['utility_miou']:.2f} |"
        )
    lines += [
        "",
        "## Error-linked evidence",
        "",
        "The following contrasts use the preselected probability sample. "
        "A source-support descriptor includes density and local shape; it is not a semantic inventory of asset subtypes.",
        "",
        "| Model → dataset / class / radius | Outside source descriptor support | Error inside | Error outside |",
        "|---|---:|---:|---:|",
    ]

    def pct(value):
        return "unavailable" if value is None else f"{100 * value:.1f}%"

    for r in support:
        if next(t["source"] for t in tasks if t["id"] == r["model"]) == r["domain"]:
            continue
        lines.append(
            f"| {r['model']} → {r['domain']} / {r['class']} / {r['radius_m']}m | {pct(r['outside_support_fraction'])} | "
            f"{pct(r['error_inside_support'])} | {pct(r['error_outside_support'])} |"
        )
    cards = evidence_cards(store, tasks, canonical, support, features, probes)
    lines += [
        "",
        "## Research-direction evidence cards",
        "",
        "These declared descriptive screens are not significance tests or a causal ranking. "
        "No direction is selected merely because it appeared in the original plan.",
        "",
        "| Direction | Qualifying contrasts | Status |",
        "|---|---:|---|",
    ]
    for name, item in cards.items():
        lines.append(f"| {name.replace('_', ' ')} | {len(item['signatures'])} | {item['status']} |")
    lines += [
        "",
        "Review the class-specific contrasts and counterevidence in `direction_evidence.json`, "
        "including probe validity and label/decision alternatives, before shortlisting up to three training directions.",
    ]
    lines += [
        "",
        "## Interpretation of the three primary questions",
        "",
        "The shortlist remains a scientific review of actual results, rather than three preselected remedies.",
        "",
        "- **Coverage and acquisition compatibility.** Review the separate2m and broad-scale support/error contrasts, "
        "class composition, "
        "return/density strata and subtype records before separating missing examples from changed acquisition cues.",
        "- **Input-cue robustness.** Review `paired_probe_results.csv` alongside `probe_validity.csv` and the clean "
        "source/target strata. Prioritize only vulnerabilities associated with actual transfer errors. A destructive "
        "shuffle or thinning drop alone is not evidence for a new augmentation recipe.",
        "- **Feature learning or processing on supported examples.** Compare target feature/error associations "
        "with own-domain controls. Inspect supported target errors "
        "and label/decision alternatives before choosing an architecture or objective.",
        "",
        "## Competing explanations and unresolved questions",
        "",
        "- `native_and_decision_metrics.json` separates native subtype confusions, ignored-output competition and "
        "alternative decision rules; neither alternative changes the canonical scores.",
        "- `model_error_agreement.json` identifies shared failures and method-specific rescues. Historical method "
        "differences do not isolate their training intervention or establish seed repeatability.",
        "- `inference_boundaries.csv` and split-depth matrices test concentration near actual inference boundaries. "
        "Their association does not establish missing scene context during training.",
        "- `sampled_subtypes.csv` and the review gallery support manual label/morphology review. DALES supplies no "
        "equivalent tower subtype label; absence of that label cannot establish absence of tower-like training shapes.",
        "- No result here identifies focal/Lovasz, normalization, a backbone family or a newly added feature as a "
        "proven remedy. Those variables need trained, source-competent controls after problem-fit review.",
        "- Neighbourhoods with insufficient points, tile-edge truncation and inadequate feature-bank coverage are "
        "reported as unavailable evidence, not successful controls.",
        "",
        "## Output map",
        "",
        "- `measurement_coverage.json`: required checkpoints, classes, geometry and probe completion.",
        "- `class_deltas.csv`, `tile_bootstrap.json`: every class tradeoff and tile-based uncertainty.",
        "- `error_conditioned_inputs.csv`, `pooled_input_features.csv`, `conditional_error_strata.csv`: "
        "exact inputs versus errors.",
        "- `source_support.csv`, `feature_support.csv`, `error_conditioned_geometry.csv`: support and error comparisons.",
        "- `gallery/review.html`: illustrative, labelled local context; raw identities retained.",
        "",
        "## Decision boundary",
        "",
        "Do not spend a training run from a marginal distribution shift or a feature-distance "
        "plot alone. Select a candidate only when the affected classes, measured failure signature, competing explanation, "
        "proposed mechanism and all-class/source-competence contrast are explicit. These datasets have already informed "
        "development; they are not untouched future-domain evidence.",
    ]
    measures["limitations"].extend(
        [
            "point samples are not asset instances",
            "source tower subtype unlabelled",
            "observational associations are not causal effects",
            "historical selection gap: DALES Mix3D38163",
            "two development datasets; no multi-seed confirmation",
        ]
    )
    store.text("Research_Decision_Report.md", "\n".join(lines) + "\n")
