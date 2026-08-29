#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_loader import load_yaml
from src.dg_audit import (
    audit_bev_projection,
    audit_mix3d_boundary,
    audit_occupancy_perturbations,
    occupancy_summary,
)
from src.mix3d import ALSMix3DConfig, stable_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-pass BAB, occupancy-consistency, and BEV feasibility audit"
    )
    p.add_argument("--audit-config", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--validate-only", action="store_true")
    return p.parse_args()


def _finite_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio when defined, otherwise NaN for JSON null conversion."""
    denominator = float(denominator)
    if denominator == 0.0:
        return float("nan")
    return float(numerator) / denominator


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_finite_json(value), indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key))
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fields:
                value = _finite_json(row.get(key))
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, separators=(",", ":"), sort_keys=True)
                clean[key] = "" if value is None else value
            writer.writerow(clean)


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def _audit_cfg(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError("audit config must contain a YAML mapping")
    return cfg


def _build_dataset(training_cfg: Dict[str, Any]):
    # Keep the CLI's --help/--validate-only paths usable on login nodes where
    # the CUDA/Minkowski training environment is not loaded.
    from src.dist import DistEnv
    from train import build_dataloaders

    cfg = copy.deepcopy(training_cfg)
    data = cfg.setdefault("data", {})
    data["num_workers"] = 0
    data["batch_size"] = 1
    data.setdefault("mix3d", {})["enabled"] = False
    cfg.setdefault("model", {}).setdefault("aux_heads", {}).setdefault("bev", {})[
        "enabled"
    ] = False
    if str(data.get("dataset", "")).lower() == "eclair":
        # The audit samples base tiles exactly once; weighted repetition is irrelevant.
        data["sampling"] = {"mode": "tiles"}
    train_loader, _, _ = build_dataloaders(
        cfg, DistEnv(enabled=False), eclair_run_cache_dir=None
    )
    return train_loader.dataset


def _sample_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    count = max(1, int(count))
    if count <= int(length):
        return np.linspace(0, length - 1, count, dtype=np.int64).tolist()

    # Cover every source file before repeating deterministic indices. Repeated
    # DALES indices still produce different crops because sample_i contributes
    # to the sample seed.
    full_cycles, remainder = divmod(count, int(length))
    indices = list(range(int(length))) * int(full_cycles)
    if remainder:
        indices.extend(
            np.linspace(0, length - 1, remainder, dtype=np.int64).tolist()
        )
    return indices


def _stratified_class_presence_specs(
    class_counts: np.ndarray,
    *,
    class_ids: Sequence[int],
    uniform_samples: int,
    samples_per_class: int,
    support_bins: int,
    minimum_total_unique: int,
    seed: int,
) -> list[Dict[str, Any]]:
    """Build deterministic, deduplicated audit samples with rare-class quotas.

    The returned records describe unique host tiles. A tile may belong to the
    unbiased ``uniform`` stratum and to one or more ``class_<id>`` strata. The
    class quotas are filled across low/medium/high support ranks rather than by
    taking only the richest tiles. ``uniform_fill`` guarantees a useful total
    sample count but must not be treated as an unbiased population stratum.
    """
    counts = np.asarray(class_counts, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[0] <= 0 or counts.shape[1] <= 0:
        raise ValueError("class_counts must have shape [tiles, classes].")
    if np.any(counts < 0):
        raise ValueError("class_counts must be non-negative.")

    n_tiles, n_classes = counts.shape
    classes = tuple(dict.fromkeys(int(v) for v in class_ids))
    if not classes:
        raise ValueError("stratified sampling requires at least one class id.")
    bad = [c for c in classes if c < 0 or c >= n_classes]
    if bad:
        raise ValueError(f"stratified class ids outside [0,{n_classes}): {bad}")

    uniform_n = min(n_tiles, max(0, int(uniform_samples)))
    per_class_n = max(0, int(samples_per_class))
    n_bins = max(1, int(support_bins))
    minimum_n = min(n_tiles, max(0, int(minimum_total_unique)))

    # Dict insertion order is the deterministic execution order.
    selected: Dict[int, set[str]] = {}

    def add(index: int, stratum: str) -> None:
        selected.setdefault(int(index), set()).add(str(stratum))

    for index in _sample_indices(n_tiles, uniform_n):
        add(index, "uniform")

    for class_id in classes:
        eligible = np.flatnonzero(counts[:, class_id] > 0).astype(np.int64)
        # Any already-selected eligible tile counts toward this class quota.
        already = [idx for idx in selected if counts[idx, class_id] > 0]
        for index in already:
            add(index, f"class_{class_id}")

        need = max(0, min(per_class_n, int(eligible.size)) - len(already))
        if need == 0:
            continue

        available = np.asarray(
            [int(idx) for idx in eligible if int(idx) not in selected],
            dtype=np.int64,
        )
        if available.size == 0:
            continue

        # Rank by log support and split the rank order into equal-sized bins.
        # This samples small, medium, and large structures without thresholds
        # tied to one sensor's raw point density.
        support = np.log1p(counts[available, class_id].astype(np.float64))
        rank_order = np.lexsort((available, support))
        ranked = available[rank_order]
        bins = [b.copy() for b in np.array_split(ranked, min(n_bins, ranked.size)) if b.size]
        rng = np.random.default_rng(stable_seed(seed, "audit_stratum", class_id))
        for bi, values in enumerate(bins):
            bins[bi] = values[rng.permutation(values.size)]

        # Round-robin across support bins so no one support regime dominates.
        cursors = [0] * len(bins)
        chosen: list[int] = []
        while len(chosen) < need:
            progressed = False
            for bi, values in enumerate(bins):
                if cursors[bi] >= values.size:
                    continue
                chosen.append(int(values[cursors[bi]]))
                cursors[bi] += 1
                progressed = True
                if len(chosen) >= need:
                    break
            if not progressed:
                break
        for index in chosen:
            add(index, f"class_{class_id}")

    # A tile chosen for one class may contain other requested classes too.
    # Record every true class-stratum membership after selection.
    for index in list(selected):
        for class_id in classes:
            if counts[index, class_id] > 0:
                add(index, f"class_{class_id}")

    if len(selected) < minimum_n:
        remaining = np.asarray(
            [idx for idx in range(n_tiles) if idx not in selected], dtype=np.int64
        )
        rng = np.random.default_rng(stable_seed(seed, "audit_uniform_fill"))
        if remaining.size:
            remaining = remaining[rng.permutation(remaining.size)]
        for index in remaining[: minimum_n - len(selected)]:
            add(int(index), "uniform_fill")
            for class_id in classes:
                if counts[int(index), class_id] > 0:
                    add(int(index), f"class_{class_id}")

    def stratum_key(value: str) -> tuple[int, int | str]:
        if value == "uniform":
            return (0, 0)
        if value.startswith("class_"):
            return (1, int(value.split("_", 1)[1]))
        return (2, value)

    specs: list[Dict[str, Any]] = []
    for index, strata in selected.items():
        class_strata = [c for c in classes if f"class_{c}" in strata]
        specs.append(
            {
                "host_index": int(index),
                "sampling_mode": "stratified_class_presence",
                "sampling_strata": sorted(strata, key=stratum_key),
                "stratum_class_ids": class_strata,
                "selection_class_point_counts": {
                    str(c): int(counts[index, c]) for c in classes
                },
                "is_uniform_sample": int("uniform" in strata),
            }
        )
    return specs


def _uniform_sample_specs(length: int, count: int) -> list[Dict[str, Any]]:
    return [
        {
            "host_index": int(index),
            "sampling_mode": "uniform",
            "sampling_strata": ["uniform"],
            "stratum_class_ids": [],
            "selection_class_point_counts": {},
            "is_uniform_sample": 1,
        }
        for index in _sample_indices(length, count)
    ]


def _sampling_overview(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    strata: Dict[str, set[str]] = {}
    for record in records:
        host = str(record["host_source"])
        for stratum in record.get("sampling_strata", ["uniform"]):
            strata.setdefault(str(stratum), set()).add(host)
    modes = sorted({str(r.get("sampling_mode", "uniform")) for r in records})
    stratified = modes != ["uniform"]
    return {
        "modes": modes,
        "total_selected_records": len(records),
        "unique_host_sources": len({str(r["host_source"]) for r in records}),
        "stratum_unique_hosts": {k: len(v) for k, v in sorted(strata.items())},
        "aggregate_scope": (
            "all_selected_stratified_records_not_population_unbiased"
            if stratified
            else "uniform_records"
        ),
        "population_estimate_stratum": "uniform",
    }


def _eclair_class_counts_for_audit(ds: Any, cache_path: Path) -> np.ndarray:
    """Load/reuse an audit-local ECLAIR tile-by-class count matrix."""
    expected_names = np.asarray([str(v) for v in ds.names])
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            counts = np.asarray(cached["counts"], dtype=np.int64)
            names = np.asarray(cached["names"]).astype(str)
        if counts.shape == (len(ds.names), int(ds.num_classes)) and np.array_equal(
            names, expected_names
        ):
            return counts

    counts = np.asarray(ds._load_or_compute_label_counts(), dtype=np.int64)
    if counts.shape != (len(ds.names), int(ds.num_classes)):
        raise RuntimeError(
            "Unexpected ECLAIR class-count shape: "
            f"{counts.shape}, expected {(len(ds.names), int(ds.num_classes))}."
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, counts=counts, names=expected_names)
    tmp.replace(cache_path)
    return counts


def _different_index(length: int, host_idx: int, rng: np.random.Generator) -> int:
    if length < 2:
        raise RuntimeError("candidate audit needs at least two source files")
    j = int(rng.integers(0, length - 1))
    return j + int(j >= int(host_idx))


def _eclair_pair(ds, host_idx: int, donor_idx: int, seed: int):
    host_name = ds.names[int(host_idx)]
    donor_name = ds.names[int(donor_idx)]
    host, _ = ds._prepare_training_payload(fname=host_name, sample_seed=int(seed))
    donor, _ = ds._prepare_training_payload(
        fname=donor_name,
        sample_seed=stable_seed(seed, host_name, donor_name, "audit_donor"),
    )
    return host, donor, str(host_name), str(donor_name)


def _dales_payload(ds, index: int, seed: int):
    path = ds.files[int(index)]
    raw, xyz, intensity = ds._load_mix_tile(path)
    rep = int(seed) % max(1, int(ds.crop_cfg.crops_per_tile_per_epoch))
    _, payload, _ = ds._sample_mix_crop(
        path=path,
        tile_i=int(index),
        rep_i=rep,
        raw=raw,
        xyz=xyz,
        intensity_scaled=intensity,
        sample_seed_override=int(seed),
    )
    return payload, str(path)


def _pair(ds, dataset: str, host_idx: int, donor_idx: int, seed: int):
    if dataset == "eclair":
        return _eclair_pair(ds, host_idx, donor_idx, seed)
    if dataset == "dales":
        host, host_name = _dales_payload(ds, host_idx, seed)
        donor, donor_name = _dales_payload(
            ds, donor_idx, stable_seed(seed, host_name, donor_idx, "audit_donor")
        )
        return host, donor, host_name, donor_name
    raise ValueError(dataset)


def _projector_bounds(training_cfg: Mapping[str, Any]) -> tuple[float, ...]:
    proj = (
        (((training_cfg.get("model", {}) or {}).get("aux_heads", {}) or {}).get("bev", {}) or {}).get(
            "projector", {}
        )
        or {}
    )
    return (
        float(proj.get("x_min_m", 0.0)),
        float(proj.get("x_max_m", 100.0)),
        float(proj.get("y_min_m", 0.0)),
        float(proj.get("y_max_m", 100.0)),
        float(proj.get("z_min_m", -100.0)),
        float(proj.get("z_max_m", 200.0)),
    )


def _candidate_projector_bounds(
    bev_cfg: Mapping[str, Any], dataset: str
) -> tuple[float, ...]:
    candidate = bev_cfg.get("candidate_projection", {}) or {}
    per_dataset = candidate.get("bounds_by_dataset", {}) or {}
    bounds = per_dataset.get(str(dataset), {}) or {}
    required = ("x_min_m", "x_max_m", "y_min_m", "y_max_m")
    missing = [key for key in required if key not in bounds]
    if missing:
        raise ValueError(
            f"bev.candidate_projection.bounds_by_dataset.{dataset} missing {missing}"
        )
    return (
        float(bounds["x_min_m"]),
        float(bounds["x_max_m"]),
        float(bounds["y_min_m"]),
        float(bounds["y_max_m"]),
        float(bounds.get("z_min_m", -1.0)),
        float(bounds.get("z_max_m", 1.0)),
    )


def _median(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(vals)) if vals else None


def _min(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return min(vals) if vals else None


def _summarize_dataset(
    dataset: str,
    records: Sequence[Mapping[str, Any]],
    *,
    utility_ids: Sequence[int],
    bev_static_map_risk: bool,
    gates: Mapping[str, Any],
) -> Dict[str, Any]:
    class_support = []
    if records and records[0].get("occupancy"):
        num_classes = len(records[0]["occupancy"][0]["point_class_counts"])
        for class_id in range(num_classes):
            point_presence = 0
            voxel_presence = 0
            point_total = 0
            voxel_total = 0
            for rec in records:
                point_row = rec["occupancy"][0]
                input_row = min(
                    rec["occupancy"],
                    key=lambda row: abs(
                        float(row["voxel_m"]) - float(rec["input_voxel_m"])
                    ),
                )
                points = int(point_row["point_class_counts"][class_id])
                voxels = int(input_row["voxel_class_counts"][class_id])
                point_total += points
                voxel_total += voxels
                point_presence += int(points > 0)
                voxel_presence += int(voxels > 0)
            class_support.append(
                {
                    "class_id": class_id,
                    "samples_with_points": point_presence,
                    "samples_with_input_voxels": voxel_presence,
                    "total_points": point_total,
                    "total_input_voxels": voxel_total,
                    "is_utility": class_id in {int(v) for v in utility_ids},
                }
            )

    bab = [r["bab"] for r in records]
    applied = [r for r in bab if int(r.get("applied", 0)) == 1]
    bab_summary = {
        "applied_rate": len(applied) / max(1, len(bab)),
        "median_cut_component_fraction": _median(
            r.get("cut_component_fraction_all") for r in applied
        ),
        "median_removed_vs_inserted_class_js": _median(
            r.get("removed_vs_inserted_class_js") for r in applied
        ),
    }
    seam_width = float(gates.get("bab_seam_width_m", 1.0))
    seam_rows = [
        s
        for r in applied
        for s in r.get("seam_bands", [])
        if abs(float(s["width_m"]) - seam_width) < 1e-9
    ]
    bab_summary.update(
        {
            "decision_seam_width_m": seam_width,
            "median_cross_xy_nn_m": _median(s.get("cross_xy_nn_median_m") for s in seam_rows),
            "median_cross_abs_dz_m": _median(s.get("cross_abs_dz_median_m") for s in seam_rows),
            "median_cross_label_mismatch_fraction": _median(
                s.get("cross_label_mismatch_fraction") for s in seam_rows
            ),
        }
    )
    cut_gate = float(gates.get("bab_cut_component_fraction", 0.10))
    bab_summary["bab_worth_training"] = bool(
        bab_summary["applied_rate"] >= float(gates.get("bab_min_applied_rate", 0.75))
        and bab_summary["median_cut_component_fraction"] is not None
        and bab_summary["median_cut_component_fraction"] >= cut_gate
    )
    bab_summary["gate_note"] = (
        "A positive gate shows that semantic component cutting is frequent; it does not prove "
        "that current prediction errors are seam-caused."
    )

    grouped: Dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for rec in records:
        for row in rec["ocons"]:
            grouped.setdefault((str(row["method"]), float(row["strength"])), []).append(row)
    ocons_summary = []
    for (method, strength), rows in sorted(grouped.items()):
        protected_ret = []
        disappeared = 0
        for row in rows:
            disappeared += int(row["protected_classes_disappeared"])
            for pc in row["per_class"]:
                if int(pc["class_id"]) in {int(c) for c in utility_ids}:
                    protected_ret.append(pc["active_voxel_retention"])
        item = {
            "method": method,
            "strength": strength,
            "median_active_voxel_jaccard": _median(r["active_voxel_jaccard"] for r in rows),
            "median_changed_active_voxel_fraction": _median(
                r["changed_active_voxel_fraction"] for r in rows
            ),
            "minimum_utility_active_voxel_retention": _min(protected_ret),
            "protected_disappearances": disappeared,
        }
        lo = float(gates.get("ocons_changed_voxel_fraction_min", 0.08))
        hi = float(gates.get("ocons_changed_voxel_fraction_max", 0.35))
        retention = float(gates.get("ocons_utility_retention_min", 0.70))
        changed = item["median_changed_active_voxel_fraction"]
        item["eligible"] = bool(
            method == "active_voxel_mask"
            and changed is not None
            and lo <= changed <= hi
            and item["minimum_utility_active_voxel_retention"] is not None
            and item["minimum_utility_active_voxel_retention"] >= retention
            and disappeared == 0
        )
        ocons_summary.append(item)

    bev_grouped: Dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for rec in records:
        for row in rec["bev"]:
            key = (str(row["projection_variant"]), float(row["resolution_m"]))
            bev_grouped.setdefault(key, []).append(row)
    bev_summary = []
    for (projection_variant, resolution), rows in sorted(bev_grouped.items()):
        policy_rows = []
        for row in rows:
            for policy, values in row["policies"].items():
                policy_rows.append((policy, values))
        policies = {}
        for policy in sorted({p for p, _ in policy_rows}):
            vals = [v for p, v in policy_rows if p == policy]
            policies[policy] = {
                "median_utility_min_retention": _median(
                    v.get("utility_min_retention") for v in vals
                )
            }
        item = {
            "projection_variant": projection_variant,
            "resolution_m": resolution,
            "xy_frame": str(rows[0]["xy_frame"]),
            "z_filter": str(rows[0]["z_filter"]),
            "height_slicing": str(rows[0]["height_slicing"]),
            "median_in_bounds_active_voxel_fraction": _median(
                r["in_bounds_active_voxel_fraction"] for r in rows
            ),
            "median_multi_class_cell_fraction": _median(
                r["multi_class_cell_fraction"] for r in rows
            ),
            "median_z_sliced_multi_class_cell_fraction": _median(
                r["z_sliced_multi_class_cell_fraction"] for r in rows
            ),
            "policies": policies,
            "input_index_to_deep_feature_map_risk": bool(bev_static_map_risk),
        }
        bounds_ok = (
            item["median_in_bounds_active_voxel_fraction"] is not None
            and item["median_in_bounds_active_voxel_fraction"]
            >= float(gates.get("bev_in_bounds_fraction_min", 0.95))
        )
        best_single = max(
            (v["median_utility_min_retention"] for v in policies.values() if v["median_utility_min_retention"] is not None),
            default=None,
        )
        single_ok = best_single is not None and best_single >= float(
            gates.get("bev_utility_retention_min", 0.90)
        )
        item["utility_evidence_available"] = best_single is not None
        item["single_label_training_ready"] = bool(
            bounds_ok and single_ok and not bev_static_map_risk
        )
        item["multi_label_recommended"] = (
            None if best_single is None else bool(not single_ok)
        )
        bev_summary.append(item)

    return {
        "dataset": dataset,
        "samples": len(records),
        "unique_host_sources": len({str(r["host_source"]) for r in records}),
        "class_support": class_support,
        "bab": bab_summary,
        "ocons": ocons_summary,
        "bev": bev_summary,
        "recommended_immediate_training_candidates": {
            "bab": bool(bab_summary["bab_worth_training"]),
            "ocons": any(bool(r["eligible"]) for r in ocons_summary),
            "bev_current_implementation": any(
                bool(r["single_label_training_ready"])
                for r in bev_summary
                if r["projection_variant"] == "configured_current"
            ),
            "bev_candidate_projection": any(
                bool(r["single_label_training_ready"])
                for r in bev_summary
                if r["projection_variant"] == "centered_candidate"
            ),
        },
    }


def _cross_dataset_occupancy(records: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    grouped: Dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for rec in records:
        for row in rec["occupancy"]:
            grouped.setdefault((str(rec["dataset"]), float(row["voxel_m"])), []).append(row)
    datasets = sorted({k[0] for k in grouped})
    sizes = sorted({k[1] for k in grouped})
    out = []
    for size in sizes:
        row: Dict[str, Any] = {"voxel_m": size}
        for dataset in datasets:
            vals = grouped.get((dataset, size), [])
            row[dataset] = {
                "median_active_voxels_per_1000_points": _median(
                    v["active_voxels_per_1000_points"] for v in vals
                ),
                "median_points_per_active_voxel": _median(
                    v["points_per_active_voxel_mean"] for v in vals
                ),
                "median_singleton_voxel_fraction": _median(
                    v["singleton_voxel_fraction"] for v in vals
                ),
            }
        if len(datasets) == 2:
            a, b = datasets
            av = row[a]["median_active_voxels_per_1000_points"]
            bv = row[b]["median_active_voxels_per_1000_points"]
            row["active_voxels_per_point_ratio"] = None if av is None or bv is None else _safe_ratio(av, bv)
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    audit_path = args.audit_config.resolve()
    audit = _audit_cfg(audit_path)
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else _resolve(audit.get("out_dir", "runs/dg_candidate_audit"), repo_root)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    training_paths = audit.get("training_configs", [])
    if not training_paths:
        raise ValueError("audit config needs training_configs")
    training_paths = [_resolve(p, repo_root) for p in training_paths]
    for path in training_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.validate_only:
        print(json.dumps({"status": "ok", "training_configs": [str(p) for p in training_paths]}, indent=2))
        return

    seed = int(audit.get("seed", 1337))
    sample_count = int(args.samples or audit.get("samples_per_dataset", 6))
    voxel_sizes = [float(v) for v in audit.get("shared_voxel_sizes_m", [0.2, 0.5, 1.0])]
    bab_cfg = audit.get("bab", {}) or {}
    occ_cfg = audit.get("ocons", {}) or {}
    bev_cfg = audit.get("bev", {}) or {}
    gates = audit.get("decision_gates", {}) or {}
    utility_by_dataset = audit.get("utility_class_ids", {}) or {}
    sampling_by_dataset = ((audit.get("sampling", {}) or {}).get("by_dataset", {}) or {})

    resolved = copy.deepcopy(audit)
    resolved["training_configs"] = [str(p) for p in training_paths]
    resolved["out_dir"] = str(out_dir)
    resolved["samples_per_dataset"] = sample_count
    resolved["samples_argument_scope"] = (
        "uniform mode only; stratified modes use sampling.by_dataset quotas"
    )
    _write_json(out_dir / "config_resolved.json", resolved)

    all_records: list[Dict[str, Any]] = []
    summaries = []
    stratum_summaries = []
    for config_path in training_paths:
        training_cfg = load_yaml(config_path)
        dataset = str(training_cfg["data"]["dataset"]).lower()
        print(f"[audit] building {dataset} dataset from {config_path}", flush=True)
        ds = _build_dataset(training_cfg)
        length = len(ds.names) if dataset == "eclair" else len(ds.files)
        dataset_sampling = sampling_by_dataset.get(dataset, {}) or {}
        sampling_mode = str(dataset_sampling.get("mode", "uniform")).lower()
        if sampling_mode == "uniform":
            sample_specs = _uniform_sample_specs(length, sample_count)
        elif sampling_mode == "stratified_class_presence":
            if dataset != "eclair":
                raise ValueError(
                    "stratified_class_presence is currently implemented only for ECLAIR."
                )
            if not bool(dataset_sampling.get("deduplicate_tiles", True)):
                raise ValueError("The decision audit requires deduplicate_tiles=true.")
            class_counts = _eclair_class_counts_for_audit(
                ds, out_dir / "eclair_tile_class_counts.npz"
            )
            sample_specs = _stratified_class_presence_specs(
                class_counts,
                class_ids=[int(v) for v in dataset_sampling.get("class_ids", [5, 6, 7])],
                uniform_samples=int(dataset_sampling.get("uniform_samples", 25)),
                samples_per_class=int(dataset_sampling.get("samples_per_class", 30)),
                support_bins=int(dataset_sampling.get("support_bins", 3)),
                minimum_total_unique=int(dataset_sampling.get("minimum_total_unique", 75)),
                seed=stable_seed(seed, dataset, "stratified_audit"),
            )
            overview: Dict[str, int] = {}
            for spec in sample_specs:
                for stratum in spec["sampling_strata"]:
                    overview[str(stratum)] = overview.get(str(stratum), 0) + 1
            print(
                f"[audit] {dataset} stratified selection unique={len(sample_specs)} "
                f"strata={json.dumps(overview, sort_keys=True)}",
                flush=True,
            )
        else:
            raise ValueError(f"Unknown audit sampling mode for {dataset}: {sampling_mode}")
        ls = training_cfg["data"]["label_space"]
        num_classes = int(ls["num_classes"])
        ignore_index = int(ls["ignore_index"])
        patch = training_cfg["data"]["patch"]
        input_voxel_m = float(patch["voxel_size"]) * float(patch["coord_norm_factor"])
        mix_cfg = ALSMix3DConfig.from_cfg(training_cfg["data"].get("mix3d", {}))
        utility_ids = [int(c) for c in utility_by_dataset.get(dataset, [])]
        if not utility_ids:
            utility_ids = [5, 6] if dataset == "dales" else [5, 6, 7]
        bounds = _projector_bounds(training_cfg)
        candidate_bev_cfg = bev_cfg.get("candidate_projection", {}) or {}
        candidate_bev_enabled = bool(candidate_bev_cfg.get("enabled", True))
        candidate_bounds = (
            _candidate_projector_bounds(bev_cfg, dataset)
            if candidate_bev_enabled
            else None
        )
        aux_bev = (((training_cfg.get("model", {}) or {}).get("aux_heads", {}) or {}).get("bev", {}) or {})
        levels = [str(v) for v in aux_bev.get("levels", ["block8"])]
        mode = str((aux_bev.get("projector", {}) or {}).get("mode", "pool")).lower()
        # Dataset maps index input voxels; block5..8 contain different active-site arrays.
        static_map_risk = mode == "select" and any(level.startswith("block") for level in levels)

        dataset_records: list[Dict[str, Any]] = []
        for sample_i, sample_spec in enumerate(sample_specs):
            host_idx = int(sample_spec["host_index"])
            sample_seed = stable_seed(seed, dataset, sample_i, host_idx, "candidate_audit")
            rng = np.random.default_rng(sample_seed)
            donor_idx = _different_index(length, host_idx, rng)
            print(
                f"[audit] {dataset} sample {sample_i + 1}/{len(sample_specs)} "
                f"host={host_idx} donor={donor_idx} "
                f"strata={'+'.join(sample_spec['sampling_strata'])}",
                flush=True,
            )
            host, donor, host_name, donor_name = _pair(
                ds, dataset, host_idx, donor_idx, sample_seed
            )
            current_bev = audit_bev_projection(
                host,
                input_voxel_m=input_voxel_m,
                resolutions_m=[float(v) for v in bev_cfg.get("resolutions_m", [0.2, 0.5, 1.0])],
                bounds_xyz_m=bounds,
                num_classes=num_classes,
                ignore_index=ignore_index,
                utility_class_ids=utility_ids,
                z_slices=int(bev_cfg.get("z_slices", 4)),
                hidden_dim=int(aux_bev.get("hidden_dim", 64)),
                xy_frame="native",
                z_filter="bounds",
                height_slicing="fixed_bounds",
            )
            for row in current_bev:
                row["projection_variant"] = "configured_current"

            candidate_bev = []
            if candidate_bev_enabled and candidate_bounds is not None:
                candidate_bev = audit_bev_projection(
                    host,
                    input_voxel_m=input_voxel_m,
                    resolutions_m=[
                        float(v)
                        for v in candidate_bev_cfg.get(
                            "resolutions_m", bev_cfg.get("resolutions_m", [0.2, 0.5, 1.0])
                        )
                    ],
                    bounds_xyz_m=candidate_bounds,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    utility_class_ids=utility_ids,
                    z_slices=int(candidate_bev_cfg.get("z_slices", bev_cfg.get("z_slices", 4))),
                    hidden_dim=int(aux_bev.get("hidden_dim", 64)),
                    xy_frame=str(candidate_bev_cfg.get("xy_frame", "bbox_centered")),
                    z_filter=str(candidate_bev_cfg.get("z_filter", "none")),
                    height_slicing=str(candidate_bev_cfg.get("height_slicing", "quantile")),
                )
                for row in candidate_bev:
                    row["projection_variant"] = "centered_candidate"

            record = {
                "dataset": dataset,
                "sample_index": sample_i,
                "host_index": int(host_idx),
                "donor_index": int(donor_idx),
                "host_source": host_name,
                "donor_source": donor_name,
                "sampling_mode": str(sample_spec["sampling_mode"]),
                "sampling_strata": list(sample_spec["sampling_strata"]),
                "stratum_class_ids": list(sample_spec["stratum_class_ids"]),
                "selection_class_point_counts": dict(
                    sample_spec["selection_class_point_counts"]
                ),
                "is_uniform_sample": int(sample_spec["is_uniform_sample"]),
                "input_voxel_m": input_voxel_m,
                "occupancy": occupancy_summary(
                    host["xyz"],
                    host["y_train"],
                    voxel_sizes_m=voxel_sizes,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ),
                "bab": audit_mix3d_boundary(
                    host,
                    donor,
                    cfg=mix_cfg,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    voxel_edge_m=input_voxel_m,
                    seam_widths_m=[float(v) for v in bab_cfg.get("seam_widths_m", [0.5, 1.0, 2.0])],
                    component_cell_m=float(bab_cfg.get("component_cell_m", 1.0)),
                    rng=np.random.default_rng(stable_seed(sample_seed, "bab")),
                ),
                "ocons": audit_occupancy_perturbations(
                    host,
                    input_voxel_m=input_voxel_m,
                    strengths=[float(v) for v in occ_cfg.get("strengths", [0.1, 0.2, 0.3])],
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    protected_class_ids=utility_ids,
                    protected_min_voxels=int(occ_cfg.get("protected_min_voxels", 3)),
                    component_voxel_m=float(occ_cfg.get("component_voxel_m", 1.0)),
                    seed=stable_seed(sample_seed, "ocons"),
                ),
                "bev": current_bev + candidate_bev,
                "bev_static": {
                    "levels": levels,
                    "projector_mode": mode,
                    "input_index_to_deep_feature_map_risk": static_map_risk,
                    "configured_bounds_xyz_m": list(bounds),
                    "candidate_bounds_xyz_m": (
                        None if candidate_bounds is None else list(candidate_bounds)
                    ),
                },
            }
            dataset_records.append(record)
            all_records.append(record)
        dataset_summary = _summarize_dataset(
            dataset,
            dataset_records,
            utility_ids=utility_ids,
            bev_static_map_risk=static_map_risk,
            gates=gates,
        )
        dataset_summary["sampling"] = _sampling_overview(dataset_records)
        summaries.append(dataset_summary)

        if sampling_mode != "uniform":
            strata = sorted(
                {
                    str(stratum)
                    for record in dataset_records
                    for stratum in record.get("sampling_strata", [])
                }
            )
            for stratum in strata:
                subset = [
                    record
                    for record in dataset_records
                    if stratum in record.get("sampling_strata", [])
                ]
                if not subset:
                    continue
                stratum_summary = _summarize_dataset(
                    dataset,
                    subset,
                    utility_ids=utility_ids,
                    bev_static_map_risk=static_map_risk,
                    gates=gates,
                )
                stratum_summary["stratum"] = stratum
                stratum_summary["interpretation"] = (
                    "population_unbiased_subset"
                    if stratum == "uniform"
                    else "class_enriched_diagnostic_subset"
                )
                stratum_summaries.append(stratum_summary)

    summary_payload = {
        "datasets": summaries,
        "stratum_summaries": stratum_summaries,
        "cross_dataset_occupancy": _cross_dataset_occupancy(all_records),
    }
    _write_json(out_dir / "summary.json", summary_payload)
    with (out_dir / "samples.jsonl").open("w") as f:
        for record in all_records:
            f.write(json.dumps(_finite_json(record), sort_keys=True) + "\n")

    occupancy_rows = []
    bab_rows = []
    bab_seam_rows = []
    bab_class_rows = []
    ocons_rows = []
    ocons_class_rows = []
    bev_rows = []
    bev_class_rows = []
    bev_pair_rows = []
    sampling_rows = []
    class_support_rows = [
        {"dataset": summary["dataset"], **row}
        for summary in summaries
        for row in summary["class_support"]
    ]
    for rec in all_records:
        base = {
            k: rec[k]
            for k in (
                "dataset",
                "sample_index",
                "host_index",
                "donor_index",
                "host_source",
                "donor_source",
                "sampling_mode",
                "sampling_strata",
                "stratum_class_ids",
                "selection_class_point_counts",
                "is_uniform_sample",
            )
        }
        sampling_rows.append(dict(base))
        for row in rec["occupancy"]:
            occupancy_rows.append({**base, **row})
        bab = {k: v for k, v in rec["bab"].items() if k not in ("seam_bands", "component_cuts")}
        bab_rows.append({**base, **bab})
        for row in rec["bab"].get("seam_bands", []):
            bab_seam_rows.append({**base, **row})
        for row in rec["bab"].get("component_cuts", []):
            bab_class_rows.append({**base, **row})
        for row in rec["ocons"]:
            scalar = {k: v for k, v in row.items() if k != "per_class"}
            ocons_rows.append({**base, **scalar})
            for cls in row["per_class"]:
                ocons_class_rows.append(
                    {**base, "method": row["method"], "strength": row["strength"], **cls}
                )
        for row in rec["bev"]:
            scalar = {
                k: v
                for k, v in row.items()
                if k not in ("policies", "top_collision_pairs", "class_in_bounds_retention", "multi_label_class_positive_cells")
            }
            bev_rows.append({**base, **scalar})
            for class_id in range(len(row["class_in_bounds_retention"])):
                class_row = {
                    **base,
                    "projection_variant": row["projection_variant"],
                    "resolution_m": row["resolution_m"],
                    "class_id": class_id,
                    "in_bounds_retention": row["class_in_bounds_retention"][class_id],
                    "multi_label_positive_cells": row["multi_label_class_positive_cells"][class_id],
                }
                for policy, values in row["policies"].items():
                    class_row[f"{policy}_voxel_retention"] = values["class_voxel_retention"][class_id]
                    class_row[f"{policy}_positive_cell_recall"] = values["class_positive_cell_recall"][class_id]
                bev_class_rows.append(class_row)
            for pair in row["top_collision_pairs"]:
                bev_pair_rows.append(
                    {
                        **base,
                        "projection_variant": row["projection_variant"],
                        "resolution_m": row["resolution_m"],
                        **pair,
                    }
                )

    for name, rows in (
        ("occupancy.csv", occupancy_rows),
        ("bab_samples.csv", bab_rows),
        ("bab_seams.csv", bab_seam_rows),
        ("bab_classes.csv", bab_class_rows),
        ("ocons.csv", ocons_rows),
        ("ocons_classes.csv", ocons_class_rows),
        ("bev.csv", bev_rows),
        ("bev_classes.csv", bev_class_rows),
        ("bev_collision_pairs.csv", bev_pair_rows),
        ("class_support.csv", class_support_rows),
        ("sampling_manifest.csv", sampling_rows),
    ):
        _write_csv(out_dir / name, rows)

    print(json.dumps(_finite_json(summary_payload), indent=2), flush=True)
    print(f"[audit] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
