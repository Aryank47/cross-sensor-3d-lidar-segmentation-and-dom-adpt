from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .features import build_features
from .label_maps import eclair_native_to_train_ids
from .sampling_types import COMMON_CLASS_NAMES, DatasetBundle, WindowSpec
from .voxelization import voxelize_from_q

try:
    from scipy.spatial import cKDTree
    from scipy.stats import wasserstein_distance

    SCIPY_AVAILABLE = True
except Exception:
    cKDTree = None
    wasserstein_distance = None
    SCIPY_AVAILABLE = False


def stable_rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "|".join([str(seed), *[str(part) for part in parts]])
    digest = hashlib.sha1(payload.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def summary_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    names = ["mean", "std", "min", "q05", "q25", "median", "q75", "q95", "max"]
    if values.size == 0:
        return {f"{prefix}_{name}": np.nan for name in names}
    q = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    stats = [np.mean(values), np.std(values), np.min(values), *q, np.max(values)]
    return {f"{prefix}_{name}": float(value) for name, value in zip(names, stats)}


def coefficient_of_variation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan
    mean = float(np.mean(values))
    return np.nan if abs(mean) < 1e-12 else float(np.std(values) / mean)


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values[np.isfinite(values)], 0.0, None)
    if values.size == 0:
        return np.nan
    total = float(values.sum())
    if total <= 0:
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float(2 * np.sum(ranks * sorted_values) / (n * total) - (n + 1) / n)


def normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if counts.size <= 1 or total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(p)) / math.log(counts.size))


def build_density_grid(
    xy_window_local: np.ndarray,
    window_size_m: float,
    requested_resolution_m: float,
) -> tuple[np.ndarray, float, float]:
    bins = max(1, int(round(window_size_m / requested_resolution_m)))
    actual_resolution = window_size_m / bins
    counts, _, _ = np.histogram2d(
        xy_window_local[:, 1],
        xy_window_local[:, 0],
        bins=(bins, bins),
        range=((0.0, window_size_m), (0.0, window_size_m)),
    )
    cell_area = actual_resolution * actual_resolution
    return counts.astype(np.int64), actual_resolution, cell_area


def compute_window_row(
    bundle: DatasetBundle,
    spec: WindowSpec,
    xyz_window_tile_local: np.ndarray,
    native_labels: np.ndarray,
    common_labels: np.ndarray,
    return_number: np.ndarray,
    number_of_returns: np.ndarray,
) -> Dict[str, Any]:
    n = int(xyz_window_tile_local.shape[0])
    rn = np.asarray(return_number, dtype=np.int64)
    nor = np.asarray(number_of_returns, dtype=np.int64)
    common = np.asarray(common_labels, dtype=np.int64)
    return {
        "dataset": bundle.name,
        "tile_index": spec.tile_index,
        "tile_id": spec.tile_id,
        "tile_path": spec.tile_path,
        "cohort": spec.cohort,
        "selection_reason": spec.selection_reason,
        "focus_common_id": spec.focus_common_id,
        "window_key": spec.key,
        "window_size_m": spec.window_size_m,
        "x_min_m": spec.x_min_m,
        "y_min_m": spec.y_min_m,
        "area_m2": spec.area_m2,
        "point_count": n,
        "raw_density_pts_m2": n / spec.area_m2,
        "first_return_count": int(np.sum(rn == 1)),
        "first_return_density_pts_m2": float(np.sum(rn == 1) / spec.area_m2),
        "single_return_count": int(np.sum((rn == 1) & (nor == 1))),
        "single_return_density_pts_m2": float(np.sum((rn == 1) & (nor == 1)) / spec.area_m2),
        "later_return_count": int(np.sum(rn > 1)),
        "later_return_density_pts_m2": float(np.sum(rn > 1) / spec.area_m2),
        "later_return_fraction": float(np.mean(rn > 1)),
        "valid_common_fraction": float(np.mean(common != 0)),
        "z_span_m": float(np.ptp(xyz_window_tile_local[:, 2])),
        "z_above_window_min_median": float(np.median(xyz_window_tile_local[:, 2] - xyz_window_tile_local[:, 2].min())),
        "native_label_min": int(np.min(native_labels)),
        "native_label_max": int(np.max(native_labels)),
    }


def compute_grid_rows(
    bundle: DatasetBundle,
    spec: WindowSpec,
    xyz_window_tile_local: np.ndarray,
    return_number: np.ndarray,
    number_of_returns: np.ndarray,
    resolutions_m: Sequence[float],
) -> list[Dict[str, Any]]:
    xy = np.asarray(xyz_window_tile_local[:, :2], dtype=np.float64).copy()
    xy[:, 0] -= spec.x_min_m
    xy[:, 1] -= spec.y_min_m
    rn = np.asarray(return_number, dtype=np.int64)
    nor = np.asarray(number_of_returns, dtype=np.int64)
    subsets = {
        "all_returns": np.ones(xy.shape[0], dtype=bool),
        "first_returns": rn == 1,
        "single_returns": (rn == 1) & (nor == 1),
        "later_returns": rn > 1,
    }
    rows: list[Dict[str, Any]] = []
    for subset_name, mask in subsets.items():
        for requested_res in resolutions_m:
            grid, actual_res, cell_area = build_density_grid(xy[mask], spec.window_size_m, float(requested_res))
            counts = grid.ravel().astype(np.float64)
            density = counts / cell_area
            occupied = counts > 0
            row: Dict[str, Any] = {
                "dataset": bundle.name,
                "tile_index": spec.tile_index,
                "tile_id": spec.tile_id,
                "cohort": spec.cohort,
                "window_key": spec.key,
                "window_size_m": spec.window_size_m,
                "subset": subset_name,
                "grid_resolution_requested_m": float(requested_res),
                "grid_resolution_actual_m": float(actual_res),
                "cell_area_m2": float(cell_area),
                "number_of_cells": int(counts.size),
                "occupied_cells": int(occupied.sum()),
                "occupied_fraction": float(np.mean(occupied)),
                "empty_fraction": float(np.mean(~occupied)),
                "density_cv_all_cells": coefficient_of_variation(density),
                "density_gini_all_cells": gini(density),
                "occupancy_entropy_normalized": normalized_entropy(counts),
            }
            row.update(summary_stats(density, "cell_density_pts_m2"))
            row.update(summary_stats(density[occupied], "occupied_cell_density_pts_m2"))
            rows.append(row)
    return rows


def _subsample_indices(n: int, maximum: int, seed: int, *parts: object) -> np.ndarray:
    if n <= maximum:
        return np.arange(n, dtype=np.int64)
    return stable_rng(seed, *parts).choice(n, size=maximum, replace=False)


def compute_knn_rows(
    bundle: DatasetBundle,
    spec: WindowSpec,
    xyz_window_tile_local: np.ndarray,
    common_labels: np.ndarray,
    return_number: np.ndarray,
    number_of_returns: np.ndarray,
    *,
    k_values: Sequence[int],
    max_points_per_subset: int,
    max_points_per_class: int,
    min_class_points: int,
    class_ids: Sequence[int],
    seed: int,
) -> list[Dict[str, Any]]:
    if not SCIPY_AVAILABLE:
        return []
    xyz = np.asarray(xyz_window_tile_local, dtype=np.float64)
    rn = np.asarray(return_number, dtype=np.int64)
    nor = np.asarray(number_of_returns, dtype=np.int64)
    common = np.asarray(common_labels, dtype=np.int64)
    subsets: list[tuple[str, Optional[int], np.ndarray, int]] = [
        ("all_returns", None, np.ones(xyz.shape[0], dtype=bool), max_points_per_subset),
        ("first_returns", None, rn == 1, max_points_per_subset),
        ("single_returns", None, (rn == 1) & (nor == 1), max_points_per_subset),
    ]
    for class_id in class_ids:
        mask = common == int(class_id)
        if int(mask.sum()) >= min_class_points:
            subsets.append(("common_class", int(class_id), mask, max_points_per_class))

    rows: list[Dict[str, Any]] = []
    for subset_name, class_id, mask, maximum in subsets:
        subset = xyz[mask]
        if subset.shape[0] < 2:
            continue
        indices = _subsample_indices(subset.shape[0], maximum, seed, bundle.name, spec.key, subset_name, class_id)
        points = subset[indices]
        maximum_k = min(max(int(k) for k in k_values), points.shape[0] - 1)
        if maximum_k < 1:
            continue
        for coordinate_space, coordinates in (("xy", points[:, :2]), ("xyz", points)):
            tree = cKDTree(coordinates)
            distances, _ = tree.query(coordinates, k=maximum_k + 1, workers=-1)
            if distances.ndim == 1:
                distances = distances[:, None]
            for k in k_values:
                if int(k) > maximum_k:
                    continue
                values = distances[:, int(k)]
                row: Dict[str, Any] = {
                    "dataset": bundle.name,
                    "tile_index": spec.tile_index,
                    "tile_id": spec.tile_id,
                    "cohort": spec.cohort,
                    "window_key": spec.key,
                    "window_size_m": spec.window_size_m,
                    "subset": subset_name,
                    "common_class_id": class_id,
                    "common_class_name": COMMON_CLASS_NAMES.get(class_id),
                    "coordinate_space": coordinate_space,
                    "k": int(k),
                    "points_available": int(subset.shape[0]),
                    "points_used": int(points.shape[0]),
                }
                row.update(summary_stats(values, "distance_m"))
                rows.append(row)
    return rows


def compute_class_rows(
    bundle: DatasetBundle,
    spec: WindowSpec,
    xyz_window_tile_local: np.ndarray,
    common_labels: np.ndarray,
    *,
    occupied_area_grid_m: float,
) -> list[Dict[str, Any]]:
    xyz = np.asarray(xyz_window_tile_local, dtype=np.float64)
    common = np.asarray(common_labels, dtype=np.int64)
    xy = xyz[:, :2].copy()
    xy[:, 0] -= spec.x_min_m
    xy[:, 1] -= spec.y_min_m
    rows: list[Dict[str, Any]] = []
    for class_id, class_name in COMMON_CLASS_NAMES.items():
        mask = common == class_id
        count = int(mask.sum())
        if count:
            grid, actual_res, cell_area = build_density_grid(xy[mask], spec.window_size_m, occupied_area_grid_m)
            occupied_cells = int(np.sum(grid > 0))
            occupied_area = occupied_cells * cell_area
            z_rel = xyz[mask, 2] - float(xyz[:, 2].min())
        else:
            actual_res = float(occupied_area_grid_m)
            occupied_area = 0.0
            z_rel = np.empty((0,), dtype=np.float64)
        row: Dict[str, Any] = {
            "dataset": bundle.name,
            "tile_index": spec.tile_index,
            "tile_id": spec.tile_id,
            "cohort": spec.cohort,
            "window_key": spec.key,
            "window_size_m": spec.window_size_m,
            "common_class_id": class_id,
            "common_class_name": class_name,
            "point_count": count,
            "point_fraction_all_points": count / max(1, common.size),
            "density_pts_m2_full_window": count / spec.area_m2,
            "occupied_area_grid_m": actual_res,
            "occupied_area_m2": occupied_area,
            "density_pts_per_occupied_m2": count / occupied_area if occupied_area > 0 else np.nan,
        }
        row.update(summary_stats(z_rel, "height_above_window_min_m"))
        rows.append(row)
    return rows


def _point_train_labels(bundle: DatasetBundle, native_labels: np.ndarray) -> np.ndarray:
    native = np.asarray(native_labels, dtype=np.int64)
    if bundle.dataset_kind == "eclair":
        undefined_id = int(bundle.training_config["data"]["label_space"].get("eclair_undefined_id", 0))
        return eclair_native_to_train_ids(native, undefined_id=undefined_id, ignore_index=bundle.ignore_index).astype(
            np.int64, copy=False
        )
    safe = np.clip(native, 0, bundle.dataset.label_lut.size - 1)
    return bundle.dataset.label_lut[safe].astype(np.int64, copy=False)


def _model_coordinates(
    bundle: DatasetBundle,
    spec: WindowSpec,
    xyz_window_tile_local: np.ndarray,
    coordinate_contract: str,
) -> np.ndarray:
    xyz = np.asarray(xyz_window_tile_local, dtype=np.float64).copy()
    contract = coordinate_contract.lower().strip()
    if contract == "native_pipeline":
        if bundle.dataset_kind == "dales":
            xyz[:, 0] -= spec.x_min_m + 0.5 * spec.window_size_m
            xyz[:, 1] -= spec.y_min_m + 0.5 * spec.window_size_m
        # ECLAIR get_raw() already uses the whole-tile local origin, matching training.
        return xyz
    if contract == "common_window_min":
        xyz -= xyz.min(axis=0, keepdims=True)
        return xyz
    raise ValueError(f"Unknown coordinate_contract={coordinate_contract}")


def build_native_tile_voxel_context(
    bundle: DatasetBundle,
    raw: Mapping[str, Any],
    *,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """
    Build the exact native full-tile voxel context for ECLAIR.

    ECLAIR source training voxelizes an entire tile. Deriving subwindow occupancy
    from this context preserves cross-boundary feature/label pooling. DALES
    training uses crop-centered inputs, so its native context must be built per
    window and this function returns None.
    """
    if bundle.dataset_kind != "eclair":
        return None
    xyz = np.asarray(raw["xyz"], dtype=np.float64)
    native = np.asarray(raw["native_labels"], dtype=np.int64)
    y_train = _point_train_labels(bundle, native)
    xyz_normalized = (xyz / float(bundle.patch_cfg.coord_norm_factor)).astype(np.float32, copy=False)
    features = build_features(
        xyz_local=xyz_normalized,
        intensity=raw.get("intensity") if bundle.feat_cfg.use_intensity else None,
        return_number=raw.get("return_number") if bundle.feat_cfg.use_return_number else None,
        number_of_returns=raw.get("number_of_returns") if bundle.feat_cfg.use_number_of_returns else None,
        rgb=raw.get("rgb") if bundle.feat_cfg.use_rgb else None,
        cfg=bundle.feat_cfg,
    )
    q = np.floor(xyz_normalized / float(bundle.patch_cfg.voxel_size)).astype(np.int32)
    q = np.ascontiguousarray(q, dtype=np.int32)
    vx = voxelize_from_q(
        q_int32=q,
        feats_p_f32=features,
        labels_p_i64=y_train,
        ignore_index=bundle.ignore_index,
        cfg=bundle.voxel_cfg,
        rng=stable_rng(seed, bundle.name, "full_tile_native_voxel"),
        return_maps=True,
        num_classes_hint=bundle.num_train_classes,
    )
    inverse = np.asarray(vx["inverse_map"], dtype=np.int64)
    occupied = int(np.asarray(vx["coords_u"]).shape[0])
    return {
        "inverse_map": inverse,
        "points_per_voxel": np.bincount(inverse, minlength=occupied),
        "labels_u_common": bundle.train_to_common(np.asarray(vx["labels_u"], dtype=np.int64)),
        "voxel_size_normalized": float(bundle.patch_cfg.voxel_size),
        "voxel_size_physical_m": float(bundle.patch_cfg.voxel_size) * float(bundle.patch_cfg.coord_norm_factor),
    }


def _voxel_view_specs(
    bundle: DatasetBundle,
    controlled_physical_sizes_m: Sequence[float],
) -> list[tuple[str, str, float, float]]:
    coord_norm = float(bundle.patch_cfg.coord_norm_factor)
    native_norm = float(bundle.patch_cfg.voxel_size)
    views = [
        (
            "native_pipeline",
            "native_pipeline",
            native_norm,
            native_norm * coord_norm,
        )
    ]
    for physical in controlled_physical_sizes_m:
        physical = float(physical)
        views.append(
            (
                f"controlled_{physical:.3f}m",
                "common_window_min",
                physical / coord_norm,
                physical,
            )
        )
    return views


def compute_voxel_rows(
    bundle: DatasetBundle,
    spec: WindowSpec,
    xyz_window_tile_local: np.ndarray,
    native_labels: np.ndarray,
    common_labels: np.ndarray,
    intensity: Optional[np.ndarray],
    return_number: np.ndarray,
    number_of_returns: np.ndarray,
    rgb: Optional[np.ndarray],
    *,
    controlled_physical_sizes_m: Sequence[float],
    max_points_for_voxelization: int,
    seed: int,
    tile_point_indices: Optional[np.ndarray] = None,
    native_tile_context: Optional[Mapping[str, Any]] = None,
) -> list[Dict[str, Any]]:
    n = int(xyz_window_tile_local.shape[0])
    if n > int(max_points_for_voxelization):
        return [
            {
                "dataset": bundle.name,
                "tile_index": spec.tile_index,
                "tile_id": spec.tile_id,
                "cohort": spec.cohort,
                "window_key": spec.key,
                "window_size_m": spec.window_size_m,
                "view_name": "SKIPPED_TOO_MANY_POINTS",
                "raw_points": n,
                "max_points_for_voxelization": int(max_points_for_voxelization),
            }
        ]

    y_train = _point_train_labels(bundle, native_labels)
    raw_common = np.asarray(common_labels, dtype=np.int64)
    rows: list[Dict[str, Any]] = []

    for (
        view_name,
        coordinate_contract,
        voxel_size_normalized,
        physical_size,
    ) in _voxel_view_specs(bundle, controlled_physical_sizes_m):
        if view_name == "native_pipeline" and native_tile_context is not None:
            if tile_point_indices is None:
                raise ValueError("native_tile_context requires tile_point_indices")
            full_inverse = np.asarray(native_tile_context["inverse_map"], dtype=np.int64)
            selected_voxel_ids = np.unique(full_inverse[np.asarray(tile_point_indices, dtype=np.int64)])
            labels_u_common = np.asarray(native_tile_context["labels_u_common"], dtype=np.int64)[selected_voxel_ids]
            full_points_per_voxel = np.asarray(native_tile_context["points_per_voxel"], dtype=np.int64)
            points_per_voxel = full_points_per_voxel[selected_voxel_ids]
            occupied_voxels = int(selected_voxel_ids.size)
            base = {
                "dataset": bundle.name,
                "tile_index": spec.tile_index,
                "tile_id": spec.tile_id,
                "cohort": spec.cohort,
                "window_key": spec.key,
                "window_size_m": spec.window_size_m,
                "view_name": view_name,
                "coordinate_contract": "full_tile_native_then_window_slice",
                "coord_norm_factor": float(bundle.patch_cfg.coord_norm_factor),
                "voxel_size_normalized": float(native_tile_context["voxel_size_normalized"]),
                "voxel_size_physical_m": float(native_tile_context["voxel_size_physical_m"]),
                "feat_pool": bundle.voxel_cfg.feat_pool,
                "label_pool": bundle.voxel_cfg.label_pool,
                "scope": "all",
                "common_class_id": None,
                "common_class_name": None,
                "raw_points": n,
                "occupied_voxels": occupied_voxels,
                "occupied_voxels_per_m2": occupied_voxels / spec.area_m2,
                "points_per_occupied_voxel_mean": float(np.mean(points_per_voxel)),
                "points_per_occupied_voxel_median": float(np.median(points_per_voxel)),
                "singleton_voxel_fraction": float(np.mean(points_per_voxel == 1)),
                "multi_point_voxel_fraction": float(np.mean(points_per_voxel > 1)),
            }
            base.update(summary_stats(points_per_voxel, "points_per_voxel"))
            rows.append(base)
            window_inverse = full_inverse[np.asarray(tile_point_indices, dtype=np.int64)]
            for class_id, class_name in COMMON_CLASS_NAMES.items():
                model_count = int(np.sum(labels_u_common == class_id))
                point_mask = raw_common == class_id
                footprint_count = int(np.unique(window_inverse[point_mask]).size) if np.any(point_mask) else 0
                rows.append(
                    {
                        "dataset": bundle.name,
                        "tile_index": spec.tile_index,
                        "tile_id": spec.tile_id,
                        "cohort": spec.cohort,
                        "window_key": spec.key,
                        "window_size_m": spec.window_size_m,
                        "view_name": view_name,
                        "coordinate_contract": "full_tile_native_then_window_slice",
                        "coord_norm_factor": float(bundle.patch_cfg.coord_norm_factor),
                        "voxel_size_normalized": float(native_tile_context["voxel_size_normalized"]),
                        "voxel_size_physical_m": float(native_tile_context["voxel_size_physical_m"]),
                        "feat_pool": bundle.voxel_cfg.feat_pool,
                        "label_pool": bundle.voxel_cfg.label_pool,
                        "scope": "common_class",
                        "common_class_id": class_id,
                        "common_class_name": class_name,
                        "raw_points": int(point_mask.sum()),
                        "occupied_voxels": occupied_voxels,
                        "model_label_voxels": model_count,
                        "model_label_voxels_per_m2": model_count / spec.area_m2,
                        "class_footprint_voxels": footprint_count,
                        "class_footprint_voxels_per_m2": footprint_count / spec.area_m2,
                        "model_label_fraction_of_all_voxels": model_count / max(1, occupied_voxels),
                        "class_footprint_fraction_of_all_voxels": footprint_count / max(1, occupied_voxels),
                    }
                )
            continue

        xyz_model_m = _model_coordinates(bundle, spec, xyz_window_tile_local, coordinate_contract)
        xyz_normalized = (xyz_model_m / float(bundle.patch_cfg.coord_norm_factor)).astype(np.float32, copy=False)
        features = build_features(
            xyz_local=xyz_normalized,
            intensity=intensity if bundle.feat_cfg.use_intensity else None,
            return_number=return_number if bundle.feat_cfg.use_return_number else None,
            number_of_returns=number_of_returns if bundle.feat_cfg.use_number_of_returns else None,
            rgb=rgb if bundle.feat_cfg.use_rgb else None,
            cfg=bundle.feat_cfg,
        )
        q = np.floor(xyz_normalized / float(voxel_size_normalized)).astype(np.int32)
        q = np.ascontiguousarray(q, dtype=np.int32)
        rng = stable_rng(seed, bundle.name, spec.key, view_name, "voxel")
        vx = voxelize_from_q(
            q_int32=q,
            feats_p_f32=features,
            labels_p_i64=y_train,
            ignore_index=bundle.ignore_index,
            cfg=bundle.voxel_cfg,
            rng=rng,
            return_maps=True,
            num_classes_hint=bundle.num_train_classes,
        )
        inverse = np.asarray(vx["inverse_map"], dtype=np.int64)
        labels_u_train = np.asarray(vx["labels_u"], dtype=np.int64)
        labels_u_common = bundle.train_to_common(labels_u_train)
        occupied_voxels = int(np.asarray(vx["coords_u"]).shape[0])
        points_per_voxel = np.bincount(inverse, minlength=occupied_voxels)

        base = {
            "dataset": bundle.name,
            "tile_index": spec.tile_index,
            "tile_id": spec.tile_id,
            "cohort": spec.cohort,
            "window_key": spec.key,
            "window_size_m": spec.window_size_m,
            "view_name": view_name,
            "coordinate_contract": coordinate_contract,
            "coord_norm_factor": float(bundle.patch_cfg.coord_norm_factor),
            "voxel_size_normalized": float(voxel_size_normalized),
            "voxel_size_physical_m": float(physical_size),
            "feat_pool": bundle.voxel_cfg.feat_pool,
            "label_pool": bundle.voxel_cfg.label_pool,
            "scope": "all",
            "common_class_id": None,
            "common_class_name": None,
            "raw_points": n,
            "occupied_voxels": occupied_voxels,
            "occupied_voxels_per_m2": occupied_voxels / spec.area_m2,
            "points_per_occupied_voxel_mean": float(np.mean(points_per_voxel)),
            "points_per_occupied_voxel_median": float(np.median(points_per_voxel)),
            "singleton_voxel_fraction": float(np.mean(points_per_voxel == 1)),
            "multi_point_voxel_fraction": float(np.mean(points_per_voxel > 1)),
        }
        base.update(summary_stats(points_per_voxel, "points_per_voxel"))
        rows.append(base)

        for class_id, class_name in COMMON_CLASS_NAMES.items():
            model_count = int(np.sum(labels_u_common == class_id))
            point_mask = raw_common == class_id
            footprint_count = int(np.unique(inverse[point_mask]).size) if np.any(point_mask) else 0
            rows.append(
                {
                    "dataset": bundle.name,
                    "tile_index": spec.tile_index,
                    "tile_id": spec.tile_id,
                    "cohort": spec.cohort,
                    "window_key": spec.key,
                    "window_size_m": spec.window_size_m,
                    "view_name": view_name,
                    "coordinate_contract": coordinate_contract,
                    "coord_norm_factor": float(bundle.patch_cfg.coord_norm_factor),
                    "voxel_size_normalized": float(voxel_size_normalized),
                    "voxel_size_physical_m": float(physical_size),
                    "feat_pool": bundle.voxel_cfg.feat_pool,
                    "label_pool": bundle.voxel_cfg.label_pool,
                    "scope": "common_class",
                    "common_class_id": class_id,
                    "common_class_name": class_name,
                    "raw_points": int(point_mask.sum()),
                    "occupied_voxels": occupied_voxels,
                    "model_label_voxels": model_count,
                    "model_label_voxels_per_m2": model_count / spec.area_m2,
                    "class_footprint_voxels": footprint_count,
                    "class_footprint_voxels_per_m2": footprint_count / spec.area_m2,
                    "model_label_fraction_of_all_voxels": model_count / max(1, occupied_voxels),
                    "class_footprint_fraction_of_all_voxels": footprint_count / max(1, occupied_voxels),
                }
            )
    return rows


def js_divergence(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    low = min(float(a.min()), float(b.min()))
    high = max(float(a.max()), float(b.max()))
    if high <= low:
        return 0.0
    edges = np.linspace(low, high, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa.astype(np.float64) + 1e-12
    pb = pb.astype(np.float64) + 1e-12
    pa /= pa.sum()
    pb /= pb.sum()
    midpoint = 0.5 * (pa + pb)
    return float(0.5 * np.sum(pa * np.log(pa / midpoint)) + 0.5 * np.sum(pb * np.log(pb / midpoint)))


def tile_cluster_comparison(
    dataframe: Any,
    *,
    metric: str,
    group_columns: Sequence[str],
    dataset_order: Sequence[str],
    iterations: int,
    seed: int,
) -> list[Dict[str, Any]]:
    if metric not in dataframe.columns:
        return []
    first_name, second_name = dataset_order
    rows: list[Dict[str, Any]] = []
    grouped = dataframe.groupby(list(group_columns), dropna=False) if group_columns else [((), dataframe)]
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_info = dict(zip(group_columns, group_key))
        tile_values: Dict[str, Dict[str, np.ndarray]] = {}
        for dataset_name in dataset_order:
            subset = group[group["dataset"] == dataset_name]
            values_by_tile = {
                str(tile_id): tile_group[metric].dropna().to_numpy(dtype=np.float64)
                for tile_id, tile_group in subset.groupby("tile_id")
                if not tile_group[metric].dropna().empty
            }
            tile_values[dataset_name] = values_by_tile
        if not tile_values[first_name] or not tile_values[second_name]:
            continue
        first_tile_aggregate = np.array([np.median(v) for v in tile_values[first_name].values()])
        second_tile_aggregate = np.array([np.median(v) for v in tile_values[second_name].values()])
        observed = float(np.median(first_tile_aggregate) - np.median(second_tile_aggregate))
        rng = stable_rng(seed, metric, *group_key, "tile_bootstrap")
        first_tiles = list(tile_values[first_name])
        second_tiles = list(tile_values[second_name])
        diffs = np.empty(iterations, dtype=np.float64)
        for iteration in range(iterations):
            sampled_first = rng.choice(first_tiles, size=len(first_tiles), replace=True)
            sampled_second = rng.choice(second_tiles, size=len(second_tiles), replace=True)
            first_values = np.concatenate([tile_values[first_name][str(tile)] for tile in sampled_first])
            second_values = np.concatenate([tile_values[second_name][str(tile)] for tile in sampled_second])
            diffs[iteration] = np.median(first_values) - np.median(second_values)
        low, high = np.quantile(diffs, [0.025, 0.975])
        rows.append(
            {
                **group_info,
                "metric": metric,
                "dataset_a": first_name,
                "dataset_b": second_name,
                "difference_definition": f"median({first_name}) - median({second_name})",
                "number_of_tiles_a": len(first_tiles),
                "number_of_tiles_b": len(second_tiles),
                "tile_median_a": float(np.median(first_tile_aggregate)),
                "tile_median_b": float(np.median(second_tile_aggregate)),
                "tile_median_difference_a_minus_b": observed,
                "tile_cluster_bootstrap_ci95_low": float(low),
                "tile_cluster_bootstrap_ci95_high": float(high),
                "js_divergence_on_tile_medians": js_divergence(first_tile_aggregate, second_tile_aggregate),
                "wasserstein_on_tile_medians": (
                    float(wasserstein_distance(first_tile_aggregate, second_tile_aggregate)) if SCIPY_AVAILABLE else np.nan
                ),
            }
        )
    return rows
