from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import laspy
import numpy as np

from .sampling_types import DatasetBundle, Footprint, TileRecord

REQUIRED_STANDARD_DIMENSIONS = {
    "return_number",
    "number_of_returns",
}


def inspect_las_header(path: Path) -> Dict[str, Any]:
    with laspy.open(path) as reader:
        header = reader.header
        dimensions = [str(name) for name in header.point_format.dimension_names]
        dimensions_lower = {name.lower() for name in dimensions}
        crs = None
        try:
            parsed = header.parse_crs()
            crs = parsed.to_string() if parsed is not None else None
        except Exception:
            crs = None

        has_classification = "classification" in dimensions_lower or "raw_classification" in dimensions_lower
        return {
            "path": str(path),
            "las_version": str(header.version),
            "point_format_id": int(header.point_format.id),
            "point_count_header": int(header.point_count),
            "dimensions": json.dumps(dimensions),
            "has_classification": bool(has_classification),
            "has_return_number": "return_number" in dimensions_lower,
            "has_number_of_returns": "number_of_returns" in dimensions_lower,
            "x_min_header": float(header.mins[0]),
            "y_min_header": float(header.mins[1]),
            "z_min_header": float(header.mins[2]),
            "x_max_header": float(header.maxs[0]),
            "y_max_header": float(header.maxs[1]),
            "z_max_header": float(header.maxs[2]),
            "span_x_header_m": float(header.maxs[0] - header.mins[0]),
            "span_y_header_m": float(header.maxs[1] - header.mins[1]),
            "span_z_header_m": float(header.maxs[2] - header.mins[2]),
            "scale_x": float(header.scales[0]),
            "scale_y": float(header.scales[1]),
            "scale_z": float(header.scales[2]),
            "crs": crs,
        }


def validate_header_contract(path: Path, header: Dict[str, Any]) -> None:
    missing = []
    if not header["has_classification"]:
        missing.append("classification/raw_classification")
    if not header["has_return_number"]:
        missing.append("return_number")
    if not header["has_number_of_returns"]:
        missing.append("number_of_returns")
    if missing:
        raise ValueError(
            f"LAS file is unsuitable for density/return/class analysis: {path}; "
            f"missing dimensions={missing}; dimensions={header['dimensions']}"
        )


def validate_raw_contract(
    bundle: DatasetBundle,
    tile: TileRecord,
    raw: Dict[str, Any],
    header: Dict[str, Any],
    *,
    strict_returns: bool = True,
) -> Dict[str, Any]:
    required = {"xyz", "native_labels", "return_number", "number_of_returns"}
    missing = required - set(raw)
    if missing:
        raise KeyError(f"{bundle.name}/{tile.tile_id}: get_raw() missing keys {sorted(missing)}")

    xyz = np.asarray(raw["xyz"])
    labels = np.asarray(raw["native_labels"])
    rn = np.asarray(raw["return_number"])
    nor = np.asarray(raw["number_of_returns"])
    n = int(xyz.shape[0])

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"{bundle.name}/{tile.tile_id}: xyz shape must be [N,3], got {xyz.shape}")
    if xyz.dtype != np.float64:
        raise TypeError(f"{bundle.name}/{tile.tile_id}: expected get_raw()['xyz'] float64, got {xyz.dtype}")
    for name, values in {
        "native_labels": labels,
        "return_number": rn,
        "number_of_returns": nor,
    }.items():
        if values.ndim != 1 or values.shape[0] != n:
            raise ValueError(f"{bundle.name}/{tile.tile_id}: {name} shape must be [{n}], got {values.shape}")

    if not np.isfinite(xyz).all():
        raise ValueError(f"{bundle.name}/{tile.tile_id}: xyz contains NaN/Inf")
    if n != int(header["point_count_header"]):
        raise ValueError(
            f"{bundle.name}/{tile.tile_id}: get_raw count {n:,} differs from LAS header " f"{int(header['point_count_header']):,}"
        )

    labels_i64 = labels.astype(np.int64, copy=False)
    observed = set(np.unique(labels_i64).tolist())
    unexpected = observed - bundle.expected_native_ids
    if unexpected:
        raise ValueError(
            f"{bundle.name}/{tile.tile_id}: unexpected native label IDs {sorted(unexpected)}; "
            f"expected subset of {sorted(bundle.expected_native_ids)}"
        )

    common = bundle.raw_to_common(labels_i64)
    valid_common = common != 0
    if not np.any(valid_common):
        raise ValueError(
            f"{bundle.name}/{tile.tile_id}: all points map to common ignore. "
            "This usually means the wrong files or wrong label convention were used."
        )

    rn_i64 = rn.astype(np.int64, copy=False)
    nor_i64 = nor.astype(np.int64, copy=False)
    invalid_return = (rn_i64 < 1) | (nor_i64 < 1) | (rn_i64 > nor_i64)
    invalid_count = int(invalid_return.sum())
    if strict_returns and invalid_count:
        examples = np.column_stack((rn_i64[invalid_return], nor_i64[invalid_return]))[:10]
        raise ValueError(
            f"{bundle.name}/{tile.tile_id}: {invalid_count:,} invalid return pairs; " f"examples={examples.tolist()}"
        )

    return {
        "dataset": bundle.name,
        "tile_index": tile.tile_index,
        "tile_id": tile.tile_id,
        "path": str(tile.path),
        "point_count": n,
        "xyz_dtype": str(xyz.dtype),
        "observed_native_ids": json.dumps(sorted(int(x) for x in observed)),
        "mapped_common_fraction": float(np.mean(valid_common)),
        "ignored_common_fraction": float(np.mean(~valid_common)),
        "return_number_min": int(rn_i64.min()),
        "return_number_max": int(rn_i64.max()),
        "number_of_returns_min": int(nor_i64.min()),
        "number_of_returns_max": int(nor_i64.max()),
        "invalid_return_pairs": invalid_count,
        "observed_span_x_m": float(np.ptp(xyz[:, 0])),
        "observed_span_y_m": float(np.ptp(xyz[:, 1])),
        "observed_span_z_m": float(np.ptp(xyz[:, 2])),
    }


def determine_footprint(
    xyz_local: np.ndarray,
    cfg: Dict[str, Any],
    *,
    dataset_name: str,
    tile_id: str,
) -> Footprint:
    xyz = np.asarray(xyz_local, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[0] == 0 or xyz.shape[1] < 2:
        raise ValueError(f"{dataset_name}/{tile_id}: expected a non-empty [N, >=2] coordinate array, " f"got shape={xyz.shape}")

    x_min = float(np.min(xyz[:, 0]))
    y_min = float(np.min(xyz[:, 1]))
    span_x = float(np.ptp(xyz[:, 0]))
    span_y = float(np.ptp(xyz[:, 1]))
    mode = str(cfg.get("mode", "point_bounds")).lower().strip()

    if mode == "point_bounds":
        return Footprint(
            width_m=span_x,
            height_m=span_y,
            source="point_bounds",
            observed_span_x_m=span_x,
            observed_span_y_m=span_y,
            x_min_m=x_min,
            y_min_m=y_min,
        )

    if mode != "nominal_square":
        raise ValueError(f"Unknown footprint mode={mode!r}")

    nominal = float(cfg["nominal_size_m"])
    minimum_fraction = float(cfg.get("min_extent_fraction", 0.90))
    exclude_partial = bool(cfg.get("exclude_partial_tiles", True))
    overshoot_tolerance_m = float(cfg.get("max_extent_overshoot_m", 1.0))
    if span_x > nominal + overshoot_tolerance_m or span_y > nominal + overshoot_tolerance_m:
        raise ValueError(
            f"{dataset_name}/{tile_id}: observed span={span_x:.3f}x{span_y:.3f}m exceeds "
            f"nominal={nominal:.3f}m by more than tolerance={overshoot_tolerance_m:.3f}m. "
            "Check coordinate units or nominal footprint configuration."
        )
    fraction_x = span_x / nominal
    fraction_y = span_y / nominal
    partial = fraction_x < minimum_fraction or fraction_y < minimum_fraction
    if partial and exclude_partial:
        raise RuntimeError(
            f"SKIP_PARTIAL_TILE::{dataset_name}/{tile_id}: observed span={span_x:.3f}x{span_y:.3f}m, "
            f"nominal={nominal:.3f}m, required_fraction={minimum_fraction:.3f}"
        )

    return Footprint(
        width_m=nominal,
        height_m=nominal,
        source="nominal_square",
        observed_span_x_m=span_x,
        observed_span_y_m=span_y,
        x_min_m=x_min,
        y_min_m=y_min,
    )
