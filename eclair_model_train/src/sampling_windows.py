from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .sampling_types import Footprint, TileRecord, WindowSpec


@dataclass(frozen=True)
class CandidateSummary:
    spec: WindowSpec
    point_count: int
    first_return_count: int
    single_return_count: int
    later_return_count: int
    class_counts: Dict[int, int]


def stable_rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "|".join([str(seed), *[str(part) for part in parts]])
    digest = hashlib.sha1(payload.encode("utf-8")).digest()
    derived = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(derived)


def _integral_image(counts: np.ndarray) -> np.ndarray:
    return np.pad(counts.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def _rect_sum(ii: np.ndarray, x0: int, y0: int, width: int, height: int) -> int:
    x1 = x0 + width
    y1 = y0 + height
    return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def _histogram_counts(
    xy: np.ndarray,
    footprint: Footprint,
    grid_m: float,
    mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, float]:
    nx = max(1, int(math.ceil(footprint.width_m / grid_m)))
    ny = max(1, int(math.ceil(footprint.height_m / grid_m)))
    actual_x = footprint.width_m / nx
    actual_y = footprint.height_m / ny
    points = xy if mask is None else xy[mask]
    hist, _, _ = np.histogram2d(
        points[:, 1],
        points[:, 0],
        bins=(ny, nx),
        range=(
            (footprint.y_min_m, footprint.y_min_m + footprint.height_m),
            (footprint.x_min_m, footprint.x_min_m + footprint.width_m),
        ),
    )
    return hist.astype(np.int64), actual_x, actual_y


def build_candidate_summaries(
    *,
    tile: TileRecord,
    xyz_local: np.ndarray,
    common_labels: np.ndarray,
    return_number: np.ndarray,
    number_of_returns: np.ndarray,
    footprint: Footprint,
    window_sizes_m: Sequence[float],
    candidates_per_size: int,
    selection_grid_m: float,
    seed: int,
) -> list[CandidateSummary]:
    xy = np.asarray(xyz_local[:, :2], dtype=np.float64)
    rn = np.asarray(return_number, dtype=np.int64)
    nor = np.asarray(number_of_returns, dtype=np.int64)
    common = np.asarray(common_labels, dtype=np.int64)

    all_hist, dx, dy = _histogram_counts(xy, footprint, selection_grid_m)
    first_hist, _, _ = _histogram_counts(xy, footprint, selection_grid_m, rn == 1)
    single_hist, _, _ = _histogram_counts(xy, footprint, selection_grid_m, (rn == 1) & (nor == 1))
    later_hist, _, _ = _histogram_counts(xy, footprint, selection_grid_m, rn > 1)
    class_hists = {
        class_id: _histogram_counts(xy, footprint, selection_grid_m, common == class_id)[0] for class_id in range(1, 8)
    }

    all_ii = _integral_image(all_hist)
    first_ii = _integral_image(first_hist)
    single_ii = _integral_image(single_hist)
    later_ii = _integral_image(later_hist)
    class_ii = {class_id: _integral_image(hist) for class_id, hist in class_hists.items()}

    ny, nx = all_hist.shape
    results: list[CandidateSummary] = []
    candidate_id = 0

    for size_m in window_sizes_m:
        width_bins = int(round(float(size_m) / dx))
        height_bins = int(round(float(size_m) / dy))
        if width_bins <= 0 or height_bins <= 0 or width_bins > nx or height_bins > ny:
            continue
        effective_w = width_bins * dx
        effective_h = height_bins * dy
        if abs(effective_w - float(size_m)) > max(dx, 1e-6) * 0.05 or abs(effective_h - float(size_m)) > max(dy, 1e-6) * 0.05:
            raise ValueError(
                f"window_size={size_m}m is not compatible with selection grid actual resolution "
                f"dx={dx}, dy={dy}; choose divisible values"
            )

        rng = stable_rng(seed, tile.dataset, tile.path, size_m, "candidate_windows")
        possible_x = nx - width_bins + 1
        possible_y = ny - height_bins + 1
        total_positions = possible_x * possible_y
        n = min(int(candidates_per_size), total_positions)
        flat_positions = rng.choice(total_positions, size=n, replace=False)

        for flat in flat_positions.tolist():
            y0_bin = int(flat // possible_x)
            x0_bin = int(flat % possible_x)
            spec = WindowSpec(
                dataset=tile.dataset,
                tile_index=tile.tile_index,
                tile_id=tile.tile_id,
                tile_path=str(tile.path),
                cohort="candidate",
                selection_reason="candidate",
                window_size_m=float(size_m),
                x_min_m=float(footprint.x_min_m + x0_bin * dx),
                y_min_m=float(footprint.y_min_m + y0_bin * dy),
                candidate_id=candidate_id,
            )
            results.append(
                CandidateSummary(
                    spec=spec,
                    point_count=_rect_sum(all_ii, x0_bin, y0_bin, width_bins, height_bins),
                    first_return_count=_rect_sum(first_ii, x0_bin, y0_bin, width_bins, height_bins),
                    single_return_count=_rect_sum(single_ii, x0_bin, y0_bin, width_bins, height_bins),
                    later_return_count=_rect_sum(later_ii, x0_bin, y0_bin, width_bins, height_bins),
                    class_counts={
                        class_id: _rect_sum(ii, x0_bin, y0_bin, width_bins, height_bins) for class_id, ii in class_ii.items()
                    },
                )
            )
            candidate_id += 1

    return results


def _intersection_fraction(a: WindowSpec, b: WindowSpec) -> float:
    x_overlap = max(0.0, min(a.x_max_m, b.x_max_m) - max(a.x_min_m, b.x_min_m))
    y_overlap = max(0.0, min(a.y_max_m, b.y_max_m) - max(a.y_min_m, b.y_min_m))
    intersection = x_overlap * y_overlap
    if intersection <= 0:
        return 0.0
    return intersection / min(a.area_m2, b.area_m2)


def _accept_non_overlapping(
    candidate: WindowSpec,
    selected: Sequence[WindowSpec],
    max_overlap_fraction: float,
) -> bool:
    return all(
        _intersection_fraction(candidate, existing) <= max_overlap_fraction
        for existing in selected
        if math.isclose(existing.window_size_m, candidate.window_size_m)
    )


def select_primary_windows(
    candidates: Sequence[CandidateSummary],
    *,
    windows_per_size: int,
    min_points: int,
    max_overlap_fraction: float,
    seed: int,
) -> list[WindowSpec]:
    selected: list[WindowSpec] = []
    sizes = sorted({item.spec.window_size_m for item in candidates})
    for size in sizes:
        valid = [item for item in candidates if item.spec.window_size_m == size and item.point_count >= min_points]
        rng = stable_rng(seed, valid[0].spec.dataset if valid else "none", size, "primary_select")
        order = rng.permutation(len(valid)).tolist() if valid else []
        chosen: list[WindowSpec] = []
        for index in order:
            item = valid[int(index)]
            spec = WindowSpec(
                **{
                    **item.spec.__dict__,
                    "cohort": "primary_random",
                    "selection_reason": "unbiased_random",
                }
            )
            if _accept_non_overlapping(spec, chosen, max_overlap_fraction):
                chosen.append(spec)
            if len(chosen) >= windows_per_size:
                break
        selected.extend(chosen)
    return selected


def select_diagnostic_windows(
    candidates: Sequence[CandidateSummary],
    *,
    focus_common_ids: Sequence[int],
    windows_per_class_per_size: int,
    min_points: int,
    min_focus_points: int,
    max_overlap_fraction: float,
) -> list[WindowSpec]:
    selected: list[WindowSpec] = []
    sizes = sorted({item.spec.window_size_m for item in candidates})
    for size in sizes:
        for class_id in focus_common_ids:
            ranked = [
                item
                for item in candidates
                if item.spec.window_size_m == size
                and item.point_count >= min_points
                and item.class_counts.get(int(class_id), 0) >= min_focus_points
            ]
            ranked.sort(
                key=lambda item: (
                    item.class_counts.get(int(class_id), 0) / max(1, item.point_count),
                    item.class_counts.get(int(class_id), 0),
                ),
                reverse=True,
            )
            chosen: list[WindowSpec] = []
            for item in ranked:
                spec = WindowSpec(
                    **{
                        **item.spec.__dict__,
                        "cohort": "diagnostic_stratified",
                        "selection_reason": f"class_rich_common_{int(class_id)}",
                        "focus_common_id": int(class_id),
                    }
                )
                if _accept_non_overlapping(spec, chosen, max_overlap_fraction):
                    chosen.append(spec)
                if len(chosen) >= windows_per_class_per_size:
                    break
            selected.extend(chosen)
    return selected


def extract_window_indices(xyz_local: np.ndarray, spec: WindowSpec) -> np.ndarray:
    xyz = np.asarray(xyz_local)
    mask = (xyz[:, 0] >= spec.x_min_m) & (xyz[:, 0] < spec.x_max_m) & (xyz[:, 1] >= spec.y_min_m) & (xyz[:, 1] < spec.y_max_m)
    return np.flatnonzero(mask)
