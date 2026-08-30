from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .mix3d import ALSMix3DConfig, MixResult, compose_crop_replace


def _rows(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).reshape(-1)


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0.0 else float("nan")


def _quantile(a: np.ndarray, q: float) -> float:
    return float(np.quantile(a, q)) if a.size else float("nan")


def _class_counts(labels: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    valid = (y != int(ignore_index)) & (y >= 0) & (y < int(num_classes))
    return np.bincount(y[valid], minlength=int(num_classes))[: int(num_classes)].astype(np.int64)


def _majority_labels(
    labels: np.ndarray,
    inverse: np.ndarray,
    n_groups: int,
    *,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    out = np.full(int(n_groups), int(ignore_index), dtype=np.int64)
    valid = (
        (labels != int(ignore_index))
        & (labels >= 0)
        & (labels < int(num_classes))
    )
    if not np.any(valid):
        return out
    flat = inverse[valid] * int(num_classes) + labels[valid]
    counts = np.bincount(flat, minlength=int(n_groups) * int(num_classes)).reshape(
        int(n_groups), int(num_classes)
    )
    present = counts.sum(axis=1) > 0
    out[present] = counts[present].argmax(axis=1)
    return out


def voxelize_labels_numpy(
    xyz_m: np.ndarray,
    labels: np.ndarray,
    *,
    voxel_m: float,
    num_classes: int,
    ignore_index: int,
) -> Dict[str, np.ndarray]:
    """Dependency-free equivalent of the repository's majority-label quantization."""
    xyz = np.asarray(xyz_m)
    y = np.asarray(labels, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or y.shape != (xyz.shape[0],):
        raise ValueError("xyz/labels must have shapes [N,3] and [N].")
    if not np.isfinite(voxel_m) or float(voxel_m) <= 0.0:
        raise ValueError("voxel_m must be finite and positive.")
    q = np.floor(xyz / float(voxel_m)).astype(np.int64, copy=False)
    coords, first, inverse, counts = np.unique(
        q, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    labels_u = _majority_labels(
        y,
        inverse,
        int(coords.shape[0]),
        num_classes=int(num_classes),
        ignore_index=int(ignore_index),
    )
    return {
        "coords": coords,
        "labels": labels_u,
        "first": first.astype(np.int64, copy=False),
        "inverse": inverse.astype(np.int64, copy=False),
        "counts": counts.astype(np.int64, copy=False),
    }


def _js_divergence(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    a = np.asarray(counts_a, dtype=np.float64)
    b = np.asarray(counts_b, dtype=np.float64)
    if a.sum() <= 0 or b.sum() <= 0:
        return float("nan")
    p = a / a.sum()
    q = b / b.sum()
    m = 0.5 * (p + q)

    def kl(x: np.ndarray, y: np.ndarray) -> float:
        nz = x > 0
        return float(np.sum(x[nz] * np.log2(x[nz] / y[nz])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def occupancy_summary(
    xyz_m: np.ndarray,
    labels: np.ndarray,
    *,
    voxel_sizes_m: Sequence[float],
    num_classes: int,
    ignore_index: int,
) -> list[Dict[str, Any]]:
    rows = []
    point_counts = _class_counts(labels, num_classes, ignore_index)
    for size in voxel_sizes_m:
        vx = voxelize_labels_numpy(
            xyz_m,
            labels,
            voxel_m=float(size),
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        counts = vx["counts"]
        vox_class = _class_counts(vx["labels"], num_classes, ignore_index)
        rows.append(
            {
                "voxel_m": float(size),
                "points": int(np.asarray(xyz_m).shape[0]),
                "active_voxels": int(counts.size),
                "active_voxels_per_1000_points": 1000.0
                * _safe_ratio(counts.size, np.asarray(xyz_m).shape[0]),
                "points_per_active_voxel_mean": float(counts.mean()) if counts.size else 0.0,
                "points_per_active_voxel_p95": _quantile(counts, 0.95),
                "singleton_voxel_fraction": float(np.mean(counts == 1)) if counts.size else 0.0,
                "point_class_counts": point_counts.tolist(),
                "voxel_class_counts": vox_class.tolist(),
            }
        )
    return rows


def _component_sizes(coords: np.ndarray, *, dimensions: int = 3) -> list[int]:
    """Small audit-only connected components on an occupied lattice."""
    if coords.size == 0:
        return []
    pts = {tuple(int(v) for v in row[:dimensions]) for row in np.asarray(coords)}
    sizes: list[int] = []
    while pts:
        root = pts.pop()
        todo = [root]
        n = 0
        while todo:
            cur = todo.pop()
            n += 1
            for axis in range(dimensions):
                for delta in (-1, 1):
                    nxt = list(cur)
                    nxt[axis] += delta
                    key = tuple(nxt)
                    if key in pts:
                        pts.remove(key)
                        todo.append(key)
        sizes.append(n)
    return sizes


def _class_geometry(
    xyz: np.ndarray,
    labels: np.ndarray,
    class_id: int,
    *,
    component_voxel_m: float,
) -> Dict[str, float]:
    pts = np.asarray(xyz)[np.asarray(labels) == int(class_id)]
    if pts.size == 0:
        return {
            "points": 0,
            "xy_extent_m": 0.0,
            "z_extent_m": 0.0,
            "components": 0,
            "largest_component_fraction": float("nan"),
        }
    q = np.unique(np.floor(pts / float(component_voxel_m)).astype(np.int64), axis=0)
    sizes = _component_sizes(q, dimensions=3)
    return {
        "points": int(pts.shape[0]),
        "xy_extent_m": float(np.linalg.norm(np.ptp(pts[:, :2], axis=0))),
        "z_extent_m": float(np.ptp(pts[:, 2])),
        "components": int(len(sizes)),
        "largest_component_fraction": _safe_ratio(max(sizes, default=0), sum(sizes)),
    }


def _voxel_set_metrics(base_coords: np.ndarray, pert_coords: np.ndarray) -> Dict[str, float]:
    a = np.unique(_rows(np.asarray(base_coords)))
    b = np.unique(_rows(np.asarray(pert_coords)))
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    return {
        "active_voxels": int(b.size),
        "active_voxel_retention": _safe_ratio(b.size, a.size),
        "active_voxel_jaccard": _safe_ratio(inter, union),
        "changed_active_voxel_fraction": 1.0 - _safe_ratio(inter, a.size),
    }


def audit_occupancy_perturbations(
    payload: Mapping[str, Any],
    *,
    input_voxel_m: float,
    strengths: Sequence[float],
    num_classes: int,
    ignore_index: int,
    protected_class_ids: Sequence[int],
    protected_min_voxels: int,
    component_voxel_m: float,
    seed: int,
) -> list[Dict[str, Any]]:
    """Compare point thinning with model-visible occupied-voxel masking."""
    xyz = np.asarray(payload["xyz"])
    y = np.asarray(payload["y_train"], dtype=np.int64)
    base = voxelize_labels_numpy(
        xyz,
        y,
        voxel_m=float(input_voxel_m),
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    base_point_counts = _class_counts(y, num_classes, ignore_index)
    base_vox_counts = _class_counts(base["labels"], num_classes, ignore_index)
    protected = {int(c) for c in protected_class_ids if 0 <= int(c) < int(num_classes)}
    base_geometry = {
        cls: _class_geometry(xyz, y, cls, component_voxel_m=float(component_voxel_m))
        for cls in protected
    }
    out: list[Dict[str, Any]] = []

    for si, strength in enumerate(strengths):
        p = float(strength)
        if not 0.0 < p < 1.0:
            raise ValueError("occupancy strengths must be in (0,1).")
        for method in ("point_thin", "active_voxel_mask"):
            rng = np.random.default_rng(int(seed) + si * 1009 + (0 if method == "point_thin" else 53))
            if method == "point_thin":
                keep_points = rng.random(xyz.shape[0]) >= p
            else:
                keep_vox = rng.random(base["coords"].shape[0]) >= p
                # Class-preserving safeguard: force a minimum active support for
                # protected classes that were present in the unperturbed view.
                for cls in protected:
                    ids = np.where(base["labels"] == cls)[0]
                    need = min(int(protected_min_voxels), int(ids.size))
                    have = int(keep_vox[ids].sum())
                    if have < need:
                        candidates = ids[~keep_vox[ids]]
                        if candidates.size:
                            chosen = rng.choice(candidates, size=need - have, replace=False)
                            keep_vox[chosen] = True
                keep_points = keep_vox[base["inverse"]]

            if not np.any(keep_points):
                continue
            xyz_p = xyz[keep_points]
            y_p = y[keep_points]
            pert = voxelize_labels_numpy(
                xyz_p,
                y_p,
                voxel_m=float(input_voxel_m),
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
            point_counts = _class_counts(y_p, num_classes, ignore_index)
            vox_counts = _class_counts(pert["labels"], num_classes, ignore_index)
            per_class = []
            protected_disappeared = 0
            for cls in range(int(num_classes)):
                geom0 = base_geometry.get(cls)
                geom1 = (
                    _class_geometry(xyz_p, y_p, cls, component_voxel_m=float(component_voxel_m))
                    if cls in protected
                    else None
                )
                disappeared = bool(base_vox_counts[cls] > 0 and vox_counts[cls] == 0)
                if cls in protected and disappeared:
                    protected_disappeared += 1
                per_class.append(
                    {
                        "class_id": cls,
                        "point_retention": _safe_ratio(point_counts[cls], base_point_counts[cls]),
                        "active_voxel_retention": _safe_ratio(vox_counts[cls], base_vox_counts[cls]),
                        "disappeared": int(disappeared),
                        "components_before": None if geom0 is None else int(geom0["components"]),
                        "components_after": None if geom1 is None else int(geom1["components"]),
                        "largest_component_fraction_before": None if geom0 is None else geom0["largest_component_fraction"],
                        "largest_component_fraction_after": None if geom1 is None else geom1["largest_component_fraction"],
                        "xy_extent_retention": None if geom0 is None else _safe_ratio(geom1["xy_extent_m"], geom0["xy_extent_m"]),
                        "z_extent_retention": None if geom0 is None else _safe_ratio(geom1["z_extent_m"], geom0["z_extent_m"]),
                    }
                )
            row: Dict[str, Any] = {
                "method": method,
                "strength": p,
                "retained_points": int(keep_points.sum()),
                "point_retention": float(keep_points.mean()),
                "protected_classes_disappeared": int(protected_disappeared),
                "per_class": per_class,
            }
            row.update(_voxel_set_metrics(base["coords"], pert["coords"]))
            out.append(row)
    return out


def _distance_to_rect_boundary(xy: np.ndarray, rect: Tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    x = xy[:, 0]
    y = xy[:, 1]
    return np.minimum.reduce((np.abs(x - x0), np.abs(x - x1), np.abs(y - y0), np.abs(y - y1)))


def _cross_provenance_nn(
    host_xyz: np.ndarray,
    donor_xyz: np.ndarray,
    host_y: np.ndarray,
    donor_y: np.ndarray,
) -> Dict[str, float]:
    if host_xyz.shape[0] == 0 or donor_xyz.shape[0] == 0:
        return {}
    # Cap only the query side; the deterministic stride avoids RNG dependence.
    if donor_xyz.shape[0] > 20000:
        take = np.linspace(0, donor_xyz.shape[0] - 1, 20000, dtype=np.int64)
        donor_xyz = donor_xyz[take]
        donor_y = donor_y[take]
    try:
        from scipy.spatial import cKDTree

        dist, idx = cKDTree(host_xyz[:, :2]).query(donor_xyz[:, :2], k=1, workers=1)
    except Exception:
        # Portable fallback for small smoke tests.
        if host_xyz.shape[0] > 10000:
            take = np.linspace(0, host_xyz.shape[0] - 1, 10000, dtype=np.int64)
            host_xyz = host_xyz[take]
            host_y = host_y[take]
        d2 = ((donor_xyz[:, None, :2] - host_xyz[None, :, :2]) ** 2).sum(axis=2)
        idx = d2.argmin(axis=1)
        dist = np.sqrt(d2[np.arange(d2.shape[0]), idx])
    dz = np.abs(donor_xyz[:, 2] - host_xyz[idx, 2])
    mismatch = donor_y != host_y[idx]
    return {
        "cross_xy_nn_median_m": _quantile(np.asarray(dist), 0.5),
        "cross_xy_nn_p95_m": _quantile(np.asarray(dist), 0.95),
        "cross_abs_dz_median_m": _quantile(dz, 0.5),
        "cross_abs_dz_p95_m": _quantile(dz, 0.95),
        "cross_label_mismatch_fraction": float(np.mean(mismatch)),
    }


def _cut_components_2d(
    xyz: np.ndarray,
    labels: np.ndarray,
    removed: np.ndarray,
    kept: np.ndarray,
    *,
    class_id: int,
    cell_m: float,
) -> Tuple[int, int]:
    take = np.asarray(labels) == int(class_id)
    if not np.any(take):
        return 0, 0
    cells = np.floor(np.asarray(xyz)[take, :2] / float(cell_m)).astype(np.int64)
    rem = np.asarray(removed, dtype=bool)[take]
    kep = np.asarray(kept, dtype=bool)[take]
    unique, inv = np.unique(cells, axis=0, return_inverse=True)
    flags = np.zeros((unique.shape[0], 2), dtype=bool)
    np.logical_or.at(flags[:, 0], inv, rem)
    np.logical_or.at(flags[:, 1], inv, kep)
    lookup = {tuple(row): i for i, row in enumerate(unique.tolist())}
    unseen = set(lookup)
    total = 0
    cut = 0
    while unseen:
        root = unseen.pop()
        todo = deque([root])
        has_removed = False
        has_kept = False
        while todo:
            cur = todo.popleft()
            f = flags[lookup[cur]]
            has_removed |= bool(f[0])
            has_kept |= bool(f[1])
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in unseen:
                    unseen.remove(nxt)
                    todo.append(nxt)
        total += 1
        cut += int(has_removed and has_kept)
    return cut, total


def audit_mix3d_boundary(
    host: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    cfg: ALSMix3DConfig,
    num_classes: int,
    ignore_index: int,
    voxel_edge_m: float,
    seam_widths_m: Sequence[float],
    component_cell_m: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    result: MixResult = compose_crop_replace(
        host=host,
        donor=donor,
        cfg=cfg,
        num_classes=num_classes,
        voxel_edge_m=voxel_edge_m,
        rng=rng,
        return_trace=True,
    )
    d = result.diagnostics
    row: Dict[str, Any] = {
        "applied": int(d.applied),
        "skip_code": int(d.skip_code),
        "requested_area_fraction": float(cfg.replacement_area_fraction),
        "actual_area_fraction": _safe_ratio(
            d.replacement_side_m**2, float(host["context_side_xy_m"]) ** 2
        ),
        "host_removed_fraction": _safe_ratio(d.host_removed, d.host_points),
        "output_to_host_points_ratio": _safe_ratio(d.output_points, d.host_points),
        "guard_band_m": float(d.guard_band_m),
        "height_shift_m": float(d.height_shift_m),
        "cross_provenance_voxels": int(d.cross_provenance_voxels),
        "removed_vs_inserted_class_js": _js_divergence(
            np.asarray(d.host_removed_class_counts),
            np.asarray(d.donor_inserted_class_counts),
        ),
    }
    if not d.applied or result.trace is None:
        return row

    tr = result.trace
    hxyz = np.asarray(host["xyz"])
    hy = np.asarray(host["y_train"], dtype=np.int64)
    dxyz = np.asarray(donor["xyz"])[tr.donor_region_mask].astype(np.float64, copy=True)
    dxyz += np.asarray(tr.donor_translation_xyz_m, dtype=np.float64)
    dy = np.asarray(donor["y_train"], dtype=np.int64)[tr.donor_region_mask]
    hkeep_xyz = hxyz[tr.host_keep_mask]
    hkeep_y = hy[tr.host_keep_mask]

    seam_rows = []
    for width in seam_widths_m:
        hs = _distance_to_rect_boundary(hkeep_xyz[:, :2], tr.host_xyxy_m) <= float(width)
        ds = _distance_to_rect_boundary(dxyz[:, :2], tr.host_xyxy_m) <= float(width)
        seam = {
            "width_m": float(width),
            "host_points": int(hs.sum()),
            "donor_points": int(ds.sum()),
            "donor_to_host_point_ratio": _safe_ratio(ds.sum(), hs.sum()),
        }
        seam.update(_cross_provenance_nn(hkeep_xyz[hs], dxyz[ds], hkeep_y[hs], dy[ds]))
        seam_rows.append(seam)
    row["seam_bands"] = seam_rows

    cuts = []
    total_cut = 0
    total_components = 0
    for cls in range(int(num_classes)):
        n_cut, n_total = _cut_components_2d(
            hxyz,
            hy,
            tr.host_expanded_mask,
            tr.host_keep_mask,
            class_id=cls,
            cell_m=float(component_cell_m),
        )
        total_cut += n_cut
        total_components += n_total
        cuts.append(
            {
                "class_id": cls,
                "cut_components": int(n_cut),
                "total_components": int(n_total),
                "cut_component_fraction": _safe_ratio(n_cut, n_total),
            }
        )
    row["component_cuts"] = cuts
    row["cut_component_fraction_all"] = _safe_ratio(total_cut, total_components)
    return row


def _bev_policy_labels(
    flat: np.ndarray,
    labels: np.ndarray,
    z: np.ndarray,
    *,
    policy: str,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(flat, kind="mergesort")
    fs = flat[order]
    starts = np.r_[0, 1 + np.flatnonzero(fs[1:] != fs[:-1])]
    ends = np.r_[starts[1:], fs.size]
    selected_labels = []
    selected_flat = []
    for start, end in zip(starts, ends):
        ids = order[start:end]
        if policy == "first":
            chosen = ids[0]
            label = int(labels[chosen])
        elif policy == "last":
            chosen = ids[-1]
            label = int(labels[chosen])
        elif policy == "highest_z":
            chosen = ids[int(np.argmax(z[ids]))]
            label = int(labels[chosen])
        elif policy == "majority":
            bc = np.bincount(labels[ids], minlength=int(num_classes))
            label = int(bc.argmax())
        else:
            raise ValueError(policy)
        selected_flat.append(int(fs[start]))
        selected_labels.append(label)
    return np.asarray(selected_flat, dtype=np.int64), np.asarray(selected_labels, dtype=np.int64)


def audit_bev_projection(
    payload: Mapping[str, Any],
    *,
    input_voxel_m: float,
    resolutions_m: Sequence[float],
    bounds_xyz_m: Tuple[float, float, float, float, float, float],
    num_classes: int,
    ignore_index: int,
    utility_class_ids: Sequence[int],
    z_slices: int,
    hidden_dim: int,
    xy_frame: str = "native",
    z_filter: str = "bounds",
    height_slicing: str = "fixed_bounds",
) -> list[Dict[str, Any]]:
    xyz = np.asarray(payload["xyz"])
    y = np.asarray(payload["y_train"], dtype=np.int64)
    vx = voxelize_labels_numpy(
        xyz,
        y,
        voxel_m=float(input_voxel_m),
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    coords_m = vx["coords"].astype(np.float64) * float(input_voxel_m)
    labels = vx["labels"]
    valid_label = (labels != int(ignore_index)) & (labels >= 0) & (labels < int(num_classes))
    xy_frame = str(xy_frame).lower()
    z_filter = str(z_filter).lower()
    height_slicing = str(height_slicing).lower()
    if xy_frame not in {"native", "bbox_centered"}:
        raise ValueError(f"Unsupported BEV xy_frame: {xy_frame}")
    if z_filter not in {"bounds", "none"}:
        raise ValueError(f"Unsupported BEV z_filter: {z_filter}")
    if height_slicing not in {"fixed_bounds", "quantile"}:
        raise ValueError(f"Unsupported BEV height_slicing: {height_slicing}")

    projection_coords = coords_m.copy()
    projection_center_xy = np.zeros(2, dtype=np.float64)
    if xy_frame == "bbox_centered" and projection_coords.shape[0]:
        projection_center_xy = 0.5 * (
            projection_coords[:, :2].min(axis=0)
            + projection_coords[:, :2].max(axis=0)
        )
        projection_coords[:, :2] -= projection_center_xy

    x0, x1, y0, y1, z0, z1 = (float(v) for v in bounds_xyz_m)
    z_ok = (
        np.ones(labels.shape[0], dtype=bool)
        if z_filter == "none"
        else (projection_coords[:, 2] >= z0) & (projection_coords[:, 2] < z1)
    )
    in_bounds = (
        valid_label
        & (projection_coords[:, 0] >= x0)
        & (projection_coords[:, 0] < x1)
        & (projection_coords[:, 1] >= y0)
        & (projection_coords[:, 1] < y1)
        & z_ok
    )
    total_class = _class_counts(labels, num_classes, ignore_index)
    in_class = _class_counts(labels[in_bounds], num_classes, ignore_index)
    utility = {int(c) for c in utility_class_ids}
    rows: list[Dict[str, Any]] = []

    for res in resolutions_m:
        res = float(res)
        width = max(1, int(math.ceil((x1 - x0) / res)))
        height = max(1, int(math.ceil((y1 - y0) / res)))
        c = projection_coords[in_bounds]
        lab = labels[in_bounds]
        if c.shape[0]:
            px = np.clip(np.floor((c[:, 0] - x0) / res).astype(np.int64), 0, width - 1)
            py = np.clip(np.floor((c[:, 1] - y0) / res).astype(np.int64), 0, height - 1)
            flat = py * width + px
            unique_cells, inv, per_cell = np.unique(flat, return_inverse=True, return_counts=True)
            class_presence = np.zeros((unique_cells.size, int(num_classes)), dtype=bool)
            class_presence[inv, lab] = True
            n_classes_cell = class_presence.sum(axis=1)
            multi = n_classes_cell > 1
            pair_counts = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
            for a in range(int(num_classes)):
                for b in range(a + 1, int(num_classes)):
                    pair_counts[a, b] = int(np.sum(class_presence[:, a] & class_presence[:, b]))
        else:
            flat = np.empty(0, dtype=np.int64)
            unique_cells = np.empty(0, dtype=np.int64)
            per_cell = np.empty(0, dtype=np.int64)
            class_presence = np.zeros((0, int(num_classes)), dtype=bool)
            multi = np.empty(0, dtype=bool)
            pair_counts = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)

        policies: Dict[str, Any] = {}
        for policy in ("first", "last", "majority", "highest_z"):
            _, selected = _bev_policy_labels(
                flat, lab, c[:, 2] if c.size else np.empty(0), policy=policy, num_classes=num_classes
            ) if flat.size else (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
            selected_counts = _class_counts(selected, num_classes, ignore_index)
            positive_cells = class_presence.sum(axis=0).astype(np.int64)
            policies[policy] = {
                "selected_cell_class_counts": selected_counts.tolist(),
                "class_voxel_retention": [
                    _safe_ratio(selected_counts[k], in_class[k]) for k in range(int(num_classes))
                ],
                "class_positive_cell_recall": [
                    _safe_ratio(selected_counts[k], positive_cells[k]) for k in range(int(num_classes))
                ],
                "utility_min_retention": min(
                    (
                        _safe_ratio(selected_counts[k], positive_cells[k])
                        for k in utility
                        if positive_cells[k] > 0
                    ),
                    default=float("nan"),
                ),
            }

        zslice_collision = float("nan")
        if flat.size and int(z_slices) > 1:
            if height_slicing == "quantile":
                edges = np.quantile(c[:, 2], np.linspace(0.0, 1.0, int(z_slices) + 1))
                zs = np.searchsorted(edges[1:-1], c[:, 2], side="right").astype(
                    np.int64, copy=False
                )
            elif z1 > z0:
                zs = np.clip(
                    np.floor((c[:, 2] - z0) / ((z1 - z0) / int(z_slices))).astype(np.int64),
                    0,
                    int(z_slices) - 1,
                )
            else:
                zs = np.zeros(c.shape[0], dtype=np.int64)
            flat3 = flat * int(z_slices) + zs
            uc3, inv3 = np.unique(flat3, return_inverse=True)
            cp3 = np.zeros((uc3.size, int(num_classes)), dtype=bool)
            cp3[inv3, lab] = True
            zslice_collision = float(np.mean(cp3.sum(axis=1) > 1)) if uc3.size else 0.0

        pair_records = [
            {"class_a": a, "class_b": b, "cells": int(pair_counts[a, b])}
            for a in range(int(num_classes))
            for b in range(a + 1, int(num_classes))
            if pair_counts[a, b] > 0
        ]
        pair_records.sort(key=lambda r: r["cells"], reverse=True)
        logits_bytes = int(num_classes) * height * width * 4
        feature_bytes = int(hidden_dim) * height * width * 4
        rows.append(
            {
                "resolution_m": res,
                "xy_frame": xy_frame,
                "z_filter": z_filter,
                "height_slicing": height_slicing,
                "projection_center_x_m": float(projection_center_xy[0]),
                "projection_center_y_m": float(projection_center_xy[1]),
                "configured_bounds_xyz_m": [x0, x1, y0, y1, z0, z1],
                "observed_z_min_m": float(coords_m[:, 2].min()) if coords_m.size else float("nan"),
                "observed_z_max_m": float(coords_m[:, 2].max()) if coords_m.size else float("nan"),
                "height": height,
                "width": width,
                "input_active_voxels": int(vx["coords"].shape[0]),
                "in_bounds_active_voxels": int(in_bounds.sum()),
                "in_bounds_active_voxel_fraction": _safe_ratio(in_bounds.sum(), valid_label.sum()),
                "class_in_bounds_retention": [
                    _safe_ratio(in_class[k], total_class[k]) for k in range(int(num_classes))
                ],
                "occupied_cells": int(unique_cells.size),
                "multi_class_cell_fraction": float(np.mean(multi)) if multi.size else 0.0,
                "voxels_per_occupied_cell_mean": float(per_cell.mean()) if per_cell.size else 0.0,
                "z_slices": int(z_slices),
                "z_sliced_multi_class_cell_fraction": zslice_collision,
                "policies": policies,
                "multi_label_class_positive_cells": class_presence.sum(axis=0).astype(int).tolist(),
                "top_collision_pairs": pair_records[:20],
                "dense_logits_mib_per_sample": logits_bytes / (1024.0**2),
                "dense_hidden_mib_per_sample": feature_bytes / (1024.0**2),
            }
        )
    return rows
