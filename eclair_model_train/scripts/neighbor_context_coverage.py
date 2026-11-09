"""Stage 1. Bound attainable context coverage using frozen ECLAIR test tiles."""

import hashlib
import json
from pathlib import Path

import laspy
import numpy as np


ARCHIVE = Path("/scratch/m23csa510/e0_results/transfer_evidence/transfer_evidence_v1")
MANIFEST_HASH = "ade55826fd93d1dbc984f10fb752e8c7cabeb3d6f0e033d63478a5835ba409de"
RADIUS = 16
CELL = 100
CLASS_NAMES = ("ignored", "ground", "vegetation", "buildings", "wires", "poles", "fence", "vehicles")
LUT = np.asarray([0, 0, 1, 2, 3, 0, 4, 4, 5, 5, 6, 7])
ROUTES = ((3, (1, 2)), (6, (2, 3)), (7, (2, 6)), (5, (2,)), (4, (5,)))


def read(path):
    return json.loads(path.read_bytes())


def required_cells(xy, cell, present):
    """Optimistic full disk coverage from nominal grid cells, without reading neighbors."""
    possible = np.ones(len(xy), dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            key = (cell[0] + dx, cell[1] + dy)
            if key in present:
                continue
            lower = np.asarray(key) * CELL
            separation = np.maximum(np.maximum(lower - xy, xy - (lower + CELL)), 0)
            possible &= np.einsum("ij,ij->i", separation, separation) >= RADIUS**2
    return possible


def tile_info(member):
    path = Path(member["path"])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (member["bytes"], member["mtime_ns"]):
        raise ValueError(f"Frozen test tile changed: {path}")
    with laspy.open(path) as f:
        h = f.header
        lower, upper = h.mins[:2], h.maxs[:2]
        cell = tuple(int(value) for value in np.floor(lower / CELL))
        if not np.all(upper <= (np.asarray(cell) + 1) * CELL):
            raise ValueError(f"Tile does not fit the nominal grid: {path}")
        return {
            "path": str(path), "cell": cell, "lower": lower.tolist(), "upper": upper.tolist(),
            "points": h.point_count, "system": h.system_identifier,
            "project_id": str(h.uuid), "source_id": h.file_source_id,
            "crs_record": any(v.user_id in {"LASF_Projection", "liblas"} for v in h.vlrs),
        }


def scan_tile(info, present, wanted):
    names = ("all", "inside_raw_tile", "possibly_complete_with_test_neighbors",
             "possibly_recovered_by_test_neighbors")
    counts = {name: np.zeros(8, dtype=int) for name in names}
    lower, upper = np.asarray(info["lower"]), np.asarray(info["upper"])
    seen_min, seen_max = np.full(2, np.inf), np.full(2, -np.inf)
    with laspy.open(info["path"]) as reader:
        for points in reader.chunk_iterator(500_000):
            xy = np.column_stack((points.x, points.y))
            seen_min = np.minimum(seen_min, xy.min(0))
            seen_max = np.maximum(seen_max, xy.max(0))
            gt = LUT[np.asarray(points.classification)]
            counts["all"] += np.bincount(gt, minlength=8)
            keep = np.isin(gt, wanted)
            xy, gt = xy[keep], gt[keep]
            inside = np.minimum(xy - lower, upper - xy).min(1) >= RADIUS
            possible = required_cells(xy, tuple(info["cell"]), present)
            for name, mask in zip(names[1:], (inside, possible, possible & ~inside)):
                counts[name] += np.bincount(gt[mask], minlength=8)
    if (counts["all"].sum() != info["points"] or not np.allclose(seen_min, lower, rtol=0, atol=1e-8)
            or not np.allclose(seen_max, upper, rtol=0, atol=1e-8)):
        raise ValueError(f"Raw point count or bounds differ from header: {info['path']}")
    return {key: value.tolist() for key, value in counts.items()}


def route_support(rows, key):
    by_tile = np.asarray([row["counts"][key] for row in rows])
    result = []
    for pos, neg in ROUTES:
        sides = []
        for values in (by_tile[:, pos], by_tile[:, list(neg)].sum(1)):
            sides.append({"points": int(values.sum()), "tiles": int((values > 0).sum())})
        result.append({"class": CLASS_NAMES[pos], "sides": sides,
                       "pass": all(s["points"] >= 100 and s["tiles"] >= 5 for s in sides)})
    return result


def coverage():
    path = ARCHIVE / "study_manifest.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != MANIFEST_HASH:
        raise ValueError("Unexpected frozen manifest")
    manifest = read(path)
    identity = manifest["identity"]
    status = read(ARCHIVE / "STATUS.json")
    if status["identity"] != identity or not status["all_required_stages_complete"]:
        raise ValueError("Canonical archive is incomplete")
    members = manifest["datasets"]["eclair/test"]["members"]
    if len(members) != 125:
        raise ValueError("Unexpected test inventory")
    rows = [tile_info(member) for member in members]
    cells = {tuple(row["cell"]): i for i, row in enumerate(rows)}
    if len(cells) != len(rows):
        raise ValueError("Multiple test tiles occupy the same nominal grid cell")
    for i, row in enumerate(rows):
        a, b = row["cell"]
        row["grid_neighbors"] = sorted(cells[k] for k in
                                      ((a+x, b+y) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y)
                                      if k in cells)
        marker = read(ARCHIVE / f"test/eclair_{i:04d}.json")
        if marker["identity"] != identity:
            raise ValueError(f"Unexpected archive tile identity: {i}")
        expected = np.asarray(marker["models"]["dales_29364"]["support"])
        row["counts"] = scan_tile(row, cells, np.flatnonzero(expected[1:]) + 1)
        if not np.array_equal(expected[1:], row["counts"]["all"][1:]):
            raise ValueError(f"Raw class counts disagree with canonical archive: {i}")
        if (i + 1) % 25 == 0:
            print(f"Checked {i + 1}/{len(rows)} test tiles", flush=True)
    candidate_support = route_support(rows, "possibly_complete_with_test_neighbors")
    recovered_support = route_support(rows, "possibly_recovered_by_test_neighbors")
    return {
        "manifest_sha256": MANIFEST_HASH, "radius_m": RADIUS, "cell_size_m": CELL,
        "class_names": CLASS_NAMES, "coverage_is_upper_bound": True,
        "coverage_basis": "Each frozen test grid cell is treated as completely filled; no validation or train data.",
        "same_acquisition_verified": False,
        "same_acquisition_reason": "Grid adjacency alone does not verify acquisition provenance.",
        "inside_raw_tile_support": route_support(rows, "inside_raw_tile"),
        "optimistic_complete_support": candidate_support,
        "optimistic_recovered_only_support": recovered_support,
        "all_routes_possible": all(row["pass"] for row in candidate_support),
        "status": "needs_verified_neighbor_coverage" if all(row["pass"] for row in candidate_support)
                  else "stopped_insufficient_test_coverage",
        "stage_2": "not_run",
        "stage_3": "not_run",
        "tiles": rows,
    }


def self_check():
    xy = np.asarray([[50., 50.], [99., 50.], [99., 99.], [85., 85.]])
    origin = (0, 0)
    assert required_cells(xy, origin, {origin}).tolist() == [True, False, False, False]
    assert required_cells(xy, origin, {origin, (1, 0), (0, 1)}).tolist() == [True, True, False, True]
    assert required_cells(xy, origin, {origin, (1, 0), (0, 1), (1, 1)}).all()
    left_tangent = np.asarray([[16., 50.], [15.999, 50.]])
    assert required_cells(left_tangent, origin, {origin}).tolist() == [True, False]


if __name__ == "__main__":
    self_check()
    print(json.dumps(coverage(), allow_nan=False), flush=True)
