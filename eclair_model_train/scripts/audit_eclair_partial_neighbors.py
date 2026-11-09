"""Inventory actual held-out neighboring points, with a provenance stop before inference."""

import json
from collections import Counter
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

from neighbor_context_coverage import ARCHIVE, CELL, CLASS_NAMES, LUT, MANIFEST_HASH, read, tile_info
from run_scale_signal_probe import sha


DISTANCE_BANDS = [2, 4, 8, 16, 32]


def load_anchors(index, identity):
    stem = ARCHIVE / f"geometry/eclair_test_{index:04d}"
    marker = read(stem.with_suffix(".json"))
    path = stem.with_suffix(".npz")
    if marker["identity"] != identity or marker["artifacts"] != [
        {"path": str(path.relative_to(ARCHIVE)), "sha256": sha(path)}
    ]:
        raise ValueError(f"Unexpected anchor archive {path}")
    with np.load(path, allow_pickle=False) as z:
        keep = ((z["flags"] & 1) != 0) & (z["gt"] > 0)
        return z["ids"][keep], z["xyz"][keep], z["gt"][keep]


def neighbor_tree(info):
    data = laspy.read(info["path"])
    xyz = np.asarray(data.xyz, dtype=np.float64)
    if len(xyz) != info["points"] or not np.isfinite(xyz).all():
        raise ValueError(f"Unexpected raw tile content {info['path']}")
    if not (np.allclose(xyz.min(0)[:2], info["lower"], rtol=0, atol=1e-8)
            and np.allclose(xyz.max(0)[:2], info["upper"], rtol=0, atol=1e-8)):
        raise ValueError(f"Unexpected raw tile bounds {info['path']}")
    return cKDTree(xyz), data


def verify_anchor_coordinates(index, info, identity):
    ids, local, gt = load_anchors(index, identity)
    raw = laspy.read(info["path"])
    xyz = np.asarray(raw.xyz, dtype=np.float64)
    global_points = xyz[ids]
    if (not np.array_equal(LUT[np.asarray(raw.classification)[ids]], gt)
            or not np.allclose(global_points - xyz.min(0), local, rtol=0, atol=1e-8)):
        raise ValueError(f"Saved points and labels do not align with {info['path']}")
    return global_points, gt


def main():
    path = ARCHIVE / "study_manifest.json"
    if sha(path) != MANIFEST_HASH:
        raise ValueError("Frozen study manifest changed")
    manifest = read(path)
    status = read(ARCHIVE / "STATUS.json")
    if status["identity"] != manifest["identity"] or not status["all_required_stages_complete"]:
        raise ValueError("Canonical archive is incomplete")
    members = manifest["datasets"]["eclair/test"]["members"]
    rows = [tile_info(member) for member in members]
    cells = {tuple(row["cell"]): i for i, row in enumerate(rows)}
    if len(cells) != len(rows) or len(rows) != 125:
        raise ValueError("Unexpected ECLAIR test grid")
    incoming = {}
    for i, row in enumerate(rows):
        a, b = row["cell"]
        incoming[i] = sorted(cells[k] for k in ((a+x, b+y) for x in (-1, 0, 1) for y in (-1, 0, 1)
                                                if x or y) if k in cells)
    nearest, labels, distances_by_pair = {}, {}, []
    provenance = Counter()
    for j, row in enumerate(rows):
        recipients = incoming[j]
        if not recipients:
            continue
        tree, raw = neighbor_tree(row)
        for i in recipients:
            # Reopening a recipient bounds memory to two raw tiles at any time.
            points, gt = verify_anchor_coordinates(i, rows[i], manifest["identity"])
            dist, _ = tree.query(points)
            nearest[i] = np.minimum(nearest.get(i, np.full(len(points), np.inf)), dist)
            labels[i] = gt
            peer = rows[i]
            pair = ("missing_crs" if not (row["crs_record"] and peer["crs_record"]) else
                    "missing_project_or_source_id" if not (row["source_id"] and peer["source_id"]
                    and int(row["project_id"].replace("-", ""), 16)
                    and int(peer["project_id"].replace("-", ""), 16)) else
                    "acquisition_not_independently_verified")
            provenance[pair] += 1
            distances_by_pair.append({
                "recipient": i, "neighbor": j, "verification": pair,
                "sample_within_m": [np.bincount(gt[dist < r], minlength=8).tolist() for r in DISTANCE_BANDS],
            })
        del tree, raw
        if (j + 1) % 25 == 0:
            print(f"Read candidate neighborhoods through tile {j + 1}", flush=True)
    tiles = []
    for i, row in enumerate(rows):
        tiles.append({
            "tile": Path(row["path"]).name, "neighbors": incoming[i],
            "sample_within_m": [np.bincount(labels[i][nearest[i] < r], minlength=8).tolist()
                                for r in DISTANCE_BANDS] if i in nearest else [[0] * 8 for _ in DISTANCE_BANDS],
        })
    result = {
        "manifest_sha256": MANIFEST_HASH, "grid_cell_m": CELL, "distance_bands_m": DISTANCE_BANDS,
        "counts_are": "Unweighted archived sample points near real neighboring XYZ; adjacency and numeric alignment only.",
        "class_names": CLASS_NAMES, "candidate_directed_pairs": len(distances_by_pair),
        "provenance_counts": dict(provenance), "verified_acquisition_pairs": 0,
        "decision": "unresolved_neighbor_acquisition_provenance" if distances_by_pair else "no_adjacent_test_tiles",
        "checkpoint_inference": "not_run",
        "tiles": tiles, "pairs": distances_by_pair,
    }
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
