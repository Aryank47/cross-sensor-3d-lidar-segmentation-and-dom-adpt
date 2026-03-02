#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from src.config_loader import load_yaml
from src.data_dales import _find_dales_files


def _read_stats_csv(path: Path) -> List[dict]:
    import csv

    rows = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            row["tile_xg"] = int(row["tile_xg"])
            row["tile_yg"] = int(row["tile_yg"])
            row["tile_x0_m"] = float(row["tile_x0_m"])
            row["tile_y0_m"] = float(row["tile_y0_m"])
            row["point_count"] = int(row["point_count"])
            row["counts_json"] = json.loads(row["counts_json"])
            rows.append(row)
    return rows


def _neighbors4(b: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = b
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


@dataclass
class Block:
    bx: int
    by: int
    tile_paths: List[str]
    tile_count: int
    counts: np.ndarray  # [C]
    points: int


def _weighted_prop_distance(p: np.ndarray, q: np.ndarray, w: np.ndarray) -> float:
    # L1 distance on proportions with weights
    return float(np.sum(w * np.abs(p - q)))


def _safe_props(counts: np.ndarray) -> np.ndarray:
    s = float(np.sum(counts))
    if s <= 0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts.astype(np.float64) / s


def _build_blocks(rows: List[dict], *, num_classes: int, block_size_m: float) -> Dict[Tuple[int, int], Block]:
    tmp_tiles = defaultdict(list)
    tmp_counts = defaultdict(lambda: np.zeros((num_classes,), dtype=np.int64))
    tmp_points = defaultdict(int)

    for r in rows:
        x0 = float(r["tile_x0_m"])
        y0 = float(r["tile_y0_m"])
        bx = int(math.floor(x0 / block_size_m))
        by = int(math.floor(y0 / block_size_m))

        key = (bx, by)
        tmp_tiles[key].append(r["path"])
        tmp_points[key] += int(r["point_count"])

        c = np.zeros((num_classes,), dtype=np.int64)
        for k, v in r["counts_json"].items():
            ci = int(k)
            if 0 <= ci < num_classes:
                c[ci] = int(v)
        tmp_counts[key] += c

    blocks = {}
    for (bx, by), paths in tmp_tiles.items():
        blocks[(bx, by)] = Block(
            bx=bx,
            by=by,
            tile_paths=paths,
            tile_count=len(paths),
            counts=tmp_counts[(bx, by)],
            points=tmp_points[(bx, by)],
        )
    return blocks


def _region_grow_best(
    *,
    blocks: Dict[Tuple[int, int], Block],
    seed_block: Tuple[int, int],
    target_tiles: int,
    target_prop: np.ndarray,
    class_weights: np.ndarray,
    required_min_points: Dict[int, int],
) -> Optional[Set[Tuple[int, int]]]:
    if seed_block not in blocks:
        return None

    region: Set[Tuple[int, int]] = set([seed_block])
    frontier: Set[Tuple[int, int]] = set([nb for nb in _neighbors4(seed_block) if nb in blocks and nb not in region])

    # running totals
    tiles = blocks[seed_block].tile_count
    counts = blocks[seed_block].counts.copy()

    # if target_tiles==1 this is enough
    while tiles < target_tiles:
        if not frontier:
            break

        best_nb = None
        best_score = None  # higher is better improvement

        cur_prop = _safe_props(counts)
        cur_dist = _weighted_prop_distance(cur_prop, target_prop, class_weights)

        for nb in list(frontier):
            b = blocks[nb]
            new_counts = counts + b.counts
            new_prop = _safe_props(new_counts)
            new_dist = _weighted_prop_distance(new_prop, target_prop, class_weights)
            improvement = cur_dist - new_dist

            # tie-break: prefer adding fewer tiles to hit target smoothly
            # and prefer blocks with more rare points (implicitly via weights)
            if (best_score is None) or (improvement > best_score):
                best_score = improvement
                best_nb = nb

        if best_nb is None:
            break

        region.add(best_nb)
        b = blocks[best_nb]
        tiles += b.tile_count
        counts += b.counts

        frontier.remove(best_nb)
        for nb2 in _neighbors4(best_nb):
            if nb2 in blocks and nb2 not in region:
                frontier.add(nb2)

        # stop if we overshoot too much (optional)
        if tiles >= target_tiles:
            break

    # constraint check: required_min_points for some classes
    for cls, minp in required_min_points.items():
        if int(counts[cls]) < int(minp):
            return None

    return region


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="DALES YAML config (for num_classes + roots + label map).")
    ap.add_argument("--stats_csv", required=True, help="CSV from dales_compute_tile_stats.py")
    ap.add_argument("--out_dir", required=True, help="Output directory for split manifests.")
    ap.add_argument("--val_fraction", type=float, default=0.10)
    ap.add_argument("--block_size_m", type=float, default=2000.0, help="Spatial block size in meters (e.g., 2000 = 2km).")
    ap.add_argument("--seed_trials", type=int, default=50, help="How many seed blocks to try.")
    ap.add_argument("--weight_beta", type=float, default=0.5, help="Class weighting: w ~ 1/(freq^beta).")
    ap.add_argument("--min_points", type=str, default="5:5000,6:5000", help="Min points constraints e.g. '5:5000,6:5000'")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data = cfg["data"]
    if str(data.get("dataset", "dales")).lower() != "dales":
        raise ValueError("This split builder expects data.dataset=dales")

    num_classes = int(data["label_space"]["num_classes"])

    rows = _read_stats_csv(Path(args.stats_csv))
    if len(rows) == 0:
        raise RuntimeError("No rows found in stats CSV.")

    # global target distribution across TRAIN ROOT
    global_counts = np.zeros((num_classes,), dtype=np.int64)
    for r in rows:
        for k, v in r["counts_json"].items():
            ci = int(k)
            if 0 <= ci < num_classes:
                global_counts[ci] += int(v)

    global_prop = global_counts.astype(np.float64) / max(1.0, float(global_counts.sum()))

    # class weights: w ~ 1/(freq^beta)
    eps = 1e-12
    beta = float(args.weight_beta)
    w = 1.0 / np.maximum(eps, global_prop) ** beta
    w = w / np.mean(w)  # normalize average weight ~ 1

    # required min points for key utility classes
    required_min_points: Dict[int, int] = {}
    if args.min_points.strip():
        for kv in args.min_points.split(","):
            kv = kv.strip()
            if not kv:
                continue
            c, p = kv.split(":")
            required_min_points[int(c)] = int(p)

    # blocks
    block_size_m = float(args.block_size_m)
    blocks = _build_blocks(rows, num_classes=num_classes, block_size_m=block_size_m)
    block_keys = list(blocks.keys())

    # target tiles count (based on tiles, not points) — consistent with your previous approach
    total_tiles = sum(b.tile_count for b in blocks.values())
    target_val_tiles = max(1, int(round(total_tiles * float(args.val_fraction))))

    # Choose seed candidates: blocks with most weighted rare mass
    # Score block by sum(w[c] * counts[c])
    scores = []
    for k, b in blocks.items():
        score = float(np.sum(w * b.counts.astype(np.float64)))
        scores.append((score, k))
    scores.sort(reverse=True)
    seed_candidates = [k for _, k in scores[: max(1, int(args.seed_trials))]]

    best_region = None
    best_dist = None
    best_tiles = None

    for si, seed in enumerate(seed_candidates, start=1):
        region = _region_grow_best(
            blocks=blocks,
            seed_block=seed,
            target_tiles=target_val_tiles,
            target_prop=global_prop,
            class_weights=w,
            required_min_points=required_min_points,
        )
        if region is None:
            continue

        # compute region stats
        reg_counts = np.zeros((num_classes,), dtype=np.int64)
        reg_tiles = 0
        for bk in region:
            reg_counts += blocks[bk].counts
            reg_tiles += blocks[bk].tile_count

        reg_prop = reg_counts.astype(np.float64) / max(1.0, float(reg_counts.sum()))
        dist = _weighted_prop_distance(reg_prop, global_prop, w)

        if (best_dist is None) or (dist < best_dist):
            best_dist = dist
            best_region = region
            # best_tiles = reg_tiles

        if si % 10 == 0:
            print(f"[search] tried {si}/{len(seed_candidates)} seeds. current best dist={best_dist:.6f}", flush=True)

    if best_region is None:
        raise RuntimeError("Failed to build a val region satisfying constraints. Relax min_points or try larger val_fraction.")

    # Expand blocks -> tile paths
    val_tiles: List[str] = []
    for bk in sorted(best_region):
        val_tiles.extend(blocks[bk].tile_paths)
    val_set = set(val_tiles)

    # Train tiles are all train-root tiles not in val
    all_tiles = [r["path"] for r in rows]
    train_tiles = [p for p in all_tiles if p not in val_set]

    # Test tiles: always from dales_test_root (official)
    test_files = _find_dales_files(Path(data["dales_test_root"]))
    test_tiles = [str(p.resolve()) for p in test_files]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "train.txt").write_text("\n".join(train_tiles) + "\n")
    (out_dir / "val.txt").write_text("\n".join(val_tiles) + "\n")
    (out_dir / "test.txt").write_text("\n".join(test_tiles) + "\n")

    # Summary
    def counts_for_paths(paths: List[str]) -> np.ndarray:
        m = {r["path"]: r for r in rows}
        csum = np.zeros((num_classes,), dtype=np.int64)
        for p in paths:
            rr = m.get(p)
            if rr is None:
                continue
            for k, v in rr["counts_json"].items():
                ci = int(k)
                if 0 <= ci < num_classes:
                    csum[ci] += int(v)
        return csum

    c_train = counts_for_paths(train_tiles)
    c_val = counts_for_paths(val_tiles)
    tot_train = max(1, int(c_train.sum()))
    tot_val = max(1, int(c_val.sum()))

    summary = {
        "block_size_m": block_size_m,
        "val_fraction_target_tiles": float(args.val_fraction),
        "total_tiles": int(len(all_tiles)),
        "train_tiles": int(len(train_tiles)),
        "val_tiles": int(len(val_tiles)),
        "test_tiles": int(len(test_tiles)),
        "global_freq": (global_counts / max(1, global_counts.sum())).tolist(),
        "train_freq": (c_train / tot_train).tolist(),
        "val_freq": (c_val / tot_val).tolist(),
        "weighted_distance_val_vs_global": float(best_dist),
        "required_min_points": required_min_points,
        "seed_trials": int(args.seed_trials),
        "weight_beta": float(args.weight_beta),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n[done] wrote split manifests:")
    print("  ", out_dir / "train.txt")
    print("  ", out_dir / "val.txt")
    print("  ", out_dir / "test.txt")
    print("  ", out_dir / "summary.json")
    print(f"[val] tiles={len(val_tiles)} (target~{target_val_tiles}), weighted_dist={best_dist:.6f}", flush=True)


if __name__ == "__main__":
    main()
