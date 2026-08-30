from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import laspy
import numpy as np
from src.config_loader import load_yaml
from src.data_dales import _find_dales_files

_TILE_RE = re.compile(r"(?P<x>\d+)[_\-](?P<y>\d+)")


def _parse_tile_xy_from_name(path: Path) -> Optional[Tuple[int, int]]:
    """
    DALES files look like: 5080_54400.las -> x0=5080, y0=54400 (grid units).
    Often these map to meters by *100 (based on your EDA mins/maxs).
    We keep raw grid indices; conversion handled later.
    """
    m = _TILE_RE.search(path.stem)
    if not m:
        return None
    return int(m.group("x")), int(m.group("y"))


def _grid_to_meters(xg: int, yg: int) -> Tuple[float, float]:
    """
    Heuristic: DALES naming uses 100m grid units (5080 -> 508000 m).
    If numbers look small (< 1e6), multiply by 100 to get meters.
    """
    scale = 100.0
    return float(xg) * scale, float(yg) * scale


def _native_to_train_counts(
    native_counts_0_255: np.ndarray,
    label_map: Dict[int, int],
    *,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    """
    Map native LAS classification codes -> train IDs via label_map.
    """
    train_counts = np.zeros((num_classes,), dtype=np.int64)
    for native_id, cnt in enumerate(native_counts_0_255.tolist()):
        if cnt <= 0:
            continue
        t = int(label_map.get(int(native_id), ignore_index))
        if t == ignore_index:
            continue
        if 0 <= t < num_classes:
            train_counts[t] += int(cnt)
    return train_counts


def _stream_native_label_hist(
    path: Path, *, chunk_size: int = 2_000_000
) -> Tuple[int, np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Streaming histogram of LAS 'classification' (native labels).
    Returns (point_count, hist[256], mins(x,y,z), maxs(x,y,z))
    """
    hist = np.zeros((256,), dtype=np.int64)
    with laspy.open(str(path)) as f:
        h = f.header
        mins = (float(h.mins[0]), float(h.mins[1]), float(h.mins[2]))
        maxs = (float(h.maxs[0]), float(h.maxs[1]), float(h.maxs[2]))
        n_total = int(h.point_count)

        dim_names = set(f.header.point_format.dimension_names)
        if "classification" not in dim_names:
            raise RuntimeError(f"'classification' dim missing in {path}")

        for points in f.chunk_iterator(chunk_size):
            cls = np.asarray(points["classification"], dtype=np.uint8)
            # bincount to 256
            hist += np.bincount(cls, minlength=256).astype(np.int64)
    return n_total, hist, mins, maxs


@dataclass
class TileRow:
    path: str
    tile_xg: int
    tile_yg: int
    tile_x0_m: float
    tile_y0_m: float
    header_min_x: float
    header_min_y: float
    header_min_z: float
    header_max_x: float
    header_max_y: float
    header_max_z: float
    point_count: int
    # counts per train class
    counts_json: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Your DALES YAML config (for roots + label map).")
    ap.add_argument("--out_csv", required=True, help="Output CSV path.")
    ap.add_argument("--chunk_size", type=int, default=2_000_000)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data = cfg["data"]

    if str(data.get("dataset", "dales")).lower() != "dales":
        raise ValueError("This script expects data.dataset = dales")

    train_root = Path(data["dales_train_root"])
    files = sorted(_find_dales_files(train_root))

    label_map_cfg = data.get("dales_label_map_native_to_train", None)
    if label_map_cfg is None:
        raise ValueError("data.dales_label_map_native_to_train is required.")
    label_map = {int(k): int(v) for k, v in label_map_cfg.items()}

    num_classes = int(data["label_space"]["num_classes"])
    ignore_index = int(data["label_space"]["ignore_index"])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[TileRow] = []

    # global counts (useful sanity check)
    global_train_counts = np.zeros((num_classes,), dtype=np.int64)

    for i, p in enumerate(files, start=1):
        # parse name -> grid
        xy = _parse_tile_xy_from_name(p)
        if xy is None:
            # fallback: use header mins, and synthesize grid by /100
            n_total, hist, mins, maxs = _stream_native_label_hist(p, chunk_size=args.chunk_size)
            xg = int(round(mins[0] / 100.0))
            yg = int(round(mins[1] / 100.0))
        else:
            xg, yg = xy
            n_total, hist, mins, maxs = _stream_native_label_hist(p, chunk_size=args.chunk_size)

        x0m, y0m = _grid_to_meters(xg, yg)

        train_counts = _native_to_train_counts(
            hist,
            label_map,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        global_train_counts += train_counts

        row = TileRow(
            path=str(p.resolve()),
            tile_xg=int(xg),
            tile_yg=int(yg),
            tile_x0_m=float(x0m),
            tile_y0_m=float(y0m),
            header_min_x=float(mins[0]),
            header_min_y=float(mins[1]),
            header_min_z=float(mins[2]),
            header_max_x=float(maxs[0]),
            header_max_y=float(maxs[1]),
            header_max_z=float(maxs[2]),
            point_count=int(n_total),
            counts_json=json.dumps({str(ci): int(train_counts[ci]) for ci in range(num_classes)}),
        )
        rows.append(row)

        if i % 25 == 0 or i == len(files):
            print(f"[stats] {i}/{len(files)} processed. last={p.name}", flush=True)

    # write CSV
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "path",
                "tile_xg",
                "tile_yg",
                "tile_x0_m",
                "tile_y0_m",
                "header_min_x",
                "header_min_y",
                "header_min_z",
                "header_max_x",
                "header_max_y",
                "header_max_z",
                "point_count",
                "counts_json",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.path,
                    r.tile_xg,
                    r.tile_yg,
                    r.tile_x0_m,
                    r.tile_y0_m,
                    r.header_min_x,
                    r.header_min_y,
                    r.header_min_z,
                    r.header_max_x,
                    r.header_max_y,
                    r.header_max_z,
                    r.point_count,
                    r.counts_json,
                ]
            )

    tot = max(1, int(global_train_counts.sum()))
    freq = global_train_counts / tot * 100.0
    print("\n[global] train-id counts:", global_train_counts.tolist())
    print("[global] freq%:", [round(float(x), 4) for x in freq.tolist()])
    print(f"[done] wrote stats CSV: {out_csv}")


if __name__ == "__main__":
    main()
