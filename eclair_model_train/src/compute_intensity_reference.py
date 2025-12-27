# scripts/compute_intensity_reference.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .data_eclair import _load_eclair_split_list, _read_las_arrays, _resolve_pc_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eclair_root", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--meta_filename", default="labels.json")
    ap.add_argument(
        "--allowed_review_categories", nargs="*", default=None
    )  # None = no filtering
    ap.add_argument("--intensity_divisor", type=float, default=65535.0)
    ap.add_argument("--bins", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    eclair_root = Path(args.eclair_root)
    cats: Optional[Sequence[str]] = args.allowed_review_categories

    names = _load_eclair_split_list(
        eclair_root,
        args.split,
        meta_filename=args.meta_filename,
        allowed_review_categories=cats,
    )

    bins = int(args.bins)
    hist = np.zeros((bins,), dtype=np.int64)

    def bin_idx(x01: np.ndarray) -> np.ndarray:
        x01 = np.clip(x01, 0.0, 1.0)
        return np.minimum((x01 * (bins - 1)).astype(np.int64), bins - 1)

    for i, fname in enumerate(names, start=1):
        pc = _resolve_pc_path(eclair_root, fname)
        raw = _read_las_arrays(pc)
        inten = raw["intensity"]
        if inten is None:
            continue
        x01 = (inten / float(args.intensity_divisor)).astype(np.float32, copy=False)
        idx = bin_idx(x01)
        hist += np.bincount(idx, minlength=bins)

        if i % 50 == 0 or i == len(names):
            print(f"[ref] processed {i}/{len(names)} tiles")

    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= max(1.0, cdf[-1])

    # quantiles for probs 0..1
    probs = np.linspace(0.0, 1.0, 1001)
    # invert cdf: find smallest bin where cdf >= p
    q_bins = np.searchsorted(cdf, probs, side="left")
    q_bins = np.clip(q_bins, 0, bins - 1)
    quantiles = (q_bins / float(bins - 1)).astype(np.float32)

    out = {
        "bins": bins,
        "probs": probs.tolist(),
        "quantiles": quantiles.tolist(),  # in [0,1]
        "split": args.split,
        "allowed_review_categories": cats,
        "intensity_divisor": float(args.intensity_divisor),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote reference quantiles -> {args.out}")


if __name__ == "__main__":
    main()
