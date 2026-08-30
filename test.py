# import argparse
# import logging
# import random
# import sys
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import numpy as np

# try:
#     import laspy
# except ImportError:
#     print("Error: laspy is not installed. Run 'pip install laspy'")
#     sys.exit(1)


# # -----------------------------------------------------------------------------
# # 1. THE ROBUST READER (The Code You Need to Verify)
# # -----------------------------------------------------------------------------
# def _read_las_arrays(path: Path) -> Dict[str, np.ndarray]:
#     """
#     Reads LAS/LAZ files using the robust property-access method to avoid
#     bit-packing bugs and handle missing dimensions safely.
#     """
#     try:
#         las = laspy.read(str(path))
#     except Exception as e:
#         raise RuntimeError(f"Failed to read LAS file {path}: {e}")

#     # Standardize XYZ to float32
#     xyz = np.array(las.xyz, dtype=np.float32)

#     def _get_dim(
#         attr_name: str, fallback_names: List[str] = None
#     ) -> Optional[np.ndarray]:
#         # Priority 1: Direct property access (handles bit-unpacking/scaling)
#         if hasattr(las, attr_name):
#             val = getattr(las, attr_name)
#             return np.array(val)

#         # Priority 2: Dictionary access (fallback for non-standard names)
#         # Check standard dimension names case-insensitively
#         dims_lower = set(d.lower() for d in las.point_format.dimension_names)

#         if attr_name.lower() in dims_lower:
#             return np.array(las[attr_name])

#         if fallback_names:
#             for name in fallback_names:
#                 if name.lower() in dims_lower:
#                     return np.array(las[name])
#         return None

#     # Intensity
#     intensity = _get_dim("intensity")
#     if intensity is None:
#         intensity = np.zeros((xyz.shape[0],), dtype=np.float32)
#     else:
#         intensity = intensity.astype(np.float32)

#     # Returns (CRITICAL: Use property access)
#     rn = _get_dim("return_number")
#     nor = _get_dim("number_of_returns")

#     # Defaults for sparse/simple formats
#     if rn is None:
#         rn = np.ones((xyz.shape[0],), dtype=np.int64)
#     else:
#         rn = rn.astype(np.int64)

#     if nor is None:
#         nor = np.ones((xyz.shape[0],), dtype=np.int64)
#     else:
#         nor = nor.astype(np.int64)

#     # Labels
#     labels = _get_dim("classification", fallback_names=["raw_classification"])
#     if labels is None:
#         # Warn but don't crash for a test script, though training might need labels
#         print(f"  [WARN] No classification found in {path.name}, using zeros.")
#         labels = np.zeros((xyz.shape[0],), dtype=np.int64)
#     else:
#         labels = labels.astype(np.int64)

#     # Color (optional)
#     rgb = None
#     red = _get_dim("red")
#     green = _get_dim("green")
#     blue = _get_dim("blue")

#     if red is not None and green is not None and blue is not None:
#         # Check if 16-bit
#         max_val = max(red.max(), green.max(), blue.max())
#         scale = 1.0
#         if max_val > 255:
#             scale = 1.0 / 65535.0

#         r = red.astype(np.float32) * scale
#         g = green.astype(np.float32) * scale
#         b = blue.astype(np.float32) * scale
#         rgb = np.stack([r, g, b], axis=1)

#     return {
#         "xyz": xyz,
#         "intensity": intensity,
#         "return_number": rn,
#         "number_of_returns": nor,
#         "rgb": rgb,
#         "native_labels": labels,
#     }


# # def _read_las_arrays(path: Path) -> Dict[str, np.ndarray]:

# #     # laspy.read loads all points in memory; ok for ECLAIR tiles
# #     las = laspy.read(str(path))

# #     # xyz float64 -> float32
# #     xyz = las.xyz.astype(np.float32, copy=True)

# #     def _dim(name: str) -> Optional[np.ndarray]:
# #         if name in set(las.point_format.dimension_names):
# #             arr = las[name]
# #             # laspy sometimes returns a SubFieldView
# #             return getattr(arr, "array", arr)
# #         return None

# #     intensity = _dim("intensity")
# #     return_number = _dim("return_number")
# #     number_of_returns = _dim("number_of_returns")

# #     # ECLAIR labels can be in 'classification' or 'raw_classification' depending on export
# #     gt_key = (
# #         "classification"
# #         if "classification" in set(las.point_format.dimension_names)
# #         else "raw_classification"
# #     )
# #     native_labels = _dim(gt_key)

# #     rgb = None
# #     if all(
# #         k in set(las.point_format.dimension_names) for k in ("red", "green", "blue")
# #     ):
# #         r = _dim("red").astype(np.float32)
# #         g = _dim("green").astype(np.float32)
# #         b = _dim("blue").astype(np.float32)
# #         # ECLAIR sometimes stores 16-bit colors; we will scale later if enabled.
# #         rgb = np.stack([r, g, b], axis=1)

# #     if native_labels is None:
# #         raise RuntimeError(
# #             f"Missing classification labels in {path} (looked for 'classification' or 'raw_classification')"
# #         )

# #     return {
# #         "xyz": xyz,
# #         "intensity": intensity.astype(np.float32) if intensity is not None else None,
# #         "return_number": (
# #             return_number.astype(np.int64) if return_number is not None else None
# #         ),
# #         "number_of_returns": (
# #             number_of_returns.astype(np.int64)
# #             if number_of_returns is not None
# #             else None
# #         ),
# #         "rgb": rgb,
# #         "native_labels": native_labels.astype(np.int64),
# #     }


# # -----------------------------------------------------------------------------
# # 2. STATISTICAL VALIDATION LOGIC
# # -----------------------------------------------------------------------------
# def analyze_dataset(name: str, files: List[Path], n_samples: int):
#     print(f"\n{'='*60}")
#     print(f"ANALYZING DATASET: {name}")
#     print(f"{'='*60}")

#     if not files:
#         print("  No files found.")
#         return

#     # Sample files if too many
#     if len(files) > n_samples:
#         selected = random.sample(files, n_samples)
#         print(f"  Selected {n_samples} random files from {len(files)} total.")
#     else:
#         selected = files
#         print(f"  Analyzing all {len(files)} files.")

#     # Aggregate stats
#     stats = {
#         "return_values": set(),
#         "nor_values": set(),
#         "intensity_min": float("inf"),
#         "intensity_max": float("-inf"),
#         "has_rgb_count": 0,
#         "label_values": set(),
#         "points_total": 0,
#     }

#     errors = []

#     for p in selected:
#         print(f"  Reading: {p.name} ... ", end="")
#         try:
#             data = _read_las_arrays(p)
#             n = data["xyz"].shape[0]
#             stats["points_total"] += n

#             # 1. CHECK RETURNS (The Bug Finder)
#             rn = data["return_number"]
#             nor = data["number_of_returns"]

#             rn_uniq = np.unique(rn)
#             nor_uniq = np.unique(nor)

#             stats["return_values"].update(rn_uniq.tolist())
#             stats["nor_values"].update(nor_uniq.tolist())

#             # Assertion 1: Returns should be small integers
#             if rn.max() > 15:
#                 raise ValueError(
#                     f"Suspicious Return Number max={rn.max()}. Bit-packing bug likely!"
#                 )
#             if nor.max() > 15:
#                 raise ValueError(
#                     f"Suspicious Number of Returns max={nor.max()}. Bit-packing bug likely!"
#                 )
#             if (
#                 rn.min() < 0
#             ):  # Should be at least 1 usually, 0 strictly implies undefined
#                 raise ValueError(f"Negative Return Number found!")

#             # 2. CHECK INTENSITY
#             inten = data["intensity"]
#             stats["intensity_min"] = min(stats["intensity_min"], float(inten.min()))
#             stats["intensity_max"] = max(stats["intensity_max"], float(inten.max()))

#             # 3. CHECK RGB
#             if data["rgb"] is not None:
#                 stats["has_rgb_count"] += 1
#                 if data["rgb"].max() > 1.0 + 1e-6:
#                     raise ValueError(
#                         f"RGB not normalized! Max value: {data['rgb'].max()}"
#                     )

#             # 4. CHECK LABELS
#             lbls = data["native_labels"]
#             stats["label_values"].update(np.unique(lbls).tolist())

#             # 5. CHECK DIMENSIONS
#             if not (n == len(inten) == len(rn) == len(nor) == len(lbls)):
#                 raise ValueError(
#                     f"Array length mismatch! xyz={n}, inten={len(inten)}, rn={len(rn)}"
#                 )

#             print("OK")

#         except Exception as e:
#             print("FAILED")
#             errors.append((p.name, str(e)))

#     # --- REPORT ---
#     print(f"\n  --- {name} SUMMARY ---")
#     if errors:
#         print(f"  [!!] ERRORS FOUND IN {len(errors)} FILES:")
#         for fname, err in errors:
#             print(f"    - {fname}: {err}")
#     else:
#         print("  [OK] No read errors or sanity check failures.")

#     print(f"  Total Points Read: {stats['points_total']}")

#     # Returns Report
#     rn_sorted = sorted(list(stats["return_values"]))
#     nor_sorted = sorted(list(stats["nor_values"]))
#     print(f"  Return Numbers Found: {rn_sorted}")
#     print(f"  Num of Returns Found: {nor_sorted}")

#     if max(rn_sorted) > 7:
#         print(
#             "  [WARNING] Return numbers > 7. This is technically valid for LAS 1.4 but rare. Verify if expected."
#         )
#     else:
#         print("  [PASS] Return numbers are within standard range (1-7).")

#     # Intensity Report
#     print(
#         f"  Intensity Range: [{stats['intensity_min']:.2f}, {stats['intensity_max']:.2f}]"
#     )
#     if name.upper() == "DALES" and stats["intensity_max"] <= 0.0:
#         print(
#             "  [INFO] DALES intensity is all zero. This is expected if source files lack intensity."
#         )
#     elif stats["intensity_max"] > 0:
#         print("  [PASS] Intensity data detected.")

#     # RGB Report
#     print(f"  Files with RGB: {stats['has_rgb_count']} / {len(selected)}")

#     # Label Report
#     print(f"  Unique Labels Found: {sorted(list(stats['label_values']))}")


# # -----------------------------------------------------------------------------
# # 3. MAIN RUNNER
# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Verify LAS/LAZ reader implementation."
#     )
#     parser.add_argument(
#         "--dales_dir", type=str, help="Directory containing DALES .las files"
#     )
#     parser.add_argument(
#         "--eclair_dir", type=str, help="Directory containing ECLAIR .laz files"
#     )
#     parser.add_argument(
#         "--samples", type=int, default=10, help="Number of files to check per dataset"
#     )

#     args = parser.parse_args()

#     # Check ECLAIR
#     if args.eclair_dir:
#         p = Path(args.eclair_dir)
#         files = sorted(list(p.rglob("*.laz")) + list(p.rglob("*.las")))
#         analyze_dataset("ECLAIR", files, args.samples)
#     else:
#         print("Skipping ECLAIR (no --eclair_dir provided)")

#     # Check DALES
#     if args.dales_dir:
#         p = Path(args.dales_dir)
#         files = sorted(list(p.rglob("*.laz")) + list(p.rglob("*.las")))
#         analyze_dataset("DALES", files, args.samples)
#     else:
#         print("Skipping DALES (no --dales_dir provided)")


from pathlib import Path

import numpy as np
import torch


def stats(name, v, max_uniques=20):
    # Convert to numpy for uniform handling
    if torch.is_tensor(v):
        a = v.detach().cpu().numpy()
        src = "torch"
    else:
        a = np.asarray(v)
        src = "numpy/other"

    print(f"\n== {name} ==")
    print(f"source={src} shape={a.shape} dtype={a.dtype}")

    if a.size == 0:
        print("EMPTY")
        return

    # numeric stats
    if np.issubdtype(a.dtype, np.number):
        print(f"min={a.min()} max={a.max()} mean={a.mean()}")
    else:
        print("non-numeric dtype; skipping min/max/mean")

    # uniques / distribution (only if 1D)
    if a.ndim == 1:
        u = np.unique(a)
        if u.size <= max_uniques:
            print("uniques:", u)
        else:
            print(f"uniques_count={u.size} head={u[:max_uniques]}")

        # Small histogram for int-like things
        if np.issubdtype(a.dtype, np.integer):
            vals, cnt = np.unique(a, return_counts=True)
            pairs = list(zip(vals.tolist(), cnt.tolist()))
            # print only first few pairs if huge
            print("value->count (head):", pairs[:30])


def check_returns(rn, nor):
    rn = np.asarray(rn)
    nor = np.asarray(nor)

    print("\n==== Return-number sanity rules ====")
    print(
        "Rule-of-thumb expected (LAS decoded): rn in [1..7], nor in [1..7], and rn <= nor"
    )

    bad_rn = np.sum((rn < 0) | (rn > 7))
    bad_nor = np.sum((nor < 0) | (nor > 7))
    bad_rel = np.sum(rn > nor)

    print(f"bad_rn_outside_0..7: {bad_rn} / {rn.size}")
    print(f"bad_nor_outside_0..7: {bad_nor} / {nor.size}")
    print(f"rn_gt_nor: {bad_rel} / {rn.size}")

    # If you see many values > 7 (like 9, 17, 33, 65...), that screams packed-byte bug.
    # Show a few offenders:
    offenders = rn[(rn > 7) | (rn < 0)]
    if offenders.size:
        print("rn offenders (head):", offenders[:20])
    offenders2 = nor[(nor > 7) | (nor < 0)]
    if offenders2.size:
        print("nor offenders (head):", offenders2[:20])


def main(pt_path):
    obj = torch.load(pt_path, map_location="cpu")
    print("Loaded:", pt_path)
    print("Type:", type(obj))
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict cache, got {type(obj)}")

    print("Dict keys:", list(obj.keys()))

    # print stats for each key
    for k in obj.keys():
        stats(k, obj[k])

    # extra checks for returns
    if "return_number" in obj and "number_of_returns" in obj:
        check_returns(obj["return_number"], obj["number_of_returns"])

    # intensity special-case sanity
    if "intensity" in obj:
        inten = np.asarray(obj["intensity"])
        print("\n==== Intensity quick check ====")
        print("unique intensity count:", np.unique(inten).size)
        print("unique intensity head:", np.unique(inten)[:10])


if __name__ == "__main__":
    main("/scratch/m23csa510/eclair_cache/train/pointcloud_69.pt")
