import json

import numpy as np
import pandas as pd

ECLAIR_TO_COMMON = {
    0: 0,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 0,
    6: 4,
    7: 4,
    8: 5,
    9: 5,
    10: 6,
    11: 7,
}
DALES_TO_COMMON = {0: 0, 1: 1, 2: 2, 3: 7, 4: 7, 5: 4, 6: 6, 7: 5, 8: 3}

COMMON_NAMES = {
    0: "ignore",
    1: "ground",
    2: "vegetation",
    3: "buildings",
    4: "wires",
    5: "poles",
    6: "fence",
    7: "vehicle",
}


def parse_hist(s: str) -> dict[int, int]:
    if not isinstance(s, str) or not s or s == "{}":
        return {}
    d = json.loads(s)
    return {int(k): int(v) for k, v in d.items()}


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    s = x.sum()
    return x / s if s > 0 else x


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def compute_common_from_patch_stats(csv_path: str) -> None:
    harm = pd.read_csv(csv_path)

    results = []
    all_point_props = {}
    all_presence = {}

    for ds in ["ECLAIR", "DALES"]:
        df = harm[harm["dataset"] == ds].copy()
        mapping = ECLAIR_TO_COMMON if ds == "ECLAIR" else DALES_TO_COMMON

        # point-wise aggregation
        common_counts = np.zeros(8, dtype=np.int64)

        # patch presence
        present = np.zeros((len(df), 8), dtype=bool)

        for i, s in enumerate(df["class_hist_nonzero"].values):
            d = parse_hist(s)
            for nid, cnt in d.items():
                if cnt <= 0:
                    continue
                cid = mapping.get(nid, 0)
                if 0 <= cid < 8:
                    common_counts[cid] += cnt
                    if cid > 0:
                        present[i, cid] = True

        # exclude ignore for point proportions
        point_props = normalize(common_counts[1:])
        presence = present[:, 1:].mean(axis=0)

        all_point_props[ds] = point_props
        all_presence[ds] = presence

        for k in range(1, 8):
            results.append(
                {
                    "dataset": ds,
                    "class_id": k,
                    "class": COMMON_NAMES[k],
                    "points": int(common_counts[k]),
                    "point_%_excl_ignore": float(point_props[k - 1] * 100.0),
                    "patch_presence_%": float(presence[k - 1] * 100.0),
                    "num_patches": int(len(df)),
                }
            )

    # global JS divergence over common priors (excl ignore)
    js = js_divergence(all_point_props["ECLAIR"], all_point_props["DALES"])
    print(f"[common-class] JS divergence over class priors (excl ignore): {js:.6f}")

    out = pd.DataFrame(results)
    out.to_csv("common_class_stats.csv", index=False)

    # add a “comparison view” table
    comp = []
    for k in range(1, 8):
        comp.append(
            {
                "class": COMMON_NAMES[k],
                "ECLAIR_point_%": float(all_point_props["ECLAIR"][k - 1] * 100.0),
                "DALES_point_%": float(all_point_props["DALES"][k - 1] * 100.0),
                "ECLAIR_patch_presence_%": float(all_presence["ECLAIR"][k - 1] * 100.0),
                "DALES_patch_presence_%": float(all_presence["DALES"][k - 1] * 100.0),
                "DALES/ECLAIR_point_%_ratio": float(
                    all_point_props["DALES"][k - 1]
                    / max(all_point_props["ECLAIR"][k - 1], 1e-12)
                ),
            }
        )
    pd.DataFrame(comp).to_csv("common_class_comparison.csv", index=False)

    print("[done] wrote: common_class_stats.csv, common_class_comparison.csv")


if __name__ == "__main__":
    compute_common_from_patch_stats("../logs/domain_eda_out/harmonized_patch_stats.csv")
