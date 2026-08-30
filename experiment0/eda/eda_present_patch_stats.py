#!/usr/bin/env python3
import json

import numpy as np
import pandas as pd

# -----------------------------
# Mappings (as you defined)
# Common IDs:
# 0 ignore, 1 ground, 2 vegetation, 3 buildings, 4 wires, 5 poles, 6 fence, 7 vehicle
# -----------------------------
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

FOCUS = [4, 5]  # wires, poles
COOC_WITH = [1, 2, 3, 6, 7]  # ground, vegetation, buildings, fence, vehicle

QUANTILES = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]


def parse_hist(s: str) -> dict[int, int]:
    if not isinstance(s, str) or not s or s == "{}":
        return {}
    d = json.loads(s)
    return {int(k): int(v) for k, v in d.items()}


def remap_to_common(hist_native: dict[int, int], mapping: dict[int, int]) -> np.ndarray:
    """Return common_counts[0..7] for this patch."""
    cc = np.zeros(8, dtype=np.int64)
    for nid, cnt in hist_native.items():
        if cnt <= 0:
            continue
        cid = mapping.get(nid, 0)
        if 0 <= cid < 8:
            cc[cid] += cnt
    return cc


def qstats(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0, **{f"q{int(q*100):02d}": np.nan for q in QUANTILES}}
    qs = np.quantile(arr, QUANTILES)
    out = {"n": int(arr.size)}
    for q, v in zip(QUANTILES, qs):
        out[f"q{int(q*100):02d}"] = float(v)
    return out


def main(csv_path: str = "../logs/domain_eda_out/harmonized_patch_stats.csv"):
    df = pd.read_csv(csv_path)

    # outputs
    rows_present = []
    rows_cooc = []

    for ds in ["ECLAIR", "DALES"]:
        sub = df[df["dataset"] == ds].copy()
        mapping = ECLAIR_TO_COMMON if ds == "ECLAIR" else DALES_TO_COMMON

        # Build per-patch common counts matrix: (num_patches, 8)
        common_counts = np.zeros((len(sub), 8), dtype=np.int64)
        for i, s in enumerate(sub["class_hist_nonzero"].values):
            native = parse_hist(s)
            common_counts[i] = remap_to_common(native, mapping)

        # ---- 1) N_c | (N_c > 0) for each class (focus + optionally all) ----
        for c in range(1, 8):
            vals = common_counts[:, c]
            present_vals = vals[vals > 0]
            st = qstats(present_vals)
            rows_present.append(
                {"dataset": ds, "class_id": c, "class": COMMON_NAMES[c], **st}
            )

        # ---- 2) Co-occurrence: P(other present | focus present) ----
        for focus in FOCUS:
            focus_present = common_counts[:, focus] > 0
            denom = int(focus_present.sum())

            for other in COOC_WITH:
                other_present = common_counts[:, other] > 0
                num = int((focus_present & other_present).sum())
                p = (num / denom) if denom > 0 else np.nan

                rows_cooc.append(
                    {
                        "dataset": ds,
                        "focus_class": COMMON_NAMES[focus],
                        "other_class": COMMON_NAMES[other],
                        "num_focus_patches": denom,
                        "num_both_patches": num,
                        "P(other | focus)": float(p) if p == p else np.nan,
                    }
                )

            # also: how many points of "other" when focus present (conditional density)
            for other in COOC_WITH:
                other_counts_when_focus = common_counts[focus_present, other]
                other_counts_when_focus = other_counts_when_focus[
                    other_counts_when_focus > 0
                ]
                st = qstats(other_counts_when_focus)
                rows_cooc.append(
                    {
                        "dataset": ds,
                        "focus_class": COMMON_NAMES[focus],
                        "other_class": COMMON_NAMES[other],
                        "num_focus_patches": denom,
                        "num_both_patches": int(
                            (focus_present & (common_counts[:, other] > 0)).sum()
                        ),
                        "P(other | focus)": (
                            float(
                                (focus_present & (common_counts[:, other] > 0)).sum()
                                / denom
                            )
                            if denom > 0
                            else np.nan
                        ),
                        "note": "counts_of_other_given_focus_present_and_other_present",
                        **st,
                    }
                )

    out_present = pd.DataFrame(rows_present)
    out_cooc = pd.DataFrame(rows_cooc)

    out_present.to_csv("present_patch_counts_quantiles.csv", index=False)
    out_cooc.to_csv("focus_cooccurrence_stats.csv", index=False)

    # Print the key lines for wires/poles quickly
    key = out_present[out_present["class"].isin(["wires", "poles"])].copy()
    key = key[
        ["dataset", "class", "n", "q50", "q90", "q95", "q99", "q100"]
    ].sort_values(["class", "dataset"])
    print("\n=== KEY: N_c | (N_c>0) quantiles (counts per 100m patch) ===")
    print(key.to_string(index=False))

    print(
        "\n[done] wrote: present_patch_counts_quantiles.csv, focus_cooccurrence_stats.csv"
    )


if __name__ == "__main__":
    main()
