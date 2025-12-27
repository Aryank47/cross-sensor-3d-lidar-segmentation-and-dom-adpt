import json

import numpy as np
import pandas as pd

raw = pd.read_csv("./logs/domain_eda_out/raw_file_stats.csv")
harm = pd.read_csv("./logs/domain_eda_out/harmonized_patch_stats.csv")
gap = pd.read_csv("./logs/domain_eda_out/domain_gap_ranked.csv")


def safe_load_list(x):
    if not isinstance(x, str):
        return None
    x = x.strip()
    if not x or x.lower() == "nan":
        return None
    try:
        return json.loads(x)
    except:
        return None


raw["intensity_q"] = raw["intensity_q"].apply(safe_load_list)
raw["int_q50"] = raw["intensity_q"].apply(
    lambda a: np.nan if a is None else float(a[2])
)

print("\n=== Domain gap ranking ===")
print(gap)

print("\n=== Intensity median per dataset (raw files) ===")
print(raw.groupby("dataset")["int_q50"].agg(["count", "median", "min", "max"]))

harm["sparse_lt_1000"] = harm["point_count"] < 1000
print("\n=== Fraction of near-empty 100m patches (<1000 pts) ===")
print(harm.groupby("dataset")["sparse_lt_1000"].mean())
