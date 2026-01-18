# /src/label_maps.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

ECLAIR_CLASS_NAMES_11 = [
    "Unassigned",
    "Ground",
    "Vegetation",
    "Buildings",
    "Noise",
    "Transmission wires",
    "Distribution wires",
    "Poles",
    "Transmission towers",
    "Fence",
    "Vehicle",
]

# DALES has 9 semantic classes (native label ids).
# Fill these with your repo’s naming convention if you already have one elsewhere.
DALES_CLASS_NAMES_9 = [
    "ground",
    "vegetation",
    "cars",
    "trucks",
    "powerlines",
    "poles",
    "fences",
    "buildings",
    "unknown_or_other",
]


@dataclass(frozen=True)
class LabelSpace:
    num_classes: int
    ignore_index: int = -100

    class_names: Optional[List[str]] = None


def eclair_native_to_train_ids(
    native_labels: np.ndarray,
    *,
    undefined_id: int = 0,
    ignore_index: int = -100,
) -> np.ndarray:
    """
    Map ECLAIR native ids to contiguous [0..10] for the 11 classes (1..11),
    and map undefined_id (0) to ignore_index.

    Native:
      0 Undefined -> ignore
      1..11 semantic classes -> shift by -1
    """
    y = native_labels.astype(np.int64, copy=True)
    out = np.full_like(y, fill_value=ignore_index, dtype=np.int64)
    mask = y != undefined_id
    out[mask] = y[mask] - 1
    return out
