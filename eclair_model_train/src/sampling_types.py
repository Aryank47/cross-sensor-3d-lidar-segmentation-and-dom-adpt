from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

COMMON_IGNORE_ID = 0
COMMON_CLASS_NAMES: Dict[int, str] = {
    1: "ground",
    2: "vegetation",
    3: "buildings",
    4: "wires",
    5: "poles",
    6: "fence",
    7: "vehicle",
}


@dataclass(frozen=True)
class TileRecord:
    dataset: str
    tile_index: int
    tile_id: str
    path: Path


@dataclass(frozen=True)
class Footprint:
    width_m: float
    height_m: float
    source: str
    observed_span_x_m: float
    observed_span_y_m: float
    # Lower-left corner in the coordinate frame returned by get_raw().
    x_min_m: float = 0.0
    y_min_m: float = 0.0


@dataclass(frozen=True)
class WindowSpec:
    dataset: str
    tile_index: int
    tile_id: str
    tile_path: str
    cohort: str
    selection_reason: str
    window_size_m: float
    x_min_m: float
    y_min_m: float
    candidate_id: int
    focus_common_id: Optional[int] = None

    @property
    def x_max_m(self) -> float:
        return self.x_min_m + self.window_size_m

    @property
    def y_max_m(self) -> float:
        return self.y_min_m + self.window_size_m

    @property
    def area_m2(self) -> float:
        return self.window_size_m * self.window_size_m

    @property
    def key(self) -> str:
        focus = "none" if self.focus_common_id is None else str(self.focus_common_id)
        return (
            f"{self.dataset}|{self.tile_path}|{self.cohort}|{focus}|"
            f"{self.window_size_m:.6f}|{self.x_min_m:.6f}|{self.y_min_m:.6f}|"
            f"{self.candidate_id}"
        )


@dataclass
class DatasetBundle:
    name: str
    training_config_path: Path
    training_config: Dict[str, Any]
    dataset: Any
    tile_records: list[TileRecord]
    native_to_common_lut: np.ndarray
    train_to_common_lut: np.ndarray
    expected_native_ids: set[int]
    native_class_names: Mapping[int, str]
    patch_cfg: Any
    feat_cfg: Any
    voxel_cfg: Any
    ignore_index: int
    num_train_classes: int
    dataset_kind: str
    source_split: str

    def raw_to_common(self, native_labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(native_labels, dtype=np.int64)
        if labels.size == 0:
            return np.empty((0,), dtype=np.int64)
        if labels.min() < 0 or labels.max() >= self.native_to_common_lut.size:
            raise ValueError(
                f"{self.name}: native label out of LUT range: "
                f"min={int(labels.min())}, max={int(labels.max())}, "
                f"lut_size={self.native_to_common_lut.size}"
            )
        return self.native_to_common_lut[labels]

    def train_to_common(self, train_labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(train_labels, dtype=np.int64)
        out = np.full(labels.shape, COMMON_IGNORE_ID, dtype=np.int64)
        valid = labels != int(self.ignore_index)
        if np.any(valid):
            valid_values = labels[valid]
            if valid_values.min() < 0 or valid_values.max() >= self.train_to_common_lut.size:
                raise ValueError(
                    f"{self.name}: train label out of LUT range: "
                    f"min={int(valid_values.min())}, max={int(valid_values.max())}, "
                    f"lut_size={self.train_to_common_lut.size}"
                )
            out[valid] = self.train_to_common_lut[valid_values]
        return out
