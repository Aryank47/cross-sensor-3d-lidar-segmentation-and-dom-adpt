from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import laspy
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism is often slower; enable only if you really want bitwise repeatability.
    torch.backends.cudnn.benchmark = True


@dataclass
class StepTimer:
    t0: float = time.time()
    last: float = time.time()
    steps: int = 0

    def tick(self) -> float:
        now = time.time()
        dt = now - self.last
        self.last = now
        self.steps += 1
        return dt

    def elapsed(self) -> float:
        return time.time() - self.t0


class CSVLogger:
    """Append-only CSV logger (one row per epoch)."""

    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._initialized = self.path.exists()
        if not self._initialized:
            with self.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            self._initialized = True

    def log(self, row: Dict[str, Any]) -> None:
        # Ensure stable columns
        out = {k: row.get(k, "") for k in self.fieldnames}
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(out)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def atomic_save_torch(state: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def format_seconds(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    r = s - 60 * m
    if m < 60:
        return f"{m}m{r:.0f}s"
    h = m // 60
    m = m % 60
    return f"{h}h{m}m"


def env_or(default: str, env_key: str) -> str:
    return os.environ.get(env_key, default)


def read_las_arrays_robust(path: Path) -> Dict[str, np.ndarray]:
    """
    Reads LAS/LAZ files using robust property-access to avoid bit-packing bugs.
    Standardizes output keys to: xyz, intensity, return_number, number_of_returns, rgb, native_labels.
    """
    try:
        las = laspy.read(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to read LAS file {path}: {e}")

    # Standardize XYZ to float32
    xyz = np.array(las.xyz, dtype=np.float64)

    def _get_dim(
        attr_name: str, fallback_names: List[str] = None
    ) -> Optional[np.ndarray]:
        # Priority 1: Direct property access (handles bit-unpacking/scaling)
        if hasattr(las, attr_name):
            val = getattr(las, attr_name)
            return np.array(val)

        # Priority 2: Dictionary access (fallback for non-standard names)
        # Check standard dimension names case-insensitively
        dims_lower = set(d.lower() for d in las.point_format.dimension_names)

        if attr_name.lower() in dims_lower:
            return np.array(las[attr_name])

        if fallback_names:
            for name in fallback_names:
                if name.lower() in dims_lower:
                    return np.array(las[name])
        return None

    # Intensity
    intensity = _get_dim("intensity")
    if intensity is None:
        intensity = np.zeros((xyz.shape[0],), dtype=np.float32)
    else:
        intensity = intensity.astype(np.float32)

    # Returns (CRITICAL FIX: Use property access)
    rn = _get_dim("return_number")
    nor = _get_dim("number_of_returns")

    if rn is None:
        rn = np.ones((xyz.shape[0],), dtype=np.int64)
    else:
        rn = rn.astype(np.int64)

    if nor is None:
        nor = np.ones((xyz.shape[0],), dtype=np.int64)
    else:
        nor = nor.astype(np.int64)

    # Labels
    labels = _get_dim("classification", fallback_names=["raw_classification"])
    if labels is None:
        # Fallback for datasets that might be unlabeled
        # print(f" [WARN] No classification found in {path.name}, using zeros.") # Optional logging
        labels = np.zeros((xyz.shape[0],), dtype=np.int64)
    else:
        labels = labels.astype(np.int64)

    # RGB
    rgb = None
    red = _get_dim("red")
    green = _get_dim("green")
    blue = _get_dim("blue")

    if red is not None and green is not None and blue is not None:
        max_val = max(red.max(), green.max(), blue.max())
        scale = 1.0
        if max_val > 255:
            scale = 1.0 / 65535.0
        r = red.astype(np.float32) * scale
        g = green.astype(np.float32) * scale
        b = blue.astype(np.float32) * scale
        rgb = np.stack([r, g, b], axis=1)

    return {
        "xyz": xyz,
        "intensity": intensity,
        "return_number": rn,
        "number_of_returns": nor,
        "rgb": rgb,
        "native_labels": labels,
    }
