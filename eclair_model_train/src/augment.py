from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class AugmentConfig:
    enabled: bool = True
    random_flip_xy: bool = True
    random_rotate_deg: Tuple[float, float] = (0.0, 360.0)
    scale_range: Tuple[float, float] = (0.95, 1.05)
    jitter_std: float = 0.01


def augment_xyz(
    xyz: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator
) -> np.ndarray:
    """
    Apply simple geometric augmentations in-place-ish on local XYZ.

    Notes:
    - For aerial LiDAR, we rotate around Z.
    - Flip is done around the tile's max extent (keeps coords mostly within the tile box).
    """
    if not cfg.enabled:
        return xyz

    out = xyz.astype(np.float32, copy=True)

    # Random rotation around Z
    a0, a1 = cfg.random_rotate_deg
    if a1 > a0:
        theta = np.deg2rad(rng.uniform(a0, a1))
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        out[:, :2] = out[:, :2] @ R.T

    # Random flip
    if cfg.random_flip_xy:
        if rng.random() < 0.5:
            x_max = float(out[:, 0].max())
            out[:, 0] = x_max - out[:, 0]
        if rng.random() < 0.5:
            y_max = float(out[:, 1].max())
            out[:, 1] = y_max - out[:, 1]

    # Random scaling
    s0, s1 = cfg.scale_range
    if s1 > s0:
        scale = float(rng.uniform(s0, s1))
        out *= scale

    # Jitter
    if cfg.jitter_std and cfg.jitter_std > 0:
        out += rng.normal(0.0, cfg.jitter_std, size=out.shape).astype(np.float32)

    return np.ascontiguousarray(out, dtype=np.float32)
