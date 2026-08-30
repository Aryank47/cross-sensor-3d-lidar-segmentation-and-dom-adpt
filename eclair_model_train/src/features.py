from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FeatureConfig:
    use_intensity: bool = True
    intensity_divisor: float = 65535.0
    returns_onehot_k: int = 5
    use_rgb: bool = False
    rgb_divisor: float = 65535.0
    include_coords: bool = False  # If True, append (x,y,z) as features (not paper-like)


def _one_hot(values_1_based: np.ndarray, k: int) -> np.ndarray:
    """
    One-hot encode 1-based integer values into size-k vectors.
    Values outside [1..k] are clipped.
    """
    v = values_1_based.astype(np.int64)
    v = np.clip(v, 1, k) - 1
    out = np.zeros((v.shape[0], k), dtype=np.float32)
    out[np.arange(v.shape[0]), v] = 1.0
    return out


def build_features(
    xyz_local: np.ndarray,
    intensity: Optional[np.ndarray],
    return_number: Optional[np.ndarray],
    number_of_returns: Optional[np.ndarray],
    rgb: Optional[np.ndarray],
    cfg: FeatureConfig,
) -> np.ndarray:
    feats: list[np.ndarray] = []

    if cfg.use_intensity:
        if intensity is None:
            raise ValueError("use_intensity=True but intensity is None")
        inten = intensity.astype(np.float32) / float(cfg.intensity_divisor)
        inten = np.clip(inten, 0.0, 1.0)
        feats.append(inten.reshape(-1, 1))

    if return_number is not None and number_of_returns is not None:
        feats.append(_one_hot(return_number, cfg.returns_onehot_k))
        feats.append(_one_hot(number_of_returns, cfg.returns_onehot_k))

    if cfg.use_rgb:
        if rgb is None:
            raise ValueError("use_rgb=True but rgb is None")
        rgb_f = rgb.astype(np.float32) / float(cfg.rgb_divisor)
        feats.append(np.clip(rgb_f, 0.0, 1.0))

    if cfg.include_coords:
        feats.append(xyz_local.astype(np.float32))

    if not feats:
        # MinkowskiEngine requires some feature; common fallback is a constant 1.
        feats.append(np.ones((xyz_local.shape[0], 1), dtype=np.float32))

    return np.concatenate(feats, axis=1).astype(np.float32, copy=False)
