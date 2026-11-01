# /src/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FeatureConfig:
    # intensity
    use_intensity: bool = True
    intensity_divisor: float = 65535.0

    # returns
    returns_onehot_k: int = 5
    use_return_number: bool = True
    use_number_of_returns: bool = True

    # optional extras
    use_rgb: bool = False
    include_coords: bool = False  # append xyz_local if True
    return_input_mode: str = "observed"

    def __post_init__(self):
        if self.return_input_mode not in ("observed", "constant_pair"):
            raise ValueError(f"Unknown return_input_mode={self.return_input_mode!r}")
        if self.return_input_mode == "constant_pair" and (
            self.returns_onehot_k != 5 or not self.use_return_number or not self.use_number_of_returns
            or self.use_intensity or self.use_rgb or self.include_coords
        ):
            raise ValueError("constant_pair requires exactly two K=5 return groups and no other feature channels")


def feature_dim(cfg: FeatureConfig) -> int:
    d = 0
    if cfg.use_intensity:
        d += 1
    if cfg.use_return_number:
        d += int(cfg.returns_onehot_k)
    if cfg.use_number_of_returns:
        d += int(cfg.returns_onehot_k)
    if cfg.use_rgb:
        d += 3
    if cfg.include_coords:
        d += 3
    return d


def _onehot_returns(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x)
    # LiDAR returns are typically 1..k; clamp defensively
    idx = np.clip(x.astype(np.int64), 1, k) - 1
    return np.eye(k, dtype=np.float32)[idx]


def _maybe_scale_intensity(intensity: np.ndarray, divisor: float) -> np.ndarray:
    """
    Accept either raw intensity (0..65535) or already-scaled (0..1).
    If max <= ~1, treat as scaled; else divide by divisor.
    """
    x = np.asarray(intensity).astype(np.float32, copy=False)
    if x.size == 0:
        return x.reshape(-1, 1)
    if float(x.max()) <= 1.0 + 1e-3:
        return x.reshape(-1, 1)
    return (x / float(divisor)).reshape(-1, 1)


def build_features(
    *,
    xyz_local: np.ndarray,
    intensity: Optional[np.ndarray],
    return_number: Optional[np.ndarray],
    number_of_returns: Optional[np.ndarray],
    rgb: Optional[np.ndarray],
    cfg: FeatureConfig,
    return_number_1h: Optional[np.ndarray] = None,
    number_of_returns_1h: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Returns float32 features [N, C] as numpy array (so callers can index with numpy idx).
    Option A: cfg.use_intensity=False -> intensity channel is omitted.
    """
    if cfg.return_input_mode == "constant_pair":
        # Retain nonzero occupancy features and the original channel count.
        # Ignore raw AND cached return fields; never mutate the dataset/cache.
        feats = np.zeros((len(xyz_local), 10), dtype=np.float32)
        feats[:, (0, 5)] = 1.0
        return feats
    parts: list[np.ndarray] = []

    # intensity
    if cfg.use_intensity:
        if intensity is None:
            raise ValueError("cfg.use_intensity=True but intensity=None")
        parts.append(_maybe_scale_intensity(intensity, cfg.intensity_divisor))

    # returns
    if cfg.use_return_number:
        if return_number_1h is not None:
            rn = return_number_1h.astype(np.float32, copy=False)
        else:
            rn = _onehot_returns(return_number, int(cfg.returns_onehot_k))
        parts.append(rn)

    if cfg.use_number_of_returns:
        if number_of_returns_1h is not None:
            nor = number_of_returns_1h.astype(np.float32, copy=False)
        else:
            nor = _onehot_returns(number_of_returns, int(cfg.returns_onehot_k))
        parts.append(nor)

    # rgb
    if cfg.use_rgb:
        if rgb is None:
            raise ValueError("cfg.use_rgb=True but rgb=None")
        parts.append(np.asarray(rgb, dtype=np.float32))

    # coords-as-features
    if cfg.include_coords:
        parts.append(np.asarray(xyz_local, dtype=np.float32))

    if not parts:
        raise ValueError("No features enabled; feature vector is empty.")

    feats = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    return np.ascontiguousarray(feats, dtype=np.float32)


def infer_in_channels(cfg: FeatureConfig) -> int:
    c = 0
    if cfg.use_intensity:
        c += 1
    if cfg.use_return_number:
        c += int(cfg.returns_onehot_k)
    if cfg.use_number_of_returns:
        c += int(cfg.returns_onehot_k)
    if cfg.use_rgb:
        c += 3
    if cfg.include_coords:
        c += 3
    return int(c)


def assert_checkpoint_feature_config(checkpoint: dict, source_cfg: dict) -> None:
    """Identical tensor shapes do not make observed and constant inputs compatible."""
    from dataclasses import asdict

    requested = FeatureConfig(**source_cfg["data"]["features"])
    saved = checkpoint.get("cfg") if isinstance(checkpoint, dict) else None
    if saved is None:
        if requested.return_input_mode != "observed":
            raise ValueError("A constant-input checkpoint must contain its source feature configuration")
        return  # preserve support for legacy observed-input state dictionaries
    actual = FeatureConfig(**saved["data"]["features"])
    if asdict(actual) != asdict(requested):
        raise ValueError("Checkpoint feature configuration differs from requested evaluation inputs")
