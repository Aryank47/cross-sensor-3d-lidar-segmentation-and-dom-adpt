from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ALSBEVConfig:
    enabled: bool = False
    feature_level: str = "block8"
    xy_frame: str = "bbox_centered"
    half_extent_m: float = 90.0
    resolution_m: float = 0.50
    y_flip: bool = True
    height_mode: str = "quantile"
    height_slices: int = 4
    feature_pool: str = "max"
    input_channels: int = 128
    pre_pool_dim: int = 16
    decoder_hidden_dim: int = 64
    target_mode: str = "height_sliced_multilabel"
    bce_weight: float = 0.50
    dice_weight: float = 0.50
    loss_weight: float = 0.50
    loss_warmup_epochs: int = 10
    metric_threshold: float = 0.50
    min_in_bounds_fraction: float = 0.995
    min_height_edge_gap_m: float = 0.01

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "ALSBEVConfig":
        raw = (((cfg.get("model", {}) or {}).get("aux_heads", {}) or {}).get("bev_als", {}) or {})
        loss = raw.get("loss", {}) or {}
        out = ALSBEVConfig(
            enabled=bool(raw.get("enabled", False)),
            feature_level=str(raw.get("feature_level", "block8")),
            xy_frame=str(raw.get("xy_frame", "bbox_centered")),
            half_extent_m=float(raw.get("half_extent_m", 90.0)),
            resolution_m=float(raw.get("resolution_m", 0.50)),
            y_flip=bool(raw.get("y_flip", True)),
            height_mode=str(raw.get("height_mode", "quantile")),
            height_slices=int(raw.get("height_slices", 4)),
            feature_pool=str(raw.get("feature_pool", "max")),
            input_channels=int(raw.get("input_channels", 128)),
            pre_pool_dim=int(raw.get("pre_pool_dim", 16)),
            decoder_hidden_dim=int(raw.get("decoder_hidden_dim", 64)),
            target_mode=str(raw.get("target_mode", "height_sliced_multilabel")),
            bce_weight=float(loss.get("bce_weight", 0.50)),
            dice_weight=float(loss.get("dice_weight", 0.50)),
            loss_weight=float(loss.get("max_weight", 0.50)),
            loss_warmup_epochs=int(loss.get("warmup_epochs", 10)),
            metric_threshold=float(raw.get("metric_threshold", 0.50)),
            min_in_bounds_fraction=float(raw.get("min_in_bounds_fraction", 0.995)),
            min_height_edge_gap_m=float(raw.get("min_height_edge_gap_m", 0.01)),
        )
        if out.enabled:
            if out.feature_level != "block8" or out.xy_frame != "bbox_centered":
                raise ValueError("BEV-ALS v1 requires feature_level=block8 and xy_frame=bbox_centered.")
            if out.height_mode != "quantile" or out.height_slices != 4:
                raise ValueError("BEV-ALS v1 requires four quantile height slices.")
            if out.feature_pool != "max" or out.target_mode != "height_sliced_multilabel":
                raise ValueError("BEV-ALS v1 requires max pooling and height_sliced_multilabel targets.")
            if out.half_extent_m <= 0 or out.resolution_m <= 0:
                raise ValueError("BEV-ALS extent and resolution must be positive.")
            cells = (2.0 * out.half_extent_m) / out.resolution_m
            if not math.isclose(cells, round(cells), rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("2*half_extent_m must be divisible by resolution_m.")
            if out.input_channels <= 0 or out.pre_pool_dim <= 0 or out.decoder_hidden_dim <= 0:
                raise ValueError("BEV-ALS channel dimensions must be positive.")
            if not math.isclose(out.bce_weight + out.dice_weight, 1.0, abs_tol=1e-6):
                raise ValueError("BEV-ALS BCE and Dice weights must sum to 1.")
            if out.bce_weight < 0.0 or out.dice_weight < 0.0:
                raise ValueError("BEV-ALS BCE and Dice weights must be non-negative.")
            if out.loss_weight < 0.0 or out.loss_warmup_epochs < 0:
                raise ValueError("BEV-ALS loss weight and warmup epochs must be non-negative.")
            if not (0.0 < out.metric_threshold < 1.0):
                raise ValueError("BEV-ALS metric_threshold must be in (0,1).")
            if not (0.0 < out.min_in_bounds_fraction <= 1.0):
                raise ValueError("BEV-ALS min_in_bounds_fraction must be in (0,1].")
            if out.min_height_edge_gap_m < 0.0:
                raise ValueError("BEV-ALS min_height_edge_gap_m must be non-negative.")
        return out

    @property
    def grid_size(self) -> int:
        return int(round((2.0 * self.half_extent_m) / self.resolution_m))


def bev_als_weight_for_epoch(cfg: ALSBEVConfig, epoch: int) -> float:
    if not cfg.enabled:
        return 0.0
    if cfg.loss_warmup_epochs <= 0:
        return float(cfg.loss_weight)
    return float(cfg.loss_weight) * min(1.0, max(0.0, float(epoch) / float(cfg.loss_warmup_epochs)))
