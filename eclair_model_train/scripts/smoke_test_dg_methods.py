#!/usr/bin/env python3
"""Synthetic contract tests for O-CONS and the ALS-specific BEV branch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bev_als_config import ALSBEVConfig
from src.bev_als_head import ALSHeightSlicedProjector, als_bev_multilabel_loss
from src.bev_als_targets import build_als_bev_target
from src.ocons import OConsConfig, make_masked_view, match_sparse_coordinates, ocons_consistency_loss


def test_ocons() -> None:
    n = 200
    coords = torch.zeros((n, 4), dtype=torch.int32)
    coords[:, 1] = torch.arange(n, dtype=torch.int32)
    feats = torch.randn(n, 10)
    labels = torch.arange(n, dtype=torch.int64) % 8
    cfg = OConsConfig(
        enabled=True,
        mask_fraction=0.10,
        protected_class_ids=(5, 6),
        protected_min_voxels=3,
        protected_min_fraction=0.70,
    )
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = make_masked_view(coords, feats, labels, cfg, num_classes=8, generator=g1)
    b = make_masked_view(coords, feats, labels, cfg, num_classes=8, generator=g2)
    assert torch.equal(a.clean_input_indices, b.clean_input_indices)
    assert abs(a.diagnostics.actual_mask_fraction - 0.10) < 0.01
    for class_id in (5, 6):
        before = int(a.diagnostics.class_before[class_id])
        after = int(a.diagnostics.class_after[class_id])
        assert after >= min(before, max(3, int(np.ceil(0.70 * before))))
    idx, coverage = match_sparse_coordinates(coords, a.coordinates, require_all=True)
    assert coverage == 1.0 and torch.equal(coords.index_select(0, idx), a.coordinates)
    clean_logits = torch.randn(a.coordinates.shape[0], 8)
    pert_logits = clean_logits.clone().requires_grad_(True)
    loss, diag = ocons_consistency_loss(clean_logits, pert_logits, a.labels, cfg, num_classes=8)
    assert float(loss.item()) < 1e-6
    assert float(diag["agreement"].item()) == 1.0
    loss.backward()


def test_bev_als() -> None:
    cfg = ALSBEVConfig(
        enabled=True,
        half_extent_m=2.0,
        resolution_m=1.0,
        height_slices=4,
        input_channels=128,
        pre_pool_dim=4,
        decoder_hidden_dim=8,
    )
    coords = np.asarray(
        [[-1, -1, 0], [-1, -1, 0], [0, 0, 1], [1, 1, 2], [1, 1, 3]],
        dtype=np.int32,
    )
    labels = np.asarray([5, 6, 0, 1, 4], dtype=np.int64)
    target = build_als_bev_target(
        coords,
        labels,
        num_classes=8,
        meters_per_voxel=1.0,
        cfg=cfg,
        ignore_index=-100,
    )
    shifted = build_als_bev_target(
        coords + np.asarray([100, -80, 7], dtype=np.int32),
        labels,
        num_classes=8,
        meters_per_voxel=1.0,
        cfg=cfg,
        ignore_index=-100,
    )
    assert np.array_equal(target.target_u8, shifted.target_u8)
    assert float(target.diagnostics["in_bounds_fraction"]) == 1.0
    assert int(target.diagnostics["multi_label_cells"]) >= 1

    class Sparse:
        pass

    sparse = Sparse()
    sparse.C = torch.cat([torch.zeros((coords.shape[0], 1), dtype=torch.int32), torch.from_numpy(coords)], dim=1)
    sparse.F = torch.randn(coords.shape[0], 128, requires_grad=True)
    sparse.tensor_stride = (1, 1, 1)
    projector = ALSHeightSlicedProjector(cfg)
    dense, diag = projector(
        sparse,
        meters_per_voxel=1.0,
        center_xy_m=torch.from_numpy(target.frame.center_xy_m)[None],
        height_edges_m=torch.from_numpy(target.frame.height_edges_m)[None],
    )
    assert dense.shape == (1, 16, 4, 4)
    assert float(diag["feature_in_bounds_fraction"].item()) == 1.0
    projector_loss = dense.square().mean()
    projector_loss.backward()

    assert sparse.F.grad is not None
    assert bool(torch.isfinite(sparse.F.grad).all())
    assert any(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in projector.parameters())

    target_t = torch.from_numpy(target.target_u8)[None]
    occupied_t = torch.from_numpy(target.occupied_u8)[None]
    logits = torch.where(target_t.bool(), torch.tensor(8.0), torch.tensor(-8.0)).requires_grad_(True)
    loss, metrics = als_bev_multilabel_loss(logits, target_t, occupied_t, cfg)
    assert float(metrics["f1"].item()) == 1.0
    assert tuple(metrics["tp_by_slice_class"].shape) == (4, 8)
    loss.backward()


def main() -> None:
    test_ocons()
    test_bev_als()
    print("O-CONS and BEV-ALS synthetic smoke tests passed")


if __name__ == "__main__":
    main()
