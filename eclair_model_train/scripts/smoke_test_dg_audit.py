#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dg_audit import (
    audit_bev_projection,
    audit_mix3d_boundary,
    audit_occupancy_perturbations,
    occupancy_summary,
)
from src.mix3d import ALSMix3DConfig
from scripts.audit_dg_candidates import (
    _cross_dataset_occupancy,
    _sample_indices,
    _sampling_overview,
    _stratified_class_presence_specs,
    _summarize_dataset,
)


def _sample(source: str, seed: int):
    rng = np.random.default_rng(seed)
    n = 12000
    xyz = rng.uniform((0.0, 0.0, 0.0), (40.0, 40.0, 15.0), size=(n, 3))
    labels = rng.integers(0, 4, size=n, dtype=np.int64)
    # Guarantee protected thin classes have support.
    labels[:300] = 2
    labels[300:600] = 3
    return {
        "xyz": xyz.astype(np.float32),
        "y_train": labels,
        "return_number": np.ones(n, dtype=np.int64),
        "number_of_returns": np.ones(n, dtype=np.int64),
        "intensity": None,
        "rgb": None,
        "rn_1h_u8": None,
        "nor_1h_u8": None,
        "source_id": source,
        "context_side_xy_m": 40.0,
    }


def main() -> None:
    host = _sample("host", 1)
    donor = _sample("donor", 2)
    occ = occupancy_summary(
        host["xyz"], host["y_train"], voxel_sizes_m=[0.5, 1.0], num_classes=4, ignore_index=-100
    )
    assert len(occ) == 2 and occ[0]["active_voxels"] > 0

    ocons = audit_occupancy_perturbations(
        host,
        input_voxel_m=0.5,
        strengths=[0.2],
        num_classes=4,
        ignore_index=-100,
        protected_class_ids=[2, 3],
        protected_min_voxels=3,
        component_voxel_m=1.0,
        seed=9,
    )
    assert {r["method"] for r in ocons} == {"point_thin", "active_voxel_mask"}
    masked = next(r for r in ocons if r["method"] == "active_voxel_mask")
    assert masked["changed_active_voxel_fraction"] > 0.1
    assert masked["protected_classes_disappeared"] == 0

    cfg = ALSMix3DConfig(
        enabled=True,
        replacement_area_fraction=0.25,
        guard_band_voxels=1,
        min_region_points=100,
        min_replacement_side_m=2.0,
        verify_cross_provenance_collisions=True,
    )
    bab = audit_mix3d_boundary(
        host,
        donor,
        cfg=cfg,
        num_classes=4,
        ignore_index=-100,
        voxel_edge_m=0.5,
        seam_widths_m=[0.5, 1.0],
        component_cell_m=1.0,
        rng=np.random.default_rng(11),
    )
    assert bab["applied"] == 1
    assert len(bab["seam_bands"]) == 2
    assert len(bab["component_cuts"]) == 4

    bev = audit_bev_projection(
        host,
        input_voxel_m=0.5,
        resolutions_m=[0.5, 1.0],
        bounds_xyz_m=(0.0, 40.0, 0.0, 40.0, 0.0, 15.0),
        num_classes=4,
        ignore_index=-100,
        utility_class_ids=[2, 3],
        z_slices=4,
        hidden_dim=32,
    )
    assert len(bev) == 2
    assert bev[0]["in_bounds_active_voxel_fraction"] > 0.9
    assert 0.0 <= bev[0]["multi_class_cell_fraction"] <= 1.0
    assert set(bev[0]["policies"]) == {"first", "last", "majority", "highest_z"}

    shifted = dict(host)
    shifted["xyz"] = host["xyz"] + np.asarray([1000.0, -500.0, 250.0], dtype=np.float32)
    candidate_bev = audit_bev_projection(
        shifted,
        input_voxel_m=0.5,
        resolutions_m=[0.5],
        bounds_xyz_m=(-30.0, 30.0, -30.0, 30.0, -1.0, 1.0),
        num_classes=4,
        ignore_index=-100,
        utility_class_ids=[2, 3],
        z_slices=4,
        hidden_dim=32,
        xy_frame="bbox_centered",
        z_filter="none",
        height_slicing="quantile",
    )
    assert candidate_bev[0]["in_bounds_active_voxel_fraction"] > 0.99
    assert candidate_bev[0]["z_sliced_multi_class_cell_fraction"] <= candidate_bev[0]["multi_class_cell_fraction"]

    repeated = _sample_indices(length=3, count=8)
    assert len(repeated) == 8
    assert set(repeated) == {0, 1, 2}

    # Deterministic class-presence stratification: preserve a uniform stratum,
    # meet class quotas when support exists, deduplicate hosts, and fill the
    # requested minimum number of unique samples.
    tile_counts = np.zeros((18, 5), dtype=np.int64)
    tile_counts[[0, 2, 4, 6, 8, 10, 12, 14], 2] = [1, 2, 4, 8, 16, 32, 64, 128]
    tile_counts[[1, 2, 5, 6, 9, 10, 13, 14], 3] = [2, 3, 5, 7, 11, 13, 17, 19]
    tile_counts[[3, 6, 7, 10, 11, 14, 15, 17], 4] = [1, 3, 9, 27, 81, 5, 15, 45]
    specs_a = _stratified_class_presence_specs(
        tile_counts,
        class_ids=[2, 3, 4],
        uniform_samples=4,
        samples_per_class=6,
        support_bins=3,
        minimum_total_unique=12,
        seed=1337,
    )
    specs_b = _stratified_class_presence_specs(
        tile_counts,
        class_ids=[2, 3, 4],
        uniform_samples=4,
        samples_per_class=6,
        support_bins=3,
        minimum_total_unique=12,
        seed=1337,
    )
    assert specs_a == specs_b
    assert len(specs_a) >= 12
    assert len({s["host_index"] for s in specs_a}) == len(specs_a)
    assert sum(s["is_uniform_sample"] for s in specs_a) == 4
    for class_id in (2, 3, 4):
        members = [s for s in specs_a if f"class_{class_id}" in s["sampling_strata"]]
        assert len(members) >= 6
        assert all(tile_counts[s["host_index"], class_id] > 0 for s in members)
    sampling_overview = _sampling_overview(
        [
            {
                **spec,
                "host_source": f"tile_{spec['host_index']}",
            }
            for spec in specs_a
        ]
    )
    assert sampling_overview["unique_host_sources"] == len(specs_a)
    assert sampling_overview["stratum_unique_hosts"]["uniform"] == 4
    assert sampling_overview["aggregate_scope"] == (
        "all_selected_stratified_records_not_population_unbiased"
    )

    for row in bev:
        row["projection_variant"] = "configured_current"
    for row in candidate_bev:
        row["projection_variant"] = "centered_candidate"
    aggregate = _summarize_dataset(
        "synthetic",
        [
            {
                "host_source": "host",
                "input_voxel_m": 0.5,
                "occupancy": occ,
                "bab": bab,
                "ocons": ocons,
                "bev": bev + candidate_bev,
            }
        ],
        utility_ids=[2, 3],
        bev_static_map_risk=True,
        gates={},
    )
    assert aggregate["samples"] == 1
    assert aggregate["unique_host_sources"] == 1
    assert len(aggregate["class_support"]) == 4
    assert {row["projection_variant"] for row in aggregate["bev"]} == {
        "configured_current",
        "centered_candidate",
    }

    cross = _cross_dataset_occupancy(
        [
            {
                "dataset": "dales",
                "occupancy": [
                    {
                        "voxel_m": 0.5,
                        "active_voxels_per_1000_points": 800.0,
                        "points_per_active_voxel_mean": 1.25,
                        "singleton_voxel_fraction": 0.75,
                    }
                ],
            },
            {
                "dataset": "eclair",
                "occupancy": [
                    {
                        "voxel_m": 0.5,
                        "active_voxels_per_1000_points": 400.0,
                        "points_per_active_voxel_mean": 2.5,
                        "singleton_voxel_fraction": 0.5,
                    }
                ],
            },
        ]
    )
    assert len(cross) == 1
    assert cross[0]["active_voxels_per_point_ratio"] == 2.0
    print("DG candidate audit smoke test passed")


if __name__ == "__main__":
    main()
