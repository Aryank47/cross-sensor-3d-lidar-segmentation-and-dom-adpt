from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mix3d import ALSMix3DConfig, compose_crop_replace


def _sample(source_id: str, seed: int, side_m: float = 100.0):
    rng = np.random.default_rng(seed)
    n = 30000
    xyz = np.column_stack(
        (
            rng.uniform(0.0, side_m, n),
            rng.uniform(0.0, side_m, n),
            rng.uniform(0.0, 30.0, n),
        )
    ).astype(np.float32)
    return {
        "xyz": xyz,
        "y_train": rng.integers(0, 11, n, dtype=np.int64),
        "return_number": rng.integers(1, 6, n, dtype=np.int64),
        "number_of_returns": rng.integers(1, 6, n, dtype=np.int64),
        "intensity": None,
        "rgb": None,
        "rn_1h_u8": None,
        "nor_1h_u8": None,
        "source_id": source_id,
        "context_side_xy_m": side_m,
    }



def _roundoff_boundary_sample(source_id: str, *, donor: bool):
    """Construct a float32 extent that reproduces high < low by round-off."""
    rng = np.random.default_rng(123 if donor else 456)
    n = 5000
    if donor:
        x0 = np.float32(28.611933)
        x1 = np.float32(70.626915)
        y0 = np.float32(-10.0)
        y1 = np.float32(80.0)
    else:
        x0 = np.float32(-100.0)
        x1 = np.float32(100.0)
        y0 = np.float32(-100.0)
        y1 = np.float32(100.0)
    xyz = np.column_stack(
        (
            rng.uniform(float(x0), float(x1), n),
            rng.uniform(float(y0), float(y1), n),
            rng.uniform(0.0, 10.0, n),
        )
    ).astype(np.float32)
    # Force the measured extrema to the exact float32 endpoints.
    xyz[0, 0], xyz[1, 0] = x0, x1
    xyz[0, 1], xyz[1, 1] = y0, y1
    return {
        "xyz": xyz,
        "y_train": rng.integers(0, 3, n, dtype=np.int64),
        "return_number": np.ones(n, dtype=np.int64),
        "number_of_returns": np.ones(n, dtype=np.int64),
        "intensity": None,
        "rgb": None,
        "rn_1h_u8": None,
        "nor_1h_u8": None,
        "source_id": source_id,
        "context_side_xy_m": 200.0,
    }

def main():
    cfg = ALSMix3DConfig(
        enabled=True,
        replacement_area_fraction=0.25,
        guard_band_voxels=1,
        min_region_points=100,
        verify_cross_provenance_collisions=True,
    )
    host = _sample("host", 1)
    donor = _sample("donor", 2)

    a = compose_crop_replace(
        host=host,
        donor=donor,
        cfg=cfg,
        num_classes=11,
        voxel_edge_m=0.5,
        rng=np.random.default_rng(99),
    )
    b = compose_crop_replace(
        host=host,
        donor=donor,
        cfg=cfg,
        num_classes=11,
        voxel_edge_m=0.5,
        rng=np.random.default_rng(99),
    )

    assert a.diagnostics.applied == 1
    assert a.diagnostics.cross_provenance_voxels == 0
    assert abs(a.diagnostics.replacement_side_m - 50.0) < 1e-6
    assert a.sample["xyz"].shape[0] == a.sample["y_train"].shape[0]
    assert a.sample["xyz"].shape[0] == a.sample["return_number"].shape[0]
    assert np.array_equal(a.sample["xyz"], b.sample["xyz"])
    assert np.array_equal(a.sample["y_train"], b.sample["y_train"])
    assert len(a.diagnostics.host_removed_class_counts) == 11
    assert len(a.diagnostics.donor_inserted_class_counts) == 11
    assert len(a.diagnostics.output_class_counts) == 11
    assert len(a.diagnostics.donor_host_context_pairs) == 11 * 11
    assert sum(a.diagnostics.output_class_counts) == a.diagnostics.output_points

    # Regression: a float32 donor span can round so that dmax - side is a few
    # micro-metres below dmin. This must be treated as a single-point interval,
    # not raised as NumPy's "high - low < 0" error.
    boundary_cfg = ALSMix3DConfig(
        enabled=True,
        replacement_area_fraction=1.0,
        guard_band_voxels=0,
        min_region_points=1,
        min_replacement_side_m=1.0,
        verify_cross_provenance_collisions=False,
    )
    boundary = compose_crop_replace(
        host=_roundoff_boundary_sample("boundary-host", donor=False),
        donor=_roundoff_boundary_sample("boundary-donor", donor=True),
        cfg=boundary_cfg,
        num_classes=3,
        voxel_edge_m=0.2,
        rng=np.random.default_rng(7),
    )
    assert boundary.diagnostics.applied == 1

    # Labels must be retained-host labels followed by donor labels; composition
    # is hard-label concatenation, not interpolation.
    inserted = a.diagnostics.donor_inserted
    assert inserted > 0
    assert np.isin(a.sample["y_train"][-inserted:], donor["y_train"]).all()
    print("Mix3D core smoke test passed:", a.diagnostics)


if __name__ == "__main__":
    main()
