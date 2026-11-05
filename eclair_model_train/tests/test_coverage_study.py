import numpy as np
import pytest
from src.coverage_study import TileCoverage, geometry_bins, geometry_eval_rows


def test_same_native_label_choice_and_cache_identity():
    points = np.array(
        [[1.0, 1.0, 2.0], [70.0, 2.0, 3.0], [1.0, 2.0, 4.0], [4.0, 2.0, 4.0]]
    )
    labels = np.array([7, 7, 8, 8])
    rows = [[i, labels[i], 5 if i < 2 else 3, i % 2, *p] for i, p in enumerate(points)]
    tile = TileCoverage(rows, points, labels)
    rng = np.random.default_rng(1)
    choices = [tile.choose(0, 7, rng) for _ in range(100)]
    assert set(choices) == {0, 1}
    hits = tile.crop_hits(points, np.array([0, 2]), 0, 10)
    assert hits.sum() == 2 and hits[5, 0] == hits[3, 0] == 1
    with pytest.raises(ValueError, match="raw cache"):
        TileCoverage(rows, points + 1, labels)


def test_anchor_identity_and_geometry_bin_confusion():
    data = {
        "ids": np.array([1, 3]),
        "xyz": np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        "gt": np.array([5, 5]),
        "native": np.array([7, 7]),
        "geometry": np.zeros((2, 6, 22), dtype=np.float32),
    }
    data["geometry"][1, 2, 4] = 0.9
    data["geometry"][1, 5, 9] = 5.0
    medians = {str(c): [0.5, 3.0] for c in (3, 4, 5, 6, 7)}
    assert geometry_bins(data, medians).tolist() == [0, 3]
    xyz = np.array([[i, 0.0, 0.0] for i in range(4)])
    native = np.array([0, 7, 0, 7])
    gt = np.array([0, 5, 0, 5])
    prediction = np.array([0, 4, 0, 5])
    counts, pairs = geometry_eval_rows(
        data, {"xyz": xyz}, native, gt, prediction, medians
    )
    assert counts[5][0][4] == counts[5][3][5] == 1
    assert pairs["prediction"] == [4, 5]
    with pytest.raises(ValueError, match="align"):
        geometry_eval_rows(data, {"xyz": xyz[::-1]}, native, gt, prediction, medians)
