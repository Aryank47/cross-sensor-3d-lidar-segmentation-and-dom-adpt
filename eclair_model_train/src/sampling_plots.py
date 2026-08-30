from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .sampling_metrics import build_density_grid
from .sampling_types import COMMON_CLASS_NAMES, WindowSpec

matplotlib.use("Agg")


def _rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "|".join([str(seed), *[str(part) for part in parts]])
    digest = hashlib.sha1(payload.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _local_xy(xyz: np.ndarray, spec: WindowSpec) -> np.ndarray:
    out = np.asarray(xyz, dtype=np.float64).copy()
    out[:, 0] -= spec.x_min_m
    out[:, 1] -= spec.y_min_m
    return out


def _sample_with_probability(
    xyz: np.ndarray,
    probability: float,
    *,
    seed: int,
    key: str,
) -> np.ndarray:
    if probability >= 1.0:
        return xyz
    generator = _rng(seed, key)
    mask = generator.random(xyz.shape[0]) < probability
    return xyz[mask]


def save_pair_overlay(
    *,
    first_name: str,
    first_spec: WindowSpec,
    first_data: Dict[str, np.ndarray],
    second_name: str,
    second_spec: WindowSpec,
    second_data: Dict[str, np.ndarray],
    output_path: Path,
    scatter_budget_for_denser_window: int,
    heatmap_resolution_m: float,
    dpi: int,
    seed: int,
    focus_common_id: Optional[int] = None,
) -> None:
    first_xyz = _local_xy(first_data["xyz"], first_spec)
    second_xyz = _local_xy(second_data["xyz"], second_spec)
    first_rn = np.asarray(first_data["return_number"], dtype=np.int64)
    second_rn = np.asarray(second_data["return_number"], dtype=np.int64)
    first_common = np.asarray(first_data["common_labels"], dtype=np.int64)
    second_common = np.asarray(second_data["common_labels"], dtype=np.int64)

    maximum_n = max(first_xyz.shape[0], second_xyz.shape[0])
    probability = min(1.0, scatter_budget_for_denser_window / max(1, maximum_n))
    first_plot = _sample_with_probability(first_xyz, probability, seed=seed, key=f"{first_spec.key}|all")
    second_plot = _sample_with_probability(second_xyz, probability, seed=seed, key=f"{second_spec.key}|all")

    first_first = _sample_with_probability(first_xyz[first_rn == 1], probability, seed=seed, key=f"{first_spec.key}|first")
    second_first = _sample_with_probability(
        second_xyz[second_rn == 1],
        probability,
        seed=seed,
        key=f"{second_spec.key}|first",
    )

    first_grid, actual_res, cell_area = build_density_grid(first_xyz[:, :2], first_spec.window_size_m, heatmap_resolution_m)
    second_grid, _, _ = build_density_grid(second_xyz[:, :2], second_spec.window_size_m, heatmap_resolution_m)
    first_density = first_grid / cell_area
    second_density = second_grid / cell_area
    shared_vmax = max(
        1.0,
        float(np.quantile(np.concatenate((first_density.ravel(), second_density.ravel())), 0.99)),
    )

    figure, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    axes[0, 0].scatter(first_plot[:, 0], first_plot[:, 1], s=0.3, alpha=0.35, rasterized=True)
    axes[0, 0].set_title(f"{first_name}: all returns\n{first_xyz.shape[0] / first_spec.area_m2:.2f} points/m²")
    axes[0, 1].scatter(second_plot[:, 0], second_plot[:, 1], s=0.3, alpha=0.35, rasterized=True)
    axes[0, 1].set_title(f"{second_name}: all returns\n{second_xyz.shape[0] / second_spec.area_m2:.2f} points/m²")
    axes[0, 2].scatter(
        first_plot[:, 0],
        first_plot[:, 1],
        s=0.3,
        alpha=0.22,
        label=first_name,
        rasterized=True,
    )
    axes[0, 2].scatter(
        second_plot[:, 0],
        second_plot[:, 1],
        s=0.3,
        alpha=0.22,
        label=second_name,
        rasterized=True,
    )
    axes[0, 2].set_title(f"All-return overlay (shared display probability={probability:.4f})")
    axes[0, 2].legend(markerscale=6)

    im_first = axes[1, 0].imshow(
        first_density,
        origin="lower",
        extent=(0, first_spec.window_size_m, 0, first_spec.window_size_m),
        vmin=0,
        vmax=shared_vmax,
        aspect="equal",
    )
    axes[1, 0].set_title(f"{first_name}: {actual_res:.2f}m-cell density")
    axes[1, 1].imshow(
        second_density,
        origin="lower",
        extent=(0, second_spec.window_size_m, 0, second_spec.window_size_m),
        vmin=0,
        vmax=shared_vmax,
        aspect="equal",
    )
    axes[1, 1].set_title(f"{second_name}: {actual_res:.2f}m-cell density")
    figure.colorbar(im_first, ax=[axes[1, 0], axes[1, 1]], label="points/m²")

    axes[1, 2].scatter(
        first_first[:, 0],
        first_first[:, 1],
        s=0.3,
        alpha=0.25,
        label=first_name,
        rasterized=True,
    )
    axes[1, 2].scatter(
        second_first[:, 0],
        second_first[:, 1],
        s=0.3,
        alpha=0.25,
        label=second_name,
        rasterized=True,
    )
    axes[1, 2].set_title(
        "First-return overlay\n"
        f"{first_name}={np.sum(first_rn == 1) / first_spec.area_m2:.2f}, "
        f"{second_name}={np.sum(second_rn == 1) / second_spec.area_m2:.2f} first returns/m²"
    )
    axes[1, 2].legend(markerscale=6)

    for axis in axes.ravel():
        axis.set_xlim(0, min(first_spec.window_size_m, second_spec.window_size_m))
        axis.set_ylim(0, min(first_spec.window_size_m, second_spec.window_size_m))
        axis.set_aspect("equal")
        axis.set_xlabel("local x (m)")
        axis.set_ylabel("local y (m)")

    focus_text = ""
    if focus_common_id is not None:
        class_name = COMMON_CLASS_NAMES.get(int(focus_common_id), str(focus_common_id))
        first_class_density = float(np.sum(first_common == focus_common_id) / first_spec.area_m2)
        second_class_density = float(np.sum(second_common == focus_common_id) / second_spec.area_m2)
        focus_text = (
            f" | focus={class_name}: {first_name}={first_class_density:.3f}, "
            f"{second_name}={second_class_density:.3f} points/m²"
        )

    figure.suptitle(
        f"ALS sampling comparison: {first_spec.selection_reason} vs {second_spec.selection_reason}" f"{focus_text}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def save_window_diagnostic(
    *,
    dataset_name: str,
    spec: WindowSpec,
    data: Dict[str, np.ndarray],
    output_path: Path,
    scatter_budget: int,
    heatmap_resolution_m: float,
    dpi: int,
    seed: int,
) -> None:
    xyz = _local_xy(data["xyz"], spec)
    labels = np.asarray(data["common_labels"], dtype=np.int64)
    probability = min(1.0, scatter_budget / max(1, xyz.shape[0]))
    generator = _rng(seed, spec.key, "diagnostic")
    indices = np.flatnonzero(generator.random(xyz.shape[0]) < probability)
    xyz_plot = xyz[indices]
    labels_plot = labels[indices]
    grid, actual_res, cell_area = build_density_grid(xyz[:, :2], spec.window_size_m, heatmap_resolution_m)

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    axes[0].scatter(xyz_plot[:, 0], xyz_plot[:, 1], s=0.3, alpha=0.35, rasterized=True)
    axes[0].set_title(f"All points ({xyz.shape[0] / spec.area_m2:.2f} points/m²)")
    density = grid / cell_area
    image = axes[1].imshow(
        density,
        origin="lower",
        extent=(0, spec.window_size_m, 0, spec.window_size_m),
        aspect="equal",
    )
    axes[1].set_title(f"Density heatmap ({actual_res:.2f}m cells)")
    figure.colorbar(image, ax=axes[1], label="points/m²")
    for class_id, class_name in COMMON_CLASS_NAMES.items():
        mask = labels_plot == class_id
        if np.any(mask):
            axes[2].scatter(
                xyz_plot[mask, 0],
                xyz_plot[mask, 1],
                s=0.4,
                alpha=0.5,
                label=class_name,
                rasterized=True,
            )
    axes[2].set_title("Common classes")
    axes[2].legend(markerscale=5, fontsize=8)
    for axis in axes:
        axis.set_xlim(0, spec.window_size_m)
        axis.set_ylim(0, spec.window_size_m)
        axis.set_aspect("equal")
        axis.set_xlabel("local x (m)")
        axis.set_ylabel("local y (m)")
    figure.suptitle(f"{dataset_name} | {spec.cohort} | tile={spec.tile_id} | {spec.window_size_m:g}m")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def save_ecdf(
    dataframe: pd.DataFrame,
    *,
    metric: str,
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    if dataframe.empty or metric not in dataframe.columns:
        return
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for dataset_name, group in dataframe.groupby("dataset"):
        values = group[metric].dropna().to_numpy(dtype=np.float64)
        if values.size == 0:
            continue
        x = np.sort(values)
        y = np.arange(1, x.size + 1) / x.size
        axis.plot(x, y, label=f"{dataset_name} (n={x.size})")
    axis.set_xlabel(metric)
    axis.set_ylabel("Empirical cumulative probability")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
