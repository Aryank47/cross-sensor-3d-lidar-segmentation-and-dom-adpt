#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repository-native ECLAIR/DALES ALS sampling-density analysis")
    parser.add_argument("--config", type=Path, required=True, help="Analysis YAML configuration")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Project repository root containing train.py, src/, configs/",
    )
    parser.add_argument("--validate-only", action="store_true", help="Audit selected files and exit")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build bundles and list selected files only",
    )
    parser.add_argument("--max-tiles", type=int, default=None, help="Override max tiles per dataset")
    return parser.parse_args()


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def dataframe_markdown(dataframe: pd.DataFrame) -> str:
    try:
        return dataframe.to_markdown(index=False)
    except ImportError:
        return "```text\n" + dataframe.to_string(index=False) + "\n```"


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def spec_from_dict(data: Dict[str, Any]):
    from src.sampling_types import WindowSpec

    return WindowSpec(**data)


def window_payload(raw: Dict[str, np.ndarray], indices: np.ndarray, common: np.ndarray) -> Dict[str, np.ndarray]:
    payload: Dict[str, np.ndarray] = {
        "xyz": np.asarray(raw["xyz"])[indices],
        "native_labels": np.asarray(raw["native_labels"])[indices],
        "common_labels": np.asarray(common)[indices],
        "return_number": np.asarray(raw["return_number"])[indices],
        "number_of_returns": np.asarray(raw["number_of_returns"])[indices],
    }
    intensity = raw.get("intensity")
    rgb = raw.get("rgb")
    payload["intensity"] = None if intensity is None else np.asarray(intensity)[indices]
    payload["rgb"] = None if rgb is None else np.asarray(rgb)[indices]
    return payload


def load_window(bundle: Any, spec: Any) -> Dict[str, np.ndarray]:
    from src.sampling_windows import extract_window_indices

    raw = bundle.dataset.get_raw(spec.tile_index)
    common = bundle.raw_to_common(raw["native_labels"])
    indices = extract_window_indices(raw["xyz"], spec)
    return window_payload(raw, indices, common)


def write_markdown_summary(
    output_path: Path,
    window_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> None:
    lines = [
        "# ALS Sampling-Density Analysis Summary",
        "",
        "## Scope",
        "",
        "This run compares the realized raw-point and model-visible sampling distributions of the exact ECLAIR and DALES source splits selected by the project training configurations.",
        "",
        "## Dataset audit",
        "",
    ]
    if not audit_df.empty:
        for dataset_name, group in audit_df.groupby("dataset"):
            lines.append(
                f"- **{dataset_name}:** {group.shape[0]} analysed tiles, " f"{int(group['point_count'].sum()):,} points audited."
            )
    lines.extend(["", "## Primary random-window density", ""])
    primary = window_df[window_df["cohort"] == "primary_random"] if not window_df.empty else window_df
    if not primary.empty:
        summary = (
            primary.groupby(["dataset", "window_size_m"])[
                [
                    "raw_density_pts_m2",
                    "first_return_density_pts_m2",
                    "single_return_density_pts_m2",
                ]
            ]
            .median()
            .reset_index()
        )
        lines.append(dataframe_markdown(summary))
    else:
        lines.append("No primary windows were produced.")
    lines.extend(["", "## Tile-aware comparisons", ""])
    if not comparison_df.empty:
        selected = comparison_df[
            comparison_df["metric"].isin(
                [
                    "raw_density_pts_m2",
                    "first_return_density_pts_m2",
                    "distance_m_median",
                    "occupied_voxels_per_m2",
                ]
            )
        ].head(50)
        lines.append(dataframe_markdown(selected))
    else:
        lines.append("No comparison table was produced.")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- First-return density is treated as a pulse-layout proxy, not as a guaranteed reconstruction of emitted pulse density.",
            "- Global conclusions use the `primary_random` cohort only; class-stratified windows are diagnostic and are excluded from population estimates.",
            "- Statistical intervals use tile-cluster bootstrap rather than treating overlapping windows or individual points as independent.",
            "- Native-pipeline voxel views reproduce each training configuration; controlled views apply common physical voxel sizes to both datasets.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    if not (repo_root / "src").exists():
        raise SystemExit(f"Invalid repo root: {repo_root}; expected src/ directory")
    sys.path.insert(0, str(repo_root))

    from src.config_loader import load_yaml
    from src.sampling_datasets import build_dataset_bundle, select_tile_records
    from src.sampling_metrics import (
        SCIPY_AVAILABLE,
        build_native_tile_voxel_context,
        compute_class_rows,
        compute_grid_rows,
        compute_knn_rows,
        compute_voxel_rows,
        compute_window_row,
        tile_cluster_comparison,
    )
    from src.sampling_plots import save_ecdf, save_pair_overlay, save_window_diagnostic
    from src.sampling_types import COMMON_CLASS_NAMES
    from src.sampling_validation import determine_footprint, inspect_las_header, validate_header_contract, validate_raw_contract
    from src.sampling_windows import (
        build_candidate_summaries,
        extract_window_indices,
        select_diagnostic_windows,
        select_primary_windows,
    )

    analysis_cfg_path = args.config.expanduser().resolve()
    cfg = load_yaml(analysis_cfg_path)
    output_dir = resolve_path(repo_root, cfg["run"]["output_dir"])
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    windows_dir = output_dir / "sampled_windows"
    for directory in (output_dir, tables_dir, figures_dir, windows_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["run"].get("seed", 1337))
    dataset_order = [str(x) for x in cfg.get("dataset_order", ["ECLAIR", "DALES"])]
    bundles: Dict[str, Any] = {}
    selected_tiles: Dict[str, list[Any]] = {}

    for dataset_key in ("eclair", "dales"):
        dcfg = cfg["datasets"][dataset_key]
        bundle = build_dataset_bundle(
            name=dataset_key,
            training_cfg_path=resolve_path(repo_root, dcfg["training_config"]),
            native_mapping_path=resolve_path(repo_root, dcfg["native_to_common_mapping"]),
            train_mapping_path=resolve_path(repo_root, dcfg["train_to_common_mapping"]),
            split=str(dcfg.get("split", "train")),
            use_project_cache=bool(dcfg.get("use_project_cache", True)),
            require_project_cache=dcfg.get("require_project_cache"),
            seed=seed,
        )
        bundles[bundle.name] = bundle
        max_tiles = args.max_tiles if args.max_tiles is not None else dcfg.get("max_tiles")
        selected_tiles[bundle.name] = select_tile_records(
            bundle.tile_records,
            max_tiles=None if max_tiles is None else int(max_tiles),
            seed=seed,
            dataset_name=bundle.name,
        )

    mapping_audit = {
        name: {
            "training_config": str(bundle.training_config_path),
            "source_split": bundle.source_split,
            "native_to_common": {
                str(native_id): int(bundle.native_to_common_lut[native_id]) for native_id in sorted(bundle.expected_native_ids)
            },
            "train_to_common": {
                str(train_id): int(bundle.train_to_common_lut[train_id]) for train_id in range(bundle.num_train_classes)
            },
            "common_names": COMMON_CLASS_NAMES,
            "patch": dataclasses.asdict(bundle.patch_cfg),
            "features": dataclasses.asdict(bundle.feat_cfg),
            "voxelization": dataclasses.asdict(bundle.voxel_cfg),
        }
        for name, bundle in bundles.items()
    }
    save_json(output_dir / "mapping_and_pipeline_contract.json", mapping_audit)
    save_json(
        output_dir / "run_manifest.json",
        {
            "analysis_config": str(analysis_cfg_path),
            "repo_root": str(repo_root),
            "timestamp_unix": time.time(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy_available": SCIPY_AVAILABLE,
            "selected_tiles": {
                name: [dataclasses.asdict(record) for record in records] for name, records in selected_tiles.items()
            },
            "config": cfg,
        },
    )

    if args.dry_run:
        print(
            json.dumps(
                {name: [str(x.path) for x in records] for name, records in selected_tiles.items()},
                indent=2,
            )
        )
        return 0

    audit_rows: list[Dict[str, Any]] = []
    skipped_rows: list[Dict[str, Any]] = []
    window_rows: list[Dict[str, Any]] = []
    grid_rows: list[Dict[str, Any]] = []
    knn_rows: list[Dict[str, Any]] = []
    class_rows: list[Dict[str, Any]] = []
    voxel_rows: list[Dict[str, Any]] = []
    selected_specs: Dict[str, Any] = {}
    selection_rows: list[Dict[str, Any]] = []
    diagnostic_figure_count: Dict[str, int] = {name: 0 for name in bundles}

    sampling_cfg = cfg["sampling"]
    metrics_cfg = cfg["metrics"]
    plot_cfg = cfg["plots"]

    for dataset_name in dataset_order:
        bundle = bundles[dataset_name]
        dcfg = cfg["datasets"][dataset_name.lower()]
        for tile in selected_tiles[dataset_name]:
            print(
                f"[{dataset_name}] reading tile {tile.tile_index}: {tile.path}",
                flush=True,
            )
            header = inspect_las_header(tile.path)
            validate_header_contract(tile.path, header)
            raw = bundle.dataset.get_raw(tile.tile_index)
            audit = validate_raw_contract(
                bundle,
                tile,
                raw,
                header,
                strict_returns=bool(cfg["validation"].get("strict_returns", True)),
            )
            audit_rows.append({**header, **audit})
            common = bundle.raw_to_common(raw["native_labels"])

            if args.validate_only:
                del raw, common
                continue

            try:
                footprint = determine_footprint(
                    raw["xyz"],
                    dcfg["footprint"],
                    dataset_name=dataset_name,
                    tile_id=tile.tile_id,
                )
            except RuntimeError as exc:
                if str(exc).startswith("SKIP_PARTIAL_TILE::"):
                    skipped_rows.append(
                        {
                            "dataset": dataset_name,
                            "tile_index": tile.tile_index,
                            "tile_id": tile.tile_id,
                            "path": str(tile.path),
                            "reason": str(exc),
                        }
                    )
                    print(str(exc), flush=True)
                    del raw, common
                    continue
                raise

            native_tile_context = (
                build_native_tile_voxel_context(bundle, raw, seed=seed)
                if bool(metrics_cfg.get("native_eclair_full_tile_voxel_context", True))
                and bool(metrics_cfg.get("model_voxel_enabled", True))
                else None
            )

            candidate_count = int(sampling_cfg["windows_per_tile_per_size"]) * int(sampling_cfg.get("candidate_multiplier", 6))
            candidates = build_candidate_summaries(
                tile=tile,
                xyz_local=raw["xyz"],
                common_labels=common,
                return_number=raw["return_number"],
                number_of_returns=raw["number_of_returns"],
                footprint=footprint,
                window_sizes_m=[float(x) for x in sampling_cfg["window_sizes_m"]],
                candidates_per_size=candidate_count,
                selection_grid_m=float(sampling_cfg.get("selection_grid_m", 1.0)),
                seed=seed,
            )
            primary_specs = select_primary_windows(
                candidates,
                windows_per_size=int(sampling_cfg["windows_per_tile_per_size"]),
                min_points=int(sampling_cfg["min_points_per_window"]),
                max_overlap_fraction=float(sampling_cfg.get("max_overlap_fraction", 0.25)),
                seed=seed,
            )
            diagnostic_specs = select_diagnostic_windows(
                candidates,
                focus_common_ids=[int(x) for x in sampling_cfg["diagnostic_common_class_ids"]],
                windows_per_class_per_size=int(sampling_cfg.get("diagnostic_windows_per_class_per_size", 1)),
                min_points=int(sampling_cfg["min_points_per_window"]),
                min_focus_points=int(sampling_cfg.get("min_focus_points", 25)),
                max_overlap_fraction=float(sampling_cfg.get("max_overlap_fraction", 0.25)),
            )
            selection_rows.append(
                {
                    "dataset": dataset_name,
                    "tile_index": tile.tile_index,
                    "tile_id": tile.tile_id,
                    "path": str(tile.path),
                    "footprint_x_min_m": footprint.x_min_m,
                    "footprint_y_min_m": footprint.y_min_m,
                    "footprint_width_m": footprint.width_m,
                    "footprint_height_m": footprint.height_m,
                    "candidate_count": len(candidates),
                    "candidates_meeting_min_points": sum(
                        item.point_count >= int(sampling_cfg["min_points_per_window"]) for item in candidates
                    ),
                    "primary_windows_selected": len(primary_specs),
                    "diagnostic_windows_selected": len(diagnostic_specs),
                }
            )

            print(
                f"[{dataset_name}] window selection tile={tile.tile_id}: "
                f"frame=[{footprint.x_min_m:.3f}, "
                f"{footprint.x_min_m + footprint.width_m:.3f}] x "
                f"[{footprint.y_min_m:.3f}, "
                f"{footprint.y_min_m + footprint.height_m:.3f}], "
                f"candidates={len(candidates)}, primary={len(primary_specs)}, "
                f"diagnostic={len(diagnostic_specs)}",
                flush=True,
            )

            candidate_by_id = {item.spec.candidate_id: item for item in candidates}
            for spec in [*primary_specs, *diagnostic_specs]:
                indices = extract_window_indices(raw["xyz"], spec)
                if indices.size < int(sampling_cfg["min_points_per_window"]):
                    continue
                candidate_estimate = candidate_by_id[spec.candidate_id].point_count
                exact_difference = int(indices.size) - int(candidate_estimate)
                if abs(exact_difference) > int(cfg["validation"].get("max_candidate_count_difference", 4)):
                    raise RuntimeError(
                        f"Candidate integral-count mismatch for {spec.key}: "
                        f"estimate={candidate_estimate}, exact={indices.size}, diff={exact_difference}"
                    )

                payload = window_payload(raw, indices, common)
                selected_specs[spec.key] = dataclasses.asdict(spec)
                window_rows.append(
                    {
                        **compute_window_row(
                            bundle,
                            spec,
                            payload["xyz"],
                            payload["native_labels"],
                            payload["common_labels"],
                            payload["return_number"],
                            payload["number_of_returns"],
                        ),
                        "candidate_count_estimate": int(candidate_estimate),
                        "candidate_exact_difference": exact_difference,
                    }
                )
                grid_rows.extend(
                    compute_grid_rows(
                        bundle,
                        spec,
                        payload["xyz"],
                        payload["return_number"],
                        payload["number_of_returns"],
                        [float(x) for x in metrics_cfg["grid_resolutions_m"]],
                    )
                )
                class_rows.extend(
                    compute_class_rows(
                        bundle,
                        spec,
                        payload["xyz"],
                        payload["common_labels"],
                        occupied_area_grid_m=float(metrics_cfg.get("class_occupied_area_grid_m", 1.0)),
                    )
                )
                if bool(metrics_cfg.get("knn_enabled", True)):
                    knn_rows.extend(
                        compute_knn_rows(
                            bundle,
                            spec,
                            payload["xyz"],
                            payload["common_labels"],
                            payload["return_number"],
                            payload["number_of_returns"],
                            k_values=[int(x) for x in metrics_cfg["knn_k_values"]],
                            max_points_per_subset=int(metrics_cfg["max_knn_points_per_subset"]),
                            max_points_per_class=int(metrics_cfg["max_knn_points_per_class"]),
                            min_class_points=int(metrics_cfg["min_class_points_for_knn"]),
                            class_ids=[int(x) for x in metrics_cfg["knn_common_class_ids"]],
                            seed=seed,
                        )
                    )
                if bool(metrics_cfg.get("model_voxel_enabled", True)):
                    voxel_rows.extend(
                        compute_voxel_rows(
                            bundle,
                            spec,
                            payload["xyz"],
                            payload["native_labels"],
                            payload["common_labels"],
                            payload["intensity"],
                            payload["return_number"],
                            payload["number_of_returns"],
                            payload["rgb"],
                            controlled_physical_sizes_m=[float(x) for x in metrics_cfg["controlled_physical_voxel_sizes_m"]],
                            max_points_for_voxelization=int(metrics_cfg["max_points_for_model_voxelization"]),
                            seed=seed,
                            tile_point_indices=indices,
                            native_tile_context=native_tile_context,
                        )
                    )

                if bool(plot_cfg.get("save_individual_diagnostics", True)) and diagnostic_figure_count[dataset_name] < int(
                    plot_cfg.get("max_individual_diagnostics_per_dataset", 12)
                ):
                    save_window_diagnostic(
                        dataset_name=dataset_name,
                        spec=spec,
                        data=payload,
                        output_path=figures_dir
                        / "individual_windows"
                        / f"{dataset_name}_{spec.cohort}_{spec.tile_id}_{spec.candidate_id}.png",
                        scatter_budget=int(plot_cfg["scatter_budget_for_denser_window"]),
                        heatmap_resolution_m=float(plot_cfg["heatmap_resolution_m"]),
                        dpi=int(plot_cfg["dpi"]),
                        seed=seed,
                    )
                    diagnostic_figure_count[dataset_name] += 1

                if bool(cfg["export"].get("save_selected_window_npz", False)):
                    np.savez_compressed(
                        windows_dir / f"{dataset_name}_{short_hash(spec.key)}_{spec.cohort}.npz",
                        **{k: v for k, v in payload.items() if v is not None},
                        window_spec_json=json.dumps(dataclasses.asdict(spec)),
                    )
                del payload
            del raw, common, native_tile_context

    audit_df = pd.DataFrame(audit_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    window_df = pd.DataFrame(window_rows)
    grid_df = pd.DataFrame(grid_rows)
    knn_df = pd.DataFrame(knn_rows)
    class_df = pd.DataFrame(class_rows)
    voxel_df = pd.DataFrame(voxel_rows)
    selection_df = pd.DataFrame(selection_rows)

    audit_df.to_csv(tables_dir / "tile_reader_audit.csv", index=False)
    skipped_df.to_csv(tables_dir / "skipped_tiles.csv", index=False)
    window_df.to_csv(tables_dir / "window_metrics.csv", index=False)
    grid_df.to_csv(tables_dir / "grid_density_metrics.csv", index=False)
    knn_df.to_csv(tables_dir / "knn_spacing_metrics.csv", index=False)
    class_df.to_csv(tables_dir / "class_density_metrics.csv", index=False)
    voxel_df.to_csv(tables_dir / "model_voxel_metrics.csv", index=False)
    save_json(output_dir / "selected_windows.json", selected_specs)
    selection_df.to_csv(
        tables_dir / "window_selection_audit.csv",
        index=False,
    )

    if args.validate_only:
        print(f"Validation complete. Audit: {tables_dir / 'tile_reader_audit.csv'}")
        return 0

    comparison_rows: list[Dict[str, Any]] = []
    primary_window = window_df[window_df["cohort"] == "primary_random"] if not window_df.empty else window_df
    for metric in [
        "raw_density_pts_m2",
        "first_return_density_pts_m2",
        "single_return_density_pts_m2",
        "later_return_density_pts_m2",
        "later_return_fraction",
    ]:
        comparison_rows.extend(
            tile_cluster_comparison(
                primary_window,
                metric=metric,
                group_columns=["window_size_m"],
                dataset_order=dataset_order,
                iterations=int(cfg["statistics"]["bootstrap_iterations"]),
                seed=seed,
            )
        )

    missing_primary = [
        dataset_name
        for dataset_name in dataset_order
        if primary_window.empty or primary_window[primary_window["dataset"] == dataset_name].empty
    ]

    if missing_primary:
        raise RuntimeError(
            "Sampling analysis is incomplete: no primary_random windows were produced for "
            f"{missing_primary}. See {tables_dir / 'window_selection_audit.csv'} and "
            f"{tables_dir / 'skipped_tiles.csv'}."
        )
    if not grid_df.empty:
        primary_grid = grid_df[grid_df["cohort"] == "primary_random"]
        for metric in [
            "occupied_fraction",
            "density_cv_all_cells",
            "density_gini_all_cells",
            "cell_density_pts_m2_median",
        ]:
            comparison_rows.extend(
                tile_cluster_comparison(
                    primary_grid,
                    metric=metric,
                    group_columns=[
                        "window_size_m",
                        "subset",
                        "grid_resolution_requested_m",
                    ],
                    dataset_order=dataset_order,
                    iterations=int(cfg["statistics"]["bootstrap_iterations"]),
                    seed=seed,
                )
            )
    if not knn_df.empty:
        primary_knn = knn_df[knn_df["cohort"] == "primary_random"]
        for metric in ["distance_m_median", "distance_m_q95"]:
            comparison_rows.extend(
                tile_cluster_comparison(
                    primary_knn,
                    metric=metric,
                    group_columns=[
                        "window_size_m",
                        "subset",
                        "common_class_id",
                        "coordinate_space",
                        "k",
                    ],
                    dataset_order=dataset_order,
                    iterations=int(cfg["statistics"]["bootstrap_iterations"]),
                    seed=seed,
                )
            )
    if not voxel_df.empty and "scope" in voxel_df.columns:
        primary_voxel = voxel_df[(voxel_df["cohort"] == "primary_random") & (voxel_df["scope"] == "all")]
        for metric in [
            "occupied_voxels_per_m2",
            "points_per_occupied_voxel_mean",
            "singleton_voxel_fraction",
        ]:
            comparison_rows.extend(
                tile_cluster_comparison(
                    primary_voxel,
                    metric=metric,
                    group_columns=[
                        "window_size_m",
                        "view_name",
                        "voxel_size_physical_m",
                    ],
                    dataset_order=dataset_order,
                    iterations=int(cfg["statistics"]["bootstrap_iterations"]),
                    seed=seed,
                )
            )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(tables_dir / "tile_cluster_statistical_comparisons.csv", index=False)

    if not primary_window.empty:
        for size, group in primary_window.groupby("window_size_m"):
            save_ecdf(
                group,
                metric="raw_density_pts_m2",
                output_path=figures_dir / f"ecdf_raw_density_{float(size):g}m.png",
                title=f"Raw all-return density ({float(size):g}m random windows)",
                dpi=int(plot_cfg["dpi"]),
            )
            save_ecdf(
                group,
                metric="first_return_density_pts_m2",
                output_path=figures_dir / f"ecdf_first_return_density_{float(size):g}m.png",
                title=f"First-return density ({float(size):g}m random windows)",
                dpi=int(plot_cfg["dpi"]),
            )

    plot_size = float(plot_cfg["professor_window_size_m"])
    primary_at_size = (
        primary_window[np.isclose(primary_window["window_size_m"], plot_size)] if not primary_window.empty else primary_window
    )
    if all(name in bundles for name in dataset_order) and not primary_at_size.empty:
        representatives: Dict[str, Dict[str, Any]] = {}
        for dataset_name in dataset_order:
            group = primary_at_size[primary_at_size["dataset"] == dataset_name].sort_values("raw_density_pts_m2")
            if group.empty:
                continue
            representatives[dataset_name] = {
                "low": group.iloc[int(round(0.10 * (len(group) - 1)))],
                "median": group.iloc[int(round(0.50 * (len(group) - 1)))],
                "high": group.iloc[int(round(0.90 * (len(group) - 1)))],
            }
        if all(name in representatives for name in dataset_order):
            for descriptor in ("low", "median", "high"):
                first_row = representatives[dataset_order[0]][descriptor]
                second_row = representatives[dataset_order[1]][descriptor]
                first_spec = spec_from_dict(selected_specs[str(first_row["window_key"])])
                second_spec = spec_from_dict(selected_specs[str(second_row["window_key"])])
                save_pair_overlay(
                    first_name=dataset_order[0],
                    first_spec=first_spec,
                    first_data=load_window(bundles[dataset_order[0]], first_spec),
                    second_name=dataset_order[1],
                    second_spec=second_spec,
                    second_data=load_window(bundles[dataset_order[1]], second_spec),
                    output_path=figures_dir / f"professor_overlay_{descriptor}.png",
                    scatter_budget_for_denser_window=int(plot_cfg["scatter_budget_for_denser_window"]),
                    heatmap_resolution_m=float(plot_cfg["heatmap_resolution_m"]),
                    dpi=int(plot_cfg["dpi"]),
                    seed=seed,
                )

    if not class_df.empty and not window_df.empty:
        diagnostic_windows = window_df[
            (window_df["cohort"] == "diagnostic_stratified") & np.isclose(window_df["window_size_m"], plot_size)
        ]
        diagnostic_classes = [int(x) for x in sampling_cfg["diagnostic_common_class_ids"]]
        for class_id in diagnostic_classes:
            pair_rows: Dict[str, Any] = {}
            for dataset_name in dataset_order:
                candidate_keys = diagnostic_windows[
                    (diagnostic_windows["dataset"] == dataset_name) & (diagnostic_windows["focus_common_id"] == class_id)
                ]["window_key"]
                candidates = class_df[
                    (class_df["dataset"] == dataset_name)
                    & (class_df["common_class_id"] == class_id)
                    & (class_df["window_key"].isin(candidate_keys))
                ].sort_values("density_pts_per_occupied_m2", ascending=False)
                if not candidates.empty:
                    pair_rows[dataset_name] = candidates.iloc[0]
            if all(name in pair_rows for name in dataset_order):
                first_spec = spec_from_dict(selected_specs[str(pair_rows[dataset_order[0]]["window_key"])])
                second_spec = spec_from_dict(selected_specs[str(pair_rows[dataset_order[1]]["window_key"])])
                save_pair_overlay(
                    first_name=dataset_order[0],
                    first_spec=first_spec,
                    first_data=load_window(bundles[dataset_order[0]], first_spec),
                    second_name=dataset_order[1],
                    second_spec=second_spec,
                    second_data=load_window(bundles[dataset_order[1]], second_spec),
                    output_path=figures_dir / f"professor_overlay_class_{class_id}_{COMMON_CLASS_NAMES[class_id]}.png",
                    scatter_budget_for_denser_window=int(plot_cfg["scatter_budget_for_denser_window"]),
                    heatmap_resolution_m=float(plot_cfg["heatmap_resolution_m"]),
                    dpi=int(plot_cfg["dpi"]),
                    seed=seed,
                    focus_common_id=class_id,
                )

    write_markdown_summary(output_dir / "ANALYSIS_SUMMARY.md", window_df, comparison_df, audit_df)
    print(f"Analysis complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
