# Repository-Native ALS Sampling-Density Analysis

This package implements the ECLAIR–DALES sampling-density study on top of the current project code rather than using a separate LAS parser.

## Reused project contracts

The implementation reuses:

- `EclairTiles` and its `labels.json` split/review filtering;
- `DalesTiles` and the DALES split manifests;
- `read_las_arrays_robust()` through each dataset's `get_raw()` method;
- `mapping_eclair_to_common.yaml` and `mapping_dales_to_common.yaml`;
- `train_id_to_common_id.yaml` and `train_id_to_common_dales.yaml`;
- `FeatureConfig` and `build_features()`;
- `VoxelizationConfig.from_cfg()` and `voxelize_from_q()`;
- the source configurations' coordinate normalization, voxel sizes, feature pooling, and majority label pooling.

The project common IDs remain unchanged:

| ID | Class |
|---:|---|
| 0 | ignore |
| 1 | ground |
| 2 | vegetation |
| 3 | buildings |
| 4 | wires |
| 5 | poles |
| 6 | fence |
| 7 | vehicle |

## Install into the project repository

Copy the package contents into the repository root:

```bash
rsync -av als_sampling_analysis_package/ /path/to/eclair_model_train/
cd /path/to/eclair_model_train
```

This adds files under existing `scripts/`, `src/analysis/`, `configs/`, and `tests/` directories. It does not replace the current training files.

## Environment

Use the same environment as training. The analysis additionally requires SciPy, pandas, matplotlib, and laspy:

```bash
python -m pip install scipy pandas matplotlib "laspy[lazrs]"
```

Confirm the environment variables referenced by the source training configurations are available:

```bash
echo "$ECLAIR_ROOT"
echo "$ECLAIR_CACHE_ROOT"
echo "$DALES_TRAIN_ROOT"
echo "$DALES_TEST_ROOT"
echo "$DALES_CACHE_ROOT_DROP_I"
```

## Execution sequence

### 1. Dry run: resolve exact source files

```bash
python scripts/analyze_als_sampling_density.py \
  --repo-root . \
  --config configs/als_sampling_density.yaml \
  --dry-run
```

### 2. Strict reader and label audit

Start with two tiles per dataset:

```bash
python scripts/analyze_als_sampling_density.py \
  --repo-root . \
  --config configs/als_sampling_density.yaml \
  --validate-only \
  --max-tiles 2
```

Inspect:

```text
runs/als_sampling_density/tables/tile_reader_audit.csv
runs/als_sampling_density/mapping_and_pipeline_contract.json
runs/als_sampling_density/tables/skipped_tiles.csv
```

The command fails on missing LAS dimensions, invalid return pairs, unexpected native label IDs, point-count mismatch, or all-ignore mapping.

### 3. Pilot analysis

Use `--max-tiles 2` without `--validate-only`:

```bash
python scripts/analyze_als_sampling_density.py \
  --repo-root . \
  --config configs/als_sampling_density.yaml \
  --max-tiles 2
```

### 4. Main analysis

```bash
python scripts/analyze_als_sampling_density.py \
  --repo-root . \
  --config configs/als_sampling_density.yaml
```

A SLURM template is provided at `scripts/run_als_sampling_density.slurm`.

## Primary outputs

```text
runs/als_sampling_density/
├── ANALYSIS_SUMMARY.md
├── mapping_and_pipeline_contract.json
├── run_manifest.json
├── selected_windows.json
├── figures/
│   ├── professor_overlay_low.png
│   ├── professor_overlay_median.png
│   ├── professor_overlay_high.png
│   ├── professor_overlay_class_4_wires.png
│   ├── professor_overlay_class_5_poles.png
│   ├── professor_overlay_class_6_fence.png
│   ├── professor_overlay_class_7_vehicle.png
│   └── individual_windows/
└── tables/
    ├── tile_reader_audit.csv
    ├── skipped_tiles.csv
    ├── window_metrics.csv
    ├── grid_density_metrics.csv
    ├── knn_spacing_metrics.csv
    ├── class_density_metrics.csv
    ├── model_voxel_metrics.csv
    └── tile_cluster_statistical_comparisons.csv
```

## Scientific design

### Primary random cohort

Only `primary_random` windows are used for population-density estimates and tile-cluster statistical comparisons. They are sampled independently of semantic content.

### Diagnostic stratified cohort

`diagnostic_stratified` windows are selected for wires, poles, fences, and vehicles. They are used for content-controlled visualizations and class-specific diagnostics, never for global density estimates.

### Raw sampling views

The analysis reports:

- all-return density;
- first-return density as a pulse-layout proxy;
- single-return and later-return density;
- multi-scale grid heterogeneity;
- XY and XYZ k-nearest-neighbour spacing for all, first, and single returns;
- class-conditioned raw density.

### Model-visible views

The native view uses each source configuration exactly:

- DALES: `coord_norm_factor=10`, normalized `voxel_size=0.02`, physical edge `0.20m`;
- ECLAIR: values from the selected ECLAIR training configuration, currently normalized `voxel_size=0.04`, physical edge `0.40m`.

Controlled views apply the same physical voxel sizes to both datasets. Feature and label pooling still use the source configuration's `VoxelizationConfig`.

### Statistical unit

Confidence intervals resample tiles, not points or individual windows. This avoids treating millions of correlated points as independent observations.

### Qualitative overlays

The two datasets use one shared point-display probability derived from the denser paired window. Unlike equal point caps, this preserves the visible relative density.

## Tests

Pure window/mapping tests:

```bash
PYTHONPATH=. pytest -q tests/test_sampling_mapping_contract.py tests/test_sampling_windows.py
```

VM integration tests using real files and caches:

```bash
ALS_SAMPLING_INTEGRATION=1 PYTHONPATH=. pytest -q tests/test_sampling_integration.py
```

## Colab usage

Run the main extraction on the VM. Export figures and CSVs to Drive. Set `export.save_selected_window_npz=true` only for a small pilot when interactive Colab visualization of point crops is needed.
