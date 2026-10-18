# Validation Report

## Completed checks

- `ruff check`: passed with no remaining findings.
- `ruff format`: applied to the Python implementation.
- `python -m py_compile`: passed for the CLI, all `src/analysis` modules, and tests.
- Pure window-selection tests: 2 passed.
- Mapping-contract plus window tests in a reconstructed project tree: 3 passed.
- Reader-parity integration tests on generated LAS files: 2 passed for ECLAIR and DALES.
- Full synthetic end-to-end run completed and generated:
  - reader audit;
  - random-window metrics;
  - grid-density metrics;
  - model-voxel metrics;
  - tile-cluster comparisons;
  - ECDF plots;
  - professor-facing low/median/high overlays.
- `--validate-only` synthetic run completed successfully.

## What was exercised

The synthetic integration setup used valid LAS 1.4 point-format 7 files for ECLAIR and LAS 1.2 point-format 1 files for DALES, with classification, return number, number of returns, intensity, and ECLAIR RGB fields. It exercised:

- the project `read_las_arrays_robust()` path through `EclairTiles.get_raw()` and `DalesTiles.get_raw()`;
- ECLAIR metadata split/review filtering;
- DALES manifest resolution;
- native/common mappings;
- raw density and return metrics;
- candidate integral-grid counting versus exact extraction;
- full-tile ECLAIR native voxel slicing;
- crop-faithful DALES native voxelization;
- controlled physical voxel views;
- tile-cluster comparisons;
- density-preserving overlays.

## Remaining VM validation

The package has not been run against the user's complete real ECLAIR and DALES datasets in this environment. The required next step is the provided two-tile VM sequence:

1. `--dry-run`
2. `--validate-only --max-tiles 2`
3. full pilot with `--max-tiles 2`
4. inspect audit CSVs and figures
5. run the main configured study

The real VM integration tests should also be run with:

```bash
ALS_SAMPLING_INTEGRATION=1 PYTHONPATH=. pytest -q tests/test_sampling_integration.py
```
