# DG candidate audit: BAB, O-CONS, and BEV

This change adds a no-training audit that evaluates all three candidate directions on the exact source-side preprocessing used by the current ECLAIR and DALES configurations. It does not modify LAS/LAZ data and it does not enable BAB, occupancy consistency, or BEV training.

## What is measured

### Boundary-aware blending (BAB gate)

- requested versus actual replacement area;
- Mix3D application and skip rates;
- host-point removal and donor-point insertion;
- class-distribution Jensen–Shannon divergence between removed and inserted regions;
- cross-provenance nearest-neighbour XY distance, height discontinuity, and label mismatch in 0.5 m, 1 m, and 2 m seam bands;
- per-class coherent 2D components intersected by the replacement boundary.

The BAB gate is positive only when Mix3D is usually applicable and at least 10% of the audited coherent components are cut. This establishes that destructive cuts are frequent enough to justify a BAB run; it does not by itself prove that a trained model's errors occur at seams.

### Occupancy consistency (O-CONS gate)

Two perturbations are compared at the model's actual input voxel size:

- point thinning, which may leave the active sparse tensor nearly unchanged;
- active-voxel masking, which guarantees a model-visible occupancy change.

For each 10%, 20%, and 30% strength the audit reports active-voxel Jaccard overlap, changed-voxel fraction, per-class point/voxel retention, complete class disappearance, component fragmentation, and spatial-extent retention. Wire and pole classes receive a minimum-support safeguard in the active-voxel-mask candidate.

The default gate accepts only active-voxel masking with an 8–35% median active-site change, no protected-class disappearance, and at least 70% utility-class active-voxel retention. These limits can be changed in `configs/dg_candidate_audit.yaml` before results are produced.

### LiDOG-inspired BEV gate

- fraction of input active voxels retained by the configured XYZ bounds;
- per-class bounds retention;
- occupied-cell and multi-class-cell rates at 0.2 m, 0.5 m, and 1 m;
- first, last, majority, and highest-Z single-label positive-cell recall (plus voxel-retention diagnostics);
- multi-label positive cells;
- collision class pairs;
- four-height-slice collision rate;
- dense feature/logit memory estimates.

The audit also emits a static blocker when an input-voxel `selected_idx` map is supplied to a deep feature level such as `block8`. Those arrays are in different coordinate/index spaces and must not be assumed to align.

## Files added or changed

- `src/dg_audit.py`: NumPy audit kernels.
- `scripts/audit_dg_candidates.py`: repository/config-aware entry point.
- `configs/dg_candidate_audit.yaml`: auditable sampling, perturbation, and gate settings.
- `scripts/smoke_test_dg_audit.py`: dependency-light synthetic regression test.
- `scripts/slurm_audit_dg_candidates.sh`: HPC launcher.
- `src/mix3d.py`: opt-in Mix3D trace masks; normal training behavior is unchanged.
- `src/bev_head.py`: uses a safe dataclass `default_factory` for the projector config.

## Run sequence

From `eclair_model_train`:

```bash
python -m py_compile train.py src/*.py scripts/*.py
python scripts/smoke_test_mix3d.py
python scripts/smoke_test_dg_audit.py
python scripts/audit_dg_candidates.py \
  --audit-config configs/dg_candidate_audit.yaml \
  --validate-only
```

The first real-data pass should use one sample per dataset:

```bash
python scripts/audit_dg_candidates.py \
  --audit-config configs/dg_candidate_audit.yaml \
  --samples 1 \
  --out-dir /scratch/m23csa510/e0_results/dg_candidate_audit_smoke
```

After checking the smoke output, submit the six-sample pass:

```bash
sbatch scripts/slurm_audit_dg_candidates.sh
```

To change the audit size without editing files:

```bash
sbatch --export=ALL,SAMPLES=12 scripts/slurm_audit_dg_candidates.sh
```

## Outputs

`summary.json` is the decision-level result. `samples.jsonl` preserves the complete nested measurements. The CSV files support plotting and manual review:

- `occupancy.csv`
- `bab_samples.csv`, `bab_seams.csv`, `bab_classes.csv`
- `ocons.csv`, `ocons_classes.csv`
- `bev.csv`, `bev_classes.csv`, `bev_collision_pairs.csv`
- `config_resolved.json`

## Decision after the pass

Use the DALES summary first because DALES→ECLAIR is the harder locked direction.

1. If at least one O-CONS setting is eligible, it is the first independent training candidate.
2. Train BAB only if its gate is positive and the per-class table shows cuts in classes implicated by the M1 regressions or utility objects—not merely abundant ground/vegetation components.
3. Do not enable the current BEV head while the static deep-feature index-map risk is present or bounds retention is below 95%. If single-label utility retention is below 90%, implement a multi-label or height-sliced target before training.
4. If both O-CONS and a corrected BEV pass their gates, they are the two highest-value parallel training runs. BAB remains the conditional Mix3D repair.
