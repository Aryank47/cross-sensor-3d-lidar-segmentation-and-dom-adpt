# O-CONS-10 and BEV-ALS-H4-ML implementation and run guide

## 1. Frozen experiment contract

Both discovery methods train from the same DALES recipe as the final 120 m
M1-25 run, with Mix3D disabled. The frozen comparator is
`configs/m0_dales_frozen_29364.yaml`.

| Method | Config | Mix3D | New mechanism |
|---|---|---:|---|
| O-CONS-10 | `configs/dg_dales_ocons10.yaml` | off | 10% post-voxel active-coordinate masking plus clean-teacher consistency |
| BEV-ALS-H4-ML | `configs/dg_dales_bev_als_h4_ml.yaml` | off | four crop-relative height slices with multi-label BEV supervision |

`method.name` is validated before data loading. A run fails if its enabled
flags do not match the declared method. The resolved contract is written to
`method_contract.json` and embedded in every checkpoint.

## 2. O-CONS-10 behavior

For every collated sparse batch, the implementation creates a second view by
removing 10% of the already voxelized active coordinates. This guarantees that
the perturbation is visible to MinkowskiEngine; it is not merely a raw-point
change that disappears during voxelization.

DALES train IDs 5 (poles) and 6 (power lines) are protected independently in
each sample. At least `max(3, ceil(0.70 * class_support))` active coordinates
are retained for each present protected class. A protected-class disappearance
is a hard runtime error.

The clean and perturbed views both receive supervised segmentation loss. Clean
logits are detached and aligned to perturbed output coordinates by an exact,
collision-free integer-coordinate lookup. The total loss is

```text
0.5 * L_seg(clean)
+ 0.5 * L_seg(perturbed)
+ lambda(epoch) * 0.5 * [KL_micro + KL_macro_present]
```

`lambda` ramps linearly from 0.05 at epoch 1 to 0.50 at epoch 10. The macro KL
contains only classes with at least eight perturbed voxels in that step.

## 3. BEV-ALS-H4-ML behavior

This is a new ALS-specific branch. It does not use the old trial BEV projector,
its absolute `[-10,10]` bounds, or `bev_selected_idx`.

| Property | Selected design |
|---|---|
| Source feature | native `block8` sparse tensor, tensor stride 1 |
| XY frame | center of the augmented crop's voxel bounding box |
| Bounds | `[-90,90)` m around that center |
| Cell size | 0.50 m |
| Dense grid | 360 × 360 |
| Height representation | per-crop 25%, 50%, and 75% voxel-height quantiles |
| Slices | four |
| Feature pooling | per-slice/per-cell maximum after 128→16 channel reduction |
| Target | independent class-presence bits in every slice/cell |
| Decoder | three 3×3 conv-BN-ReLU blocks, then slice/class logits |
| Inference | auxiliary projection/head skipped; only 3D segmentation is used |

The crop-relative frame makes the target invariant to global XY and Z offsets.
Multi-label targets preserve two classes occupying the same vertical slice and
BEV cell. Height slices retain coarse vertical structure without imposing an
absolute ALS elevation range.

The BEV objective is

```text
L_seg + lambda(epoch) * [0.5 * BCE_occupied + 0.5 * Dice_present]
```

Unoccupied slice/cells are excluded from BCE so the enormous empty background
does not dominate. The BEV weight ramps from 0.05 at epoch 1 to 0.50 at epoch
10. Both target and `block8` in-bounds fractions must be at least 0.995; falling
below this is a hard correctness error.

## 4. Apply and static verification

Run from `eclair_model_train`:

```bash
git apply --check --verbose DG_OCONS_BEV_ALS_IMPLEMENTATION_v2.patch
git apply --verbose DG_OCONS_BEV_ALS_IMPLEMENTATION_v2.patch

python -m py_compile train.py src/*.py scripts/*.py
bash -n scripts/slurm_dg_method_diagnostic.sh
bash -n scripts/slurm_dg_method_train.sh

python scripts/smoke_test_mix3d.py
python scripts/smoke_test_dg_audit.py
python scripts/smoke_test_dg_methods.py
```

The last smoke test checks O-CONS determinism, protection and exact coordinate
matching; BEV translation invariance, collision preservation, native feature
projection, loss shape, perfect-prediction metrics, and backward propagation.

## 5. Environment and preflight

```bash
export DALES_TRAIN_ROOT=/scratch/m23csa510/dales/dales/train
export DALES_TEST_ROOT=/scratch/m23csa510/dales/dales/test
export DALES_CACHE_ROOT_DROP_I=/scratch/m23csa510/dales/dales_cache

python scripts/preflight_dg_method.py \
  --config configs/dg_dales_ocons10_diagnostic.yaml \
  --expect-method ocons \
  --build-model

python scripts/preflight_dg_method.py \
  --config configs/dg_dales_bev_als_h4_ml_diagnostic.yaml \
  --expect-method bev_als \
  --build-model
```

Preflight resolves config inheritance, verifies frozen scientific parity,
rejects Mix3D/method leakage, validates class IDs and BEV dimensions, and
instantiates the actual model.

## 6. Small diagnostic training

The diagnostic configs run one epoch, exactly three training batches, no
validation, and no final test. They still perform one optimizer step because
the frozen gradient accumulation is three. Their artifacts are explicitly
marked as diagnostic and must not be reported as experimental results.

Submit both one-GPU diagnostics in parallel:

```bash
sbatch \
  --export=ALL,CONFIG_SRC=configs/dg_dales_ocons10_diagnostic.yaml,METHOD_TAG=ocons \
  scripts/slurm_dg_method_diagnostic.sh

sbatch \
  --export=ALL,CONFIG_SRC=configs/dg_dales_bev_als_h4_ml_diagnostic.yaml,METHOD_TAG=bev_als \
  scripts/slurm_dg_method_diagnostic.sh
```

For an allocated interactive GPU, run them sequentially:

```bash
python train.py --config configs/dg_dales_ocons10_diagnostic.yaml
python train.py --config configs/dg_dales_bev_als_h4_ml_diagnostic.yaml
```

The Slurm jobs write to:

```text
/scratch/m23csa510/e0_results/diagnostic_ocons_JOBID
/scratch/m23csa510/e0_results/diagnostic_bev_als_JOBID
```

## 7. Diagnostic acceptance gates

Common gates:

- `diagnostic_complete.json` exists and has `status=ok`;
- `method_contract.json` names exactly one intended method;
- `metrics.csv` has one row with finite training losses;
- `backbone_grad_norm` is finite and greater than zero;
- `gpu_peak_reserved_mib` fits with safe headroom for the full job;
- `checkpoints/last.pt` exists;
- no Mix3D columns are active.

O-CONS gates:

- `ocons_actual_mask_fraction` is approximately 0.10;
- `ocons_active_jaccard` is approximately 0.90;
- `ocons_coordinate_match_coverage` equals 1.0;
- `ocons_protected_disappearances` equals 0;
- pole and power-line retention in `ocons_class_diagnostics.csv` is at least
  0.70 when those classes are present;
- clean, perturbed, micro-KL and macro-KL losses are finite;
- epoch-1 `aux_weight` is 0.05.

BEV-ALS gates:

- `bev_als_in_bounds_fraction` and
  `bev_als_feature_in_bounds_fraction` are at least 0.995;
- `method_aux_grad_norm` is finite and greater than zero;
- BEV, BCE and Dice losses are finite;
- epoch-1 `aux_weight` is 0.05;
- `bev_als_slice_class_diagnostics.csv` has 32 rows (4 slices × 8 classes)
  for the epoch and nonzero support for classes present in the sampled crops;
- wire/pole `bounds_retention` is at least 0.995 whenever supported.

Inspect concise outputs with:

```bash
column -s, -t < RUN_DIR/metrics.csv
cat RUN_DIR/method_contract.json
cat RUN_DIR/diagnostic_complete.json
column -s, -t < RUN_DIR/ocons_class_diagnostics.csv       # O-CONS
column -s, -t < RUN_DIR/bev_als_slice_class_diagnostics.csv  # BEV-ALS
```

## 8. Full parallel discovery training

Only after both diagnostic jobs pass:

```bash
sbatch \
  --export=ALL,CONFIG_SRC=configs/dg_dales_ocons10.yaml,METHOD_TAG=ocons \
  scripts/slurm_dg_method_train.sh

sbatch \
  --export=ALL,CONFIG_SRC=configs/dg_dales_bev_als_h4_ml.yaml,METHOD_TAG=bev_als \
  scripts/slurm_dg_method_train.sh
```

The full-job preflight uses `--full`, so diagnostic limits cannot silently leak
into a scientific run. Both jobs use seed 1337 and two-GPU DDP.

## 9. Evaluation sequence

First run a one-tile cross-domain integration check for each completed run:

```bash
sbatch --export=ALL,CASE=dales_to_eclair,LIMIT_TILES=1,DALES_RUN_DIR=RUN_DIR,DALES_CKPT=RUN_DIR/checkpoints/best.pt,DALES_CFG=RUN_DIR/config_resolved.json,EVAL_TAG=METHOD_smoke \
  scripts/slurm_cross_eval_common.sh
```

Then run the frozen 120/120 canonical evaluation by omitting `LIMIT_TILES` and
keeping `WINDOW_M=120,STRIDE_M=120`. Compare DALES→ECLAIR common-space mIoU,
wires, poles, buildings, fences and vehicles against M0 and M1-25, and compare
DALES→DALES source retention. Do not select a winner from auxiliary training
metrics; they establish application correctness, not cross-domain quality.

## 10. Files added or changed

- `train.py`: method validation, O-CONS dual view, BEV-ALS loss, DDP reductions,
  debug limits, checkpoint contracts, method diagnostics and gradient checks.
- `src/ocons.py`: masking, protection, exact coordinate alignment and KL.
- `src/bev_als_config.py`, `src/bev_als_targets.py`, `src/bev_als_head.py`:
  ALS frame, targets, projection, decoder and loss.
- `src/model.py`: training-only BEV-ALS wrapper on native `block8` features.
- `src/data_dales.py`, `src/data_eclair.py`: target/frame creation and collation.
- `src/method_config.py`: isolation contract.
- `src/config_loader.py`: recursive `extends` with deep merge and cycle checks.
- `configs/*dg_dales*`: frozen, full and diagnostic configs.
- `scripts/preflight_dg_method.py`, `scripts/smoke_test_dg_methods.py` and the
  two Slurm entry points.
