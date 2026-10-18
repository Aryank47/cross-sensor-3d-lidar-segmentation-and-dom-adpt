# ALS-safe Mix3D M1 — manual implementation and execution guide

This guide is written against the flattened code snapshot supplied on 12 July 2026.
The accompanying `Mix3D_ALS_manual.patch` contains the complete code for:

- new `src/mix3d.py`;
- changes to `src/data_eclair.py`;
- changes to `src/data_dales.py`;
- changes to `src/metrics.py`;
- changes to `train.py`;
- new `scripts/smoke_test_mix3d.py`.

No evaluation source file, model source file, loss source file, mapping file, cache
precomputation script, or new SLURM file is required.

## 1. Scientific scope

M1 is source-only context randomisation:

- ECLAIR source training mixes ECLAIR host and ECLAIR donor tiles only.
- DALES source training mixes DALES host and DALES donor crops only.
- Host and donor must come from different source files.
- Features and source-native train labels move with their points.
- Composition happens after independent ordinary source augmentation and before
  final feature pooling and voxelisation.
- The implementation performs crop replacement, not full-scene union. Report it
  as **ALS-safe Mix3D-style crop replacement**, not literal original Mix3D.
- Validation, source test and cross-domain evaluation never mix samples.

The original Mix3D implementation concatenates two independently augmented scenes
and voxelises the union afterward. It uses hard point labels without label
interpolation (Nekrasov et al., 3DV 2021). Crop replacement is the ALS-specific
adaptation used here to avoid complete-scene density doubling.

## 2. Back up the files

From `eclair_model_train`:

```bash
cp train.py train.py.before_mix3d
cp src/data_eclair.py src/data_eclair.py.before_mix3d
cp src/data_dales.py src/data_dales.py.before_mix3d
```

Do not overwrite or edit `src/eval_cross_common.py`.

## 3. Apply the code patch

Place `Mix3D_ALS_manual.patch` in the `eclair_model_train` repository root.

First check that it matches the current files:

```bash
patch --dry-run -p1 < Mix3D_ALS_manual.patch
```

Expected files:

```text
checking file src/data_eclair.py
checking file src/data_dales.py
checking file src/metrics.py
checking file train.py
checking file src/mix3d.py
checking file scripts/smoke_test_mix3d.py
```

If the dry run is clean, apply it:

```bash
patch -p1 < Mix3D_ALS_manual.patch
```

The patch was itself applied to a reconstruction of the supplied flattened
snapshot and passed Python syntax compilation.

If the actual repository has later unrelated edits and a hunk does not apply,
do not force it with a reject/fuzz option. Apply that hunk manually at the shown
function and compare it with the patch.

## 4. What the code changes do

### `src/mix3d.py`

Defines:

- validated `ALSMix3DConfig`;
- stable process-independent seed derivation;
- different-source donor selection;
- host-region and donor-region selection;
- one-voxel physical guard band;
- local fifth-percentile height alignment;
- point-aligned attribute and hard-label concatenation;
- optional cross-provenance voxel-collision verification;
- explicit diagnostic and skip codes;
- raw point-count histograms for removed host points, inserted donor points and
  final mixed output, separately for every native training class;
- donor-class by retained-host-context co-occurrence matrices.

Skip codes are:

```text
0 applied/no skip
1 probability gate
2 no different donor
3 invalid/insufficient region
4 cross-provenance voxel collision
5 DALES composed output exceeded hard voxel budget
```

### `src/data_eclair.py`

- keeps the ordinary 29306 host seed and augmentation stream;
- independently loads and augments one different ECLAIR donor;
- measures partial ECLAIR context size before rotation;
- composes before `build_features()` and `voxelize_from_q()`;
- uses the unchanged ECLAIR feature, pooling and label contracts;
- carries Mix3D metadata through sparse collation;
- uses worker-visible epoch state for persistent workers.

For the canonical ECLAIR representation:

```text
physical voxel edge = 0.05 × 10 = 0.5 m
one-voxel guard band = 0.5 m
100 m + 100 m nominal replacement side = 50 m
```

Partial ECLAIR tiles derive a smaller side from their observed pre-augmentation
minimum XY span. If a valid region containing at least the configured number of
points cannot be found, the unmodified host is returned and the skip is logged.

### `src/data_dales.py`

- leaves the original M0 crop path unchanged when Mix3D is disabled;
- for M1, reproduces the canonical 29364 crop-centre, retry, shrink, soft/hard
  voxel-budget and rare-preserving fallback logic;
- returns the accepted post-augmentation crop before final voxel aggregation;
- samples the donor crop with the same frozen DALES crop policy from a different
  500 m source file;
- replaces a region at raw-point level and then performs final voxelisation;
- retries region composition when the final mixed output exceeds the canonical
  700,000-voxel hard limit, then falls back to the unchanged host if necessary;
- preserves host crop diagnostics and adds Mix3D diagnostics.

The 500 m DALES files are only raw source containers. They are never combined
or sent to the model. Both host and donor are independently reduced to accepted
crop contexts first.

For canonical 29364:

```text
nominal crop side = 120 m
physical voxel edge = 0.02 × 10 = 0.2 m
one-voxel guard band = 0.2 m
120 m + 120 m replacement side = 60 m
```

If either crop shrinks, the replacement side is based on the smaller accepted
crop size.

### `train.py`

- parses `data.mix3d` once;
- enables it only for the training dataset;
- explicitly constructs ordinary non-mixed validation/test datasets;
- rejects M1 with BEV enabled;
- supports disabling the very large run-scoped ECLAIR cache;
- avoids attempting to remove a disabled/nonexistent run cache;
- aggregates Mix3D diagnostics across all DDP ranks.
- reports source-validation precision and recall for every class, in addition to
  IoU and F1;
- prunes optional epoch checkpoints immediately rather than at the end of the run;
- records checkpoint, run-directory and run-cache sizes every epoch;
- stops cleanly if a configured per-run storage ceiling is exceeded.

The model, loss, optimiser, scheduler, gradient accumulation, source validation,
checkpoint selection and pointwise evaluation functions are unchanged.

## 5. Static and core smoke checks

After activating `e0_env` in the same way as the existing SLURM files:

```bash
python -m py_compile \
  train.py \
  src/mix3d.py \
  src/data_eclair.py \
  src/data_dales.py \
  src/metrics.py \
  scripts/smoke_test_mix3d.py

python scripts/smoke_test_mix3d.py
```

The second command should report `Mix3D core smoke test passed` and show:

- `applied=1`;
- a 50 m synthetic replacement side;
- zero cross-provenance voxels;
- aligned geometry, labels and return fields;
- deterministic repeated composition.

## 6. Create the two M1 configurations from exact canonical runs

Do not copy the current static `e0_eclair_dropI.yaml`, because that file contains
the later 0.04 m sensitivity setting. Copy the actual canonical job YAMLs:

```bash
cp /scratch/m23csa510/e0_results/e0_eclair_repro_29306/e0_eclair_29306.yaml \
  configs/m1_eclair_mix3d.yaml

cp /scratch/m23csa510/e0_results/e0_dales_train_29364/e0_dales_29364.yaml \
  configs/m1_dales_mix3d.yaml
```

### ECLAIR M1 additions

In `configs/m1_eclair_mix3d.yaml`, change only `run.name`, allow the SLURM to
overwrite `run.out_dir`, and add the following keys inside the existing `data:`
mapping:

```yaml
run:
  name: m1_eclair_mix3d
  keep_epoch_snapshots: false
  keep_last_k_epoch_snapshots: 0
  best_checkpoint_model_only: true
  epoch_snapshots_model_only: true
  max_run_dir_gb: 20.0

data:
  # Operational only. The persistent raw ECLAIR cache remains enabled.
  run_cache_enabled: false
  run_cache_precompute_returns_onehot: false

  mix3d:
    enabled: true
    policy: crop_replace
    probability: 1.0
    replacement_area_fraction: 0.25
    guard_band_voxels: 1
    donor_file_strategy: uniform_different_source_file
    max_donor_attempts: 8
    max_region_attempts: 8
    min_region_points: 2000
    min_replacement_side_m: 5.0
    height_alignment: q05_local
    verify_cross_provenance_collisions: true
    diagnostics: true
```

Do not create a second `run:` or `data:` mapping. Merge the keys into the
existing mappings.

Verify that the resulting ECLAIR M1 file still contains:

```yaml
epochs: 100
data:
  batch_size: 2
  grad_accum_steps: 2
  patch:
    coord_norm_factor: 10.0
    voxel_size: 0.05
  voxelization:
    feat_pool: mean_all
    label_pool: majority
  features:
    use_intensity: false
    use_return_number: true
    use_number_of_returns: true
loss:
  alpha: null
  lovasz:
    enabled: true
    weight: 0.5
eval:
  mode: point
model:
  aux_heads:
    bev:
      enabled: false
```

### DALES M1 additions

In `configs/m1_dales_mix3d.yaml`, change `run.name` and add only the Mix3D block
inside the existing `data:` mapping:

```yaml
run:
  name: m1_dales_mix3d
  keep_epoch_snapshots: false
  keep_last_k_epoch_snapshots: 0
  best_checkpoint_model_only: true
  epoch_snapshots_model_only: true
  max_run_dir_gb: 20.0

data:
  mix3d:
    enabled: true
    policy: crop_replace
    probability: 1.0
    replacement_area_fraction: 0.25
    guard_band_voxels: 1
    donor_file_strategy: uniform_different_source_file
    max_donor_attempts: 8
    max_region_attempts: 8
    min_region_points: 2000
    min_replacement_side_m: 5.0
    height_alignment: q05_local
    verify_cross_provenance_collisions: true
    diagnostics: true
```

Verify that canonical DALES settings remain:

```yaml
epochs: 200
data:
  batch_size: 1
  grad_accum_steps: 3
  cache_kind: raw
  require_cache: true
  patch:
    coord_norm_factor: 10.0
    voxel_size: 0.02
  sampling:
    mode: crops
    crop_size_xy_m: 120.0
    crops_per_tile_per_epoch: 12
    crops_per_item: 1
    rare_center_prob: 0.6
    max_voxels_per_crop_soft: 650000
    max_voxels_per_crop_hard: 700000
  voxelization:
    feat_pool: mean_all
    label_pool: majority
loss:
  alpha: [0.53, 0.53, 1.07, 1.6, 0.53, 1.6, 1.07, 1.07]
  lovasz:
    enabled: true
    weight: 0.7
eval:
  mode: point
model:
  aux_heads:
    bev:
      enabled: false
```

## 7. Validate the resolved configuration before submitting

```bash
python - <<'PY'
from src.config_loader import load_yaml

for path in ("configs/m1_eclair_mix3d.yaml", "configs/m1_dales_mix3d.yaml"):
    cfg = load_yaml(path)
    d = cfg["data"]
    assert d["mix3d"]["enabled"] is True
    assert cfg["model"]["aux_heads"]["bev"]["enabled"] is False
    assert cfg["eval"]["mode"] == "point"
    if d["dataset"] == "eclair":
        assert d["patch"]["voxel_size"] == 0.05
        assert cfg["loss"]["alpha"] is None
        assert d["run_cache_enabled"] is False
    else:
        assert d["patch"]["voxel_size"] == 0.02
        assert d["sampling"]["crop_size_xy_m"] == 120.0
        assert d["sampling"]["crops_per_item"] == 1
        assert d["sampling"]["max_voxels_per_crop_hard"] == 700000
    print(path, "OK")
PY
```

## 8. One-real-batch integration checks

Run these on a compute node with the same modules and environment as the
training SLURMs. They do not train a model.

### ECLAIR

```bash
python - <<'PY'
from src.config_loader import load_yaml
from src.dist import DistEnv
from train import build_dataloaders

cfg = load_yaml("configs/m1_eclair_mix3d.yaml")
cfg["data"]["num_workers"] = 0
tr, va, te = build_dataloaders(cfg, DistEnv(enabled=False), eclair_run_cache_dir=None)
b = next(iter(tr))
print("train", b["coords"].shape, b["feats"].shape, b["labels"].shape)
print("mix applied", b["meta_mix_applied"], "skip", b["meta_mix_skip_code"])
assert "meta_mix_applied" in b
assert int(b["meta_mix_cross_provenance_voxels"].sum()) == 0
vb = next(iter(va))
assert "meta_mix_applied" not in vb
print("ECLAIR real-batch M1 check passed")
PY
```

### DALES

```bash
python - <<'PY'
from src.config_loader import load_yaml
from src.dist import DistEnv
from train import build_dataloaders

cfg = load_yaml("configs/m1_dales_mix3d.yaml")
cfg["data"]["num_workers"] = 0
tr, va, te = build_dataloaders(cfg, DistEnv(enabled=False))
b = next(iter(tr))
print("train", b["coords"].shape, b["feats"].shape, b["labels"].shape)
print("crop", b["meta_crop_size_xy_m"], "voxels", b["meta_n_vox"])
print("mix applied", b["meta_mix_applied"], "skip", b["meta_mix_skip_code"])
assert "meta_mix_applied" in b
assert int(b["meta_mix_cross_provenance_voxels"].sum()) == 0
assert int(b["meta_n_vox"].max()) <= 700000
vb = next(iter(va))
assert "meta_mix_applied" not in vb
print("DALES real-batch M1 check passed")
PY
```

A skip on a particular host is not automatically an error. Inspect the skip
code. A persistent low `mix_rate` is an error requiring investigation before a
full run.

## 9. Reuse the existing training SLURMs

No new SLURM file is necessary. Make the configuration line in each existing
training launcher overridable.

### `eclair_train.slurm`

Replace:

```bash
RUN_DIR="${SCRATCH_OUTROOT}/e0_eclair_repro_${SLURM_JOB_ID}"
CONFIG_SRC="configs/e0_eclair_dropI.yaml"
CONFIG_RUN="${RUN_DIR}/e0_eclair_${SLURM_JOB_ID}.yaml"
```

with:

```bash
RUN_PREFIX="${RUN_PREFIX:-e0_eclair_repro}"
RUN_DIR="${SCRATCH_OUTROOT}/${RUN_PREFIX}_${SLURM_JOB_ID}"
CONFIG_SRC="${CONFIG_SRC:-configs/e0_eclair_dropI.yaml}"
CONFIG_STEM="$(basename "${CONFIG_SRC%.*}")"
CONFIG_RUN="${RUN_DIR}/${CONFIG_STEM}_${SLURM_JOB_ID}.yaml"
```

### `dales_train.slurm`

Replace:

```bash
RUN_DIR="${SCRATCH_OUTROOT}/e0_dales_train_${SLURM_JOB_ID}"
CONFIG_SRC="configs/e0_dales_dropI.yaml"
CONFIG_RUN="${RUN_DIR}/e0_dales_${SLURM_JOB_ID}.yaml"
```

with:

```bash
RUN_PREFIX="${RUN_PREFIX:-e0_dales_train}"
RUN_DIR="${SCRATCH_OUTROOT}/${RUN_PREFIX}_${SLURM_JOB_ID}"
CONFIG_SRC="${CONFIG_SRC:-configs/e0_dales_dropI.yaml}"
CONFIG_STEM="$(basename "${CONFIG_SRC%.*}")"
CONFIG_RUN="${RUN_DIR}/${CONFIG_STEM}_${SLURM_JOB_ID}.yaml"
```

All module loading, CUDA paths, `e0_env`, DDP and GPU settings remain unchanged.

## 10. Submit training

```bash
sbatch --export=ALL,CONFIG_SRC=configs/m1_eclair_mix3d.yaml,RUN_PREFIX=m1_eclair_mix3d \
  eclair_train.slurm

sbatch --export=ALL,CONFIG_SRC=configs/m1_dales_mix3d.yaml,RUN_PREFIX=m1_dales_mix3d \
  dales_train.slurm
```

During the first epoch, verify:

- no target dataset path is opened;
- `meta_mix_applied` is present in training batches;
- validation runs without Mix3D metadata;
- `mix_rate` is reasonably close to 1, allowing documented partial/invalid-region
  and DALES budget skips;
- `mix_cross_provenance_voxels` is exactly zero;
- DALES `train_vox_mean` remains under the hard budget;
- no `_run_cache` directory grows for ECLAIR M1;
- `best.pt` is selected using source validation mIoU.

The code deliberately does not cache mixed samples.

## 11. Diagnostic evidence captured by each run

The revised implementation deliberately separates three questions.

### 11.1 Was the implementation valid?

`metrics.csv` records:

```text
mix_rate
mix_skip_rate
mix_probability_skip_rate
mix_no_donor_skip_rate
mix_invalid_region_skip_rate
mix_collision_skip_rate
mix_output_budget_skip_rate
mix_cross_provenance_voxels
mix_output_to_host_points_ratio_mean/std
mix_replacement_side_m_mean/std
mix_height_shift_m_mean/std
mix_abs_height_shift_m_mean
```

An M1 result is not scientifically interpretable if mixing rarely occurred,
cross-provenance collisions are nonzero, output-to-host point ratios drift
substantially, or extreme height shifts indicate implausible geometry.

### 11.2 What training distribution was actually generated?

`metrics.csv` also records donor-selection coverage:

```text
mix_donor_unique_count
mix_donor_selection_entropy
mix_donor_max_share
```

`mix_class_diagnostics.csv` contains one row per epoch and native training class:

```text
host_removed_points
donor_inserted_points
mixed_output_points
donor_presence_samples
```

This reveals whether random file selection still produced a strongly skewed
class intervention. For example, it can show that vehicle or pole points were
almost never inserted even though the global mix rate was high.

`mix_context_diagnostics.json` stores, per epoch:

- the donor-file usage histogram;
- a `donor class × retained host-context class` presence matrix.

This matrix does not prove that the model used context, but it verifies whether
the intervention actually exposed classes to novel source-scene co-occurrences.
If the matrix remains almost identical or highly concentrated, the run did not
strongly test the context-prior hypothesis.

All these artifacts are aggregate CSV/JSON files. They are normally only a few
megabytes and do not store point clouds or per-sample tensors.

### 11.3 What happened to class behaviour?

Source validation in `metrics.csv` now includes per-class:

```text
val_iou_<class>
val_f1_<class>
val_precision_<class>
val_recall_<class>
```

The existing post-hoc common-space evaluator already writes target-domain
per-class precision and recall. Therefore utility-class degradation can be
separated into:

- recall loss: more false negatives / missed utility objects;
- precision loss: more false positives / over-prediction;
- both: broad class confusion or representation damage.

Do not run target validation every epoch or use target results for checkpoint
selection. That would contaminate the source-only DG protocol. Target precision,
recall and IoU remain post-hoc measurements from the locked source-selected
checkpoint.

## 12. Storage and checkpoint retention

The supplied `train.py` does not normally accumulate one full checkpoint every
epoch. It atomically overwrites `last.pt`, overwrites `best.pt` when validation
improves, and creates `epoch_*.pt` only when `run.keep_epoch_snapshots=true`.
The old `keep_last_k_epoch_snapshots` key was present in configurations but was
not enforced by the original code.

The earlier approximately 96 GB incident was consistent with `_run_cache`, not
with `last.pt`/`best.pt`. ECLAIR M1 explicitly disables that run cache.

The revised code additionally:

1. deletes interrupted `*.tmp` checkpoint files at startup;
2. deletes stale `epoch_*.pt` files at startup when snapshots are disabled;
3. enforces `keep_last_k_epoch_snapshots` immediately after every snapshot save;
4. permits model-only `best.pt` and optional model-only epoch snapshots;
5. logs `checkpoint_dir_gb`, `run_dir_gb`, and `run_cache_gb` every epoch;
6. stops the whole DDP job if `run.max_run_dir_gb` is exceeded.

For initial M1 runs use:

```yaml
run:
  keep_epoch_snapshots: false
  keep_last_k_epoch_snapshots: 0
  best_checkpoint_model_only: true
  epoch_snapshots_model_only: true
  max_run_dir_gb: 20.0
```

This retains only:

- `last.pt`, with optimiser/scheduler/scaler state for resumption;
- `best.pt`, containing model state and configuration for inference/evaluation.

Before deleting anything from an existing run, identify the actual consumer:

```bash
RUN=/scratch/m23csa510/e0_results/<run_name>

du -h --max-depth=2 "${RUN}" 2>/dev/null | sort -hr | head -50
find "${RUN}/checkpoints" -maxdepth 1 -type f \
  -printf '%s %p\n' 2>/dev/null | sort -nr | head -50
```

Your training outputs are written to `/scratch`, not `/csehome`. The reported
scratch quota is 200 GB with approximately 105 GB used, so quota headroom is
about 95 GB even though filesystem-wide `df` reports much more free space.
`/csehome` is globally full but your reported home usage is about 22 GB against
a 70 GB quota. Continue keeping training outputs, caches and checkpoints under
`/scratch`; do not allow a config with `run.out_dir: ./runs/...` to bypass the
SLURM override.

## 13. Evaluation impact

The new code does **not** change evaluation mathematics or the evaluation data
path.

Reasons:

1. `train.py` passes Mix3D only to `train_ds` and explicitly disables it for
   `val_ds` and `test_ds`.
2. `src.eval_cross_common` does not use the M1 training branch and does not pass
   a Mix3D configuration to either dataset constructor.
3. Both dataset constructors default to `ALSMix3DConfig(enabled=False)`.
4. Model architecture, output class count, checkpoint state keys, feature stack,
   voxel size and train-ID spaces are unchanged.
5. Common-space mapping files and inverse point mapping are unchanged.

The M1 `config_resolved.json` contains an extra `data.mix3d` block. This is safe:
the cross evaluator reads the source model/representation settings it needs and
the dataset constructors remain non-mixed unless training explicitly passes an
enabled Mix3D object.

Do not remove the Mix3D block from the source config used for evaluation; it is
part of the experiment record.

## 14. Run source and cross-domain evaluation

The supplied cross-evaluation SLURM defaults to the canonical M0 runs. Override
only the source run directory.

Assume:

```bash
E_M1_RUN=/scratch/m23csa510/e0_results/m1_eclair_mix3d_<ECLAIR_JOB_ID>
D_M1_RUN=/scratch/m23csa510/e0_results/m1_dales_mix3d_<DALES_JOB_ID>
```

### ECLAIR M1 source checkpoint

```bash
sbatch --export=ALL,CASE=eclair_to_eclair,ECLAIR_RUN_DIR="${E_M1_RUN}" \
  scripts/slurm_cross_eval_common.sh

sbatch --export=ALL,CASE=eclair_to_dales,ECLAIR_RUN_DIR="${E_M1_RUN}" \
  scripts/slurm_cross_eval_common.sh
```

For `eclair_to_dales`, the target DALES configuration remains the canonical
29364 configuration through the script default.

### DALES M1 source checkpoint

```bash
sbatch --export=ALL,CASE=dales_to_dales,DALES_RUN_DIR="${D_M1_RUN}" \
  scripts/slurm_cross_eval_common.sh

sbatch --export=ALL,CASE=dales_to_eclair,DALES_RUN_DIR="${D_M1_RUN}" \
  scripts/slurm_cross_eval_common.sh
```

For `dales_to_eclair`, the target ECLAIR configuration remains canonical 29306
through the script default.

Keep the existing explicit mappings:

```text
configs/train_id_to_common_id.yaml
configs/train_id_to_common_dales.yaml
configs/mapping_eclair_to_common.yaml
configs/mapping_dales_to_common.yaml
```

Do not infer or modify mappings for M1.

## 15. Required comparison

Compare M1 against the matched canonical source recipe, not historical
bug-affected runs:

- ECLAIR M0: run 29306;
- DALES M0: run 29364.

Report:

- source-native test mIoU and macro-F1;
- common-space target mIoU and macro-F1;
- all seven common per-class IoUs;
- utility mean IoU;
- `mix_rate` and all skip/point/side/collision diagnostics;
- mean and standard deviation over matched seeds once the first seed is verified.

A gain in cross-domain mIoU with stable source performance supports the
context-prior hypothesis. A source-only gain without cross-domain gain is source
regularisation, not evidence that context randomisation reduced the domain gap.
A regression accompanied by low mix rate, nonzero collisions, or frequent
output-budget skips is not an interpretable negative result until the engineering
issue is resolved.

## 16. Interpreting M1 outcomes

Logging can establish whether M1 was implemented and exercised correctly; it
cannot by itself prove that the baseline model relied on context. Use the
following interpretation hierarchy.

| Observation | Most defensible interpretation |
| --- | --- |
| Low mix rate, high invalid-region/budget skips, nonzero collisions, or large density/height anomalies | Engineering/intervention failure. The hypothesis was not tested cleanly. |
| Valid mixing, but utility classes were rarely present in donor regions | Random source-file mixing gave inadequate utility-class exposure. This is an M1 coverage limitation, not evidence against context priors. |
| Valid, diverse mixing; source precision/recall and source mIoU degrade | Context randomisation or synthetic seams damaged source learning. Inspect class exposure, height shifts and output ratios before concluding Mix3D is unsuitable. |
| Source stable, target precision worsens while recall is stable | M1 increased target false positives; context disruption may have weakened useful negative-context constraints. |
| Source stable, target recall worsens while precision is stable | M1 increased missed target utilities; local morphology/support shift is likely more important than context randomisation. |
| Source stable, target metrics unchanged across matched seeds | Evidence does not support context randomisation as a major solution for this transfer direction. It does not prove that context priors are absent. |
| Source stable, target improves consistently across matched seeds | Supports the claim that reducing source-context dependence improves transfer, subject to the logged intervention-validity checks. |

If a correct and well-exercised M1 run fails, the scientifically acceptable
conclusion is limited:

> Under the tested ALS-safe crop-replacement policy and source recipe, random
> source-context recombination did not reduce the observed cross-survey gap.

Do not conclude that ECLAIR/DALES models have no context priors. That stronger
claim would require a separate causal diagnostic or controlled context-ablation
experiment. Run at least three matched seeds before interpreting small changes.

## Reference

Nekrasov, A., Schult, J., Litany, O., Leibe, B., & Engelmann, F. (2021).
*Mix3D: Out-of-Context Data Augmentation for 3D Scenes*. International Conference
on 3D Vision (3DV). arXiv:2110.02210.
