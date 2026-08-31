# O-CONS and BEV-ALS two-epoch DDP pilot

This pilot processes the complete DALES training sampler for two epochs using
two GPUs and four DataLoader workers per rank. Validation runs at epoch 2; the
final test set is intentionally skipped.

## Local verification

```bash
cd eclair_model_train
python -m py_compile train.py src/*.py scripts/*.py
python scripts/smoke_test_dg_methods.py

export DALES_TRAIN_ROOT=/scratch/m23csa510/dales/dales/train
export DALES_TEST_ROOT=/scratch/m23csa510/dales/dales/test
export DALES_CACHE_ROOT_DROP_I=/scratch/m23csa510/dales/dales_cache

python scripts/preflight_dg_method.py \
  --config configs/dg_dales_ocons10_pilot.yaml \
  --expect-method ocons \
  --build-model

python scripts/preflight_dg_method.py \
  --config configs/dg_dales_bev_als_h4_ml_pilot.yaml \
  --expect-method bev_als \
  --build-model
```

## Submit the pilots

```bash
sbatch \
  --export=ALL,CONFIG_SRC=configs/dg_dales_ocons10_pilot.yaml,METHOD_TAG=ocons \
  scripts/slurm_dg_method_pilot.sh

sbatch \
  --export=ALL,CONFIG_SRC=configs/dg_dales_bev_als_h4_ml_pilot.yaml,METHOD_TAG=bev_als \
  scripts/slurm_dg_method_pilot.sh
```

Each successful run writes `pilot_acceptance.json`. The important decisions are:

- `correctness_pass`: DDP, gradients, method diagnostics, validation, and both
  `last.pt` and `best.pt` passed.
- `single_job_time_pass`: the buffered 200-epoch estimate fits within 36 hours.
- `ready_for_resumable_full_run`: correctness and storage passed even if the
  36-hour limit requires more than one Slurm allocation.

The validation confusion matrices use rows as ground truth and columns as
predictions. `val_confusion_latest.json` and `val_confusion_best.json` are
overwritten in place, so they do not accumulate with epoch count.

## Full training

Submit a new run only after the corresponding pilot passes correctness:

```bash
sbatch \
  --export=ALL,CONFIG_SRC=configs/dg_dales_ocons10.yaml,METHOD_TAG=ocons \
  scripts/slurm_dg_method_train.sh

sbatch \
  --export=ALL,CONFIG_SRC=configs/dg_dales_bev_als_h4_ml.yaml,METHOD_TAG=bev_als \
  scripts/slurm_dg_method_train.sh
```

The full configurations overwrite one resumable `last.pt` every epoch and keep
one model-only `best.pt`; they do not accumulate epoch snapshots.

If a job reaches the 36-hour limit, resume in the same run directory:

```bash
sbatch \
  --export=ALL,CONFIG_SRC=configs/dg_dales_ocons10.yaml,METHOD_TAG=ocons,RESUME_RUN_DIR=/scratch/m23csa510/e0_results/dg_ocons_JOB_ID \
  scripts/slurm_dg_method_train.sh
```

Use `METHOD_TAG=bev_als` and the BEV run directory for a BEV-ALS resume.
