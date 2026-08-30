# ECLAIR E0 Backbone (paper-like) — MinkowskiEngine

This folder contains a clean, modular training pipeline to reproduce the **ECLAIR** backbone experiment:

- backbone: **Res16UNet14C**
- loss: **Focal**
- features: **intensity + return_number(one-hot) + number_of_returns(one-hot)**
- coordinate normalization + voxel quantization

## 1) Environment variables

Set these before running:

```bash
export ECLAIR_ROOT=/path/to/ECLAIR          # contains labels.json and pointclouds/
export ECLAIR_CACHE_ROOT=/path/to/cache     # optional but recommended
```

## 2) Optional: build cache (faster I/O)

```bash
python scripts/precompute_eclair_cache.py --eclair_root "$ECLAIR_ROOT" --cache_root "$ECLAIR_CACHE_ROOT"
```

## 3) Train

```bash
python train.py --config configs/e0_eclair.yaml
```

Outputs will be written to `run.out_dir` (see config), including:

- `metrics.csv` (per-epoch loss, mIoU, macro-F1, per-class IoU/F1)
- `checkpoints/{best.pt,last.pt,epoch_XXX.pt}`
- `test_metrics.json`

## Notes on exact reproduction

The ECLAIR paper specifies model/voxel/feature choices, but does not provide
all numeric hyperparameters (e.g., augmentation ranges and StepLR gamma) in the text excerpt.
Those are exposed in `configs/e0_eclair.yaml` for easy tuning.
