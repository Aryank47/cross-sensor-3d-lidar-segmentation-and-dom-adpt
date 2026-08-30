# DALES Experiments Report

**Project:** Cross-sensor 3D LiDAR Segmentation & Domain Adaptation (ECLAIR ↔ DALES)  
**Dataset:** DALES (LAS 1.2, point_format=1)  
**Model:** MinkowskiEngine Res16UNet14C (3D)  
**Features:** returns-only (Return Number + Number of Returns one-hot, k=5 → `in_channels=10`)  
**Intensity:** disabled (`use_intensity=false`) due to DALES intensity being effectively dead/zero in our EDA  
**Eval:** point-wise evaluation using OOM-safe sliding window inference

---

## 1) Motivation and Setup

### Why “Drop Intensity”?

Prior EDA indicated DALES **intensity values are effectively all zero**, so intensity is not informative and can create misleading feature-mismatch relative to ECLAIR. Therefore, all DALES experiments below use:

- `features.use_intensity: false`
- `returns_onehot_k: 5` (Return Number + Number of Returns → 10 channels)

### Common training setup (shared unless stated)

- **Patch / voxelization**
  - `coord_norm_factor=10.0`
  - `voxel_size=0.02` (normalized space)
- **Augmentation**
  - random flip XY, random rotation 0–360°, scale 0.95–1.05, jitter std 0.01
- **Cache**
  - `cache_kind: raw` (raw point cache), `require_cache: true`, `write_cache: false`
- **Sampling**
  - `sampling.mode: crops` (crop-based training for OOM safety + better tile coverage)
- **Eval**
  - `eval.mode: point`
  - window inference: `size_xy_m=50`, `stride_xy_m=25`, mean-logits aggregation

> Note: DALES is highly imbalanced; utility classes (poles, power_lines) are extremely rare (see Section 5).

---

## 2) Experiment Summary Table

| Exp ID    | Run Dir                             | Voxelization (`feat_pool`, `label_pool`) | Crop Size | Loss                        |       mIoU |   macro-F1 | Notes                                                                    |
| --------- | ----------------------------------- | ---------------------------------------- | --------: | --------------------------- | ---------: | ---------: | ------------------------------------------------------------------------ |
| **E1**    | `/scratch/.../e0_dales_train_28656` | `sample_first`, `first`                  |       20m | Focal (alpha=null)          | **0.5637** |     0.6642 | baseline pooling + rare centering (poles, power_lines)                   |
| **E2**    | `/scratch/.../e0_dales_train_28722` | `mean_all`, `majority`                   |       20m | Focal (alpha=null)          | **0.6093** |     0.7141 | improved voxel feature/label pooling + slightly stronger rare centering  |
| **E3** ✅ | `/scratch/.../e0_dales_train_28797` | `mean_all`, `majority`                   |       40m | Focal + **per-class alpha** | **0.6888** | **0.7809** | + training speed improvements + weighted rare-centering + larger context |

**Best so far:** **E3** (mIoU **0.6888**, macro-F1 **0.7809**)

---

## 3) Detailed Results

### E1 — Baseline pooling (`sample_first` / `first`) with 20×20 crops

**Run:** `/scratch/m23csa510/e0_results/e0_dales_train_28656`  
**Key config:**

- voxelization: `feat_pool=sample_first`, `label_pool=first`
- sampling: crops, `crop_size_xy_m=20`, `crops_per_tile_per_epoch=8`
- rare centering: `rare_center_prob=0.5`, `rare_center_class_ids=[5,6]` (poles, power_lines)
- loss: focal, `alpha=null`, `gamma=2.0`

**Test metrics:**

- **mIoU:** 0.5637
- **macro-F1:** 0.6642
- per-class IoU:
  - ground 0.9185
  - vegetation 0.8729
  - cars 0.5640
  - trucks 0.00028 _(very low)_
  - buildings 0.7671
  - poles 0.3252
  - power_lines 0.7271
  - fences 0.3342

**Observation:**

- Strong on dominant classes (ground/veg/buildings) but weak utility assets (poles/fences) and extremely poor trucks.

---

### E2 — Improved voxel pooling (`mean_all` / `majority`) with 20×20 crops

**Run:** `/scratch/m23csa510/e0_results/e0_dales_train_28722`  
**Key config changes vs E1:**

- voxelization: `feat_pool=mean_all`, `label_pool=majority`
- sampling: `crops_per_tile_per_epoch=12`
- rare centering: `rare_center_prob=0.6`, `rare_center_class_ids=[3,5,6]`

**Test metrics:**

- **mIoU:** 0.6093
- **macro-F1:** 0.7141
- per-class IoU:
  - ground 0.9279
  - vegetation 0.8919
  - cars 0.6568
  - trucks 0.1288
  - buildings 0.8116
  - poles 0.3126
  - power_lines 0.7733
  - fences 0.3715

**Observation:**

- Switching to `mean_all` / `majority` improved overall mIoU and stabilized trucks from ~0 → ~0.13 IoU.
- Poles still lag significantly (0.31 IoU), suggesting rare-class supervision remains a key bottleneck.

---

### E3 ✅ — Larger crops (40×40), per-class alpha, weighted rare centering, and training speed improvements

**Run:** `/scratch/m23csa510/e0_results/e0_dales_train_28797`  
**Key config changes vs E2:**

- crop size: `crop_size_xy_m=40.0`
- training efficiency: crop bundling `crops_per_item=3`, `prefetch_factor=4`, `grad_accum_steps=2`
- weighted rare centering:
  - `rare_center_class_ids=[3,5,6]`
  - `rare_center_class_weights={5:2.0, 6:3.0}`
  - `rare_center_balance_beta=0.5`
- **loss alpha enabled**:
  - `[0.53, 0.53, 1.07, 1.60, 0.53, 1.60, 1.07, 1.07]` (order: ground, vegetation, cars, trucks, buildings, poles, power_lines, fences)

**Test metrics:**

- **mIoU:** **0.6888**
- **macro-F1:** **0.7809**
- per-class IoU:
  - ground 0.9484
  - vegetation 0.9091
  - cars 0.7432
  - trucks 0.1300
  - buildings 0.8836
  - poles 0.5499
  - power_lines 0.8188
  - fences 0.5274

**Observation:**

- This is a large jump: **+0.0795 mIoU** from E2.
- Major gains for **poles** (0.31 → **0.55**) and **fences** (0.37 → **0.53**), indicating:
  1. larger spatial context (40m crops), and
  2. per-class alpha + weighted rare centering  
     jointly improved rare utility assets.

---

## 4) Training Efficiency Improvement (Time Comparison)

Based on Slurm job timing:

- **Old run time:** ~9h 24m (job 28722)
- **New run time:** ~5h 17m (job 28797)

**Net improvement:** ~**44% faster** end-to-end training.

Likely contributors (as implemented in codebase):

- crop bundling (`crops_per_item=3`) reduces repeated tile load/processing per epoch
- moving heavy augmentation away from full-tile operations and focusing on crop-level work
- `prefetch_factor=4` + `persistent_workers` for DataLoader throughput
- reduced `grad_accum_steps` after bundling to maintain effective batch behavior

---

## 5) Dataset Class Imbalance (Tile-level point counts)

Computed from DALES split tiles (train/val/test).  
Order: `[ground, vegetation, cars, trucks, buildings, poles, power_lines, fences]`

### TRAIN

- counts: `[158,606,409, 109,721,698, 2,265,583, 354,711, 51,485,927, 247,189, 728,257, 1,394,389]`
- freq%: `[48.83, 33.78, 0.70, 0.11, 15.85, 0.076, 0.224, 0.429]`

### VAL

- freq%: `[52.66, 30.10, 0.862, 1.069, 14.71, 0.081, 0.194, 0.322]`

### TEST

- freq%: `[50.66, 30.50, 0.787, 0.113, 17.25, 0.068, 0.170, 0.459]`

**Key takeaway:** utility assets are extremely rare:

- poles: ~0.07–0.08%
- power_lines: ~0.17–0.22%
  This motivates rare-aware sampling and loss weighting.

---

## 6) Key Takeaways So Far

1. **Voxel pooling matters:** moving from (`sample_first`, `first`) → (`mean_all`, `majority`) improved mIoU significantly (E1 → E2).
2. **Bigger context helps:** increasing crop size from 20m → 40m improved rare utility classes substantially (E2 → E3).
3. **Rare targeting works when combined with context + loss weighting:** per-class alpha + weighted rare centering improved poles/fences strongly in E3.
4. **Efficiency improvements enabled faster iteration** (~44% reduction in training time), making broader sweeps feasible.

---

## 7) Next Training Ladder Plan (Planned Experiments)

**Goal:** improve DALES performance further in a controlled manner, avoiding confounded comparisons.

### Rung A — Crop-size sweep (short runs)

Keep E3 settings fixed, vary crop size and voxel budgets:

- 40m (baseline winner)
- 60m
- 80m
- 100m

For each: run **25–30 epochs** and compare:

- overall mIoU
- utility assets IoU (poles, power_lines, fences)

### Rung B — Add Lovász (IoU-optimized loss) after selecting best crop size

Enable Lovász with a gentle ramp:

- warmup 10 epochs, ramp 20 epochs
- start weight 0.2–0.3

### Rung C — Evaluation-only upgrades

- Enable rot4 TTA for final reporting (no training change)
- Optionally reduce eval stride for more overlap if boundary artifacts appear

---

### Rung D — `feature_pool` + `label_pool` combinations to try.

- mean_all + first (isolates label pooling effect)

- sample_first + majority (isolates feature pooling effect)

- max_all + majority (tests “sharp” feature pooling)

- random + majority (tests stochastic pooling; may help DG but can add variance)

---

## 8) Artifact Locations (for reproducibility)

- E1: `/scratch/m23csa510/e0_results/e0_dales_train_28656`
- E2: `/scratch/m23csa510/e0_results/e0_dales_train_28722`
- E3: `/scratch/m23csa510/e0_results/e0_dales_train_28797`

Each directory contains:

- `config_resolved.json`
- `metrics.csv`
- `test_metrics.json`
- `checkpoints/{best.pt,last.pt,...}`

---

## 9) Notes / Known Limitations

- DALES intensity channel is disabled due to being non-informative (0-valued) in our EDA.
- Point-wise evaluation uses windowed voxel inference + projection to points. This is OOM-safe but may depend on window stride/overlap.
- Rare classes remain statistically fragile; comparisons should ideally be verified across ≥2 random seeds once the best configuration is identified.

---
