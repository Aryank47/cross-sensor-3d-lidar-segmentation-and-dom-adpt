## A) Overall performance comparison (runs vs Paper)

### A1) ECLAIR — feature-matched comparison (returns-only)

config uses **returns-only** (10-dim: return_number one-hot (5) + number_of_returns one-hot (5)), **no intensity**, **no RGB**, which aligns best with the paper’s `freturn` row (Melekhov et al., arXiv 2024).

| Metric   | ECLAIR | Paper ECLAIR (freturn) | Δ (ours − Paper) |
| -------- | -----: | ---------------------: | ---------------: |
| Macro-F1 | 0.8314 |                 0.8420 |      **−0.0106** |
| mIoU     | 0.7486 |                 0.7663 |      **−0.0177** |

(Melekhov et al., arXiv 2024)

### A2) ECLAIR — best paper model (not feature-matched; for context)

Paper’s best reported architecture entry is **Res16UNet14C** with **macro-F1 0.845 / mIoU 77.29%** (their best setting includes different feature choices than our drop-intensity run). (Melekhov et al., arXiv 2024; Choy et al., CVPR 2019)

| Metric   | Our ECLAIR | Paper ECLAIR (Res16UNet14C best) |           Δ |
| -------- | ---------: | -------------------------------: | ----------: |
| Macro-F1 |     0.8314 |                           0.8450 | **−0.0136** |
| mIoU     |     0.7486 |                           0.7729 | **−0.0243** |

(Melekhov et al., arXiv 2024; Choy et al., CVPR 2019)

### A3) DALES — comparison to KPConv (different backbone; for context)

DALES paper reports **mean IoU 0.811** for KPConv. our Res16UNet14C mean IoU is **0.724**. (Varney et al., arXiv 2020; Thomas et al., ICCV 2019)

| Metric   | our DALES | Paper DALES (KPConv) |           Δ |
| -------- | --------: | -------------------: | ----------: |
| Mean IoU |    0.7240 |               0.8110 | **−0.0870** |
| Macro-F1 |    0.8271 |   N/A (not reported) |           — |

(Varney et al., arXiv 2020; Thomas et al., ICCV 2019)

---

## B) Class-wise comparison (detailed)

### B1) ECLAIR class-wise: our run vs Paper `freturn` (returns-only)

Paper reports “macro per-class F1/IoU” over the **11 important classes (1..11)** (excluding Undefined=0). our 11-class ordering matches **[Unassigned, Ground, Vegetation, Buildings, Noise, Trans wires, Dist wires, Poles, Trans towers, Fence, Vehicles]** after our `native_id−1` remap. (Melekhov et al., arXiv 2024)

> All paper IoU values below are converted from % to decimals for direct comparison.

| Class               | Our F1 | Paper F1 |        ΔF1 | Our IoU | Paper IoU |       ΔIoU |
| ------------------- | -----: | -------: | ---------: | ------: | --------: | ---------: |
| Unassigned          |  0.275 |    0.290 |     −0.015 |   0.159 |     0.172 |     −0.013 |
| Ground              |  0.971 |    0.990 |     −0.019 |   0.943 |     0.983 | **−0.040** |
| Vegetation          |  0.990 |    0.990 |     +0.000 |   0.980 |     0.984 |     −0.004 |
| Buildings           |  0.940 |    0.930 |     +0.010 |   0.886 |     0.868 | **+0.018** |
| Noise               |  0.834 |    0.830 |     +0.004 |   0.715 |     0.710 |     +0.005 |
| Transmission wires  |  0.981 |    0.990 |     −0.009 |   0.963 |     0.989 | **−0.026** |
| Distribution wires  |  0.843 |    0.900 | **−0.057** |   0.729 |     0.823 | **−0.094** |
| Poles (Dist. poles) |  0.750 |    0.690 | **+0.060** |   0.600 |     0.523 | **+0.077** |
| Transmission towers |  0.951 |    0.960 |     −0.009 |   0.906 |     0.916 |     −0.010 |
| Fence               |  0.770 |    0.830 | **−0.060** |   0.626 |     0.707 | **−0.081** |
| Vehicles            |  0.841 |    0.860 |     −0.019 |   0.726 |     0.753 |     −0.027 |

(Melekhov et al., arXiv 2024)

**Biggest deficits (IoU):**

- Distribution wires **−9.4 pts**
- Fence **−8.1 pts**
- Ground **−4.0 pts**
- Transmission wires **−2.6 pts**
  (Melekhov et al., arXiv 2024)

**Biggest gains (IoU):**

- Poles **+7.7 pts**
- Buildings **+1.8 pts**
  (Melekhov et al., arXiv 2024)

---

### B2) DALES class-wise: Our run vs Paper KPConv

Paper DALES table reports IoU for **8 classes** (ground, buildings, cars, trucks, poles, power lines, fences, veg). Our DALES run has 8 classes too (ground, vegetation, cars, trucks, buildings, poles, power_lines, fences); below is aligned by class name. (Varney et al., arXiv 2020; Thomas et al., ICCV 2019)

| Class       | Our IoU | Paper IoU (KPConv) |       ΔIoU | Our F1 |
| ----------- | ------: | -----------------: | ---------: | -----: |
| ground      |   0.837 |              0.971 | **−0.134** |  0.911 |
| vegetation  |   0.745 |              0.941 | **−0.196** |  0.854 |
| buildings   |   0.582 |              0.966 | **−0.384** |  0.736 |
| cars        |   0.880 |              0.853 | **+0.027** |  0.936 |
| trucks      |   0.351 |              0.419 |     −0.068 |  0.520 |
| poles       |   0.706 |              0.750 |     −0.044 |  0.828 |
| power_lines |   0.814 |              0.955 | **−0.141** |  0.898 |
| fences      |   0.876 |              0.635 | **+0.241** |  0.934 |

(Varney et al., arXiv 2020; Thomas et al., ICCV 2019)

**Largest gaps:** Buildings (−38.4 pts), Vegetation (−19.6 pts), Power lines (−14.1 pts), Ground (−13.4 pts). (Varney et al., arXiv 2020)

---

| Run                        | Key changes vs baseline  |       mIoU |      wires |      poles |       fence |     vehicle |
| -------------------------- | ------------------------ | ---------: | ---------: | ---------: | ----------: | ----------: |
| Baseline (46727326c2)      | sample_first, no TTA     |     0.5942 |     0.7696 |     0.2027 |      0.3212 |      0.1621 |
| **Best (660b3f3f21)**      | **mean_all + rot4**      | **0.5998** | **0.7886** | **0.2132** |      0.3250 |      0.1616 |
| **Runner-up (5d1033af15)** | rot4 only (sample_first) | **0.5995** |     0.7872 | **0.2153** | (not shown) | (not shown) |

| Run                        | Key changes vs baseline            |       mIoU |      wires |  poles |  fence | vehicle |
| -------------------------- | ---------------------------------- | ---------: | ---------: | -----: | -----: | ------: |
| Baseline (46727326c2)      | sample_first, no TTA               |     0.0490 |     0.3230 | 0.0154 | 0.0037 |  0.0010 |
| **Best (36f0394584)**      | **coord_norm_factor=8 + mean_all** | **0.0603** | **0.4097** | 0.0086 | 0.0019 |  0.0010 |
| **Runner-up (5849333e41)** | **voxel_size=0.04 + mean_all**     | **0.0603** | **0.4096** | 0.0086 | 0.0019 |  0.0010 |

---

For DALES you’re doing crop sampling in point space before voxelization. Each training iteration picks a 20m × 20m spatial window from a large DALES tile/file, keeps only the points inside that window (optionally biased toward rare classes like poles/wires), then voxelizes only those points into MinkowskiEngine sparse tensors. Crop size and voxel size are both hyperparameters (data/representation knobs), not “features”, and tuning them is totally legitimate — they control memory, context, and how thin structures survive quantization.

1. What exactly is “crop sampling” in your DALES pipeline?

DALES scenes are huge. Feeding a full scene into a sparse UNet can explode memory because sparse UNets keep many active sites across multiple resolutions.

So instead of dropping voxels globally, your DALES loader does:

Load raw arrays (xyz + return attributes + native labels) from the raw cache.

(Optional) Augment the points (flip/scale/jitter).

Choose a crop center (𝑐𝑥,𝑐𝑦)

Select points in a square window around that center:
𝐶={𝑖∣∣𝑥𝑖−𝑐𝑥∣≤𝑠/2, ∣𝑦𝑖−𝑐𝑦∣≤𝑠/2}

where 𝑠=crop_size_xy_m (e.g., 20m).

(Optional) Recenter crop coords to crop-local frame (important for BEV).

Build per-point features (return_number one-hot + number_of_returns one-hot).

Voxelize those points with voxel_size and pooling rules.

Return {coords, feats, labels, (maps...), (bev_labels...)} to the trainer.

Key answer:

✅ Crop sampling happens before voxelization.

That’s also why it prevents OOM: you never voxelize the full giant scene—only the crop.

---

2. How is the crop center chosen?

You configured:

rare_center_prob: 0.5

rare_center_class_ids: [5,6] (poles, power_lines in your train ID space)

crops_per_tile_per_epoch: 8

So conceptually, per file/tile per epoch:

For each crop:

with probability 0.5, pick a random point belonging to a rare class (pole/powerline) as the crop center

otherwise, pick a random point from the scene as crop center

This biases training to include thin/rare classes more often, rather than hoping random uniform crops contain poles/wires.

If the selected crop is too small (not enough points) or too large (too many voxels), the loader retries (you set resample_tries: 25) and can shrink crop size.

---

3. How is crop size determined during runtime?

You set:

crop_size_xy_m: 20.0

min_crop_size_xy_m: 8.0

shrink_ratio: 0.8

max_shrink_steps: 5

crop_min_points: 2000

voxel budgets:

max_voxels_per_crop_soft: 120000

max_voxels_per_crop_hard: 160000

So runtime crop sizing is:

Start with s = 20.0m.

Select points in that crop.

If:

too few points (<2000) → resample a different center, or

estimated/actual voxels exceed soft budget → shrink crop:

𝑠←0.8

repeat until:

under budget, or

reached min_crop_size_xy_m, or

max shrink steps used.

If still too big → apply a hard fallback (e.g., subsample points but preserve rare classes), so you don’t crash.

So crop size is a starting hyperparam but becomes adaptive to stay OOM-safe.

---

4. Does crop sampling happen after voxelization?

No — and it matters.

If you cropped after voxelization:

you’d have to voxelize the full scene first (expensive and can OOM)

then discard many voxels (wasteful)

and you’d still have indexing/mapping complexities

Cropping in point space is the standard way to keep memory bounded.

---

5. Are voxel size and crop size “hyperparameters”? Can we tune them?

Yes, absolutely.

Crop size (crop_size_xy_m) is a hyperparameter controlling:

context (bigger crops see more surroundings; better for buildings/roads)

memory (bigger crops → more points → more voxels → more GPU memory)

rare class frequency (smaller crops + rare-center sampling can concentrate poles/wires more often)

boundary effects (too small crops can cut structures and confuse the model)

Voxel size (patch.voxel_size) is a hyperparameter controlling:

resolution (smaller voxels preserve thin wires/poles better)

compute/memory (smaller voxels create more active sites)

aliasing (large voxels can erase thin structures)

Neither is a “feature” like intensity or returns — they are representation / sampling knobs.

Tuning them is fair game, and in LiDAR segmentation papers, voxel size and crop/window size are routinely treated as key experimental knobs because they directly affect sparse tensor density and performance.

---

6. How crop size and voxel size interact (important intuition)

A rough mental model:

number of points in a crop grows ~ with area: 𝑁∝𝑠^2

number of voxels grows with both area and voxel size:
smaller voxel_size → many more unique voxels per meter.

So:

doubling crop size can quadruple points

halving voxel size can increase voxel count dramatically (especially in dense regions)

That’s why your pipeline has both:

crop-size adaptation

voxel budgets

---

7. Practical guidance for your current config

You’re currently at:

crop: 20m

voxel size: 0.02

soft/hard voxels: 120k / 160k

stride in eval: 50m non-overlap (for point eval)

This is a reasonable starting point for 2×GPU runs, but you should expect:

wires/poles are sensitive to voxel size

buildings/ground benefit from larger crop context

So you can tune:

crop_size_xy_m (e.g., 15, 20, 25)

voxel_size (e.g., 0.02 vs 0.03)

rare_center_prob (0.5 → 0.7 if utility assets are still weak)

One subtle point (ties to your earlier BEV question)

If you want BEV bounds fixed to [0,20]×[0,20], you either:

crop-center/recenter into crop frame, or

use symmetric bounds [-10,10] and shift indices like LiDOG

Otherwise BEV will “see” an empty map for most crops.
