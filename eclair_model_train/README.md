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
