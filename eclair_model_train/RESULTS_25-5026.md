# Cross-Sensor 3D LiDAR Semantic Segmentation: Baseline and Domain-Gap Evaluation Report

## Quick Summary

This report summarizes the final baseline evaluation protocol and results for **ECLAIR ↔ DALES cross-sensor 3D LiDAR semantic segmentation**. The evaluation is now scientifically clean because we report two distinct metric spaces separately:

1. **Native/train label-space evaluation**
   Used to compare each model with its own train-time/source-domain baseline.

2. **Common 7-class label-space evaluation**
   Used for same-domain anchors and cross-domain domain-gap analysis.

The final official **test-split** source-only cross-domain results are:

| Source model | Target dataset | Label space    |   mIoU | Macro-F1 | Absolute mIoU gap | Relative mIoU drop |
| ------------ | -------------- | -------------- | -----: | -------: | ----------------: | -----------------: |
| ECLAIR       | ECLAIR         | common 7-class | 0.8535 |   0.9157 |                 — |                  — |
| ECLAIR       | DALES          | common 7-class | 0.6272 |   0.7212 |            0.2263 |              26.5% |
| DALES        | DALES          | common 7-class | 0.8292 |   0.9007 |                 — |                  — |
| DALES        | ECLAIR         | common 7-class | 0.5819 |   0.6823 |            0.2473 |              29.8% |

The full-labelled-target diagnostic runs confirm the same trend:

| Source model | Target dataset | Split               |   mIoU | Macro-F1 | Relative mIoU drop |
| ------------ | -------------- | ------------------- | -----: | -------: | -----------------: |
| ECLAIR       | DALES          | all labelled target | 0.6115 |   0.7040 |              28.4% |
| DALES        | ECLAIR         | all labelled target | 0.5936 |   0.6991 |              28.4% |

The main conclusion is that **source-only transfer loses roughly 26–30% relative common-space mIoU across the two datasets**, with the largest degradation concentrated in **poles, fence, vehicles, and direction-specific building/wire behavior**.

---

# 1. Objective

The goal of this evaluation is to quantify the **source-only cross-sensor domain gap** between two aerial LiDAR semantic segmentation datasets:

- **ECLAIR**
- **DALES**

The trained models are evaluated in four principal directions:

1. **ECLAIR → ECLAIR**
   Same-domain source sanity and source-domain anchor.

2. **DALES → DALES**
   Same-domain source sanity and source-domain anchor.

3. **ECLAIR → DALES**
   Cross-domain transfer from ECLAIR-trained model to DALES target data.

4. **DALES → ECLAIR**
   Cross-domain transfer from DALES-trained model to ECLAIR target data.

The key research question is:

> How much performance is lost when a 3D LiDAR semantic segmentation model trained on one dataset is evaluated on another dataset under a shared semantic taxonomy?

---

# 2. Evaluation Protocol

## 2.1 Model and Feature Contract

The evaluated models use a sparse 3D convolutional backbone based on the MinkowskiEngine-style sparse convolutional framework (Choy et al., CVPR 2019). The feature contract is kept consistent with the source checkpoints used in the project:

- `return_number` one-hot encoding with `k=5`
- `number_of_returns` one-hot encoding with `k=5`
- total input channels: `10`
- intensity disabled
- RGB disabled
- coordinate features disabled

This returns-only setup is important for cross-sensor evaluation because DALES intensity is not reliable/useful in the current project setup. Therefore, source-only transfer should not depend on an ECLAIR-only intensity signal.

## 2.2 Pointwise Evaluation

The final evaluation is performed as **pointwise semantic segmentation evaluation**.

For each target tile:

1. Raw point cloud data is loaded from the raw dataset or raw cache.
2. The target tile is processed using windowed sparse inference.
3. Points in each window are voxelized.
4. The model predicts logits over occupied sparse voxels.
5. Voxel logits are mapped back to original points.
6. Metrics are accumulated over points.

This matters because the final metric is evaluated over original points, not over sparse voxels. This avoids making the metric depend on the voxelization density of a particular dataset.

## 2.3 Windowed Inference

For DALES, full tiles are very large and cannot be safely processed as a single sparse tensor. Therefore, evaluation uses windowed inference.

The current evaluation used:

```text
window_size_m = 120
window_stride_m = 120
aggregation = mean_logits
max_voxels_per_forward = 180000
```

This allows full-tile evaluation while keeping memory bounded.

Importantly, **training-time DALES crop sampling is not used as the evaluation metric**. Crop sampling was used during training to make optimization feasible. Evaluation reconstructs predictions over the target tile using windowed inference and computes pointwise metrics over the full evaluation point set.

---

# 3. Label-Space Protocol

A major risk in this project was mixing different label spaces. The final protocol separates them explicitly.

## 3.1 Native / Train Label Spaces

### DALES train-space labels

The DALES model predicts 8 train classes:

| Train ID | DALES class |
| -------: | ----------- |
|        0 | ground      |
|        1 | vegetation  |
|        2 | cars        |
|        3 | trucks      |
|        4 | buildings   |
|        5 | poles       |
|        6 | power_lines |
|        7 | fences      |

This label space is used for DALES same-domain native/train-space evaluation.

### ECLAIR train-space labels

The ECLAIR model predicts 11 train classes:

| Train ID | ECLAIR class        |
| -------: | ------------------- |
|        0 | Unassigned          |
|        1 | Ground              |
|        2 | Vegetation          |
|        3 | Buildings           |
|        4 | Noise               |
|        5 | Transmission wires  |
|        6 | Distribution wires  |
|        7 | Poles               |
|        8 | Transmission towers |
|        9 | Fence               |
|       10 | Vehicles            |

This label space is used for ECLAIR same-domain native/train-space evaluation.

## 3.2 Common 7-Class Taxonomy

For cross-domain evaluation, native labels cannot be compared directly because the datasets have different semantic granularity. Therefore, all cross-domain results use the following common taxonomy:

| Common ID | Common class |
| --------: | ------------ |
|         0 | ignore       |
|         1 | ground       |
|         2 | vegetation   |
|         3 | buildings    |
|         4 | wires        |
|         5 | poles        |
|         6 | fence        |
|         7 | vehicle      |

The important merges are:

### DALES → common

| DALES class | Common class |
| ----------- | ------------ |
| ground      | ground       |
| vegetation  | vegetation   |
| cars        | vehicle      |
| trucks      | vehicle      |
| buildings   | buildings    |
| poles       | poles        |
| power_lines | wires        |
| fences      | fence        |

### ECLAIR → common

| ECLAIR class        | Common class |
| ------------------- | ------------ |
| Unassigned          | ignore       |
| Ground              | ground       |
| Vegetation          | vegetation   |
| Buildings           | buildings    |
| Noise               | ignore       |
| Transmission wires  | wires        |
| Distribution wires  | wires        |
| Poles               | poles        |
| Transmission towers | poles        |
| Fence               | fence        |
| Vehicles            | vehicle      |

This mapping is the reason native/train-space and common-space metrics are not directly comparable.

---

# 4. Metrics

The report uses:

## 4.1 Per-class IoU

For class (c):

[
IoU_c = \frac{TP_c}{TP_c + FP_c + FN_c}
]

## 4.2 Mean IoU

[
mIoU = \frac{1}{C}\sum_{c=1}^{C} IoU_c
]

where the `ignore` class is excluded from the mean.

## 4.3 Per-class F1

[
F1_c = \frac{2TP_c}{2TP_c + FP_c + FN_c}
]

## 4.4 Macro-F1

[
MacroF1 = \frac{1}{C}\sum_{c=1}^{C} F1_c
]

again excluding the ignored class.

## 4.5 Domain Gap

For a source model (S), the absolute domain gap is computed as:

[
Gap_{S \rightarrow T} = mIoU_{S \rightarrow S}^{common} - mIoU_{S \rightarrow T}^{common}
]

The relative drop is:

[
RelativeDrop\_{S \rightarrow T}
==============================

\frac{
mIoU*{S \rightarrow S}^{common} - mIoU*{S \rightarrow T}^{common}
}{
mIoU\_{S \rightarrow S}^{common}
}
\times 100
]

---

# 5. Same-Domain Evaluation Results

## 5.1 DALES → DALES

The DALES checkpoint was evaluated in both DALES train/native space and common space.

### DALES train-space result

| Metric           |       Value |
| ---------------- | ----------: |
| mIoU             |      0.7586 |
| Macro-F1         |      0.8410 |
| Tiles            |          11 |
| Total points     | 136,643,891 |
| Evaluated points | 135,962,320 |

### DALES train-space per-class results

| Class       |    IoU |     F1 | Precision | Recall |    Support |
| ----------- | -----: | -----: | --------: | -----: | ---------: |
| ground      | 0.9598 | 0.9795 |    0.9696 | 0.9896 | 68,871,897 |
| vegetation  | 0.9270 | 0.9621 |    0.9685 | 0.9558 | 41,464,228 |
| cars        | 0.8142 | 0.8976 |    0.8806 | 0.9152 |  1,070,554 |
| trucks      | 0.2718 | 0.4274 |    0.5568 | 0.3468 |    154,142 |
| buildings   | 0.9339 | 0.9658 |    0.9835 | 0.9488 | 23,454,294 |
| poles       | 0.6630 | 0.7974 |    0.8172 | 0.7785 |     92,724 |
| power_lines | 0.8916 | 0.9427 |    0.9581 | 0.9278 |    230,412 |
| fences      | 0.6071 | 0.7555 |    0.7710 | 0.7406 |    624,069 |

### DALES common-space result

| Metric           |       Value |
| ---------------- | ----------: |
| mIoU             |      0.8292 |
| Macro-F1         |      0.9007 |
| Tiles            |          11 |
| Total points     | 136,643,891 |
| Evaluated points | 135,962,320 |

### DALES common-space per-class results

| Common class |    IoU |     F1 | Precision | Recall |    Support |
| ------------ | -----: | -----: | --------: | -----: | ---------: |
| ground       | 0.9598 | 0.9795 |    0.9696 | 0.9896 | 68,871,897 |
| vegetation   | 0.9270 | 0.9621 |    0.9685 | 0.9558 | 41,464,228 |
| buildings    | 0.9339 | 0.9658 |    0.9835 | 0.9488 | 23,454,294 |
| wires        | 0.8916 | 0.9427 |    0.9581 | 0.9278 |    230,412 |
| poles        | 0.6630 | 0.7974 |    0.8172 | 0.7785 |     92,724 |
| fence        | 0.6071 | 0.7555 |    0.7710 | 0.7406 |    624,069 |
| vehicle      | 0.8217 | 0.9022 |    0.9082 | 0.8962 |  1,224,696 |

### Interpretation

The DALES common-space mIoU is higher than the train-space mIoU:

```text
train-space mIoU  = 0.7586
common-space mIoU = 0.8292
```

This is expected because common-space merges:

```text
cars + trucks → vehicle
```

In train-space, the `trucks` class has low IoU:

```text
trucks IoU = 0.2718
```

After merging cars and trucks into `vehicle`, car↔truck confusions are no longer penalized. The merged vehicle class achieves:

```text
vehicle IoU = 0.8217
```

Therefore, the common-space improvement is not a model improvement; it is a taxonomy effect.

---

## 5.2 ECLAIR → ECLAIR

The ECLAIR checkpoint was evaluated in both ECLAIR train/native space and common space.

### ECLAIR train-space result

| Metric           |      Value |
| ---------------- | ---------: |
| mIoU             |     0.7792 |
| Macro-F1         |     0.8534 |
| Tiles            |        125 |
| Total points     | 55,653,830 |
| Evaluated points | 55,653,830 |

### ECLAIR train-space per-class results

| Train ID | ECLAIR class        |    IoU |     F1 | Precision | Recall |    Support |
| -------: | ------------------- | -----: | -----: | --------: | -----: | ---------: |
|        0 | Unassigned          | 0.1882 | 0.3167 |    0.2751 | 0.3732 |      4,821 |
|        1 | Ground              | 0.9677 | 0.9836 |    0.9745 | 0.9929 | 26,787,280 |
|        2 | Vegetation          | 0.9690 | 0.9842 |    0.9930 | 0.9756 | 28,511,243 |
|        3 | Buildings           | 0.8666 | 0.9285 |    0.9661 | 0.8938 |     81,468 |
|        4 | Noise               | 0.7117 | 0.8316 |    0.8869 | 0.7828 |      1,022 |
|        5 | Transmission wires  | 0.9947 | 0.9973 |    0.9963 | 0.9984 |    212,257 |
|        6 | Distribution wires  | 0.9160 | 0.9562 |    0.9721 | 0.9407 |     23,436 |
|        7 | Poles               | 0.7535 | 0.8594 |    0.9097 | 0.8144 |      7,010 |
|        8 | Transmission towers | 0.8763 | 0.9341 |    0.9259 | 0.9424 |      9,828 |
|        9 | Fence               | 0.6792 | 0.8089 |    0.8934 | 0.7391 |      9,690 |
|       10 | Vehicles            | 0.6484 | 0.7867 |    0.7161 | 0.8729 |      5,775 |

### ECLAIR common-space result

| Metric           |      Value |
| ---------------- | ---------: |
| mIoU             |     0.8535 |
| Macro-F1         |     0.9157 |
| Tiles            |        125 |
| Total points     | 55,653,830 |
| Evaluated points | 55,647,987 |

### ECLAIR common-space per-class results

| Common class |    IoU |     F1 | Precision | Recall |    Support |
| ------------ | -----: | -----: | --------: | -----: | ---------: |
| ground       | 0.9678 | 0.9836 |    0.9745 | 0.9929 | 26,787,280 |
| vegetation   | 0.9690 | 0.9843 |    0.9931 | 0.9756 | 28,511,243 |
| buildings    | 0.8677 | 0.9291 |    0.9674 | 0.8938 |     81,468 |
| wires        | 0.9911 | 0.9956 |    0.9962 | 0.9949 |    235,693 |
| poles        | 0.8339 | 0.9094 |    0.9294 | 0.8903 |     16,838 |
| fence        | 0.6926 | 0.8184 |    0.9168 | 0.7391 |      9,690 |
| vehicle      | 0.6526 | 0.7898 |    0.7212 | 0.8729 |      5,775 |

### Interpretation

The ECLAIR common-space mIoU is higher than the train-space mIoU:

```text
train-space mIoU  = 0.7792
common-space mIoU = 0.8535
```

This is expected for two reasons.

First, the common taxonomy ignores:

```text
Unassigned → ignore
Noise      → ignore
```

The difference in evaluated points confirms this exactly:

```text
train-space evaluated points  = 55,653,830
common-space evaluated points = 55,647,987
difference                   = 5,843
```

The ignored train-space supports are:

```text
Unassigned support = 4,821
Noise support      = 1,022
total              = 5,843
```

Second, common-space merges:

```text
Transmission wires + Distribution wires → wires
Poles + Transmission towers             → poles
```

Therefore, the common-space result is the correct same-domain anchor for cross-domain comparison, while the train-space result is the correct value for native/source-domain evaluation.

---

# 6. Cross-Domain Test-Split Results

The domain-gap results use the held-out target test split.

## 6.1 ECLAIR → DALES Test

| Metric           |       Value |
| ---------------- | ----------: |
| mIoU             |      0.6272 |
| Macro-F1         |      0.7212 |
| Tiles            |          11 |
| Total points     | 136,643,891 |
| Evaluated points | 135,962,320 |

### Per-class results

| Common class |    IoU |     F1 | Precision | Recall |    Support |
| ------------ | -----: | -----: | --------: | -----: | ---------: |
| ground       | 0.9491 | 0.9739 |    0.9551 | 0.9935 | 68,871,897 |
| vegetation   | 0.8828 | 0.9377 |    0.9618 | 0.9148 | 41,464,228 |
| buildings    | 0.9074 | 0.9515 |    0.9605 | 0.9426 | 23,454,294 |
| wires        | 0.7866 | 0.8806 |    0.8117 | 0.9622 |    230,412 |
| poles        | 0.2836 | 0.4419 |    0.8406 | 0.2997 |     92,724 |
| fence        | 0.4315 | 0.6029 |    0.7452 | 0.5062 |    624,069 |
| vehicle      | 0.1493 | 0.2598 |    0.9256 | 0.1511 |  1,224,696 |

### Interpretation

The ECLAIR-trained model transfers well for large structural classes:

```text
ground IoU     = 0.9491
vegetation IoU = 0.8828
buildings IoU  = 0.9074
wires IoU      = 0.7866
```

However, it performs poorly on smaller and object-like classes:

```text
poles IoU   = 0.2836
vehicle IoU = 0.1493
fence IoU   = 0.4315
```

The vehicle result is particularly revealing:

```text
vehicle precision = 0.9256
vehicle recall    = 0.1511
```

This means the ECLAIR model is very conservative on DALES vehicles. When it predicts vehicle, it is usually correct, but it misses most DALES vehicle points.

Similarly, for poles:

```text
poles precision = 0.8406
poles recall    = 0.2997
```

This indicates low recall rather than random overprediction.

---

## 6.2 DALES → ECLAIR Test

| Metric           |      Value |
| ---------------- | ---------: |
| mIoU             |     0.5819 |
| Macro-F1         |     0.6823 |
| Tiles            |        125 |
| Total points     | 55,653,830 |
| Evaluated points | 55,647,987 |

### Per-class results

| Common class |    IoU |     F1 | Precision | Recall |    Support |
| ------------ | -----: | -----: | --------: | -----: | ---------: |
| ground       | 0.9421 | 0.9702 |    0.9861 | 0.9548 | 26,787,280 |
| vegetation   | 0.9478 | 0.9732 |    0.9608 | 0.9859 | 28,511,243 |
| buildings    | 0.2412 | 0.3887 |    0.2789 | 0.6408 |     81,468 |
| wires        | 0.8818 | 0.9372 |    0.9960 | 0.8850 |    235,693 |
| poles        | 0.1415 | 0.2480 |    0.1817 | 0.3902 |     16,838 |
| fence        | 0.4673 | 0.6370 |    0.6243 | 0.6502 |      9,690 |
| vehicle      | 0.4516 | 0.6222 |    0.5385 | 0.7368 |      5,775 |

### Interpretation

The DALES-trained model transfers well for:

```text
ground IoU     = 0.9421
vegetation IoU = 0.9478
wires IoU      = 0.8818
```

However, it performs poorly on:

```text
buildings IoU = 0.2412
poles IoU     = 0.1415
```

The building result is directionally different from ECLAIR→DALES. In DALES→ECLAIR:

```text
building precision = 0.2789
building recall    = 0.6408
```

This indicates overprediction of buildings on ECLAIR.

For poles:

```text
poles precision = 0.1817
poles recall    = 0.3902
```

This means the model both overpredicts and misses many pole points. Poles are therefore one of the most unstable classes in cross-domain transfer.

---

# 7. Test-Split Domain Gap

## 7.1 ECLAIR Source Model

Same-domain common-space anchor:

```text
ECLAIR→ECLAIR common mIoU = 0.85354000
ECLAIR→ECLAIR macro-F1    = 0.91574899
```

Cross-domain result:

```text
ECLAIR→DALES common mIoU = 0.62719719
ECLAIR→DALES macro-F1    = 0.72118062
```

Absolute mIoU gap:

[
0.85354000 - 0.62719719 = 0.22634281
]

Relative mIoU drop:

[
\frac{0.22634281}{0.85354000} \times 100 = 26.5%
]

Macro-F1 gap:

[
0.91574899 - 0.72118062 = 0.19456837
]

Relative Macro-F1 drop:

[
\frac{0.19456837}{0.91574899} \times 100 = 21.3%
]

## 7.2 DALES Source Model

Same-domain common-space anchor:

```text
DALES→DALES common mIoU = 0.82917554
DALES→DALES macro-F1    = 0.90074192
```

Cross-domain result:

```text
DALES→ECLAIR common mIoU = 0.58190714
DALES→ECLAIR macro-F1    = 0.68233979
```

Absolute mIoU gap:

[
0.82917554 - 0.58190714 = 0.24726840
]

Relative mIoU drop:

[
\frac{0.24726840}{0.82917554} \times 100 = 29.8%
]

Macro-F1 gap:

[
0.90074192 - 0.68233979 = 0.21840213
]

Relative Macro-F1 drop:

[
\frac{0.21840213}{0.90074192} \times 100 = 24.3%
]

## 7.3 Summary Table

| Source | Same-domain target | Cross-domain target | Same-domain common mIoU | Cross-domain common mIoU | Absolute gap | Relative drop |
| ------ | ------------------ | ------------------- | ----------------------: | -----------------------: | -----------: | ------------: |
| ECLAIR | ECLAIR             | DALES               |                  0.8535 |                   0.6272 |       0.2263 |         26.5% |
| DALES  | DALES              | ECLAIR              |                  0.8292 |                   0.5819 |       0.2473 |         29.8% |

The DALES→ECLAIR direction has a slightly larger official test-split mIoU drop.

---

# 8. Full-Labelled-Target Diagnostic Results

The official domain-gap numbers above use held-out test splits. In addition, full-labelled-target diagnostic runs were performed.

These are **not official held-out test metrics**, because they include target train/validation data. They are included only to test whether the observed cross-domain pattern is stable across more target scenes.

## 8.1 ECLAIR → DALES-All

| Metric           |       Value |
| ---------------- | ----------: |
| mIoU             |      0.6115 |
| Macro-F1         |      0.7040 |
| Tiles            |          40 |
| Total points     | 505,311,573 |
| Evaluated points | 497,632,442 |

### Per-class results

| Common class |    IoU |     F1 | Precision | Recall |     Support |
| ------------ | -----: | -----: | --------: | -----: | ----------: |
| ground       | 0.9362 | 0.9671 |    0.9427 | 0.9927 | 246,893,458 |
| vegetation   | 0.8685 | 0.9296 |    0.9638 | 0.8978 | 162,282,348 |
| buildings    | 0.8987 | 0.9466 |    0.9525 | 0.9409 |  80,362,827 |
| wires        | 0.7909 | 0.8833 |    0.8190 | 0.9585 |   1,030,298 |
| poles        | 0.2141 | 0.3527 |    0.3803 | 0.3288 |     369,648 |
| fence        | 0.4318 | 0.6032 |    0.7312 | 0.5133 |   2,136,996 |
| vehicle      | 0.1402 | 0.2459 |    0.9136 | 0.1421 |   4,556,867 |

Compared with ECLAIR→DALES test:

```text
test mIoU = 0.6272
all mIoU  = 0.6115
difference = -0.0157
```

The full-target result is slightly lower than the test result, but close enough to support the same interpretation.

## 8.2 DALES → ECLAIR-All

| Metric           |       Value |
| ---------------- | ----------: |
| mIoU             |      0.5936 |
| Macro-F1         |      0.6991 |
| Tiles            |       1,246 |
| Total points     | 582,042,807 |
| Evaluated points | 581,091,255 |

### Per-class results

| Common class |    IoU |     F1 | Precision | Recall |     Support |
| ------------ | -----: | -----: | --------: | -----: | ----------: |
| ground       | 0.9388 | 0.9684 |    0.9813 | 0.9558 | 297,177,970 |
| vegetation   | 0.9307 | 0.9641 |    0.9468 | 0.9821 | 270,351,788 |
| buildings    | 0.6151 | 0.7617 |    0.8502 | 0.6899 |   9,748,819 |
| wires        | 0.7180 | 0.8359 |    0.9915 | 0.7224 |   2,361,670 |
| poles        | 0.0991 | 0.1804 |    0.1622 | 0.2031 |     346,672 |
| fence        | 0.3297 | 0.4958 |    0.5127 | 0.4801 |     541,869 |
| vehicle      | 0.5239 | 0.6876 |    0.6537 | 0.7252 |     562,467 |

Compared with DALES→ECLAIR test:

```text
test mIoU = 0.5819
all mIoU  = 0.5936
difference = +0.0117
```

The full-target result is slightly higher than the test result. This indicates that the ECLAIR test split is somewhat harder for some classes, especially buildings, but the overall source-only gap remains consistent.

## 8.3 Full-Target Diagnostic Domain Gap

Using the same-domain test common-space anchors:

| Source | Target | Split               | Same-domain anchor | Cross mIoU | Absolute gap | Relative drop |
| ------ | ------ | ------------------- | -----------------: | ---------: | -----------: | ------------: |
| ECLAIR | DALES  | all-labelled target |             0.8535 |     0.6115 |       0.2421 |         28.4% |
| DALES  | ECLAIR | all-labelled target |             0.8292 |     0.5936 |       0.2356 |         28.4% |

The full-target diagnostics show nearly symmetric relative mIoU drops:

```text
ECLAIR→DALES-all relative drop = 28.4%
DALES→ECLAIR-all relative drop = 28.4%
```

This reinforces the conclusion that the observed domain gap is not merely a test-split artifact.

---

# 9. Overall Result Summary

## 9.1 Main Result Table

| Source | Target | Split               | Label space    |   mIoU | Macro-F1 | mIoU gap | Relative mIoU drop |
| ------ | ------ | ------------------- | -------------- | -----: | -------: | -------: | -----------------: |
| ECLAIR | ECLAIR | test                | common 7-class | 0.8535 |   0.9157 |        — |                  — |
| ECLAIR | DALES  | test                | common 7-class | 0.6272 |   0.7212 |   0.2263 |              26.5% |
| ECLAIR | DALES  | all-labelled target | common 7-class | 0.6115 |   0.7040 |   0.2421 |              28.4% |
| DALES  | DALES  | test                | common 7-class | 0.8292 |   0.9007 |        — |                  — |
| DALES  | ECLAIR | test                | common 7-class | 0.5819 |   0.6823 |   0.2473 |              29.8% |
| DALES  | ECLAIR | all-labelled target | common 7-class | 0.5936 |   0.6991 |   0.2356 |              28.4% |

---

# 10. Key Observations

## 10.1 Same-domain source models are strong

Both source models perform well in same-domain common-space evaluation:

```text
ECLAIR→ECLAIR common mIoU = 0.8535
DALES→DALES common mIoU  = 0.8292
```

This confirms that the source checkpoints are not weak models. The cross-domain degradation is therefore meaningful and cannot be explained simply by failed source training.

## 10.2 Native/train-space and common-space metrics must not be mixed

The same checkpoint can produce different mIoU values depending on label taxonomy.

For DALES:

```text
train-space mIoU  = 0.7586
common-space mIoU = 0.8292
```

For ECLAIR:

```text
train-space mIoU  = 0.7792
common-space mIoU = 0.8535
```

This is expected because the common space merges or ignores some train-space classes. Therefore:

- native/train-space results are used for source-domain reporting;
- common-space results are used for cross-domain reporting.

This separation is necessary for scientific validity.

## 10.3 Large geometric classes transfer well

Across both cross-domain directions, ground and vegetation remain high:

| Direction         | Ground IoU | Vegetation IoU |
| ----------------- | ---------: | -------------: |
| ECLAIR→DALES test |     0.9491 |         0.8828 |
| DALES→ECLAIR test |     0.9421 |         0.9478 |
| ECLAIR→DALES all  |     0.9362 |         0.8685 |
| DALES→ECLAIR all  |     0.9388 |         0.9307 |

These classes are large, spatially continuous, and geometrically stable. They are less sensitive to sensor/domain differences.

## 10.4 Sparse object classes dominate the domain gap

The largest failures occur in:

```text
poles
fence
vehicles
```

For ECLAIR→DALES test:

```text
poles IoU   = 0.2836
vehicle IoU = 0.1493
```

For DALES→ECLAIR test:

```text
poles IoU     = 0.1415
buildings IoU = 0.2412
```

For the full-target diagnostic:

```text
ECLAIR→DALES-all poles IoU = 0.2141
ECLAIR→DALES-all vehicle IoU = 0.1402
DALES→ECLAIR-all poles IoU = 0.0991
DALES→ECLAIR-all fence IoU = 0.3297
```

This suggests that the cross-sensor gap is driven less by broad surface recognition and more by small, sparse, object-level semantics.

## 10.5 ECLAIR→DALES mainly suffers from low recall for vehicles and poles

For ECLAIR→DALES test:

```text
vehicle precision = 0.9256
vehicle recall    = 0.1511
poles precision   = 0.8406
poles recall      = 0.2997
```

For ECLAIR→DALES-all:

```text
vehicle precision = 0.9136
vehicle recall    = 0.1421
poles precision   = 0.3803
poles recall      = 0.3288
```

This means the ECLAIR source model is conservative on DALES vehicles and poles. It often misses them, but when it predicts vehicle in particular, it is usually correct.

## 10.6 DALES→ECLAIR has direction-specific building and pole behavior

On the ECLAIR test split:

```text
buildings IoU = 0.2412
poles IoU     = 0.1415
```

On all labelled ECLAIR data:

```text
buildings IoU = 0.6151
poles IoU     = 0.0991
```

The large difference in building IoU between test and all-labelled ECLAIR indicates that the ECLAIR test split is particularly difficult for DALES→ECLAIR building transfer. However, poles remain weak in both splits.

## 10.7 Full-target diagnostics support the test-split conclusion

The official test-split relative drops are:

```text
ECLAIR→DALES test = 26.5%
DALES→ECLAIR test = 29.8%
```

The full-labelled-target diagnostic drops are:

```text
ECLAIR→DALES-all = 28.4%
DALES→ECLAIR-all = 28.4%
```

This consistency strengthens the conclusion that the source-only domain gap is real and robust.

---

# 11. Scientific Interpretation

The evaluation shows that source-only 3D LiDAR semantic segmentation transfers reasonably well for dominant geometric classes but degrades substantially for sparse and utility-relevant object classes.

The common-space results are especially important because they remove incompatible label granularity between ECLAIR and DALES. Even after this fair label-space alignment, cross-domain transfer loses around one-quarter to one-third of same-domain mIoU.

This indicates that the remaining gap is not only a label-taxonomy problem. It is likely driven by:

1. point-density and sampling differences,
2. scene-composition differences,
3. different object context distributions,
4. annotation-policy differences,
5. geometric scale and visibility differences for small structures.

The most domain-sensitive classes are poles, fences, and vehicles. These are small, sparse, and context-dependent. They are much more sensitive to local geometry, density, and dataset-specific acquisition conditions than ground or vegetation.

---

# 12. Implications for the Next Stage

These results provide a clean baseline for the next phase of the thesis.

The next-stage domain generalization or adaptation method should focus on improving:

```text
poles
fence
vehicles
direction-specific building/wire failures
```

The broad surface classes are already strong, so future methods should not be judged only by global mIoU. Per-class changes are essential.

In particular, a good domain adaptation method should ideally:

1. maintain ground and vegetation performance,
2. improve low-recall vehicle transfer in ECLAIR→DALES,
3. improve pole recognition in both directions,
4. reduce DALES→ECLAIR building instability,
5. improve fence transfer without sacrificing large-class performance.

---

# 13. Final Conclusion

The final evaluation protocol is now scientifically clean because it separates:

```text
native/train-space source-domain evaluation
```

from:

```text
common-space cross-domain evaluation
```

The same-domain train-space results validate the source checkpoints:

```text
ECLAIR train/native mIoU = 0.7792
DALES train/native mIoU  = 0.7586
```

The same-domain common-space results provide fair source anchors:

```text
ECLAIR common mIoU = 0.8535
DALES common mIoU  = 0.8292
```

The official test-split cross-domain results show substantial source-only domain gaps:

```text
ECLAIR→DALES mIoU = 0.6272, relative drop = 26.5%
DALES→ECLAIR mIoU = 0.5819, relative drop = 29.8%
```

The full-labelled-target diagnostics confirm this conclusion:

```text
ECLAIR→DALES-all mIoU = 0.6115, relative drop = 28.4%
DALES→ECLAIR-all mIoU = 0.5936, relative drop = 28.4%
```

Therefore, the project has established a defensible source-only baseline: both models are strong in-domain, but cross-sensor transfer remains substantially degraded, especially for sparse utility-relevant classes. This provides a strong motivation for the next phase of domain generalization or adaptation.

---

# References

- Choy, C., Gwak, J., & Savarese, S. (2019). **4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks**. _CVPR 2019_.
- Berman, M., Triki, A. R., & Blaschko, M. B. (2018). **The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-over-Union Measure in Neural Networks**. _CVPR 2018_.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). **Focal Loss for Dense Object Detection**. _ICCV 2017_.
- Varney, N., Asari, V. K., & Graehling, Q. (2020). **DALES: A Large-scale Aerial LiDAR Data Set for Semantic Segmentation**. _arXiv pre-print, 2020_.
- Melekhov, I., et al. (2024). **ECLAIR: A High-Fidelity Aerial LiDAR Dataset for Semantic Segmentation**. _arXiv pre-print, 2024_.
