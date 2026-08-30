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

---

---

---

---

### Quick summary

The _order_ doesn’t matter **only if you remap both sides consistently**: you must convert the dataset’s native labels into your train-ID space **before computing loss**, and during evaluation you must interpret predictions **in that same train-ID space** (or map both prediction+GT into a common space). If you ever mix spaces (GT in native IDs but model outputs in train IDs), then yes—the model will “learn the wrong thing”.

---

You’re thinking about it exactly the right way: **a class index is just a name-tag**. If the name-tags are swapped on the ground truth, the model will learn swapped semantics.

So why do people say “order doesn’t matter”?

Because in segmentation, the model never sees the _string_ “car” or “building”. It sees:

- an integer label per point (ground truth): `y ∈ {0,1,2,...,C-1}`
- a vector of logits per point (prediction): `z ∈ R^C`

Loss compares them by index.

### The only thing that matters is the mapping function

Let:

- `native_id` = what’s stored in DALES `.las` (0..8)
- `train_id` = what your model uses internally (0..7, plus ignore)

You define a mapping:

[
f:\ \text{native_id} \rightarrow \text{train_id}
]

Example (conceptually):
`native 8 (buildings) -> train 4 (buildings)`
`native 3 (cars) -> train 2 (cars)`
etc.

Now the training pipeline is correct if it does:

1. **Ground-truth mapping**
   [
   y_{\text{train}} = f(y_{\text{native}})
   ]

2. **Model predicts in train space**
   [
   \hat{y}_{\text{train}} = \arg\max_k z_k
   ]

3. **Loss uses train labels**
   Cross-entropy (or focal) is computed with `y_train` indexing into the logits.

If all three happen, then whether “cars” is train index 2 or 7 is irrelevant. It’s just a permutation of labels.

### When order _does_ matter (the failure mode you’re describing)

It breaks if you accidentally do this:

- Ground truth stays in native IDs (e.g., “cars” = 3)
- But your model outputs train IDs (e.g., “cars” = 2)

Then loss will punish the model for predicting the “correct semantic class” because it’s using the wrong index reference frame.

Concretely:

- A point is truly “car”.
- In DALES native labels, that’s `3`.
- Your model predicts train index `2` (because you decided train_id 2 = car).
- If you forgot to map GT to train space, the loss compares `2` vs `3` and says “wrong”, and the model learns nonsense.

### Simple mental model: label spaces are coordinate systems

Think of label IDs as coordinate systems:

- DALES native space: (\mathcal{Y}\_{native})
- Your model train space: (\mathcal{Y}\_{train})

Training and evaluation must never mix coordinate systems without converting.

### How to sanity-check you’re not mixing spaces

You can do a brutally effective check:

- Pick a batch.
- Print unique GT labels **after mapping** and confirm they are exactly in `{0..7} ∪ {-100}`.
- Print unique predicted labels (argmax) and confirm they are in `{0..7}`.
- If you ever see `8` in mapped GT or `-100` in predictions, something is off.

Also: if you have a `class_names` array, it must match the **train-id ordering**, not the native ordering.

---

### Bottom line (your exact question)

> “If I say index 3 is building and tiles are encoded with index 3 as cars… model learns wrong, right?”

✅ Correct — **unless** you remap the tiles’ `3 (cars)` into your internal train-id for cars before training/eval.
If you do that mapping correctly everywhere, then index numbers are arbitrary labels and order doesn’t matter.

You’re not missing anything — you’re just correctly spotting the one way this goes off the rails: **mismatched label spaces**.

---

---

---

---

### Quick mental model

- **`feat_pool` + `label_pool` decide what happens when many points fall into the same voxel.**
- **Focal vs Lovász decide what kind of mistakes get punished more.**
- **`alpha` is just a loss weight per class** (it does **not** oversample data by itself).

---

## 1) `sample_first, first` vs `mean_all, majority` (and what else exists)

### Why this exists

You voxelize a point cloud: many points → one voxel. You must:

1. choose **one feature vector** for the voxel (`feat_pool`)
2. choose **one label** for the voxel (`label_pool`)

### `feat_pool` options (feature aggregation inside a voxel)

Assume voxel (v) has point features ({x*i}*{i=1}^{n_v}).

- **`sample_first`**: pick the feature of the first/representative point
  [
  f(v)=x_{i_0}
  ]
  Fast, but noisy/unstable because “first” depends on ordering.

- **`mean_all`**: average all point features in that voxel
  [
  f(v)=\frac{1}{n_v}\sum_{i=1}^{n_v} x_i
  ]
  More stable; reduces randomness; typically better for dense LiDAR.

Other common ones you already support:

- **`max_all`**: elementwise max
  [
  f(v)=\max_i x_i
  ]
  Can emphasize strong activations but can be noisy.
- **`random`**: sample a random point’s features (with RNG)
  [
  f(v)=x_{i \sim \text{Uniform}(1..n_v)}
  ]
  Adds stochasticity like augmentation; can help generalization but hurts determinism.

### `label_pool` options (label aggregation inside a voxel)

Assume voxel has point labels ({y_i}).

- **`first`**: label = label of representative point
  [
  y(v)=y_{i_0}
  ]
  Can be wrong if the representative point is not typical.

- **`majority`**: label = most frequent label in voxel
  [
  y(v)=\arg\max_c #{i: y_i=c}
  ]
  Usually more correct and stable, especially near object boundaries.

You also might see:

- **`last`** (you use this in BEV label pooling sometimes): take label of last point / last in ordering (similar issues as `first`).

### Why `mean_all + majority` helped you

DALES is dense and noisy; with small voxel size, many points collide per voxel.
Using **`mean_all`** reduces feature noise; using **`majority`** reduces label noise. Less noise → better learning.

### What combinations should we try?

If you want a _scientific, minimal_ sweep, do this:

**Recommended sweep (most informative):**

1. `mean_all + majority` ✅ (your current best baseline)
2. `mean_all + first` (isolates label pooling effect)
3. `sample_first + majority` (isolates feature pooling effect)
4. `max_all + majority` (tests “sharp” feature pooling)
5. `random + majority` (tests stochastic pooling; may help DG but can add variance)

Avoid spending time on `sample_first + first` again unless you’re sanity-checking, because it’s typically the noisiest pair (and your E1 confirms it).

---

## 2) What is Lovász loss, and why add it?

### The goal difference

- **Focal loss** is a _per-voxel classification loss_ (like cross-entropy) that focuses on hard examples.
- **Lovász-Softmax loss** is designed to _directly optimize IoU_ (Intersection-over-Union), which is what you report as mIoU.

### Focal loss (what it does)

For a voxel with true class (y), predicted probability for that class (p_t), focal loss is:

[
\mathcal{L}_{focal} = -\alpha_y (1-p_t)^\gamma \log(p_t)
]

- If the model is already confident ((p_t \to 1)), the factor ((1-p_t)^\gamma \to 0), so **easy examples get down-weighted**.
- If the model is wrong / uncertain, loss stays large → **focus on hard examples**.
- Great when you have imbalance and many easy “background-ish” voxels.

### Lovász-Softmax (what it does)

IoU for a class depends on **global set overlaps**, not independent voxels.
Lovász loss builds a differentiable surrogate to IoU by:

- taking per-voxel “errors” (how wrong the prediction is)
- sorting them
- weighting them in a way that approximates IoU improvement

You can think of it as: **“punish the errors that matter most for IoU”**.

### Why add it _after_ focal is working

Lovász can be a bit unstable early because:

- it depends on sorting errors; early predictions are garbage → gradients can be noisy
- small classes can be especially jumpy

So your plan “enable Lovász later with warmup/ramp” is exactly right:

- first learn a decent classifier with focal
- then slowly blend in Lovász to push IoU up

### Focal vs Lovász in one line

- **Focal:** “learn to classify hard voxels”
- **Lovász:** “optimize IoU directly (set-level metric)”

And you can combine them:
[
\mathcal{L} = \mathcal{L}*{focal} + \lambda(t),\mathcal{L}*{lovasz}
]
where (\lambda(t)) ramps up with epoch (t).

---

## 3) What do the `alpha` values do?

### Not oversampling

**Alpha does not oversample data.** It does not change how often a class appears in batches.
It only changes how much its errors contribute to the loss.

### What it actually does

In focal loss (and weighted CE), `alpha_y` is a **class weight**:

[
\mathcal{L}_{focal} = -\alpha_y (1-p_t)^\gamma \log(p_t)
]

- If `alpha` for poles is larger, then **mistakes on poles cost more**.
- This helps when poles are rare: otherwise the optimizer mostly learns ground/veg/buildings.

### Relationship to sampling

There are _two independent levers_:

1. **Sampling / rare centering**: changes the data distribution the model sees
   → more rare-class voxels per epoch.

2. **Alpha weights**: changes the gradient importance per error
   → rare-class errors punch above their frequency.

Using both is common and often synergistic:

- sampling ensures the model actually sees enough rare examples
- alpha ensures it doesn’t ignore them when it does

### A practical note for your case

Your DALES split stats show poles/power*lines are \_extremely rare*.
So alpha helps, but it’s not magic: if a class barely appears in crops, alpha can’t fix “no signal”. That’s why your big jump likely came from the _combo_:

- larger crops (more context + more rare hits)
- weighted rare-centering
- alpha weighting
- better voxel pooling

---

### Opinion (clearly labeled)

**My opinion:** For the next “clean” ladder rung, keep the winning core (`mean_all+majority`, crop 40m, alpha on) and only change **one axis at a time**:

1. pooling ablation (2–3 runs max)
2. crop size sweep (40→60→80→100, short 25–30 epoch probes)
3. Lovász ramp on best configuration

That keeps the results interpretable and prevents “config soup” where you can’t tell what helped.

If you continue down this route, you’re not just tuning—you’re doing actual experimental science, which is the fun kind.

---

---

---

---

---

### Quick answer

They’re **not doing the same thing**.

- **`alpha`** is a **per-class weight** inside focal/CE (changes gradient _importance_ by class).
- **Lovász** is an **IoU-optimizing surrogate** (changes the _shape_ of the objective to align with mIoU).

So yes: **it’s totally valid to run them together**, but you should **verify with a small ablation** because they can “over-correct” rare classes if your sampling is already aggressive.

---

## Why they’re not the same (even though both “help rare classes”)

### `alpha` (class weighting) — local, per-voxel

In focal loss:
[
\mathcal{L}_{focal} = -\alpha_y (1-p_t)^\gamma \log(p_t)
]

- `alpha_y` says: “errors on class (y) matter more.”
- It does **not** know about IoU or global overlaps.
- It’s basically _rebalancing the class contributions_ to the training signal.

### Lovász — global, set-level (IoU-ish)

Lovász-Softmax tries to directly push IoU up by focusing on the ranking of errors that change IoU the most.

- It is **not** “class weighting”; it’s **metric shaping**.
- It can improve mIoU even when class weights are off, because it attacks the metric directly.

So they’re different levers:

- `alpha`: “who gets to shout louder?”
- Lovász: “are we optimizing the right game (IoU) instead of proxy (CE/focal)?”

---

## Should you enable both together?

### Most common, sane recipe (what I’d do first)

Keep what already worked for you (**focal + alpha**) and then **add Lovász gradually**:
[
\mathcal{L} = \mathcal{L}*{focal,\alpha} + \lambda(t),\mathcal{L}*{lovasz}
]
Where (\lambda(t)) ramps up after warmup.

This is exactly what your `FocalLovaszLoss` wrapper is for.

### When you might _turn off_ alpha after enabling Lovász

If you see any of these symptoms after turning Lovász on (while keeping alpha):

- Rare classes become over-predicted (precision tanks; lots of false positives)
- Common classes (ground/veg/buildings) regress hard
- Training becomes unstable/jittery (val mIoU oscillates more than usual)

That usually means you’re “stacking too many rare boosters”:
**rare-centering sampling + alpha + focal((\gamma)) + Lovász** can become too aggressive.

In that case, you don’t have to choose only one; you can **reduce intensity**:

- keep Lovász, but **reduce alpha** (closer to 1.0), or
- keep alpha, but **lower Lovász weight**, or
- keep both, but **lower rare_center_prob / weights**, or
- keep both, but reduce focal gamma from 2 → 1 (less “hard-example amplification”).

---

## Minimal, scientific ablation (3 runs, high signal)

Given your current best run uses **focal + alpha + bigger crops + rare centering**, do:

1. **Baseline:** `focal + alpha`, `lovasz.enabled=false`
2. **Add Lovász:** `focal + alpha + lovasz(ramp)`
3. **Lovász without alpha:** `focal(alpha=null) + lovasz(ramp)`

Keep everything else identical.

**Interpretation:**

- If (2) > (1): Lovász is helping.
- If (3) ≈ (2): alpha is not needed once Lovász is on (you can simplify).
- If (2) < (1) but (3) > (1): alpha + Lovász together is too much; use Lovász but drop alpha.
- If both (2) and (3) < (1): Lovász isn’t helping in your setup; keep it off.

---

## Practical settings to start with (so it doesn’t blow up)

If you enable Lovász, don’t slam it at full weight from epoch 1.

- `lovasz.enabled: true`
- `warmup_epochs: 10`
- `ramp_epochs: 10`
- `weight: 0.3 to 0.7` (start 0.5 is fine)
- `classes: present` (good default; avoids weirdness on absent classes)

And **keep alpha exactly as-is** for the first “Add Lovász” ablation run—don’t change two knobs at once.

---

### Opinion (clearly labeled)

**My opinion:** Start with **both enabled** (focal+alpha + Lovász ramp), because you already proved alpha helps, and Lovász often gives an extra mIoU bump once the classifier is “reasonable.” Then use the 3-run ablation above to decide whether to keep alpha long-term.

If you paste the current `loss:` block you want to try next, I can give you the exact YAML variants for those 3 runs (only the minimal diffs).

---

---

---

---

---

### Quick answer

`alpha` helps by **multiplying the loss (and therefore the gradient)** for specific classes. So a mistake on _trucks/poles/power_lines_ can count **2–3× more** than an equally-bad mistake on _ground/veg/buildings_. That makes the optimizer “care” about those errors instead of drowning them under the ocean of ground/veg points.

Below is a concrete numeric walk-through (with your exact alpha list).

---

## 1) What alpha actually does (numerically)

For multi-class CE (and focal is CE with extra factors), the per-sample loss is:

[
\text{CE} = -\log(p_{y})
]

With class-weighting (`alpha`):

[
\text{Weighted CE} = -\alpha_{y}\log(p_{y})
]

So alpha is a **scale factor** on the loss for samples whose true class is (y).

### Your alpha values (DALES 8 classes)

Order: `ground, vegetation, cars, trucks, buildings, poles, power_lines, fences`

- ground: **0.53**
- vegetation: **0.53**
- cars: **1.07**
- trucks: **1.60**
- buildings: **0.53**
- poles: **1.60**
- power_lines: **1.07**
- fences: **1.07**

So compared to ground, a truck/pole error is weighted:
[
\frac{1.60}{0.53} \approx 3.02\times
]

That’s already a very clear “pay attention to trucks/poles” bias.

---

## 2) Concrete example with the same prediction quality

Assume the model predicts the correct class with probability (p_y = 0.2) (pretty bad, but common early or for rare classes).

### Without alpha

[
\text{CE} = -\log(0.2) = 1.609
]

### With alpha

- If this sample is **ground**:
  [
  \text{Weighted CE} = 0.53 \times 1.609 = 0.853
  ]
- If this sample is **truck**:
  [
  \text{Weighted CE} = 1.60 \times 1.609 = 2.574
  ]

So the _same_ quality mistake produces **3× larger loss** when it’s a truck vs ground.
And since gradients scale with the loss multiplier, the parameter update “push” is ~3× stronger too.

---

## 3) What happens inside a batch (why it matters with your class imbalance)

From your class frequencies (TRAIN split, tile counts):

- ground ≈ **48.83%**
- vegetation ≈ **33.78%**
- buildings ≈ **15.85%**
- trucks ≈ **0.109%** (super rare)

Imagine a batch with 10,000 labeled points (easy numbers):

- ground points: 4883
- veg points: 3378
- buildings: 1585
- trucks: ~11 (because 0.109%)

Now assume the model’s average CE per point is roughly:

- ground/veg/buildings: 0.30 (they’re easy)
- trucks: 2.00 (hard)

### Total loss contribution WITHOUT alpha

- ground: 4883 × 0.30 = 1464.9
- veg: 3378 × 0.30 = 1013.4
- buildings: 1585 × 0.30 = 475.5
- trucks: 11 × 2.00 = 22.0

Trucks barely exist in the total signal.

### WITH alpha

- ground weighted: 1464.9 × 0.53 = 776.4
- veg weighted: 1013.4 × 0.53 = 537.1
- buildings weighted: 475.5 × 0.53 = 252.0
- trucks weighted: 22.0 × 1.60 = 35.2

Trucks are still small, but they become **60% stronger**, while the massive classes get **almost halved**. Net effect: the optimizer stops being hypnotized by ground/veg.

And in _your actual setup_, you also do **rare-centering crops**, which increases the truck/pole presence in sampled crops. Alpha then makes those rare points even more “important” when they appear.

---

## 4) How this interacts with focal loss (important nuance)

Focal loss:
[
\mathcal{L}_{focal} = -\alpha_y(1-p_t)^\gamma \log(p_t)
]

Two knobs:

- `alpha_y`: “class importance”
- ((1-p_t)^\gamma): “hard-example importance”

### Numeric focal example (γ = 2)

Take two samples:

**A) Easy sample:** (p_t=0.9)
[
(1-0.9)^2=0.01,\quad -\log(0.9)=0.105
]
Base focal (no alpha): (0.01 \times 0.105 = 0.00105)

- If it’s ground: ×0.53 ⇒ **0.00056**
- If it’s truck: ×1.60 ⇒ **0.00168**

**B) Hard sample:** (p_t=0.2)
[
(1-0.2)^2=0.64,\quad -\log(0.2)=1.609
]
Base focal: (0.64 \times 1.609 = 1.0298)

- ground: ×0.53 ⇒ **0.5458**
- truck: ×1.60 ⇒ **1.6477**

So alpha scales both easy and hard samples, but focal already boosts hard ones—together they do:

- **hard + rare** gets a _big_ gradient
- **easy + common** becomes almost irrelevant

That’s usually what you want in heavy imbalance segmentation.

---

## 5) So how is alpha helping _right now_ in your runs?

From your results:

- Without alpha (28722): **mIoU ~0.609**
- With alpha + bigger crops (28797): **mIoU ~0.689**

There are multiple changes between these runs (crop size 40, crops_per_item, rare-weighting), but alpha is very plausibly contributing because:

- trucks are extremely rare globally,
- even with rare-centering, you still need the optimizer to “treat rare mistakes as expensive.”

Alpha’s role is not oversampling. It’s **loss reweighting**—a gradient shaping trick.

---

### If you want, here’s a quick sanity test (no new training)

Log per-class _weighted loss contribution_ during training for 200 steps:

- compute `loss_per_point` (or per-voxel) and aggregate by label
- compare with alpha on/off
  You’ll literally see rare classes occupy a much larger slice of the loss pie when alpha is enabled.

(That’s the most satisfying “I can see the physics” proof.)

---

---

---

---

# DALES Train/Val Split Strategy (Manifest-based, Distribution-aware)

## Context / Problem

The DALES dataset (in our local setup) provides **train/** and **test/**, but **no official validation split**.  
A validation split is required to:

- select the best checkpoint (`best.pt`) during training,
- monitor overfitting / convergence,
- compare experiment variants (sampling, voxelization, loss weighting) in a consistent way.

### Why a naive random split is risky

DALES has **extreme class imbalance**, especially for utility assets:

- poles: ~0.07–0.10%
- power_lines: ~0.17–0.40%
- trucks: ~0.1–0.2% (rare and noisy)

With only ~29 training tiles total, a random 10% val split (≈ 3 tiles) can easily create a **validation set whose class distribution is an outlier** (e.g., trucks accidentally becoming ~10× more frequent than train/test).  
This makes validation metrics misleading and can cause the “best checkpoint” to be selected for the wrong reasons.

## What We Changed (High level)

We replaced the old random val split with a **manifest-based, distribution-aware split**:

- We compute **tile-level class statistics** (counts per class per tile).
- We search across multiple random seeds to find a 3-tile validation subset that:
  1. stays close to the global distribution, and
  2. explicitly avoids pathological rare-class spikes.
- We write the split as **manifests**:
  - `train.txt`, `val.txt`, `test.txt`
- Training uses these manifests so that the split is:
  - reproducible,
  - consistent across experiments,
  - aligned with cached files.

This approach is “scientific enough” for iterative experimentation while keeping the workflow simple.

## Artifacts Produced

All split artifacts live in:

- `tile_stats.csv`  
  Per-tile statistics used to design the split:
  - file path
  - tile grid coordinates (parsed from filename)
  - header bounds
  - point count
  - `counts_json`: per-class point counts in **train-id space**

- `manifests/train.txt`, `manifests/val.txt`, `manifests/test.txt`  
  Plaintext lists of `.las` paths used by the pipeline.

- `manifests/summary.json`  
  Summary of the final chosen split, including:
  - number of tiles in each split
  - global/train/val frequency vectors
  - a weighted distance score: `weighted_distance_val_vs_global`

## How the Split is Computed

### Step 1 — Compute per-tile stats

For each DALES tile in the training set:

1. Load point labels (`classification` / `native_labels`).
2. Map native labels → train IDs using `dales_label_map_native_to_train`.
3. Count points per class.
4. Save results to `tile_stats.csv`.

### Step 2 — Define the objective for the validation set

We choose `val_fraction_target_tiles ≈ 0.10` → `val_tiles ≈ 3`.

We evaluate candidate val subsets using a **weighted distance** between:

- `val_freq` and `global_freq`

Weights emphasize rare classes (especially poles/power_lines), so the search prefers validation subsets that preserve meaningful signal on utility assets.

The summary file stores:

- `global_freq`, `train_freq`, `val_freq`
- `weighted_distance_val_vs_global`
- number of seed trials performed (`seed_trials`)

### Step 3 — Fix the split via manifests

Once the best candidate subset is found:

- write `train.txt`, `val.txt`, `test.txt`
- write `summary.json`
- rebuild the cache using those file lists

This ensures training/evaluation always use the exact same split unless we intentionally regenerate manifests.

## Why This Is Better Than the Old Split

The old split was created by random shuffling and slicing:

- With only ~3 validation tiles, random selection can produce **distribution outliers**.
- We observed validation class frequencies that deviated significantly from train/test for rare classes (especially trucks).

The new split improves:

- **representativeness**: validation distribution is closer to global
- **stability**: avoids extreme rare-class spikes driven by randomness
- **reproducibility**: manifests lock the split for all future experiments

## How to Use This Split in Training

1. Ensure cache exists for the manifest split:
   - `.../dales_cache/dales_dropI/raw/train/*.pt`
   - `.../dales_cache/dales_dropI/raw/val/*.pt`
   - `.../dales_cache/dales_dropI/raw/test/*.pt`

2. Train using the manifest paths (recommended pattern):
   - Training reads `train.txt`
   - Validation reads `val.txt`
   - Testing reads `test.txt`

3. Retrain the best performing model config using the new manifests to measure impact:
   - this isolates “split impact” from other hyperparameter changes.

## Notes / Limitations

- Validation remains small (≈ 3 tiles), so some variance is expected.
- This is a practical compromise: it improves scientific validity without adding heavyweight complexity.
- If needed later, we can add stricter spatial separation or exclusion buffers, but it is not required for initial iteration.

## Current Split Summary (example)

From `manifests/summary.json`:

- `total_tiles`: 29
- `train_tiles`: 26
- `val_tiles`: 3
- `test_tiles`: 11
- `weighted_distance_val_vs_global`: ~0.0134 (lower is better)

The manifests provide the exact tile list used in each split.
