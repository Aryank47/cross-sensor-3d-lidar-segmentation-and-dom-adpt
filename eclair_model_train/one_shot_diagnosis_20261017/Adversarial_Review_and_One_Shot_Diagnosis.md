# Adversarial review and one-shot diagnostic specification

17 October 2026. This supersedes the priority ranking in the 11 October research review. It does not replace the canonical benchmark or claim new training results.

## 1. Verdict

The central conclusion survives: **the supplied experiments do not establish backbone capacity as the primary blocker**. This means architecture remains unidentified, not that architecture is irrelevant. The existing comparisons change augmentation or auxiliary training on the same core backbone, so they cannot isolate architecture.

I retract the stronger interpretation that geometry preservation is already the best-supported next intervention. It is a plausible architectural candidate. **Label-prior/head/exposure effects are at least as important to investigate, and the current records already contain stronger evidence of their relevance than the earlier explanation acknowledged.** None of these observations identifies a unique causal explanation.

The next action is one finite collection using the existing datasets, raw caches and eight checkpoints. It produces one evidence bundle. Then select one controlled training comparison from the decision table below; do not open a sequence of unrelated diagnostic scripts. The collection does not search or train architectures.

Internal sources: [experiment audit](C:/Users/aryan/.codex/.chatgpt-projects/g-p-68aee25f88d08191a1e79d3fc3f68de5/output/Project_State_and_Evidence_Audit.md), [implementation audit](C:/Users/aryan/.codex/.chatgpt-projects/g-p-68aee25f88d08191a1e79d3fc3f68de5/output/Code_and_Training_Evidence_Update.md), [confusion-matrix inventory](C:/Users/aryan/.codex/.chatgpt-projects/g-p-68aee25f88d08191a1e79d3fc3f68de5/output/experiment_evidence_inventory.json). The new script embeds the eight resolved configurations and 16 canonical matrices from those records.

## 2. Claims challenged and disposition

| Earlier claim or possible interpretation | Adversarial finding | Disposition |
|---|---|---|
| Canonical aggregate results are numerically correct | Independently recomputed all 16 seven-class mIoUs from the integer matrices; all match within $10^{-12}$ | Supported |
| All three methods failed | Cross-direction improvements span 1.36–4.19 percentage points; useful class gains coexist with regressions | Reject that interpretation |
| Architecture is not the blocker | No controlled architecture comparison exists | Too strong. Architecture is unresolved |
| Geometry preservation is the strongest next method | Voxel scales and shared predictions establish a constraint, but not its contribution to the observed error; finer-grid D-trained models also fail | Downgrade to candidate |
| Vehicle problems have one common mechanism | E→D primarily misses vehicles; D→E has a major vehicle false-positive problem | Reject a single undifferentiated vehicle diagnosis |
| Existing results lack class-prior information | Exact target test composition is derivable from the existing matrices | Corrected; no new run is needed for these totals |
| Class-prior differences prove label shift | They do not establish unchanged class-conditional geometry/features or source training exposure | Unsupported causal claim |
| Non-common vehicle predictions prove an ontology defect | They establish an error destination. Native Noise versus Unassigned and examples remain missing | Diagnostic clue, not defect proven |
| BEV tested another representation at inference | The local auxiliary head is inactive during regular inference and is supervised after voxel-majority labeling | Correct description required |
| OCons tested all density invariance | It tests one mask/output-consistency recipe, with class protection and eligibility choices | Too broad |
| A higher coarse-grid oracle mIoU bounds the model | Majority back-projection does not maximize macro IoU; its accuracy ceiling is a different quantity | Explicitly prohibited inference |
| Changing inference windows isolates context | Each forward subtracts its coordinate minimum, so window changes can also change strided-grid phase | New confound identified from code |
| Return-feature ablation identifies harmful inputs | Frozen-weight ablation is an out-of-distribution intervention, not retraining evidence | Sensitivity only |
| A preprint was definitely unpublished | No accepted venue was verified; that is not proof of non-publication | Publication status remains unverified |
| One diagnosis can establish the best future architecture | Untrained alternatives and seed variance cannot be recovered from existing checkpoints | Impossible from the proposed collection |

Sparse point–voxel processing and search are distinct contributions in SPVNAS. Its results make it a relevant candidate, not a demonstrated remedy for these ALS transfers. Search rankings also require validation rather than assuming a shared-weight score is definitive. (Tang et al., ECCV 2020; Yu et al., ICLR 2020.) [P1–P2]

## 3. Important findings recovered from existing results

### 3.1 Opposite vehicle error patterns

| Direction and model | Vehicle IoU | Precision | Recall |
|---|---:|---:|---:|
| E→D M0 | 14.93% | 92.56% | 15.11% |
| E→D OCons | 19.71% | 90.99% | 20.11% |
| D→E M0 | 25.18% | 27.59% | 74.25% |
| D→E OCons | 55.97% | 75.53% | 68.36% |

OCons improves D→E vehicle IoU largely by improving precision while recall decreases. In E→D, the dominant remaining problem is low recall. Therefore, proposing the same rare-class emphasis in both directions without checking false positives would be poorly justified. These are exact consequences of the saved matrices, not new model evaluations.

### 3.2 Test composition is already known

| Common class | ECLAIR test points | ECLAIR valid-point share | DALES test points | DALES valid-point share |
|---|---:|---:|---:|---:|
| Ground | 26,787,280 | 48.1370% | 68,871,897 | 50.6551% |
| Vegetation | 28,511,243 | 51.2350% | 41,464,228 | 30.4969% |
| Building | 81,468 | 0.1464% | 23,454,294 | 17.2506% |
| Wire | 235,693 | 0.4235% | 230,412 | 0.1695% |
| Pole | 16,838 | 0.0303% | 92,724 | 0.0682% |
| Fence | 9,690 | 0.0174% | 624,069 | 0.4590% |
| Vehicle | 5,775 | 0.0104% | 1,224,696 | 0.9008% |

Vehicle point prevalence differs by about 87 times between these test sets; building prevalence differs by about 118 times. This is consistent with investigating classifier/prior effects. It does not establish pure label shift, describe training-crop exposure, count independent objects, or establish an 87-fold difference in vehicle instances.

The same evidence shows that ground and vegetation account for over 99% of valid ECLAIR test points. An aggregate raw-point accuracy ceiling can therefore look reassuring while rare-class constraints remain substantial. Per-class quantities are essential. This does not invalidate the seven-class mean-IoU objective.

### 3.3 Class-aware regularization is already partly present

The DALES recipe includes class-dependent focal weights and rare-center sampling. OCons has a class-average component, but requires at least eight retained voxels per eligible class in the collated batch. It protects utility classes, not every weak class: vehicles and buildings are not in the protected lists. Therefore, neither “class imbalance was never addressed” nor “class imbalance has already been ruled out” is accurate. These are implementation facts; their effect requires measurement or a controlled intervention.

## 4. Why the existing records cannot answer every remaining question

An aggregate confusion matrix records counts of true/predicted classes. It does not record which points shared a voxel, their spatial arrangement, the native output IDs merged into a common column, or the actual crop/forward context. Multiple very different causes can produce the same matrix.

For example, an E→D vehicle point predicted as non-common could be:

- A point in a mixed voxel whose features and supervision favor another class.
- A geometrically distinguishable point assigned to a competing source output.
- A point whose return features or context differ from the source training examples.

The matrix establishes the error count but cannot distinguish these possibilities. Existing aggregate training CSVs likewise do not reconstruct per-step historical class exposure. The one-shot job obtains these joint measurements from the actual data and frozen models; it does not rerun experiments merely to rediscover their mIoUs.

## 5. The single entry point

[diagnose_project.py](C:/Users/aryan/.codex/.chatgpt-projects/g-p-68aee25f88d08191a1e79d3fc3f68de5/output/one_shot_diagnostics/diagnose_project.py) is self-contained apart from the existing project checkout, data/caches, weights and training environment. It uses the real project loaders, model builder, voxelizer and evaluator. It includes frozen evidence and expected run identities, so a separate experiment list need not be assembled.

### Full collection scope

| Component | Scope | Measurement and intended use |
|---|---|---|
| Existing evidence | 16 canonical matrices | Independently check all mIoUs and recover already-known class composition |
| Preflight | All eight checkpoints, mappings, six splits, source/cache paths and source files | Collect missing inputs together; check exact membership counts, disjointness, hashes and config drift before inference |
| Data scan | E: 1,059/62/125 train/val/test tiles; D: 26/3/11; 1,286 total | Actual point/return distributions, fingerprints, class counts and geometry constraints |
| Voxel diagnostics | 0.1, 0.2, 0.4 and 0.5 m on every scanned tile | Common-majority and native-majority-then-map survival, mixed cells and accuracy ceiling; no model retraining at these resolutions |
| Frozen inference | All eight checkpoints on both complete test sets: 16 evaluations, 1,088 checkpoint–tile combinations | Native error destinations, per-tile matrices, confidence margins, example points and exact historical confusion comparison |
| Existing historical controls, if accessible | E 29355 at 0.4 m mean, E 30404 at 0.4 m max, D 30406 trained with 140 m crops | Reuse existing trained alternatives for up to six additional evaluations; expose configuration differences and preserve historical-revision uncertainty |
| Feature observations | Sample up to 64 voxels/class/tile at block8 | Class-conditioned source/target feature moments within a checkpoint; descriptive shift, not causal separability |
| Normalization | First normalization layers and block8 observations, plus stored running statistics | Sampled activation moments and immutable-buffer checks; no target-statistic updates |
| Effective context | Every observed successful sparse forward | Active voxels, extents, coordinate phase, OOM splits and bounding-box boundary proxies |
| Training-data replay | Each run at first/middle/final epoch, up to 24 selected items per epoch | Current-loader accepted voxel exposure, crop shrinking/caps, Mix/BEV metadata, OCons retention; not exact historical DDP replay |
| Frozen sensitivity probes | One class-centred 50 m patch per class/domain/validation-or-test split where available | Paired tests listed below; bounded cohort, not exhaustive perturbation coverage |
| Uncertainty | Paired tile bootstrap for method-versus-M0 differences | Conditional tile uncertainty, including absent-class bootstrap frequency; not seed variance |

The eight canonical checkpoints use the same core backbone, with the BEV training wrapper where applicable. The optional historical controls also use that family and test existing input-resolution/pooling/crop recipes; they are not promoted to canonical M0. Their weights are reused automatically if accessible and otherwise recorded as unavailable. There are no trained Point Transformer, superpoint or NAS alternatives in the supplied package to evaluate. Creating and training those is a subsequent experiment, not a missing diagnostic variable.

### Frozen sensitivity tests

Each model uses its source validation patches and both datasets' test patches. At most 168 canonical model–patch tasks run (231 including all three accessible historical comparators), with a clean baseline and six interventions each:

1. Shuffle return-number/number-of-returns pairs jointly, preserving their patch marginal distribution.
2. Replace return inputs with a valid constant pair, explicitly marked as an inference-time out-of-distribution test.
3. Retain approximately half the raw points, preserving coordinate-minimum anchors.
4. Remove approximately 10% of active voxels without class protection; distinguish this from the trained protected OCons mask.
5. Halve the inference window: a joint context/scheduling/phase intervention.
6. Shift sparse coordinates by one XY voxel inside the forward, preserving features and active-set relative geometry, then align outputs back: a fixed-context strided-grid phase probe.

For thinning, compare against clean predictions on exactly the same retained points, not against the full clean scene. Report class survival and absent classes. Patches can overlap, and one class-centred patch is not a representative random sample of every object of that class. No significance claim is assigned to this cohort.

The job also collects sampled local geometry and two voxel-grid phases on these patches. Geometry uses explicitly reported nearest-neighbor support; density and shape descriptors are not assumed to be independent.

### What the measurements can establish

- **Native errors:** which source outputs receive missing vehicles, with target-native subclasses retained separately.
- **Representation constraint:** how much class mixing and label disappearance occurs at each grid, and how sensitive that is to grid phase.
- **Classifier sensitivity:** how native argmax, common probability summation and exclusion of non-common outputs change errors without retraining. These remain separate diagnostic decision rules.
- **Exposure:** whether seeded current-loader samples systematically omit or underrepresent classes after the actual sampling/capping pipeline.
- **Support/input sensitivity:** how frozen predictions respond to controlled inputs and scheduling.
- **Reproduction:** whether the available checkpoints and current evaluator produce exactly the saved canonical matrices.

### What they cannot establish

- A unique causal decomposition into density, priors, geometry and architecture.
- That removing a feature during training will help because its frozen ablation changes predictions.
- That aligning two class centroids will improve discrimination.
- That a feature is irrecoverably absent merely because majority-label mIoU is low.
- That NAS is necessary, or which untrained architecture wins.
- Historical exact class exposure, independent seed uncertainty, missing sensor trajectories or independent instance counts.
- A guarantee that the next training intervention solves cross-domain generalization.

These limitations are properties of the available evidence. No amount of additional scripting can infer all counterfactual trained-model outcomes from the current checkpoints.

## 6. A finite decision after the bundle

Use the following table once, rather than requesting another exploratory diagnostic script. The options are controlled research interventions, not claims that a particular result will occur.

| Evidence pattern in the collection | First training comparison | Attribution boundary |
|---|---|---|
| Reproduction, mapping or cache mismatch | Resolve the identified provenance/contract issue, then resume the same job | Do not interpret changed scores as research gains |
| Low weak-class survival and substantial mixed-cell conflicts across resolutions/phases | Matched M0 versus one fixed geometry-preserving or finer-resolution variant | Finer resolution and a new operator are separate factors; do not silently change both |
| High geometric feasibility, strong extra-output/decision-margin effects, or disproportionate exposure | One source-trained classifier/exposure intervention versus M0; freeze the encoder for a head-only test if that is the question | A common-class training head changes the label/training objective; it is not a pure head-size comparison |
| Geometry and exposure look adequate, but conditional input/support sensitivity and feature confusion persist | One fixed-backbone feature/class-geometry objective versus M0 | The perturbation and objective are explicit adaptations, not proven ALS remedies |
| Several mechanisms have material evidence | One small comparison containing M0 and at most two orthogonal corrections | Do not combine all changes into a single uninterpretable method |

Then repeat only the winning training comparison with matched seeds before investing in broad NAS. If a manual architectural change improves the relevant classes at matched conditions, search a constrained architecture family with a declared source-based selection rule and an equal-budget random baseline. SPVNAS and NAS evaluation literature motivate this comparison; they do not guarantee a gain here. (Tang et al., ECCV 2020; Yu et al., ICLR 2020.) [P1–P2]

No universal numerical cutoff is asserted for “material.” We should assess effect size, affected classes, clean-source behavior, tile concentration, operational priorities and measurement uncertainty together. A large global accuracy ceiling does not dismiss rare-class constraints.

The objective remains the project's known ECLAIR↔DALES benchmark and application classes. A new dataset is not required to execute this collection. An untouched future-domain claim would require separate validation; that is not an instruction to delay the current work.

## 7. Validation and operational limits

The local scientific and workflow tests cover rejection-column semantics, ignored labels, valid native train ID zero, native/common merge order, majority ties, brute-force verification of the accuracy ceiling, a counterexample to treating majority mIoU as a ceiling, exact quantization order, paired comparisons, resume invalidation, archive hashes and static project API compatibility. All 16 canonical mIoUs were independently reproduced from the supplied matrices.

**The complete GPU path has not been run here.** The local environment lacks PyTorch, MinkowskiEngine, datasets and actual weights. The exported cluster environment includes SciPy 1.15.3, but runtime compatibility of the instrumented job still needs execution on the existing compute node. No inference results, measured runtime or identified root cause are claimed from an unexecuted path.

The job uses one visible GPU sequentially and the existing environment, installs nothing, and trains nothing. It is a full collection, not a smoke test. Whole-data voxel analysis and loader replay add CPU/I/O work beyond historical evaluation runtimes, so no unsupported completion-time promise is made. Constructors may ensure existing cache directories exist; cache payloads and checkpoints are not rewritten.

Completed stages are reused only when the script, arguments, source, checkpoints and recorded data/cache identities match. Missing or failed work remains visible in STATUS.json. An interrupted run resumes with the same command and output directory. The upload archive excludes checkpoint weights and working raw patches.

## References

**P1.** Tang, H., Liu, Z., Zhao, S., Lin, Y., Lin, J., Wang, H., & Han, S. (2020). *Searching efficient 3D architectures with sparse point-voxel convolution*. European Conference on Computer Vision (ECCV). [Full paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730681.pdf).

**P2.** Yu, K., Sciuto, C., Jaggi, M., Musat, C., & Salzmann, M. (2020). *Evaluating the search phase of neural architecture search*. International Conference on Learning Representations (ICLR). [Full paper](https://arxiv.org/pdf/1902.08142); [accepted conference record](https://iclr.cc/virtual_2020/poster_H1loF2NFwr.html).

**P3.** Zhao, J., Gao, Y., Xia, Y., Nie, S., Yue, J., Sun, W., & Xia, S. (2026). *Cross-view domain generalization via geometric consistency for LiDAR semantic segmentation*. arXiv:2602.14525. [Metadata](https://arxiv.org/abs/2602.14525). This concept was introduced in an arXiv pre-print; peer-reviewed publication status remains unverified. Metadata without a verified venue does not prove non-publication.
