# Bidirectional transfer evidence study

This is evaluation of existing weights, not training, NAS or target adaptation.
The historical references are tuned project models, not untouched ECLAIR-paper baselines.

## Start one job

From the cluster, with `e0_env` active:

```bash
cd /csehome/m23csa510/lidar_experiments/cross-sensor-3d-lidar-segmentation-and-dom-adpt
git pull --ff-only origin improve-train-code
cd eclair_model_train
python -B scripts/submit_transfer_evidence_study.py --study transfer_evidence_v1
```

The launcher checks live **user quota**, checkpoint/config SHA256 values, source-validation records,
the queue limit, and the actual `cn07` node exclusion before submission. It needs at least 10 GiB
of scratch user-quota headroom. Outputs are capped at 2 GiB, including logs and temporary artifacts.
No older checkpoint, cache, result or document is automatically deleted, moved or uploaded.

To validate admission without writing/submitting:

```bash
python -B scripts/submit_transfer_evidence_study.py --study transfer_evidence_v1 --check-only
```

One allocation requests one GPU, eight CPUs, 64 GiB RAM and 36 hours; `cn07` is excluded both in
the launcher and Slurm header. GPU/Minkowski checks run in the allocation. No GPU work runs on
the login node. If compute-node quota reporting is unavailable, the validated login-node admission
snapshot can be used for at most 48 hours. This is not a storage reservation; other jobs can consume quota.

## Checkpoint scope

Ten exact `best.pt` files are pinned in `configs/transfer_evidence/study.json`:

| Source | Reference | Mix3D | OCons | BEV | New reference | Context |
|---|---|---|---|---|---|---|
| ECLAIR | 29306 / ep90 | 38162 / ep72 | 39770 / ep70 | 39798 / ep72 | — | — |
| DALES | 29364 / ep94 | 38163 / ep198 | 39769 / ep198 | 39797 / ep192 | 43929 / ep162 | 43930 / ep200 |

The October 21 read-only audit verified all files and epochs. Nine checkpoint source scores match
their CSV validation maxima. DALES Mix3D38163 has the archived/live epoch198 checkpoint but its
current CSV lacks that epoch; this reviewed provenance gap is explicit in both manifest and report.
Its complete canonical matrix must still reproduce. No automatic replacement with another model is allowed.

Integer versus string configuration keys are normalized before comparing checkpoint/resolved
inference contracts. Feature/model/grid/label-contract differences remain fatal.

## Work performed

1. Pin code, mappings, checkpoint/config bytes and raw-file/cache/split identities. Strictly load all
   native 8/11-class models, including BEV wrappers and the extra context block.
2. GPU observer-off/on and frozen-state checks on real windows from both datasets, for every model.
3. Raw neighbourhood samples from all 1,221 training/test tiles; verify their numeric payloads against
   previous complete scans. Reuse existing joint return histograms. Six metric radii: 0.5, 1, 2, 4, 8, 16 m; local and broader support are reported separately.
4. Full canonical tests: ten checkpoints × both datasets = twenty evaluations, 1,360 model/tile pairs.
   Reproduce each archived complete integer matrix exactly before interpreting new evidence.
5. Up to 32 fixed class-covering source windows per source dataset for frozen feature reference banks.
6. Twenty-four fixed windows per test dataset, all models, two fixed seeds: valid joint-return shuffling
   and 50% observation thinning. Identity controls and actual changed-voxel/input counts accompany them.
7. Consolidated all-class report, conditional-error tables, source support, feature observations,
   native subtype/decision comparisons, paired tile uncertainty and a bounded review gallery.

Canonical predictions use native argmax **before** common-class mapping. Dataset/feature/grid
semantics remain owned by the checkpoint. Full canonical windows are 120 m / 120 m; FP32, no TTA,
mean-logit aggregation and the existing 180k-voxel/depth8 splitting rules are unchanged.
Only one model is resident on the GPU at a time; other models stay on the CPU. Auxiliary BEV
activation caches are cleared between uses without changing model parameters or normalization buffers.

Diagnostic probe windows are separate fixed physical crops snapped to each model's existing grid.
Their clean controls use the same crop/split as their perturbed inputs. They are not replacements for
full-tile canonical scores. Thinning comparisons score exactly the retained raw point identities.

## Resume and completion

```bash
python -B scripts/submit_transfer_evidence_study.py --study transfer_evidence_v1 --resume
```

Resume only after the previous job terminates. A study lock prevents two runners writing concurrently.
Exact code/input identity and artifact checksums are mandatory. Completed geometry tiles, complete
test tiles (all ten models plus agreement), source windows and probes are reused. An interrupted test
tile may repeat; full per-point predictions are deliberately not persisted. No automatic requeue occurs.
An incompatible code/checkpoint/data change requires review and a new study name.

Default output:

```text
/scratch/m23csa510/e0_results/transfer_evidence/transfer_evidence_v1/
```

Read `STATUS.json`. `needs_resume` is an incomplete study, even if Slurm exits successfully.
Only `status=complete`, `completion.json` and `measurement_coverage.json` together establish
collection completion. `failed` includes the reason. Neither a partial report nor Slurm `COMPLETED`
alone establishes scientific validity.

## Main outputs

- `Research_Decision_Report.md`, `canonical_metrics.csv`, `class_deltas.csv`: all-class results and tradeoffs.
- `canonical_matrices.json`, `tile_bootstrap.json`: exact benchmark reproduction and paired tile uncertainty.
- `error_conditioned_inputs.csv`, `pooled_input_features.csv`: actual inputs joined to prediction errors.
- `native_inputs/*.npz`: complete native-subtype × common-prediction × joint-return counters, including ignored labels.
- `error_conditioned_geometry.csv`, `conditional_error_strata.csv`: density/shape/label-mixture error associations.
- `source_support.csv`, `feature_support.csv`: raw geometry support and separate within-model learned-feature observations.
- `native_and_decision_metrics.json`: native/subtype confusion, ignored-output competition and diagnostic alternative rules.
- `paired_probe_results.csv`, `probe_validity.csv`: same-point sensitivity and actual effective input changes.
- `model_error_agreement.json`: persistent errors and rescues across checkpoints.
- `gallery/review.html`, SVGs, snippets and raw-point index: up to 56 illustrative local contexts.
- `measurement_schema.json`, `measurement_coverage.json`, `study_manifest.json`, `resource_summary.json`, `logs/storage_admission.json`.

Dense full-cloud logits/features and checkpoint copies are never saved. Sampled features stay FP32.
Native labels and diagnostic sample selections never enter the model's feature tensors.

## Interpretation guards

- The primary sample is uniform within each common class, with recorded inclusion probabilities.
  Native/tile supplements are flagged separately; they are not silently population-weighted.
- Geometry queries use full raw neighbourhoods, not point-capped inference crops. Fewer than eight
  neighbours means unavailable shape statistics. Tile-edge-truncated geometry is excluded from support comparisons.
- Nearest-other-label distance is right-censored at the maximum 16 m radius. Local Z extent is not height above terrain.
- Source support uses source-only normalization and a leave-one-source-tile-out distance reference.
  Missing support in this descriptor is not proof of a missing semantic subtype. DALES tower labels remain unavailable.
- Native-max confidence is tabulated against **common-class prediction correctness**; probability-merged confidence
  and native-argmax decisions are explicitly different quantities. Alternative rules never change canonical scores.
- Frozen probes test sensitivity of these weights. Shuffling is not a physical ALS sensor simulator;
  thinning is not evidence that another OCons-style training recipe will solve transfer.
- Historical method comparisons have training/provenance differences. Context's controlled comparator is 43929.
  Source-feature bank coverage is reported; inadequate coverage is not a successful representation control.
- Geometry/feature associations, support bins and tile bootstrap intervals cannot uniquely assign cause to
  architecture, focal/Lovasz, normalization or source coverage. No automatic next-training submission occurs.

The final evidence review should shortlist at most three directions using class-specific signatures,
counter-explanations, source competence and all-class guardrails. An inconclusive result is retained.

## Local checks

With NumPy and SciPy available, run from `eclair_model_train`:

```bash
python -B -m unittest discover -s tests -p test_transfer_evidence.py -v
```

The CPU tests compare frozen original window/projection primitives from commit 3e39f78 with the
observer-enabled evaluator under a controlled backend, including overlaps and recursive splits.
They do not replace the real sparse-kernel GPU checks performed before bulk collection.
