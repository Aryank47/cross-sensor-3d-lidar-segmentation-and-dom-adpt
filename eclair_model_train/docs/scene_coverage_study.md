# DALES → ECLAIR source scene coverage screen

## Decision this study buys

Does spreading source crops across a wider set of measured spatial contexts improve balanced transfer? The treatment changes crop centers within the same tile and native center label. It cannot add missing examples. It tests the whole sampling policy, including downstream changes in class exposure and crop content. It does not identify the physical sensor effect or prove that a geometry bin is an object subtype.

## Frozen arms and admission

- Fresh control and candidate use the 200-epoch, 120 m DALES architecture reference recipe. Keep MinkUNet14C, observed returns, native labels, focal/Lovász, optimizer, seed, source checkpoint selection and canonical point evaluation fixed.
- Build the catalogue from the 26 DALES training geometry files in the completed transfer study. Verify archive hashes and inventory against the runtime data. Use source-class medians of local 2 m linearity and broad 16 m height span to form four measured geometry bins. Exclude truncated and nonfinite anchors. These are neighborhood descriptors, not verified instance counts or tower labels.
- Candidate keeps the original tile and native center label. For half of eligible proposals it draws a geometry bin uniformly, then draws one of its occupied 60 m cells. Anchor coordinates, crop membership and reported centers all use the same tile-local coordinates as the DALES loader. Leave the existing proposal alone when fewer than two bins are available. This is a fixed policy, with no target-driven choice of bins, rates or weights.
- Catalogue admission requires at least three geometry bins represented in at least four tiles per monitored class. In the control allocation, replay two frozen crop draws per tile for both arms before training. Require at least 10% accepted center replacements, at least two monitored classes gaining at least two crops with anchors from their control's least-covered bin, and total accepted voxel workload within 20%. A miss stops training. The candidate waits for the complete control.
- Both allocations exclude cn07 and use the existing quota and checkpoint limits. Use only the pinned committed code, existing read-only caches and bounded outputs.

## Records and granularity

| File | Contents |
|---|---|
| coverage_catalog.json | Source tiles, verified anchor coordinates/point IDs, class/bin definitions and distinct source tile support |
| coverage_replay.json | Both arms at the same 52 source tile/crop slots, accepted centers, replacements, crop sizes, geometry-bin hits and voxel class counts |
| study_epochs.jsonl and metrics.csv | Per-epoch and per-rank accepted label counts, crop metadata, center coordinates/classes, geometry-bin hits, updates/skips, validation, timing and memory |
| evaluation/{checkpoint}/{domain}/tile_*.json | All 11 DALES and 125 ECLAIR test tiles per endpoint, full common confusion, native GT by common prediction, geometry-stratified confusion and paired prediction identities on the saved anchor subset |
| coverage_comparison.json | Matched best/final comparisons, source and target class/aggregate deltas, paired tile intervals, subtype results and anchor transition counts, plus audit and decision flags |

Count training geometry hits on retained raw anchors in accepted crops. Post-voxel class counts are a separate measure. Use the full-test matrices for headline IoUs, precision and recall. Use the fixed geometry cohorts for conditional recall/confusion with support counts. They are sampled points and do not support full-population precision estimates or a count of independent objects.

## Prespecified decision

- Execution mismatch or exposure not delivered: diagnose the implementation or treatment. No conclusion about coverage. The full-run exposure check requires at least 10% accepted center replacements and a rise of at least two percentage points in the least-covered bin crop rate for at least two monitored classes.
- Valid exposure change with balanced improvement: confirm this policy before expansion. The best endpoint requires at least +1 point on both target aggregates, positive paired-tile lower interval bounds and no target class loss beyond two points. Source aggregates may fall at most one point and any source class at most two at both endpoints. Final target aggregate changes must stay positive with class losses no greater than two points.
- Valid exposure change but a class or source guard fails: diagnose redistribution from the full and paired-anchor counts. Close this fixed policy for balanced deployment.
- Valid exposure change without a worthwhile balanced gain: close this fixed policy. Inspect which geometry bins and native subtypes remain weak. A more diverse source inventory, source-competent model, or different acquisition/representation test needs its own evidence.

This is a single-seed development comparison in one direction. It does not claim unseen-acquisition generalization, prove a unique root cause, or trigger more jobs automatically.

## Launch

From the committed cluster checkout in e0_env, run:

```bash
python -B scripts/submit_architecture_study.py --study dales_scene_coverage_s1337_v2 \
  --arms coverage_control coverage_candidate --hours 36 \
  --out-root /scratch/m23csa510/e0_results/coverage_studies \
  --coverage-geometry /scratch/m23csa510/e0_results/transfer_evidence/transfer_evidence_v1
```
