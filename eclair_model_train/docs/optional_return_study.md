# Optional-return development candidate

This submits exactly one ECLAIR training, with the approved fixed p=0.5 whole-return-group restriction during training. Geometry, labels, accepted source exposure, backbone, loss and100-epoch budget are frozen. Primary validation/inference always uses observed returns.

From the repository's `eclair_model_train` folder with `e0_env` active:

```bash
python scripts/submit_optional_return_study.py --study eclair_optional_s1337_v2 --hours 36 --dry-run
python scripts/submit_optional_return_study.py --study eclair_optional_s1337_v2 --hours 36
```

The submitter checks live user quota (10GiB minimum), excludes `cn07`, enforces the queue limit, copies hash-pinned reference evidence, and exports committed code into the new study directory. Queued and running allocations use that immutable export; a later repository pull cannot change their code. No historical files or caches are changed/deleted.

Default output root:
`/scratch/m23csa510/e0_results/optional_return_studies/eclair_optional_s1337_v2/`

The first submission attempt (`v1`) stopped before `sbatch` because Git required the subtree archive to run from the repository root. Its empty directory is preserved; no training was submitted for it. The corrected export is covered by a real-repository regression test.

Logs: `/scratch/m23csa510/e0_logs/optional_returns_JOBID.{out,err}`.

All stages run inside the single allocation: reference/data/software admission; source-only cache/masking/GPU/checkpoint/paired-inference preflight; two-GPU training; complete source/target evaluation of source-selected best and final; best-only constant-input diagnostic; decision report; compact results bundle. No reverse-direction job is automatically submitted.

## Saved evidence

- `candidate/STATUS.json`, `training_status.json`: actual phase/completion.
- `code_provenance.json`, `candidate/source_manifest.sha256`, `candidate/source_snapshot.tar.gz`: exact code and commit.
- `reference_manifest.json`, `reference/`: immutable hashes and text results for completed controls46265/46266, plus historical benchmark rows.
- `decision_rules.json`, `candidate/study_specification.json`: frozen probability, thresholds, primary/secondary views and stop rules.
- `candidate/config_resolved.{json,yaml}`, `runtime_identity.json`, `dataset_inventory.json`, `storage_admission.json`, `preflight.json`: admitted recipe, software, data and actual GPU checks.
- `candidate/metrics.csv`: native-eleven training/validation metrics and memory/timing.
- `candidate/study_epochs.jsonl`: native/common class exposure; geometry/label/file hashes; actual updates/skips; masked/unmasked sample and class exposure per rank.
- `candidate/checkpoint_selection.json`: source-only selection. Only best/last checkpoints retained.
- `candidate/evaluation_manifest.json`: evaluated identities and explicit endpoint directories.
- `candidate/evaluation/{best,last}/MODEL_SHA/{eclair,dales}/`: complete canonical integer matrices, per-tile scores, native subtype information and hash-verified completion.
- Best tile JSON additionally stores sparse `[native_GT, observed_native_prediction, constant_native_prediction, count]` rows. `constant_diagnostic.json` aggregates paired correct/new/persistent errors. It is diagnostic, never a target-selected deployment mode.
- `report.json`, `report.md`: all class/aggregate source/target deltas versus completed controls, tile intervals, final-weight vetoes, diagnostic comparisons and advancement decision.
- `collection_manifest.json`, `results_bundle.tar.gz`: compact future-analysis package, excluding checkpoints, raw point clouds and exported source tree.

## Stop and resume

Nonfinite objectives, rank disagreement and a14th AMP skip cause a numerical-screen failure; no automatic restart or precision change. Source exposure is checked against the completed observed control after every epoch. Storage is capped at2GiB per candidate; duplicate dataset caches are disabled.

Allocation deadline exits as `needs_resume` at completed-epoch/tile boundaries. Explicit resume uses the same immutable code and original state:

```bash
python scripts/submit_optional_return_study.py --study eclair_optional_s1337_v2 --resume --hours 36
```

If the queued job waits more than48hours and compute-node live quota is unavailable, refresh its login-node admission without cancelling/resubmitting:

```bash
python scripts/submit_optional_return_study.py --study eclair_optional_s1337_v2 --refresh-admission
```

Completed files stay intact. A numerical failure needs review instead of automatic resume. Slurm `COMPLETED` alone is not sufficient: require candidateSTATUS complete, report complete and all canonical completion markers. The report may validly recommend stopping the method rather than advancement.

This is one development seed with reused controls and one source direction. Tile intervals are not seed replication. The original full/constant study's exact-update failure remains unchanged. A report permitting reverse testing is a decision for later work, not authorization inside this job to launch further trainings.
