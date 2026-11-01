# ECLAIR return-input study

This study trains two fresh models with the same frozen ECLAIR 29306 recipe:

| Arm | Inputs at training and both-domain evaluation |
|---|---|
| observed | Existing two five-channel return encodings |
| constant | Every occupied voxel receives `[1,0,0,0,0,1,0,0,0,0]` |

The comparison asks whether informative return attributes help or harm transfer
under this recipe, and whether a model trained without those attributes retains
source competence. It cannot identify a unique root cause or establish that
all geometry representations behave the same way. It is one paired seed in
ECLAIR-to-DALES; a third training is deliberately not submitted.

## Launch and resume

From `eclair_model_train` with `e0_env` active:

```bash
python scripts/submit_feature_contract_study.py --study eclair_returns_s1337_v1 --hours 36 --dry-run
python scripts/submit_feature_contract_study.py --study eclair_returns_s1337_v1 --hours 36
```

The submission validates the login-node user quota, requires 10 GiB headroom,
checks the two-job limit, resolves and excludes `cn07`, and submits one
two-GPU allocation per arm. Allocation time is a limit, not a measured ETA.
Scientific code must be committed. No old outputs or caches are deleted.

Each allocation first checks existing cache coverage, exact split identities,
source-only cache preprocessing parity, feature/checkpoint mode compatibility,
initialization equality, and a real GPU forward/backward/checkpoint roundtrip.
Training starts only after those checks pass. No target labels select a model.
Shared caches are required; duplicate per-run caches are disabled in both arms.
Outputs are capped at 2 GiB per arm with best and last checkpoints only.

If `STATUS.json` says `needs_resume`, inspect logs and resume explicitly:

```bash
python scripts/submit_feature_contract_study.py --study eclair_returns_s1337_v1 --arms observed --resume --hours 36
```

Use the affected arm. Resume requires the same scientific code, dataset/cache
identities and software versions; it restores completed-epoch training state
and reuses completed evaluation tiles. It never resubmits automatically.

## Locations and completion

Logs: `/scratch/m23csa510/e0_logs/returns_{observed,constant}_JOBID.{out,err}`.
Output root: `/scratch/m23csa510/e0_results/feature_contract_studies/eclair_returns_s1337_v1/`.

| Output | What it establishes |
|---|---|
| `submission.json` | Actual jobs, arms and resume submissions |
| `<arm>/STATUS.json`, `training_status.json` | Current phase and actual completion, independently of Slurm exit |
| `<arm>/config_resolved.{yaml,json}`, `study_specification.json` | Exact frozen recipe and input mode |
| `<arm>/source_manifest.sha256`, `source_snapshot.tar.gz`, `git_commit.txt`, `runtime_identity.json` | Code, software, world size and hardware identities |
| `<arm>/dataset_inventory.json`, `storage_admission.json`, `preflight.json` | Split/cache identities, admitted quota and completed input/GPU checks |
| `<arm>/metrics.csv` | Every epoch, native-eleven validation mIoU, loss, F1, per-class metrics and available memory/timing |
| `<arm>/study_epochs.jsonl` | Common-seven source validation, class exposure, coordinate/label/file exposure hashes, actual updates and AMP skips by rank |
| `<arm>/checkpoint_selection.json` | Best source-native validation epoch; target labels never used |
| `<arm>/evaluation_manifest.json` | Best and last checkpoint identities and evaluation coverage |
| `<arm>/evaluation/MODEL_HASH/{eclair,dales}/tile_*.json` | Full-cloud per-tile integer common/native confusion matrices and subtype counts |
| `<arm>/evaluation/MODEL_HASH/{eclair,dales}/metrics.json`, `complete.json` | Full 125-tile ECLAIR or 11-tile DALES scores and hash-verified completion |
| `comparison.json`, `comparison.md` | Matched-execution checks, all-class and utility deltas, paired tile intervals, ordered outcome |
| `collection_manifest.json`, `results_bundle.tar.gz` | Compact hashed evidence bundle for later analysis; excludes checkpoints and raw clouds |

Best is selected using source-native validation mIoU; final weights are secondary.
Both use the canonical full test splits, 120 m windows/stride, mean logits,
no TTA and no inference AMP. Native eleven and common-seven scores stay distinct.
Ignored predictions remain false negatives for scored ground-truth classes.
Full native matrices retain ignored categories and wire/pole subtypes.

The paired report is automatic after both arms complete. It verifies equal
scientific configurations (apart from input mode), code, inventories,
initialization, accepted exposure and optimizer-update/skip budgets. A mismatch
invalidates the controlled comparison rather than being hidden in the score.

## Reading the outcome

All deltas are constant minus observed, in percentage points. The source guard
allows at most 1 pp loss in each aggregate and 2 pp in each class. The target
uses the same material-loss bands. The report applies these ordered labels:

1. `invalid_execution`: the intended matched comparison was not achieved.
2. `source_competence_failed`: restricting returns failed the source guard.
3. `observed_material_target_regression`: at least one target loss exceeds its band.
4. `strict_observed_positive_confirmation_candidate`: both target means improve,
   every target class is nondecreasing, and the source guard passes.
5. `practical_preservation_confirmation_candidate`: losses stay inside the bands
   and marginal paired tile lower intervals also clear those bounds.
6. `unresolved_practical_comparison`: no decisive screen above was satisfied.

Tile intervals describe these fixed test tiles. They do not measure training
seed variation, simultaneous confidence, causal uniqueness or future-domain
performance. A positive or preservation candidate needs independent confirmation
before becoming a claimed method improvement. Do not choose checkpoints using
target results or call improved source curves cross-domain success.
