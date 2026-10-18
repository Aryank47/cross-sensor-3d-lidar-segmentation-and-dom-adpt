# DALES-to-ECLAIR architecture pilot

This is a three-arm, single-seed architecture comparison, not an implemented NAS
search algorithm or proof that the backbone causes the domain gap.

The reference preserves the current canonical 29364 training recipe: 120 m nominal
DALES crop, 0.2 m physical input grid, returns-only inputs, existing sampling,
weighted focal implementation and scheduled Lovasz. The current focal alpha is
inside weighted CE before exponentiation; this study records and preserves it.

| Arm       | Stage block counts | Change                                           |
| --------- | ------------------ | ------------------------------------------------ |
| reference | 1,1,1,1,1,1,1,1    | Existing MinkUNet14C                             |
| context   | 1,1,2,1,1,1,1,1    | Extra 128-to-128 block at encoder stride 8       |
| fine      | 1,1,1,1,1,1,1,2    | Extra 128-to-128 block at final decoder stride 1 |

Shared weights start identically; the two added blocks also start identically.
Added parameter counts match each other, but computation does not. Model creation
does not advance the caller's CPU/CUDA RNG. Baseline/default model state keys are
preserved. Candidate checkpoints strictly reconstruct their saved architecture.

## Submit from the cluster login node

```bash
cd /csehome/m23csa510/lidar_experiments/cross-sensor-3d-lidar-segmentation-and-dom-adpt
git switch improve-train-code
git pull --ff-only origin improve-train-code
cd eclair_model_train
python scripts/submit_architecture_study.py --study dales_arch_s1337 --arms reference context
```

This submits two independent two-GPU jobs, each with a 36-hour allocation. It
checks the total existing queue against the default limit of two, verifies quota
and filesystem headroom, resolves cn07's actual scheduler spelling, and excludes
it. All shipped study/legacy GPU batch scripts also default to excluding cn07.
The compute entry point refuses to run on cn07 even if manually mis-scheduled.

If the account actually permits three queued jobs, all three may instead be
submitted together with `--arms reference context fine --max-queued 3`. Do not
submit the same study/arm twice. A per-arm allocation lock prevents concurrent
writes. The study name must be new on initial submission.

When a slot is free, submit the remaining arm:

```bash
python scripts/submit_architecture_study.py --study dales_arch_s1337 --arms fine
```

An allocation may finish training but leave evaluation pending, or stop at a
completed epoch before wall time. Continue the same arm without starting a new
experiment:

```bash
python scripts/submit_architecture_study.py --study dales_arch_s1337 --arms reference --resume
```

Replace `reference` with the arm requiring continuation. `--hours 48` is available
only if the partition permits it. No automatic requeue or hidden allocation chain
is created. Queue limits and the full combined training cost still apply.

Outputs live under:

`/scratch/m23csa510/e0_results/architecture_studies/dales_arch_s1337/<arm>/`

Read `STATUS.json`. `needs_resume` is incomplete work, not a completed result;
`failed` preserves the error rather than claiming success. Slurm logs are in
`/scratch/m23csa510/e0_logs/arch_<arm>_<jobid>.{out,err}`.

## Checks before long training

The same allocation checks exact frozen configuration parity, source/data/cache
identities and storage. It then exercises an actual accepted training batch on a
GPU: coordinate/label alignment, finite loss and gradients, bounded classifier
audit without mutating `.grad`, RNG restoration, optimizer step, strict checkpoint
save/load, and matching initialization across arms. Temporary smoke checkpoints
are removed from their own temporary directory. No training result is claimed
from this smoke test. Long training begins only after it passes.

The loader uses fresh workers and a separate explicitly seeded generator each
epoch in study mode. DALES still uses its existing per-item/rank/epoch crop seed.
This makes worker initialization consistent across uninterrupted and resumed
runs. Per-rank Python/NumPy/CPU/CUDA RNG and actual update/AMP-skip counters are
saved. Resume requires the same world size, source code and scientific settings.
Ordinary non-study loader/model behavior is retained. The study explicitly aligns
sparse output labels by coordinates rather than assuming row order.

## Evidence and selection

- `study_epochs.jsonl`: all-rank accepted native/common label counts, sample class
  presence, crop/shrink/fallback metadata, update counters and full source
  validation confusion-derived all-class/utility measures.
- `loss_gradient_audit.jsonl`: first microbatch per rank/epoch, separate focal and
  Lovasz scalars, and local class-conditioned **focal-only classifier** gradient
  norms. These are diagnostic objectives, not a per-class decomposition of Lovasz.
- `metrics.csv`: legacy source validation, losses, resource and timing curves.
- `checkpoint_selection.json`: source native all-class validation selects
  `best.pt`. Utility-best is scalar/epoch tracking only; no target epoch selection.
- `evaluation/<model_hash>/{dales,eclair}/metrics.json`: full raw-point test results
  under the canonical 120 m/120 m, mean-logit, non-AMP, 180k-voxel forward cap,
  recursive-depth-8 contract. Native argmax is mapped to common IDs unchanged.
  Native GT rows retain ordinary pole/tower and wire subtypes; per-tile matrices
  are compact and resumable. Full logits/point clouds are never written.
- `bounded_probes.json`: class-centered, capped 50 m cohorts; paired thinning,
  one-voxel grid phase, constant-return OOD and fixed-core variable-halo probes;
  stage/class feature moments and 1 m radius geometry/error strata. These bounded
  probes are not canonical test scores or physical sensor simulations.
- `frozen_head_results.json`: bounded source-train features fit a ridge readout;
  source validation chooses its regularization. Target labels never fit or select
  a readout. Features remain in RAM and are not saved as a bank.
- `source_manifest.sha256`, `source_snapshot.tar.gz`, `dataset_inventory.json`,
  `evaluation_manifest.json`, `resource_summary.json`, `bundle_manifest.json`:
  provenance, checkpoint/model identity, cache/split identity and storage evidence.

Best and final weights are both evaluated; identical model hashes are deduplicated.
Evaluation reuses completed per-tile work after an allocation ends. Bounded probes
restart as a group if interrupted before their completion marker. They do not
silently omit failed required stages.

The conservative observed screen requires improved target seven-class and utility
means, no target class IoU loss, and no source common mIoU loss. This is a strict
decision policy, not a statistical test. All arms and class tradeoffs must be
reported. The already-inspected ECLAIR target makes architecture selection from
these results exploratory; replication and an independent acquisition are still
needed. Failure of these insertions does not rule out all NAS or representations.

After the submitted arms complete, compare their fixed primary checkpoints on the
login node (NumPy required, no GPU):

```bash
python scripts/compare_architecture_study.py --study-dir /scratch/m23csa510/e0_results/architecture_studies/dales_arch_s1337
```

Paired tile intervals are descriptive; they are not uncertainty across training
seeds or independent acquisitions. A present but incomplete arm is refused.

## Storage

Each arm is capped at 2 GiB generated output, with 128 MiB for compact diagnostics.
The user-reported scratch quota had approximately 92 GiB below its soft limit;
the submission helper checks the current quota again rather than trusting that
old value. It admits two 2 GiB arms plus 2 GiB additional headroom by default.

Quota must be readable on the login node at every submission, including resume.
The launcher passes a timestamped quota receipt through Slurm's job environment.
The allocation checks its user, study directory, arms, budget and maximum age of
48 hours. It prefers a live compute-node quota query. If that query is unavailable
(for example, a remote mount has no matching quota row), it records and announces
use of the login-node observation in `storage_admission.json`. A readable live
quota below the required budget still stops the job. Missing or stale receipts
are refused. Filesystem free-space, inode and per-run limits remain active.
The receipt does not reserve quota: unrelated writes by this account can consume
space after submission, and a filesystem free-space check cannot detect that.

If an attempt failed before training and the launcher code has since changed,
start with a new study name, for example `dales_arch_s1337_qfix`. Keep the old
directory as the failure record. Do not use `--resume` across a code change:
the source-manifest check intentionally refuses it. This does not discard any
trained checkpoint in the old directory.

Retain only full-resume `last.pt` and model-only `best.pt`. Atomic replacement needs
space for the previous and new checkpoint simultaneously. Storage is checked
before each save; errors are propagated across DDP ranks. The previous atomic
resume file is never deleted to make room. If interrupted between replacing last
and best, resume repairs best from last when last was the newly selected best.

No intermediate epoch/milestone weights, separate utility-best, copied final
checkpoint, dataset duplication or new augmented cache is created. Existing raw
caches are required and read-only. Source learning curves are retained, but
arbitrary retrospective analysis of unsaved historical weights is unavailable.
There is no automatic deletion of historical runs or dataset caches.

## Local validation

CPU checks (NumPy, PyYAML and pytest):

```bash
python -m pytest -q tests/test_architecture_study.py
python -m compileall -q train.py src scripts
```

GPU/MinkowskiEngine execution is checked by compute-node preflight, not claimed
from the laptop's static checks. No search method, new loss, or new density
regularizer is introduced in this pilot.
