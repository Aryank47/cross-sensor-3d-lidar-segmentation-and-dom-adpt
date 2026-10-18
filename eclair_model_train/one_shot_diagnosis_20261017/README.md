# Run one collection

Copy `diagnose_project.py` into the existing cluster checkout. In the same GPU environment used for the canonical training/evaluation, run from that checkout:

```bash
python diagnose_project.py --repo . --out /scratch/m23csa510/project_diagnosis_20261017
```

This is the complete entry point. There is no separate configuration-generation, data-statistics, inference or packaging command. Use an existing GPU allocation; this command does not submit a scheduler job or activate another environment.

## Included automatically

- Eight canonical runs: ECLAIR/DALES M0 29306/29364; Mix3D 38162/38163; OCons 39770/39769; BEV 39798/39797.
- Automatically reuse historical 29355 (0.4 m mean), 30404 (0.4 m max) and 30406 (140 m training crop) if their existing weights are accessible. Missing historical weights are recorded, not fabricated or required to complete the canonical collection.
- Frozen resolved configurations, expected checkpoint paths and all 16 canonical confusion matrices.
- All six dataset splits, exact canonical split counts and label-map validation.
- Four physical grid sizes on all 1,286 tiles, complete test inference for all eight checkpoints, sampled feature/normalization observations, training-loader replay and bounded paired perturbations.
- Stage-level completion records, hashes, discrepancy checks, summaries and an upload archive.

The script uses the project's actual `src.eval_cross_common`, loaders, voxelizer and model builder. It does not train, search architectures, update normalization statistics, replace benchmark rules, install packages, or modify weights/cache payloads. Existing cache directories may be checked/created by the original constructors.

## Required existing environment

Use the original project's working CUDA/PyTorch/MinkowskiEngine environment, including NumPy, PyYAML, SciPy and the project's existing loader dependencies. SciPy is included in the uploaded cluster environment. The script runs sequentially on one visible GPU; do not launch multiple distributed copies.

The actual best weights, raw caches, source files required by stat-based cache keys, ECLAIR metadata and DALES manifests must be accessible. They were not all uploaded to this chat, so the full GPU path has not been validated here. Local numerical/workflow tests are provided separately and are not a substitute for a cluster run.

## If existing paths changed

Only if the results root moved:

```bash
python diagnose_project.py --repo . --results-root /actual/results/root --out /actual/diagnostic/output
```

`--path-map OLD=NEW` remaps path prefixes inside embedded configurations. Manifest contents and stat-keyed caches still have to be compatible with the original loaders; the script does not migrate/rekey datasets or caches. `--overrides` accepts a JSON object with optional `runs` entries keyed by run ID (for example an explicit checkpoint path) and `datasets` entries containing full target configurations. Prefer unchanged original paths for this collection.

The preflight collects missing inputs/configuration discrepancies into `preflight.json` before starting GPU inference. Critical source changes stop the default run; `--allow-code-drift` explicitly permits a recorded checkout difference, but does not certify compatibility or canonical reproduction. Do not use it to hide an unexplained model or feature-contract change.

## Finite scope

The default collection scans all tiles at 0.1/0.2/0.4/0.5 m, then evaluates all 16 canonical checkpoint/domain combinations. The three accessible historical comparators add at most six evaluations, separately labelled as exploratory. It samples up to 24 training items at each of three epochs per run. The sensitivity cohort uses one class-centred 50 m patch per class/domain/validation-or-test split where available, at most 168 canonical model–patch tasks (231 including all historical comparators), each with a baseline plus six interventions. These are deliberately bounded probes, not exhaustive coverage of every perturbation or historical training step.

Inference timing includes instrumentation. Do not report it as production throughput or promise a runtime from old evaluation logs. No inference-only collection can establish independent seed variability, train an absent architecture, or guarantee that the next method solves generalization.

## One result to return

At completion, return this archive:

```text
/scratch/m23csa510/project_diagnosis_20261017_results.tar.gz
```

Start with `STATUS.json`, then `MEASUREMENTS_AND_DECISIONS.md` and `diagnostic_results.json`. Detailed stage records preserve the measurements needed for review. The archive excludes checkpoint weights and working raw patches.

Status `complete` means the declared collection completed and the historical matrices matched. `complete_with_reproduction_discrepancies` means collection completed but exact reproduction failed. `failed`, `incomplete`, `interrupted`, `audit_only` and `preflight_only` are not a completed scientific collection.

Rerun the same command with the same output path after an interruption. Completed stages are reused only with matching identities; failed work is retried. This is resume support, not a request to run a succession of new diagnostic scripts.

## Files accompanying the entry point

`test_diagnose_project.py` is the local test suite, not another cluster diagnostic to run. `Adversarial_Review_and_One_Shot_Diagnosis.md` documents corrections, variables and decision boundaries. `validation.json` records what was actually checked locally.
