# Frozen mentor comparison: one bounded development pass

## Question and selected mechanism

Can an observed-return student benefit more from a source-selected geometry-input mentor than from equally implemented observed-input mentoring, while meeting the original source and target requirements?

Two fresh students use the same original MinkUNet14C, initialization, observed inputs, source examples, base focal/Lovasz loss and100-epoch schedule. Arm0 has the observed mentor46265/best72; arm1 has the constant-return mentor46266/best68. Both existing mentors were selected from source validation only. Mentors are frozen, evaluated on the student's augmented source coordinates, and absent at deployment. No new inference branch, NAS search, source-label ontology change, or target-data training.

We selected full native-eleven probability guidance to test competing-class information. A proposed binary true-class-only objective was rejected during review because it reduces to supervised sample reweighting. Separating target/non-target information has published precedent, but this source-correct ALS mentor comparison is our project adaptation, not a published validated ALS remedy. (Zhao et al., CVPR2022.)

## Fixed guidance objective

- Mentor forward: FP32, evaluation mode, no gradients. Same source coordinates and pooled features; only the geometry mentor receives the existing valid nonzero constant pair instead of return attributes.
- Match sparse output coordinates exactly before using predictions. Keep eleven native output probabilities.
- A source voxel is eligible only if its GT belongs to common classes1..7 and the mentor's native argmax equals its native GT. Do not merge subtype disagreements into correctness.
- Guidance is full KL(mentor probabilities || student probabilities), temperature1. No `q>p` filter, uncertainty gate or target-fitted selector.
- For each common class present in a rank's batch, sum eligible-point KL and divide by **all valid points of that class**, including zero contributions from ineligible points. Average those class means, then the existing DDP rank/microbatch reduction. This controls tiny eligible-set inflation; it is still a class-balanced auxiliary objective, matched in both arms.
- Add weight0.1, linearly ramped over epochs1..10. This is a fixed engineering screen, not a literature-prescribed ALS optimum. No sweep.
- Source-native eleven-class focal/Lovasz supervision remains unchanged on all valid training labels, including Unassigned/Noise. Validation and deployment use observed returns. Source-native validation selects best; final is secondary.

Correctness filtering avoids imposing an incorrect mentor's hard source label. It does not guarantee preservation or target robustness. A correct mentor may soften a stronger student or teach a source-specific correlation. Mentors differ in competence, learned weights, selected epochs and numerical histories; this estimates mentor-specific value, not an isolated causal feature effect.

## Controls, evidence and decisions

The fresh observed-mentor student controls the auxiliary framework. The original observed46265 remains an additional required benchmark; a weak fresh control cannot earn advancement. The old exact-update failure is not rewritten.

Both arms record per-class valid/eligible counts, mentor entropy, competing-class and ignored-category probability mass, KL, guidance gradient magnitude/direction, fixed mentor state hashes every epoch, source exposure hashes, applied updates/skips, memory and timing. Source-only GPU preflight verifies the objective/gradients, output alignment, mentor immutability, initialization, source-cache parity, checkpoint roundtrip and paired inference. The report flags absent class guidance; permits no hidden temperature/weight changes.

Advancement requires source preservation and target gains against **both** the fresh control and original observed reference: source aggregate losses<=1pp and each common class<=2pp loss; target seven-class and utility gains>=1pp, each class loss<=2pp and both aggregate tile-interval lower bounds>0. Final endpoints must pass the source guard and retain positive target aggregate gains and class guards. One seed/one direction remains development evidence, not independent replication.

- Geometry mentor passes both references: eligible for later reverse-pair consideration; no automatic submission.
- Both mentor arms improve similarly: credit generic guidance, without geometry-specific advancement.
- Candidate beats a weak auxiliary control but fails original-reference requirements: stop.
- Correctly executed negative or uncertain result: stop this fixed mentor recipe; no teacher/weight/temperature/filter sweep and no claim that every feature method was disproved.
- Invalid execution, missing class signal or source-incompetent fresh control: explicitly report the limitation; no automatic retraining.

## Submission

From `eclair_model_train`, with `e0_env` active:

```bash
python -B scripts/submit_mentor_study.py --study eclair_mentor_s1337_v1 --hours 36 --dry-run
python -B scripts/submit_mentor_study.py --study eclair_mentor_s1337_v1 --hours 36
```

One Slurm array contains two tasks, serialized with `%1`. Each task receives two GPUs and its own36-hour limit. Node `cn07` is excluded; automatic Slurm requeue is disabled. The array counts as two training tasks and does not reduce their combined cost.

Root: `/scratch/m23csa510/e0_results/mentor_studies/eclair_mentor_s1337_v1/`.
Logs: `/scratch/m23csa510/e0_logs/mentor_ARRAYID_0.{out,err}` and `_1.{out,err}`.
Task0: `observed_mentor`; task1: `geometry_mentor`. Scheduler may run task1 first; comparison waits for both complete.

Each task runs admission, source preflight, training, all125 ECLAIR and11 DALES tests for source-selected best/final, best-only paired constant-input diagnostics, arm report and compact bundle. The last completed arm also writes `comparison.json`, `comparison.md` and `comparison_results.tar.gz`. No target checkpoint selection or deployment-mode selection.

Code is exported immutably from the committed repository. Hash-pinned reference text is copied into the new study; existing mentor checkpoints/data/caches are read without changes. No duplicate raw caches or mentor checkpoint copies. At least10GiB live quota headroom is required; each arm has a2GiB run cap and128MiB diagnostic cap. Only best/last student checkpoints are retained in these new directories.

Explicit resume, only after review:

```bash
python -B scripts/submit_mentor_study.py --study eclair_mentor_s1337_v1 --resume-arm geometry_mentor --hours 36
python -B scripts/submit_mentor_study.py --study eclair_mentor_s1337_v1 --refresh-admission
```

A14th AMP skip, nonfinite objective, disagreement between rank counters, changed mentor state, changed source exposure, runtime/code mismatch or insufficient storage stops the affected arm. Deadline stopping preserves completed work as `needs_resume`; nothing automatically resubmits.

## References

Zhao, B., Cui, Q., Song, R., Qiu, Y., & Liang, J. (2022). Decoupled knowledge distillation. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*,11953–11962. https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html
