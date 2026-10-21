# Matched classifier control

## Question

Can an alternative classifier, already fitted using DALES training features and selected using DALES validation,
improve full-test transfer while the learned backbone stays fixed?

The completed architecture study saved this classifier, but its diagnostic scores used a smaller point sample than
the original classifier's scores. Those scores are not a valid direct comparison. This evaluation applies the saved
classifier to the original canonical DALES/ECLAIR test sets and compares it with the stored original-classifier results.

This is an evaluation-only control. It does not train a backbone, refit a classifier, search target labels, or isolate
the effect of focal/Lovasz loss. Source fitting used bounded, class-capped feature cohorts, so a negative result
cannot establish that every classifier repair would fail.

## Run

From the repository's `eclair_model_train` directory in `e0_env` on the login node:

```bash
python -B scripts/submit_readout_control.py \
  --study-dir /scratch/m23csa510/e0_results/architecture_studies/dales_arch_s1337_qfix
```

The submitter checks the original artifacts, code hashes, quota and queue, then submits **one Slurm job**, one GPU,
12-hour maximum by default, with **cn07 excluded**. Evaluation runs sequentially inside that allocation, not on the
login node. `--check-only` performs admission checks without submitting or writing output. `--hours 24` requests
a longer allocation; the allowed range is 1–36 hours. No automatic requeue or reconnect is used.

Default endpoints are source-selected best (primary) and final (secondary), for reference and context. Context best
and final have identical weights and are evaluated once, making **three unique models and six domain evaluations**.
The stored original-classifier full-test results are reused, rather than rerunning those models unnecessarily.

For a job that reports `needs_resume`, submit the same command with `--resume`. Only tiles with matching protocol
signatures are reused. Input or evaluator changes invalidate resume. A failed job requires inspecting its error first.

## Integrity and storage

- Original study directories, checkpoints, saved heads and caches are inputs only. No original metrics/status are changed.
- Default output is a new `readout_control_v1` folder beside the reference/context arms. Other output locations can
  be supplied with `--output-dir`; existing arm folders and their ancestors are protected.
- Original source files must match their archived manifest. Dataset membership, raw/cache identity, mappings, features,
  quantization, 120 m test window/stride, FP32 inference and mean-logit aggregation must match the original protocol.
- The eight native classifier outputs are retained. Cars/trucks are merged only after native argmax, as in the original
  evaluator. Common-seven IoU is computed from the resulting confusion matrix, not by averaging native class IoUs.
- The GPU smoke check verifies the saved standardized classifier projection and unchanged backbone features/buffers
  on a bounded DALES validation patch before full-test inference. The backbone is frozen and kept in evaluation mode.
- Output is capped at **128 MiB**, with a **512 MiB** free-space reserve. Login admission checks user quota. If compute
  quota cannot resolve the network mount, a matching login receipt less than 48 hours old is used and recorded;
  it is an observation, not a reservation. Filesystem headroom and the output cap remain enforced.
- Only compact JSON statistics are saved. No new checkpoints, prediction clouds, feature banks or caches are created.

## Outputs and interpretation

`STATUS.json` distinguishes `complete`, `needs_resume` and `failed`; Slurm exit alone is not completion.
`progress.json` identifies the active model/domain. Completed tile JSONs also show inference progress.

`comparison.json` indexes per-model comparisons; each model's comparison contains:

- Original versus replacement common-seven mIoU, wire/pole mean and all seven per-class IoUs, with percentage-point deltas.
- Precision, recall, confusion matrices, GT support and pole/tower subtype recall, for both DALES and ECLAIR.
- Checks that both classifiers were scored on the same protocol, tiles and class/subtype support.
- The existing conservative balanced screen: target overall and utility improve, no target class declines, and source
  overall is preserved. This is an observed screen, not a significance test. Individual class changes remain visible.

If balanced transfer improves with the backbone fixed, prioritize how class decisions are learned in the next
controlled training experiment. It does not prove that representation issues are solved or identify one loss as the cause.
If it fails, this particular source-trained classifier replacement is insufficient. Do not infer that the backbone is
incapable, or automatically launch a loss sweep or more depth variants. Interpret source and target class tradeoffs
before committing the next full-training slot. Best-checkpoint results stay primary; final results are a sensitivity check.

## Local validation

```bash
python -B -m unittest discover -s tests -p test_readout_control.py -v
bash -n scripts/slurm_readout_control.sh
```

CPU tests exercise projection, source-only selection, original-code/input guards, exact support/protocol matching,
endpoint deduplication, storage admission, resume identity and class-tradeoff rejection. CUDA/Minkowski execution
is checked by the in-allocation smoke test; CPU tests do not validate the cluster GPU runtime.
