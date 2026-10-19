# Read-only DALES intensity audit

From the repository's `eclair_model_train` directory, in the existing `e0_env` environment:

```bash
python -B scripts/audit_dales_intensity.py \
  /scratch/m23csa510/dales/dales/train \
  /scratch/m23csa510/dales/dales/test
```

This CPU-only command reads the raw LAS files, not the drop-intensity caches. It
does not train, use a GPU, access checkpoints, open network connections, or write
files. It reads every point's uint16 intensity field directly from the records.
It uses NumPy, already required by the training environment; it does not need laspy.
The full scan reads substantial dataset data: use a CPU allocation if cluster
policy does not permit this work in a login terminal. Any Slurm allocation must
exclude `cn07`; the checker itself does not submit a job.

Output consists of one JSON line per file and a final summary. Confirm all of:

- `complete_for_supplied_roots: true`
- `errors: 0`
- `points_fully_checked` is positive, and file/point coverage matches the dataset
- `nonzero_intensity_points: 0` and `all_intensity_zero: true`

Only then conclude that the supplied files have unpopulated intensity. This is
not a statement about every DALES release or files outside those roots. An
incomplete audit must not be interpreted as proof of all-zero intensity.

Only uncompressed LAS 1.2 point format 1, matching the archived DALES format, is
supported. Other formats, compressed LAZ, truncated files, and missing or empty
input roots are reported as errors (exit code 2). Uppercase extensions are
included, and repeated files are counted only once. An empty LAS file does not
establish all-zero intensity. The checker emits file size and modification time
and rejects a file that changes during its scan.

Run the CPU-only verification suite without writing bytecode:

```bash
python -B -m unittest discover -s tests -p test_audit_dales_intensity.py -v
```

The fixtures in these tests are synthetic. Passing them is not a DALES audit.
