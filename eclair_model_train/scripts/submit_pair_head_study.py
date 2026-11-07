"""Queue one bounded frozen-head allocation using the completed 47054 run."""

import argparse
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess

from submit_architecture_study import (ADMISSION_RESERVE_BYTES, RUN_BUDGET_BYTES,
                                       quota_admission, quota_headroom, resolve_excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.study_dir.resolve()
    scratch = Path(f"/scratch/{getpass.getuser()}").resolve()
    baseline = scratch / "e0_results/coverage_studies/dales_scene_coverage_contents_s1337_v1/coverage_control"
    status = json.loads((baseline / "STATUS.json").read_text())
    if status["status"] != "complete" or status["job_id"] != "47054":
        parser.error("Original 47054 must be complete")
    if not out.is_relative_to(scratch) or out.exists():
        parser.error("Choose a new output directory inside your scratch allocation")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip():
        parser.error("Clean tracked checkout required")
    queued = subprocess.check_output(["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"], text=True).splitlines()
    if len(queued) >= 3 or "pair_head" in queued:
        parser.error("Queue limit reached or the frozen-head job is already queued")
    headroom = quota_headroom(scratch)
    budget = RUN_BUDGET_BYTES + ADMISSION_RESERVE_BYTES
    if min(headroom, shutil.disk_usage(scratch).free) < budget:
        parser.error("Insufficient scratch quota or filesystem space")
    nodes = subprocess.check_output(["sinfo", "-N", "-h", "-o", "%N"], text=True).splitlines()
    excluded = resolve_excluded(nodes)
    logs = scratch / "e0_logs"
    if not logs.is_dir():
        parser.error("Expected existing e0_logs directory")
    env = os.environ.copy()
    env.update(PAIR_HEAD_COMMIT=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               PAIR_HEAD_BASELINE=str(baseline), PAIR_HEAD_STUDY=str(out),
               STUDY_QUOTA_ADMISSION=json.dumps(quota_admission(out, ["pair_head"], headroom)))
    # Reserve this new study name before submission so accidental repeats cannot queue it twice.
    out.mkdir(parents=True, exist_ok=False)
    job = subprocess.check_output(["sbatch", "--parsable", "--no-requeue", "--export=ALL",
                                  f"--exclude={excluded}", "--job-name=pair_head", f"--chdir={root}",
                                  f"--output={logs}/pair_head_%j.out", f"--error={logs}/pair_head_%j.err",
                                  str(root / "scripts/slurm_pair_head_study.sh")], env=env, text=True).strip()
    receipt = {"job_id": job.split(";")[0], "study_dir": str(out / "pair_head"), "excluded": excluded,
               "quota_headroom_bytes": headroom, "commit": env["PAIR_HEAD_COMMIT"]}
    (out / "submission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
