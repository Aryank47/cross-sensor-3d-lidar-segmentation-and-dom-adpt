#!/usr/bin/env python3
"""Allocated, exclusive, resumable runner. No job submission or model training."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.submit_architecture_study import quota_headroom  # noqa: E402
from src.transfer_evidence import ASSETS, read_json, run_study, safe_output  # noqa: E402


def validate_admission(receipt, output, now=None):
    cfg = read_json(ASSETS / "study.json")
    age = (time.time() if now is None else now) - float(receipt["checked_at_unix"])
    if not (
        receipt["version"] == 1
        and receipt["user"] == getpass.getuser()
        and Path(receipt["output_dir"]).resolve() == Path(output).resolve()
        and -300 <= age <= 48 * 3600
        and receipt["required_bytes"] == cfg["minimum_quota_headroom_bytes"]
        and receipt["headroom_bytes"] >= receipt["required_bytes"]
        and receipt["budget_bytes"] == cfg["output_budget_bytes"]
    ):
        raise ValueError("Missing, stale or insufficient login-node user-quota admission")


def main():
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = safe_output(args.output)
    receipt = json.loads(os.environ["TRANSFER_QUOTA_ADMISSION"])
    validate_admission(receipt, output)
    try:
        live = quota_headroom(output)
        mode = "live_compute_quota"
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        live, mode = None, "login_admission_snapshot_compute_query_unavailable"
        print(
            f"Compute quota query unavailable ({exc}); using validated login snapshot, not a reservation.", flush=True
        )
    if live is not None and live < receipt["required_bytes"]:
        raise OSError("Live user quota is below the admitted 10 GiB headroom")
    output.mkdir(parents=True, exist_ok=True)
    # Lock is inside the new study only. No other process may resume/write concurrently.
    with (output / "logs/runner.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        admission = {
            "mode": mode,
            "live_quota_headroom_bytes": live,
            "login_snapshot": receipt,
            "existing_results_deleted": False,
            "monitor_restarted": False,
        }
        # Keep admission under logs so first-run manifest binding sees no unowned artifacts.
        (output / "logs/storage_admission.json").write_text(json.dumps(admission, indent=2))
        run_study(output, resume=args.resume, deadline=float(os.environ["TRANSFER_DEADLINE_UNIX"]))


if __name__ == "__main__":
    main()
