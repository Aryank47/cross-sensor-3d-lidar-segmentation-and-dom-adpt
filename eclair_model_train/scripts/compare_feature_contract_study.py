#!/usr/bin/env python3
"""Run automatically after each arm; compare only fully completed paired studies."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.feature_contract_study import compare_study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", required=True)
    args = parser.parse_args()
    root = Path(args.study_dir).resolve()
    with (root / ".comparison.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = compare_study(root)
        print(json.dumps(result), flush=True)
        if result["status"] == "complete":
            from scripts.collect_feature_contract_results import collect

            print(json.dumps(collect(root)), flush=True)


if __name__ == "__main__":
    main()
