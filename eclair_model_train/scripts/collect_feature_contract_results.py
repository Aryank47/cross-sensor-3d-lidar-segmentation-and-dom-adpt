#!/usr/bin/env python3
"""Package the compact completed evidence; exclude checkpoint and raw data files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.feature_contract_study import ARMS
from src.study_runtime import atomic_json, sha256


def collect(root):
    root = Path(root).resolve()
    for arm in ARMS:
        if (
            json.loads((root / arm / "STATUS.json").read_text()).get("status")
            != "complete"
        ):
            raise ValueError("Collection requires both study completion markers")
    if json.loads((root / "comparison.json").read_text()).get("status") != "complete":
        raise ValueError("Paired report is incomplete")
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or "checkpoints" in path.parts:
            continue
        if path.name == "collection_manifest.json" or path.suffix not in (
            ".json",
            ".jsonl",
            ".csv",
            ".yaml",
            ".md",
            ".txt",
            ".sha256",
        ):
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError("Evidence path escaped the study directory")
        files.append(path)
    if sum(p.stat().st_size for p in files) > 256 * 1024**2:
        raise OSError("Compact evidence exceeds the 256 MiB collection bound")
    manifest = root / "collection_manifest.json"
    atomic_json(
        manifest,
        {
            "files": {str(p.relative_to(root)): sha256(p) for p in files},
            "checkpoints_included": False,
            "raw_point_clouds_included": False,
            "primary_endpoint": "source_selected_best",
        },
    )
    output = root / "results_bundle.tar.gz"
    temporary = root / "results_bundle.tar.gz.tmp"
    with tarfile.open(temporary, "w:gz") as archive:
        for path in [*files, manifest]:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    os.replace(temporary, output)
    return {
        "bundle": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.study_dir)))


if __name__ == "__main__":
    main()
