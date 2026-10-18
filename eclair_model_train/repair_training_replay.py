#!/usr/bin/env python3
"""Repair only the failed training replays; retain the completed diagnosis.

Imports the hash-verified original diagnostic, inserts its missing observer
return, and records the repair separately. No scan or model inference is run.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import logging
from pathlib import Path
import sys
import tarfile
import textwrap
import traceback
import types

ORIGINAL_SCRIPT_SHA256 = "25fb7ca26e259d07765c5178f76c98dbd89b984364014b9ae35146c4576abb5a"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def patch_replay_source(source):
    """Strictly patch the reviewed defect, not arbitrary project code."""
    tree = ast.parse(textwrap.dedent(source))
    observers = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.name == "observe_training_voxel"]
    if len(observers) != 1:
        raise RuntimeError("Expected one reviewed voxel observer")
    observer = observers[0]
    if any(isinstance(n, ast.Return) for n in ast.walk(observer)):
        raise RuntimeError("Observer already has a return: this is not the expected original defect")
    observer.body.append(ast.Return(value=ast.Name(id="vx", ctx=ast.Load())))
    # Preserve a full per-item traceback if an additional problem surfaces.
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value if isinstance(k, ast.Constant) else None for k in node.keys]
            if set(keys) == {"epoch", "index", "error"}:
                node.keys.append(ast.Constant(value="traceback"))
                node.values.append(ast.Call(func=ast.Attribute(value=ast.Name(id="traceback", ctx=ast.Load()),
                    attr="format_exc", ctx=ast.Load()), args=[], keywords=[]))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def verify_original_bundle(root, manifest):
    """The original launcher appends one final log line after manifest creation."""
    errors, append_only = [], []
    resolved = root.resolve()
    for entry in manifest:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(resolved) or not path.is_file():
            errors.append({"path": entry["path"], "reason": "missing or outside original output"})
            continue
        if sha(path) == entry["sha256"]:
            continue
        if entry["path"] == "run.log" and path.stat().st_size >= entry["bytes"]:
            with path.open("rb") as f:
                prefix = hashlib.sha256(f.read(entry["bytes"])).hexdigest()
            if prefix == entry["sha256"]:
                append_only.append(entry["path"])
                continue
        errors.append({"path": entry["path"], "reason": "hash mismatch"})
    return {"entries": len(manifest), "errors": errors, "append_only_logs": append_only}


def load_verified_diagnostic(script, expected_sha):
    if expected_sha != ORIGINAL_SCRIPT_SHA256 or sha(script) != expected_sha:
        raise RuntimeError("Original diagnostic differs from the verified completed run. Keep its diagnose_project.py unchanged for this repair.")
    spec = importlib.util.spec_from_file_location("verified_original_diagnosis", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-out", default="/scratch/m23csa510/project_diagnosis_20261017")
    parser.add_argument("--diagnostic-script", help="Unchanged original diagnose_project.py used for the run")
    args = parser.parse_args()
    parent = Path(args.parent_out).resolve()
    repair = parent / "training_replay_repair_v1"
    repair.mkdir(parents=True, exist_ok=True)
    old_spec = json.loads((parent / "run_specification.json").read_text())
    old_status = json.loads((parent / "STATUS.json").read_text())
    old_results = json.loads((parent / "diagnostic_results.json").read_text())
    manifest = json.loads((parent / "bundle_manifest.json").read_text())
    script = Path(args.diagnostic_script) if args.diagnostic_script else Path(old_spec["arguments"]["repo"]) / "diagnose_project.py"
    if not script.is_file() and not args.diagnostic_script:
        script = Path(old_spec["arguments"]["repo"]) / "one_shot_diagnosis_20261017" / "diagnose_project.py"
    diag = load_verified_diagnostic(script, old_spec["script_sha256"])
    verification = verify_original_bundle(parent, manifest)
    diag.write_json(repair / "original_bundle_verification.json", verification)
    if verification["errors"]:
        raise RuntimeError("Original output changed; see original_bundle_verification.json. No replay started.")
    failures = old_status.get("failures", [])
    if not failures or any(not f.get("stage", "").startswith("training_replay_") for f in failures):
        raise RuntimeError("This targeted repair expects only training-replay failures")
    if not all(r["complete_coverage"] and r["exact_historical_confusion_match"] for r in old_results["evaluations"]):
        raise RuntimeError("Canonical inference must already be complete and reproduced")
    run_ids = {f["stage"].removeprefix("training_replay_") for f in failures}
    original_source = inspect.getsource(diag.Runner.replay_training)
    fixed_source = patch_replay_source(original_source)
    fixed_file = repair / "fixed_replay_training.py"
    fixed_file.write_text(fixed_source, encoding="utf-8")
    namespace = {}
    exec(compile(fixed_source, str(fixed_file), "exec"), diag.__dict__, namespace)
    options = copy.deepcopy(old_spec["arguments"])
    options["out"] = str(repair)
    runner = diag.Runner(argparse.Namespace(**options))
    success = False
    repair_results = []
    caught = None
    try:
        runner.prepare()  # Identity checks only: no raw scan, loader item, or inference.
        differences = []
        for field in ("source_hashes", "mapping_hashes", "label_maps", "datasets", "runs"):
            if diag.clean_json(runner.spec.get(field)) != old_spec.get(field):
                differences.append(field)
        for field in ("torch", "numpy"):
            if runner.spec.get("runtime", {}).get(field) != old_spec.get("runtime", {}).get(field):
                differences.append("runtime." + field)
        if differences:
            raise RuntimeError("Inputs differ from the completed run: " + ", ".join(differences))
        metadata = {"parent_signature": json.loads((parent / "preflight.json").read_text())["signature"],
            "original_script_sha256": old_spec["script_sha256"],
            "repair_program_sha256": sha(__file__), "fixed_function_sha256": sha(fixed_file),
            "fix": "Return the original voxelizer result from the observer; add per-item tracebacks.",
            "scope": "Only failed training-data replays; completed scans/inference/probes are retained.",
            "input_identity_checks_passed": True, "run_ids": sorted(run_ids)}
        runner.spec["repair"] = metadata
        runner.signature = diag.digest_obj(runner.spec)
        diag.write_json(repair / "run_specification.json", runner.spec)
        diag.write_json(repair / "repair_metadata.json", metadata)
        runner.replay_training = types.MethodType(namespace["replay_training"], runner)
        for run in runner.runs:
            if run["run"] in run_ids:
                result = runner.task("training_replay_" + run["run"], lambda r=run: runner.replay_training(r))
                if result is not None:
                    repair_results.append(result)
        success = len(repair_results) == len(run_ids) and not runner.failures
        repaired_status = {"status": "complete" if success else "failed", "scope": "training_replay_repair_only",
            "events": runner.events, "failures": runner.failures, "completed_runs": len(repair_results),
            "expected_runs": len(run_ids), "repair": metadata}
        diag.write_json(repair / "STATUS.json", repaired_status)
        results = copy.deepcopy(old_results)
        if success:
            results["training_exposure"] = diag.summarize_diagnostics([], [], repair_results, {})["training_exposure"]
            results["training_replay_repair"] = metadata
            diag.write_json(repair / "consolidated_diagnostic_results.json", results)
            diag.write_json(repair / "consolidated_STATUS.json", {
                "status": "complete_with_training_replay_repair", "original_status": "failed",
                "original_failures_resolved": sorted(run_ids), "remaining_failures": [],
                "canonical_confusion_matrices_reproduced": 16,
                "note": "Original failure records remain preserved. New replay records have a separate repair signature."})
            (repair / "consolidated_READ_RESULTS.md").write_text(
                "# Repaired diagnostic collection\n\n"
                "The original dataset scans, 22 model/domain evaluations and 231 probes are retained.\n"
                "Only the failed training replays were rerun after correcting the observer return.\n"
                "Read root STATUS.json and diagnostic_results.json for consolidated results.\n"
                "Original status/results are preserved under original_run/. Repair provenance,\n"
                "per-item data and a separate signature are under training_replay_repair_v1/.\n", encoding="utf-8")
    except BaseException as exc:
        caught = exc
        diag.write_json(repair / "STATUS.json", {"status": "failed", "error": str(exc),
            "traceback": traceback.format_exc(), "events": runner.events, "failures": runner.failures})
        runner.log.exception("Targeted repair failed; original measurements remain unchanged")
    finally:
        for h in list(runner.log.handlers):
            h.flush()
            h.close()
            runner.log.removeHandler(h)
    output = parent.parent / (parent.name + "_repaired_results.tar.gz")
    replacements = {"STATUS.json": repair / "consolidated_STATUS.json",
                    "diagnostic_results.json": repair / "consolidated_diagnostic_results.json",
                    "READ_RESULTS.md": repair / "consolidated_READ_RESULTS.md"} if success else {}
    entries = []
    files = []
    for p in parent.rglob("*"):
        if not p.is_file() or "working_patches" in p.parts or "__pycache__" in p.parts or p.name.endswith((".tmp", ".gz")) or p.name == "repaired_bundle_manifest.json":
            continue
        name = p.relative_to(parent).as_posix()
        if name == "bundle_manifest.json":
            files.append((p, "original_run/bundle_manifest.json"))
        elif name in ("STATUS.json", "diagnostic_results.json", "READ_RESULTS.md"):
            files.append((p, "original_run/" + name))
            files.append((replacements.get(name, p), name))
        else:
            files.append((p, name))
    for p, name in files:
        entries.append({"path": name, "sha256": sha(p), "bytes": p.stat().st_size})
    bundle_manifest = repair / "repaired_bundle_manifest.json"
    bundle_manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    with tarfile.open(output, "w:gz") as tar:
        for p, name in files:
            tar.add(p, arcname=name)
        tar.add(bundle_manifest, arcname="bundle_manifest.json")
    print("Targeted replay repair:", "complete" if success else "failed", flush=True)
    print("Result archive:", output, flush=True)
    if caught:
        print(str(caught), file=sys.stderr)
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
