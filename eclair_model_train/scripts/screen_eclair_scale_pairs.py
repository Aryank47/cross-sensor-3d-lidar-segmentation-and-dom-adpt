"""Source-only scale selection, followed by one locked ECLAIR check when admissible."""

import json
from pathlib import Path

import numpy as np

import run_scale_signal_probe as probe
from neighbor_context_coverage import MANIFEST_HASH


WIDE = (4, 8, 16)
FEATURES = ['count', 'eigenvalue_middle', 'linearity', 'planarity',
            'principal_vertical', 'z_extent', 'edge_truncated']


def admissible(support):
    return all(route["passed"] for route in support)


def align_sources(banks):
    keys = {r: set(zip(bank["tile"], bank["ids"])) for r, bank in banks.items()}
    common = set.intersection(*keys.values())
    aligned = {}
    for r, bank in banks.items():
        keep = np.fromiter(((tile, pt) in common for tile, pt in zip(bank["tile"], bank["ids"])),
                           dtype=bool, count=len(bank["ids"]))
        aligned[r] = {k: v[keep] if isinstance(v, np.ndarray) else v for k, v in bank.items()}
    for bank in aligned.values():
        for key in ("tile", "ids", "y", "native", "w", "common"):
            if not np.array_equal(bank[key], aligned[WIDE[0]][key]):
                raise ValueError(f"Shared source cohorts disagree on {key}")
    return aligned


def report(bank, predictions):
    measures = {name: probe.measure(bank, value) for name, value in predictions.items()}
    return {
        "comparisons": probe.compare(measures),
        "arms": {
            name: {**detail, "balanced_error_pct": float(probe.score(cells.sum(0))),
                   "route_error_and_support_by_tile": cells.tolist(),
                   "confusion_by_tile": matrix.tolist()}
            for name, (cells, matrix, detail) in measures.items()
        },
    }


def passes(comparisons, target=False):
    single = ("local", "wide")
    useful = all(row["gain_pp"] >= probe.SPEC["minimum_gain_pp"] for row in comparisons.values())
    stable = all(comparisons[k]["leave_one_tile_gain_range"][0] > 0 for k in single)
    guards = all(ok for k in single for _, ok in comparisons[k]["guards"].values())
    if target:
        stable &= all(comparisons[k]["valid_bootstrap_fraction"] >= probe.SPEC["min_valid_resample_fraction"]
                      and comparisons[k]["paired_95"][0] is not None
                      and comparisons[k]["paired_95"][0] > 0 for k in single)
    return {"minimum_gain": useful, "tile_stability": stable, "class_guards": guards}


def safe_json(value):
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    probe.sanity_check()
    archive = probe.ROOT
    path = archive / "study_manifest.json"
    if probe.sha(path) != MANIFEST_HASH:
        raise ValueError("Frozen manifest changed")
    manifest, schema = probe.read(path), probe.read(archive / "measurement_schema.json")
    status = probe.read(archive / "STATUS.json")
    if status["identity"] != manifest["identity"] or not status["all_required_stages_complete"]:
        raise ValueError("Canonical archive is incomplete")
    radii = schema["radii_m"]
    if [schema["geometry_variables"][i] for i in (0, 2, 4, 5, 7, 9, 12)] != FEATURES:
        raise ValueError("Unexpected geometry schema")
    catalog = Path(__file__).resolve().parents[1] / "configs/transfer_evidence/scan_catalog.json"
    if (archive / "scan_catalog.json").exists():
        catalog = archive / "scan_catalog.json"
    if probe.sha(catalog) != "319a400c63d34a84449f3dd244f0b3836337b57dcf233159e06539dca300320d":
        raise ValueError("Source support catalog differs from the frozen study")

    def load(domain, split, r):
        bank = probe.bank(domain, split, manifest, radius_indices=(radii.index(2), radii.index(r)),
                          local_nuisance_only=True, exact_tile_weights=True)
        if len(bank["inventory"]) != (26 if domain == "dales" else 125):
            raise ValueError("Unexpected tile inventory")
        if domain == "eclair":
            bank["full_test_support"] = [probe.read(archive / f"test/eclair_{i:04d}.json")["models"]["dales_29364"]["support"]
                                         for i in range(125)]
        return bank

    source = align_sources({r: load("dales", "train", r) for r in WIDE})
    folds = np.array_split(np.random.default_rng(probe.SEED).permutation(26), probe.SPEC["folds"])
    source_support = probe.support(source[WIDE[0]])
    result = {
        "manifest_sha256": MANIFEST_HASH, "radii_m": [2, *WIDE], "source_tile_folds": [fold.tolist() for fold in folds],
        "source_common_support": source_support,
        "screen": {k: v for k, v in probe.SPEC.items() if k not in
                   ("radii_m", "common_inputs", "coverage_fallback", "fit_weighting")},
        "inputs_shared_by_all_arms": ["log_2m_point_count", "return_number", "number_of_returns"],
        "sampling": "Primary sample only. Inverse inclusion weights computed from tile-class quotas and raw population.",
        "source_selection": "Shared source cohort across all radii. Rank by smaller gain over either single view, then shorter radius.",
        "target_access": "Only the first source-qualified pair with adequate target support is scored.",
        "source": {},
    }
    if not admissible(source_support):
        result["decision"] = "unresolved_shared_source_support"
        print(json.dumps(safe_json(result), allow_nan=False))
        return

    ranked = []
    for r, bank in source.items():
        predictions = {}
        for name, slots in probe.ARMS.items():
            out = np.full((len(bank["y"]), 7), np.nan)
            for fold in folds:
                held = np.isin(bank["tile"], fold)
                if held.any():
                    out[held], = probe.fit_predict(bank, ~held, [(bank, held)], slots)
            if not np.isfinite(out).all():
                raise ValueError("Incomplete out-of-fold predictions")
            predictions[name] = out
        analysis = report(bank, predictions)
        checks = passes(analysis["comparisons"])
        result["source"][r] = {**analysis, "checks": checks}
        if all(checks.values()):
            ranked.append((-min(analysis["comparisons"][k]["gain_pp"] for k in ("local", "wide")), r))
        print(f"Completed DALES 2 m + {r} m", flush=True)
    result["ranked_source_qualified"] = [r for _, r in sorted(ranked)]
    result["target_coverage"] = {}
    result["decision"] = "no_source_pair_qualified" if not ranked else "unresolved_target_support"
    for _, r in sorted(ranked):
        target = load("eclair", "test", r)
        coverage = probe.support(target)
        result["target_coverage"][r] = {"support": coverage, "inventory": target["inventory"]}
        if not admissible(coverage):
            continue
        result["locked_pair_m"] = [2, r]
        print(f"Locked 2 m + {r} m before ECLAIR scoring", flush=True)
        train = source[r]
        predictions = {}
        for name, slots in probe.ARMS.items():
            predictions[name], = probe.fit_predict(train, slice(None), [(target, slice(None))], slots)
        result["target"] = report(target, predictions)
        checks = passes(result["target"]["comparisons"], target=True)
        result["target"]["checks"] = checks
        result["decision"] = "advance_to_matched_training_review" if all(checks.values()) else "locked_pair_did_not_advance"
        break
    print(json.dumps(safe_json(result), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
