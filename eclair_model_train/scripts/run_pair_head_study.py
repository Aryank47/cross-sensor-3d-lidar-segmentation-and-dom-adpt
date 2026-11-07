"""Fit and screen two native DALES classifiers on one frozen 47054 backbone."""

import argparse
import copy
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 1337
ARM = "pair_head"
BASELINE_MODEL = "6433a22d614a68d274b4f2a4449abd37003b700db2381f826ce61e4136a04b05"
FIT_EPOCHS = 8
LEARNING_RATE = 0.0003
MARGIN = 1.0
PAIR_WEIGHT = 0.1
PAIR_LIMIT = 32
CLASS_CROP_LIMIT = 128
BATCH_ROWS = 4096
# Native training IDs for ground, vegetation, building, pole and wire.
PAIRS = ((0, 4), (1, 4), (1, 5), (6, 5))


def pair_loss(logits, labels, initial_logits):
    import torch

    terms = []
    counts = []
    for first, second in PAIRS:
        for actual, rival in ((first, second), (second, first)):
            gap = initial_logits[:, actual] - initial_logits[:, rival]
            rows = torch.where((labels == actual) & (gap < MARGIN))[0]
            rows = rows[torch.argsort(gap[rows], stable=True)[:PAIR_LIMIT]]
            counts.append(rows.numel())
            if rows.numel():
                terms.append((MARGIN + logits[rows, rival] - logits[rows, actual]).relu().mean())
    return (torch.stack(terms).mean() if terms else logits.sum() * 0), counts


def contrasts(metrics):
    cm = metrics["confusion_matrix"]
    support = metrics["support"]
    if min(support[1:]) == 0:
        raise ValueError("Every common class must have held-out support")
    return {
        "background_to_building": (cm[1][3] + cm[2][3]) / (support[1] + support[2]),
        "building_recall": metrics["recall"]["buildings"],
        "pole_to_vegetation": cm[5][2] / support[5],
        "pole_recall": metrics["recall"]["poles"],
        "wire_to_pole": cm[4][5] / support[4],
    }


def gate(candidate, reference, *, target):
    c, r = contrasts(candidate), contrasts(reference)
    aggregate_gain = 0.01 if target else -0.005
    class_floor = 0.02 if target else 0.01
    return {
        "background_to_building_down": c["background_to_building"] < r["background_to_building"],
        "building_recall_preserved": c["building_recall"] >= r["building_recall"] - 0.005,
        "pole_to_vegetation_down": c["pole_to_vegetation"] < r["pole_to_vegetation"],
        "pole_recall_up": c["pole_recall"] >= r["pole_recall"] + 0.005,
        "wire_to_pole_no_increase": c["wire_to_pole"] <= r["wire_to_pole"],
        "wire_iou_preserved": candidate["per_class_iou"]["wires"] >= reference["per_class_iou"]["wires"] - 0.005,
        "all_classes_preserved": all(candidate["per_class_iou"][k] >= v - class_floor
                                    for k, v in reference["per_class_iou"].items()),
        **{key: candidate[key] >= reference[key] + aggregate_gain
           for key in ("all_seven_miou", "utility_miou")},
    }


def collect_features(model, cfg, baseline, device):
    import json
    import MinkowskiEngine as ME
    import torch
    from src.dist import DistEnv
    from train import build_dataloaders, _labels_in_output_order

    loader, val_loader, _ = build_dataloaders(cfg, DistEnv(False))
    dataset = loader.dataset
    if dataset.write_cache or not dataset.require_cache or len(loader) != 312:
        raise ValueError("Expected 312 original read-only DALES training crops")
    catalog = json.loads(Path(cfg["data"]["sampling"]["coverage_catalog"]).read_text())
    files = {str(Path(x).resolve()) for x in dataset.files}
    if files != {str(Path(x["path"]).resolve()) for x in catalog["inventory"]["members"]}:
        raise ValueError("Training members differ from the original control catalogue")
    if files & {str(Path(x).resolve()) for x in val_loader.dataset.files}:
        raise ValueError("Train/validation leakage")
    if json.loads((baseline / "STATUS.json").read_text())["job_id"] != "47054":
        raise ValueError("Wrong source job")
    dataset.set_epoch(1)
    loader.generator.manual_seed(SEED + 1009)
    rng = torch.Generator().manual_seed(SEED)
    features, labels, captured = [], [], []
    hook = model.final.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    try:
        with torch.no_grad():
            for batch in loader:
                coords, y = batch["coords"].to(device), batch["labels"].to(device)
                sparse = ME.SparseTensor(batch["feats"].to(device), coordinates=coords, device=device)
                output = model(sparse)
                if len(captured) != 1:
                    raise RuntimeError("Missing or repeated final-layer input")
                x = captured.pop()
                if not torch.equal(x.C, output.C):
                    raise RuntimeError("Final classifier changed sparse coordinate order")
                if not torch.equal(coords.cpu(), x.C.cpu()):
                    y, coverage = _labels_in_output_order(coords, y, x.C)
                    if coverage != 1.0:
                        raise RuntimeError("Incomplete native labels for classifier features")
                y = y.cpu()
                indices = []
                for label in range(8):
                    choices = torch.where(y == label)[0]
                    if choices.numel() > CLASS_CROP_LIMIT:
                        choices = choices[torch.randperm(choices.numel(), generator=rng)[:CLASS_CROP_LIMIT]]
                    indices.append(choices)
                ids = torch.cat(indices)
                features.append(x.F.index_select(0, ids.to(device)).float().cpu())
                labels.append(y[ids])
                if len(features) == 1:
                    scores = output.F.index_select(0, ids.to(device)).float()
                    kernel = model.final.kernel.reshape(-1, 8)
                    torch.testing.assert_close(features[-1].to(device) @ kernel + model.final.bias,
                                               scores, atol=0.0002, rtol=0.0001)
                if len(features) % 26 == 0:
                    print(f"Frozen features: {len(features)}/312 DALES crops", flush=True)
    finally:
        hook.remove()
    return torch.cat(features), torch.cat(labels)


def fit(model, cfg, baseline, device):
    import torch
    from train import build_loss

    features, labels = collect_features(model, cfg, baseline, device)
    counts = labels.bincount(minlength=8).tolist()
    if min(counts) < 128 or not features.isfinite().all():
        raise ValueError("Insufficient or invalid classifier fitting examples")
    initial = torch.nn.Linear(features.shape[1], 8, device=device)
    with torch.no_grad():
        initial.weight.copy_(model.final.kernel.reshape(-1, 8).T)
        initial.bias.copy_(model.final.bias.reshape(8))
    initial.requires_grad_(False)
    heads = [copy.deepcopy(initial).requires_grad_(True) for _ in range(2)]
    optimizers = [torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=0.0) for head in heads]
    criterion = build_loss(cfg).to(device)
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(200)  # Keep the reference's fully ramped Lovasz weight in both arms.
    rng = torch.Generator().manual_seed(SEED)
    audit = []
    for epoch in range(1, FIT_EPOCHS + 1):
        pair_counts = [0] * 8
        totals = [0.0, 0.0, 0.0]
        for ids in torch.randperm(len(labels), generator=rng).split(BATCH_ROWS):
            x, y = features[ids].to(device), labels[ids].to(device)
            for index, (head, optimizer) in enumerate(zip(heads, optimizers, strict=True)):
                optimizer.zero_grad(set_to_none=True)
                logits = head(x)
                loss = criterion(logits, y)
                totals[index] += float(loss.detach())
                if index == 1:
                    margin, used = pair_loss(logits, y, initial(x))
                    pair_counts = [a + b for a, b in zip(pair_counts, used, strict=True)]
                    loss = loss + PAIR_WEIGHT * margin
                    totals[2] += float(margin.detach())
                if not loss.isfinite():
                    raise RuntimeError("Nonfinite classifier objective")
                loss.backward()
                optimizer.step()
        audit.append({"epoch": epoch, "sum_batch_losses": totals, "directed_pair_examples": pair_counts})
        print(f"Head fit {epoch}/{FIT_EPOCHS}: directed examples {pair_counts}", flush=True)
    return [h.state_dict() for h in (initial, *heads)], counts, audit


def run(args, out, status):
    import json
    import torch
    from run_architecture_study import model_digest
    from src.model import build_model
    from src.study_analysis import evaluate_tiles
    from src.study_runtime import atomic_json, sha256
    from submit_architecture_study import RUN_BUDGET_BYTES, allocation_quota_admission

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != os.environ["PAIR_HEAD_COMMIT"] or subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "train.py", "src", "scripts", "configs"], check=False,
    ).returncode:
        raise RuntimeError("Study requires the clean submitted commit")
    if socket.gethostname().split(".")[0].lower() == "cn07":
        raise RuntimeError("Excluded node cn07")
    atomic_json(out / "storage_admission.json", allocation_quota_admission(
        out, RUN_BUDGET_BYTES, allowed_arms=(ARM,),
    ))
    baseline = args.baseline.resolve()
    manifest = json.loads((baseline / "evaluation_manifest.json").read_text())
    if (manifest["checkpoints"]["best"]["model_sha256"] != BASELINE_MODEL
            or manifest["checkpoints"]["best"]["epoch"] != 200):
        raise ValueError("Unexpected checkpoint provenance")
    path = baseline / "checkpoints" / "best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if model_digest(checkpoint["model_state"]) != BASELINE_MODEL:
        raise ValueError("Reference model digest mismatch")
    cfg = json.loads((baseline / "config_resolved.json").read_text())
    if cfg["data"]["sampling"].get("coverage_resample") is not False:
        raise ValueError("Expected the untreated 47054 crop loader")
    (out / "config_resolved.yaml").write_text(json.dumps(cfg, indent=2))  # JSON is valid YAML.
    atomic_json(out / "protocol.json", {"commit": commit, "checkpoint_file_sha256": sha256(path),
                "seed": SEED, "fit_epochs": FIT_EPOCHS, "learning_rate": LEARNING_RATE,
                "pair_weight": PAIR_WEIGHT, "margin": MARGIN, "pairs_native": PAIRS,
                "pair_direction_cap_per_batch": PAIR_LIMIT, "class_crop_cap": CLASS_CROP_LIMIT,
                "batch_rows": BATCH_ROWS, "fixed_final_head": True})
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    model = build_model(10, 8, cfg=cfg).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    status("fitting")
    states, counts, audit = fit(model, cfg, baseline, device)
    atomic_json(out / "fit_audit.json", {"native_label_counts": counts, "epochs": audit})
    if any(count < 128 for count in audit[0]["directed_pair_examples"]):
        status("complete", decision="insufficient_source_pair_support", eclair_evaluated=False)
        return
    torch.save([{key: value.cpu() for key, value in head.items()} for head in states], out / "head_weights.pt")

    def backbone_digest():
        return model_digest({k: v for k, v in model.state_dict().items() if not k.startswith("final.")})

    frozen = backbone_digest()
    if model_digest(model.state_dict()) != BASELINE_MODEL:
        raise RuntimeError("Feature extraction changed the frozen model")
    logger = logging.getLogger(ARM)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    deadline = float(os.environ["PAIR_HEAD_DEADLINE_UNIX"])
    names = ("original", "refit", "pair_margin")

    def evaluate(domain, split):
        result = {}
        for name, state in zip(names, states, strict=True):
            status("evaluating", domain=domain, split=split, head=name)
            with torch.no_grad():
                model.final.kernel.copy_(state["weight"].T.reshape_as(model.final.kernel))
                model.final.bias.copy_(state["bias"].reshape_as(model.final.bias))
            stage = out / f"{domain}_{split}" / name
            ident = model_digest(model.state_dict())
            if not evaluate_tiles(model, cfg, ident, domain, stage, deadline, logger, split=split):
                raise TimeoutError(f"Incomplete {domain} {split} for {name}")
            if backbone_digest() != frozen:
                raise RuntimeError("Evaluation changed frozen backbone parameters or buffers")
            result[name] = json.loads((stage / domain / "metrics.json").read_text())
            expected_tiles = {("dales", "val"): 3, ("dales", "test"): 11, ("eclair", "test"): 125}
            if len(result[name]["tiles"]) != expected_tiles[domain, split]:
                raise RuntimeError("Incomplete tile set")
        if split == "test":
            saved = json.loads((baseline / "evaluation" / BASELINE_MODEL / domain / "metrics.json").read_text())
            keys = ("dataset", "features", "patch", "voxelization", "window_m", "stride_m", "aggregation",
                    "amp", "source_mapping_sha256", "target_mapping_sha256", "coverage_catalog_sha256")
            if any(result["original"]["contract"][k] != saved["contract"][k] for k in keys):
                raise RuntimeError(f"Original {domain} evaluation contract mismatch")
            if result["original"]["confusion_matrix"] != saved["confusion_matrix"]:
                raise RuntimeError(f"Original {domain} canonical matrix parity failed")
        return result

    validation = evaluate("dales", "val")
    source_checks = {name: gate(validation["pair_margin"], validation[name], target=False)
                     for name in names[:2]}
    atomic_json(out / "source_gate.json", {"checks": source_checks,
                "error_rates": {name: contrasts(m) for name, m in validation.items()}})
    if not all(all(checks.values()) for checks in source_checks.values()):
        status("complete", decision="stop_pair_margin_source_gate", eclair_evaluated=False)
        return
    source = evaluate("dales", "test")
    target = evaluate("eclair", "test")
    target_checks = {name: gate(target["pair_margin"], target[name], target=True) for name in names[:2]}
    source_preserved = all(source["pair_margin"][key] >= source["original"][key] - 0.01
                           for key in ("all_seven_miou", "utility_miou")) and all(
        source["pair_margin"]["per_class_iou"][k] >= v - 0.02
        for k, v in source["original"]["per_class_iou"].items()
    )
    advance = source_preserved and all(all(checks.values()) for checks in target_checks.values())
    atomic_json(out / "test_decision.json", {"target_checks": target_checks,
                "source_test_preserved": source_preserved, "advance_to_confirmation": advance,
                "error_rates": {name: contrasts(m) for name, m in target.items()},
                "original_canonical_matrix_parity": True})
    status("complete", decision="confirm_readout" if advance else "close_tested_readout", eclair_evaluated=True)


def main():
    from src.study_runtime import atomic_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.study_dir.resolve() / ARM
    out.mkdir(parents=True, exist_ok=False)

    def status(state, **fields):
        atomic_json(out / "STATUS.json", {"status": state, "job_id": os.environ.get("SLURM_JOB_ID"),
                    "updated_unix": time.time(), **fields})

    status("preflight")
    try:
        run(args, out, status)
    except Exception as exc:
        status("incomplete" if isinstance(exc, TimeoutError) else "failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
