"""Resumable point-level evaluation and bounded, source-selected feature probes.

No target labels fit a head or select a checkpoint. Only compact statistics are
persisted. Full inference reuses the canonical point/voxel projection routine.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .study_metrics import confusion, summary
from .study_runtime import atomic_json, sha256, storage_guard


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def load_dataset(domain, split, source_cfg, logger):
    from . import eval_cross_common as ev

    # Use archived canonical target split/configuration, never the study arm's labels.
    run = "e0_dales_train_29364" if domain == "dales" else "e0_eclair_repro_29306"
    path = Path(f"/scratch/m23csa510/e0_results/{run}/config_resolved.json")
    cfg = ev.load_config_any(path)
    manifest = ev.get_dales_manifest_for_split(cfg, split) if domain == "dales" else None
    if domain == "dales" and not manifest:
        raise RuntimeError(f"Exact DALES {split} manifest required; no inferred split")
    ds = ev.build_target_dataset_one_split(
        target_domain=domain,
        target_cfg=cfg,
        source_cfg=source_cfg,
        split=split,
        target_manifest=manifest,
        force_raw_cache=True,
        logger=logger,
    )
    if hasattr(ds, "write_cache"):
        ds.write_cache = False
    if hasattr(ds, "require_cache"):
        ds.require_cache = True
    native = ev.mapping_to_lut(
        ev.load_yaml_int_map(f"configs/mapping_{domain}_to_common.yaml"),
        size=256,
        default=0,
        name="native_to_common",
    )
    import importlib

    module = importlib.import_module("src.data_" + domain)
    members = []
    for member in ds.files if hasattr(ds, "files") else ds.names:
        path = Path(member) if domain == "dales" else module._resolve_pc_path(ds.eclair_root, member)
        cache = (
            ds._cache_dir / (module._cache_key_for_dales_raw(path=path) + ".pt")
            if domain == "dales"
            else ds._cache_path(member)
        )
        if not path.is_file() or not cache.is_file():
            raise FileNotFoundError(f"Required raw source/cache missing: {path}, {cache}")
        members.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "cache": str(cache),
                "cache_bytes": cache.stat().st_size,
                "cache_mtime_ns": cache.stat().st_mtime_ns,
            }
        )
    identity = {
        "domain": domain,
        "split": split,
        "members": members,
        "target_config_sha256": sha256(Path(f"/scratch/m23csa510/e0_results/{run}/config_resolved.json")),
        "manifest_sha256": sha256(manifest) if manifest else None,
    }
    if domain == "eclair":
        identity["labels_json_sha256"] = sha256(
            Path(cfg["data"]["eclair_root"]) / cfg.get("eclair", {}).get("meta_filename", "labels.json")
        )
    return ds, native, identity


def evaluate_tiles(model, cfg, checkpoint_id, domain, out, deadline, logger):
    import torch

    from . import eval_cross_common as ev

    ds, lut, identity = load_dataset(domain, "test", cfg, logger)
    train_lut = ev.mapping_to_lut(
        ev.load_yaml_int_map("configs/train_id_to_common_dales.yaml"),
        size=8,
        default=0,
        name="source_to_common",
    )
    ev.validate_source_train_to_common_mapping(
        source_cfg=cfg,
        source_domain="dales",
        source_train_to_common=train_lut,
        source_native_to_common_lut=ev.mapping_to_lut(
            ev.load_yaml_int_map("configs/mapping_dales_to_common.yaml"),
            size=256,
            default=0,
            name="source_native",
        ),
        logger=logger,
    )
    contract = {
        "checkpoint": checkpoint_id,
        "dataset": identity,
        "model": cfg["model"],
        "features": cfg["data"]["features"],
        "patch": cfg["data"]["patch"],
        "voxelization": cfg["data"].get("voxelization"),
        "window_m": 120,
        "stride_m": 120,
        "aggregation": "mean_logits",
        "amp": False,
        "source_mapping_sha256": sha256("configs/train_id_to_common_dales.yaml"),
        "target_mapping_sha256": sha256(f"configs/mapping_{domain}_to_common.yaml"),
    }
    signature = fingerprint(contract)
    stage = Path(out) / domain
    stage.mkdir(parents=True, exist_ok=True)
    matrices, subtype = [], np.zeros((256, 8), dtype=np.int64)
    tile_rows = []
    for index in range(len(ds)):
        path = stage / f"tile_{index:04d}.json"
        result = json.loads(path.read_text()) if path.exists() else None
        raw = None
        # Include raw path and cache identities, not only split membership.
        if result is None or result.get("signature") != signature:
            if time.time() > deadline - 900:
                return False
            raw = ds.get_raw(index)
            tick = time.monotonic()
            logits = ev.infer_tile_point_logits(
                model=model,
                raw=raw,
                source_cfg=cfg,
                device=torch.device("cuda:0"),
                amp=False,
                window_size_m=120.0,
                window_stride_m=120.0,
                aggregation="mean_logits",
                max_voxels_per_forward=180000,
                max_split_depth=8,
                logger=logger,
                target_domain=domain,
            )
            for start in range(0, len(logits), 1000000):
                if not np.isfinite(logits[start : start + 1000000]).all():
                    raise RuntimeError(f"Nonfinite logits in {domain} test tile {index}")
            pred = train_lut[logits.argmax(1)]
            native = ev.get_native_labels(raw, target_domain=domain).astype(np.int64)
            if np.any((native < 0) | (native > 255)):
                raise ValueError("Native labels exceed recorded taxonomy")
            gt = lut[native]
            cm = confusion(gt, pred)
            native_cm = np.bincount(native * 8 + pred, minlength=256 * 8).reshape(256, 8)
            result = {
                "signature": signature,
                "tile": index,
                "path": str(raw.get("path", index)),
                "matrix": cm.tolist(),
                "native_gt_by_common_prediction": native_cm.tolist(),
                "seconds": time.monotonic() - tick,
            }
            storage_guard(stage, 64 * 1024, cfg["study"]["storage"])
            atomic_json(path, result)
            del logits, raw
        matrices.append(np.asarray(result["matrix"], dtype=np.int64))
        subtype += np.asarray(result["native_gt_by_common_prediction"], dtype=np.int64)
        tile_rows.append({"tile": index, **summary(result["matrix"])})
    atomic_json(
        stage / "metrics.json",
        {
            **summary(np.sum(matrices, axis=0)),
            "native_gt_by_common_prediction": subtype.tolist(),
            "tiles": tile_rows,
            "contract": contract,
            "subtype_recall": {
                str(i): (float(subtype[i, lut[i]] / subtype[i].sum()) if subtype[i].sum() else None)
                for i in range(256)
                if subtype[i].sum() and lut[i] in (4, 5)
            },
        },
    )
    return True


def take_raw(raw, mask):
    n = len(raw["xyz"])
    return {k: v[mask] if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v for k, v in raw.items()}


def infer_patch(model, raw, cfg, phase=0, features=False):
    """Single bounded forward with a FIXED quantization origin; no window splitting."""
    import MinkowskiEngine as ME
    import torch

    from . import eval_cross_common as ev
    from .features import FeatureConfig, build_features
    from .voxelization import VoxelizationConfig, voxelize_from_q

    norm = float(cfg["data"]["patch"]["coord_norm_factor"])
    voxel = float(cfg["data"]["patch"]["voxel_size"])
    xyz = np.asarray(raw["xyz"], np.float32) / norm
    fc = FeatureConfig(**cfg["data"]["features"])
    fp = build_features(
        xyz_local=xyz,
        intensity=raw.get("intensity"),
        return_number=raw.get("return_number"),
        number_of_returns=raw.get("number_of_returns"),
        rgb=None,
        cfg=fc,
    )
    q = np.floor(xyz / voxel).astype(np.int32)
    q[:, :2] += int(phase)
    vx = voxelize_from_q(
        q_int32=q,
        feats_p_f32=np.asarray(fp, np.float32),
        labels_p_i64=None,
        ignore_index=-100,
        cfg=VoxelizationConfig.from_cfg(cfg["data"]),
        rng=None,
        return_maps=True,
        num_classes_hint=8,
    )
    coords = np.column_stack((np.zeros(len(vx["coords_u"]), np.int32), vx["coords_u"])).astype(np.int32)
    coords_t = torch.from_numpy(coords).to("cuda")
    st = ME.SparseTensor(torch.from_numpy(vx["feats_u"]).float().to("cuda"), coordinates=coords_t)
    captured = {}
    handles = []
    if features:
        for name in ("block3", "block8"):

            def tap(_module, _inputs, output, name=name):
                captured[name] = output

            handles.append(getattr(model, name).register_forward_hook(tap))
    try:
        with torch.inference_mode():
            output = model(st)
            scores = ev.align_sparse_output_to_input_coords(
                out=output, coords_in_t=coords_t, device=torch.device("cuda")
            )
            logits = scores.float().cpu().numpy()[vx["inverse_map"]]
            if not np.isfinite(logits).all():
                raise RuntimeError("Nonfinite bounded-probe logits")
            feats = None
            moments = {}
            if features:
                values = ev.align_sparse_output_to_input_coords(
                    out=captured["block8"], coords_in_t=coords_t, device=torch.device("cuda")
                )
                feats = values.float().cpu().numpy()[vx["inverse_map"]]
                for name, tensor in captured.items():
                    v = tensor.F.detach().float()
                    moments[name] = {
                        "active_sites": len(v),
                        "mean": v.mean(0).cpu().tolist(),
                        "variance": v.var(0, unbiased=False).cpu().tolist(),
                    }
    finally:
        for handle in handles:
            handle.remove()
    return logits, feats, moments


def cohorts(domain, split, cfg, logger, per_class=1):
    from . import eval_cross_common as ev

    ds, lut, _ = load_dataset(domain, split, cfg, logger)
    found = {c: 0 for c in range(1, 8)}
    for index in range(len(ds)):
        raw = ds.get_raw(index)
        native = ev.get_native_labels(raw, target_domain=domain).astype(np.int64)
        gt = lut[native]
        for cls in range(1, 8):
            ids = np.flatnonzero(gt == cls)
            if not len(ids) or found[cls] >= per_class:
                continue
            center = raw["xyz"][ids[len(ids) // 2]]
            mask = (np.abs(raw["xyz"][:, :2] - center[:2]) <= 25).all(1)
            chosen = np.flatnonzero(mask)
            seed = 1337 + 7919 * cls + 101 * index
            if len(chosen) > 20000:
                chosen = np.sort(np.random.default_rng(seed).choice(chosen, 20000, replace=False))
                # Keep the class-center anchor even for a very rare class.
                if ids[len(ids) // 2] not in chosen:
                    chosen[0] = ids[len(ids) // 2]
                    chosen.sort()
            patch = take_raw(raw, chosen)
            patch["xyz"] = patch["xyz"].copy()
            origin = np.floor(patch["xyz"].min(0) / 0.2) * 0.2
            patch["xyz"] -= origin
            found[cls] += 1
            yield {
                "raw": patch,
                "gt": gt[chosen],
                "native": native[chosen],
                "domain": domain,
                "split": split,
                "tile": index,
                "class": cls,
                "center": center - origin,
                "origin": origin,
                "seed": seed,
                "sampling": "class-centered 50m patch, uniform cap 20000 points plus retained focal anchor; diagnostic only",
            }
        if all(v >= per_class for v in found.values()):
            break


def bounded_analysis(model, cfg, out, deadline, logger):
    import torch
    from scipy.spatial import cKDTree

    from . import eval_cross_common as ev

    train_lut = np.asarray((1, 2, 7, 7, 3, 5, 4, 6))
    native_train_lut, _ = ev.source_native_to_train_lut_from_config(
        source_cfg=cfg, source_domain="dales", logger=logger
    )
    bn_before = fingerprint(
        {
            k: v.cpu().tolist()
            for k, v in model.state_dict().items()
            if "running_" in k or "num_batches_tracked" in k
        }
    )
    reports, banks = [], {}
    for domain, split, count in (
        ("dales", "train", 2),
        ("dales", "val", 1),
        ("dales", "test", 1),
        ("eclair", "test", 1),
    ):
        bank = banks.setdefault((domain, split), {"x": [], "y": [], "gt": []})
        for case in cohorts(domain, split, cfg, logger, per_class=count):
            if time.time() > deadline - 900:
                return False
            raw, gt = case["raw"], case["gt"]
            logits, features, moments = infer_patch(model, raw, cfg, features=True)
            pred = train_lut[logits.argmax(1)]
            row = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in case.items()
                if k not in ("raw", "gt", "native")
            }
            row.update(clean=summary(confusion(gt, pred)), stage_moments=moments, variants={})
            class_features = {}
            chosen = []
            for cls in range(1, 8):
                ids = np.flatnonzero(gt == cls)
                selected = (
                    ids[np.linspace(0, len(ids) - 1, min(len(ids), 128)).astype(int)] if len(ids) else []
                )
                chosen.extend(selected)
                if len(selected):
                    class_features[str(cls)] = {
                        "count": len(selected),
                        "mean": features[selected].mean(0).tolist(),
                        "variance": features[selected].var(0).tolist(),
                    }
            row["bounded_class_features"] = class_features
            chosen = np.asarray(chosen, dtype=int)
            bank["x"].append(features[chosen])
            bank["gt"].append(gt[chosen])
            bank["y"].append(native_train_lut[case["native"][chosen]] if domain == "dales" else gt[chosen])
            if split == "test":
                rng = np.random.default_rng(case["seed"])
                keep = rng.random(len(gt)) < 0.5
                keep[0] = True
                constant = copy.copy(raw)
                for key in ("return_number", "number_of_returns"):
                    if key in raw:
                        constant[key] = np.ones_like(raw[key])
                center = np.asarray(case["center"])
                halo = (np.abs(raw["xyz"][:, :2] - center[:2]) <= 15).all(1)
                core = (np.abs(raw["xyz"][:, :2] - center[:2]) <= 10).all(1)
                cases = [
                    ("thinning_50pct", take_raw(raw, keep), keep, 0),
                    ("grid_phase_xy_one_voxel", raw, np.ones(len(gt), bool), 1),
                    ("constant_returns_ood", constant, np.ones(len(gt), bool), 0),
                ]
                if core.any():
                    cases.append(("fixed_core_halo15_vs25", take_raw(raw, halo), halo, 0))
                for name, variant, retained, phase in cases:
                    scores, _, _ = infer_patch(model, variant, cfg, phase=phase)
                    vpred = train_lut[scores.argmax(1)]
                    support = core[retained] if name.startswith("fixed_core") else np.ones(len(vpred), bool)
                    base = pred[retained][support]
                    truth = gt[retained][support]
                    row["variants"][name] = {
                        "changed": summary(confusion(truth, vpred[support])),
                        "paired_clean": summary(confusion(truth, base)),
                        "paired_points": len(truth),
                        "disagreement": float((base != vpred[support]).mean()),
                        "fixed_quantization_origin": True,
                        "recursive_splitting": False,
                    }
                anchors = chosen[np.linspace(0, len(chosen) - 1, min(512, len(chosen))).astype(int)]
                tree = cKDTree(raw["xyz"])
                neighbours = tree.query_ball_point(raw["xyz"][anchors], r=1.0)
                density = np.asarray([len(n) for n in neighbours])
                linearity, verticality = np.full(len(anchors), np.nan), np.full(len(anchors), np.nan)
                for j, neighbours_j in enumerate(neighbours):
                    if len(neighbours_j) < 3:
                        continue
                    points = raw["xyz"][neighbours_j]
                    values, vectors = np.linalg.eigh(np.cov(points.T))
                    if values[-1] > 1e-12:
                        linearity[j] = (values[-1] - values[-2]) / values[-1]
                        verticality[j] = abs(vectors[2, -1])
                strata = {}
                for label, mask in (
                    ("neighbors_le4", density <= 4),
                    ("neighbors_gt4", density > 4),
                    ("linear_ge_0_8", linearity >= 0.8),
                    ("linear_vertical_ge_0_8", (linearity >= 0.8) & (verticality >= 0.8)),
                ):
                    ids = anchors[mask]
                    strata[label] = summary(confusion(gt[ids], pred[ids]))
                row["geometry"] = {
                    "radius_m": 1.0,
                    "scope": "capped diagnostic cohort, not raw acquisition density",
                    "anchor_count": len(anchors),
                    "strata": strata,
                }
            reports.append(row)
    # A source-only frozen ridge readout: fit on source TRAIN, choose lambda on source VAL.
    # Features are bounded in RAM; no feature-bank files or target fitting are performed.
    train = banks[("dales", "train")]
    val = banks[("dales", "val")]
    x, y = np.concatenate(train["x"]), np.concatenate(train["y"])
    xv, yv = np.concatenate(val["x"]), np.concatenate(val["gt"])
    valid = (y >= 0) & (y < 8)
    x, y = x[valid].astype(np.float64), y[valid]
    mean, std = x.mean(0), x.std(0).clip(0.001)

    def design(values):
        return np.column_stack(((values - mean) / std, np.ones(len(values))))

    a, av = design(x), design(xv)
    counts = np.bincount(y, minlength=8)
    weights = 1.0 / counts[y].clip(1)
    weights /= weights.mean()
    target = np.eye(8)[y]
    candidates = []
    for penalty in (0.1, 1.0, 10.0):
        gram = a.T @ (a * weights[:, None]) + penalty * np.eye(a.shape[1])
        coefficients = np.linalg.solve(gram, a.T @ (target * weights[:, None]))
        score = summary(confusion(yv, train_lut[(av @ coefficients).argmax(1)]))["all_seven_miou"]
        candidates.append((score, -penalty, coefficients))
    score, negative_penalty, coefficients = max(candidates, key=lambda item: (item[0], item[1]))
    heads = {
        "penalty": -negative_penalty,
        "selected_source_val_miou": score,
        "fit_source_native_counts": counts.tolist(),
        "target_labels_used_for_fit_or_selection": False,
        "inverse_native_frequency_weighted_ridge": True,
        "standardization_mean": mean.tolist(),
        "standardization_std": std.tolist(),
        "coefficients": coefficients.tolist(),
        "source_val_candidates": [{"penalty": -c[1], "miou": c[0]} for c in candidates],
        "scope": "bounded diagnostic cohorts; not canonical full-test readout results",
        "evaluations": {},
    }
    for (domain, split), bank in banks.items():
        xx, gg = np.concatenate(bank["x"]), np.concatenate(bank["gt"])
        heads["evaluations"][f"{domain}/{split}"] = summary(
            confusion(gg, train_lut[(design(xx) @ coefficients).argmax(1)])
        )
    bn_after = fingerprint(
        {
            k: v.cpu().tolist()
            for k, v in model.state_dict().items()
            if "running_" in k or "num_batches_tracked" in k
        }
    )
    if bn_before != bn_after:
        raise RuntimeError("Analysis mutated normalization buffers")
    out = Path(out)
    storage_guard(out, 8 * 1024**2, cfg["study"]["storage"])
    atomic_json(
        out / "bounded_probes.json",
        {
            "cases": reports,
            "bn_buffers_unchanged": True,
            "interpretation": "Sensitivity and readout diagnostics do not identify a unique training cause",
        },
    )
    atomic_json(out / "frozen_head_results.json", heads)
    atomic_json(
        out / "bounded_complete.json",
        {name: sha256(out / name) for name in ("bounded_probes.json", "frozen_head_results.json")},
    )
    torch.cuda.empty_cache()
    return True
