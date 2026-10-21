"""Audited, frozen-weight transfer evidence. Canonical inference stays authoritative."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import shutil
import signal
import time
from pathlib import Path

import numpy as np

from .study_analysis import fingerprint, load_dataset
from .study_metrics import confusion, summary
from .study_runtime import sha256
from .transfer_evidence_sampling import (
    choose_windows,
    conditional_inputs,
    confidence_arrays,
    geometry,
    payload_digest,
    perturb,
    sample_tile,
    sampling_plan,
    seed_for,
    take_raw,
)

ASSETS = Path(__file__).resolve().parents[1] / "configs/transfer_evidence"


class NeedsResume(RuntimeError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical_object(obj):
    return json.loads(json.dumps(obj, default=str))


def config_contract(cfg):
    return canonical_object(
        {
            "model": cfg["model"],
            "features": cfg["data"]["features"],
            "patch": cfg["data"]["patch"],
            "voxelization": cfg["data"].get("voxelization"),
            "voxel_feat_mode": cfg["data"].get("voxel_feat_mode"),
            "label_space": cfg["data"].get("label_space"),
            "label_map": cfg["data"].get("dales_label_map_native_to_train"),
        }
    )


def safe_output(path):
    out = Path(path).expanduser().resolve()
    expected = Path(f"/scratch/{__import__('getpass').getuser()}/e0_results/transfer_evidence").resolve()
    if out.parent != expected or not out.name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Output must be a named study directly under {expected}")
    return out


def verify_checkpoints(settings):
    """File digests are live-audited, not inferred from an experiment name."""
    tasks = []
    for item in settings["models"]:
        checkpoint = Path(item["checkpoint"])
        config = checkpoint.parents[1] / "config_resolved.json"
        if sha256(checkpoint) != item["checkpoint_sha256"] or sha256(config) != item["config_sha256"]:
            raise ValueError(f"Pinned checkpoint/config changed: {item['id']}")
        cfg = read_json(config)
        if cfg["data"]["dataset"] != item["source"]:
            raise ValueError("Source dataset differs from checkpoint inventory")
        tasks.append({**item, "cfg": cfg})
    if len(tasks) != 10 or len({x["id"] for x in tasks}) != 10:
        raise ValueError("Expected ten distinct checkpoint tasks")
    return tasks


def validate_selection(obj, task, rows):
    if int(obj["epoch"]) != task["epoch"] or abs(float(obj["best_val_miou"]) - task["best_val_miou"]) > 1e-12:
        raise ValueError("Live checkpoint epoch/score differs from audited best.pt")
    if config_contract(obj["cfg"]) != config_contract(task["cfg"]):
        raise ValueError(f"Inference-critical checkpoint/resolved configuration mismatch: {task['id']}")
    if task["source_validation_verified"]:
        valid = [r for r in rows if r.get("val_miou", "").lower() not in ("", "nan", "none")]
        selected = [r for r in valid if int(r["epoch"]) == task["epoch"]]
        if len(selected) != 1 or abs(float(selected[0]["val_miou"]) - task["best_val_miou"]) > 1e-12:
            raise ValueError("Source-validation row does not support the frozen selection")
        if abs(max(float(r["val_miou"]) for r in valid) - task["best_val_miou"]) > 1e-12:
            raise ValueError("Checkpoint is not the recorded source-validation maximum")
    elif task["id"] != "dales_38163" or not task.get("selection_warning"):
        raise ValueError("Unreviewed historical selection gap")


class Store:
    def __init__(self, root, settings, deadline):
        self.root, self.settings, self.deadline = Path(root), settings, deadline
        self.stop = False
        for sig in (signal.SIGTERM, *([signal.SIGUSR1] if hasattr(signal, "SIGUSR1") else [])):
            signal.signal(sig, self.request_stop)

    def request_stop(self, *_):
        self.stop = True

    def heartbeat(self):
        if self.stop or time.time() > self.deadline - 900:
            raise NeedsResume("Deadline/signal: completed tile/stage records are reusable")

    def admit(self, write_bytes=0):
        self.heartbeat()
        used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        if used + write_bytes > self.settings["output_budget_bytes"]:
            raise OSError("2 GiB output cap reached; no existing results will be deleted")
        if shutil.disk_usage(self.root).free < write_bytes + 512 * 1024**2:
            raise OSError("Filesystem free-space reserve exhausted")

    def json(self, name, value):
        data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        self.admit(len(data))
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)

    def npz(self, name, **values):
        self.admit(sum(np.asarray(v).nbytes for v in values.values()) + 65536)
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("wb") as f:
            np.savez_compressed(f, **values)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        return {"path": name, "sha256": sha256(path)}

    def text(self, name, value):
        data = value.encode("utf-8")
        self.admit(len(data))
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(data)
        os.replace(temp, path)

    def complete(self, name):
        path = self.root / name
        if not path.exists():
            return None
        result = read_json(path)
        if result.get("identity") != self.identity:
            raise ValueError(f"Stale stage identity: {name}")
        for item in result.get("artifacts", []):
            if sha256(self.root / item["path"]) != item["sha256"]:
                raise ValueError(f"Stage artifact checksum mismatch: {item['path']}")
        return result


def state_digest(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def clear_eval_cache(model):
    """Release auxiliary-hook activations, never parameters or BN statistics."""
    if hasattr(model, "base"):
        if hasattr(model, "_feature"):
            model._feature = None
        if hasattr(model, "_feat_cache"):
            model._feat_cache.clear()


class ModelPool(dict):
    """Keep one GPU model resident so the checkpoint matrix does not alter memory admission."""

    active = None

    def __getitem__(self, key):
        if self.active != key:
            if self.active is not None:
                previous = super().__getitem__(self.active)
                clear_eval_cache(previous)
                previous.to("cpu")
            super().__getitem__(key).to("cuda:0")
            self.active = key
        return super().__getitem__(key)


class Observer:
    """Read-only point-aligned feature tap, with exact overlap/split accounting."""

    def __init__(self, anchors, cfg, heartbeat=lambda: None):
        self.anchors, self.cfg, self.heartbeat = np.asarray(anchors, np.int64), cfg, heartbeat
        self.captured = None
        self.forward_sizes, self.split_sizes = [], []

    def prepared(self, coords, feats, inverse):
        self.coords, self.feats, self.inverse = coords, feats, inverse
        self.anchor_voxels = inverse[self.anchors]
        self.counts = np.zeros(len(self.anchors), np.int32)
        self.feature_sum = None
        self.edge = np.full(len(coords), np.inf, np.float32)
        self.subset_edge = np.full(len(coords), np.inf, np.float32)
        self.depth = np.zeros(len(coords), np.uint8)
        self.spacing = float(self.cfg["data"]["patch"]["voxel_size"]) * float(
            self.cfg["data"]["patch"]["coord_norm_factor"]
        )

    def window(self, selected, x0, y0, width):
        self.heartbeat()
        xy = self.coords[selected, :2]
        distance = np.minimum.reduce(
            [xy[:, 0] - x0, xy[:, 1] - y0, x0 + width - 1 - xy[:, 0], y0 + width - 1 - xy[:, 1]]
        )
        self.edge[selected] = np.minimum(self.edge[selected], distance.astype(np.float32) * self.spacing)

    def split(self, selected, depth):
        self.heartbeat()
        self.depth[selected] = np.maximum(self.depth[selected], depth + 1)
        if len(self.split_sizes) < 20000:
            self.split_sizes.append([len(selected), int(depth)])

    def hook(self, model):
        base = model.base if hasattr(model, "base") else model

        def tap(_module, args):
            self.captured = args[0]

        return base.final.register_forward_pre_hook(tap)

    def forwarded(self, selected, coords_t, device):
        from . import eval_cross_common as ev

        self.heartbeat()
        xy = self.coords[selected, :2]
        distances = np.minimum(xy - xy.min(0), xy.max(0) - xy).min(1) * self.spacing
        self.subset_edge[selected] = np.minimum(self.subset_edge[selected], distances)
        order = np.argsort(selected)
        sorted_ids = selected[order]
        pos = np.searchsorted(sorted_ids, self.anchor_voxels)
        found = pos < len(sorted_ids)
        found[found] &= sorted_ids[pos[found]] == self.anchor_voxels[found]
        keep = np.flatnonzero(found)
        if len(keep):
            if self.captured is None:
                raise RuntimeError("Final decoder feature tap did not fire")
            values = ev.align_sparse_output_to_input_coords(out=self.captured, coords_in_t=coords_t, device=device)
            values = values[order[pos[keep]]].detach().float().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite decoder features")
            if self.feature_sum is None:
                self.feature_sum = np.zeros((len(self.anchors), values.shape[1]), np.float64)
            self.feature_sum[keep] += values
            self.counts[keep] += 1
        self.captured = None
        if len(self.forward_sizes) < 20000:
            self.forward_sizes.append(len(selected))

    def features(self):
        if len(self.anchors) and (self.feature_sum is None or np.any(self.counts == 0)):
            raise ValueError("Sampled decoder features lack complete window coverage")
        return (
            (self.feature_sum / self.counts[:, None]).astype(np.float32)
            if len(self.anchors)
            else np.empty((0, 128), np.float32)
        )


def prepare_grid(raw, cfg):
    """Diagnostic-window preparation uses the identical feature/voxel primitives."""
    from .features import FeatureConfig, build_features
    from .voxelization import VoxelizationConfig, voxelize_from_q

    patch = cfg["data"]["patch"]
    xyz = np.asarray(raw["xyz"], dtype=np.float32) / float(patch["coord_norm_factor"])
    feats = build_features(
        xyz_local=xyz,
        intensity=raw.get("intensity"),
        return_number=raw.get("return_number"),
        number_of_returns=raw.get("number_of_returns"),
        rgb=raw.get("rgb"),
        cfg=FeatureConfig(**cfg["data"]["features"]),
    )
    q = np.floor(xyz / float(patch["voxel_size"])).astype(np.int32)
    return voxelize_from_q(
        q_int32=q,
        feats_p_f32=feats,
        labels_p_i64=None,
        ignore_index=-100,
        cfg=VoxelizationConfig.from_cfg(cfg["data"]),
        rng=None,
        return_maps=True,
        num_classes_hint=int(cfg["model"]["out_channels"]),
    )


def window_ids(raw, spec, cfg):
    """Physical selection snaps to that model's existing input-grid boundaries."""
    patch = cfg["data"]["patch"]
    spacing = float(patch["coord_norm_factor"]) * float(patch["voxel_size"])
    q = np.floor(
        (np.asarray(raw["xyz"], np.float32) / float(patch["coord_norm_factor"])) / float(patch["voxel_size"])
    ).astype(np.int32)
    x0, y0 = int(np.floor(spec["x0_m"] / spacing)), int(np.floor(spec["y0_m"] / spacing))
    width = int(np.ceil(120.0 / spacing))
    return np.flatnonzero((q[:, 0] >= x0) & (q[:, 0] < x0 + width) & (q[:, 1] >= y0) & (q[:, 1] < y0 + width))


def infer_window(model, cfg, raw, anchors, store, logger, observer=True):
    import torch

    from . import eval_cross_common as ev

    vx = prepare_grid(raw, cfg)
    tap = Observer(anchors, cfg, store.heartbeat) if observer else None
    if tap:
        tap.prepared(vx["coords_u"], vx["feats_u"], vx["inverse_map"])
    scores = ev.recursively_infer_subset(
        model=model,
        coords_u=vx["coords_u"],
        feats_u=vx["feats_u"],
        sel=np.arange(len(vx["coords_u"]), dtype=np.int64),
        num_classes=int(cfg["model"]["out_channels"]),
        device=torch.device("cuda:0"),
        amp=False,
        max_voxels_per_forward=180000,
        depth=0,
        max_depth=8,
        logger=logger,
        observer=tap,
    )
    logits = scores[vx["inverse_map"]]
    clear_eval_cache(model)
    return logits, tap, vx


def check_raw(raw, domain, lut, expected):
    from . import eval_cross_common as ev

    native = ev.get_native_labels(raw, target_domain=domain).astype(np.int64)
    if np.any((native < 0) | (native > 255)):
        raise ValueError("Native label out of taxonomy range")
    gt = lut[native]
    if len(native) != expected["points"] or not np.array_equal(
        np.bincount(native, minlength=256), expected["native_histogram"]
    ):
        raise ValueError("Raw point/label support changed since diagnosis")
    if not np.array_equal(np.bincount(gt, minlength=8), expected["common_histogram"]):
        raise ValueError("Common ontology differs from audited data")
    if payload_digest(raw) != expected["numeric_payload_sha256"]:
        raise ValueError("Raw numeric payload differs from audited scan")
    return native, gt


def geometry_stage(store, datasets, catalog, settings):
    banks, windows = {}, {}
    for (domain, split), (ds, lut, _) in datasets.items():
        group = f"{domain}/{split}"
        plan = sampling_plan(catalog, domain, split, settings)
        store.json(f"sampling/{domain}_{split}.json", plan)
        groups = []
        for i in range(len(ds)):
            marker = f"geometry/{domain}_{split}_{i:04d}.json"
            row = store.complete(marker)
            if row is None:
                store.admit()
                raw = ds.get_raw(i)
                native, gt = check_raw(raw, domain, lut, catalog[f"{group}/{i}"])
                ids, flags, probability = sample_tile(gt, native, plan, i, seed_for(settings["seed"], group))
                geo = geometry(raw["xyz"], gt, ids, settings["radii_m"], store.heartbeat)
                artifact = store.npz(
                    marker.replace(".json", ".npz"),
                    ids=ids,
                    flags=flags,
                    probability=probability,
                    xyz=np.asarray(raw["xyz"])[ids],
                    gt=gt[ids],
                    native=native[ids],
                    geometry=geo,
                    rn=np.asarray(raw["return_number"])[ids],
                    nor=np.asarray(raw["number_of_returns"])[ids],
                )
                row = {"identity": store.identity, "artifacts": [artifact], "anchors": len(ids)}
                store.json(marker, row)
                del raw, geo
            with np.load(store.root / row["artifacts"][0]["path"], allow_pickle=False) as a:
                groups.append((i, a["ids"], a["xyz"], a["gt"]))
            banks[(domain, split, i)] = row["artifacts"][0]["path"]
            if domain == "dales" or i % 25 == 0:
                print(f"Geometry {group}: tile {i + 1}/{len(ds)}", flush=True)
        windows[group] = choose_windows(
            groups,
            settings["source_windows"] if split == "train" else settings["probe_windows"],
            seed_for(settings["seed"], group, "windows"),
        )
        store.json(f"sampling/{domain}_{split}_windows.json", windows[group])
        print(f"Geometry {group}: {len(ds)} tiles complete", flush=True)
    return banks, windows


def load_models(tasks, logger):
    import csv

    import torch

    from . import eval_cross_common as ev

    models, maps = {}, {}
    for task in tasks:
        cfg = task["cfg"]
        obj = torch.load(task["checkpoint"], map_location="cpu")
        with (Path(task["checkpoint"]).parents[1] / "metrics.csv").open() as f:
            validate_selection(obj, task, list(csv.DictReader(f)))
        model = ev.build_model(
            in_channels=ev.infer_in_channels(ev.FeatureConfig(**cfg["data"]["features"])),
            out_channels=int(cfg["model"]["out_channels"]),
            D=int(cfg["model"].get("D", 3)),
            cfg=cfg,
        )
        model.load_state_dict(obj["model_state"], strict=True)
        del obj
        model.eval()
        source = task["source"]
        mapping = "configs/train_id_to_common_dales.yaml" if source == "dales" else "configs/train_id_to_common_id.yaml"
        lut = ev.mapping_to_lut(
            ev.load_yaml_int_map(mapping), size=int(cfg["model"]["out_channels"]), default=0, name="source"
        )
        native = ev.mapping_to_lut(
            ev.load_yaml_int_map(f"configs/mapping_{source}_to_common.yaml"), size=256, default=0, name="native"
        )
        ev.validate_source_train_to_common_mapping(
            source_cfg=cfg,
            source_domain=source,
            source_train_to_common=lut,
            source_native_to_common_lut=native,
            logger=logger,
        )
        models[task["id"]], maps[task["id"]] = model, lut
    return ModelPool(models), maps


def full_inference(store, datasets, banks, tasks, models, maps, settings, logger):
    import torch

    from . import eval_cross_common as ev

    gallery_counts = {}
    for path in sorted((store.root / "test").glob("*.json")) if (store.root / "test").exists() else []:
        for item in store.complete(str(path.relative_to(store.root))).get("gallery", []):
            k = tuple(item["group"])
            gallery_counts[k] = gallery_counts.get(k, 0) + 1
    for domain in ("dales", "eclair"):
        ds, lut, _ = datasets[(domain, "test")]
        for i in range(len(ds)):
            marker = f"test/{domain}_{i:04d}.json"
            if store.complete(marker):
                continue
            store.admit()
            raw = ds.get_raw(i)
            native = ev.get_native_labels(raw, target_domain=domain).astype(np.int64)
            gt = lut[native]
            with np.load(store.root / banks[(domain, "test", i)], allow_pickle=False) as anchor:
                ids = anchor["ids"]
            tile = {"identity": store.identity, "artifacts": [], "models": {}, "agreement": {}}
            predictions = {}
            for task in tasks:
                key, cfg = task["id"], task["cfg"]
                tick = time.monotonic()
                model = models[key]
                before = state_digest(model)
                tap = Observer(ids, cfg, store.heartbeat)
                logits = ev.infer_tile_point_logits(
                    model=model,
                    raw=raw,
                    source_cfg=cfg,
                    device=torch.device("cuda:0"),
                    target_domain=domain,
                    logger=logger,
                    observer=tap,
                    **settings["protocol"],
                )
                clear_eval_cache(model)
                if before != state_digest(model):
                    raise ValueError("Inference changed model weights or normalization buffers")
                if not np.isfinite(logits).all():
                    raise ValueError("Nonfinite canonical logits")
                if i == 0:
                    # Complete first tile, not a small unrelated patch: numerical path is unchanged.
                    original = ev.infer_tile_point_logits(
                        model=model,
                        raw=raw,
                        source_cfg=cfg,
                        device=torch.device("cuda:0"),
                        target_domain=domain,
                        logger=logger,
                        **settings["protocol"],
                    )
                    if not np.array_equal(logits, original):
                        raise ValueError("Observer changed canonical logits; stop before further claims")
                    clear_eval_cache(model)
                    del original
                pred_native = logits.argmax(1)
                pred = maps[key][pred_native].astype(np.uint8)
                predictions[key] = pred
                cm = confusion(gt, pred)
                native_cm = np.bincount(
                    native * int(cfg["model"]["out_channels"]) + pred_native,
                    minlength=256 * int(cfg["model"]["out_channels"]),
                ).reshape(256, -1)
                native_common_cm = np.bincount(native * 8 + pred, minlength=256 * 8).reshape(256, 8)
                joint, invalid = conditional_inputs(gt, pred, raw["return_number"], raw["number_of_returns"])
                native_joint, native_invalid = conditional_inputs(
                    native, pred, raw["return_number"], raw["number_of_returns"], classes=256, ignore_zero=False
                )
                pooled_sum = np.zeros((64, tap.feats.shape[1]), np.float64)
                pooled_square = np.zeros_like(pooled_sum)
                confidence_hist = np.zeros((8, 2, 10), np.int64)
                confidence_sums = np.zeros((8, 2, 10), np.float64)
                common_confidence_hist = np.zeros_like(confidence_hist)
                margin_hist = np.zeros((8, 8, 9), np.int64)
                boundary_cm = np.zeros((4, 8, 8), np.int64)
                subset_cm = np.zeros_like(boundary_cm)
                split_cm = np.zeros((9, 8, 8), np.int64)
                alternative = np.zeros((8, 8), np.int64)
                probability_merge = np.zeros((8, 8), np.int64)
                for start in range(0, len(gt), 250000):
                    sl = slice(start, start + 250000)
                    c = confidence_arrays(logits[sl], maps[key])
                    g, p = gt[sl], pred[sl]
                    valid = g > 0
                    wrong = p != g
                    binid = np.minimum((c["native_confidence"] * 10).astype(int), 9)
                    k = (g[valid] * 2 + wrong[valid]) * 10 + binid[valid]
                    confidence_hist += np.bincount(k, minlength=160).reshape(8, 2, 10)
                    confidence_sums += np.bincount(k, weights=c["native_confidence"][valid], minlength=160).reshape(
                        8, 2, 10
                    )
                    cb = np.minimum((c["mapped_class_probability"] * 10).astype(int), 9)
                    common_confidence_hist += np.bincount(
                        (g[valid] * 2 + wrong[valid]) * 10 + cb[valid], minlength=160
                    ).reshape(8, 2, 10)
                    pair = g[valid] * 8 + p[valid]
                    margin_bin = np.digitize(
                        c["native_logit_margin"][valid], [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
                    )
                    margin_hist += np.bincount(pair * 9 + margin_bin, minlength=64 * 9).reshape(8, 8, 9)
                    fv = tap.feats[tap.inverse[sl]][valid].astype(np.float64)
                    for channel in range(fv.shape[1]):
                        pooled_sum[:, channel] += np.bincount(pair, weights=fv[:, channel], minlength=64)
                        pooled_square[:, channel] += np.bincount(pair, weights=fv[:, channel] ** 2, minlength=64)
                    edge = np.digitize(tap.edge[tap.inverse[sl]], [0.5, 2.0, 5.0])
                    subset_edge = np.digitize(tap.subset_edge[tap.inverse[sl]], [0.5, 2.0, 5.0])
                    depth = tap.depth[tap.inverse[sl]]
                    for b in range(4):
                        boundary_cm[b] += confusion(g[edge == b], p[edge == b])
                        subset_cm[b] += confusion(g[subset_edge == b], p[subset_edge == b])
                    for b in range(9):
                        split_cm[b] += confusion(g[depth == b], p[depth == b])
                    restricted = logits[sl].copy()
                    restricted[:, maps[key] == 0] = -np.inf
                    alternative += confusion(g, maps[key][restricted.argmax(1)])
                    probability_merge += confusion(g, c["common_probability_argmax"])
                anchor_confidence = confidence_arrays(logits[ids], maps[key])
                arrays = {
                    "point_ids": ids,
                    "features": tap.features(),
                    "feature_observations": tap.counts,
                    "pooled_inputs": tap.feats[tap.inverse[ids]],
                    "gt": gt[ids],
                    "native": native[ids],
                    "window_edge_m": tap.edge[tap.inverse[ids]],
                    "actual_subset_edge_m": tap.subset_edge[tap.inverse[ids]],
                    "split_depth": tap.depth[tap.inverse[ids]],
                    **anchor_confidence,
                    "joint_returns": joint,
                    "invalid_returns": invalid,
                    "native_return_joint": native_joint,
                    "native_invalid_returns": native_invalid,
                    "pooled_sum": pooled_sum,
                    "pooled_square": pooled_square,
                    "confidence_hist": confidence_hist,
                    "confidence_sums": confidence_sums,
                    "common_confidence_hist": common_confidence_hist,
                    "native_logit_margin_hist": margin_hist,
                    "window_edge_confusions": boundary_cm,
                    "actual_subset_edge_confusions": subset_cm,
                    "split_depth_confusions": split_cm,
                    "native_gt_by_native_prediction": native_cm,
                    "native_gt_by_common_prediction": native_common_cm,
                }
                artifact = store.npz(f"test/{domain}_{i:04d}_{key}.npz", **arrays)
                tile["artifacts"].append(artifact)
                tile["models"][key] = {
                    **summary(cm),
                    "artifact": artifact["path"],
                    "restricted_common_outputs": summary(alternative),
                    "probability_merge_rule": summary(probability_merge),
                    "seconds": time.monotonic() - tick,
                    "state_unchanged": True,
                    "observer_parity_checked": i == 0,
                    "forwards": len(tap.forward_sizes),
                    "split_sizes": tap.split_sizes[:100],
                }
                del logits, tap, arrays
                gc.collect()
            keys = list(predictions)
            for a, key in enumerate(keys):
                for other in keys[a + 1 :]:
                    correct, correct2 = predictions[key] == gt, predictions[other] == gt
                    valid = gt > 0
                    pair = np.bincount(gt[valid] * 4 + correct[valid] * 2 + correct2[valid], minlength=32).reshape(8, 4)
                    tile["agreement"][key + "/" + other] = pair.tolist()
            tile["gallery"] = capture_gallery(
                store, raw, native, gt, ids, predictions, tasks, domain, i, gallery_counts
            )
            tile["artifacts"].extend(item["artifact"] for item in tile["gallery"])
            store.json(marker, tile)
            print(f"Canonical {domain} tile {i + 1}/{len(ds)}: ten checkpoints complete", flush=True)
            del predictions, raw


def capture_gallery(store, raw, native, gt, anchors, predictions, tasks, domain, tile_index, counts):
    from scipy.spatial import cKDTree

    cross = [t["id"] for t in tasks if t["source"] != domain]
    wrong = np.stack([predictions[k][anchors] != gt[anchors] for k in cross])
    kinds = {"persistent": wrong.all(0), "rescued": wrong.any(0) & ~wrong.all(0)}
    result, tree = [], None
    for cls in range(1, 8):
        for kind, mask in kinds.items():
            group = (domain, cls, kind)
            remaining = 2 - counts.get(group, 0)
            candidates = anchors[mask & (gt[anchors] == cls)]
            if remaining <= 0 or not len(candidates):
                continue
            rng = np.random.default_rng(seed_for(domain, tile_index, cls, kind, "gallery"))
            for point in rng.choice(candidates, min(remaining, len(candidates)), replace=False):
                if tree is None:
                    tree = cKDTree(np.asarray(raw["xyz"], np.float64), copy_data=False)
                neighbours = np.asarray(tree.query_ball_point(raw["xyz"][point], 16.0), np.int64)
                total = len(neighbours)
                if total > 4096:
                    neighbours = np.sort(rng.choice(neighbours, 4096, replace=False))
                    if point not in neighbours:
                        neighbours[0] = point
                        neighbours.sort()
                slot = counts.get(group, 0)
                name = f"gallery/{domain}_{cls}_{kind}_{slot}.npz"
                artifact = store.npz(
                    name,
                    xyz=(raw["xyz"][neighbours] - raw["xyz"][point]).astype(np.float32),
                    point_ids=neighbours,
                    native=native[neighbours],
                    gt=gt[neighbours],
                    predictions=np.stack([predictions[k][neighbours] for k in cross]),
                )
                result.append(
                    {
                        "group": list(group),
                        "tile": tile_index,
                        "anchor_point": int(point),
                        "native_label": int(native[point]),
                        "raw_neighbourhood_points": total,
                        "saved_points": len(neighbours),
                        "radius_m": 16.0,
                        "cross_models": cross,
                        "anchor_predictions": {k: int(predictions[k][point]) for k in cross},
                        "selection": "post-hoc persistent/rescued examples among fixed anchors, not population estimates",
                        "artifact": artifact,
                    }
                )
                counts[group] = slot + 1
    return result


def source_features(store, datasets, windows, tasks, models, settings, logger):
    from scipy.spatial import cKDTree

    from . import eval_cross_common as ev

    for domain in ("dales", "eclair"):
        ds, lut, _ = datasets[(domain, "train")]
        last_tile, raw, tree = None, None, None
        for wi, spec in sorted(enumerate(windows[domain + "/train"]), key=lambda x: x[1]["tile"]):
            marker = f"source/{domain}_{wi:03d}.json"
            if store.complete(marker):
                continue
            store.admit()
            if last_tile != spec["tile"]:
                raw, tree = None, None
                gc.collect()
                raw = ds.get_raw(spec["tile"])
                native = ev.get_native_labels(raw, target_domain=domain).astype(np.int64)
                gt = lut[native]
                tree = cKDTree(np.asarray(raw["xyz"], np.float64), copy_data=False)
                last_tile = spec["tile"]
            row = {"identity": store.identity, "spec": spec, "models": {}, "artifacts": []}
            geo_cache = {}
            for task in [t for t in tasks if t["source"] == domain]:
                cfg, key = task["cfg"], task["id"]
                ids = window_ids(raw, spec, cfg)
                if not len(ids):
                    raise ValueError("Preselected source window is unexpectedly empty")
                local = []
                for cls in range(1, 8):
                    candidates = np.flatnonzero(gt[ids] == cls)
                    rng = np.random.default_rng(seed_for(domain, wi, cls, "source_features"))
                    local.extend(rng.choice(candidates, min(128, len(candidates)), replace=False).tolist())
                local = np.asarray(sorted(local), np.int64)
                cloud = take_raw(raw, ids)
                before = state_digest(models[key])
                _, tap, _ = infer_window(models[key], cfg, cloud, local, store, logger)
                if before != state_digest(models[key]):
                    raise ValueError("Source feature extraction updated weights/buffers")
                # Geometry is computed against the complete tile, not the inference crop.
                geo_key = hashlib.sha256(ids[local].tobytes()).hexdigest()
                if geo_key not in geo_cache:
                    geo_cache[geo_key] = geometry(
                        raw["xyz"], gt, ids[local], settings["radii_m"], store.heartbeat, tree=tree
                    )
                geo = geo_cache[geo_key]
                artifact = store.npz(
                    f"source/{domain}_{wi:03d}_{key}.npz",
                    features=tap.features(),
                    point_ids=ids[local],
                    gt=gt[ids[local]],
                    native=native[ids[local]],
                    geometry=geo,
                    xyz=np.asarray(raw["xyz"])[ids[local]],
                )
                row["models"][key] = {
                    "artifact": artifact["path"],
                    "class_support": np.bincount(gt[ids[local]], minlength=8).tolist(),
                }
                row["artifacts"].append(artifact)
                del cloud, tap
            store.json(marker, row)
            print(f"Source {domain}: window {wi + 1}/{len(windows[domain + '/train'])}", flush=True)
        print(f"Source feature windows {domain}: complete", flush=True)
        del raw, tree


def effective_change(clean, changed):
    def keys(coords):
        a = np.ascontiguousarray(coords, dtype=np.int32)
        return a.view(np.dtype((np.void, 12))).ravel()

    _, a, b = np.intersect1d(keys(clean["coords_u"]), keys(changed["coords_u"]), return_indices=True)
    fraction = float(np.any(clean["feats_u"][a] != changed["feats_u"][b], axis=1).mean()) if len(a) else None
    return {
        "clean_voxels": len(clean["coords_u"]),
        "changed_voxels": len(changed["coords_u"]),
        "shared_voxels": len(a),
        "changed_pooled_fraction_on_shared_voxels": fraction,
        "lost_voxel_fraction": 1.0 - len(a) / max(1, len(clean["coords_u"])),
    }


def paired_probes(store, datasets, windows, tasks, models, maps, settings, logger):
    from . import eval_cross_common as ev

    for domain in ("dales", "eclair"):
        ds, lut, _ = datasets[(domain, "test")]
        last_tile, raw = None, None
        for wi, spec in sorted(enumerate(windows[domain + "/test"]), key=lambda x: x[1]["tile"]):
            marker = f"probes/{domain}_{wi:03d}.json"
            if store.complete(marker):
                continue
            store.admit()
            if last_tile != spec["tile"]:
                raw = ds.get_raw(spec["tile"])
                last_tile = spec["tile"]
            row = {"identity": store.identity, "spec": spec, "models": {}, "artifacts": []}
            for task in tasks:
                cfg, key, model = task["cfg"], task["id"], models[task["id"]]
                ids = window_ids(raw, spec, cfg)
                cloud = take_raw(raw, ids)
                gt = lut[ev.get_native_labels(cloud, target_domain=domain).astype(np.int64)]
                before = state_digest(model)
                clean, _, grid = infer_window(model, cfg, cloud, [], store, logger, observer=False)
                identity_raw, _ = perturb(cloud, "identity", 0)
                identical, _, identity_grid = infer_window(model, cfg, identity_raw, [], store, logger, observer=False)
                if not np.array_equal(clean, identical) or not np.array_equal(
                    grid["feats_u"], identity_grid["feats_u"]
                ):
                    raise ValueError("Identity probe changed logits or voxel inputs")
                original_pred = maps[key][clean.argmax(1)]
                entries = []
                for mode in ("return_shuffle", "thin50"):
                    for seed in settings["probe_seeds"]:
                        store.heartbeat()
                        changed_raw, retained = perturb(cloud, mode, seed)
                        changed, _, changed_grid = infer_window(
                            model, cfg, changed_raw, [], store, logger, observer=False
                        )
                        pred = maps[key][changed.argmax(1)]
                        reference = original_pred[retained]
                        valid = gt[retained] > 0
                        entries.append(
                            {
                                "mode": mode,
                                "seed": seed,
                                "points_before": len(gt),
                                "points_after": len(retained),
                                "clean_on_retained": summary(confusion(gt[retained], reference)),
                                "perturbed": summary(confusion(gt[retained], pred)),
                                "prediction_change_by_class": np.bincount(
                                    gt[retained][valid],
                                    weights=(pred[valid] != reference[valid]).astype(np.int64),
                                    minlength=8,
                                )
                                .astype(int)
                                .tolist(),
                                "effective_change": effective_change(grid, changed_grid),
                            }
                        )
                        del changed, changed_raw, changed_grid
                if before != state_digest(model):
                    raise ValueError("Probe updated checkpoint weights/buffers")
                row["models"][key] = {
                    "identity_control_passed": True,
                    "state_unchanged": True,
                    "clean": summary(confusion(gt, original_pred)),
                    "probes": entries,
                }
                del cloud, clean, grid, identical, identity_grid
            store.json(marker, row)
            print(f"Probes {domain}: window {wi + 1}/{len(windows[domain + '/test'])}", flush=True)
        print(f"Paired probes {domain}: complete", flush=True)


def bind_manifest(out, settings, tasks, identities, resume):
    root = Path(__file__).resolve().parents[1]
    files = (
        sorted((root / "src").glob("*.py"))
        + sorted(ASSETS.glob("*.json"))
        + sorted((root / "configs").glob("mapping*.yaml"))
        + sorted((root / "configs").glob("train_id_to_common*.yaml"))
        + [
            root / "scripts/run_transfer_evidence_study.py",
            root / "scripts/submit_transfer_evidence_study.py",
            root / "scripts/slurm_transfer_evidence_study.sh",
        ]
    )
    manifest = {
        "version": 1,
        "settings": settings,
        "tasks": tasks,
        "datasets": identities,
        "code": {str(p.relative_to(root)): sha256(p) for p in files},
        "target_fitting_or_checkpoint_selection": False,
    }
    identity = fingerprint(manifest)
    path = Path(out) / "study_manifest.json"
    if path.exists():
        if not resume or read_json(path)["identity"] != identity:
            raise ValueError("Resume requires identical code, checkpoint, mapping, config and dataset identities")
    elif resume:
        raise ValueError("No manifest to resume")
    elif any(p.name != "logs" for p in Path(out).iterdir()):
        raise ValueError("Refuse existing output contents")
    return identity, {"identity": identity, **manifest}


def gpu_preflight(store, datasets, tasks, models, logger):
    import torch
    from scipy.spatial import cKDTree

    rows = []
    for domain in ("dales", "eclair"):
        raw = datasets[(domain, "test")][0].get_raw(0)
        for task in tasks:
            store.heartbeat()
            ids = window_ids(raw, {"x0_m": 0.0, "y0_m": 0.0}, task["cfg"])
            cloud = take_raw(raw, ids)
            anchors = np.linspace(0, len(ids) - 1, min(32, len(ids)), dtype=np.int64)
            before = state_digest(models[task["id"]])
            tick = time.monotonic()
            clean, _, _ = infer_window(models[task["id"]], task["cfg"], cloud, [], store, logger, observer=False)
            observed, tap, _ = infer_window(models[task["id"]], task["cfg"], cloud, anchors, store, logger)
            if not np.array_equal(clean, observed) or before != state_digest(models[task["id"]]):
                raise ValueError("GPU observer/state parity failed before the collection")
            if not np.isfinite(tap.features()).all():
                raise ValueError("GPU source feature tap is invalid")
            rows.append(
                {
                    "model": task["id"],
                    "domain": domain,
                    "seconds": time.monotonic() - tick,
                    "points": len(ids),
                    "observer_logit_max_abs_error": 0.0,
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            del clean, observed, cloud, tap
        del raw
    # Confirm scientific-neighbourhood dependency before traversing all caches.
    assert cKDTree(np.eye(3)).query(np.zeros(3))[0] > 0
    store.json(
        "gpu_preflight.json",
        {"checks": rows, "all_passed": True, "runtime_is_measured_on_windows_not_a_full_study_eta": True},
    )


def run_study(output, resume=False, deadline=float("inf")):
    from .transfer_evidence_report import create_report, verify_canonical

    started, bound = time.monotonic(), False
    settings, catalog = read_json(ASSETS / "study.json"), read_json(ASSETS / "scan_catalog.json")
    out = safe_output(output)
    out.mkdir(parents=True, exist_ok=True)
    store = Store(out, settings, deadline)
    logger = logging.getLogger("transfer_evidence")
    logger.setLevel(logging.WARNING)
    logger.addHandler(logging.StreamHandler())
    try:
        tasks = verify_checkpoints(settings)
        datasets, identities = {}, {}
        for domain in ("dales", "eclair"):
            for split in ("train", "test"):
                ds, lut, identity = load_dataset(domain, split, tasks[0]["cfg"], logger)
                if getattr(ds, "write_cache", False) or getattr(ds, "_run_cache_dir", None) is not None:
                    raise ValueError("Diagnostic datasets must not write shared or run caches")
                if len(ds) != settings[f"{split}_tiles"][domain]:
                    raise ValueError("Dataset membership count changed")
                datasets[(domain, split)] = ds, lut, identity
                identities[domain + "/" + split] = identity
        store.identity, manifest = bind_manifest(out, settings, tasks, identities, resume)
        bound = True
        store.json("study_manifest.json", manifest)
        from .transfer_evidence_sampling import GEOMETRY_NAMES

        store.json(
            "measurement_schema.json",
            {
                "geometry_variables": list(GEOMETRY_NAMES),
                "radii_m": settings["radii_m"],
                "sample_flags": {"1": "probability", "2": "native_supplement", "4": "tile_supplement"},
                "raw_return_bins": "0..15 exact; 16 means >=16; invalid separately counted",
                "nearest_other_label_m": "NaN means no other known label within maximum radius; right-censored",
                "minimum_neighbours_for_shape": 8,
                "feature_storage": "float32",
                "common_prediction": "native argmax then mapping; ignore prediction remains FN",
            },
        )
        store.json("STATUS.json", {"status": "running", "identity": store.identity, "stage": "gpu_preflight"})
        models, maps = load_models(tasks, logger)
        from .study_metrics import COMMON_NAMES

        store.json(
            "mapping_manifest.json",
            {
                "common_names": list(COMMON_NAMES),
                "native_to_common": {d: datasets[(d, "test")][1].tolist() for d in ("dales", "eclair")},
                "model_train_to_common": {k: v.tolist() for k, v in maps.items()},
                "native_mapping_documents": {
                    d: (ASSETS.parent / f"mapping_{d}_to_common.yaml").read_text() for d in ("dales", "eclair")
                },
            },
        )
        gpu_preflight(store, datasets, tasks, models, logger)
        store.json("STATUS.json", {"status": "running", "identity": store.identity, "stage": "geometry"})
        banks, windows = geometry_stage(store, datasets, catalog, settings)
        store.json("STATUS.json", {"status": "running", "identity": store.identity, "stage": "canonical"})
        full_inference(store, datasets, banks, tasks, models, maps, settings, logger)
        verify_canonical(out, tasks, settings)  # No interpretation of unreproduced scores.
        store.json("STATUS.json", {"status": "running", "identity": store.identity, "stage": "source_features"})
        source_features(store, datasets, windows, tasks, models, settings, logger)
        store.json("STATUS.json", {"status": "running", "identity": store.identity, "stage": "probes"})
        paired_probes(store, datasets, windows, tasks, models, maps, settings, logger)
        del models
        gc.collect()
        store.json("STATUS.json", {"status": "running", "identity": store.identity, "stage": "report"})
        coverage = create_report(store, tasks, banks, windows, settings)
        import resource

        import torch

        store.json(
            "resource_summary.json",
            {
                "elapsed_seconds_this_allocation": time.monotonic() - started,
                "max_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "output_bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
                "existing_files_deleted": False,
            },
        )
        store.json(
            "completion.json",
            {
                "identity": store.identity,
                "required_measurements": coverage,
                "report_sha256": sha256(out / "Research_Decision_Report.md"),
            },
        )
        store.json(
            "STATUS.json",
            {
                "status": "complete",
                "identity": store.identity,
                "completed_at_unix": time.time(),
                "all_required_stages_complete": True,
            },
        )
        print("Transfer evidence study complete. Read Research_Decision_Report.md", flush=True)
    except NeedsResume as exc:
        # A signal must not prevent the small terminal status write.
        store.stop, store.deadline = False, float("inf")
        store.json("STATUS.json", {"status": "needs_resume", "reason": str(exc)})
        print(str(exc), flush=True)
    except Exception as exc:
        if not bound:
            print("Input/resume preflight failed; existing study state was not changed", flush=True)
            raise
        store.stop, store.deadline = False, float("inf")
        try:
            store.json("STATUS.json", {"status": "failed", "reason": repr(exc), "all_required_stages_complete": False})
        except Exception:
            print("Cannot persist failure status; inspect Slurm error log", flush=True)
        raise
