"""Fixed, error-independent samples and paired observation probes. CPU only."""

from __future__ import annotations

import hashlib

import numpy as np

GEOMETRY_NAMES = (
    "count",
    "eigenvalue_large",
    "eigenvalue_middle",
    "eigenvalue_small",
    "linearity",
    "planarity",
    "scattering",
    "principal_vertical",
    "normal_vertical",
    "z_extent",
    "class_purity",
    "ignored_fraction",
    "edge_truncated",
    "nearest_other_label_m",
) + tuple("neighbour_common_fraction_" + str(i) for i in range(8))


def seed_for(*parts):
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:16], 16)


def payload_digest(raw):
    h = hashlib.sha256()
    for key in ("xyz", "native_labels", "return_number", "number_of_returns"):
        if raw.get(key) is not None:
            arr = np.ascontiguousarray(raw[key])
            h.update(key.encode())
            h.update(str((arr.shape, str(arr.dtype))).encode())
            h.update(memoryview(arr).cast("B"))
    return h.hexdigest()


def allocate(counts, budget, seed):
    """Hypergeometric allocation equals a uniform sample over the pooled class."""
    counts = np.asarray(counts, dtype=np.int64)
    if np.any(counts < 0) or budget < 0:
        raise ValueError("Invalid population or sample size")
    rng = np.random.default_rng(seed)
    remaining = min(int(budget), int(counts.sum()))
    total = int(counts.sum())
    out = np.zeros_like(counts)
    for i, size in enumerate(counts):
        out[i] = rng.hypergeometric(int(size), total - int(size), remaining) if remaining and total else 0
        remaining -= int(out[i])
        total -= int(size)
    assert remaining == 0
    return out


def sampling_plan(catalog, domain, split, cfg):
    rows = [catalog[f"{domain}/{split}/{i}"] for i in range(cfg[f"{split}_tiles"][domain])]
    common = np.asarray([r["common_histogram"] for r in rows], dtype=np.int64)
    native = np.asarray([r["native_histogram"] for r in rows], dtype=np.int64)
    quotas, subquotas = np.zeros_like(common), np.zeros_like(native)
    totals = common.sum(0)
    for cls in range(1, 8):
        quotas[:, cls] = allocate(common[:, cls], cfg["class_points"], seed_for(cfg["seed"], domain, split, cls))
    for cls in np.flatnonzero(native.sum(0)):
        subquotas[:, cls] = allocate(native[:, cls], cfg["subtype_points"], seed_for("native", domain, split, int(cls)))
    ceiling = int(quotas.sum() + subquotas.sum() + (common[:, 1:] > 0).sum())
    if ceiling > cfg["anchor_cap"]:
        raise ValueError(f"Frozen anchor design exceeds cap: {domain}/{split} {ceiling}")
    return {
        "class_quotas": quotas.tolist(),
        "native_quotas": subquotas.tolist(),
        "class_population": totals.tolist(),
        "maximum_anchors": ceiling,
    }


def sample_tile(gt, native, plan, tile, seed):
    """Flags: 1 probability sample; 2 native supplement; 4 tile supplement."""
    gt, native = np.asarray(gt), np.asarray(native)
    selected, flags, probabilities = {}, {}, {}
    for cls in range(1, 8):
        ids = np.flatnonzero(gt == cls)
        n = int(plan["class_quotas"][tile][cls])
        if n > len(ids):
            raise ValueError("Raw class support differs from audited scan")
        rng = np.random.default_rng(seed_for(seed, tile, cls, "probability"))
        chosen = rng.choice(ids, size=n, replace=False) if n else []
        p = min(1.0, sum(r[cls] for r in plan["class_quotas"]) / max(1, plan["class_population"][cls]))
        for idx in chosen:
            idx = int(idx)
            selected[idx], flags[idx], probabilities[idx] = True, flags.get(idx, 0) | 1, p
        if len(ids):
            idx = int(ids[np.random.default_rng(seed_for(seed, tile, cls, "tile")).integers(len(ids))])
            selected[idx], flags[idx] = True, flags.get(idx, 0) | 4
    for cls, n in enumerate(plan["native_quotas"][tile]):
        if not n:
            continue
        ids = np.flatnonzero(native == cls)
        rng = np.random.default_rng(seed_for(seed, tile, cls, "native"))
        for idx in rng.choice(ids, size=n, replace=False):
            idx = int(idx)
            selected[idx], flags[idx] = True, flags.get(idx, 0) | 2
    ids = np.asarray(sorted(selected), dtype=np.int64)
    return (
        ids,
        np.asarray([flags[int(i)] for i in ids], np.uint8),
        np.asarray([probabilities.get(int(i), 0.0) for i in ids], np.float64),
    )


def geometry(raw_xyz, gt, anchor_ids, radii, heartbeat=lambda: None, tree=None):
    """Query full raw neighbourhoods; never downsample before computing support."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(raw_xyz, dtype=np.float64)
    tree = tree if tree is not None else cKDTree(xyz, copy_data=False)
    lower, upper = xyz.min(0), xyz.max(0)
    out = np.full((len(anchor_ids), len(radii), len(GEOMETRY_NAMES)), np.nan, np.float32)
    for i, idx in enumerate(anchor_ids):
        if i % 32 == 0:
            heartbeat()
        ids = np.asarray(tree.query_ball_point(xyz[idx], max(radii)), dtype=np.int64)
        delta = xyz[ids] - xyz[idx]
        dist2 = np.einsum("ij,ij->i", delta, delta)
        other = (gt[ids] > 0) & (gt[ids] != gt[idx])
        nearest = float(np.sqrt(dist2[other].min())) if other.any() else np.nan
        for j, radius in enumerate(radii):
            keep = dist2 <= radius**2
            points, labels = xyz[ids[keep]], gt[ids[keep]]
            edge = bool(np.any(xyz[idx, :2] - lower[:2] < radius) or np.any(upper[:2] - xyz[idx, :2] < radius))
            row = out[i, j]
            row[0], row[12], row[13] = len(points), edge, nearest
            row[9] = np.ptp(points[:, 2]) if len(points) else np.nan
            known = labels > 0
            row[10] = np.mean(labels[known] == gt[idx]) if known.any() and gt[idx] else np.nan
            row[11] = np.mean(~known) if len(labels) else np.nan
            if len(labels):
                row[14:22] = np.bincount(labels, minlength=8) / len(labels)
            if len(points) < 8:
                continue
            centered = points - points.mean(0)
            values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
            values = np.maximum(values[::-1], 0)
            row[1:4] = values
            if values[0] > 1e-12:
                row[4:7] = [
                    (values[0] - values[1]) / values[0],
                    (values[1] - values[2]) / values[0],
                    values[2] / values[0],
                ]
                row[7:9] = abs(vectors[2, -1]), abs(vectors[2, 0])
    return out


def geometry_descriptor(values, radius_index=2):
    """A declared input-support descriptor; excludes all label-derived columns."""
    a = np.asarray(values)[:, radius_index]  # default physical 2 m radius
    result = np.column_stack((np.log1p(a[:, 0]), a[:, 4:10]))
    result[a[:, 12] != 0] = np.nan  # a clipped tile neighbourhood is not absent source support
    return result


def choose_windows(anchor_groups, count, seed):
    """Fixed class-covering windows, selected without predictions or confidences."""
    choices = []
    for tile, ids, xyz, gt in anchor_groups:
        for i, point in enumerate(ids):
            if gt[i] > 0:
                x0, y0 = np.maximum(0, np.floor(xyz[i, :2] - 60)).tolist()
                choices.append(
                    {"tile": int(tile), "point": int(point), "x0_m": float(x0), "y0_m": float(y0), "class": int(gt[i])}
                )
    rng = np.random.default_rng(seed)
    rng.shuffle(choices)
    used, out = set(), []
    for slot in range(count):
        wanted = slot % 7 + 1
        candidates = [c for c in choices if c["class"] == wanted and (c["tile"], c["x0_m"], c["y0_m"]) not in used]
        if not candidates:
            candidates = [c for c in choices if (c["tile"], c["x0_m"], c["y0_m"]) not in used]
        if not candidates:
            break
        item = candidates[0]
        used.add((item["tile"], item["x0_m"], item["y0_m"]))
        out.append(item)
    return out


def take_raw(raw, ids):
    n = len(raw["xyz"])
    return {
        k: v[ids] if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v
        for k, v in raw.items()
        if k not in ("return_number_1h", "number_of_returns_1h")
    }


def perturb(raw, mode, seed):
    rng = np.random.default_rng(seed)
    ids = np.arange(len(raw["xyz"]), dtype=np.int64)
    if mode == "thin50":
        ids = np.sort(rng.choice(ids, max(1, len(ids) // 2), replace=False))
        return take_raw(raw, ids), ids
    result = take_raw(raw, ids)
    if mode == "return_shuffle":
        rn, nor = np.asarray(raw["return_number"]), np.asarray(raw["number_of_returns"])
        valid = np.flatnonzero(np.isfinite(rn) & np.isfinite(nor) & (rn >= 1) & (nor >= rn))
        perm = rng.permutation(valid)
        result["return_number"], result["number_of_returns"] = rn.copy(), nor.copy()
        result["return_number"][valid], result["number_of_returns"][valid] = rn[perm], nor[perm]
    elif mode != "identity":
        raise ValueError("Unknown, unregistered probe")
    return result, ids


def confidence_arrays(logits, train_to_common):
    values = np.asarray(logits, dtype=np.float32)
    exp = np.exp(values - values.max(1, keepdims=True))
    probabilities = exp / exp.sum(1, keepdims=True)
    pred = values.argmax(1)
    top = np.partition(values, -2, axis=1)[:, -2:]
    common = np.zeros((len(values), 8), np.float32)
    for cls, mapped in enumerate(train_to_common):
        common[:, mapped] += probabilities[:, cls]
    mapped = np.asarray(train_to_common)[pred]
    return {
        "pred_native": pred,
        "pred": mapped,
        "native_confidence": probabilities.max(1),
        "mapped_class_probability": common[np.arange(len(values)), mapped],
        "native_logit_margin": top[:, 1] - top[:, 0],
        "common_probability_argmax": common.argmax(1),
    }


def conditional_inputs(gt, pred, rn, nor, classes=8, ignore_zero=True):
    """Exact raw-value bins 0..15 and >=16; separate validity counters."""
    rn, nor = np.asarray(rn), np.asarray(nor)
    finite = np.isfinite(rn) & np.isfinite(nor)
    a = np.clip(np.where(finite, rn, 0), 0, 16).astype(np.int64)
    b = np.clip(np.where(finite, nor, 0), 0, 16).astype(np.int64)
    valid = (gt >= (1 if ignore_zero else 0)) & (gt < classes)
    key = ((gt[valid] * 8 + pred[valid]) * 17 + a[valid]) * 17 + b[valid]
    joint = np.bincount(key, minlength=classes * 8 * 17 * 17).reshape(classes, 8, 17, 17)
    invalid = (~finite) | (rn < 1) | (nor < rn) | (rn != np.floor(rn)) | (nor != np.floor(nor))
    bad = np.bincount(gt[valid & invalid] * 8 + pred[valid & invalid], minlength=classes * 8).reshape(classes, 8)
    return joint, bad
