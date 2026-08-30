# /src/eval_dales_preproc_sweep_wo_intensity.py
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml

from .config_loader import load_yaml
from .model import build_model
from .utils import read_las_arrays_robust

try:
    import MinkowskiEngine as ME
except Exception as e:
    raise RuntimeError("MinkowskiEngine is required.") from e

# CUDA AMP autocast helper: prefer torch.cuda.amp.autocast when available,
# otherwise fall back to a no-op contextmanager to support CPU-only or older PyTorch.
from contextlib import nullcontext

try:
    from torch.cuda.amp import autocast as _torch_autocast  # type: ignore
except Exception:
    _torch_autocast = None


def _maybe_autocast(enabled: bool):
    if _torch_autocast is None:
        return nullcontext()
    return _torch_autocast(enabled=enabled)


def _cfg_digest(cfg: dict) -> dict:
    """Keep only fields that must match between train-time cfg and eval-time cfg."""
    d = {}
    # model contract
    d["model.in_channels"] = cfg.get("model", {}).get("in_channels")
    d["model.out_channels"] = cfg.get("model", {}).get("out_channels")
    d["model.D"] = cfg.get("model", {}).get("D")

    # dataset/label-space contract
    data = cfg.get("data", {})
    d["data.dataset"] = data.get("dataset")
    ls = data.get("label_space", {})
    d["label_space.num_classes"] = ls.get("num_classes")
    d["label_space.ignore_index"] = ls.get("ignore_index")
    d["label_space.class_names"] = ls.get("class_names")

    # feature contract (anything that changes channel semantics)
    feat = data.get("features", {})
    d["features.use_intensity"] = feat.get("use_intensity")
    d["features.returns_onehot_k"] = feat.get("returns_onehot_k")
    d["features.use_rgb"] = feat.get("use_rgb")
    d["features.include_coords"] = feat.get("include_coords")
    return d


def _assert_cfg_matches_ckpt(eval_cfg: dict, ckpt: dict):
    train_cfg = ckpt.get("cfg", None)
    if train_cfg is None:
        print("[warn] ckpt has no 'cfg' – cannot verify config/label-space contract.")
        return

    # ckpt['cfg'] might be OmegaConf; try to coerce
    if not isinstance(train_cfg, dict):
        try:
            # OmegaConf supports .to_container(resolve=True)
            train_cfg = train_cfg.to_container(resolve=True)
        except Exception:
            pass
    if not isinstance(train_cfg, dict):
        print(f"[warn] ckpt['cfg'] is {type(train_cfg)} – cannot verify contract.")
        return

    a = _cfg_digest(eval_cfg)
    b = _cfg_digest(train_cfg)

    mismatches = {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}

    if mismatches:
        print("\n[ERROR] Eval config does NOT match checkpoint train config for critical fields:")
        print(json.dumps(mismatches, indent=2, default=str))
        raise SystemExit(2)


# -------------------------
# Logging utilities
# -------------------------
def setup_logger(out_dir: Path, name: str, level=logging.INFO) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(out_dir / "run.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Train-space LUT loader (native -> train) from checkpoint cfg
# -------------------------
def load_native_to_train_from_ckpt(
    ckpt_path: str,
    logger: logging.Logger,
) -> Tuple[np.ndarray, int, Optional[int], bool, Tuple[int, ...]]:
    """
    Returns:
      lut_native_to_train: (256,) int64
      train_ignore_index: int
      train_max_voxels: Optional[int]
      train_class_aware_max_voxels: bool
      rare_class_ids: Tuple[int,...]

    We try to read from ckpt["cfg"]["data"] using common key patterns.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_raw = ckpt.get("cfg", None)
    cfg = _to_plain_dict_cfg(cfg_raw)
    if cfg is None:
        raise RuntimeError("Checkpoint has no usable 'cfg' dict; cannot derive native->train LUT.")

    data = cfg.get("data", {})
    if not isinstance(data, dict):
        raise RuntimeError("Checkpoint cfg['data'] missing or not a dict; cannot derive LUT.")

    ls = data.get("label_space", {}) if isinstance(data.get("label_space", {}), dict) else {}
    train_ignore_index = int(ls.get("ignore_index", -100))

    # preproc knobs (optional)
    pre = data.get("preproc", {}) if isinstance(data.get("preproc", {}), dict) else {}
    train_max_voxels = pre.get("max_voxels", None)
    train_max_voxels = int(train_max_voxels) if train_max_voxels is not None else None
    train_class_aware_max_voxels = bool(pre.get("class_aware_max_voxels", False))
    rare_class_ids = tuple(int(x) for x in pre.get("rare_class_ids", [2, 3, 5, 6, 7]))

    # Mapping dict search (most specific first)
    cand = None
    direct_keys = [
        "dales_label_map_native_to_train",
        "label_map_native_to_train",
        "native_label_map_to_train",
        "native_to_train",
    ]
    for k in direct_keys:
        v = data.get(k, None)
        if isinstance(v, dict) and v:
            cand = v
            logger.info(f"[trainmap] using cfg.data.{k} for native->train LUT ({len(v)} entries)")
            break

    # Also allow nested label_maps
    if cand is None:
        lm = data.get("label_maps", None)
        if isinstance(lm, dict):
            for k, v in lm.items():
                if not isinstance(v, dict) or not v:
                    continue
                if re.search(r"native.*train", str(k)):
                    cand = v
                    logger.info(f"[trainmap] using cfg.data.label_maps.{k} for native->train LUT ({len(v)} entries)")
                    break

    if cand is None:
        raise RuntimeError(
            "Could not find native->train label map in checkpoint cfg. "
            "Expected something like cfg.data.dales_label_map_native_to_train (or similar)."
        )

    lut = np.full((256,), train_ignore_index, dtype=np.int64)
    for k, v in cand.items():
        kk = int(k)
        if 0 <= kk < 256:
            lut[kk] = int(v)

    return lut, train_ignore_index, train_max_voxels, train_class_aware_max_voxels, rare_class_ids


# -------------------------
# YAML mapping loaders
# -------------------------
def load_yaml_map_int_dict(path: str) -> Dict[int, int]:
    mp = yaml.safe_load(Path(path).read_text())
    out: Dict[int, int] = {}
    for k, v in mp.items():
        out[int(k)] = int(v)
    return out


def load_yaml_lut_fixed(path: str, *, size: int, default: int = 0) -> np.ndarray:
    """
    Fixed-size LUT with safe default.
    - For DALES native labels: use size=256, default=0 (ignore)
    - For pred_to_common override: use size=out_channels, default=0 (ignore)
    """
    mp = yaml.safe_load(Path(path).read_text())
    lut = np.full((size,), int(default), dtype=np.int64)
    for k, v in mp.items():
        kk = int(k)
        if 0 <= kk < size:
            lut[kk] = int(v)
    return lut


def build_pred_to_common_from_eclair_map(
    eclair_native_to_common: Dict[int, int],
    *,
    out_channels: int,
) -> np.ndarray:
    """
    Model predicts train_id in [0..out_channels-1].
    ECLAIR native ids are expected to be train_id + 1.
    """
    pred_to_common = np.zeros((out_channels,), dtype=np.int64)
    for train_id in range(out_channels):
        native_id = train_id + 1
        pred_to_common[train_id] = int(eclair_native_to_common.get(native_id, 0))
    return pred_to_common


def infer_ckpt_domain(ckpt_path: str, *, out_channels: Optional[int] = None, logger: Optional[logging.Logger] = None) -> str:
    """
    Best-effort detection of the *source domain* the checkpoint was trained on.
    We need this only to pick the correct native->common LUT (DALES vs ECLAIR).
    """
    domain = "unknown"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("cfg", {})
        data = cfg.get("data", {}) if isinstance(cfg, dict) else {}
        if isinstance(data, dict):
            # Strong signals (explicit label map keys)
            if "dales_label_map_native_to_train" in data:
                domain = "dales"
            elif "eclair_label_map_native_to_train" in data:
                domain = "eclair"
            else:
                # Weaker signals (dataset name or paths)
                ds = str(data.get("dataset", "")).lower()
                if "dales" in ds:
                    domain = "dales"
                elif "eclair" in ds:
                    domain = "eclair"
    except Exception:
        pass

    # Heuristic fallback based on class count if still unknown
    if domain == "unknown" and out_channels is not None:
        if int(out_channels) == 8:
            domain = "dales"
        elif int(out_channels) == 11:
            domain = "eclair"

    if logger is not None:
        logger.info(f"[ckpt] inferred_source_domain={domain} (out_channels={out_channels})")
    return domain


def build_pred_to_common_from_ckpt_lut(
    *,
    native_to_train_lut: np.ndarray,  # (256,) source native -> train id
    native_to_common_lut: np.ndarray,  # (256,) source native -> common id
    out_channels: int,
    train_ignore_index: int,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Derive pred_to_common (train_id -> common_id) from the checkpoint's native->train LUT.

    Steps:
      1) invert native->train into train->native (first-hit wins)
      2) compose with native->common to get train->common
    """
    train_to_native = np.full((out_channels,), -1, dtype=np.int64)
    for nid in range(int(native_to_train_lut.shape[0])):
        tid = int(native_to_train_lut[nid])
        if tid == int(train_ignore_index):
            continue
        if tid < 0 or tid >= out_channels:
            continue
        if train_to_native[tid] != -1 and train_to_native[tid] != nid:
            logger.warning(f"[map] train_id {tid} maps to multiple native ids ({train_to_native[tid]}, {nid}); keeping first.")
            continue
        train_to_native[tid] = nid

    pred_to_common = np.zeros((out_channels,), dtype=np.int64)  # default common ignore=0
    for tid in range(out_channels):
        nid = int(train_to_native[tid])
        if nid == -1:
            pred_to_common[tid] = 0
        else:
            pred_to_common[tid] = int(native_to_common_lut[nid])

    logger.info(f"[map] pred_to_common(train->common)={pred_to_common.tolist()}")
    return pred_to_common


def _sha1_file(path: str, chunk_bytes: int = 8 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _strip_module_prefix(state_dict: dict) -> dict:
    # Handles DDP checkpoints that save keys as "module.xxx"
    if not state_dict:
        return state_dict

    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


# -------------------------
# Metrics
# -------------------------
def confusion_from_labels(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    ignore_index: int = 0,
) -> np.ndarray:
    assert gt.shape == pred.shape
    mask = gt != ignore_index
    gt = gt[mask].astype(np.int64, copy=False)
    pred = pred[mask].astype(np.int64, copy=False)

    gt = np.clip(gt, 0, num_classes - 1)
    pred = np.clip(pred, 0, num_classes - 1)

    idx = gt * num_classes + pred
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape((num_classes, num_classes))


def iou_from_confusion(
    cm: np.ndarray,
    ignore_index: Optional[int] = 0,
    *,
    ignore_empty: bool = False,  # if True, exclude classes with denom==0 from mIoU
) -> Tuple[float, List[float]]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    denom = tp + fp + fn  # note: no epsilon here
    iou = np.divide(tp, np.maximum(denom, 1e-12))

    if ignore_empty:
        iou = iou.copy()
        iou[denom == 0] = np.nan

    if ignore_index is not None:
        if not (0 <= ignore_index < cm.shape[0]):
            raise ValueError(f"ignore_index={ignore_index} out of range for cm shape {cm.shape}")
        iou = iou.copy()
        iou[ignore_index] = np.nan

    miou = float(np.nanmean(iou))
    return miou, iou.tolist()


def read_dales_las(path: Path) -> Dict[str, np.ndarray]:
    """
    Robust reader wrapper for DALES inference.
    Matches old output signature: keys = [xyz, intensity, return_number, number_of_returns, cls]
    """
    data = read_las_arrays_robust(path)

    # Preserve original logging behavior
    # print(
    #     "DALES intensity stats:",
    #     float(data["intensity"].min()),
    #     float(data["intensity"].max()),
    #     float(np.median(data["intensity"])),
    # )

    return {
        "xyz": data["xyz"],
        "intensity": data["intensity"],
        "return_number": data["return_number"],
        "number_of_returns": data["number_of_returns"],
        "cls": data["native_labels"],  # <--- Renamed 'native_labels' to 'cls' to match your old code
    }


# -------------------------
# Patch iterator (non-overlapping default)
# -------------------------
def iter_xy_patches_indices(xyz: np.ndarray, patch_size_m: float, patch_stride_m: float) -> Iterable[np.ndarray]:
    x = xyz[:, 0]
    y = xyz[:, 1]
    xmin, ymin = float(x.min()), float(y.min())

    gx = np.floor((x - xmin) / patch_stride_m).astype(np.int32)
    gy = np.floor((y - ymin) / patch_stride_m).astype(np.int32)

    key = (gx.astype(np.int64) << 32) | (gy.astype(np.int64) & 0xFFFFFFFF)
    order = np.argsort(key, kind="mergesort")
    key_sorted = key[order]

    start = 0
    n = len(order)
    while start < n:
        k = key_sorted[start]
        end = start + 1
        while end < n and key_sorted[end] == k:
            end += 1

        idx = order[start:end]
        cx = int(gx[idx[0]])
        cy = int(gy[idx[0]])
        x0 = xmin + cx * patch_stride_m
        y0 = ymin + cy * patch_stride_m
        x1 = x0 + patch_size_m
        y1 = y0 + patch_size_m

        inside = idx[(x[idx] >= x0) & (x[idx] < x1) & (y[idx] >= y0) & (y[idx] < y1)]
        if inside.size > 0:
            yield inside

        start = end


# -------------------------
# ECLAIR-consistent feature builder (locks dim=11)
# -------------------------
def _one_hot_1_based(values_1_based: np.ndarray, k: int) -> np.ndarray:
    v = values_1_based.astype(np.int64, copy=False)
    v = np.clip(v, 1, k) - 1
    out = np.zeros((v.shape[0], k), dtype=np.float32)
    out[np.arange(v.shape[0]), v] = 1.0
    return out


# -------------------------
# Optional: linearity proxy (feature surgery / structure-aware reps)
# -------------------------
def compute_linearity(
    xyz: np.ndarray,
    k: int,
    logger: logging.Logger,
    *,
    max_points: int = 50000,
    seed: int = 1337,
) -> Optional[np.ndarray]:
    """
    Returns linearity in [0,1] per point using local covariance eigenvalues.
    Requires sklearn; if missing, returns None.
    """
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
    except Exception:
        logger.warning("[lin] sklearn not available -> linearity-based modes disabled.")
        return None

    n = xyz.shape[0]
    if n < 3:
        return np.zeros((n,), dtype=np.float32)

    rng = np.random.default_rng(seed)
    k_eff = min(k, n)

    def _linearity_core(x: np.ndarray) -> np.ndarray:
        nn = NearestNeighbors(n_neighbors=min(k_eff, x.shape[0]), algorithm="auto")
        nn.fit(x)
        _, inds = nn.kneighbors(x, return_distance=True)
        lin = np.zeros((x.shape[0],), dtype=np.float32)
        for i in range(x.shape[0]):
            neigh = x[inds[i]]
            c = neigh - neigh.mean(axis=0, keepdims=True)
            cov = (c.T @ c) / max(1, c.shape[0])
            w = np.linalg.eigvalsh(cov)
            w = np.sort(w)[::-1]  # λ1>=λ2>=λ3
            l1, l2 = float(w[0]), float(w[1])
            lin[i] = (l1 - l2) / (l1 + 1e-6)
        return np.clip(lin, 0.0, 1.0).astype(np.float32, copy=False)

    # Hard bound: if too many points, compute on a subset and assign via 1-NN
    if n > int(max_points):
        sub = rng.choice(n, size=int(max_points), replace=False)
        xyz_sub = xyz[sub]
        lin_sub = _linearity_core(xyz_sub)
        nn1 = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn1.fit(xyz_sub)
        _, ind1 = nn1.kneighbors(xyz, return_distance=True)
        return lin_sub[ind1[:, 0]].astype(np.float32, copy=False)

    return _linearity_core(xyz)


# -------------------------
# Preprocessing spec (scientifically valid)
# -------------------------
@dataclass(frozen=True)
class PreprocSpec:
    # Baseline matches training:
    # - local coords per patch (always applied in code)
    # - coord_norm_factor default 10
    # - voxel_size default 0.05 (in normalized coords)
    coord_norm_factor: float = 10.0
    voxel_size: float = 0.05  # in normalized space, like training
    z_scale: float = 1.0  # anisotropic scaling on normalized z (generic; targets span_z shift)

    # Feature transforms
    returns_mode: str = "as_is"  # onehot|zero|constant|drop_rn|drop_nor
    returns_k: int = 5

    # Voxel representative strategy (does NOT change eval set)
    # Model input has 1 feature vector per occupied voxel.
    # We choose how to compute that feature vector from all points in the voxel.
    voxel_feat_mode: str = (
        "sample_first"
        # sample_first|sample_random|sample_max_intensity|sample_max_linearity|mean_all|mean_thin
    )
    mean_thin_p: float = 0.7  # used if voxel_feat_mode == mean_thin
    knn_k: int = 16  # for linearity modes
    linearity_max_points: int = 50000
    linearity_seed: int = 1337

    # TTA
    tta_mode: str = "none"  # none|rot4

    # Patch filter (IMPORTANT: does NOT affect "full" metrics, only "filtered" metrics)
    patch_filter_mode: str = "none"  # none|min_points|min_occ_vox
    min_points: int = 1000
    min_occ_vox: int = 1000

    def to_id(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.md5(payload).hexdigest()[:10]


def _canon(v):
    # Canonicalize floats to avoid 0.04 vs 0.04000000000000001 mismatches
    if isinstance(v, float):
        return round(v, 8)
    if isinstance(v, dict):
        return {k: _canon(v[k]) for k in sorted(v.keys())}
    if isinstance(v, list):
        return [_canon(x) for x in v]
    return v


def _parse_csv_floats(s: str) -> List[float]:
    """
    Parse a comma-separated list of floats, e.g. "0.02,0.03,0.04".
    Empty tokens are ignored.
    """
    out: List[float] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def spec_key_from_dict(d: Dict) -> Tuple:
    # Stable tuple key over ALL spec fields
    items = []
    for k in sorted(d.keys()):
        items.append((k, _canon(d[k])))
    return tuple(items)


def spec_key_from_spec(s: PreprocSpec) -> Tuple:
    return spec_key_from_dict(dataclasses.asdict(s))


# -------------------------
# Rotation TTA
# -------------------------
def rotate_z(xyz: np.ndarray, deg: int) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (xyz @ R.T).astype(np.float32, copy=False)


# -------------------------
# EDA-alignment stats per patch
# -------------------------
def patch_alignment_stats(
    xyz_local: np.ndarray,
    xyz_norm: np.ndarray,
    voxel_size: float,
) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    n = xyz_local.shape[0]
    stats["n_points"] = float(n)
    if n == 0:
        return stats

    # z span in meters (local coords)
    z = xyz_local[:, 2]
    stats["z_span_m"] = float(np.max(z) - np.min(z))

    # voxel occupancy
    q = np.floor(xyz_norm / float(voxel_size)).astype(np.int32, copy=False)
    q = np.ascontiguousarray(q, dtype=np.int32)
    keys = q[:, 0].astype(np.int64) * 73856093 ^ q[:, 1].astype(np.int64) * 19349663 ^ q[:, 2].astype(np.int64) * 83492791
    uniq, cnt = np.unique(keys, return_counts=True)
    stats["occ_vox"] = float(len(uniq))
    stats["pts_per_vox_mean"] = float(np.mean(cnt))
    stats["pts_per_vox_med"] = float(np.median(cnt))
    return stats


LINEARITY_VOX_MODES = {"sample_max_linearity"}


def spec_uses_linearity(spec: PreprocSpec) -> bool:
    return spec.voxel_feat_mode in LINEARITY_VOX_MODES


# -------------------------
# Build per-point features (dim=11) then per-voxel features (dim=11)
# -------------------------


def _cap_unique_idx_class_aware(
    *,
    unique_idx: np.ndarray,  # [Nv] indices into patch arrays
    gt_train: np.ndarray,  # [N] train ids per point (incl ignore_index)
    max_voxels: int,
    class_aware: bool,
    rare_ids: list[int],
    ignore_index: int,
    seed: int = 1337,
) -> np.ndarray:
    Nv = int(unique_idx.shape[0])
    if max_voxels <= 0 or Nv <= max_voxels:
        return unique_idx

    rng = np.random.default_rng(seed)

    if not class_aware:
        return rng.choice(unique_idx, size=max_voxels, replace=False)

    # class-aware: always keep all rare-class voxels (up to budget),
    # then fill remaining with uniform random from the rest.
    gt_u = gt_train[unique_idx]  # [Nv]
    rare_mask = np.isin(gt_u, np.array(rare_ids, dtype=gt_u.dtype)) & (gt_u != ignore_index)

    rare_idx = unique_idx[rare_mask]
    nonrare_idx = unique_idx[~rare_mask]

    if rare_idx.size >= max_voxels:
        return rng.choice(rare_idx, size=max_voxels, replace=False)

    remaining = max_voxels - int(rare_idx.size)
    if nonrare_idx.size <= remaining:
        return np.concatenate([rare_idx, nonrare_idx], axis=0)

    fill = rng.choice(nonrare_idx, size=remaining, replace=False)
    return np.concatenate([rare_idx, fill], axis=0)


def build_point_features_returns_only(
    rn: np.ndarray,
    nor: np.ndarray,
    spec: PreprocSpec,
) -> np.ndarray:
    """
    Returns:
      - feats_points: (N, 2 * returns_k)
    Features: [one_hot(return_number, returns_k), one_hot(number_of_returns, returns_k)].
    """

    # returns one-hot like training (k=5)
    rn_oh = _one_hot_1_based(rn, spec.returns_k)
    nor_oh = _one_hot_1_based(nor, spec.returns_k)

    # returns transform (dim preserved)
    if spec.returns_mode == "as_is":
        pass
    elif spec.returns_mode == "drop":
        rn_oh[:] = 0.0
        nor_oh[:] = 0.0
    elif spec.returns_mode == "const1":
        rn_oh[:] = 0.0
        nor_oh[:] = 0.0
        rn_oh[:, 0] = 1.0
        nor_oh[:, 0] = 1.0
    elif spec.returns_mode == "uniform":
        rn_oh[:] = 1.0 / float(spec.returns_k)
        nor_oh[:] = 1.0 / float(spec.returns_k)
    else:
        raise ValueError(f"Unknown returns_mode: {spec.returns_mode}")

    feats_points = np.concatenate([rn_oh, nor_oh], axis=1).astype(np.float32, copy=False)
    expected_dim = 2 * spec.returns_k
    assert feats_points.shape[1] == expected_dim, (
        f"Feature dim mismatch: got {feats_points.shape[1]}, " f"expected {expected_dim} (= 2 * returns_k)."
    )

    return feats_points


# def voxelize_with_inverse(
#     xyz_norm: np.ndarray,
#     voxel_size: float,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Returns:
#       - q_u (M,3) unique voxel coords (int32)
#       - unique_idx (M,) indices of representative points (training-like)
#       - inv (N,) mapping each point -> voxel index in [0..M-1]
#     """
#     q = np.floor(xyz_norm / float(voxel_size)).astype(np.int32, copy=False)
#     q = np.ascontiguousarray(q, dtype=np.int32)
#     q_t = torch.from_numpy(q).int().contiguous()
#     # ME returns:
#     #   - unique coords tensor (unused here)
#     #   - unique indices into original points
#     #   - inverse map point->unique index
#     _, unique_idx_t, inv_t = ME.utils.sparse_quantize(q_t, return_index=True, return_inverse=True)
#     unique_idx = unique_idx_t.cpu().numpy().astype(np.int64, copy=False)
#     inv = inv_t.cpu().numpy().astype(np.int64, copy=False)
#     q_u = q[unique_idx]
#     q_u = np.ascontiguousarray(q_u, dtype=np.int32)
#     return q_u, unique_idx, inv


def voxelize_with_inverse(xyz_norm: np.ndarray, voxel_size: float, *, logger: Optional[logging.Logger] = None):
    q = np.floor(xyz_norm / voxel_size).astype(np.int32, copy=False)
    q = np.ascontiguousarray(q, dtype=np.int32)

    # IMPORTANT: keep coords on CPU for ME utils
    q_t = torch.from_numpy(q).int().contiguous()  # CPU tensor

    # sparse_quantize returns (unique_coords, unique_idx, inverse)
    q_u_t, unique_idx_t, inv_t = ME.utils.sparse_quantize(q_t, return_index=True, return_inverse=True)

    q_u = q_u_t.cpu().numpy().astype(np.int32, copy=False)
    inv = inv_t.cpu().numpy().astype(np.int64, copy=False)

    # ---- HARD sanity: inv maps points -> q_u row order ----
    if q.shape[0] > 0:
        jj = np.random.randint(0, q.shape[0], size=min(2048, q.shape[0]))
        if not np.all(q_u[inv[jj]] == q[jj]):
            raise RuntimeError("voxelize_with_inverse sanity failed: inv does not map points -> q_u ordering.")

    # ---- Build a unique_idx aligned to q_u using inv (stable, deterministic) ----
    # Sort by voxel id; mergesort preserves original point order within each voxel id.
    order = np.argsort(inv, kind="mergesort")
    inv_sorted = inv[order]
    # first index for each voxel id in ascending voxel-id order
    first_pos = np.concatenate(([0], np.flatnonzero(inv_sorted[1:] != inv_sorted[:-1]) + 1))
    unique_idx_from_inv = order[first_pos].astype(np.int64, copy=False)

    # Expect one representative per q_u row
    if unique_idx_from_inv.shape[0] != q_u.shape[0]:
        raise RuntimeError(
            f"unique_idx_from_inv size mismatch: got {unique_idx_from_inv.shape[0]} expected {q_u.shape[0]} "
            "(inv/q_u inconsistency)"
        )

    # ---- HARD invariant: q[unique_idx] must equal q_u row-wise ----
    q_from_unique = q[unique_idx_from_inv]
    if not np.array_equal(q_from_unique, q_u):
        raise RuntimeError(
            "Voxel feature/coord alignment failed: q[unique_idx_from_inv] != q_u. "
            "This would corrupt SparseTensor(feature,row) pairing."
        )

    if logger is not None:
        logger.info(f"[vox] N={q.shape[0]} M={q_u.shape[0]} inv_max={int(inv.max()) if inv.size else -1}")

    return q_u, unique_idx_from_inv, inv


def build_voxel_features_returns_only(
    feats_points: np.ndarray,
    inv: np.ndarray,
    unique_idx: np.ndarray,
    spec: PreprocSpec,
    logger: logging.Logger,
    *,
    linearity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build one feature vector per occupied voxel (M, 2 * returns_k) without changing eval set.
    Modes:
      - sample_first: training-like (use unique_idx)
      - sample_random: pick random point in voxel
      - sample_max_linearity: pick max linearity in voxel (requires linearity)
      - mean_all: mean of all points in voxel
      - mean_thin: mean of random subset in voxel
    """
    M = unique_idx.shape[0]
    if spec.voxel_feat_mode == "sample_first":
        return feats_points[unique_idx]

    # group points by voxel via inv
    order = np.argsort(inv, kind="mergesort")
    inv_sorted = inv[order]
    out = np.zeros((M, feats_points.shape[1]), dtype=np.float32)

    start = 0
    N = inv_sorted.shape[0]
    while start < N:
        v = inv_sorted[start]
        end = start + 1
        while end < N and inv_sorted[end] == v:
            end += 1
        pts_idx = order[start:end]

        if spec.voxel_feat_mode == "sample_random":
            chosen = np.random.choice(pts_idx, size=1)[0]
            out[v] = feats_points[chosen]

        elif spec.voxel_feat_mode == "sample_max_linearity":
            if linearity is None:
                logger.warning("[vox] sample_max_linearity needs linearity -> fallback to sample_first.")
                out[v] = feats_points[unique_idx[v]]
            else:
                chosen = pts_idx[np.argmax(linearity[pts_idx])]
                out[v] = feats_points[chosen]

        elif spec.voxel_feat_mode == "mean_all":
            out[v] = feats_points[pts_idx].mean(axis=0)

        elif spec.voxel_feat_mode == "mean_thin":
            p = float(spec.mean_thin_p)
            if p >= 1.0 or pts_idx.size == 1:
                out[v] = feats_points[pts_idx].mean(axis=0)
            else:
                keep = np.random.rand(pts_idx.size) < p
                if not np.any(keep):
                    keep[np.random.randint(0, pts_idx.size)] = True
                out[v] = feats_points[pts_idx[keep]].mean(axis=0)

        else:
            raise ValueError(f"Unknown voxel_feat_mode: {spec.voxel_feat_mode}")

        start = end

    return out.astype(np.float32, copy=False)


# -------------------------
# Model loading (no hallucination)
# -------------------------


def load_model_state_or_die(model: torch.nn.Module, ckpt: dict, logger) -> None:
    # Your ckpt uses "model_state"
    cand_keys = ["model_state", "model_state_dict", "state_dict", "model", "net"]

    sd = None
    for k in cand_keys:
        v = ckpt.get(k, None)
        if isinstance(v, dict) and len(v) > 0:
            sd = v
            logger.info(f"[ckpt] using state dict from key='{k}' with {len(sd)} tensors")
            break

    if sd is None:
        raise RuntimeError(f"Could not find model state dict in ckpt keys: {list(ckpt.keys())}")

    sd = _strip_module_prefix(sd)

    # STRICT load: fail if mismatch
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # Make it loud: this indicates arch mismatch or wrong checkpoint.
        raise RuntimeError(
            f"State dict mismatch.\n"
            f"  missing ({len(missing)}): {missing[:20]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:20]}"
        )
    logger.info("[ckpt] load_state_dict OK (exact match)")

    # Fingerprint a parameter for sanity
    with torch.no_grad():
        w = next(iter(model.parameters()))
        logger.info(
            f"[ckpt] first_param: shape={tuple(w.shape)} mean={w.float().mean().item():.6g} std={w.float().std().item():.6g}"
        )


def load_model_from_config(
    *,
    config_path: str,
    ckpt_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, bool, int, int]:
    """
    Load model for evaluation.

    - out_channels is taken from the config (must match the classifier head).
    - in_channels is inferred from the checkpoint's first convolution
      (conv0p1s1.kernel / weight) to avoid YAML drift.
    - Returns (model, amp_enabled, in_channels, out_channels).
    """
    cfg = load_yaml(config_path)
    model_cfg = cfg["model"]

    # ---- Out channels from config (this should match training) ----
    out_channels_cfg = int(model_cfg["out_channels"])

    # ---- Load checkpoint state dict ----
    state = torch.load(ckpt_path, map_location="cpu")
    _assert_cfg_matches_ckpt(cfg, state)
    print("[ok] cfg↔ckpt contract verified.")
    if isinstance(state, dict) and "model_state" in state:
        sd = state["model_state"]
    else:
        sd = state

    sd = _strip_module_prefix(sd)
    # ---- Infer in_channels from conv0 kernel in the checkpoint ----
    conv_key = None
    for k in sd.keys():
        # MinkowskiEngine typically uses 'conv0p1s1.kernel' but keep a fallback.
        if "conv0p1s1.kernel" in k or "conv0p1s1.weight" in k:
            conv_key = k
            break

    if conv_key is None:
        raise RuntimeError(
            "Could not find a conv0p1s1 kernel/weight in checkpoint to infer in_channels. "
            f"Available keys include: {list(sd.keys())[:20]}"
        )

    w0 = sd[conv_key]
    if w0.ndim != 3:
        raise RuntimeError(f"Expected conv0 kernel of shape [out, in, k], but got {w0.shape} " f"for key '{conv_key}'.")

    in_channels_ckpt = int(w0.shape[1])

    # Optional: sanity log if YAML in_channels differs from checkpoint.
    in_channels_cfg = int(model_cfg.get("in_channels", in_channels_ckpt))
    if in_channels_cfg != in_channels_ckpt:
        logging.getLogger(__name__).warning(
            f"[model] YAML in_channels={in_channels_cfg} but checkpoint "
            f"conv0 expects in_channels={in_channels_ckpt}. Using checkpoint value."
        )

    # ---- Build model using checkpoint-based in_channels ----
    model = build_model(
        in_channels=in_channels_ckpt,
        out_channels=out_channels_cfg,
        D=int(model_cfg.get("D", 3)),
    ).to(device)

    model.load_state_dict(sd, strict=True)
    model.eval()

    # ---- AMP flag (same logic as before) ----
    amp = bool(cfg.get("run", {}).get("amp", True)) and (device.type == "cuda")

    return model, amp, in_channels_ckpt, out_channels_cfg


# -------------------------
# Infer logits per point (keeps eval set fixed!)
# -------------------------


def _reorder_logits_to_match_coords(out: ME.SparseTensor, coords_in: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    """
    Return out.F reordered so rows align with coords_in row order.
    Works regardless of coords_in being on CPU or CUDA.
    Uses byte-wise row keys (collision-free) on CPU.
    """
    cin = coords_in.detach().cpu().to(torch.int32).contiguous().numpy()  # (N,4)
    cout = out.C.detach().cpu().to(torch.int32).contiguous().numpy()  # (N,4)

    if cout.shape[0] != cin.shape[0]:
        raise RuntimeError(f"Coord count mismatch: out={cout.shape[0]} in={cin.shape[0]}")

    # Build collision-free row keys via raw bytes view
    def rowkey(a: np.ndarray) -> np.ndarray:
        a2 = np.ascontiguousarray(a)
        return a2.view(np.dtype((np.void, a2.dtype.itemsize * a2.shape[1]))).reshape(-1)

    k_in = rowkey(cin)
    k_out = rowkey(cout)

    idx_sort = np.argsort(k_out, kind="mergesort")
    k_out_sorted = k_out[idx_sort]

    pos = np.searchsorted(k_out_sorted, k_in)
    if pos.size != k_in.size:
        raise RuntimeError("searchsorted failed unexpectedly during coord alignment")

    # Verify exact match
    if np.any(k_out_sorted[pos] != k_in):
        raise RuntimeError("Failed to align out.C to input coords (row-key mismatch).")

    idx_out = idx_sort[pos]  # numpy indices into out.F rows
    idx_out_t = torch.from_numpy(idx_out.astype(np.int64, copy=False)).to(device)

    return out.F[idx_out_t]


@torch.inference_mode()
def infer_patch_logits_per_point(
    model: torch.nn.Module,
    xyz_norm: np.ndarray,
    feats_points: np.ndarray,
    spec: PreprocSpec,
    device: torch.device,
    logger: logging.Logger,
    *,
    amp: bool,
    linearity: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Returns logits per ORIGINAL point (N,C) as a torch.Tensor on `device`.
    """
    q_u, unique_idx, inv = voxelize_with_inverse(xyz_norm, spec.voxel_size, logger=logger)

    feats_vox = build_voxel_features_returns_only(
        feats_points=feats_points,
        inv=inv,
        unique_idx=unique_idx,
        spec=spec,
        logger=logger,
        linearity=linearity,
    )

    # ME coords require batch column
    coords = np.concatenate(
        [np.zeros((q_u.shape[0], 1), dtype=np.int32), q_u],
        axis=1,
    )
    coords_t = torch.from_numpy(np.ascontiguousarray(coords, dtype=np.int32)).int().to(device)
    feats_t = torch.from_numpy(np.ascontiguousarray(feats_vox, dtype=np.float32)).float().to(device)

    # if amp and device.type == "cuda":
    #     feats_t = feats_t.half()

    st = ME.SparseTensor(features=feats_t, coordinates=coords_t, device=device)
    # Use CUDA AMP autocast wrapper; when `amp` is False or autocast unavailable this is a no-op.
    with _maybe_autocast(amp):
        out = model(st)

    # Robust alignment: prefer ME native coordinate query if available
    logits_vox = None
    if hasattr(out, "features_at_coordinates"):
        try:
            logits_vox = out.features_at_coordinates(coords_t)
        except Exception:
            logits_vox = None

    if logits_vox is None:
        # Fallback: reorder via explicit coord matching
        logits_vox = _reorder_logits_to_match_coords(out, coords_t, device=device)

    inv_t = torch.from_numpy(inv).long().to(device)
    logits_pts = logits_vox[inv_t]  # (N,C)

    # Drop references ASAP (helps long sweeps)
    del st, out, logits_vox, coords_t, feats_t, inv_t
    return logits_pts


@torch.inference_mode()
def infer_with_tta(
    model: torch.nn.Module,
    xyz_local: np.ndarray,
    rn: np.ndarray,
    nor: np.ndarray,
    spec: PreprocSpec,
    device: torch.device,
    logger: logging.Logger,
    *,
    amp: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Returns pred_train per point (N,) and alignment stats for the *unrotated* geometry.

    Optimized:
      - Build returns-only point features ONCE per patch (they do not depend on xyz).
      - Reuse these features across all TTA rotations.
    """
    # local coords like training
    xyz_centered = xyz_local - xyz_local.min(axis=0, keepdims=True)
    xyz_local = xyz_centered.astype(np.float32, copy=False)

    # normalized coords like training (+ z anisotropic scaling)
    xyz_norm = (xyz_local / float(spec.coord_norm_factor)).astype(np.float32, copy=False)
    xyz_norm = np.ascontiguousarray(xyz_norm, dtype=np.float32)
    xyz_norm[:, 2] *= float(spec.z_scale)

    # compute bounded linearity if needed
    linearity = None
    if spec_uses_linearity(spec):
        linearity = compute_linearity(
            xyz_local,
            spec.knn_k,
            logger,
            max_points=int(spec.linearity_max_points),
            seed=int(spec.linearity_seed),
        )

    # alignment stats for logging (always from unrotated coords)
    stats = patch_alignment_stats(
        xyz_local=xyz_local,
        xyz_norm=xyz_norm,
        voxel_size=spec.voxel_size,
    )

    # ---- Returns-only features: build ONCE per patch ----
    feats_points = build_point_features_returns_only(rn=rn, nor=nor, spec=spec)

    # ---- No TTA: single forward ----
    if spec.tta_mode == "none":
        logits = infer_patch_logits_per_point(
            model=model,
            xyz_norm=xyz_norm,
            feats_points=feats_points,
            spec=spec,
            device=device,
            logger=logger,
            linearity=linearity,
            amp=amp,
        )
        pred_train = torch.argmax(logits, dim=1).to(torch.int64).cpu().numpy()
        del logits
        return pred_train, stats

    # ---- rot4 TTA: 4 rotations, same features, different coords ----
    if spec.tta_mode == "rot4":
        logits_acc = None
        for deg in (0, 90, 180, 270):
            xyz_r = rotate_z(xyz_local, deg)
            xyz_r = xyz_r - xyz_r.min(axis=0, keepdims=True)
            xyz_rn = (xyz_r / float(spec.coord_norm_factor)).astype(np.float32, copy=False)
            xyz_rn = np.ascontiguousarray(xyz_rn, dtype=np.float32)
            xyz_rn[:, 2] *= float(spec.z_scale)

            lg = infer_patch_logits_per_point(
                model=model,
                xyz_norm=xyz_rn,
                feats_points=feats_points,
                spec=spec,
                device=device,
                logger=logger,
                linearity=linearity,
                amp=amp,
            )
            logits_acc = lg if logits_acc is None else (logits_acc + lg)
            del lg

        assert logits_acc is not None
        logits_acc = logits_acc / 4.0
        pred_train = torch.argmax(logits_acc, dim=1).to(dtype=torch.int64).cpu().numpy()
        del logits_acc
        return pred_train, stats

    raise ValueError(f"Unknown tta_mode: {spec.tta_mode}")


def _is_oom_exception(e: BaseException) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cudaerrormemoryallocation" in msg) or ("std::bad_alloc" in msg)


def _split_quadrants(xyz: np.ndarray) -> List[np.ndarray]:
    # xyz is patch-local array (N,3)
    x = xyz[:, 0]
    y = xyz[:, 1]
    xm = float(np.median(x))
    ym = float(np.median(y))

    q1 = (x <= xm) & (y <= ym)
    q2 = (x > xm) & (y <= ym)
    q3 = (x <= xm) & (y > ym)
    q4 = (x > xm) & (y > ym)

    masks = [q1, q2, q3, q4]
    return [np.nonzero(m)[0] for m in masks if np.any(m)]


def infer_with_tta_adaptive(
    *,
    model: torch.nn.Module,
    xyz_patch: np.ndarray,
    rn: np.ndarray,
    nor: np.ndarray,
    spec: PreprocSpec,
    device: torch.device,
    logger: logging.Logger,
    amp: bool,
    max_depth: int = 2,
    _depth: int = 0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Attempt patch inference; if OOM, split into quadrants and recurse.
    Returns pred_train aligned to xyz_patch row order.
    """
    try:
        return infer_with_tta(
            model=model,
            xyz_local=xyz_patch,
            rn=rn,
            nor=nor,
            spec=spec,
            device=device,
            logger=logger,
            amp=amp,
        )
    except (MemoryError, RuntimeError) as e:
        if not _is_oom_exception(e) or device.type != "cuda" or _depth >= max_depth:
            raise

        logger.warning(f"[oom] patch inference OOM at depth={_depth} n={xyz_patch.shape[0]} -> splitting")
        # best effort cleanup before retry
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        parts = _split_quadrants(xyz_patch)
        pred = np.empty((xyz_patch.shape[0],), dtype=np.int64)

        # Aggregate stats loosely (not super important for correctness)
        stats_sum: Dict[str, float] = {}
        stats_n = 0

        for sub_idx in parts:
            p_sub, s_sub = infer_with_tta_adaptive(
                model=model,
                xyz_patch=xyz_patch[sub_idx],
                rn=rn[sub_idx],
                nor=nor[sub_idx],
                spec=spec,
                device=device,
                logger=logger,
                amp=amp,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            pred[sub_idx] = p_sub
            stats_n += 1
            for k, v in s_sub.items():
                stats_sum[k] = stats_sum.get(k, 0.0) + float(v)

        stats_avg = {k: (v / max(1, stats_n)) for k, v in stats_sum.items()}
        return pred, stats_avg


# @torch.no_grad()
# def infer_trainstyle_voxels(
#     *,
#     model: torch.nn.Module,
#     xyz_patch: np.ndarray,  # [N,3] meters
#     rn: np.ndarray,  # [N]
#     nor: np.ndarray,  # [N]
#     spec,  # PreprocSpec
#     device: torch.device,
#     amp: bool,
#     # ---- OPTIONAL: to match training voxel cap exactly ----
#     gt_train_point: Optional[np.ndarray] = None,  # [N] train ids (or ignore)
#     train_ignore_index: int = -100,
#     max_voxels: Optional[int] = None,
#     class_aware_max_voxels: bool = False,
#     rare_class_ids: Tuple[int, ...] = (2, 3, 5, 6, 7),
#     seed: int = 1234,
#     gt_train: np.ndarray | None = None,
#     class_aware: bool = False,
#     rare_ids: list[int] | None = None,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Returns:
#       pred_vox:   [Nv] int64 predictions in train-id space (0..T-1)
#       unique_idx: [Nv] indices of representative points chosen by sparse_quantize (after any cap)
#     """
#     if model.training:
#         raise RuntimeError("Model must be in eval() during evaluation.")

#     # Always eval for inference stability
#     model.eval()
#     torch.set_grad_enabled(False)

#     xyz_patch = np.asarray(xyz_patch)
#     if xyz_patch.ndim != 2 or xyz_patch.shape[1] != 3:
#         raise ValueError(f"xyz_patch must be [N,3], got {xyz_patch.shape}")

#     N = xyz_patch.shape[0]
#     if N == 0:
#         return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

#     # Match training: local coords
#     xyz_local = xyz_patch - xyz_patch.min(axis=0, keepdims=True)

#     # Match training: normalize coords for voxelization
#     xyz_norm = (xyz_local / float(spec.coord_norm_factor)).astype(np.float32, copy=False)

#     # Match training: quantize
#     q = np.floor(xyz_norm / float(spec.voxel_size)).astype(np.int32, copy=False)
#     q = np.ascontiguousarray(q, dtype=np.int32)

#     _, unique_idx = ME.utils.sparse_quantize(q, return_index=True)
#     unique_idx = np.asarray(unique_idx)
#     q_t = torch.from_numpy(q).int().contiguous()
#     _, unique_idx_t = ME.utils.sparse_quantize(q_t, return_index=True)
#     unique_idx = unique_idx_t.cpu().numpy().astype(np.int64, copy=False)

#     if unique_idx.size == 0:
#         return np.zeros((0,), dtype=np.int64), unique_idx

#     # Optional cap (single, consistent path)
#     if max_voxels is not None and unique_idx.size > int(max_voxels):
#         if class_aware_max_voxels and gt_train_point is not None:
#             unique_idx = _cap_unique_idx_class_aware(
#                 unique_idx=unique_idx,
#                 gt_train=np.asarray(gt_train_point),
#                 max_voxels=int(max_voxels),
#                 class_aware=True,
#                 rare_ids=list(rare_class_ids),
#                 ignore_index=int(train_ignore_index),
#                 seed=int(seed),
#             )
#         else:
#             rng = np.random.default_rng(int(seed))
#             unique_idx = rng.choice(unique_idx, size=int(max_voxels), replace=False)

#     # Feature stack (returns-only)
#     unique_idx = np.asarray(unique_idx, dtype=np.int64)

#     k = int(spec.returns_k)

#     rn = np.asarray(rn).astype(np.int64, copy=False)
#     nor = np.asarray(nor).astype(np.int64, copy=False)

#     # Many LAS pipelines can include 0 or >k; one-hot expects 1..k
#     rn = np.clip(rn, 1, k)
#     nor = np.clip(nor, 1, k)

#     feats_points = build_point_features_returns_only(rn=rn, nor=nor, spec=spec)

#     q_u = q[unique_idx]
#     feats_u = feats_points[unique_idx]

#     # shape safety (Nv,3)
#     q_u = np.asarray(q_u)
#     if q_u.ndim == 1:
#         q_u = q_u.reshape(1, -1)
#     if q_u.shape[1] != 3:
#         raise ValueError(f"q_u must be [Nv,3], got {q_u.shape}")

#     # add batch column: [Nv,4]
#     coords_u = np.concatenate(
#         [np.zeros((q_u.shape[0], 1), dtype=np.int32), q_u.astype(np.int32, copy=False)],
#         axis=1,
#     )

#     coords_t = torch.from_numpy(np.ascontiguousarray(coords_u)).int()
#     feats_t = torch.from_numpy(np.ascontiguousarray(feats_u)).float().to(device, non_blocking=True)

#     st = ME.SparseTensor(feats_t, coordinates=coords_t)
#     with _maybe_autocast(amp):
#         logits = model(st).F

#     pred_vox = logits.argmax(dim=1).to("cpu").numpy().astype(np.int64, copy=False)
#     return pred_vox, unique_idx


# def infer_with_tta(
#     model: torch.nn.Module,
#     xyz_local: np.ndarray,
#     rn: np.ndarray,
#     nor: np.ndarray,
#     spec: PreprocSpec,
#     device: torch.device,
#     logger: logging.Logger,
#     *,
#     amp: bool,
# ) -> Tuple[np.ndarray, Dict[str, float]]:
#     """
#     Returns pred_train per point (N,) and alignment stats for the *unrotated* geometry.
#     """
#     # local coords like training
#     xyz_centered = xyz_local - xyz_local.min(axis=0, keepdims=True)
#     xyz_local = xyz_centered.astype(np.float32, copy=False)

#     # normalized coords like training (+ z anisotropic scaling)
#     xyz_norm = (xyz_local / float(spec.coord_norm_factor)).astype(np.float32, copy=False)
#     xyz_norm = np.ascontiguousarray(xyz_norm, dtype=np.float32)
#     xyz_norm[:, 2] *= float(spec.z_scale)

#     # compute bounded linearity if needed
#     linearity = None
#     if spec_uses_linearity(spec):
#         linearity = compute_linearity(
#             xyz_local,
#             spec.knn_k,
#             logger,
#             max_points=int(spec.linearity_max_points),
#             seed=int(spec.linearity_seed),
#         )

#     # alignment stats for logging
#     stats = patch_alignment_stats(
#         xyz_local=xyz_local,
#         xyz_norm=xyz_norm,
#         voxel_size=spec.voxel_size,
#     )

#     if spec.tta_mode == "none":
#         feats_points = build_point_features_returns_only(rn=rn, nor=nor, spec=spec)
#         logits = infer_patch_logits_per_point(
#             model=model,
#             xyz_norm=xyz_norm,
#             feats_points=feats_points,
#             spec=spec,
#             device=device,
#             logger=logger,
#             linearity=linearity,
#             amp=amp,
#         )
#         pred_train = torch.argmax(logits, dim=1).to(torch.int64).cpu().numpy()
#         del logits
#         return pred_train, stats

#     if spec.tta_mode == "rot4":
#         logits_acc = None
#         for deg in (0, 90, 180, 270):
#             xyz_r = rotate_z(xyz_local, deg)
#             xyz_r = xyz_r - xyz_r.min(axis=0, keepdims=True)
#             xyz_rn = (xyz_r / float(spec.coord_norm_factor)).astype(np.float32, copy=False)
#             xyz_rn = np.ascontiguousarray(xyz_rn, dtype=np.float32)
#             xyz_rn[:, 2] *= float(spec.z_scale)

#             # Rebuild features using rotated coords
#             # (proxy_z depends on z; keep consistent)
#             # Reuse linearity if available;
#             # it was computed on xyz_local. For strictness,
#             # recompute on rotated coords only if needed (costly).
#             # We'll keep same linearity.
#             feats_r = build_point_features_returns_only(rn=rn, nor=nor, spec=spec)
#             lg = infer_patch_logits_per_point(
#                 model=model,
#                 xyz_norm=xyz_rn,
#                 feats_points=feats_r,
#                 spec=spec,
#                 device=device,
#                 logger=logger,
#                 linearity=linearity,
#                 amp=amp,
#             )
#             logits_acc = lg if logits_acc is None else (logits_acc + lg)
#             del lg

#         assert logits_acc is not None
#         logits_acc = logits_acc / 4.0
#         pred_train = torch.argmax(logits_acc, dim=1).to(dtype=torch.int64).cpu().numpy()
#         del logits_acc
#         return pred_train, stats

#     raise ValueError(f"Unknown tta_mode: {spec.tta_mode}")


# -------------------------
# Mapping: model output IDs -> common IDs (no hallucination)
# -------------------------
def derive_pred_to_common_map(
    eclair_native_to_common: np.ndarray,
    *,
    try_import: bool,
    undefined_id: int,
    ignore_index: int,
    logger: logging.Logger,
) -> Optional[np.ndarray]:
    """
    Your mapping_eclair_to_common.yaml is native_id -> common_id.
    But the model outputs TRAIN IDs (0..out_channels-1), not native IDs.
    We derive train_id -> common_id by importing eclair_native_to_train_ids from your repo.
    If import fails, return None and require the user to pass --mapping_pred_to_common.
    """
    if not try_import:
        return None
    try:
        # Try a few common import roots (HPC runs often differ in PYTHONPATH)
        candidates = [
            "src.label_maps",
            "eclair_model_train.src.label_maps",
            "label_maps",
        ]
        eclair_native_to_train_ids = None
        for mod in candidates:
            try:
                m = importlib.import_module(mod)
                eclair_native_to_train_ids = getattr(m, "eclair_native_to_train_ids")
                break
            except Exception:
                continue
        if eclair_native_to_train_ids is None:
            raise ImportError(f"Could not import eclair_native_to_train_ids from any of {candidates}")
    except Exception as e:
        logger.warning(f"[map] Could not import src.label_maps.eclair_native_to_train_ids: {e}")
        return None

    native_ids = np.arange(len(eclair_native_to_common), dtype=np.int64)
    train_ids = eclair_native_to_train_ids(native_ids, undefined_id=undefined_id, ignore_index=ignore_index)

    # train_id -> native_id (expect mostly one-to-one for {1..11} -> {0..10})
    train_to_native: Dict[int, int] = {}
    for nid, tid in enumerate(train_ids.tolist()):
        if tid == ignore_index:
            continue
        if tid < 0:
            continue
        # If collisions exist, we keep the first and warn.
        if tid in train_to_native and train_to_native[tid] != nid:
            logger.warning(f"[map] train_id {tid} maps to multiple native ids ({train_to_native[tid]}, {nid}). Keeping first.")
            continue
        train_to_native[tid] = nid

    if not train_to_native:
        logger.warning("[map] Derived empty train_to_native map. Cannot create pred_to_common.")
        return None

    max_tid = max(train_to_native.keys())
    pred_to_common = np.zeros((max_tid + 1,), dtype=np.int64)
    for tid, nid in train_to_native.items():
        nid_clamped = int(np.clip(nid, 0, len(eclair_native_to_common) - 1))
        pred_to_common[tid] = int(eclair_native_to_common[nid_clamped])

    logger.info(f"[map] Derived pred_to_common for train ids 0..{max_tid}")
    return pred_to_common


# -------------------------
# Evaluation
# -------------------------
@dataclass
class EvalTotals:
    cm_full: np.ndarray
    cm_filtered: np.ndarray
    n_points_full: int
    n_points_filtered: int
    n_patches_full: int
    n_patches_filtered: int
    n_patches_filtered_out: int
    align_stats_sum: Dict[str, float]
    align_stats_n: int
    pred_hist_full: np.ndarray
    gt_hist_full: np.ndarray
    cm_train: np.ndarray
    gt_hist_train: np.ndarray
    pred_hist_train: np.ndarray


def empty_totals(C: int, T: int = 0) -> EvalTotals:
    return EvalTotals(
        cm_full=np.zeros((C, C), dtype=np.int64),
        cm_filtered=np.zeros((C, C), dtype=np.int64),
        n_points_full=0,
        n_points_filtered=0,
        n_patches_full=0,
        n_patches_filtered=0,
        n_patches_filtered_out=0,
        align_stats_sum={},
        align_stats_n=0,
        pred_hist_full=np.zeros((C,), dtype=np.int64),
        gt_hist_full=np.zeros((C,), dtype=np.int64),
        cm_train=np.zeros((T, T), dtype=np.int64),
        gt_hist_train=np.zeros((T,), dtype=np.int64),
        pred_hist_train=np.zeros((T,), dtype=np.int64),
    )


def add_align_stats(t: EvalTotals, stats: Dict[str, float]) -> None:
    t.align_stats_n += 1
    for k, v in stats.items():
        t.align_stats_sum[k] = t.align_stats_sum.get(k, 0.0) + float(v)


def meets_patch_filter(
    xyz_norm: np.ndarray,
    spec: PreprocSpec,
) -> bool:
    n = xyz_norm.shape[0]
    if spec.patch_filter_mode == "none":
        return True
    if spec.patch_filter_mode == "min_points":
        return n >= int(spec.min_points)
    if spec.patch_filter_mode == "min_occ_vox":
        q = np.floor(xyz_norm / float(spec.voxel_size)).astype(np.int32, copy=False)
        q = np.ascontiguousarray(q, dtype=np.int32)
        keys = q[:, 0].astype(np.int64) * 73856093 ^ q[:, 1].astype(np.int64) * 19349663 ^ q[:, 2].astype(np.int64) * 83492791
        occ = len(np.unique(keys))
        return occ >= int(spec.min_occ_vox)
    raise ValueError(f"Unknown patch_filter_mode: {spec.patch_filter_mode}")


def _collapse_score_from_metrics(metrics: dict) -> dict:
    """
    Returns a few cheap numbers to classify collapse:
      - cars+trucks fraction in train-space predictions
      - vehicle fraction in common-space predictions
    """
    out = {}
    ht = metrics.get("hist_train", {})
    pred_train = np.asarray(ht.get("pred_train", []), dtype=np.int64).reshape(-1)

    if pred_train.size >= 4:
        denom = int(pred_train.sum()) or 1
        out["cars_trucks_frac_train"] = float((pred_train[2] + pred_train[3]) / denom)
        out["pred_train_sum"] = int(denom)

    hf = metrics.get("hist_full", {})
    pred_common = np.asarray(hf.get("pred", []), dtype=np.int64).reshape(-1)
    if pred_common.size >= 8:
        denom = int(pred_common.sum()) or 1
        out["vehicle_frac_common"] = float(pred_common[7] / denom)
        out["pred_common_sum"] = int(denom)

    return out


def diagnose_ckpt_vs_evalspec(
    *,
    ckpt_path: str,
    base_spec: PreprocSpec,  # current eval spec = "Spec B"
    model: torch.nn.Module,
    amp: bool,
    dales_files: List[Path],
    map_dales_to_common: np.ndarray,
    pred_to_common: np.ndarray,
    common_class_names: List[str],
    device: torch.device,
    logger: logging.Logger,
    patch_size_m: float,
    patch_stride_m: float,
    out_dir: str,
) -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    load_model_state_or_die(model, ckpt, logger)
    model.eval()
    torch.set_grad_enabled(False)

    ckpt_cfg = ckpt.get("cfg", None)
    if ckpt_cfg is None:
        raise RuntimeError("Checkpoint has no 'cfg' key; cannot auto-align spec.")

    pre = ckpt_cfg["data"].get("preproc", {})
    train_max_voxels = pre.get("max_voxels", None)
    train_class_aware_max_voxels = bool(pre.get("class_aware_max_voxels", False))
    train_rare_ids = tuple(int(x) for x in pre.get("rare_class_ids", [2, 3, 5, 6, 7]))

    label_map = ckpt_cfg["data"].get("dales_label_map_native_to_train", None)
    if label_map is None:
        raise RuntimeError("Checkpoint cfg missing data.dales_label_map_native_to_train")

    train_ignore_index = int(ckpt_cfg["data"]["label_space"]["ignore_index"])

    lut_train = np.full((256,), train_ignore_index, dtype=np.int64)
    for k, v in label_map.items():
        kk = int(k)
        if 0 <= kk < 256:
            lut_train[kk] = int(v)

    sha1 = _sha1_file(ckpt_path)
    patch = ckpt_cfg["data"]["patch"]
    feat = ckpt_cfg["data"]["features"]
    ls = ckpt_cfg["data"]["label_space"]

    logger.info(f"[diag] ckpt_path={ckpt_path}")
    logger.info(f"[diag] ckpt_sha1={sha1}")
    logger.info(f"[diag] ckpt.patch={patch}")
    logger.info(f"[diag] ckpt.features={feat}")
    logger.info(f"[diag] ckpt.label_space.class_names={ls.get('class_names')}")

    # Spec A = checkpoint-aligned
    spec_ckpt = dataclasses.replace(
        base_spec,
        coord_norm_factor=float(patch["coord_norm_factor"]),
        voxel_size=float(patch["voxel_size"]),
        returns_k=int(feat["returns_onehot_k"]),
        voxel_feat_mode="sample_first",
    )

    # Spec B = current (whatever you passed on CLI / base_spec)
    spec_cur = base_spec

    # Run A
    mA = evaluate_run(
        model=model,
        dales_files=dales_files,
        map_dales_to_common=map_dales_to_common,
        pred_to_common=pred_to_common,
        common_class_names=common_class_names,
        spec=spec_ckpt,
        device=device,
        logger=logger,
        patch_size_m=patch_size_m,
        patch_stride_m=patch_stride_m,
        amp=amp,
        map_dales_native_to_train=lut_train,
        train_ignore_index=train_ignore_index,
        train_max_voxels=train_max_voxels,
        train_class_aware_max_voxels=train_class_aware_max_voxels,
        rare_class_ids=train_rare_ids,
    )
    logger.info(f"[diag] ckpt_spec full metrics: {mA}")
    (outp / "metrics_ckpt_spec.json").write_text(json.dumps(mA, indent=2))
    sA = _collapse_score_from_metrics(mA)
    logger.info(f"[diag] ckpt_spec collapse scores: {sA}")

    # Run B
    mB = evaluate_run(
        model=model,
        dales_files=dales_files,
        map_dales_to_common=map_dales_to_common,
        pred_to_common=pred_to_common,
        common_class_names=common_class_names,
        spec=spec_cur,
        device=device,
        logger=logger,
        patch_size_m=patch_size_m,
        patch_stride_m=patch_stride_m,
        amp=amp,
        map_dales_native_to_train=lut_train,
        train_ignore_index=train_ignore_index,
        train_max_voxels=train_max_voxels,
        train_class_aware_max_voxels=train_class_aware_max_voxels,
        rare_class_ids=train_rare_ids,
    )
    logger.info(f"[diag] current_spec full metrics: {mB}")
    (outp / "metrics_current_spec.json").write_text(json.dumps(mB, indent=2))
    sB = _collapse_score_from_metrics(mB)
    logger.info(f"[diag] current_spec collapse scores: {sB}")

    a_bad = (sA.get("cars_trucks_frac_train", 0.0) > 0.90) or (sA.get("vehicle_frac_common", 0.0) > 0.90)
    b_bad = (sB.get("cars_trucks_frac_train", 0.0) > 0.90) or (sB.get("vehicle_frac_common", 0.0) > 0.90)

    if (not a_bad) and b_bad:
        logger.info("[diag] RESULT: Universe A (eval spec mismatch).")
    elif a_bad and b_bad:
        logger.info("[diag] RESULT: Universe C (collapsed even with ckpt-aligned spec).")
    else:
        logger.info("[diag] RESULT: Inconclusive edge case.")


# def evaluate_run(
#     *,
#     model: torch.nn.Module,
#     dales_files: List[Path],
#     map_dales_to_common: np.ndarray,
#     pred_to_common: np.ndarray,
#     common_class_names: List[str],
#     spec: PreprocSpec,
#     device: torch.device,
#     logger: logging.Logger,
#     patch_size_m: float,
#     patch_stride_m: float,
#     amp: bool,
#     map_dales_native_to_train: Optional[np.ndarray] = None,
#     train_ignore_index: int = -100,
# ) -> Dict:

#     print("Evaluating...@@@@@@@@@@@@@$$$$$$$$$$$$$$$$$$$$$$$$&&&&&&&&&&&&&&&&&&&@$@#$$#@$@#$@$@$$@$@@$")

#     C = len(common_class_names)
#     totals = empty_totals(C)
#     # at top of evaluate_run (after C defined)
#     T = int(pred_to_common.shape[0])  # out_channels
#     pred_hist_train = np.zeros((T,), dtype=np.int64)
#     gt_hist_train = np.zeros((T,), dtype=np.int64)
#     cm_train = np.zeros((T, T), dtype=np.int64)

#     t0 = time.time()
#     logger.info(f"[run] spec={dataclasses.asdict(spec)}")
#     logger.info(f"[data] n_files={len(dales_files)} patch={patch_size_m} stride={patch_stride_m}")

#     for fi, fp in enumerate(dales_files, 1):
#         raw = read_dales_las(fp)
#         xyz = raw["xyz"]
#         rn = raw["return_number"]
#         nor = raw["number_of_returns"]
#         gt_native = raw["cls"]

#         # map DALES native -> common (clamp)
#         gt_native_clip = np.clip(gt_native, 0, 255)
#         gt_common = map_dales_to_common[gt_native_clip]

#         logger.info(f"[file] {fi}/{len(dales_files)} {fp.name} n_points={xyz.shape[0]}")

#         patch_i = 0
#         for idx in iter_xy_patches_indices(xyz, patch_size_m, patch_stride_m):
#             patch_i += 1

#             xyz_p = xyz[idx]
#             rn_p = rn[idx]
#             nor_p = nor[idx]
#             gt_p = gt_common[idx]

#             # local coords and normalized coords (for filter decisions)
#             xyz_local = xyz_p - xyz_p.min(axis=0, keepdims=True)
#             xyz_norm_for_filter = (xyz_local / float(spec.coord_norm_factor)).astype(np.float32, copy=False)
#             xyz_norm_for_filter = np.ascontiguousarray(xyz_norm_for_filter, dtype=np.float32)
#             xyz_norm_for_filter[:, 2] *= float(spec.z_scale)

#             # FULL scope always
#             pred_train, a_stats = infer_with_tta_adaptive(
#                 model=model,
#                 xyz_patch=xyz_p,
#                 rn=rn_p,
#                 nor=nor_p,
#                 spec=spec,
#                 device=device,
#                 logger=logger,
#                 amp=amp,
#                 max_depth=2,  # tune: 1–3
#             )

#             pred_train = pred_train.astype(np.int64, copy=False)

#             # sanity: pred ids must be within [0..T-1]
#             if int(pred_train.max()) >= T or int(pred_train.min()) < 0:
#                 raise RuntimeError(f"pred_train out of range: min={pred_train.min()} max={pred_train.max()} T={T}")

#             # --------------------------
#             # Train-space GT (MATCH TRAINING!)
#             # --------------------------
#             if map_dales_native_to_train is None:
#                 raise RuntimeError("map_dales_native_to_train is required for DALES train-space eval.")

#             # ---- accumulate train-space pred histogram (1D) ----
#             pred_hist_train += np.bincount(pred_train, minlength=T)

#             # ---- alignment stats once ----
#             add_align_stats(totals, a_stats)

#             # ---- Train-space GT for DALES: native 1..8 -> train 0..7 ; native 0 -> ignore ----
#             gt_native_patch = gt_native[idx].astype(np.int64, copy=False)
#             gt_train = map_dales_native_to_train[np.clip(gt_native_patch, 0, 255)]

#             # hist + confusion (ignore already encoded as -100)
#             m = gt_train != train_ignore_index
#             totals.gt_hist_train += np.bincount(gt_train[m], minlength=T)
#             totals.pred_hist_train += np.bincount(pred_train, minlength=T)

#             cm_train = confusion_from_labels(gt_train, pred_train, num_classes=T, ignore_index=train_ignore_index)
#             totals.cm_train += cm_train

#             if int(pred_train.max()) >= pred_to_common.shape[0]:
#                 raise RuntimeError(
#                     f"pred_to_common too small: pred_max={int(pred_train.max())} " f"but lut_size={pred_to_common.shape[0]}"
#                 )

#             pred_common = pred_to_common[pred_train]
#             valid = gt_p != 0  # match confusion_from_labels ignore_index=0
#             totals.gt_hist_full += np.bincount(gt_p[valid].astype(np.int64), minlength=C)
#             totals.pred_hist_full += np.bincount(pred_common[valid].astype(np.int64), minlength=C)

#             cm = confusion_from_labels(gt_p, pred_common, C, ignore_index=0)
#             totals.cm_full += cm
#             totals.n_points_full += int(gt_p.shape[0])
#             totals.n_patches_full += 1

#             # FILTERED scope (analysis only; does not bias "full" deltas)
#             keep_patch = meets_patch_filter(xyz_norm_for_filter, spec)
#             if keep_patch:
#                 totals.cm_filtered += cm
#                 totals.n_points_filtered += int(gt_p.shape[0])
#                 totals.n_patches_filtered += 1
#             else:
#                 totals.n_patches_filtered_out += 1

#             # periodic logs
#             if patch_i % 25 == 0:
#                 logger.info(
#                     f"[patch] {fp.name} patch_i={patch_i} n={len(idx)} "
#                     f"occ_vox={a_stats.get('occ_vox', -1):.0f} "
#                     f"ppv_mean={a_stats.get('pts_per_vox_mean', -1):.2f} "
#                     f"ppv_med={a_stats.get('pts_per_vox_med', -1):.2f} "
#                     f"z_span_m={a_stats.get('z_span_m', -1):.2f} "
#                     f"filtered_kept={totals.n_patches_filtered} filtered_out={totals.n_patches_filtered_out}"
#                 )
#                 if device.type == "cuda":
#                     logger.info(
#                         f"[cuda] alloc={torch.cuda.memory_allocated()/1e9:.3f}GB "
#                         f"reserved={torch.cuda.memory_reserved()/1e9:.3f}GB"
#                     )

#             if device.type == "cuda":
#                 reserved = torch.cuda.memory_reserved()
#                 total = torch.cuda.get_device_properties(0).total_memory
#                 if reserved > 0.90 * total:
#                     torch.cuda.empty_cache()

#         # file-level cleanup helps long sweeps
#         if device.type == "cuda":
#             torch.cuda.empty_cache()
#         gc.collect()

#     tp = np.diag(totals.cm_full).astype(np.int64)
#     fp = totals.cm_full.sum(axis=0).astype(np.int64) - tp
#     fn = totals.cm_full.sum(axis=1).astype(np.int64) - tp

#     per_class = []
#     for i, name in enumerate(common_class_names):
#         if i == 0:
#             continue
#         precision = float(tp[i] / max(1, tp[i] + fp[i]))
#         recall = float(tp[i] / max(1, tp[i] + fn[i]))
#         per_class.append(
#             {
#                 "class_id": i,
#                 "name": name,
#                 "tp": int(tp[i]),
#                 "fp": int(fp[i]),
#                 "fn": int(fn[i]),
#                 "precision": precision,
#                 "recall": recall,
#                 "gt_count": int(totals.gt_hist_full[i]),
#                 "pred_count": int(totals.pred_hist_full[i]),
#             }
#         )

#     # ---- common-space metrics (unchanged) ----
#     miou_full, iou_full = iou_from_confusion(totals.cm_full, ignore_index=0)
#     miou_filt, iou_filt = iou_from_confusion(totals.cm_filtered, ignore_index=0)

#     # ---- train-space metrics ----
#     miou_train_all, iou_train_all = iou_from_confusion(cm_train, ignore_index=None)
#     miou_train_no0, iou_train_no0 = iou_from_confusion(cm_train, ignore_index=0)  # optional

#     runtime = time.time() - t0

#     # aggregate alignment stats
#     align_avg = {}
#     if totals.align_stats_n > 0:
#         for k, v in totals.align_stats_sum.items():
#             align_avg[k] = float(v / totals.align_stats_n)

#     return {
#         "spec": dataclasses.asdict(spec),
#         "runtime_s": runtime,
#         "full": {
#             "miou": miou_full,
#             "iou_per_class": iou_full,
#             "n_points": totals.n_points_full,
#             "n_patches": totals.n_patches_full,
#         },
#         "filtered": {
#             "miou": miou_filt,
#             "iou_per_class": iou_filt,
#             "n_points": totals.n_points_filtered,
#             "n_patches_kept": totals.n_patches_filtered,
#             "n_patches_dropped": totals.n_patches_filtered_out,
#             "filter_mode": spec.patch_filter_mode,
#         },
#         "align_avg": align_avg,
#         "hist_full": {
#             "gt": totals.gt_hist_full.tolist(),
#             "pred": totals.pred_hist_full.tolist(),
#         },
#         "per_class_pr": per_class,
#         "hist_train": {
#             "gt_train": gt_hist_train.tolist(),
#             "pred_train": pred_hist_train.tolist(),
#             "train_id_to_common": pred_to_common.tolist(),
#             "train_to_common_counts": np.bincount(pred_to_common.astype(np.int64), minlength=C).tolist(),
#         },
#         "train": {
#             "miou_all": miou_train_all,
#             "iou_per_class_all": iou_train_all,
#             "miou_ignore0": miou_train_no0,
#             "iou_per_class_ignore0": iou_train_no0,
#             "cm_train": cm_train.tolist(),
#         },
#     }


def evaluate_run(
    *,
    model: torch.nn.Module,
    dales_files: List[Path],
    map_dales_to_common: np.ndarray,
    pred_to_common: np.ndarray,
    common_class_names: List[str],
    spec: PreprocSpec,
    device: torch.device,
    logger: logging.Logger,
    patch_size_m: float,
    patch_stride_m: float,
    amp: bool,
    map_dales_native_to_train: Optional[np.ndarray] = None,
    train_ignore_index: int = -100,
    train_max_voxels: Optional[int] = None,
    train_class_aware_max_voxels: bool = False,
    rare_class_ids: Tuple[int, ...] = (2, 3, 5, 6, 7),
) -> Dict:
    logger.info(f"[sanity] model.training={model.training}")

    # ---- critical: eval mode (BN/Dropout) ----
    if model.training:
        logger.info("[eval] model was in TRAIN mode; switching to EVAL mode")
    model.eval()

    C = len(common_class_names)

    # out_channels / train-space classes
    T = int(pred_to_common.shape[0])
    if T <= 0:
        raise RuntimeError(f"Invalid T={T} from pred_to_common.shape={pred_to_common.shape}")

    totals = empty_totals(C, T)

    do_train_space = map_dales_native_to_train is not None
    if do_train_space and map_dales_native_to_train.shape[0] < 256:
        raise RuntimeError(f"map_dales_native_to_train must be size 256, got {map_dales_native_to_train.shape}")

    if not do_train_space:
        logger.warning("[train] Train-space eval DISABLED (no target native->train LUT provided).")

    t0 = time.time()
    logger.info(f"[run] spec={dataclasses.asdict(spec)}")
    logger.info(f"[data] n_files={len(dales_files)} patch={patch_size_m} stride={patch_stride_m}")
    logger.info(f"[train] T={T} train_ignore_index={train_ignore_index} cm_train_dtype={totals.cm_train.dtype}")

    # dales_files = dales_files[:1]  # limit to first 10 files for debugging
    for fi, fp in enumerate(dales_files, 1):
        raw = read_dales_las(fp)
        xyz = raw["xyz"]
        rn = raw["return_number"]
        nor = raw["number_of_returns"]
        gt_native = raw["cls"]

        # DALES native -> common (for common-space eval)
        gt_native_clip = np.clip(gt_native, 0, 255)
        gt_common = map_dales_to_common[gt_native_clip]

        logger.info(f"[file] {fi}/{len(dales_files)} {fp.name} n_points={xyz.shape[0]}")

        patch_i = 0
        for idx in iter_xy_patches_indices(xyz, patch_size_m, patch_stride_m):
            patch_i += 1
            if fi == 3 and patch_i == 1:
                logger.info(f"[sanity] gt_native unique (first 20): {np.unique(gt_native)[:20]}")
                logger.info(f"[sanity] rn unique (first 20): {np.unique(rn)[:20]}")
                logger.info(f"[sanity] nor unique (first 20): {np.unique(nor)[:20]}")
                logger.info(
                    f"[sanity] rn min/max: {int(rn.min())}/{int(rn.max())}  nor min/max: {int(nor.min())}/{int(nor.max())}"
                )

            xyz_p = xyz[idx]
            rn_p = rn[idx]
            nor_p = nor[idx]
            gt_p = gt_common[idx]
            gt_native_patch = gt_native[idx].astype(np.int64, copy=False)

            # local coords and normalized coords (for filter decisions)
            xyz_local = xyz_p - xyz_p.min(axis=0, keepdims=True)
            xyz_norm_for_filter = (xyz_local / float(spec.coord_norm_factor)).astype(np.float32, copy=False)
            xyz_norm_for_filter = np.ascontiguousarray(xyz_norm_for_filter, dtype=np.float32)
            xyz_norm_for_filter[:, 2] *= float(spec.z_scale)

            # --------------------------
            # Inference (train-id per ORIGINAL point)
            # --------------------------
            pred_train, a_stats = infer_with_tta_adaptive(
                model=model,
                xyz_patch=xyz_p,
                rn=rn_p,
                nor=nor_p,
                spec=spec,
                device=device,
                logger=logger,
                amp=amp,
                max_depth=2,
            )
            pred_train = pred_train.astype(np.int64, copy=False)

            pmin = int(pred_train.min()) if pred_train.size else 0
            pmax = int(pred_train.max()) if pred_train.size else -1
            if pred_train.size and (pmax >= T or pmin < 0):
                raise RuntimeError(f"pred_train out of range: min={pmin} max={pmax} T={T}")

            add_align_stats(totals, a_stats)

            # --------------------------
            # Train-space GT (native -> train via LUT)
            # --------------------------
            gt_train = None
            if do_train_space:
                # --------------------------
                # Train-space GT (native -> train via LUT)
                # --------------------------
                gt_train = map_dales_native_to_train[np.clip(gt_native_patch, 0, 255)]

                m = gt_train != train_ignore_index
                if np.any(m):
                    totals.gt_hist_train += np.bincount(gt_train[m], minlength=T)
                    totals.pred_hist_train += np.bincount(pred_train[m], minlength=T)

                cm_tr = confusion_from_labels(
                    gt_train,
                    pred_train,
                    num_classes=T,
                    ignore_index=train_ignore_index,
                )
                totals.cm_train += cm_tr

            # --------------------------
            # Common-space eval (unchanged)
            # --------------------------

            pred_common = pred_to_common[pred_train]
            # One-time sanity for mapping correctness (first file, first patch)
            if fi == 1 and patch_i == 1:
                frac_ign = float(np.mean(gt_train == train_ignore_index)) if gt_train.size else 0.0
                logger.info(f"[sanity] T(train classes)={T} C(common)={C}")
                logger.info(f"[sanity] gt_train: frac_ignore={frac_ign:.4f} unique={np.unique(gt_train)[:20]}")
                logger.info(f"[sanity] pred_train unique={np.unique(pred_train)[:20]}")
                logger.info(f"[sanity] gt_common unique={np.unique(gt_p)[:20]}")
                logger.info(f"[sanity] pred_common unique={np.unique(pred_common)[:20]}")
                # If pred_to_common is wrong, this often becomes almost constant (e.g., all vehicle)
                pc = np.bincount(pred_common.astype(np.int64), minlength=C)
                logger.info(f"[sanity] pred_common hist={pc.tolist()}")

            valid_common = gt_p != 0  # common ignore=0
            totals.gt_hist_full += np.bincount(gt_p[valid_common].astype(np.int64), minlength=C)
            totals.pred_hist_full += np.bincount(pred_common[valid_common].astype(np.int64), minlength=C)

            cm_full = confusion_from_labels(gt_p, pred_common, C, ignore_index=0)
            totals.cm_full += cm_full
            totals.n_points_full += int(np.sum(gt_p != 0))
            totals.n_patches_full += 1

            keep_patch = meets_patch_filter(xyz_norm_for_filter, spec)
            if keep_patch:
                totals.cm_filtered += cm_full
                totals.n_points_filtered += int(np.sum(gt_p != 0))
                totals.n_patches_filtered += 1
            else:
                totals.n_patches_filtered_out += 1

            if patch_i % 25 == 0:
                logger.info(
                    f"[patch] {fp.name} patch_i={patch_i} n={len(idx)} "
                    f"occ_vox={a_stats.get('occ_vox', -1):.0f} "
                    f"ppv_mean={a_stats.get('pts_per_vox_mean', -1):.2f} "
                    f"ppv_med={a_stats.get('pts_per_vox_med', -1):.2f} "
                    f"z_span_m={a_stats.get('z_span_m', -1):.2f} "
                    f"filtered_kept={totals.n_patches_filtered} filtered_out={totals.n_patches_filtered_out}"
                )

            if device.type == "cuda":
                reserved = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                if reserved > 0.90 * total:
                    torch.cuda.empty_cache()

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # ---- common-space metrics ----
    miou_full, iou_full = iou_from_confusion(totals.cm_full, ignore_index=0)
    miou_filt, iou_filt = iou_from_confusion(totals.cm_filtered, ignore_index=0)

    miou_train_all = None
    iou_train_all = None
    miou_train_no0 = None
    iou_train_no0 = None
    if do_train_space:
        miou_train_all, iou_train_all = iou_from_confusion(totals.cm_train, ignore_index=None)
        miou_train_no0, iou_train_no0 = iou_from_confusion(totals.cm_train, ignore_index=0)  # optional

    runtime = time.time() - t0

    # aggregate alignment stats
    align_avg = {}
    if totals.align_stats_n > 0:
        for k, v in totals.align_stats_sum.items():
            align_avg[k] = float(v / totals.align_stats_n)

    # per-class precision/recall in common space (unchanged)
    tp = np.diag(totals.cm_full).astype(np.int64)
    fp = totals.cm_full.sum(axis=0).astype(np.int64) - tp
    fn = totals.cm_full.sum(axis=1).astype(np.int64) - tp

    per_class = []
    for i, name in enumerate(common_class_names):
        if i == 0:
            continue
        precision = float(tp[i] / max(1, tp[i] + fp[i]))
        recall = float(tp[i] / max(1, tp[i] + fn[i]))
        per_class.append(
            {
                "class_id": i,
                "name": name,
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "precision": precision,
                "recall": recall,
                "gt_count": int(totals.gt_hist_full[i]),
                "pred_count": int(totals.pred_hist_full[i]),
            }
        )

    return {
        "spec": dataclasses.asdict(spec),
        "runtime_s": runtime,
        "full": {
            "miou": miou_full,
            "iou_per_class": iou_full,
            "n_points": totals.n_points_full,
            "n_patches": totals.n_patches_full,
        },
        "filtered": {
            "miou": miou_filt,
            "iou_per_class": iou_filt,
            "n_points": totals.n_points_filtered,
            "n_patches_kept": totals.n_patches_filtered,
            "n_patches_dropped": totals.n_patches_filtered_out,
            "filter_mode": spec.patch_filter_mode,
        },
        "align_avg": align_avg,
        "hist_full": {
            "gt": totals.gt_hist_full.tolist(),
            "pred": totals.pred_hist_full.tolist(),
        },
        "per_class_pr": per_class,
        "hist_train": {
            "gt_train": totals.gt_hist_train.tolist(),
            "pred_train": totals.pred_hist_train.tolist(),
            "train_id_to_common": pred_to_common.tolist(),
            "train_to_common_counts": np.bincount(pred_to_common.astype(np.int64), minlength=C).tolist(),
        },
        "train": (
            {
                "miou_all": miou_train_all,
                "iou_per_class_all": iou_train_all,
                "miou_ignore0": miou_train_no0,
                "iou_per_class_ignore0": iou_train_no0,
                "cm_train": totals.cm_train.tolist(),
            }
            if do_train_space
            else None
        ),
    }


# -------------------------
# Run plan generation (staged + limited combos)
# -------------------------


def make_run_plan(base: PreprocSpec, mode: str, max_runs: int) -> List[PreprocSpec]:
    runs: List[PreprocSpec] = [base]

    # -------------------------
    # "Kill Switch" runs (must-run, science-first)
    # -------------------------
    # 1) Geom-only: remove intensity corruption, use permutation-invariant voxel features
    runs.append(dataclasses.replace(base, intensity_mode="zero", voxel_feat_mode="mean_all"))

    # 2) Statistical alignment: only meaningful when has_intensity_ref=True (will be gated later)
    runs.append(dataclasses.replace(base, intensity_mode="quantile_match", voxel_feat_mode="mean_all"))

    # 3) Structural invariance: slow mode (will be gated unless --include_slow_modes)
    runs.append(
        dataclasses.replace(
            base,
            intensity_mode="proxy_linearity",
            voxel_feat_mode="sample_max_linearity",
        )
    )

    # 4) View invariance: rot4 TTA  mean aggregation
    runs.append(dataclasses.replace(base, tta_mode="rot4", voxel_feat_mode="mean_all"))

    intensity_pool = [
        ("as_is", {}),
        ("zero", {}),
        ("constant", {"intensity_constant": 0.5}),
        ("proxy_z", {}),
        ("minmax_z", {}),
        # SLOW/OPTIONAL: only include if enabled externally
        ("proxy_linearity", {}),
        ("quantile_match", {}),  # only valid if has_intensity_ref=True
    ]
    returns_pool = [
        ("as_is", {}),
        ("drop", {}),
        ("const1", {}),
        ("uniform", {}),
    ]
    coord_pool = [0.8, 1.0, 1.2]  # scale coord_norm_factor
    voxel_pool = [0.8, 1.0, 1.2]  # scale voxel_size
    zscale_pool = [0.8, 1.0, 1.2]
    voxfeat_pool = [
        ("sample_first", {}),
        ("sample_random", {}),
        ("sample_max_intensity", {}),
        # SLOW/OPTIONAL: only include if enabled externally
        ("sample_max_linearity", {}),
        ("mean_all", {}),
        ("mean_thin", {"mean_thin_p": 0.7}),
    ]
    tta_pool = ["none", "rot4"]

    filter_pool = [
        ("none", {}),
        ("min_points", {"min_points": 1000}),
        ("min_occ_vox", {"min_occ_vox": 1000}),
    ]

    def add(r: PreprocSpec):
        if len(runs) < max_runs:
            runs.append(r)

    if mode == "staged":
        # single-factor ablations (clean science)
        for im, kw in intensity_pool:
            add(dataclasses.replace(base, intensity_mode=im, **kw))
        for rm, kw in returns_pool:
            add(dataclasses.replace(base, returns_mode=rm, **kw))
        for m in coord_pool:
            add(dataclasses.replace(base, coord_norm_factor=base.coord_norm_factor * m))
        for m in voxel_pool:
            add(dataclasses.replace(base, voxel_size=base.voxel_size * m))
        for zs in zscale_pool:
            add(dataclasses.replace(base, z_scale=zs))
        for vm, kw in voxfeat_pool:
            add(dataclasses.replace(base, voxel_feat_mode=vm, **kw))
        for tm in tta_pool:
            add(dataclasses.replace(base, tta_mode=tm))
        for fm, kw in filter_pool:
            add(dataclasses.replace(base, patch_filter_mode=fm, **kw))

        # limited combos (motivated by EDA: intensity collapse + sparsity + z-span)
        combos = [
            # robust to missing intensity + voxel mean aggregation + TTA
            dataclasses.replace(
                base,
                intensity_mode="constant",
                intensity_constant=0.5,
                voxel_feat_mode="mean_all",
                tta_mode="rot4",
            ),
            # preserve thin structures proxy: linearity rep + smaller voxels
            dataclasses.replace(
                base,
                voxel_feat_mode="sample_max_linearity",
                voxel_size=base.voxel_size * 0.8,
                intensity_mode="constant",
            ),
            # z-span compensation + smaller voxel + mean
            dataclasses.replace(
                base,
                z_scale=1.2,
                voxel_size=base.voxel_size * 0.8,
                voxel_feat_mode="mean_all",
                intensity_mode="constant",
            ),
        ]
        for c in combos:
            add(c)

    elif mode == "full":
        # broader cross product but capped
        for im, imkw in intensity_pool:
            for rm, rmkw in returns_pool:
                for cm in coord_pool:
                    for vmult in voxel_pool:
                        for zs in zscale_pool:
                            for vfm, vfkw in voxfeat_pool:
                                for tm in tta_pool:
                                    for fm, fmkw in filter_pool:
                                        if len(runs) >= max_runs:
                                            break
                                        runs.append(
                                            dataclasses.replace(
                                                base,
                                                intensity_mode=im,
                                                returns_mode=rm,
                                                coord_norm_factor=base.coord_norm_factor * cm,
                                                voxel_size=base.voxel_size * vmult,
                                                z_scale=zs,
                                                voxel_feat_mode=vfm,
                                                tta_mode=tm,
                                                patch_filter_mode=fm,
                                                **imkw,
                                                **rmkw,
                                                **vfkw,
                                                **fmkw,
                                            )
                                        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # de-dup
    seen = set()
    uniq: List[PreprocSpec] = []
    for r in runs:
        rid = r.to_id()
        if rid in seen:
            continue
        uniq.append(r)
        seen.add(rid)
    return uniq[:max_runs]


def make_decisive_returns_plan_full(
    base: PreprocSpec,
    *,
    c: float,
    has_ref: bool,
    include_slow: bool,
) -> List[PreprocSpec]:
    """
    High-signal plan intended for comparing new checkpoints.
    ~32 runs (plus quantile_match + slow modes if enabled).
    """
    runs: List[PreprocSpec] = []

    def add(s: PreprocSpec) -> None:
        runs.append(s)

    # -------------------------
    # Group A: Baselines / sanity
    # -------------------------
    add(base)
    add(dataclasses.replace(base, returns_mode="drop"))
    add(dataclasses.replace(base, returns_mode="uniform"))
    add(dataclasses.replace(base, returns_mode="const1"))

    add(dataclasses.replace(base, voxel_feat_mode="mean_all"))
    add(dataclasses.replace(base, voxel_feat_mode="mean_all", returns_mode="drop"))

    add(dataclasses.replace(base, voxel_feat_mode="mean_thin", mean_thin_p=0.7))
    add(dataclasses.replace(base, voxel_feat_mode="sample_max_intensity"))

    add(dataclasses.replace(base, tta_mode="rot4"))
    add(dataclasses.replace(base, tta_mode="rot4", voxel_feat_mode="mean_all"))

    # -------------------------
    # Group B: Intensity robustness
    # -------------------------
    add(dataclasses.replace(base, intensity_mode="constant", intensity_constant=c))
    add(dataclasses.replace(base, intensity_mode="constant", intensity_constant=c, returns_mode="drop"))

    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="mean_all",
        )
    )
    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="mean_all",
            returns_mode="drop",
        )
    )

    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="mean_all",
            tta_mode="rot4",
        )
    )
    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="mean_all",
            tta_mode="rot4",
            returns_mode="drop",
        )
    )

    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="sample_max_intensity",
        )
    )

    add(dataclasses.replace(base, intensity_mode="zero", voxel_feat_mode="mean_all"))
    add(dataclasses.replace(base, intensity_mode="proxy_z", voxel_feat_mode="mean_all"))
    add(dataclasses.replace(base, intensity_mode="minmax_z", voxel_feat_mode="mean_all"))

    if has_ref:
        add(dataclasses.replace(base, intensity_mode="quantile_match", voxel_feat_mode="mean_all"))
        add(
            dataclasses.replace(
                base,
                intensity_mode="quantile_match",
                voxel_feat_mode="mean_all",
                tta_mode="rot4",
            )
        )
        add(
            dataclasses.replace(
                base,
                intensity_mode="quantile_match",
                voxel_feat_mode="mean_all",
                tta_mode="rot4",
                returns_mode="drop",
            )
        )

    # -------------------------
    # Group C: Geometry sweeps around "best recipe"
    # -------------------------
    best = dataclasses.replace(
        base,
        intensity_mode="constant",
        intensity_constant=c,
        voxel_feat_mode="mean_all",
        tta_mode="rot4",
    )
    best_drop = dataclasses.replace(best, returns_mode="drop")

    # returns ablation on best
    for rm in ("as_is", "drop", "uniform", "const1"):
        add(dataclasses.replace(best, returns_mode=rm))

    # voxel size tweaks
    for vm in (0.8, 1.2):
        add(dataclasses.replace(best, voxel_size=base.voxel_size * vm))
        add(dataclasses.replace(best_drop, voxel_size=base.voxel_size * vm))

    # coord norm tweaks
    for cm in (0.8, 1.2):
        add(dataclasses.replace(best, coord_norm_factor=base.coord_norm_factor * cm))

    # z anisotropy tweaks
    for zs in (0.8, 1.2):
        add(dataclasses.replace(best, z_scale=zs))
        add(dataclasses.replace(best_drop, z_scale=zs))

    # -------------------------
    # Group D: Optional slow structural modes
    # -------------------------
    if include_slow:
        add(
            dataclasses.replace(
                base,
                intensity_mode="proxy_linearity",
                voxel_feat_mode="sample_max_linearity",
                tta_mode="none",
            )
        )
        add(
            dataclasses.replace(
                base,
                intensity_mode="proxy_linearity",
                voxel_feat_mode="sample_max_linearity",
                tta_mode="rot4",
            )
        )

    # De-dup deterministically by canonical spec
    seen = set()
    uniq: List[PreprocSpec] = []
    for s in runs:
        k = spec_key_from_spec(s)
        if k in seen:
            continue
        uniq.append(s)
        seen.add(k)

    return uniq


def make_decisive_returns_plan_minimal(
    base: PreprocSpec,
    *,
    c: float,
    has_ref: bool,
) -> List[PreprocSpec]:
    """
    Minimal decisive plan: ~12-16 runs.
    Goal: fast checkpoint comparison without missing the main failure modes.
    """
    runs: List[PreprocSpec] = []

    def add(s: PreprocSpec) -> None:
        runs.append(s)

    # 1) Baseline
    add(base)

    # 2) Baseline returns ablation (most common DALES generalization failure)
    add(dataclasses.replace(base, returns_mode="drop"))

    # 3) Best-known recipe (+ returns drop variant)
    best = dataclasses.replace(
        base,
        intensity_mode="constant",
        intensity_constant=c,
        voxel_feat_mode="mean_all",
        tta_mode="rot4",
    )
    add(best)
    add(dataclasses.replace(best, returns_mode="drop"))

    # 4) Constant intensity with training-like voxel rep (tests: only intensity mismatch)
    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="sample_first",
        )
    )
    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="sample_first",
            returns_mode="drop",
        )
    )

    # 5) Remove intensity entirely (tests: intensity collapse / corruption)
    add(dataclasses.replace(base, intensity_mode="zero", voxel_feat_mode="mean_all"))

    # 6) Two geometry-proxy intensities (tests: z-span/domain shift)
    add(dataclasses.replace(base, intensity_mode="proxy_z", voxel_feat_mode="mean_all"))
    add(dataclasses.replace(base, intensity_mode="minmax_z", voxel_feat_mode="mean_all"))

    # 7) Voxel rep stress tests (thin structures)
    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="sample_max_intensity",
        )
    )
    add(
        dataclasses.replace(
            base,
            intensity_mode="constant",
            intensity_constant=c,
            voxel_feat_mode="mean_thin",
            mean_thin_p=0.7,
        )
    )

    # 8) rot4 on baseline (sanity: view invariance helps even without other changes)
    add(dataclasses.replace(base, tta_mode="rot4"))

    # 9) Quantile match only if ref exists (optional but very high-signal)
    if has_ref:
        add(dataclasses.replace(base, intensity_mode="quantile_match", voxel_feat_mode="mean_all"))
        add(
            dataclasses.replace(
                base,
                intensity_mode="quantile_match",
                voxel_feat_mode="mean_all",
                tta_mode="rot4",
            )
        )

    # De-dup deterministically
    seen = set()
    uniq: List[PreprocSpec] = []
    for s in runs:
        k = spec_key_from_spec(s)
        if k in seen:
            continue
        uniq.append(s)
        seen.add(k)

    return uniq


def make_decisive_returns_plan_returns_only(
    base: PreprocSpec,
) -> List[PreprocSpec]:
    """
    Returns-only decisive plan.

    Focuses on:
      - returns usage vs. no-returns
      - voxel aggregation strategy (training-like vs permutation-invariant vs thin-structure stress)
      - rot4 TTA on both baseline and best-geometry configs
      - small geometry sweeps (voxel_size / coord_norm_factor / z_scale) around best

    Assumes:
      - Features are returns-only (no intensity fields in PreprocSpec).
      - Common label space and mappings are handled elsewhere.
    """
    runs: List[PreprocSpec] = []

    def add(s: PreprocSpec) -> None:
        runs.append(s)

    # 1) Baseline (training-like: sample_first + returns as-is)
    add(base)

    # 2) Returns ablation (most common DALES generalization failure)
    add(dataclasses.replace(base, returns_mode="drop"))

    # 3) Better voxel aggregation (permutation-invariant mean over voxel)
    best = dataclasses.replace(base, voxel_feat_mode="mean_all")
    add(best)  # mean_all + returns
    add(dataclasses.replace(best, returns_mode="drop"))  # mean_all + no-returns

    # 4) Thin-structure stress (random thinning + mean)
    add(dataclasses.replace(base, voxel_feat_mode="mean_thin", mean_thin_p=0.7))

    # 5) rot4 on baseline (view invariance with training-like voxel rep)
    add(dataclasses.replace(base, tta_mode="rot4"))

    # 6) rot4 on best geometry (high-signal configs, with and without returns)
    best_rot4 = dataclasses.replace(best, tta_mode="rot4")
    add(best_rot4)  # mean_all + rot4 + returns
    add(dataclasses.replace(best_rot4, returns_mode="drop"))  # mean_all + rot4 + no-returns

    # 7) Geometry sweeps around best (mean_all):
    #    - voxel_size: slightly finer and slightly coarser
    for vm in (0.8, 1.2):
        add(dataclasses.replace(best, voxel_size=base.voxel_size * vm))

    #    - coord_norm_factor: change effective receptive field / scaling
    for cm in (0.8, 1.2):
        add(dataclasses.replace(best, coord_norm_factor=base.coord_norm_factor * cm))

    #    - z anisotropy: compensate for different z-span statistics
    for zs in (0.8, 1.2):
        add(dataclasses.replace(best, z_scale=zs))

    # De-dup deterministically via canonical spec representation
    seen = set()
    uniq: List[PreprocSpec] = []
    for s in runs:
        k = spec_key_from_spec(s)
        if k in seen:
            continue
        uniq.append(s)
        seen.add(k)

    return uniq


# -------------------------
# Main
# -------------------------


def fit_lut_to_out_channels(pred_to_common: np.ndarray, out_channels: int, *, logger: logging.Logger) -> np.ndarray:
    """Pad/trim LUT so pred_to_common.shape[0] == out_channels (model output classes)."""
    if pred_to_common.shape[0] == out_channels:
        return pred_to_common
    if pred_to_common.shape[0] < out_channels:
        padded = np.zeros((out_channels,), dtype=np.int64)
        padded[: pred_to_common.shape[0]] = pred_to_common
        logger.warning(f"[map] pred_to_common padded: {pred_to_common.shape[0]} -> {out_channels}")
        return padded
    # larger than out_channels: trim (should be fine as long as model never outputs >= out_channels)
    logger.warning(f"[map] pred_to_common trimmed: {pred_to_common.shape[0]} -> {out_channels}")
    return pred_to_common[:out_channels].astype(np.int64, copy=False)


def _to_plain_dict_cfg(x: Any) -> Optional[dict]:
    """Make ckpt['cfg'] usable regardless of it being dict / OmegaConf / something else."""
    if isinstance(x, dict):
        return x
    # OmegaConf support if available, but don't require it.
    try:
        from omegaconf import OmegaConf  # type: ignore

        if OmegaConf.is_config(x):
            y = OmegaConf.to_container(x, resolve=True)
            return y if isinstance(y, dict) else None
    except Exception:
        pass
    return None


def _contract_digest(cfg: dict) -> dict:
    """Fields that MUST match between training ckpt cfg and eval YAML config."""
    out = {}

    model = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    data = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    feat = data.get("features", {}) if isinstance(data.get("features", {}), dict) else {}
    ls = data.get("label_space", {}) if isinstance(data.get("label_space", {}), dict) else {}

    out["data.dataset"] = str(data.get("dataset", ""))
    out["model.D"] = int(model.get("D", 3))
    out["model.out_channels"] = int(model.get("out_channels", -1))

    # feature semantics (these define channel meaning)
    out["features.use_intensity"] = bool(feat.get("use_intensity", False))
    out["features.returns_onehot_k"] = int(feat.get("returns_onehot_k", -1))
    out["features.use_rgb"] = bool(feat.get("use_rgb", False))
    out["features.include_coords"] = bool(feat.get("include_coords", False))

    # label-space semantics (these define class ordering)
    out["label_space.num_classes"] = int(ls.get("num_classes", -1))
    out["label_space.ignore_index"] = int(ls.get("ignore_index", -100))
    out["label_space.class_names"] = ls.get("class_names", None)

    return out


def assert_yaml_matches_ckpt_cfg(*, yaml_cfg: dict, ckpt_path: str, logger: logging.Logger) -> dict:
    """
    Raises immediately if the YAML config doesn't match ckpt['cfg'] on critical fields.
    Returns ckpt_cfg as a plain dict for further logging.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_cfg_raw = ckpt.get("cfg", None)
    ckpt_cfg = _to_plain_dict_cfg(ckpt_cfg_raw)
    if ckpt_cfg is None:
        raise RuntimeError(
            "Checkpoint has no usable 'cfg' dict (or OmegaConf) -> cannot validate config/label-space. "
            f"ckpt keys={list(ckpt.keys())}"
        )

    a = _contract_digest(yaml_cfg)
    b = _contract_digest(ckpt_cfg)

    mism = {k: {"yaml": a.get(k), "ckpt": b.get(k)} for k in a.keys() if a.get(k) != b.get(k)}
    if mism:
        logger.error("[FATAL] YAML config does not match checkpoint training cfg on required contract fields:")
        logger.error(json.dumps(mism, indent=2, default=str))
        raise RuntimeError(
            "Wrong --config for this --ckpt (or ckpt was trained with different label/features). "
            "Fix: pass the exact training YAML for this checkpoint."
        )

    logger.info("[ok] YAML config matches ckpt['cfg'] contract (dataset/model/features/label_space).")
    return ckpt_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        required=True,
        help="Torch checkpoint (must be a torch.nn.Module for this script)",
    )
    ap.add_argument("--mapping_dales_native_to_common", required=True)
    ap.add_argument("--mapping_eclair_native_to_common", required=True)
    ap.add_argument(
        "--mapping_pred_to_common",
        default=None,
        help="Optional: model output id -> common id YAML. " "Use if auto-derivation fails.",
    )
    ap.add_argument(
        "--mapping_native_to_train",
        default=None,
        help=(
            "Optional YAML: target native label -> model train id (size up to 256). "
            "If omitted, will try to read from checkpoint cfg (e.g., cfg.data.dales_label_map_native_to_train)."
        ),
    )
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--patch_size_m", type=float, default=100.0)
    ap.add_argument("--patch_stride_m", type=float, default=100.0)

    # baseline must match training
    ap.add_argument("--coord_norm_factor", type=float, default=10.0)
    ap.add_argument("--voxel_size", type=float, default=0.05)
    ap.add_argument("--returns_k", type=int, default=5)

    ap.add_argument("--mode", default="staged", choices=["staged", "full"])
    ap.add_argument("--max_runs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--include_slow_modes",
        action="store_true",
        help="Include slow modes (proxy_linearity / sample_max_linearity). Default: off.",
    )
    ap.add_argument(
        "--pending_specs_json",
        default=None,
        help="Optional path to pending_specs.json. If set, run ONLY these specs (resume mode).",
    )

    # mapping derivation knobs
    ap.add_argument(
        "--try_import_label_maps",
        action="store_true",
        help="Try to import src.label_maps to derive pred->common.",
    )
    ap.add_argument("--eclair_undefined_id", type=int, default=0)
    ap.add_argument("--eclair_ignore_index", type=int, default=-100)

    # optional intensity reference (ECLAIR) for quantile matching
    ap.add_argument(
        "--config",
        required=True,
        help="Training YAML config used for the checkpoint/model.",
    )
    ap.add_argument(
        "--plan",
        default="returns_only",
        choices=["returns_only", "voxel_ablation"],
        help="Run plan. For this script, only 'returns_only' (fixed decisive plan) is supported.",
    )
    ap.add_argument(
        "--decisive_size",
        default="minimal",
        choices=["minimal", "full"],
        help="Size of the decisive plan. minimal=~12-16 runs, full=~32-37 runs.",
    )
    ap.add_argument(
        "--dales_root",
        required=True,
        help=("Folder with TARGET dataset .las/.laz. " "Use DALES path for ECLAIR→DALES, and ECLAIR path for DALES→ECLAIR."),
    )
    ap.add_argument(
        "--voxel_ablation_values",
        default="0.02,0.03,0.04",
        help=(
            "CSV list of EXTRA voxel_size values (in normalized space) to evaluate in addition to "
            "the baseline --voxel_size. Only used when --plan voxel_ablation. "
            "Default gives 4 total runs when baseline is 0.05."
        ),
    )
    ap.add_argument("--diag_ckpt", action="store_true", help="Run ckpt-vs-evalspec diagnostic and exit.")

    args = ap.parse_args()
    if abs(float(args.patch_stride_m) - float(args.patch_size_m)) > 1e-6:
        raise RuntimeError(
            "patch_stride_m must equal patch_size_m unless you implement overlap fusion "
            "(otherwise points get double-counted and IoU is corrupted)."
        )

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    master = setup_logger(out_root, "preproc_sweep.master", level=logging.INFO)
    # HARD FAIL-FAST: wrong YAML passed to eval
    yaml_cfg = load_yaml(args.config)
    ckpt_cfg = assert_yaml_matches_ckpt_cfg(yaml_cfg=yaml_cfg, ckpt_path=args.ckpt, logger=master)

    # Optional: log the exact train-time class order (this catches "wrong label space" instantly)
    try:
        ls = ckpt_cfg.get("data", {}).get("label_space", {})
        master.info(f"[ckpt] train class_names={ls.get('class_names')}")
    except Exception:
        pass

    set_all_seeds(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # load mappings
    map_dales = load_yaml_lut_fixed(args.mapping_dales_native_to_common, size=256, default=0)
    # Keep BOTH:
    # 1) dict for the simple fallback assumption native_id = train_id  1
    eclair_native_to_common_dict = load_yaml_map_int_dict(args.mapping_eclair_native_to_common)
    # 2) LUT for robust derivation through eclair_native_to_train_ids (import-based)
    eclair_native_to_common_lut = load_yaml_lut_fixed(args.mapping_eclair_native_to_common, size=256, default=0)

    # derive pred_to_common
    # (IMPORTANT: pred_to_common must be based on *checkpoint source domain*, not target.)
    # model + amp + out_channels (from config)
    model, amp, in_channels, out_channels = load_model_from_config(
        config_path=args.config,
        ckpt_path=args.ckpt,
        device=device,
    )

    expected_in = 2 * int(args.returns_k)
    if in_channels != expected_in:
        raise RuntimeError(
            f"Model in_channels={in_channels} but returns-only features expect 2*returns_k={expected_in}. "
            f"Check config.model.in_channels and --returns_k."
        )

    # ---- Always load the model's native->train LUT from checkpoint (source-domain LUT) ----
    lut_model_native_to_train, train_ignore_index, train_max_voxels, train_class_aware_max_voxels, rare_class_ids = (
        load_native_to_train_from_ckpt(args.ckpt, master)
    )

    src_domain = "unknown"
    # pred_to_common
    if args.mapping_pred_to_common is not None:
        pred_to_common = load_yaml_lut_fixed(args.mapping_pred_to_common, size=out_channels, default=0)
        master.info(f"[map] Using provided mapping_pred_to_common: {args.mapping_pred_to_common}")
    else:
        # Auto-detect checkpoint source domain and derive pred_to_common from checkpoint LUT.
        src_domain = infer_ckpt_domain(args.ckpt, out_channels=out_channels, logger=master)
        if src_domain == "dales":
            # DALES-trained model: source native->common MUST be DALES mapping
            pred_to_common = build_pred_to_common_from_ckpt_lut(
                native_to_train_lut=lut_model_native_to_train,
                native_to_common_lut=map_dales,
                out_channels=out_channels,
                train_ignore_index=int(train_ignore_index),
                logger=master,
            )
            master.info("[map] pred_to_common derived from CKPT LUT + DALES native->common")
        elif src_domain == "eclair":
            # ECLAIR-trained model: source native->common MUST be ECLAIR mapping
            pred_to_common = build_pred_to_common_from_ckpt_lut(
                native_to_train_lut=lut_model_native_to_train,
                native_to_common_lut=eclair_native_to_common_lut,
                out_channels=out_channels,
                train_ignore_index=int(train_ignore_index),
                logger=master,
            )
            master.info("[map] pred_to_common derived from CKPT LUT + ECLAIR native->common")
        else:
            # Last-resort: keep your previous behavior (but loudly warn)
            master.warning(
                "[map] Could not infer checkpoint domain; falling back to legacy ECLAIR derivation paths. "
                "If this is a DALES checkpoint, you MUST provide --mapping_pred_to_common or fix ckpt cfg keys."
            )
            pred_to_common = None
            if args.try_import_label_maps:
                pred_to_common = derive_pred_to_common_map(
                    eclair_native_to_common_lut,
                    try_import=True,
                    undefined_id=int(args.eclair_undefined_id),
                    ignore_index=int(args.eclair_ignore_index),
                    logger=master,
                )
                if pred_to_common is not None:
                    pred_to_common = fit_lut_to_out_channels(pred_to_common, out_channels, logger=master)
                    master.info(f"[map] Derived pred_to_common via imported label maps (out_channels={out_channels})")
            if pred_to_common is None:
                pred_to_common = build_pred_to_common_from_eclair_map(
                    eclair_native_to_common=eclair_native_to_common_dict,
                    out_channels=out_channels,
                )
                pred_to_common = fit_lut_to_out_channels(pred_to_common, out_channels, logger=master)
                master.info(f"[map] Fallback pred_to_common using native_id=train_id+1 (out_channels={out_channels})")

    if pred_to_common is None:
        raise RuntimeError(
            "Could not construct model-output-id -> common mapping.\n"
            "Provide --mapping_pred_to_common YAML (output ids -> common ids), "
            "or run with --try_import_label_maps if your repo exposes src.label_maps.eclair_native_to_train_ids."
        )

    common_class_names = [
        "ignore",
        "ground",
        "vegetation",
        "buildings",
        "wires",
        "poles",
        "fence",
        "vehicle",
    ]
    C = len(common_class_names)

    # HARD VALIDATION: prevent silent clipping corruption
    if int(map_dales.max()) >= C:
        raise RuntimeError(
            f"mapping_dales_native_to_common outputs id {int(map_dales.max())} but C={C}. "
            f"Fix mapping or expand common_class_names."
        )
    if int(pred_to_common.max()) >= C:
        raise RuntimeError(
            f"pred_to_common outputs id {int(pred_to_common.max())} but C={C}. " f"Fix mapping_eclair_to_common / pred mapping."
        )

    # list DALES files
    droot = Path(args.dales_root)
    files = sorted(list(droot.rglob("*.las")) + list(droot.rglob("*.laz")))

    if not files:
        raise RuntimeError(f"No LAS/LAZ files found under {droot}")

    def _infer_target_domain(droot: Path, map_native_to_common_path: str) -> str:
        s = (str(droot) + " " + str(map_native_to_common_path)).lower()
        if "eclair" in s:
            return "eclair"
        if "dales" in s:
            return "dales"
        return "unknown"

    target_domain = _infer_target_domain(droot, args.mapping_dales_native_to_common)
    master.info(f"[data] inferred_target_domain={target_domain}")

    # ---- target native -> train LUT (ONLY needed for train-space metrics / class-aware voxel cap) ----
    if args.mapping_native_to_train is not None:
        lut_train = load_yaml_lut_fixed(
            args.mapping_native_to_train,
            size=256,
            default=int(train_ignore_index),
        )
        master.info(f"[trainmap] Using provided --mapping_native_to_train: {args.mapping_native_to_train}")

    elif target_domain != "unknown" and src_domain != "unknown" and target_domain != src_domain:
        # Cross-domain: checkpoint LUT is NOT the target dataset LUT.
        lut_train = None
        master.warning(
            f"[trainmap] target_domain({target_domain}) != ckpt_domain({src_domain}). "
            "Train-space metrics will be SKIPPED. "
            "Pass --mapping_native_to_train if you explicitly want train-space eval / class-aware voxel cap."
        )

    else:
        # Same-domain (or unknown): using ckpt LUT is acceptable
        lut_train = lut_model_native_to_train
        master.info("[trainmap] Using CKPT native->train LUT for GT mapping (same-domain/unknown)")

    # baseline spec (MATCH TRAINING)
    base = PreprocSpec(
        coord_norm_factor=float(args.coord_norm_factor),
        voxel_size=float(args.voxel_size),
        z_scale=1.0,
        returns_mode="as_is",
        returns_k=int(args.returns_k),
        voxel_feat_mode="sample_first",  # closest to your training cache behavior
        mean_thin_p=0.7,
        knn_k=16,
        tta_mode="none",
        patch_filter_mode="none",
        min_points=1000,
        min_occ_vox=1000,
    )

    if args.diag_ckpt:
        # Use a single logger (master) and run the one-shot diagnostic then exit.
        diagnose_ckpt_vs_evalspec(
            ckpt_path=args.ckpt,
            base_spec=base,
            model=model,
            amp=amp,
            dales_files=files,
            map_dales_to_common=map_dales,
            pred_to_common=pred_to_common,
            common_class_names=common_class_names,
            device=device,
            logger=master,
            patch_size_m=args.patch_size_m,
            patch_stride_m=args.patch_stride_m,
            out_dir=args.out_dir,
        )
        return

    # -----------------------------
    # Resume mode: run only pending run_ids, but keep the ORIGINAL plan ordering/indices
    # -----------------------------
    pending_dicts = None
    if args.pending_specs_json is not None:
        pending_path = Path(args.pending_specs_json)
        pending_dicts = json.loads(pending_path.read_text())
        master.info(f"[plan] RESUME pending-only from {pending_path} n_specs={len(pending_dicts)}")

    # plan = make_decisive_returns_plan_returns_only(base)
    # master.info(f"[plan] returns-only decisive plan n_runs={len(plan)} device={device.type}")
    # master.info(f"[paths] dales_root={droot} ckpt={args.ckpt}")
    if args.plan == "voxel_ablation":
        # Phase-1: ONLY voxel_size ablation, keep everything else fixed.
        extras = _parse_csv_floats(str(args.voxel_ablation_values))

        # Build plan: baseline first, then extras (dedup, preserve order)
        plan = []
        seen = set()
        for v in [float(base.voxel_size)] + [float(x) for x in extras]:
            vv = round(float(v), 8)
            if vv in seen:
                continue
            seen.add(vv)
            if abs(float(v) - float(base.voxel_size)) < 1e-12:
                plan.append(base)
            else:
                plan.append(dataclasses.replace(base, voxel_size=float(v)))

        master.info(f"[plan] voxel_ablation n_runs={len(plan)} device={device.type}")
        master.info(f"[plan] baseline voxel_size={base.voxel_size} + extras={extras}")
        if len(plan) != 4:
            master.warning(
                f"[plan] Expected 4 runs for Phase-1, but got {len(plan)}. "
                f"Baseline={base.voxel_size}, extras={extras}. "
                f"Tip: set --voxel_ablation_values to exactly 3 unique values."
            )
    else:
        plan = make_decisive_returns_plan_returns_only(base)
        master.info(f"[plan] returns-only decisive plan n_runs={len(plan)} device={device.type}")

    master.info(f"[paths] dales_root={droot} ckpt={args.ckpt}")
    master.info(
        f"[baseline] coord_norm_factor={base.coord_norm_factor} voxel_size={base.voxel_size} "
        f"returns_k={base.returns_k} tta_mode={base.tta_mode} voxel_feat_mode={base.voxel_feat_mode}"
    )
    pending_set = None
    if pending_dicts is not None:
        # Build plan index: canonical spec -> run_id
        plan_index: Dict[Tuple, str] = {}
        collisions = 0
        for s in plan:
            k = spec_key_from_spec(s)
            rid = s.to_id()
            if k in plan_index and plan_index[k] != rid:
                collisions += 1
            plan_index[k] = rid

        if collisions:
            master.warning(f"[resume] plan_index had {collisions} canonical collisions (unexpected, but continuing).")

        pending_set = set()
        missing = []
        for d in pending_dicts:
            k = spec_key_from_dict(d)
            rid = plan_index.get(k)
            if rid is None:
                missing.append(d)
            else:
                pending_set.add(rid)

        master.info(f"[resume] pending_matched={len(pending_set)} pending_missing={len(missing)}")

        # SANITY CHECK: log missing specs loudly (do not silently proceed)
        if missing:
            master.error(
                "[resume] Some pending specs did NOT match the current plan. "
                "This usually means your sweep generation changed, or spec fields differ."
            )
            # Print compact view
            for i, md in enumerate(missing, 1):
                master.error(f"[resume][missing {i}] {md}")
            raise RuntimeError(
                "Pending specs do not match the generated plan. "
                "Fix plan drift or regenerate pending_specs.json from this exact code version."
            )

    # run sweep
    results: Dict[str, Dict] = {}
    baseline_id = plan[0].to_id()

    for i, spec in enumerate(plan, 1):
        run_id = spec.to_id()

        # In resume mode, only execute pending run_ids, but preserve original i (folder numbering)
        if pending_set is not None and run_id not in pending_set:
            continue

        run_dir = out_root / f"run_{i:03d}_{run_id}"
        logger = setup_logger(run_dir, f"preproc_sweep.{run_id}", level=logging.INFO)
        (run_dir / "spec.json").write_text(json.dumps(dataclasses.asdict(spec), indent=2))

        try:
            # Deterministic per-spec: prevents RNG consumption leaking across runs
            spec_seed = (int(args.seed) ^ int(run_id[:8], 16)) & 0xFFFFFFFF
            set_all_seeds(spec_seed)

            model.eval()
            torch.set_grad_enabled(False)

            res = evaluate_run(
                model=model,
                dales_files=files,
                map_dales_to_common=map_dales,
                pred_to_common=pred_to_common,
                common_class_names=common_class_names,
                spec=spec,
                device=device,
                logger=logger,
                patch_size_m=args.patch_size_m,
                patch_stride_m=args.patch_stride_m,
                amp=amp,
                map_dales_native_to_train=lut_train,
                train_ignore_index=int(train_ignore_index),
                train_max_voxels=train_max_voxels,
                train_class_aware_max_voxels=train_class_aware_max_voxels,
                rare_class_ids=tuple(rare_class_ids),
            )
            (run_dir / "metrics.json").write_text(json.dumps(res, indent=2))
            results[run_id] = {"failed": False, **res}
        except Exception as e:
            logger.exception(f"[run] FAILED: {e}")
            results[run_id] = {
                "failed": True,
                "error": str(e),
                "spec": dataclasses.asdict(spec),
            }
            continue

        full = res["full"]
        master.info(
            f"[done] {i}/{len(plan)} run_id={run_id} full_miou={full['miou']:.4f} "
            f"wires={full['iou_per_class'][4]:.4f} poles={full['iou_per_class'][5]:.4f} "
            f"time={res['runtime_s']:.1f}s"
        )
        # run-level cleanup (important for long sweeps)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # -----------------------------
    # IMPORTANT: Always rebuild "results" from disk before final aggregation.
    # This makes resume mode write combined results.csv/json for ALL completed runs.
    # -----------------------------
    results = {}
    done = 0
    for rd in sorted(out_root.glob("run_[0-9][0-9][0-9]_*")):
        mp = rd / "metrics.json"
        if not mp.exists():
            continue
        try:
            res = json.loads(mp.read_text())
            rid = rd.name.split("_")[-1]
            results[rid] = {"failed": False, **res}
            done += 1
        except Exception:
            continue

    master.info(f"[reload] completed_from_disk={done}")

    # compute deltas vs baseline (FULL scope + FILTERED scope separately)
    baseline = results.get(baseline_id)
    if baseline is None or baseline.get("failed"):
        master.error("[baseline] missing/failed -> cannot compute deltas.")
        return

    b_full_miou = float(baseline["full"]["miou"])
    b_full_iou = np.array(baseline["full"]["iou_per_class"], dtype=np.float64)

    b_f_miou = float(baseline["filtered"]["miou"])
    b_f_iou = np.array(baseline["filtered"]["iou_per_class"], dtype=np.float64)

    rows = []
    for run_id, r in results.items():
        if r.get("failed"):
            continue

        full = r["full"]
        filt = r["filtered"]

        miou = float(full["miou"])
        iou = np.array(full["iou_per_class"], dtype=np.float64)
        d_miou = miou - b_full_miou
        d_iou = iou - b_full_iou

        miou_f = float(filt["miou"])
        iou_f = np.array(filt["iou_per_class"], dtype=np.float64)
        d_miou_f = miou_f - b_f_miou
        d_iou_f = iou_f - b_f_iou

        rows.append(
            {
                "run_id": run_id,
                "full_miou": miou,
                "full_miou_delta": float(d_miou),
                "full_wires": float(iou[4]),
                "full_wires_delta": float(d_iou[4]),
                "full_poles": float(iou[5]),
                "full_poles_delta": float(d_iou[5]),
                "full_fence": float(iou[6]),
                "full_fence_delta": float(d_iou[6]),
                "full_vehicle": float(iou[7]),
                "full_vehicle_delta": float(d_iou[7]),
                "filt_miou": miou_f,
                "filt_miou_delta": float(d_miou_f),
                "filt_wires_delta": float(d_iou_f[4]),
                "filt_poles_delta": float(d_iou_f[5]),
                "n_points_full": int(full["n_points"]),
                "n_patches_full": int(full["n_patches"]),
                "n_patches_kept": int(filt["n_patches_kept"]),
                "n_patches_dropped": int(filt["n_patches_dropped"]),
                "runtime_s": float(r["runtime_s"]),
                "spec": r["spec"],
                "align_avg": r.get("align_avg", {}),
            }
        )

    rows.sort(key=lambda x: x["full_miou_delta"], reverse=True)

    (out_root / "results.json").write_text(json.dumps({"baseline_id": baseline_id, "rows": rows}, indent=2))

    header = (
        "run_id,full_miou,full_miou_delta,full_wires,full_wires_delta,full_poles,full_poles_delta,"
        "full_fence,full_fence_delta,full_vehicle,full_vehicle_delta,"
        "filt_miou,filt_miou_delta,filt_wires_delta,filt_poles_delta,"
        "n_points_full,n_patches_full,n_patches_kept,n_patches_dropped,runtime_s"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['run_id']},{r['full_miou']:.6f},{r['full_miou_delta']:.6f},"
            f"{r['full_wires']:.6f},{r['full_wires_delta']:.6f},"
            f"{r['full_poles']:.6f},{r['full_poles_delta']:.6f},"
            f"{r['full_fence']:.6f},{r['full_fence_delta']:.6f},"
            f"{r['full_vehicle']:.6f},{r['full_vehicle_delta']:.6f},"
            f"{r['filt_miou']:.6f},{r['filt_miou_delta']:.6f},"
            f"{r['filt_wires_delta']:.6f},{r['filt_poles_delta']:.6f},"
            f"{r['n_points_full']},{r['n_patches_full']},{r['n_patches_kept']},{r['n_patches_dropped']},"
            f"{r['runtime_s']:.3f}"
        )
    (out_root / "results.csv").write_text("\n".join(lines))

    master.info(f"[saved] {out_root / 'results.csv'}")
    master.info(f"[saved] {out_root / 'results.json'}")
    master.info("[done] scientific preproc sweep complete.")


if __name__ == "__main__":
    main()


# import laspy


# def read_las_arrays_robust(path: Path) -> dict[str, np.ndarray]:
#     """
#     Reads LAS/LAZ files using robust property-access to avoid bit-packing bugs.
#     Standardizes output keys to: xyz, intensity, return_number, number_of_returns, rgb, native_labels.
#     """
#     try:
#         las = laspy.read(str(path))
#     except Exception as e:
#         raise RuntimeError(f"Failed to read LAS file {path}: {e}")

#     # Standardize XYZ to float32
#     xyz = np.array(las.xyz, dtype=np.float64)

#     def _get_dim(attr_name: str, fallback_names: list[str] = None) -> np.ndarray:
#         # Priority 1: Direct property access (handles bit-unpacking/scaling)
#         if hasattr(las, attr_name):
#             val = getattr(las, attr_name)
#             return np.array(val)

#         # Priority 2: Dictionary access (fallback for non-standard names)
#         # Check standard dimension names case-insensitively
#         dims_lower = set(d.lower() for d in las.point_format.dimension_names)

#         if attr_name.lower() in dims_lower:
#             return np.array(las[attr_name])

#         if fallback_names:
#             for name in fallback_names:
#                 if name.lower() in dims_lower:
#                     return np.array(las[name])
#         return None

#     # Intensity
#     intensity = _get_dim("intensity")
#     if intensity is None:
#         intensity = np.zeros((xyz.shape[0],), dtype=np.float32)
#     else:
#         intensity = intensity.astype(np.float32)

#     # Returns (CRITICAL FIX: Use property access)
#     rn = _get_dim("return_number")
#     nor = _get_dim("number_of_returns")

#     if rn is None:
#         rn = np.ones((xyz.shape[0],), dtype=np.int64)
#     else:
#         rn = rn.astype(np.int64)

#     if nor is None:
#         nor = np.ones((xyz.shape[0],), dtype=np.int64)
#     else:
#         nor = nor.astype(np.int64)

#     # Labels
#     labels = _get_dim("classification", fallback_names=["raw_classification"])
#     if labels is None:
#         # Fallback for datasets that might be unlabeled
#         # print(f" [WARN] No classification found in {path.name}, using zeros.") # Optional logging
#         labels = np.zeros((xyz.shape[0],), dtype=np.int64)
#     else:
#         labels = labels.astype(np.int64)

#     # RGB
#     rgb = None
#     red = _get_dim("red")
#     green = _get_dim("green")
#     blue = _get_dim("blue")

#     if red is not None and green is not None and blue is not None:
#         max_val = max(red.max(), green.max(), blue.max())
#         scale = 1.0
#         if max_val > 255:
#             scale = 1.0 / 65535.0
#         r = red.astype(np.float32) * scale
#         g = green.astype(np.float32) * scale
#         b = blue.astype(np.float32) * scale
#         rgb = np.stack([r, g, b], axis=1)

#     return {
#         "xyz": xyz,
#         "intensity": intensity,
#         "return_number": rn,
#         "number_of_returns": nor,
#         "rgb": rgb,
#         "native_labels": labels,
#     }


# MAP_ECLAIR_COMMON = "/csehome/m23csa510/lidar_experiments/cross-sensor-3d-lidar-segmentation-and-dom-adpt/eclair_model_train/configs/mapping_dales_to_common.yaml"
# ECLAIR_DIR = "/scratch/m23csa510/dales/dales/all"
# lut = load_yaml_lut_fixed(MAP_ECLAIR_COMMON, size=256, default=0)
# counts = np.zeros(8, dtype=np.int64)
# files = sorted(list(Path(ECLAIR_DIR).rglob("*.las")) + list(Path(ECLAIR_DIR).rglob("*.laz")))

# for fp in files:
#     data = read_las_arrays_robust(fp)
#     gt_native = data["native_labels"]
#     gt_common = lut[np.clip(gt_native, 0, 255)]
#     c = np.bincount(gt_common.astype(np.int64), minlength=8)
#     counts[:8] += c[:8]

# print("Counts per common id 0..7:", counts)
# print("Vehicle fraction:", counts[7] / counts.sum())
# print("Fence fraction:", counts[6] / counts.sum())
