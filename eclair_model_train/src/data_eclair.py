# src/data_eclair.py
from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .augment import AugmentConfig, augment_xyz
from .bev_head import BEVHeadConfig
from .bev_labels import build_bev_labels_and_selected_idx
from .features import FeatureConfig, build_features
from .label_maps import eclair_native_to_train_ids
from .utils import atomic_save_torch, read_las_arrays_robust
from .voxelization import VoxelizationConfig, voxelize_from_q


@dataclass
class PatchConfig:
    make_local_coords: bool = True
    coord_norm_factor: float = 10.0
    voxel_size: float = 0.05


def _load_eclair_split_list(
    eclair_root: Union[str, Path],
    split: str,
    *,
    meta_filename: str = "labels.json",
    allowed_review_categories: Sequence[str] = ("approved",),
) -> List[str]:
    """
    Returns a list of tile filenames for the given split.

    Supports BOTH common metadata layouts:
      (A) dict-of-splits:
          {"train": ["pointcloud_1.laz", ...], "val": [...], "test": [...]}

      (B) list-of-records (your format):
          [
            {"tile_name": "pointcloud_1.laz", "split": "train", "review_category": "approved", ...},
            ...
          ]

    Parameters
    ----------
    allowed_review_categories:
        If provided, list-format metadata will be filtered by review_category.
        Use allowed_review_categories=None to disable filtering.
    """
    eclair_root = Path(eclair_root)
    meta_path = eclair_root / meta_filename
    if not meta_path.exists():
        raise FileNotFoundError(f"ECLAIR meta file not found: {meta_path}")

    with meta_path.open("r") as f:
        meta = json.load(f)

    split = str(split).lower().strip()
    valid_splits = {"train", "val", "test"}

    if split not in valid_splits:
        raise ValueError(f"Unknown split='{split}'. Expected one of {sorted(valid_splits)}")

    # ---- Case A: dict-of-splits ----
    if isinstance(meta, dict):
        # allow either lowercase or exact keys
        keys_lower = {str(k).lower(): k for k in meta.keys()}
        if split not in keys_lower:
            raise KeyError(f"{meta_path} does not contain split '{split}'. Keys: {list(meta.keys())}")
        names = meta[keys_lower[split]]
        if not isinstance(names, list):
            raise TypeError(f"{meta_path}[{keys_lower[split]}] must be a list, got: {type(names)}")
        return sorted([str(x) for x in names])

    # ---- Case B: list-of-records ----
    if isinstance(meta, list):
        out = []
        for rec in meta:
            if not isinstance(rec, dict):
                continue
            rec_split = str(rec.get("split", "")).lower().strip()
            if rec_split != split:
                continue

            if allowed_review_categories is not None:
                cat = str(rec.get("review_category", "")).lower().strip()
                if cat not in {c.lower() for c in allowed_review_categories}:
                    continue

            name = rec.get("tile_name", None)
            if name is None:
                continue

            out.append(str(name))

        if not out:
            # Helpful debugging context
            present_splits = sorted(
                {str(r.get("split", "")).lower().strip() for r in meta if isinstance(r, dict) and "split" in r}
            )
            raise RuntimeError(
                f"No tiles found for split='{split}' after filtering.\n"
                f"meta_path={meta_path}\n"
                f"present_splits={present_splits}\n"
                f"allowed_review_categories={allowed_review_categories}"
            )

        return sorted(list(dict.fromkeys(out)))  # stable unique

    raise TypeError(f"Unsupported JSON structure in {meta_path}: {type(meta)}")


def _resolve_pc_path(eclair_root: Path, fname: str) -> Path:
    pc_dir = eclair_root / "pointclouds"
    p = pc_dir / fname
    if p.exists():
        return p
    # fallback: maybe stored as .las
    alt = p.with_suffix(".las")
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not find pointcloud file for '{fname}' under {pc_dir}")


def _read_las_arrays(path: Path) -> Dict[str, np.ndarray]:
    """
    Robust reader wrapper for ECLAIR training.
    """
    # Matches old output: xyz, intensity, return_number, number_of_returns, rgb, native_labels
    return read_las_arrays_robust(path)


def _sha1_hex(x: bytes) -> str:
    return hashlib.sha1(x).hexdigest()


def _onehot_u8(vals: np.ndarray, k: int) -> np.ndarray:
    # vals are expected 1..k, clamp to [1..k], then shift to 0..k-1
    v = vals.astype(np.int64, copy=False)
    v = np.clip(v, 1, k) - 1
    out = np.zeros((v.shape[0], k), dtype=np.uint8)
    out[np.arange(v.shape[0]), v] = 1
    return out


def _cache_key_for_eclair_raw(pc_path: Path) -> str:
    st = pc_path.stat()
    key_obj = {
        "v": "eclair_raw_cache_v1",
        "path": str(pc_path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    s = json.dumps(key_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha1_hex(s)[:24]


class EclairTiles(Dataset):
    """
    ECLAIR tile dataset (per-tile samples).
    Produces MinkowskiEngine-ready (coords, feats, labels) AFTER quantization (one label per voxel).

    This class supports optional .pt caching for faster I/O:
      cache_root/<split>/<tile_stem>.pt
    """

    def __init__(
        self,
        *,
        eclair_root: str | Path,
        split: str,
        is_train: bool,
        patch_cfg: PatchConfig,
        aug_cfg: AugmentConfig,
        feat_cfg: FeatureConfig,
        num_classes: int,
        ignore_index: int = -100,
        undefined_id: int = 0,
        use_cache: bool = True,
        cache_root: Optional[str | Path] = None,
        seed: int = 1337,
        meta_filename: str = "labels.json",
        allowed_review_categories: Optional[Sequence[str]] = ("approved",),
        voxel_cfg: Optional[VoxelizationConfig] = None,
        bev_cfg: Optional[BEVHeadConfig] = None,
        run_cache_root: Optional[str | Path] = None,
        run_cache_precompute_returns_onehot: bool = False,
    ):
        self.eclair_root = Path(eclair_root)
        self.split = split
        self.is_train = is_train
        self.patch_cfg = patch_cfg
        self.aug_cfg = aug_cfg
        self.feat_cfg = feat_cfg
        self.ignore_index = ignore_index
        self.undefined_id = undefined_id

        self.names = _load_eclair_split_list(
            self.eclair_root,
            split,
            meta_filename=meta_filename,
            allowed_review_categories=allowed_review_categories,
        )

        self.use_cache = use_cache and cache_root is not None
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self._rng = np.random.default_rng(seed + (0 if split == "train" else 10))
        self.voxel_cfg = voxel_cfg or VoxelizationConfig()
        self.bev_cfg = bev_cfg
        self.num_classes = int(num_classes)
        self.cache_kind = "raw"  # ECLAIR cache stores raw arrays only
        if self.use_cache and self.cache_root is not None:
            (self.cache_root / self.split / "raw").mkdir(parents=True, exist_ok=True)
        self.run_cache_root = Path(run_cache_root) if run_cache_root is not None else None
        self.run_cache_precompute_returns_onehot = bool(run_cache_precompute_returns_onehot)
        self._run_cache_dir = None
        if self.run_cache_root is not None:
            # run_cache_root is run-scoped (under out_dir/_run_cache/...), so no staleness across runs
            self._run_cache_dir = self.run_cache_root / "eclair" / str(self.split)
            self._run_cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _run_cache_path(self, fname: str) -> Path:
        assert self._run_cache_dir is not None
        key = _sha1_hex(fname.encode("utf-8"))[:24]
        return self._run_cache_dir / f"{key}.pt"

    def _materialize_invariant(self, raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # xyz in float64, localize once per run-config
        xyz = raw["xyz"].astype(np.float64, copy=False)
        if self.patch_cfg.make_local_coords:
            xyz = xyz - xyz.min(axis=0, keepdims=True)

        # labels in TRAIN-ID space (undefined -> ignore)
        y_native = raw.get("native_labels", None)
        if y_native is None:
            y_native = raw.get("labels", None)
        if y_native is None:
            raise KeyError("ECLAIR raw must include native_labels or labels")
        y_native = y_native.astype(np.int64, copy=False)

        y_train = y_native - 1
        y_train[y_native == int(self.undefined_id)] = int(self.ignore_index)

        out = {
            # IMPORTANT: keep existing interface keys
            "xyz": xyz,  # float64, localized if enabled
            "native_labels": y_native.astype(np.int64, copy=False),
            "y_train": y_train.astype(np.int64, copy=False),
            "intensity": raw.get("intensity", None),
            "return_number": raw.get("return_number", None),
            "number_of_returns": raw.get("number_of_returns", None),
            "rgb": raw.get("rgb", None),
        }

        if self.run_cache_precompute_returns_onehot:
            k = int(self.feat_cfg.returns_onehot_k)
            rn = out["return_number"]
            nor = out["number_of_returns"]
            if rn is not None:
                out["rn_1h_u8"] = _onehot_u8(rn, k)
            if nor is not None:
                out["nor_1h_u8"] = _onehot_u8(nor, k)

        return out

    def __len__(self) -> int:
        return len(self.names)

    def _cache_path(self, fname: str) -> Optional[Path]:
        if not self.use_cache or self.cache_root is None:
            return None

        # legacy path (backward compatible)
        legacy = self.cache_root / self.split / f"{Path(fname).stem}.pt"
        if legacy.exists():
            return legacy

        # new robust key path
        pc_path = _resolve_pc_path(self.eclair_root, fname)
        key = _cache_key_for_eclair_raw(pc_path)
        return self.cache_root / self.split / "raw" / f"{key}.pt"

    def _load_raw_base(self, fname: str) -> Dict[str, np.ndarray]:
        """
        Original raw-loading behavior: disk-cache (if enabled) else LAS/LAZ.
        Returns dict with keys: xyz, native_labels, intensity, return_number, number_of_returns, rgb.
        """

        # 1) Try cache
        cpath = self._cache_path(fname)
        if cpath is not None and cpath.exists():
            obj = torch.load(cpath, map_location="cpu")

            def _is_cache_fresh(cmeta: Optional[dict], pc_path: Path) -> bool:
                if not isinstance(cmeta, dict):
                    return True  # old caches: treat as fresh (backward compatible)
                try:
                    st = pc_path.stat()
                    return (int(cmeta.get("source_size", -1)) == int(st.st_size)) and (
                        int(cmeta.get("source_mtime_ns", -1)) == int(st.st_mtime_ns)
                    )
                except Exception:
                    return True

            # Case A: old cache stored a PyG Data object
            if isinstance(obj, Data):
                # Support both .xyz or .pos
                if hasattr(obj, "xyz"):
                    xyz = obj.xyz
                elif hasattr(obj, "pos"):
                    xyz = obj.pos
                else:
                    raise KeyError(f"{cpath} Data missing xyz/pos attributes")

                # Old cache often stored raw labels in .classification
                if hasattr(obj, "classification"):
                    native_labels = obj.classification
                elif hasattr(obj, "y"):
                    native_labels = obj.y
                else:
                    raise KeyError(f"{cpath} Data missing classification/y")

                def to_np(x):
                    if x is None:
                        return None
                    if isinstance(x, torch.Tensor):
                        return x.detach().cpu().numpy()
                    return np.asarray(x)

                out = {
                    "xyz": to_np(xyz).astype(np.float64, copy=False),
                    "native_labels": to_np(native_labels).astype(np.int64, copy=False),
                    "intensity": to_np(getattr(obj, "intensity", None)),
                    "return_number": to_np(getattr(obj, "return_number", None)),
                    "number_of_returns": to_np(getattr(obj, "number_of_returns", None)),
                    "rgb": to_np(getattr(obj, "rgb", None)),
                }

                if out["intensity"] is not None:
                    out["intensity"] = out["intensity"].astype(np.float32, copy=False)
                if out["return_number"] is not None:
                    out["return_number"] = out["return_number"].astype(np.int64, copy=False)
                if out["number_of_returns"] is not None:
                    out["number_of_returns"] = out["number_of_returns"].astype(np.int64, copy=False)
                if out["rgb"] is not None:
                    out["rgb"] = out["rgb"].astype(np.float32, copy=False)

                if "xyz" not in out or "native_labels" not in out:
                    raise TypeError(f"ECLAIR cache must be raw (xyz + native_labels). Bad cache: {cpath}")

                return out

            # Case B: new raw-cache dict (the script above writes this)
            if isinstance(obj, dict) and "xyz" in obj and "native_labels" in obj:
                # If cache meta exists, verify against current LAS/LAZ file. If stale, ignore cache.
                try:
                    pc_path = _resolve_pc_path(self.eclair_root, fname)
                    if not _is_cache_fresh(obj.get("_cache_meta", None), pc_path):
                        # stale cache -> fall back to reading LAS/LAZ (WITH dtype hygiene)
                        raw = _read_las_arrays(pc_path)

                        raw["xyz"] = raw["xyz"].astype(np.float64, copy=False)
                        if "native_labels" in raw and raw["native_labels"] is not None:
                            raw["native_labels"] = raw["native_labels"].astype(np.int64, copy=False)
                        if raw.get("intensity", None) is not None:
                            raw["intensity"] = raw["intensity"].astype(np.float32, copy=False)
                        if raw.get("return_number", None) is not None:
                            raw["return_number"] = raw["return_number"].astype(np.int64, copy=False)
                        if raw.get("number_of_returns", None) is not None:
                            raw["number_of_returns"] = raw["number_of_returns"].astype(np.int64, copy=False)
                        if raw.get("rgb", None) is not None:
                            raw["rgb"] = raw["rgb"].astype(np.float32, copy=False)

                        return raw
                except Exception:
                    pass  # don't break training if meta check fails

                # these are usually numpy arrays already, just ensure dtypes
                xyz = obj["xyz"].astype(np.float64, copy=False)
                y = obj["native_labels"].astype(np.int64, copy=False)
                intensity = obj.get("intensity", None)
                rn = obj.get("return_number", None)
                nor = obj.get("number_of_returns", None)
                rgb = obj.get("rgb", None)

                if intensity is not None:
                    intensity = intensity.astype(np.float32, copy=False)
                if rn is not None:
                    rn = rn.astype(np.int64, copy=False)
                if nor is not None:
                    nor = nor.astype(np.int64, copy=False)
                if rgb is not None:
                    rgb = rgb.astype(np.float32, copy=False)

                out = {
                    "xyz": xyz,
                    "native_labels": y,
                    "intensity": intensity,
                    "return_number": rn,
                    "number_of_returns": nor,
                    "rgb": rgb,
                }

                if "xyz" not in out or "native_labels" not in out:
                    raise TypeError(f"ECLAIR cache must be raw (xyz + native_labels). Bad cache: {cpath}")

                return out

            raise TypeError(f"Unsupported cache object type in {cpath}: {type(obj)}")

        # 2) No cache -> read LAS/LAZ
        pc_path = _resolve_pc_path(self.eclair_root, fname)
        raw = _read_las_arrays(pc_path)

        # dtype hygiene (robustness)
        raw["xyz"] = raw["xyz"].astype(np.float64, copy=False)
        if "native_labels" in raw and raw["native_labels"] is not None:
            raw["native_labels"] = raw["native_labels"].astype(np.int64, copy=False)
        if raw.get("intensity", None) is not None:
            raw["intensity"] = raw["intensity"].astype(np.float32, copy=False)
        if raw.get("return_number", None) is not None:
            raw["return_number"] = raw["return_number"].astype(np.int64, copy=False)
        if raw.get("number_of_returns", None) is not None:
            raw["number_of_returns"] = raw["number_of_returns"].astype(np.int64, copy=False)
        if raw.get("rgb", None) is not None:
            raw["rgb"] = raw["rgb"].astype(np.float32, copy=False)
        return raw

    def _load_raw(self, fname: str) -> Dict[str, np.ndarray]:
        """
        If run-cache enabled: return per-run invariants (xyz localized, y_train, optional onehots),
        otherwise return the base raw dict.
        """
        if self._run_cache_dir is not None:
            rp = self._run_cache_path(fname)
            if rp.exists():
                obj = torch.load(rp, map_location="cpu")
                if not isinstance(obj, dict) or "xyz" not in obj or "y_train" not in obj:
                    raise TypeError(f"Bad ECLAIR run-cache payload: {rp}")
                return obj

            base = self._load_raw_base(fname)
            inv = self._materialize_invariant(base)
            try:
                atomic_save_torch(inv, rp)
            except Exception as e:
                # do not kill training for cache I/O issues
                if not getattr(self, "_warned_run_cache_write", False):
                    print(
                        f"[warn] ECLAIR run-cache write failed once: {e}. Continuing without caching for this item.", flush=True
                    )
                    self._warned_run_cache_write = True

            return inv

        return self._load_raw_base(fname)

    def get_raw(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Return raw arrays for a tile (NO augmentation).
        Used by point-wise eval / window inference.
        """
        fname = self.names[idx]
        raw = self._load_raw(fname)

        xyz = raw["xyz"].astype(np.float64, copy=False)
        # If run-cache is enabled, xyz is already localized; this is idempotent either way.
        if self.patch_cfg.make_local_coords:
            xyz = xyz - xyz.min(axis=0, keepdims=True)

        out = dict(raw)
        out["xyz"] = xyz
        out["tile_name"] = str(fname)
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fname = self.names[idx]
        raw = self._load_raw(fname)

        xyz = raw["xyz"]
        # idempotent if run-cache already localized
        if self.patch_cfg.make_local_coords:
            xyz = xyz - xyz.min(axis=0, keepdims=True)

        h = zlib.crc32(fname.encode("utf-8")) & 0xFFFFFFFF
        sample_seed = (self.seed * 1000003 + self.epoch * 9176 + h) & 0x7FFFFFFF
        rng = np.random.default_rng(int(sample_seed))

        if self.is_train:
            xyz = augment_xyz(xyz, self.aug_cfg, rng)

        # Normalize coordinates before quantization
        xyz_norm = (xyz / float(self.patch_cfg.coord_norm_factor)).astype(np.float32, copy=False)
        xyz_norm = np.ascontiguousarray(xyz_norm, dtype=np.float32)

        # Labels: native -> contiguous train ids (ignore undefined)
        y_pts = raw.get("y_train", None)
        if y_pts is None:
            y_pts = eclair_native_to_train_ids(
                raw["native_labels"],
                undefined_id=self.undefined_id,
                ignore_index=self.ignore_index,
            ).astype(np.int64, copy=False)
        else:
            y_pts = y_pts.astype(np.int64, copy=False)

        # Build per-point features (only pass what is needed)
        intensity_arg = raw["intensity"] if self.feat_cfg.use_intensity else None
        rn_arg = raw["return_number"] if self.feat_cfg.use_return_number else None
        nor_arg = raw["number_of_returns"] if self.feat_cfg.use_number_of_returns else None
        rgb_arg = raw["rgb"] if self.feat_cfg.use_rgb else None

        feats_p = build_features(
            xyz_local=xyz_norm,
            intensity=intensity_arg,
            return_number=rn_arg,
            number_of_returns=nor_arg,
            rgb=rgb_arg,
            cfg=self.feat_cfg,
            return_number_1h=raw.get("rn_1h_u8", None),
            number_of_returns_1h=raw.get("nor_1h_u8", None),
        ).astype(np.float32, copy=False)

        # Quantize coords (int32 contiguous)
        q = np.floor(xyz_norm / float(self.patch_cfg.voxel_size)).astype(np.int32, copy=False)
        q = np.ascontiguousarray(q, dtype=np.int32)

        # Pool to voxels according to Task-6 voxel_cfg
        vx = voxelize_from_q(
            q_int32=q,
            feats_p_f32=feats_p,
            labels_p_i64=y_pts,
            ignore_index=int(self.ignore_index),
            cfg=self.voxel_cfg,
            rng=rng,  # used if feat_pool='random'
            return_maps=False,
            num_classes_hint=int(self.num_classes),
        )

        coords_t = torch.from_numpy(np.ascontiguousarray(vx["coords_u"], dtype=np.int32)).int()
        feats_t = torch.from_numpy(np.ascontiguousarray(vx["feats_u"], dtype=np.float32)).float()
        labels_t = torch.from_numpy(np.ascontiguousarray(vx["labels_u"], dtype=np.int64)).long()

        # --- BEV supervision (Task 8) ---
        if self.bev_cfg is not None and self.bev_cfg.enabled and self.is_train:
            m_per_vox = float(self.patch_cfg.voxel_size) * float(self.patch_cfg.coord_norm_factor)

            coords_np = coords_t.cpu().numpy().astype(np.int32, copy=False)
            labels_np = labels_t.cpu().numpy().astype(np.int64, copy=False)

            bev_labels = {}
            bev_selected = {}
            for lvl in self.bev_cfg.levels:
                lbl, sel = build_bev_labels_and_selected_idx(
                    coords_vox_int32=coords_np,
                    labels_vox_i64=labels_np,
                    voxel_ignore_index=int(self.ignore_index),
                    bev_cfg=self.bev_cfg,
                    level=str(lvl),
                    meters_per_voxel=m_per_vox,
                    rng=rng,  # deterministic per sample
                )
                bev_labels[str(lvl)] = torch.from_numpy(lbl).long()
                bev_selected[str(lvl)] = torch.from_numpy(sel).long()

            # always store dict (matches LiDOG multi-level collation)
            self_out_bev_labels = bev_labels
            self_out_bev_sel = bev_selected
        else:
            self_out_bev_labels = None
            self_out_bev_sel = None

        out = {
            "coords": coords_t,
            "feats": feats_t,
            "labels": labels_t,
            "fname": fname,
        }

        if self_out_bev_labels is not None:
            out["bev_labels"] = self_out_bev_labels
            out["bev_selected_idx"] = self_out_bev_sel

        return out


def minkowski_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    coords_list = [b["coords"] for b in batch]
    feats_list = [b["feats"] for b in batch]
    labels_list = [b["labels"] for b in batch]

    coords, feats, labels = ME.utils.sparse_collate(coords_list, feats_list, labels_list)
    fnames = [b["fname"] for b in batch]

    out = {"coords": coords, "feats": feats, "labels": labels, "fnames": fnames}

    # Optional BEV supervision (future LiDOG head)
    if "bev_labels" in batch[0]:
        bev0 = batch[0]["bev_labels"]
        if isinstance(bev0, dict):
            out["bev_labels"] = {k: torch.stack([b["bev_labels"][k] for b in batch], dim=0) for k in bev0.keys()}
        else:
            out["bev_labels"] = torch.stack([b["bev_labels"] for b in batch], dim=0)

        if "bev_selected_idx" in batch[0]:
            sel0 = batch[0]["bev_selected_idx"]
            if isinstance(sel0, dict):
                out["bev_selected_idx"] = {k: torch.stack([b["bev_selected_idx"][k] for b in batch], dim=0) for k in sel0.keys()}
            else:
                out["bev_selected_idx"] = torch.stack([b["bev_selected_idx"] for b in batch], dim=0)

    return out
