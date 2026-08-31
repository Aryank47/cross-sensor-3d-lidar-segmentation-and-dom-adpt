# src/data_eclair.py
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .augment import AugmentConfig, augment_xyz
from .bev_als_config import ALSBEVConfig
from .bev_als_targets import build_als_bev_target
from .bev_head import BEVHeadConfig
from .bev_labels import build_bev_labels_and_selected_idx
from .features import FeatureConfig, build_features
from .label_maps import eclair_native_to_train_ids
from .mix3d import (
    MIX_SKIP_NO_DONOR,
    MIX_SKIP_PROBABILITY,
    ALSMix3DConfig,
    MixDiagnostics,
    choose_different_source_index,
    compose_crop_replace,
    should_apply_mix,
    stable_seed,
    unchanged_mix_result,
)
from .utils import atomic_save_torch, read_las_arrays_robust
from .voxelization import VoxelizationConfig, voxelize_from_q


@dataclass
class PatchConfig:
    make_local_coords: bool = True
    coord_norm_factor: float = 10.0
    voxel_size: float = 0.05


@dataclass
class EclairSamplingConfig:
    """
    ECLAIR training sampler config.

    mode:
      - "tiles": original behavior; one dataset item = one ECLAIR tile.
      - "weighted_tiles": virtual weighted tile resampling for train split only.

    This does NOT crop ECLAIR tiles. It only changes how often each full tile
    appears in the training epoch.
    """

    mode: str = "tiles"  # tiles | weighted_tiles

    # Multiplier over the base number of tiles.
    # Keep 1.0 for C1-E to avoid changing the number of optimizer steps too much.
    epoch_size_multiplier: float = 1.0

    # ECLAIR train ids:
    # 5 transmission wires, 6 distribution wires, 7 poles,
    # 8 transmission towers, 9 fence, 10 vehicle.
    rare_class_ids: Tuple[int, ...] = (5, 6, 7, 8, 9, 10)

    # Optional manual class multipliers.
    rare_class_weights: Optional[Dict[int, float]] = None

    # beta=0 disables inverse-frequency correction;
    # beta=1 uses full inverse-frequency correction.
    rare_balance_beta: float = 0.5

    # Overall strength of the rare-tile boost.
    strength: float = 1.0

    # Safety clamp on final tile weights after normalization.
    min_weight: float = 0.25
    max_weight: float = 5.0

    # Stats cache is optional; it is stored under the run-cache root when available.
    use_stats_cache: bool = True

    @staticmethod
    def from_cfg(cfg: Optional[Dict[str, Any]]) -> "EclairSamplingConfig":
        cfg = cfg or {}

        raw_weights = cfg.get("rare_class_weights", None)
        if raw_weights is not None:
            raw_weights = {int(k): float(v) for k, v in dict(raw_weights).items()}

        return EclairSamplingConfig(
            mode=str(cfg.get("mode", "tiles")).lower().strip(),
            epoch_size_multiplier=float(cfg.get("epoch_size_multiplier", 1.0)),
            rare_class_ids=tuple(int(x) for x in cfg.get("rare_class_ids", [5, 6, 7, 8, 9, 10])),
            rare_class_weights=raw_weights,
            rare_balance_beta=float(cfg.get("rare_balance_beta", 0.5)),
            strength=float(cfg.get("strength", 1.0)),
            min_weight=float(cfg.get("min_weight", 0.25)),
            max_weight=float(cfg.get("max_weight", 5.0)),
            use_stats_cache=bool(cfg.get("use_stats_cache", True)),
        )


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
        require_cache: bool = False,
        require_cache_metadata: bool = False,
        seed: int = 1337,
        meta_filename: str = "labels.json",
        allowed_review_categories: Optional[Sequence[str]] = ("approved",),
        voxel_cfg: Optional[VoxelizationConfig] = None,
        bev_cfg: Optional[BEVHeadConfig] = None,
        bev_als_cfg: Optional[ALSBEVConfig] = None,
        run_cache_root: Optional[str | Path] = None,
        run_cache_precompute_returns_onehot: bool = False,
        sampling_cfg: Optional[EclairSamplingConfig] = None,
        mix3d_cfg: Optional[ALSMix3DConfig] = None,
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
        self.require_cache = bool(require_cache)
        self.require_cache_metadata = bool(require_cache_metadata)
        if self.require_cache and not self.use_cache:
            raise ValueError("ECLAIR require_cache=true requires use_cache=true and a cache_root.")
        if self.require_cache_metadata and not self.require_cache:
            raise ValueError("ECLAIR require_cache_metadata=true requires require_cache=true.")
        self.voxel_cfg = voxel_cfg or VoxelizationConfig()
        self.bev_cfg = bev_cfg
        self.bev_als_cfg = bev_als_cfg or ALSBEVConfig(enabled=False)
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
        self._epoch_shared = mp.Value("q", 0, lock=False)
        self.sampling_cfg = sampling_cfg or EclairSamplingConfig()
        self.mix3d_cfg = mix3d_cfg or ALSMix3DConfig(enabled=False)
        self._sampling_cdf: Optional[np.ndarray] = None
        self._virtual_epoch_size: Optional[int] = None
        self._sampling_summary: Optional[Dict[str, Any]] = None

        if self.mix3d_cfg.enabled and not self.is_train:
            raise ValueError("Mix3D may be enabled only for the ECLAIR training split.")
        if self.mix3d_cfg.enabled and self.bev_cfg is not None and self.bev_cfg.enabled:
            raise ValueError("Initial M1 forbids BEV auxiliary supervision.")
        if self.mix3d_cfg.enabled and self.bev_als_cfg.enabled:
            raise ValueError("The discovery BEV-ALS run must be isolated from Mix3D.")

        if self.is_train and str(self.sampling_cfg.mode).lower() in ("weighted_tiles", "rare_tiles"):
            self._init_weighted_tile_sampling()
        elif str(self.sampling_cfg.mode).lower() not in ("tiles", "weighted_tiles", "rare_tiles"):
            raise ValueError(f"Unknown ECLAIR sampling mode='{self.sampling_cfg.mode}'. " "Expected 'tiles' or 'weighted_tiles'.")

    def _init_weighted_tile_sampling(self) -> None:
        cfg = self.sampling_cfg

        if self._base_len() <= 0:
            raise RuntimeError("ECLAIR weighted sampling requested but dataset is empty.")

        counts = self._load_or_compute_label_counts()

        rare_ids = [int(c) for c in cfg.rare_class_ids if 0 <= int(c) < int(self.num_classes)]
        if len(rare_ids) == 0:
            raise ValueError("ECLAIR sampling.mode='weighted_tiles' requires at least one valid rare_class_id.")

        global_counts = counts.sum(axis=0).astype(np.float64)

        rare_global = np.array(
            [max(1.0, float(global_counts[c])) for c in rare_ids],
            dtype=np.float64,
        )

        # Relative reference among rare classes.
        # Median is safer than max because max can over-amplify the rarest class.
        rare_ref = float(np.median(rare_global))
        beta = float(cfg.rare_balance_beta)

        score = np.ones((self._base_len(),), dtype=np.float64)

        for c in rare_ids:
            present = counts[:, c] > 0
            if not np.any(present):
                continue

            user_w = 1.0
            if cfg.rare_class_weights is not None:
                user_w = float(cfg.rare_class_weights.get(int(c), 1.0))

            # Corrected relative inverse-frequency term.
            # This avoids the near-zero-weight bug from raw 1 / count^beta.
            freq_w = (rare_ref / max(1.0, float(global_counts[c]))) ** beta

            score[present] += float(cfg.strength) * user_w * float(freq_w)

        if not np.isfinite(score).all() or float(score.sum()) <= 0:
            raise RuntimeError("Invalid ECLAIR weighted sampling scores.")

        # Normalize to mean ~1, then clamp.
        score = score / max(float(score.mean()), 1e-12)
        score = np.clip(score, float(cfg.min_weight), float(cfg.max_weight))

        prob = score / max(float(score.sum()), 1e-12)

        epoch_size = int(round(self._base_len() * float(cfg.epoch_size_multiplier)))
        epoch_size = max(self._base_len(), epoch_size)

        self._sampling_cdf = np.cumsum(prob, dtype=np.float64)
        self._sampling_cdf[-1] = 1.0
        self._virtual_epoch_size = int(epoch_size)

        self._sampling_summary = {
            "mode": str(cfg.mode),
            "base_tiles": int(self._base_len()),
            "virtual_epoch_size": int(epoch_size),
            "rare_class_ids": [int(c) for c in rare_ids],
            "rare_global_counts": {str(int(c)): int(global_counts[c]) for c in rare_ids},
            "weight_min": float(score.min()),
            "weight_mean": float(score.mean()),
            "weight_max": float(score.max()),
            "n_tiles_with_any_rare": int(np.any(counts[:, rare_ids] > 0, axis=1).sum()),
        }

        print(f"[ECLAIR sampling] {json.dumps(self._sampling_summary, indent=2)}", flush=True)

    def set_epoch(self, epoch: int) -> None:
        self._epoch_shared.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch_shared.value)

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
        if self._virtual_epoch_size is not None:
            return int(self._virtual_epoch_size)
        return len(self.names)

    def _base_len(self) -> int:
        return len(self.names)

    def _resolve_item_index(self, idx: int) -> int:
        """
        Resolve virtual dataset index -> base tile index.

        For normal mode:
          idx maps to itself.

        For weighted_tiles, the mapping is deterministic for
        (seed, epoch, virtual index) and therefore also stable with persistent
        DataLoader workers and after checkpoint resume.
        """
        if self._sampling_cdf is None:
            return int(idx)
        if self._virtual_epoch_size is None or self._virtual_epoch_size <= 0:
            raise RuntimeError("ECLAIR weighted sampling has an invalid virtual epoch size.")
        virtual_idx = int(idx) % int(self._virtual_epoch_size)
        rng = np.random.default_rng(
            stable_seed(self.seed, self.epoch, virtual_idx, "weighted_tile")
        )
        u = float(rng.random())
        return min(
            self._base_len() - 1,
            int(np.searchsorted(self._sampling_cdf, u, side="right")),
        )

    @property
    def sampling_summary(self) -> Optional[Dict[str, Any]]:
        return self._sampling_summary

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
                    return not self.require_cache_metadata
                try:
                    st = pc_path.stat()
                    return (int(cmeta.get("source_size", -1)) == int(st.st_size)) and (
                        int(cmeta.get("source_mtime_ns", -1)) == int(st.st_mtime_ns)
                    )
                except OSError:
                    return False

            # Case A: old cache stored a PyG Data object
            if isinstance(obj, Data):
                if self.require_cache_metadata:
                    raise RuntimeError(
                        f"ECLAIR strict cache mode rejects metadata-free PyG cache: {cpath}"
                    )
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
                pc_path = _resolve_pc_path(self.eclair_root, fname)
                if not _is_cache_fresh(obj.get("_cache_meta", None), pc_path):
                    if self.require_cache:
                        raise RuntimeError(
                            f"ECLAIR cache is stale or lacks required metadata: {cpath}"
                        )
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

        if self.require_cache:
            raise FileNotFoundError(
                f"ECLAIR required cache entry is missing for tile={fname!r}: {cpath}"
            )

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

    def _get_raw_by_base_index(self, base_idx: int) -> Dict[str, np.ndarray]:
        if base_idx < 0 or base_idx >= self._base_len():
            raise IndexError(base_idx)

        fname = self.names[int(base_idx)]
        raw = self._load_raw(fname)

        out = dict(raw)
        out.setdefault("tile_name", str(fname))
        return out

    def get_raw(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Raw access used by evaluation code.

        Returns an unaugmented tile using the same local-coordinate
        convention expected by evaluation and training preprocessing.

        In normal mode, idx is a base tile index.
        In weighted train mode, idx may be a virtual index, so resolve it.
        """
        base_idx = self._resolve_item_index(int(idx))
        out = self._get_raw_by_base_index(base_idx)

        out = dict(out)

        xyz_raw = np.asarray(out["xyz"])

        if xyz_raw.dtype != np.float64:
            raise TypeError(f"[ECLAIR get_raw] expected raw xyz float64, " f"got {xyz_raw.dtype}, tile={out.get('tile_name')}")

        xyz = xyz_raw

        if self.patch_cfg.make_local_coords:
            xyz = xyz - xyz.min(axis=0, keepdims=True)

        out["xyz"] = xyz

        return out

    def _sampling_stats_cache_path(self) -> Optional[Path]:
        """
        Store label-count stats in the run cache when available.

        This avoids unsafe shared writes to the global raw cache under DDP.
        """
        if not bool(self.sampling_cfg.use_stats_cache):
            return None

        root = getattr(self, "run_cache_root", None)
        if root is None:
            return None

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        key_obj = {
            "v": "eclair_sampling_label_counts_v1",
            "split": str(self.split),
            "num_classes": int(self.num_classes),
            "ignore_index": int(self.ignore_index),
            "undefined_id": int(self.undefined_id),
            "n_tiles": int(self._base_len()),
            "names": list(self.names),
        }
        key = _sha1_hex(json.dumps(key_obj, sort_keys=True, separators=(",", ":")).encode("utf-8"))[:16]
        return root / f"eclair_label_counts_{self.split}_{key}.npz"

    def _load_or_compute_label_counts(self) -> np.ndarray:
        cache_path = self._sampling_stats_cache_path()

        if cache_path is not None and cache_path.exists():
            data = np.load(cache_path)
            counts = np.asarray(data["counts"], dtype=np.int64)
            if counts.shape == (self._base_len(), int(self.num_classes)):
                return counts

        counts = self._compute_label_counts()

        if cache_path is not None:
            tmp = cache_path.with_name(cache_path.name + ".tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(f, counts=counts)
            tmp.replace(cache_path)

        return counts

    def _compute_label_counts(self) -> np.ndarray:
        """
        Compute per-tile class counts in ECLAIR TRAIN-ID space.

        Shape:
          [num_base_tiles, num_classes]
        """
        counts = np.zeros((self._base_len(), int(self.num_classes)), dtype=np.int64)

        for i in range(self._base_len()):
            fname = self.names[int(i)]
            raw = self._load_raw_base(fname)

            if "y_train" in raw:
                y = raw["y_train"].astype(np.int64, copy=False)
            else:
                y_native = raw.get("native_labels", None)
                if y_native is None:
                    y_native = raw.get("labels", None)
                if y_native is None:
                    raise KeyError("ECLAIR raw must include native_labels, labels, or y_train for sampling stats.")

                y = eclair_native_to_train_ids(
                    y_native,
                    undefined_id=int(self.undefined_id),
                    ignore_index=int(self.ignore_index),
                ).astype(np.int64, copy=False)

            valid = (y != int(self.ignore_index)) & (y >= 0) & (y < int(self.num_classes))
            if np.any(valid):
                binc = np.bincount(y[valid], minlength=int(self.num_classes))
                counts[i, :] = binc[: int(self.num_classes)].astype(np.int64, copy=False)

        return counts

    def _prepare_training_payload(
        self,
        *,
        fname: str,
        sample_seed: int,
    ):
        raw = self._load_raw(fname)
        xyz = raw["xyz"].astype(np.float64, copy=False)
        if self.patch_cfg.make_local_coords:
            xyz = xyz - xyz.min(axis=0, keepdims=True)

        # Context side is measured before rotation, because rotating a square
        # enlarges its axis-aligned bounding box by up to sqrt(2).
        span_xy = np.ptp(xyz[:, :2], axis=0)
        context_side = float(np.min(span_xy))

        rng = np.random.default_rng(int(sample_seed))
        if self.is_train:
            xyz = augment_xyz(xyz, self.aug_cfg, rng)

        y_pts = raw.get("y_train", None)
        if y_pts is None:
            y_pts = eclair_native_to_train_ids(
                raw["native_labels"],
                undefined_id=self.undefined_id,
                ignore_index=self.ignore_index,
            ).astype(np.int64, copy=False)
        else:
            y_pts = y_pts.astype(np.int64, copy=False)

        payload = {
            "xyz": np.ascontiguousarray(xyz),
            "y_train": np.ascontiguousarray(y_pts, dtype=np.int64),
            "intensity": raw.get("intensity", None),
            "return_number": raw.get("return_number", None),
            "number_of_returns": raw.get("number_of_returns", None),
            "rgb": raw.get("rgb", None),
            "rn_1h_u8": raw.get("rn_1h_u8", None),
            "nor_1h_u8": raw.get("nor_1h_u8", None),
            "source_id": str(fname),
            "context_side_xy_m": context_side,
            "sample_seed": int(sample_seed),
        }
        return payload, rng

    @staticmethod
    def _attach_mix_meta(
        out: Dict[str, torch.Tensor],
        d: MixDiagnostics,
        *,
        donor_index: int,
    ) -> None:
        out["meta_mix_applied"] = torch.tensor(d.applied, dtype=torch.int64)
        out["meta_mix_skip_code"] = torch.tensor(d.skip_code, dtype=torch.int64)
        out["meta_mix_host_points"] = torch.tensor(d.host_points, dtype=torch.int64)
        out["meta_mix_host_removed"] = torch.tensor(d.host_removed, dtype=torch.int64)
        out["meta_mix_donor_inserted"] = torch.tensor(d.donor_inserted, dtype=torch.int64)
        out["meta_mix_output_points"] = torch.tensor(d.output_points, dtype=torch.int64)
        out["meta_mix_replacement_side_m"] = torch.tensor(d.replacement_side_m, dtype=torch.float32)
        out["meta_mix_guard_band_m"] = torch.tensor(d.guard_band_m, dtype=torch.float32)
        out["meta_mix_height_shift_m"] = torch.tensor(d.height_shift_m, dtype=torch.float32)
        out["meta_mix_cross_provenance_voxels"] = torch.tensor(d.cross_provenance_voxels, dtype=torch.int64)
        out["meta_mix_removed_class_counts"] = torch.tensor(d.host_removed_class_counts, dtype=torch.int64)
        out["meta_mix_donor_class_counts"] = torch.tensor(d.donor_inserted_class_counts, dtype=torch.int64)
        out["meta_mix_output_class_counts"] = torch.tensor(d.output_class_counts, dtype=torch.int64)
        n_classes = len(d.output_class_counts)
        out["meta_mix_context_pairs"] = torch.tensor(d.donor_host_context_pairs, dtype=torch.int64).reshape(n_classes, n_classes)
        out["meta_mix_donor_index"] = torch.tensor(donor_index, dtype=torch.int64)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        virtual_idx = int(idx)
        base_idx = self._resolve_item_index(virtual_idx)
        fname = self.names[base_idx]

        sample_seed = stable_seed(
            self.seed,
            self.epoch,
            fname,
            virtual_idx,
            "eclair_augment",
        )
        payload, voxel_rng = self._prepare_training_payload(
            fname=fname,
            sample_seed=int(sample_seed),
        )

        mix_diag = None
        mix_donor_index = -1
        if self.mix3d_cfg.enabled:
            mix_rng = np.random.default_rng(stable_seed(self.seed, self.epoch, fname, virtual_idx, "mix3d"))
            voxel_edge_m = float(self.patch_cfg.voxel_size) * float(self.patch_cfg.coord_norm_factor)
            if not should_apply_mix(self.mix3d_cfg, mix_rng):
                result = unchanged_mix_result(
                    payload,
                    skip_code=MIX_SKIP_PROBABILITY,
                    guard_band_m=float(self.mix3d_cfg.guard_band_voxels) * voxel_edge_m,
                    num_classes=int(self.num_classes),
                )
            else:
                donor_idx = choose_different_source_index(
                    source_ids=self.names,
                    host_index=int(base_idx),
                    rng=mix_rng,
                    max_attempts=int(self.mix3d_cfg.max_donor_attempts),
                )
                if donor_idx is None:
                    result = unchanged_mix_result(
                        payload,
                        skip_code=MIX_SKIP_NO_DONOR,
                        guard_band_m=float(self.mix3d_cfg.guard_band_voxels) * voxel_edge_m,
                        num_classes=int(self.num_classes),
                    )
                else:
                    mix_donor_index = int(donor_idx)
                    donor_name = self.names[int(donor_idx)]
                    donor_seed = stable_seed(
                        self.seed,
                        self.epoch,
                        fname,
                        donor_name,
                        virtual_idx,
                        "donor_augment",
                    )
                    donor_payload, _ = self._prepare_training_payload(
                        fname=donor_name,
                        sample_seed=donor_seed,
                    )
                    result = compose_crop_replace(
                        host=payload,
                        donor=donor_payload,
                        cfg=self.mix3d_cfg,
                        num_classes=int(self.num_classes),
                        voxel_edge_m=voxel_edge_m,
                        rng=mix_rng,
                    )
            payload = result.sample
            mix_diag = result.diagnostics

        xyz = np.asarray(payload["xyz"])
        xyz_norm = (xyz / float(self.patch_cfg.coord_norm_factor)).astype(np.float32, copy=False)
        xyz_norm = np.ascontiguousarray(xyz_norm, dtype=np.float32)

        feats_p = build_features(
            xyz_local=xyz_norm,
            intensity=payload["intensity"] if self.feat_cfg.use_intensity else None,
            return_number=(payload["return_number"] if self.feat_cfg.use_return_number else None),
            number_of_returns=(payload["number_of_returns"] if self.feat_cfg.use_number_of_returns else None),
            rgb=payload["rgb"] if self.feat_cfg.use_rgb else None,
            cfg=self.feat_cfg,
            return_number_1h=payload.get("rn_1h_u8", None),
            number_of_returns_1h=payload.get("nor_1h_u8", None),
        ).astype(np.float32, copy=False)

        q = np.floor(xyz_norm / float(self.patch_cfg.voxel_size)).astype(np.int32, copy=False)
        q = np.ascontiguousarray(q, dtype=np.int32)
        vx = voxelize_from_q(
            q_int32=q,
            feats_p_f32=feats_p,
            labels_p_i64=np.asarray(payload["y_train"], dtype=np.int64),
            ignore_index=int(self.ignore_index),
            cfg=self.voxel_cfg,
            rng=voxel_rng,
            return_maps=False,
            num_classes_hint=int(self.num_classes),
        )

        coords_t = torch.from_numpy(np.ascontiguousarray(vx["coords_u"], dtype=np.int32)).int()
        feats_t = torch.from_numpy(np.ascontiguousarray(vx["feats_u"], dtype=np.float32)).float()
        labels_t = torch.from_numpy(np.ascontiguousarray(vx["labels_u"], dtype=np.int64)).long()

        if self.bev_als_cfg.enabled and self.is_train:
            meters_per_voxel = float(self.patch_cfg.voxel_size) * float(self.patch_cfg.coord_norm_factor)
            als_target = build_als_bev_target(
                coords_t.cpu().numpy().astype(np.int32, copy=False),
                labels_t.cpu().numpy().astype(np.int64, copy=False),
                num_classes=int(self.num_classes),
                meters_per_voxel=meters_per_voxel,
                cfg=self.bev_als_cfg,
                ignore_index=int(self.ignore_index),
            )
        else:
            als_target = None

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
                    rng=voxel_rng,  # deterministic per sample
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

        if als_target is not None:
            d = als_target.diagnostics
            out["bev_als_target"] = torch.from_numpy(als_target.target_u8)
            out["bev_als_occupied"] = torch.from_numpy(als_target.occupied_u8)
            out["bev_als_center_xy_m"] = torch.from_numpy(als_target.frame.center_xy_m).float()
            out["bev_als_height_edges_m"] = torch.from_numpy(als_target.frame.height_edges_m).float()
            out["meta_bev_als_input_valid_voxels"] = torch.tensor(d["input_valid_voxels"], dtype=torch.int64)
            out["meta_bev_als_in_bounds_valid_voxels"] = torch.tensor(d["in_bounds_valid_voxels"], dtype=torch.int64)
            out["meta_bev_als_in_bounds_fraction"] = torch.tensor(d["in_bounds_fraction"], dtype=torch.float32)
            out["meta_bev_als_occupied_per_slice"] = torch.from_numpy(
                np.asarray(d["occupied_cells_per_slice"], dtype=np.int64)
            )
            out["meta_bev_als_empty_slices"] = torch.tensor(d["empty_slices"], dtype=torch.int64)
            out["meta_bev_als_multi_label_cells"] = torch.tensor(d["multi_label_cells"], dtype=torch.int64)
            out["meta_bev_als_multi_label_fraction"] = torch.tensor(d["multi_label_fraction"], dtype=torch.float32)
            out["meta_bev_als_multi_class_xy_columns"] = torch.tensor(
                d["multi_class_xy_columns"], dtype=torch.int64
            )
            out["meta_bev_als_multi_class_xy_column_fraction"] = torch.tensor(
                d["multi_class_xy_column_fraction"], dtype=torch.float32
            )
            out["meta_bev_als_height_edge_min_gap_m"] = torch.tensor(
                d["height_edge_min_gap_m"], dtype=torch.float32
            )
            out["meta_bev_als_degenerate_height_edges"] = torch.tensor(
                d["degenerate_height_edges"], dtype=torch.int64
            )
            out["meta_bev_als_class_before"] = torch.from_numpy(np.asarray(d["class_before"], dtype=np.int64))
            out["meta_bev_als_class_in_bounds"] = torch.from_numpy(np.asarray(d["class_in_bounds"], dtype=np.int64))
            out["meta_bev_als_class_retention"] = torch.from_numpy(np.asarray(d["class_retention"], dtype=np.float32))
            out["meta_bev_als_positive_cells"] = torch.from_numpy(
                np.asarray(d["positive_cells_per_slice_class"], dtype=np.int64)
            )

        if mix_diag is not None:
            self._attach_mix_meta(
                out,
                mix_diag,
                donor_index=mix_donor_index,
            )

        return out


def minkowski_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    coords_list = [b["coords"] for b in batch]
    feats_list = [b["feats"] for b in batch]
    labels_list = [b["labels"] for b in batch]

    coords, feats, labels = ME.utils.sparse_collate(coords_list, feats_list, labels_list)
    fnames = [b["fname"] for b in batch]

    out = {"coords": coords, "feats": feats, "labels": labels, "fnames": fnames}

    for k in batch[0].keys():
        if k.startswith("meta_"):
            out[k] = torch.stack([b[k] for b in batch], dim=0)

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

    if "bev_als_target" in batch[0]:
        for key in (
            "bev_als_target",
            "bev_als_occupied",
            "bev_als_center_xy_m",
            "bev_als_height_edges_m",
        ):
            out[key] = torch.stack([b[key] for b in batch], dim=0)

    return out
