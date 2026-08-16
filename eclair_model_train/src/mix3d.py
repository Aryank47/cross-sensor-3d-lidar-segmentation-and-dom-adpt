from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


MIX_SKIP_NONE = 0
MIX_SKIP_PROBABILITY = 1
MIX_SKIP_NO_DONOR = 2
MIX_SKIP_INVALID_REGION = 3
MIX_SKIP_VOXEL_COLLISION = 4
MIX_SKIP_OUTPUT_BUDGET = 5


@dataclass(frozen=True)
class ALSMix3DConfig:
    enabled: bool = False
    policy: str = "crop_replace"
    probability: float = 1.0
    replacement_area_fraction: float = 0.25
    guard_band_voxels: int = 1
    donor_file_strategy: str = "uniform_different_source_file"
    max_donor_attempts: int = 8
    max_region_attempts: int = 8
    min_region_points: int = 2000
    min_replacement_side_m: float = 5.0
    height_alignment: str = "q05_local"
    verify_cross_provenance_collisions: bool = True
    diagnostics: bool = True

    @staticmethod
    def from_cfg(cfg: Optional[Mapping[str, Any]]) -> "ALSMix3DConfig":
        d = dict(cfg or {})
        out = ALSMix3DConfig(
            enabled=bool(d.get("enabled", False)),
            policy=str(d.get("policy", "crop_replace")).lower().strip(),
            probability=float(d.get("probability", 1.0)),
            replacement_area_fraction=float(d.get("replacement_area_fraction", 0.25)),
            guard_band_voxels=int(d.get("guard_band_voxels", 1)),
            donor_file_strategy=str(
                d.get("donor_file_strategy", "uniform_different_source_file")
            ).lower().strip(),
            max_donor_attempts=int(d.get("max_donor_attempts", 8)),
            max_region_attempts=int(d.get("max_region_attempts", 8)),
            min_region_points=int(d.get("min_region_points", 2000)),
            min_replacement_side_m=float(d.get("min_replacement_side_m", 5.0)),
            height_alignment=str(d.get("height_alignment", "q05_local")).lower().strip(),
            verify_cross_provenance_collisions=bool(
                d.get("verify_cross_provenance_collisions", True)
            ),
            diagnostics=bool(d.get("diagnostics", True)),
        )
        out.validate()
        return out

    def validate(self) -> None:
        if self.policy != "crop_replace":
            raise ValueError("M1 supports only mix3d.policy='crop_replace'.")
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError("mix3d.probability must satisfy 0 <= p <= 1.")
        if not (0.0 < self.replacement_area_fraction < 1.0):
            raise ValueError("mix3d.replacement_area_fraction must be in (0, 1).")
        if self.guard_band_voxels < 0:
            raise ValueError("mix3d.guard_band_voxels must be non-negative.")
        if self.donor_file_strategy != "uniform_different_source_file":
            raise ValueError(
                "M1 supports only donor_file_strategy='uniform_different_source_file'."
            )
        if self.max_donor_attempts <= 0 or self.max_region_attempts <= 0:
            raise ValueError("Mix3D retry counts must be positive.")
        if self.min_region_points <= 0:
            raise ValueError("mix3d.min_region_points must be positive.")
        if self.min_replacement_side_m <= 0.0:
            raise ValueError("mix3d.min_replacement_side_m must be positive.")
        if self.height_alignment != "q05_local":
            raise ValueError("M1 supports only height_alignment='q05_local'.")


@dataclass(frozen=True)
class MixDiagnostics:
    applied: int
    skip_code: int
    host_points: int
    host_removed: int
    donor_inserted: int
    output_points: int
    replacement_side_m: float
    guard_band_m: float
    height_shift_m: float
    cross_provenance_voxels: int
    host_removed_class_counts: Tuple[int, ...]
    donor_inserted_class_counts: Tuple[int, ...]
    output_class_counts: Tuple[int, ...]
    donor_host_context_pairs: Tuple[int, ...]


@dataclass(frozen=True)
class MixResult:
    sample: Dict[str, Any]
    diagnostics: MixDiagnostics


_POINT_FIELDS: Tuple[str, ...] = (
    "y_train",
    "intensity",
    "return_number",
    "number_of_returns",
    "rgb",
    "rn_1h_u8",
    "nor_1h_u8",
)


def stable_seed(base_seed: int, *parts: Any) -> int:
    """Stable 31-bit seed; unlike Python hash(), this is process-independent."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(base_seed)).encode("utf-8"))
    for part in parts:
        h.update(b"\x00")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest(), byteorder="little", signed=False) & 0x7FFFFFFF


def choose_different_source_index(
    *,
    source_ids: Sequence[str],
    host_index: int,
    rng: np.random.Generator,
    max_attempts: int,
) -> Optional[int]:
    if len(source_ids) < 2:
        return None
    host_id = str(source_ids[int(host_index)])
    for _ in range(int(max_attempts)):
        j = int(rng.integers(0, len(source_ids)))
        if j != int(host_index) and str(source_ids[j]) != host_id:
            return j
    return None


def should_apply_mix(cfg: ALSMix3DConfig, rng: np.random.Generator) -> bool:
    if not cfg.enabled or cfg.probability <= 0.0:
        return False
    if cfg.probability >= 1.0:
        return True
    return bool(rng.random() < cfg.probability)


def _validate_sample(sample: Mapping[str, Any], name: str) -> int:
    if "xyz" not in sample or "y_train" not in sample:
        raise KeyError(f"{name} Mix3D payload requires xyz and y_train.")
    xyz = np.asarray(sample["xyz"])
    y = np.asarray(sample["y_train"])
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"{name}.xyz must have shape [N, 3], got {xyz.shape}.")
    if y.ndim != 1 or y.shape[0] != xyz.shape[0]:
        raise ValueError(f"{name}.y_train must have shape [N].")
    n = int(xyz.shape[0])
    if n == 0:
        raise ValueError(f"{name} Mix3D payload is empty.")
    if not np.isfinite(xyz).all():
        raise ValueError(f"{name}.xyz contains NaN/Inf.")
    for key in _POINT_FIELDS:
        value = sample.get(key, None)
        if value is not None and np.asarray(value).shape[0] != n:
            raise ValueError(f"{name}.{key} is not point-aligned with xyz.")
    side = float(sample.get("context_side_xy_m", 0.0))
    if not np.isfinite(side) or side <= 0.0:
        raise ValueError(f"{name}.context_side_xy_m must be finite and positive.")
    if not str(sample.get("source_id", "")):
        raise ValueError(f"{name}.source_id is required.")
    return n


def unchanged_mix_result(
    host: Mapping[str, Any],
    *,
    skip_code: int,
    guard_band_m: float,
    num_classes: int,
) -> MixResult:
    n = int(np.asarray(host["xyz"]).shape[0])
    output_counts = _class_counts(np.asarray(host["y_train"]), int(num_classes))
    zeros = tuple(0 for _ in range(int(num_classes)))
    zero_pairs = tuple(0 for _ in range(int(num_classes) * int(num_classes)))
    return MixResult(
        sample=dict(host),
        diagnostics=MixDiagnostics(
            applied=0,
            skip_code=int(skip_code),
            host_points=n,
            host_removed=0,
            donor_inserted=0,
            output_points=n,
            replacement_side_m=0.0,
            guard_band_m=float(guard_band_m),
            height_shift_m=0.0,
            cross_provenance_voxels=0,
            host_removed_class_counts=zeros,
            donor_inserted_class_counts=zeros,
            output_class_counts=output_counts,
            donor_host_context_pairs=zero_pairs,
        ),
    )


def _class_counts(labels: np.ndarray, num_classes: int) -> Tuple[int, ...]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    valid = (y >= 0) & (y < int(num_classes))
    counts = np.bincount(y[valid], minlength=int(num_classes))[: int(num_classes)]
    return tuple(int(x) for x in counts.tolist())


def _void_rows(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).reshape(-1)


def _cross_voxel_collision_count(
    host_xyz: np.ndarray,
    donor_xyz: np.ndarray,
    *,
    voxel_edge_m: float,
) -> int:
    if host_xyz.shape[0] == 0 or donor_xyz.shape[0] == 0:
        return 0
    qh = np.floor(host_xyz / float(voxel_edge_m)).astype(np.int64, copy=False)
    qd = np.floor(donor_xyz / float(voxel_edge_m)).astype(np.int64, copy=False)
    uh = np.unique(_void_rows(qh))
    ud = np.unique(_void_rows(qd))
    return int(np.intersect1d(uh, ud, assume_unique=True).size)


def _sample_interval_start(
    rng: np.random.Generator,
    low: float,
    high: float,
) -> float:
    """Sample an interval start while tolerating float32 endpoint round-off.

    Crop extents are measured from post-augmentation float32 coordinates. When
    ``side`` is exactly the measured span, recomputing ``max_coord - side`` in
    Python float can be a few micro-metres below ``min_coord``. NumPy rejects
    that otherwise-degenerate interval with ``high - low < 0``.

    The caller has already constrained ``side`` to the available span, so a
    non-positive interval width means there is exactly one admissible start.
    Returning the lower endpoint preserves the intended crop geometry.
    """
    lo = float(low)
    hi = float(high)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"Non-finite Mix3D sampling interval: low={lo}, high={hi}")
    if hi <= lo:
        return lo
    return float(rng.uniform(lo, hi))


def _concat_point_field(
    host: Mapping[str, Any],
    donor: Mapping[str, Any],
    key: str,
    host_keep: np.ndarray,
    donor_take: np.ndarray,
) -> Optional[np.ndarray]:
    ha = host.get(key, None)
    da = donor.get(key, None)
    if ha is None and da is None:
        return None
    if (ha is None) != (da is None):
        raise ValueError(
            f"Host/donor payload mismatch for point field '{key}' within one source dataset."
        )
    return np.concatenate((np.asarray(ha)[host_keep], np.asarray(da)[donor_take]), axis=0)


def compose_crop_replace(
    *,
    host: Mapping[str, Any],
    donor: Mapping[str, Any],
    cfg: ALSMix3DConfig,
    num_classes: int,
    voxel_edge_m: float,
    rng: np.random.Generator,
) -> MixResult:
    """Replace one host XY region with a translated region from a different source sample."""
    host_n = _validate_sample(host, "host")
    _validate_sample(donor, "donor")
    if str(host["source_id"]) == str(donor["source_id"]):
        return unchanged_mix_result(
            host,
            skip_code=MIX_SKIP_NO_DONOR,
            guard_band_m=0.0,
            num_classes=int(num_classes),
        )
    if not np.isfinite(voxel_edge_m) or voxel_edge_m <= 0.0:
        raise ValueError("voxel_edge_m must be finite and positive.")

    guard_m = float(cfg.guard_band_voxels) * float(voxel_edge_m)
    desired_side = math.sqrt(float(cfg.replacement_area_fraction)) * min(
        float(host["context_side_xy_m"]),
        float(donor["context_side_xy_m"]),
    )

    hxyz = np.asarray(host["xyz"])
    dxyz = np.asarray(donor["xyz"])
    hmin = hxyz[:, :2].min(axis=0)
    hmax = hxyz[:, :2].max(axis=0)
    dmin = dxyz[:, :2].min(axis=0)
    dmax = dxyz[:, :2].max(axis=0)

    max_host_side = float(np.min((hmax - hmin) - 2.0 * guard_m))
    max_donor_side = float(np.min(dmax - dmin))
    side = min(float(desired_side), max_host_side, max_donor_side)
    if not np.isfinite(side) or side < float(cfg.min_replacement_side_m):
        return unchanged_mix_result(
            host,
            skip_code=MIX_SKIP_INVALID_REGION,
            guard_band_m=guard_m,
            num_classes=int(num_classes),
        )

    selected = None
    for _ in range(int(cfg.max_region_attempts)):
        hx0 = _sample_interval_start(
            rng, hmin[0] + guard_m, hmax[0] - guard_m - side
        )
        hy0 = _sample_interval_start(
            rng, hmin[1] + guard_m, hmax[1] - guard_m - side
        )
        dx0 = _sample_interval_start(rng, dmin[0], dmax[0] - side)
        dy0 = _sample_interval_start(rng, dmin[1], dmax[1] - side)

        host_region = (
            (hxyz[:, 0] >= hx0)
            & (hxyz[:, 0] < hx0 + side)
            & (hxyz[:, 1] >= hy0)
            & (hxyz[:, 1] < hy0 + side)
        )
        host_expanded = (
            (hxyz[:, 0] >= hx0 - guard_m)
            & (hxyz[:, 0] < hx0 + side + guard_m)
            & (hxyz[:, 1] >= hy0 - guard_m)
            & (hxyz[:, 1] < hy0 + side + guard_m)
        )
        donor_region = (
            (dxyz[:, 0] >= dx0)
            & (dxyz[:, 0] < dx0 + side)
            & (dxyz[:, 1] >= dy0)
            & (dxyz[:, 1] < dy0 + side)
        )
        if (
            int(host_region.sum()) >= int(cfg.min_region_points)
            and int(donor_region.sum()) >= int(cfg.min_region_points)
        ):
            selected = (hx0, hy0, dx0, dy0, host_region, host_expanded, donor_region)
            break

    if selected is None:
        return unchanged_mix_result(
            host,
            skip_code=MIX_SKIP_INVALID_REGION,
            guard_band_m=guard_m,
            num_classes=int(num_classes),
        )

    hx0, hy0, dx0, dy0, host_region, host_expanded, donor_region = selected
    host_keep = ~host_expanded
    donor_take = donor_region

    host_z_ref = float(np.quantile(hxyz[host_region, 2], 0.05))
    donor_z_ref = float(np.quantile(dxyz[donor_region, 2], 0.05))
    height_shift = host_z_ref - donor_z_ref

    donor_xyz = dxyz[donor_take].astype(hxyz.dtype, copy=True)
    donor_xyz[:, 0] += hx0 - dx0
    donor_xyz[:, 1] += hy0 - dy0
    donor_xyz[:, 2] += height_shift
    host_xyz = hxyz[host_keep]

    collision_count = 0
    if cfg.verify_cross_provenance_collisions:
        collision_count = _cross_voxel_collision_count(
            host_xyz,
            donor_xyz,
            voxel_edge_m=float(voxel_edge_m),
        )
        if collision_count != 0:
            return unchanged_mix_result(
                host,
                skip_code=MIX_SKIP_VOXEL_COLLISION,
                guard_band_m=guard_m,
                num_classes=int(num_classes),
            )

    mixed: Dict[str, Any] = dict(host)
    mixed["xyz"] = np.concatenate((host_xyz, donor_xyz), axis=0)
    for key in _POINT_FIELDS:
        value = _concat_point_field(host, donor, key, host_keep, donor_take)
        if value is not None:
            mixed[key] = value
        else:
            mixed.pop(key, None)
    mixed["donor_source_id"] = str(donor["source_id"])

    out_n = int(mixed["xyz"].shape[0])
    removed_counts = _class_counts(
        np.asarray(host["y_train"])[host_expanded], int(num_classes)
    )
    donor_counts = _class_counts(
        np.asarray(donor["y_train"])[donor_region], int(num_classes)
    )
    output_counts = _class_counts(np.asarray(mixed["y_train"]), int(num_classes))
    host_retained_presence = np.asarray(
        _class_counts(np.asarray(host["y_train"])[host_keep], int(num_classes))
    ) > 0
    donor_presence = np.asarray(donor_counts) > 0
    context_pairs = np.outer(donor_presence, host_retained_presence).astype(
        np.int64, copy=False
    )
    return MixResult(
        sample=mixed,
        diagnostics=MixDiagnostics(
            applied=1,
            skip_code=MIX_SKIP_NONE,
            host_points=host_n,
            host_removed=int(host_expanded.sum()),
            donor_inserted=int(donor_region.sum()),
            output_points=out_n,
            replacement_side_m=float(side),
            guard_band_m=float(guard_m),
            height_shift_m=float(height_shift),
            cross_provenance_voxels=int(collision_count),
            host_removed_class_counts=removed_counts,
            donor_inserted_class_counts=donor_counts,
            output_class_counts=output_counts,
            donor_host_context_pairs=tuple(int(x) for x in context_pairs.reshape(-1)),
        ),
    )
