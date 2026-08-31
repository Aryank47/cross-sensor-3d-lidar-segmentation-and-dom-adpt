from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from .dg_dataset import get_dg_dataset_contract


METHOD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MethodContract:
    name: str
    mix3d_enabled: bool
    ocons_enabled: bool
    bev_als_enabled: bool
    legacy_bev_enabled: bool
    schema_version: int = METHOD_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _flag(cfg: Dict[str, Any], *path: str) -> bool:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict):
            return False
        cur = cur.get(key, {})
    return bool(cur)


def _require_keys(mapping: Dict[str, Any], keys: set[str], context: str) -> None:
    missing = sorted(keys.difference(mapping))
    if missing:
        raise ValueError(f"{context} is missing required explicit fields: {missing}")


def validate_method_contract(cfg: Dict[str, Any]) -> MethodContract:
    """Validate that exactly the intended DG method is active.

    Older baseline/Mix3D configs without ``method.name`` remain usable and are
    inferred. New O-CONS and BEV-ALS configs must declare the method explicitly.
    """

    mix3d = _flag(cfg, "data", "mix3d", "enabled")
    ocons = _flag(cfg, "ocons", "enabled")
    bev_als = _flag(cfg, "model", "aux_heads", "bev_als", "enabled")
    legacy_bev = _flag(cfg, "model", "aux_heads", "bev", "enabled")

    method_cfg = cfg.get("method", {}) or {}
    declared = str(method_cfg.get("name", "")).strip().lower()
    if not declared:
        if ocons or bev_als:
            raise ValueError("New DG methods require an explicit method.name.")
        declared = "mix3d" if mix3d else ("legacy_bev" if legacy_bev else "baseline")

    allowed = {"baseline", "mix3d", "ocons", "bev_als", "mix3d_ocons", "legacy_bev"}
    if declared not in allowed:
        raise ValueError(f"Unknown method.name={declared!r}; expected one of {sorted(allowed)}")

    expected = {
        "baseline": (False, False, False, False),
        "mix3d": (True, False, False, False),
        "ocons": (False, True, False, False),
        "bev_als": (False, False, True, False),
        "legacy_bev": (False, False, False, True),
    }
    if declared == "mix3d_ocons":
        if not bool(method_cfg.get("allow_combined", False)):
            raise ValueError("mix3d_ocons is reserved for a later experiment; set method.allow_combined=true explicitly.")
        wanted = (True, True, False, False)
    else:
        wanted = expected[declared]

    actual = (mix3d, ocons, bev_als, legacy_bev)
    if actual != wanted:
        raise ValueError(
            "Method flags do not match method.name: "
            f"name={declared!r} expected(mix3d,ocons,bev_als,legacy_bev)={wanted} actual={actual}"
        )

    if bev_als:
        bev = (((cfg.get("model", {}) or {}).get("aux_heads", {}) or {}).get("bev_als", {}) or {})
        _require_keys(
            bev,
            {
                "enabled", "feature_level", "xy_frame", "half_extent_m", "resolution_m",
                "y_flip", "height_mode", "height_slices", "feature_pool", "input_channels",
                "pre_pool_dim", "decoder_hidden_dim", "target_mode", "metric_threshold",
                "min_in_bounds_fraction", "min_height_edge_gap_m", "loss",
            },
            "model.aux_heads.bev_als",
        )
        loss = bev.get("loss", {}) or {}
        _require_keys(
            loss,
            {"bce_weight", "dice_weight", "max_weight", "warmup_epochs"},
            "model.aux_heads.bev_als.loss",
        )
        required = {
            "feature_level": "block8",
            "xy_frame": "bbox_centered",
            "height_mode": "quantile",
            "height_slices": 4,
            "feature_pool": "max",
            "target_mode": "height_sliced_multilabel",
        }
        for key, value in required.items():
            got = bev[key]
            if got != value:
                raise ValueError(f"BEV-ALS v1 requires {key}={value!r}; got {got!r}")
        if "warmup_only_bev" in bev or "bev_selected_idx" in bev:
            raise ValueError("Legacy warmup_only_bev/bev_selected_idx fields are invalid for BEV-ALS.")

    if ocons:
        raw = cfg.get("ocons", {}) or {}
        _require_keys(
            raw,
            {
                "enabled", "perturbation", "application_probability", "mask_fraction",
                "protected_class_ids", "protected_min_voxels", "protected_min_fraction",
                "seed_offset", "consistency",
            },
            "ocons",
        )
        _require_keys(
            raw.get("consistency", {}) or {},
            {"max_weight", "warmup_epochs", "temperature", "macro_min_voxels"},
            "ocons.consistency",
        )

    if ocons or bev_als:
        data = cfg.get("data", {}) or {}
        dataset_contract = get_dg_dataset_contract(str(data.get("dataset", "")))
        label_space = data.get("label_space", {}) or {}
        num_classes = int(label_space.get("num_classes", -1))
        out_channels = int((cfg.get("model", {}) or {}).get("out_channels", -1))
        if num_classes != dataset_contract.num_classes or out_channels != num_classes:
            raise ValueError(
                "DG method class dimensions do not match the dataset contract: "
                f"dataset={dataset_contract.name} expected={dataset_contract.num_classes} "
                f"label_space={num_classes} model.out_channels={out_channels}."
            )
        if ocons:
            protected = tuple(int(x) for x in (cfg.get("ocons", {}) or {}).get("protected_class_ids", []))
            if protected != dataset_contract.utility_class_ids:
                raise ValueError(
                    f"O-CONS protected_class_ids={protected} do not match "
                    f"{dataset_contract.name} utility IDs={dataset_contract.utility_class_ids}."
                )

    return MethodContract(
        name=declared,
        mix3d_enabled=mix3d,
        ocons_enabled=ocons,
        bev_als_enabled=bev_als,
        legacy_bev_enabled=legacy_bev,
    )
