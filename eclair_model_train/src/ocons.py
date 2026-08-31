from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OConsConfig:
    enabled: bool = False
    perturbation: str = "active_voxel_mask"
    application_probability: float = 1.0
    mask_fraction: float = 0.10
    protected_class_ids: Tuple[int, ...] = ()
    protected_min_voxels: int = 3
    protected_min_fraction: float = 0.70
    consistency_weight: float = 0.50
    consistency_warmup_epochs: int = 10
    temperature: float = 1.0
    macro_min_voxels: int = 8
    seed_offset: int = 91000

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "OConsConfig":
        raw = cfg.get("ocons", {}) or {}
        cons = raw.get("consistency", {}) or {}
        out = OConsConfig(
            enabled=bool(raw.get("enabled", False)),
            perturbation=str(raw.get("perturbation", "active_voxel_mask")).lower(),
            application_probability=float(raw.get("application_probability", 1.0)),
            mask_fraction=float(raw.get("mask_fraction", 0.10)),
            protected_class_ids=tuple(int(x) for x in raw.get("protected_class_ids", [])),
            protected_min_voxels=int(raw.get("protected_min_voxels", 3)),
            protected_min_fraction=float(raw.get("protected_min_fraction", 0.70)),
            consistency_weight=float(cons.get("max_weight", 0.50)),
            consistency_warmup_epochs=int(cons.get("warmup_epochs", 10)),
            temperature=float(cons.get("temperature", 1.0)),
            macro_min_voxels=int(cons.get("macro_min_voxels", 8)),
            seed_offset=int(raw.get("seed_offset", 91000)),
        )
        if out.enabled:
            if out.perturbation != "active_voxel_mask":
                raise ValueError("O-CONS v1 supports only perturbation=active_voxel_mask.")
            if out.application_probability != 1.0:
                raise ValueError("O-CONS v1 requires application_probability=1.0.")
            if not (0.0 < out.mask_fraction < 1.0):
                raise ValueError("ocons.mask_fraction must be in (0, 1).")
            if not (0.0 < out.protected_min_fraction <= 1.0):
                raise ValueError("ocons.protected_min_fraction must be in (0, 1].")
            if out.protected_min_voxels < 1 or out.temperature <= 0.0:
                raise ValueError("O-CONS minimum voxels and temperature must be positive.")
            if out.consistency_weight < 0.0 or out.consistency_warmup_epochs < 0:
                raise ValueError("O-CONS consistency weight and warmup epochs must be non-negative.")
            if out.macro_min_voxels < 1:
                raise ValueError("ocons.consistency.macro_min_voxels must be positive.")
            if len(set(out.protected_class_ids)) != len(out.protected_class_ids):
                raise ValueError("ocons.protected_class_ids must not contain duplicates.")
            if any(class_id < 0 for class_id in out.protected_class_ids):
                raise ValueError("ocons.protected_class_ids must be non-negative.")
        return out


@dataclass
class OConsDiagnostics:
    requested_mask_fraction: float
    actual_mask_fraction: float
    clean_voxels: int
    perturbed_voxels: int
    jaccard: float
    constrained_samples: int
    samples: int
    protected_min_retention_observed: float
    class_before: torch.Tensor
    class_after: torch.Tensor
    protected_disappearances: torch.Tensor


@dataclass
class OConsView:
    coordinates: torch.Tensor
    features: torch.Tensor
    labels: torch.Tensor
    clean_input_indices: torch.Tensor
    diagnostics: OConsDiagnostics


def stable_ocons_seed(base_seed: int, epoch: int, global_step: int, rank: int, seed_offset: int) -> int:
    h = hashlib.blake2b(digest_size=8)
    for value in (base_seed, epoch, global_step, rank, seed_offset):
        h.update(str(int(value)).encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "little", signed=False) & 0x7FFFFFFF


def make_masked_view(
    coordinates: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    cfg: OConsConfig,
    *,
    num_classes: int,
    generator: torch.Generator,
) -> OConsView:
    """Create a strict per-sample subset of collated active coordinates."""

    if coordinates.device.type != "cpu" or features.device.type != "cpu" or labels.device.type != "cpu":
        raise ValueError("make_masked_view expects the CPU tensors returned by the DataLoader.")
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("coordinates must be Minkowski batched coordinates [N,4].")
    n_total = int(coordinates.shape[0])
    if n_total != int(features.shape[0]) or n_total != int(labels.shape[0]):
        raise ValueError("O-CONS coordinate/feature/label row counts differ.")

    keep = torch.ones(n_total, dtype=torch.bool)
    constrained_samples = 0
    protected_min_retention_observed = 1.0
    sample_ids = torch.unique(coordinates[:, 0].to(torch.int64), sorted=True)

    invalid_protected = [class_id for class_id in cfg.protected_class_ids if class_id >= int(num_classes)]
    if invalid_protected:
        raise ValueError(
            f"O-CONS protected class IDs {invalid_protected} are outside num_classes={num_classes}."
        )

    for sample_id in sample_ids.tolist():
        sample_idx = torch.nonzero(coordinates[:, 0].to(torch.int64) == int(sample_id), as_tuple=False).squeeze(1)
        n = int(sample_idx.numel())
        requested_remove = min(n - 1, max(1, int(round(float(cfg.mask_fraction) * n)))) if n > 1 else 0
        forced_keep = torch.zeros(n, dtype=torch.bool)
        sample_labels = labels.index_select(0, sample_idx).to(torch.int64)

        for class_id in cfg.protected_class_ids:
            local = torch.nonzero(sample_labels == int(class_id), as_tuple=False).squeeze(1)
            nc = int(local.numel())
            if nc == 0:
                continue
            n_keep = min(nc, max(int(cfg.protected_min_voxels), int(math.ceil(cfg.protected_min_fraction * nc))))
            order = torch.randperm(nc, generator=generator)
            forced_keep[local.index_select(0, order[:n_keep])] = True

        eligible = torch.nonzero(~forced_keep, as_tuple=False).squeeze(1)
        n_remove = min(requested_remove, int(eligible.numel()))
        if n_remove < requested_remove:
            constrained_samples += 1
        if n_remove > 0:
            order = torch.randperm(int(eligible.numel()), generator=generator)
            remove_local = eligible.index_select(0, order[:n_remove])
            keep[sample_idx.index_select(0, remove_local)] = False

        sample_keep = keep.index_select(0, sample_idx)
        for class_id in cfg.protected_class_ids:
            before = int((sample_labels == int(class_id)).sum().item())
            if before == 0:
                continue
            after = int((sample_labels[sample_keep] == int(class_id)).sum().item())
            required = min(
                before,
                max(
                    int(cfg.protected_min_voxels),
                    int(math.ceil(float(cfg.protected_min_fraction) * before)),
                ),
            )
            if after < required:
                raise RuntimeError(
                    "O-CONS protected-class retention invariant failed: "
                    f"sample={sample_id} class={class_id} before={before} after={after} required={required}."
                )
            protected_min_retention_observed = min(
                protected_min_retention_observed,
                float(after) / float(before),
            )

    clean_indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
    class_before = torch.zeros(num_classes, dtype=torch.int64)
    class_after = torch.zeros(num_classes, dtype=torch.int64)
    valid_before = (labels >= 0) & (labels < num_classes)
    valid_after_labels = labels.index_select(0, clean_indices)
    valid_after = (valid_after_labels >= 0) & (valid_after_labels < num_classes)
    if bool(valid_before.any()):
        class_before += torch.bincount(labels[valid_before].to(torch.int64), minlength=num_classes)[:num_classes]
    if bool(valid_after.any()):
        class_after += torch.bincount(valid_after_labels[valid_after].to(torch.int64), minlength=num_classes)[:num_classes]
    disappear = torch.zeros(num_classes, dtype=torch.int64)
    for class_id in cfg.protected_class_ids:
        if int(class_before[class_id]) > 0 and int(class_after[class_id]) == 0:
            disappear[class_id] = 1
    if int(disappear.sum()) != 0:
        raise RuntimeError(f"O-CONS protected class disappearance: {torch.nonzero(disappear).flatten().tolist()}")

    n_after = int(clean_indices.numel())
    actual_mask = 1.0 - (float(n_after) / max(1, n_total))
    return OConsView(
        coordinates=coordinates.index_select(0, clean_indices),
        features=features.index_select(0, clean_indices),
        labels=labels.index_select(0, clean_indices),
        clean_input_indices=clean_indices,
        diagnostics=OConsDiagnostics(
            requested_mask_fraction=float(cfg.mask_fraction),
            actual_mask_fraction=actual_mask,
            clean_voxels=n_total,
            perturbed_voxels=n_after,
            jaccard=float(n_after) / max(1, n_total),
            constrained_samples=constrained_samples,
            samples=int(sample_ids.numel()),
            protected_min_retention_observed=float(protected_min_retention_observed),
            class_before=class_before,
            class_after=class_after,
            protected_disappearances=disappear,
        ),
    )


def _exact_coordinate_keys(reference: torch.Tensor, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError("Coordinate arrays must be [N,D] and share D.")
    combined = torch.cat([reference.to(torch.int64), query.to(torch.int64)], dim=0)
    mins = combined.min(dim=0).values
    maxs = combined.max(dim=0).values
    spans_t = maxs - mins + 1
    spans = [int(v) for v in spans_t.detach().cpu().tolist()]
    product = 1
    for span in spans:
        product *= span
        if product > (2**63 - 1):
            raise OverflowError("Exact mixed-radix coordinate key would overflow int64.")

    def encode(x: torch.Tensor) -> torch.Tensor:
        shifted = x.to(torch.int64) - mins
        key = shifted[:, 0]
        for dim in range(1, shifted.shape[1]):
            key = key * spans[dim] + shifted[:, dim]
        return key

    return encode(reference), encode(query)


def match_sparse_coordinates(reference: torch.Tensor, query: torch.Tensor, *, require_all: bool = True) -> Tuple[torch.Tensor, float]:
    """Return reference row indices for query coordinates using collision-free integer keys."""

    if query.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=query.device), 1.0
    ref_key, query_key = _exact_coordinate_keys(reference, query)
    sorted_key, order = torch.sort(ref_key)
    if sorted_key.numel() > 1 and bool((sorted_key[1:] == sorted_key[:-1]).any()):
        raise RuntimeError("Duplicate reference sparse coordinates.")
    q_sorted, _ = torch.sort(query_key)
    if q_sorted.numel() > 1 and bool((q_sorted[1:] == q_sorted[:-1]).any()):
        raise RuntimeError("Duplicate query sparse coordinates.")
    pos = torch.searchsorted(sorted_key, query_key)
    in_range = pos < sorted_key.numel()
    safe_pos = pos.clamp_max(max(0, int(sorted_key.numel()) - 1))
    matched = in_range & (sorted_key.index_select(0, safe_pos) == query_key)
    coverage = float(matched.float().mean().item())
    if require_all and not bool(matched.all()):
        raise RuntimeError(f"Sparse coordinate match coverage={coverage:.8f}, expected 1.0")
    result = torch.full_like(pos, -1, dtype=torch.int64)
    result[matched] = order.index_select(0, safe_pos[matched])
    return result, coverage


def ocons_consistency_loss(
    clean_logits_aligned: torch.Tensor,
    perturbed_logits: torch.Tensor,
    perturbed_labels: torch.Tensor,
    cfg: OConsConfig,
    *,
    num_classes: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if clean_logits_aligned.shape != perturbed_logits.shape:
        raise ValueError("Aligned clean and perturbed logits must have identical shape.")
    t = float(cfg.temperature)
    clean_p = F.softmax(clean_logits_aligned.detach().float() / t, dim=1).clamp_min(1e-8)
    pert_logp = F.log_softmax(perturbed_logits.float() / t, dim=1)
    kl_rows = (clean_p * (clean_p.log() - pert_logp)).sum(dim=1) * (t * t)
    valid = (perturbed_labels >= 0) & (perturbed_labels < num_classes)
    micro = kl_rows[valid].mean() if bool(valid.any()) else perturbed_logits.sum() * 0.0

    per_class_kl = torch.zeros(num_classes, dtype=torch.float32, device=perturbed_logits.device)
    per_class_support = torch.zeros(num_classes, dtype=torch.float32, device=perturbed_logits.device)
    macro_terms = []
    for class_id in range(num_classes):
        m = valid & (perturbed_labels == class_id)
        support = int(m.sum().item())
        per_class_support[class_id] = support
        if support:
            per_class_kl[class_id] = kl_rows[m].mean()
        if support >= int(cfg.macro_min_voxels):
            macro_terms.append(per_class_kl[class_id])
    macro = torch.stack(macro_terms).mean() if macro_terms else micro
    loss = 0.5 * micro + 0.5 * macro

    clean_pred = clean_logits_aligned.detach().argmax(dim=1)
    pert_pred = perturbed_logits.detach().argmax(dim=1)
    agreement = (clean_pred[valid] == pert_pred[valid]).float().mean() if bool(valid.any()) else loss.detach() * 0.0
    clean_conf = clean_p.max(dim=1).values[valid].mean() if bool(valid.any()) else loss.detach() * 0.0
    pert_conf = F.softmax(perturbed_logits.detach().float(), dim=1).max(dim=1).values
    pert_conf = pert_conf[valid].mean() if bool(valid.any()) else loss.detach() * 0.0
    return loss, {
        "micro_kl": micro.detach(),
        "macro_kl": macro.detach(),
        "agreement": agreement.detach(),
        "clean_confidence": clean_conf.detach(),
        "perturbed_confidence": pert_conf.detach(),
        "per_class_kl": per_class_kl.detach(),
        "per_class_support": per_class_support.detach(),
    }


def ocons_weight_for_epoch(cfg: OConsConfig, epoch: int) -> float:
    if not cfg.enabled:
        return 0.0
    warm = int(cfg.consistency_warmup_epochs)
    if warm <= 0:
        return float(cfg.consistency_weight)
    return float(cfg.consistency_weight) * min(1.0, max(0.0, float(epoch) / float(warm)))
