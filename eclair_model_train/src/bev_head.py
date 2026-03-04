# src/bev_head.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class BEVProjectorConfig:
    # Bounds in meters in the *current coordinate frame* (your tiles are local coords)
    x_min_m: float = 0.0
    x_max_m: float = 100.0
    y_min_m: float = 0.0
    y_max_m: float = 100.0
    z_min_m: float = -10.0
    z_max_m: float = 10.0

    # If img size is not specified, use meters-per-pixel
    res_m: float = 0.20

    # Match LiDOG converter: y axis flipped in the BEV image
    y_flip: bool = True

    # How to pool multiple voxels into the same BEV cell (for FEATURES)
    # max | mean
    pool: str = "max"
    # how to build BEV features:
    #   "pool" uses max/mean over all voxels in cell (current)
    #   "select" uses bev_selected_idx map (LiDOG-faithful)
    mode: str = "pool"  # pool | select
    select_source: str = "auto"  # auto | map | coords
    select_policy: str = "last"  # last | first | random


@dataclass
class BEVHeadConfig:
    enabled: bool = False

    levels: tuple[str, ...] = ("block8",)

    # Loss config
    weight: float = 0.5
    ignore_index: int = -1
    warmup_epochs: int = 0
    warmup_only_bev: bool = True

    # --- NEW for Task 8 ---
    # LiDOG BEV label collision policy: their getBEVImageNew behaves like "last write wins".
    label_pool: str = "last"  # last | first | random | majority

    # Optional: per-level BEV image size (LiDOG uses bev_img_sizes aligned with decoder_2d_levels)
    img_size_by_level: Optional[dict[str, int]] = None

    projector: BEVProjectorConfig = BEVProjectorConfig()
    hidden_dim: int = 64

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "BEVHeadConfig":
        aux = (cfg.get("model", {}) or {}).get("aux_heads", {}) or {}
        bev = aux.get("bev", {}) or {}
        if not bev:
            return BEVHeadConfig(enabled=False)

        proj = bev.get("projector", {}) or {}
        levels = tuple(bev.get("levels", ["block8"]))

        # Parse LiDOG-style bev_img_sizes aligned to levels (optional)
        img_size_by_level = None
        bev_img_sizes = bev.get("bev_img_sizes", None)
        if bev_img_sizes is not None:
            if isinstance(bev_img_sizes, (list, tuple)):
                if len(bev_img_sizes) != len(levels):
                    raise ValueError("model.aux_heads.bev.bev_img_sizes must match len(levels).")
                img_size_by_level = {lvl: int(sz) for lvl, sz in zip(levels, bev_img_sizes)}
            elif isinstance(bev_img_sizes, dict):
                img_size_by_level = {str(k): int(v) for k, v in bev_img_sizes.items()}
            else:
                raise TypeError("bev_img_sizes must be list/tuple or dict")

        return BEVHeadConfig(
            enabled=bool(bev.get("enabled", False)),
            levels=levels,
            weight=float(bev.get("weight", 0.5)),
            ignore_index=int(bev.get("ignore_index", -1)),
            warmup_epochs=int(bev.get("warmup_epochs", 0)),
            warmup_only_bev=bool(bev.get("warmup_only_bev", True)),
            label_pool=str(bev.get("label_pool", "last")).lower(),
            img_size_by_level=img_size_by_level,
            projector=BEVProjectorConfig(
                x_min_m=float(proj.get("x_min_m", 0.0)),
                x_max_m=float(proj.get("x_max_m", 100.0)),
                y_min_m=float(proj.get("y_min_m", 0.0)),
                y_max_m=float(proj.get("y_max_m", 100.0)),
                z_min_m=float(proj.get("z_min_m", -10.0)),
                z_max_m=float(proj.get("z_max_m", 10.0)),
                res_m=float(proj.get("res_m", 0.20)),
                y_flip=bool(proj.get("y_flip", True)),
                pool=str(proj.get("pool", "max")).lower(),
                mode=str(proj.get("mode", "pool")).lower(),
                select_source=str(proj.get("select_source", "auto")).lower(),
                select_policy=str(proj.get("select_policy", "last")).lower(),
            ),
            hidden_dim=int(bev.get("hidden_dim", 64)),
        )


class SparseToBEVProjector(nn.Module):
    """
    Minimal LiDOG-style idea: project 3D sparse features to a dense BEV grid.

    Input: Minkowski sparse tensor-like object with attributes:
      - x.C : [N, 1+3] int32 (batch,x,y,z) in *voxel units*
      - x.F : [N, Cin] float

    Output:
      - bev_feats: [B, Cin, H, W]
      - valid_mask: [B, H, W] bool
    """

    def __init__(self, cfg: BEVProjectorConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        x,
        *,
        meters_per_voxel: float,
        img_size: Optional[int] = None,
        selected_idx: Optional[torch.Tensor] = None,  # [B,H,W], -1 for empty
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = x.C  # [N,1+3]
        feats_all = x.F
        device = feats_all.device

        # Full batch size (must be consistent even if some samples have no in-bounds points)
        B_all = int(coords[:, 0].max().item()) + 1 if coords.numel() > 0 else 1

        # Determine grid H/W + xGrid/yGrid (needed for pool mode and coords-based select)
        x_min, x_max = float(self.cfg.x_min_m), float(self.cfg.x_max_m)
        y_min, y_max = float(self.cfg.y_min_m), float(self.cfg.y_max_m)
        z_min, z_max = float(self.cfg.z_min_m), float(self.cfg.z_max_m)

        if img_size is not None:
            H = W = int(img_size)
            xGrid = (x_max - x_min) / float(W)
            yGrid = (y_max - y_min) / float(H)
        else:
            xGrid = yGrid = float(self.cfg.res_m)
            W = max(1, int(np.ceil((x_max - x_min) / xGrid)))
            H = max(1, int(np.ceil((y_max - y_min) / yGrid)))

        mode = str(getattr(self.cfg, "mode", "pool")).lower()

        # ----------------------------------------------------
        # SELECT MODE (LiDOG-faithful “choose one per pixel”)
        # ----------------------------------------------------
        if mode == "select":
            src = str(getattr(self.cfg, "select_source", "auto")).lower()
            pol = str(getattr(self.cfg, "select_policy", "last")).lower()
            if src not in ("auto", "map", "coords"):
                raise ValueError(f"select_source must be auto|map|coords, got {src}")
            if pol not in ("last", "first", "random"):
                raise ValueError(f"select_policy must be last|first|random, got {pol}")

            Nv = int(feats_all.shape[0])

            def _use_map(si: torch.Tensor) -> bool:
                if si is None or si.dim() != 3:
                    return False
                if int(si.shape[0]) != B_all:
                    return False
                if int(si.shape[1]) != H or int(si.shape[2]) != W:
                    return False
                mx = int(si.max().item()) if si.numel() > 0 else -1
                return mx < Nv

            # ---- (A) Try map ----
            if src in ("auto", "map") and _use_map(selected_idx):
                valid = selected_idx >= 0
                idx = selected_idx.clamp_min(0).to(torch.long)
                idx_flat = idx.reshape(-1)
                gathered = feats_all.index_select(0, idx_flat)  # [B*H*W, Cin]
                Cin = int(feats_all.shape[1])
                gathered = gathered.view(B_all, H, W, Cin).permute(0, 3, 1, 2).contiguous()
                gathered = gathered * valid.unsqueeze(1).to(gathered.dtype)
                return gathered, valid

            if src == "map":
                raise ValueError(
                    "select_source='map' but selected_idx is missing/mismatched. "
                    "Use select_source='coords' or generate per-level selected_idx."
                )

            # ---- (B) Build selected_idx from coords (per-level correct) ----
            # coords in meters
            b = coords[:, 0].to(torch.int64)
            xyz_m = coords[:, 1:4].to(torch.float32) * float(meters_per_voxel)

            inb = (
                (xyz_m[:, 0] > x_min)
                & (xyz_m[:, 0] < x_max)
                & (xyz_m[:, 1] > y_min)
                & (xyz_m[:, 1] < y_max)
                & (xyz_m[:, 2] > z_min)
                & (xyz_m[:, 2] < z_max)
            )

            out_idx = torch.full((B_all * H * W,), -1, dtype=torch.long, device=device)
            if inb.any():
                idx_in = torch.nonzero(inb, as_tuple=False).squeeze(1)  # indices into feats_all
                b_in = b[inb]
                xyz_in = xyz_m[inb]

                px = torch.floor((xyz_in[:, 0] - x_min) / xGrid).to(torch.int64).clamp_(0, W - 1)
                py = torch.floor((xyz_in[:, 1] - y_min) / yGrid).to(torch.int64).clamp_(0, H - 1)
                if bool(self.cfg.y_flip):
                    py = (H - 1) - py

                flat = b_in * (H * W) + (py * W + px)  # [M]

                # pick one voxel index per flat cell, deterministically
                # use stable sort if available
                try:
                    order = torch.argsort(flat, stable=True)
                except TypeError:
                    order = torch.argsort(flat)

                flat_s = flat[order]
                idx_s = idx_in[order]

                if pol == "first":
                    keep = torch.ones_like(flat_s, dtype=torch.bool)
                    keep[1:] = flat_s[1:] != flat_s[:-1]
                elif pol == "last":
                    keep = torch.ones_like(flat_s, dtype=torch.bool)
                    keep[:-1] = flat_s[:-1] != flat_s[1:]
                else:  # random
                    # random tie-break inside each flat group
                    r = torch.rand((flat.numel(),), device=device)
                    # sort by (flat, r) then take last → max r
                    try:
                        order2 = torch.lexsort((r, flat))  # not always available
                        order = order2
                    except Exception:
                        # fallback: sort by flat then sort each group is heavy; keep last as approx
                        order = torch.argsort(flat)
                    flat_s = flat[order]
                    idx_s = idx_in[order]
                    keep = torch.ones_like(flat_s, dtype=torch.bool)
                    keep[:-1] = flat_s[:-1] != flat_s[1:]

                sel_flat = flat_s[keep]
                sel_idx = idx_s[keep]
                out_idx[sel_flat] = sel_idx

            selected = out_idx.view(B_all, H, W)
            valid = selected >= 0

            idx_flat = selected.clamp_min(0).reshape(-1)
            gathered = feats_all.index_select(0, idx_flat)
            Cin = int(feats_all.shape[1])
            gathered = gathered.view(B_all, H, W, Cin).permute(0, 3, 1, 2).contiguous()
            gathered = gathered * valid.unsqueeze(1).to(gathered.dtype)
            return gathered, valid

        # ----------------------------------------------------
        # POOL MODE (existing max/mean over voxels)
        # ----------------------------------------------------
        b = coords[:, 0].to(torch.int64)
        xyz_m = coords[:, 1:4].to(torch.float32) * float(meters_per_voxel)
        inb = (
            (xyz_m[:, 0] > x_min)
            & (xyz_m[:, 0] < x_max)
            & (xyz_m[:, 1] > y_min)
            & (xyz_m[:, 1] < y_max)
            & (xyz_m[:, 2] > z_min)
            & (xyz_m[:, 2] < z_max)
        )

        if not inb.any():
            bev = torch.zeros((B_all, feats_all.shape[1], H, W), dtype=feats_all.dtype, device=device)
            valid = torch.zeros((B_all, H, W), dtype=torch.bool, device=device)
            return bev, valid

        b = b[inb]
        xyz_m = xyz_m[inb]
        feats = feats_all[inb]

        px = torch.floor((xyz_m[:, 0] - x_min) / xGrid).to(torch.int64).clamp_(0, W - 1)
        py = torch.floor((xyz_m[:, 1] - y_min) / yGrid).to(torch.int64).clamp_(0, H - 1)
        if bool(self.cfg.y_flip):
            py = (H - 1) - py

        HW = H * W
        flat = b * HW + (py * W + px)

        Cin = feats.shape[1]
        pool = str(self.cfg.pool).lower()
        if pool == "max":
            if not hasattr(torch.Tensor, "scatter_reduce_"):
                raise RuntimeError("BEV max pooling requires torch.scatter_reduce_")
            out = torch.full((B_all * HW, Cin), -1e30, dtype=torch.float32, device=device)
            idx = flat.view(-1, 1).expand(-1, Cin)
            out.scatter_reduce_(0, idx, feats.to(torch.float32), reduce="amax", include_self=True)
            valid = out[:, 0] > -1e20
            out = out.view(B_all, H, W, Cin).permute(0, 3, 1, 2).contiguous()
            valid = valid.view(B_all, H, W)
            return out.to(feats.dtype), valid

        if pool == "mean":
            sums = torch.zeros((B_all * HW, Cin), dtype=torch.float32, device=device)
            idx = flat.view(-1, 1).expand(-1, Cin)
            sums.scatter_add_(0, idx, feats.to(torch.float32))
            counts_raw = torch.bincount(flat, minlength=B_all * HW).to(torch.float32)
            valid = counts_raw > 0
            counts = counts_raw.clamp_min(1.0)
            out = sums / counts.view(-1, 1)
            out = out.view(B_all, H, W, Cin).permute(0, 3, 1, 2).contiguous()
            valid = valid.view(B_all, H, W)
            return out.to(feats.dtype), valid

        raise ValueError(f"Unknown BEV pool='{self.cfg.pool}'")


class BEVDecoder(nn.Module):
    """Tiny decoder scaffold. You will replace/expand this to match LiDOG's Encoder2D later."""

    def __init__(self, out_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_classes, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BEVHead(nn.Module):
    """
    Projects one or more sparse feature levels to BEV and predicts semantic logits in BEV.
    Output is a dict keyed by level (LiDOG-style).
    """

    def __init__(self, *, out_classes: int, cfg: BEVHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.projector = SparseToBEVProjector(cfg.projector)
        self.decoders = nn.ModuleDict({lvl: BEVDecoder(out_classes, hidden_dim=cfg.hidden_dim) for lvl in cfg.levels})

    def forward(
        self,
        feats_by_level: Dict[str, Any],
        *,
        meters_per_voxel: float,
        bev_selected_idx: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        for lvl, st in feats_by_level.items():
            img_size = None
            sel = None
            st = feats_by_level[lvl]
            if self.cfg.img_size_by_level is not None:
                img_size = int(self.cfg.img_size_by_level.get(lvl, 0)) or None

            if bev_selected_idx is not None:
                sel = bev_selected_idx.get(lvl, None)

            bev_feats, _valid = self.projector(
                st,
                meters_per_voxel=meters_per_voxel,
                img_size=img_size,
                selected_idx=sel,
            )
            out[lvl] = self.decoders[lvl](bev_feats)
        return out
