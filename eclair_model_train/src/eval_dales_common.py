# scripts/eval_dales_common.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import MinkowskiEngine as ME
import numpy as np
import torch

from .config_loader import load_yaml
from .data_dales import (
    DalesPatchConfig,
    DalesPreprocConfig,
    DalesTiles,
    minkowski_collate_dales,
)
from .features import FeatureConfig
from .model import build_model


def load_yaml_map(path: str) -> Dict[int, int]:
    import yaml

    d = yaml.safe_load(Path(path).read_text())
    # keys can come as int or str
    out = {}
    for k, v in d.items():
        out[int(k)] = int(v)
    return out


class CommonConfusion:
    def __init__(self, num_classes: int, ignore_id: int = 0):
        self.C = int(num_classes)
        self.ignore_id = int(ignore_id)
        self.mat = torch.zeros((self.C, self.C), dtype=torch.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        # pred, gt are 1D int tensors in [0..C-1]
        m = gt != self.ignore_id
        if m.sum().item() == 0:
            return
        p = pred[m].to(torch.int64)
        g = gt[m].to(torch.int64)
        idx = g * self.C + p
        bc = torch.bincount(idx, minlength=self.C * self.C).reshape(self.C, self.C)
        self.mat += bc.cpu()

    def compute_iou(self) -> Tuple[np.ndarray, float]:
        m = self.mat.numpy().astype(np.float64)
        tp = np.diag(m)
        fp = m.sum(axis=0) - tp
        fn = m.sum(axis=1) - tp
        denom = tp + fp + fn + 1e-9
        iou = tp / denom

        # common classes are 0..7 ; ignore class 0
        valid_classes = list(range(1, self.C))
        miou = float(np.mean(iou[valid_classes]))
        return iou, miou


def build_label_lut_from_maps(
    eclair_native_to_common: Dict[int, int],
    dales_native_to_common: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      eclair_trainid_to_common: size 11, mapping train_id (0..10) -> common_id
      dales_native_to_common_lut: size 256 (safe), mapping DALES native -> common_id
    """
    # model outputs train_id 0..10 => native_id = train_id + 1
    eclair_trainid_to_common = np.zeros((11,), dtype=np.int64)
    for train_id in range(11):
        native_id = train_id + 1
        eclair_trainid_to_common[train_id] = int(
            eclair_native_to_common.get(native_id, 0)
        )

    # DALES native ids are 0..8, but make it robust
    lut = np.zeros((256,), dtype=np.int64)
    for k, v in dales_native_to_common.items():
        if 0 <= int(k) < lut.shape[0]:
            lut[int(k)] = int(v)
    return eclair_trainid_to_common, lut


@torch.no_grad()
def run_eval(
    *,
    cfg: Dict[str, Any],
    ckpt_path: str,
    dales_root: str,
    mapping_eclair_to_common: str,
    mapping_dales_to_common: str,
    out_dir: Path,
    preproc: DalesPreprocConfig,
    device: torch.device,
    dales_patch_size_m: float,
    dales_patch_stride_m: float,
    dales_min_patch_points: int,
    dales_max_patches_per_cloud: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- model ----
    model_cfg = cfg["model"]
    model = build_model(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        D=int(model_cfg.get("D", 3)),
    ).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    # supports either {"model_state": ...} (your train checkpoints) or raw state_dict
    sd = (
        state["model_state"]
        if isinstance(state, dict) and "model_state" in state
        else state
    )
    model.load_state_dict(sd, strict=True)
    model.eval()

    # ---- configs ----
    data_cfg = cfg["data"]
    patch_cfg = DalesPatchConfig(**cfg["data"]["patch"])
    feat_cfg = FeatureConfig(**cfg["data"]["features"])

    # ---- DALES patching helpers (operate in voxel-coordinate space) ----
    # In your pipeline: xyz_norm = xyz / coord_norm_factor; voxel_size is in norm units.
    # Therefore meters_per_voxel = voxel_size * coord_norm_factor.
    meters_per_voxel = float(patch_cfg.voxel_size) * float(patch_cfg.coord_norm_factor)
    if meters_per_voxel <= 0:
        raise RuntimeError(f"Invalid meters_per_voxel={meters_per_voxel}")

    def _m_to_vox(m: float) -> int:
        # ceil to be safe; minimum 1 voxel
        v = int(np.ceil(float(m) / meters_per_voxel))
        return max(1, v)

    patch_size_vox = _m_to_vox(dales_patch_size_m)
    stride_vox = _m_to_vox(dales_patch_stride_m)
    print(
        f"[dales] meters/voxel={meters_per_voxel:.3f}, patch_vox={patch_size_vox}, stride_vox={stride_vox}"
    )

    if stride_vox != patch_size_vox:
        raise RuntimeError(
            "Overlapping DALES patches (stride != patch_size) will double-count points "
            "and corrupt IoU. Use stride==patch_size, or implement fusion."
        )

    def _iter_patch_index_lists_nonoverlap(coords_xyz_cpu: torch.Tensor):
        """
        Fast path for non-overlapping windows: stride == patch_size.
        Returns list of 1D LongTensor indices for each patch.
        coords_xyz_cpu: [N,3] CPU int coords (x,y,z) (without batch column).
        """
        x = coords_xyz_cpu[:, 0]
        y = coords_xyz_cpu[:, 1]
        xmin = int(x.min().item())
        ymin = int(y.min().item())

        # patch ids in a 2D grid
        xi = ((x - xmin) // patch_size_vox).to(torch.int64)
        yi = ((y - ymin) // patch_size_vox).to(torch.int64)
        ny = int((((y.max().item()) - ymin) // patch_size_vox) + 1)
        pid = xi * ny + yi  # [N]

        order = torch.argsort(pid)
        pid_sorted = pid[order]
        uniq, counts = torch.unique_consecutive(pid_sorted, return_counts=True)

        idx_lists = []
        start = 0
        for c in counts.tolist():
            idx = order[start : start + c]
            if idx.numel() >= dales_min_patch_points:
                idx_lists.append(idx)
            start += c

        if dales_max_patches_per_cloud and dales_max_patches_per_cloud > 0:
            idx_lists = idx_lists[: int(dales_max_patches_per_cloud)]
        return idx_lists

    def _iter_patch_index_lists_general(coords_xyz_cpu: torch.Tensor):
        """
        General path for overlapping/strided windows (slower).
        Returns list of 1D LongTensor indices for each patch.
        """
        x = coords_xyz_cpu[:, 0]
        y = coords_xyz_cpu[:, 1]
        xmin, xmax = int(x.min().item()), int(x.max().item())
        ymin, ymax = int(y.min().item()), int(y.max().item())

        idx_lists = []
        for sx in range(xmin, xmax + 1, stride_vox):
            for sy in range(ymin, ymax + 1, stride_vox):
                m = (
                    (x >= sx)
                    & (x < (sx + patch_size_vox))
                    & (y >= sy)
                    & (y < (sy + patch_size_vox))
                )
                if int(m.sum().item()) >= dales_min_patch_points:
                    idx = torch.nonzero(m, as_tuple=False).squeeze(1).to(torch.int64)
                    idx_lists.append(idx)
                    if dales_max_patches_per_cloud and dales_max_patches_per_cloud > 0:
                        if len(idx_lists) >= int(dales_max_patches_per_cloud):
                            return idx_lists
        return idx_lists

    def _iter_patch_index_lists(coords_cpu: torch.Tensor):
        # coords_cpu is [N,4] (b,x,y,z). We patch per-batch-item below.
        # Here we assume coords_cpu passed is already filtered to a single batch item and we pass coords_xyz_cpu.
        raise AssertionError("Use per-batch-item helper")

    # DALES loader
    ds = DalesTiles(
        dales_root=dales_root,
        patch_cfg=patch_cfg,
        feat_cfg=feat_cfg,
        ignore_index=int(data_cfg["label_space"]["ignore_index"]),
        preproc=preproc,
        seed=int(cfg["run"]["seed"]) + 777,
    )
    print(f"[dales] n_files={len(ds)} root={dales_root}")

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=minkowski_collate_dales,
        persistent_workers=int(data_cfg.get("num_workers", 4)) > 0,
    )

    # ---- mappings ----
    e2c = load_yaml_map(mapping_eclair_to_common)
    d2c = load_yaml_map(mapping_dales_to_common)
    e_train_to_common, d_native_to_common = build_label_lut_from_maps(e2c, d2c)

    e_train_to_common_t = torch.from_numpy(e_train_to_common).to(device=device)
    d_native_to_common_t = torch.from_numpy(d_native_to_common).to(device=device)

    cm = CommonConfusion(num_classes=8, ignore_id=0)

    amp = bool(cfg["run"].get("amp", True)) and device.type == "cuda"

    for file_i, batch in enumerate(loader, start=1):
        if file_i % 10 == 0 or file_i == len(ds):
            print(f"[dales] processed files {file_i}/{len(ds)}")
        # KEEP ON CPU: avoid pushing an entire DALES cloud to GPU at once.
        coords_cpu = batch["coords"]  # [N,4] int (b,x,y,z) on CPU
        feats_cpu = batch["feats"]  # [N,C] float on CPU
        gt_native_cpu = batch["labels"]  # [N] DALES native on CPU
        total_patches = 0
        # We still support batched coords via the batch column, but loader batch_size=1 => only b=0 typically.
        batch_ids = coords_cpu[:, 0].to(torch.int64)
        uniq_b = torch.unique(batch_ids)

        for b in uniq_b.tolist():
            m_b = batch_ids == int(b)
            idx_b = torch.nonzero(m_b, as_tuple=False).squeeze(1).to(torch.int64)

            coords_b = coords_cpu[idx_b]  # [Nb,4]
            feats_b = feats_cpu[idx_b]  # [Nb,C]
            gt_b = gt_native_cpu[idx_b]  # [Nb]

            # coords xyz (x,y,z) only, CPU
            coords_xyz_b = coords_b[:, 1:4].to(torch.int32)

            # Choose patch iterator
            if stride_vox == patch_size_vox:
                patch_idx_lists = _iter_patch_index_lists_nonoverlap(coords_xyz_b)
            else:
                patch_idx_lists = _iter_patch_index_lists_general(coords_xyz_b)

            if len(patch_idx_lists) == 0:
                # If nothing passes min points, fall back to using everything (may OOM, but avoids empty eval)
                patch_idx_lists = [torch.arange(coords_b.size(0), dtype=torch.int64)]

            for idx_p in patch_idx_lists:
                total_patches += 1
                # idx_p indexes into coords_b/feats_b/gt_b (CPU)
                coords_p = coords_b[idx_p].clone()  # [Np,4]
                feats_p = feats_b[idx_p]
                gt_native_p = gt_b[idx_p]

                # Re-base patch coords to start near 0 (matches training "local coords" behavior better)
                # shift x,y,z independently; keep batch column = 0
                coords_p[:, 0] = 0
                mins = coords_p[:, 1:4].min(dim=0).values
                coords_p[:, 1:4] = coords_p[:, 1:4] - mins

                # Move only this patch to GPU
                coords_gpu = coords_p.to(device=device, non_blocking=True)
                feats_gpu = feats_p.to(device=device, non_blocking=True)

                st = ME.SparseTensor(feats_gpu, coordinates=coords_gpu, device=device)

                with torch.autocast(device_type="cuda", enabled=amp):
                    out = model(st)
                    logits = out.F  # [Np, 11]
                    pred_train = logits.argmax(dim=1)  # 0..10

                # map to common
                pred_common = e_train_to_common_t[pred_train]  # [Np]
                gt_common = d_native_to_common_t[
                    gt_native_p.to(device=device, non_blocking=True).clamp(
                        0, d_native_to_common_t.numel() - 1
                    )
                ]

                cm.update(pred_common, gt_common)

                # reduce fragmentation in long eval loops
                del (
                    st,
                    out,
                    logits,
                    pred_train,
                    coords_gpu,
                    feats_gpu,
                    pred_common,
                    gt_common,
                )
                # if device.type == "cuda" and (patch_i % 20 == 0):
                #     torch.cuda.empty_cache()
            print(f"[dales] total_patches={total_patches}")

    iou, miou = cm.compute_iou()
    preproc_dict = asdict(preproc)
    for k in ("ref_quantiles", "ref_probs", "tgt_quantiles", "tgt_probs"):
        if isinstance(preproc_dict.get(k), np.ndarray):
            preproc_dict[k] = {
                "len": int(preproc_dict[k].shape[0]),
                "min": float(np.min(preproc_dict[k])),
                "max": float(np.max(preproc_dict[k])),
            }

    res = {
        "miou_common": miou,
        "iou_per_class_common": iou.tolist(),
        "common_class_names": [
            "ignore",
            "ground",
            "vegetation",
            "buildings",
            "wires",
            "poles",
            "fence",
            "vehicle",
        ],
        "preproc": preproc_dict,
    }

    (out_dir / "metrics_common.json").write_text(json.dumps(res, indent=2))
    return res


def compute_target_quantiles_from_dales(
    *,
    dales_root: str,
    bins: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build target quantiles (in [0,1]) for DALES intensity distribution by histogram scan.
    """
    from .data_dales import _find_dales_files, _read_dales_las

    files = _find_dales_files(dales_root)
    # pass-1: estimate robust max
    maxv = 0.0
    for p in files:
        raw = _read_dales_las(p)
        inten = raw["intensity"]
        if inten is None:
            continue
        inten = inten.astype(np.float32, copy=False)
        maxv = max(maxv, float(np.percentile(inten, 99.9)))

    # if already [0,1], divisor=1.0
    divisor = 1.0 if maxv <= 1.0 + 1e-3 else maxv
    hist = np.zeros((bins,), dtype=np.int64)
    for p in files:
        raw = _read_dales_las(p)
        inten = raw["intensity"]
        if inten is None:
            continue
        inten = inten.astype(np.float32, copy=False)
        print("[debug] DALES inten dtype:", inten.dtype)
        print(
            "[debug] DALES inten min/max:", float(np.min(inten)), float(np.max(inten))
        )
        print(
            "[debug] DALES inten p50/p99:",
            float(np.percentile(inten, 50)),
            float(np.percentile(inten, 99)),
        )
        intensity_scaled = np.clip(inten / float(divisor), 0.0, 1.0)
        print("[debug] scaled max:", float(np.max(intensity_scaled)))
        idx = np.minimum((intensity_scaled * (bins - 1)).astype(np.int64), bins - 1)
        hist += np.bincount(idx, minlength=bins)

    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= max(1.0, cdf[-1])

    probs = np.linspace(0.0, 1.0, 1001).astype(np.float32)
    q_bins = np.searchsorted(cdf, probs, side="left")
    q_bins = np.clip(q_bins, 0, bins - 1)
    quantiles = (q_bins / float(bins - 1)).astype(np.float32)

    return probs.astype(np.float32), quantiles, float(divisor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dales_root", required=True)
    ap.add_argument("--mapping_eclair_to_common", required=True)
    ap.add_argument("--mapping_dales_to_common", required=True)
    ap.add_argument("--out_dir", required=True)
    # DALES patching knobs (to prevent ME OOM)
    ap.add_argument("--dales_patch_size_m", type=float, default=100.0)
    ap.add_argument("--dales_patch_stride_m", type=float, default=100.0)
    ap.add_argument("--dales_min_patch_points", type=int, default=2000)
    ap.add_argument("--dales_max_patches_per_cloud", type=int, default=0)

    # preprocessing knobs
    ap.add_argument(
        "--run_both", action="store_true", help="Run baseline + preproc and compare"
    )
    ap.add_argument(
        "--intensity_ref_json",
        default=None,
        help="Output of compute_intensity_reference.py",
    )
    ap.add_argument(
        "--preproc_intensity_mode",
        default="quantile_match",
        choices=["none", "quantile_match", "robust_standardize", "constant"],
    )
    ap.add_argument("--use_height_xy_cap", action="store_true")
    ap.add_argument("--xy_cell_size_m", type=float, default=2.0)
    ap.add_argument("--height_bins", type=int, default=5)
    ap.add_argument(
        "--caps_per_bin", nargs="+", type=int, default=[4000, 3000, 2500, 2500, 2500]
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    def _load_ref():
        if args.intensity_ref_json is None:
            raise RuntimeError("quantile_match requires --intensity_ref_json")
        ref = json.loads(Path(args.intensity_ref_json).read_text())
        ref_p = np.asarray(ref["probs"], dtype=np.float32)
        ref_q = np.asarray(ref["quantiles"], dtype=np.float32)
        return ref_p, ref_q

    results = {}

    # 1) baseline
    baseline_cfg = DalesPreprocConfig(intensity_mode="none", use_height_xy_cap=False)
    results["baseline"] = run_eval(
        cfg=cfg,
        ckpt_path=args.ckpt,
        dales_root=args.dales_root,
        mapping_eclair_to_common=args.mapping_eclair_to_common,
        mapping_dales_to_common=args.mapping_dales_to_common,
        out_dir=out_root / "baseline",
        preproc=baseline_cfg,
        device=device,
        dales_patch_size_m=float(args.dales_patch_size_m),
        dales_patch_stride_m=float(args.dales_patch_stride_m),
        dales_min_patch_points=int(args.dales_min_patch_points),
        dales_max_patches_per_cloud=int(args.dales_max_patches_per_cloud),
    )

    if args.run_both:
        feat_cfg = FeatureConfig(**cfg["data"]["features"])
        intensity_div = float(feat_cfg.intensity_divisor)
        print("[dales] model intensity_divisor =", intensity_div)
        print(
            "[debug] use_intensity:",
            feat_cfg.use_intensity,
            "intensity_divisor:",
            feat_cfg.intensity_divisor,
        )

        pre = DalesPreprocConfig(
            intensity_mode=args.preproc_intensity_mode,
            use_height_xy_cap=bool(args.use_height_xy_cap),
            xy_cell_size_m=float(args.xy_cell_size_m),
            height_bins=int(args.height_bins),
            caps_per_bin=tuple(int(x) for x in args.caps_per_bin),
        )

        if pre.intensity_mode == "quantile_match":
            ref_p, ref_q = _load_ref()
            tgt_p, tgt_q, dales_div = compute_target_quantiles_from_dales(
                dales_root=args.dales_root
            )
            pre.ref_probs = ref_p
            pre.ref_quantiles = ref_q
            pre.tgt_probs = tgt_p
            pre.tgt_quantiles = tgt_q
            pre.intensity_divisor_override = dales_div
            print(
                "[dales] intensity_divisor_override =",
                dales_div,
                "tgt_q max =",
                float(tgt_q.max()),
            )

        results["preprocessed"] = run_eval(
            cfg=cfg,
            ckpt_path=args.ckpt,
            dales_root=args.dales_root,
            mapping_eclair_to_common=args.mapping_eclair_to_common,
            mapping_dales_to_common=args.mapping_dales_to_common,
            out_dir=out_root / "preprocessed",
            preproc=pre,
            device=device,
            dales_patch_size_m=float(args.dales_patch_size_m),
            dales_patch_stride_m=float(args.dales_patch_stride_m),
            dales_min_patch_points=int(args.dales_min_patch_points),
            dales_max_patches_per_cloud=int(args.dales_max_patches_per_cloud),
        )

        # comparison
        b = results["baseline"]
        p = results["preprocessed"]
        results["delta"] = {
            "miou_common": p["miou_common"] - b["miou_common"],
            "iou_per_class_common": (
                np.asarray(p["iou_per_class_common"])
                - np.asarray(b["iou_per_class_common"])
            ).tolist(),
            "note": "delta = preprocessed - baseline",
        }

    (out_root / "summary.json").write_text(json.dumps(results, indent=2))

    # nice stdout
    print("\n=== DALES Common-space Evaluation ===")
    print(f"Baseline mIoU: {results['baseline']['miou_common']:.4f}")
    if "preprocessed" in results:
        print(f"Preproc  mIoU: {results['preprocessed']['miou_common']:.4f}")
        print(f"Δ mIoU        : {results['delta']['miou_common']:+.4f}")
        # highlight wires/poles
        wires_idx, poles_idx = 4, 5
        bw, bp = (
            results["baseline"]["iou_per_class_common"][wires_idx],
            results["baseline"]["iou_per_class_common"][poles_idx],
        )
        pw, pp = (
            results["preprocessed"]["iou_per_class_common"][wires_idx],
            results["preprocessed"]["iou_per_class_common"][poles_idx],
        )
        print(f"Wires IoU: {bw:.4f} -> {pw:.4f} (Δ {pw-bw:+.4f})")
        print(f"Poles IoU: {bp:.4f} -> {pp:.4f} (Δ {pp-bp:+.4f})")
    print(f"Saved: {out_root / 'summary.json'}\n")


if __name__ == "__main__":
    main()
