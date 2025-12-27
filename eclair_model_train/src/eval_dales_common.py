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

    # DALES loader
    ds = DalesTiles(
        dales_root=dales_root,
        patch_cfg=patch_cfg,
        feat_cfg=feat_cfg,
        ignore_index=int(data_cfg["label_space"]["ignore_index"]),
        preproc=preproc,
        seed=int(cfg["run"]["seed"]) + 777,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 1)),
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

    for batch in loader:
        coords = batch["coords"].to(device, non_blocking=True)
        feats = batch["feats"].to(device, non_blocking=True)
        gt_native = batch["labels"].to(device, non_blocking=True)  # DALES native

        st = ME.SparseTensor(feats, coordinates=coords, device=device)

        with torch.autocast(device_type="cuda", enabled=amp):
            out = model(st)
            logits = out.F  # [N, 11]
            pred_train = logits.argmax(dim=1)  # 0..10

        # map to common
        pred_common = e_train_to_common_t[pred_train]  # [N]
        gt_common = d_native_to_common_t[
            gt_native.clamp(0, d_native_to_common_t.numel() - 1)
        ]

        cm.update(pred_common, gt_common)

    iou, miou = cm.compute_iou()

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
        "preproc": asdict(preproc),
    }

    (out_dir / "metrics_common.json").write_text(json.dumps(res, indent=2))
    return res


def compute_target_quantiles_from_dales(
    *,
    dales_root: str,
    intensity_divisor: float,
    bins: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build target quantiles (in [0,1]) for DALES intensity distribution by histogram scan.
    """
    from src.data_dales import _find_dales_files, _read_dales_las

    files = _find_dales_files(dales_root)
    hist = np.zeros((bins,), dtype=np.int64)

    for p in files:
        raw = _read_dales_las(p)
        inten = raw["intensity"]
        if inten is None:
            continue
        x01 = (inten / float(intensity_divisor)).astype(np.float32, copy=False)
        x01 = np.clip(x01, 0.0, 1.0)
        idx = np.minimum((x01 * (bins - 1)).astype(np.int64), bins - 1)
        hist += np.bincount(idx, minlength=bins)

    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= max(1.0, cdf[-1])

    probs = np.linspace(0.0, 1.0, 1001)
    q_bins = np.searchsorted(cdf, probs, side="left")
    q_bins = np.clip(q_bins, 0, bins - 1)
    quantiles = (q_bins / float(bins - 1)).astype(np.float32)
    return probs.astype(np.float32), quantiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dales_root", required=True)
    ap.add_argument("--mapping_eclair_to_common", required=True)
    ap.add_argument("--mapping_dales_to_common", required=True)
    ap.add_argument("--out_dir", required=True)

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
    )

    if args.run_both:
        feat_cfg = FeatureConfig(**cfg["data"]["features"])
        intensity_div = float(feat_cfg.intensity_divisor)

        pre = DalesPreprocConfig(
            intensity_mode=args.preproc_intensity_mode,
            use_height_xy_cap=bool(args.use_height_xy_cap),
            xy_cell_size_m=float(args.xy_cell_size_m),
            height_bins=int(args.height_bins),
            caps_per_bin=tuple(int(x) for x in args.caps_per_bin),
        )

        if pre.intensity_mode == "quantile_match":
            ref_p, ref_q = _load_ref()
            tgt_p, tgt_q = compute_target_quantiles_from_dales(
                dales_root=args.dales_root, intensity_divisor=intensity_div
            )
            pre.ref_probs = ref_p
            pre.ref_quantiles = ref_q
            pre.tgt_probs = tgt_p
            pre.tgt_quantiles = tgt_q

        results["preprocessed"] = run_eval(
            cfg=cfg,
            ckpt_path=args.ckpt,
            dales_root=args.dales_root,
            mapping_eclair_to_common=args.mapping_eclair_to_common,
            mapping_dales_to_common=args.mapping_dales_to_common,
            out_dir=out_root / "preprocessed",
            preproc=pre,
            device=device,
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
