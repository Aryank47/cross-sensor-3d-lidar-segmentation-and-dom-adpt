# from pathlib import Path

# import MinkowskiEngine as ME
# import torch
# from torch.utils.data import DataLoader

# from .augment import AugmentConfig
# from .data_dales import DalesPatchConfig, DalesPreprocConfig, DalesTiles, _find_dales_files, minkowski_collate_dales
# from .features import FeatureConfig
# from .metrics import ConfusionMatrix
# from .model import build_model


# def load_ckpt(ckpt_path: str, device="cuda"):
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     print("[ckpt keys]", ckpt.keys())
#     print("[ckpt epoch]", ckpt.get("epoch", None))
#     print("[ckpt cfg model]", ckpt.get("cfg", {}).get("model", None))
#     return ckpt


# @torch.no_grad()
# def eval_train_space(model, loader, num_classes=8, ignore_index=-100, device="cuda", amp=True):
#     model.eval()
#     cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
#     for batch in loader:
#         coords = batch["coords"].to(device)
#         feats = batch["feats"].to(device)
#         labels = batch["labels"].to(device)
#         st = __import__("MinkowskiEngine").SparseTensor(feats, coordinates=coords, device=device)
#         with torch.autocast(device_type="cuda", enabled=amp):
#             logits = model(st).F
#         preds = logits.argmax(1)
#         cm.update(preds, labels)
#     return cm.compute()


# def build_lut_train_to_common():
#     # train_id_to_common_dales.yaml:
#     # 0->1, 1->2, 2->7, 3->7, 4->3, 5->5, 6->4, 7->6
#     lut = torch.tensor([1, 2, 7, 7, 3, 5, 4, 6], dtype=torch.long)
#     return lut


# def map_train_to_common(y_train: torch.Tensor, lut: torch.Tensor, ignore_index=-100, common_ignore=0):
#     # y_train: (N,) in {-100,0..7}
#     y_common = torch.full_like(y_train, common_ignore)
#     valid = y_train != ignore_index
#     y_common[valid] = lut[y_train[valid]]
#     return y_common


# @torch.no_grad()
# def eval_common_space(model, loader, lut, device="cuda", amp=True):
#     # common IDs: 0..7 where 0 is ignore
#     cm = ConfusionMatrix(num_classes=8, ignore_index=0)
#     seen = 0
#     for batch in loader:
#         coords = batch["coords"].to(device)
#         feats = batch["feats"].to(device)
#         y_tr = batch["labels"].to(device)
#         st = ME.SparseTensor(feats, coordinates=coords, device=device)
#         with torch.autocast(device_type="cuda", enabled=amp):
#             logits = model(st).F
#         print("logits shape:", tuple(logits.shape))  # (N, C)
#         print("pred max:", int(logits.argmax(1).max()))
#         p_tr = logits.argmax(1)

#         y_cm = map_train_to_common(y_tr, lut, ignore_index=-100, common_ignore=0)
#         p_cm = map_train_to_common(p_tr, lut, ignore_index=-100, common_ignore=0)  # preds never -100, but ok

#         cm.update(p_cm, y_cm)
#         seen += 1
#     print("num batches seen:", seen, "expected:", len(loader))
#     return cm.compute()


# def main(cfg_path: str, ckpt_path: str, split="test", limit_files=None):
#     ckpt = load_ckpt(ckpt_path)
#     cfg = ckpt["cfg"]  # <-- IMPORTANT: use training-time cfg snapshot

#     data = cfg["data"]
#     run = cfg["run"]
#     print("[test_dales_train_time_eval] ckpt['cfg']['data']['dales_test_root']:", data["dales_test_root"])

#     feat_cfg = FeatureConfig(**data["features"])
#     patch_cfg = DalesPatchConfig(**data["patch"])
#     preproc_cfg = DalesPreprocConfig(**data.get("preproc", {}))
#     ignore_index = int(data["label_space"]["ignore_index"])
#     label_map = {int(k): int(v) for k, v in data["dales_label_map_native_to_train"].items()}

#     root = Path(data["dales_test_root"] if split == "test" else data["dales_train_root"])
#     files = _find_dales_files(root)
#     if limit_files is not None:
#         files = files[:limit_files]

#     # Match train.py split seeds exactly:
#     split_seed = int(run["seed"]) + (2 if split == "test" else 1 if split == "val" else 0)

#     ds = DalesTiles(
#         dales_root=root,
#         files=files,
#         patch_cfg=patch_cfg,
#         feat_cfg=feat_cfg,
#         is_train=False,
#         aug_cfg=AugmentConfig(enabled=False),
#         ignore_index=ignore_index,
#         preproc=preproc_cfg,
#         seed=split_seed,  # <-- IMPORTANT
#         use_cache=bool(data.get("use_cache", True)),
#         cache_root=data.get("cache_root", None),
#         cache_subdir=str(data.get("cache_subdir", "dales_dropI")),
#         cache_key_extra=data.get("cache_key_extra", None),
#         require_cache=True,  # <-- IMPORTANT: fail fast if not cached
#         write_cache=False,
#         split_name=split,
#         label_map=label_map,
#     )
#     print("cache_dir:", ds._cache_dir)
#     if ds._cache_dir is not None:
#         print("cached files:", len(list(ds._cache_dir.glob("*.pt"))))

#     dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=minkowski_collate_dales)

#     # Build model from ckpt cfg
#     mcfg = cfg["model"]
#     model = build_model(in_channels=int(mcfg["in_channels"]), out_channels=int(mcfg["out_channels"]), D=int(mcfg.get("D", 3)))
#     model.load_state_dict(ckpt["model_state"], strict=True)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device).eval()

#     # Evaluate (train space)
#     cm = ConfusionMatrix(num_classes=int(data["label_space"]["num_classes"]), ignore_index=ignore_index)
#     with torch.no_grad():
#         for batch in dl:
#             coords = batch["coords"].to(device)
#             feats = batch["feats"].to(device)
#             y = batch["labels"].to(device)
#             st = ME.SparseTensor(feats, coordinates=coords, device=device)
#             logits = model(st).F
#             p = logits.argmax(1)
#             cm.update(p, y)

#     res = cm.compute()
#     print("[PARITY TRAIN SPACE] miou:", res.miou, "macroF1:", res.macro_f1)


# # Example:
# main(
#     "/csehome/m23csa510/lidar_experiments/cross-sensor-3d-lidar-segmentation-and-dom-adpt/eclair_model_train/configs/e0_dales_dropI.yaml",
#     "/scratch/m23csa510/e0_results/e0_dales_train_24930/checkpoints/best.pt",
#     split="test",
#     limit_files=11,
# )


# import pprint

# import numpy as np
# import torch
# import yaml

# from .features import build_features
# from .utils import read_las_arrays_robust

# ckpt_path = "/scratch/m23csa510/e0_results/e0_dales_train_24930/checkpoints/best.pt"
# cfg_map_path = "/csehome/m23csa510/lidar_experiments/cross-sensor-3d-lidar-segmentation-and-dom-adpt/eclair_model_train/configs/e0_dales_dropI.yaml"

# ckpt = torch.load(ckpt_path, map_location="cpu")

# print("\n=== CKPT keys ===")
# print(ckpt.keys())

# # Try common places where we store metadata
# meta = ckpt.get("meta", {})
# print("\n=== CKPT meta keys ===")
# print(meta.keys() if isinstance(meta, dict) else type(meta))

# # These names are guesses; print what exists in your ckpt
# candidates = [
#     ("meta.label_map_native_to_train", meta.get("label_map_native_to_train") if isinstance(meta, dict) else None),
#     ("meta.dales_label_map_native_to_train", meta.get("dales_label_map_native_to_train") if isinstance(meta, dict) else None),
#     ("label_map_native_to_train", ckpt.get("label_map_native_to_train")),
#     ("dales_label_map_native_to_train", ckpt.get("dales_label_map_native_to_train")),
#     ("label_space", ckpt.get("label_space")),
# ]

# print("\n=== CKPT mapping candidates ===")
# for k, v in candidates:
#     if v is not None:
#         print(f"\n{k}:")
#         pprint.pp(v)

# print("\n=== CKPT class names (if present) ===")
# ls = ckpt.get("label_space") or meta.get("label_space") if isinstance(meta, dict) else None
# if isinstance(ls, dict) and "class_names" in ls:
#     print(ls["class_names"])
# elif hasattr(ls, "class_names"):
#     print(ls.class_names)

# # Load cfg mapping (depends on your yaml structure)
# with open(cfg_map_path, "r") as f:
#     y = yaml.safe_load(f)

# print("\n=== CFG top-level keys ===", y.keys())
# # You may need to navigate to the exact dict path; print to locate it
# pprint.pp(y)


# p = Path("/scratch/m23csa510/dales/dales/all/5080_54400.las")

# arr = read_las_arrays_robust(p)

# rn = arr["return_number"]
# nor = arr["number_of_returns"]
# y = arr["native_labels"]

# print("[rn] unique:", np.unique(rn)[:20], "min/max:", rn.min(), rn.max())
# print("[nor] unique:", np.unique(nor)[:20], "min/max:", nor.min(), nor.max())
# print("[y] unique:", np.unique(y)[:20])

# cfg = FeatureConfig(
#     use_intensity=False,
#     intensity_divisor=65535.0,
#     returns_onehot_k=5,
#     use_rgb=False,
#     include_coords=False,
# )

# # Build a small sample to keep it fast
# idx = np.random.RandomState(0).choice(len(rn), size=200000, replace=False)
# xyz_s = arr["xyz"][idx]
# rn_s = rn[idx]
# nor_s = nor[idx]

# feat = build_features(
#     xyz_local=xyz_s, return_number=rn_s, number_of_returns=nor_s, cfg=cfg, intensity=None, rgb=None
# )  # adjust args to your signature
# feat = np.asarray(feat)

# print("feat shape:", feat.shape)
# print("feat mean/std:", feat.mean(), feat.std())
# print("num unique feature rows (sample):", np.unique(feat, axis=0).shape[0])
# print("fraction all-zero rows:", np.mean(np.all(feat == 0, axis=1)))


# p1 = [
#     Path("/scratch/m23csa510/dales/dales/all/5080_54400.las"),
#     Path("/scratch/m23csa510/dales/dales/all/5150_54325.las"),
#     Path("/scratch/m23csa510/dales/dales/all/5100_54495.las"),
#     Path("/scratch/m23csa510/dales/dales/all/5110_54495.las"),
#     Path("/scratch/m23csa510/dales/dales/all/5145_54340.las"),
#     Path("/scratch/m23csa510/dales/dales/all/5190_54400.las"),
# ]

# for p in p1:
#     arr = read_las_arrays_robust(p)
#     k = 5
#     rn = arr["return_number"]
#     nor = arr["number_of_returns"]

#     frac_rn = np.mean(rn > k)
#     frac_nor = np.mean(nor > k)
#     frac_any = np.mean((rn > k) | (nor > k))

#     print("frac(rn>k):", frac_rn)
#     print("frac(nor>k):", frac_nor)
#     print("frac(any>k):", frac_any)

#     # optional: how many points become indistinguishable due to clipping?
#     rn_clip = np.minimum(rn, k)
#     nor_clip = np.minimum(nor, k)
#     print("unique (rn,nor) before:", len(set(zip(rn, nor))))
#     print("unique (rn,nor) after :", len(set(zip(rn_clip, nor_clip))))
