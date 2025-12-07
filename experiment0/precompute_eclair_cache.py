# precompute_eclair_cache.py
from pathlib import Path

import torch
from datasets import EclairTiles
from functional import compose_transforms_from_list
from omegaconf import OmegaConf
from tqdm import tqdm


def precompute_eclair(
    eclair_dir: str,
    config_file: str = "./configs/train_e0.yaml",
    split: str = "train",
    out_root: str | None = None,
):
    """
    Precompute transformed ECLAIR tiles and save as .pt files.

    This applies the same transforms you use in training/eval
    (cfg.train_transforms / cfg.eval_transforms), but only once.
    """
    cfg = OmegaConf.load(config_file)

    if split == "train":
        transforms_cfg = cfg.train_transforms
    else:
        transforms_cfg = cfg.eval_transforms

    transforms = compose_transforms_from_list(transforms_cfg)

    dataset = EclairTiles(root=eclair_dir, split=split, transforms=transforms)

    if out_root is None:
        out_root = Path(eclair_dir) / "cache"
    else:
        out_root = Path(out_root)

    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Precomputing {split} → {out_dir}")
    for idx in tqdm(range(len(dataset)), desc=f"Precompute {split}"):
        data = dataset[idx]

        # Use a stable index-based name to preserve ordering exactly
        tile_id = f"{idx:06d}"
        torch.save(data, out_dir / f"{tile_id}.pt")


if __name__ == "__main__":
    import fire

    fire.Fire(precompute_eclair)
