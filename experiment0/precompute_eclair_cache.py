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

    errors = []
    n = len(dataset)

    print(f"[precompute] {split}: {n} tiles", flush=True)

    for idx in tqdm(range(n), desc=f"Precompute {split}"):
        try:
            data = dataset[idx]

            # Choose a stable tile_id
            if hasattr(data, "filename"):
                tile_id = Path(str(data.filename)).stem
            else:
                tile_id = f"{idx:04d}"

            out_path = out_dir / f"{tile_id}.pt"
            print(
                f"[precompute] {split} idx={idx} tile_id={tile_id} -> {out_path}",
                flush=True,
            )

            torch.save(data, out_path)

        except Exception as e:
            msg = f"[precompute][ERROR] split={split} idx={idx} error={type(e).__name__}: {e}"
            print(msg, file=sys.stderr, flush=True)
            errors.append(msg)
            # optional: break here if you don't want to continue
            break

    if errors:
        err_log = out_root / f"precompute_{split}_errors.log"
        with open(err_log, "w") as f:
            for line in errors:
                f.write(line + "\n")
        # Fail the job so you notice
        raise RuntimeError(
            f"Precompute {split} failed with {len(errors)} errors. See {err_log}"
        )


if __name__ == "__main__":
    import fire

    fire.Fire(precompute_eclair)
