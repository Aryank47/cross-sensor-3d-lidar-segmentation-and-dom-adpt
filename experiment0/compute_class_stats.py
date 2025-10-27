from pathlib import Path

import torch
import yaml
from datasets import EclairTiles
from tqdm import tqdm


def compute_class_statistics(eclair_dir: str, split: str = "train"):
    """
    Compute class frequencies on ECLAIR dataset.

    Returns:
        samples_per_cls: List[int] of length num_classes (12 for ECLAIR)
    """
    print(f"Computing class statistics for ECLAIR {split} split...")

    # Load dataset without transforms (to get raw labels)
    dataset = EclairTiles(root=eclair_dir, split=split, transforms=None)

    num_classes = 12  # ECLAIR native
    class_counts = torch.zeros(num_classes, dtype=torch.long)

    for data in tqdm(dataset, desc="Scanning tiles"):
        labels = data.classification
        for c in range(num_classes):
            class_counts[c] += (labels == c).sum()

    samples_per_cls = class_counts.tolist()

    # Print statistics
    print("\n" + "=" * 60)
    print("ECLAIR Class Distribution:")
    print("=" * 60)
    class_names = [
        "Undefined",
        "Unassigned",
        "Ground",
        "Vegetation",
        "Buildings",
        "Noise",
        "Transmission Wires",
        "Distribution Wires",
        "Poles",
        "Transmission Towers",
        "Fence",
        "Vehicles",
    ]

    total_points = sum(samples_per_cls)
    for c, (name, count) in enumerate(zip(class_names, samples_per_cls)):
        pct = 100.0 * count / total_points if total_points > 0 else 0.0
        print(f"Class {c:2d} ({name:20s}): {count:12,d} points ({pct:6.3f}%)")

    print("=" * 60)
    print(f"Total points: {total_points:,}")
    print("=" * 60 + "\n")

    return samples_per_cls


def save_class_stats(
    samples_per_cls, output_path: str = "./configs/eclair_class_counts.yaml"
):
    """Save class counts to YAML for use in training."""
    stats = {
        "samples_per_cls": samples_per_cls,
        "num_classes": len(samples_per_cls),
        "description": "Per-class sample counts for ECLAIR train split (for Class-Balanced Loss)",
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(stats, f)

    print(f"✓ Class statistics saved to: {output_path}")


if __name__ == "__main__":
    import fire

    def main(eclair_dir: str = "/data/eclair", split: str = "train"):
        samples_per_cls = compute_class_statistics(eclair_dir, split)
        save_class_stats(samples_per_cls)

        print("\nTo use Class-Balanced Loss, run:")
        print("  python train_baseline.py --loss_name class_balanced ...")

    fire.Fire(main)
