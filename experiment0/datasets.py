import json
from pathlib import Path
from typing import Dict, List, Optional

import laspy
import numpy as np
import torch
# from functional import compose_transforms_from_list
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.transforms import Compose


def read_las_arrays_robust(path: Path) -> Dict[str, np.ndarray]:
    """
    Reads LAS/LAZ files using robust property-access to avoid bit-packing bugs.
    Standardizes output keys to: xyz, intensity, return_number, number_of_returns, rgb, native_labels.
    """
    try:
        las = laspy.read(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to read LAS file {path}: {e}")

    # Standardize XYZ to float32
    xyz = np.array(las.xyz, dtype=np.float64)

    def _get_dim(
        attr_name: str, fallback_names: List[str] = None
    ) -> Optional[np.ndarray]:
        # Priority 1: Direct property access (handles bit-unpacking/scaling)
        if hasattr(las, attr_name):
            val = getattr(las, attr_name)
            return np.array(val)

        # Priority 2: Dictionary access (fallback for non-standard names)
        # Check standard dimension names case-insensitively
        dims_lower = set(d.lower() for d in las.point_format.dimension_names)

        if attr_name.lower() in dims_lower:
            return np.array(las[attr_name])

        if fallback_names:
            for name in fallback_names:
                if name.lower() in dims_lower:
                    return np.array(las[name])
        return None

    # Intensity
    intensity = _get_dim("intensity")
    if intensity is None:
        intensity = np.zeros((xyz.shape[0],), dtype=np.float32)
    else:
        intensity = intensity.astype(np.float32)

    # Returns (CRITICAL FIX: Use property access)
    rn = _get_dim("return_number")
    nor = _get_dim("number_of_returns")

    if rn is None:
        rn = np.ones((xyz.shape[0],), dtype=np.int64)
    else:
        rn = rn.astype(np.int64)

    if nor is None:
        nor = np.ones((xyz.shape[0],), dtype=np.int64)
    else:
        nor = nor.astype(np.int64)

    # Labels
    labels = _get_dim("classification", fallback_names=["raw_classification"])
    if labels is None:
        # Fallback for datasets that might be unlabeled
        # print(f" [WARN] No classification found in {path.name}, using zeros.") # Optional logging
        labels = np.zeros((xyz.shape[0],), dtype=np.int64)
    else:
        labels = labels.astype(np.int64)

    # RGB
    rgb = None
    red = _get_dim("red")
    green = _get_dim("green")
    blue = _get_dim("blue")

    if red is not None and green is not None and blue is not None:
        max_val = max(red.max(), green.max(), blue.max())
        scale = 1.0
        if max_val > 255:
            scale = 1.0 / 65535.0
        r = red.astype(np.float32) * scale
        g = green.astype(np.float32) * scale
        b = blue.astype(np.float32) * scale
        rgb = np.stack([r, g, b], axis=1)

    return {
        "xyz": xyz,
        "intensity": intensity,
        "return_number": rn,
        "number_of_returns": nor,
        "rgb": rgb,
        "native_labels": labels,
    }


class CachedEclairTiles(Dataset):
    """
    Loads pre-transformed ECLAIR tiles from disk (.pt files saved by precompute_eclair).

    Returns torch_geometric.data.Data objects ready to be batched by PyGDataLoader.
    No transforms are applied here.
    """

    def __init__(self, cache_root: str, split: str = "train"):
        self.root = Path(cache_root) / split
        self.files = sorted(self.root.glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No cached tiles found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        return torch.load(self.files[idx])


class BaseLasDataset(torch.utils.data.Dataset):
    def __init__(self, transforms: Optional[Compose] = None):
        self.transforms = transforms

    # @staticmethod
    # def _las2pyg(las: laspy.LasData, path: Path) -> Data:
    #     gt_key = (
    #         "classification"
    #         if "classification" in set(las.point_format.dimension_names)
    #         else "raw_classification"
    #     )
    #     gt = las[gt_key]
    #     gt = getattr(gt, "array", gt)
    #     data = Data(
    #         xyz=torch.from_numpy(las.xyz.copy()),
    #         intensity=torch.from_numpy(las.intensity.astype(np.int64)),
    #         classification=torch.from_numpy(np.asarray(gt).copy()).long(),
    #         return_number=torch.from_numpy(np.asarray(las.return_number)).long(),
    #         number_of_returns=torch.from_numpy(
    #             np.asarray(las.number_of_returns)
    #         ).long(),
    #         edge_of_flight_line=torch.from_numpy(np.asarray(las.edge_of_flight_line)),
    #         instance_id=(
    #             torch.from_numpy(
    #                 np.asarray(las.instance).copy().astype(np.int64)
    #             ).long()
    #             if hasattr(las, "instance")
    #             else torch.full(
    #                 (len(las.return_number),), fill_value=-1, dtype=torch.long
    #             )
    #         ),
    #         rgb=(
    #             torch.stack(
    #                 [
    #                     torch.from_numpy(las.red.astype(np.int64)),
    #                     torch.from_numpy(las.green.astype(np.int64)),
    #                     torch.from_numpy(las.blue.astype(np.int64)),
    #                 ],
    #                 dim=-1,
    #             ).long()
    #             if hasattr(las, "red")
    #             else None
    #         ),
    #         filename=str(path),
    #     )
    #     return data
    @staticmethod
    def _las2pyg(path: Path) -> Data:
        """
        Robustly reads LAS file using the shared utility and converts to PyG Data.
        Ensures Float64 precision for coordinates and proper decoding of labels.
        """
        # 1. Use the Robust Reader (Handles bit-packing & safe defaults)
        raw_dict = read_las_arrays_robust(path)

        # 2. Handle Instance ID (Specific to this dataset class)
        # We read this manually here because the generic robust reader might not extract it.
        # We use the safe property access pattern.
        las = laspy.read(str(path))
        try:
            if hasattr(las, "instance"):
                instance_id = np.array(las.instance, dtype=np.int64)
            else:
                # Fallback: check if it's in dimensions but not as a property
                # (unlikely for 'instance' but good for safety)
                dims = set(d.lower() for d in las.point_format.dimension_names)
                if "instance" in dims:
                    instance_id = np.array(las["instance"], dtype=np.int64)
                else:
                    instance_id = None
        except Exception:
            # If re-reading fails (unlikely since robust reader passed), default to None
            instance_id = None

        # 3. Convert to Torch Tensors

        # XYZ: Keep as Double (Float64) to preserve precision if coordinates are global
        # transforms (like Center) can handle Double and cast to Float later.
        pos = torch.from_numpy(raw_dict["xyz"].astype(np.float64))

        # Intensity: Float32 (from reader)
        intensity = torch.from_numpy(raw_dict["intensity"])

        # Classification: Long
        classification = torch.from_numpy(raw_dict["native_labels"]).long()

        # Returns: Long
        return_number = torch.from_numpy(raw_dict["return_number"]).long()
        number_of_returns = torch.from_numpy(raw_dict["number_of_returns"]).long()

        # RGB: Reader returns float32 [0-1] or None
        rgb = None
        if raw_dict["rgb"] is not None:
            rgb = torch.from_numpy(raw_dict["rgb"])

        # Instance ID handling
        if instance_id is not None:
            instance_t = torch.from_numpy(instance_id).long()
        else:
            # Fallback for backward compatibility: fill with -1
            instance_t = torch.full((len(pos),), -1, dtype=torch.long)

        # Create Data object
        # Note: PyG convention uses 'pos' for coordinates, but we also keep 'xyz' alias
        # if your downstream code expects it.
        data = Data(
            pos=pos,
            xyz=pos,
            intensity=intensity,
            classification=classification,
            return_number=return_number,
            number_of_returns=number_of_returns,
            rgb=rgb,
            instance_id=instance_t,
            filename=str(path),
        )

        # Add 'edge_of_flight_line' if needed for full compatibility
        # (Your old code had this; adding it back just in case)
        if hasattr(las, "edge_of_flight_line"):
            data.edge_of_flight_line = torch.from_numpy(
                np.array(las.edge_of_flight_line)
            ).long()
        else:
            data.edge_of_flight_line = torch.zeros((len(pos),), dtype=torch.long)

        return data

    def __getitem__(self, idx: int) -> Data:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class EclairTiles(BaseLasDataset):
    """Reads ECLAIR tiles using labels.json and a split (train/val/test)."""

    def __init__(self, root: str, split: str, transforms):
        super().__init__(transforms)
        self.root = Path(root)
        self.split = split
        label_file = self.root / "labels.json"
        with open(label_file, "r", encoding="utf-8") as f:
            all_tiles: List[Dict] = json.load(f)
        self.paths: List[Path] = [
            self.root / "pointclouds" / rec["tile_name"]
            for rec in all_tiles
            if rec.get("split", "") == split
        ]

    def __getitem__(self, idx: int) -> Data:
        p = self.paths[idx]
        las = laspy.read(p)
        data = self._las2pyg(las, p)
        if self.transforms:
            data = self.transforms(data)
        return data

    def __len__(self) -> int:
        return len(self.paths)


class GenericLasFolder(BaseLasDataset):
    """Scans a folder recursively for .las/.laz files. Use for DALES test if a JSON split isn't available."""

    def __init__(self, root: str, transforms):
        super().__init__(transforms)
        self.files = []
        for ext in ["*.las", "*.laz"]:
            self.files += list(Path(root).rglob(ext))
        self.files = sorted(self.files)
        if len(self.files) == 0:
            raise FileNotFoundError(f"No LAS/LAZ files found under {root}")

    def __getitem__(self, idx: int) -> Data:
        p = self.files[idx]
        # las = laspy.read(p)
        data = self._las2pyg(p)
        if self.transforms:
            data = self.transforms(data)
        return data

    def __len__(self) -> int:
        return len(self.files)
