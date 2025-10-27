import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import laspy
import numpy as np
import torch
from functional import compose_transforms_from_list
from torch_geometric.data import Data
from torch_geometric.transforms import Compose


class BaseLasDataset(torch.utils.data.Dataset):
    def __init__(self, transforms: Optional[Compose] = None):
        self.transforms = transforms

    @staticmethod
    def _las2pyg(las: laspy.LasData, path: Path) -> Data:
        gt_key = "classification" if "classification" in set(las.point_format.dimension_names) else "raw_classification"
        gt = las[gt_key]
        gt = getattr(gt, "array", gt)
        data = Data(
            xyz=torch.from_numpy(las.xyz.copy()),
            intensity=torch.from_numpy(las.intensity.astype(np.int64)),            
            classification = torch.from_numpy(np.asarray(gt).copy()).long(),
            return_number=torch.from_numpy(np.asarray(las.return_number)).long(),
            number_of_returns=torch.from_numpy(np.asarray(las.number_of_returns)).long(),
            edge_of_flight_line=torch.from_numpy(np.asarray(las.edge_of_flight_line)),
            instance_id=(
                torch.from_numpy(np.asarray(las.instance).copy().astype(np.int64)).long()
                if hasattr(las, "instance")
                else torch.full((len(las.return_number),), fill_value=-1, dtype=torch.long)
            ),
            rgb=torch.stack(
                [
                    torch.from_numpy(las.red.astype(np.int64)),
                    torch.from_numpy(las.green.astype(np.int64)),
                    torch.from_numpy(las.blue.astype(np.int64)),
                ],
                dim=-1,
            ).long()
            if hasattr(las, "red")
            else None,
            filename=str(path),
        )
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
            self.root / "pointclouds" / rec["tile_name"] for rec in all_tiles if rec.get("split", "") == split
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
    """Scans a folder recursively for .las/.laz files. Use for DALES test if a JSON split isn't available.
    """

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
        las = laspy.read(p)
        data = self._las2pyg(las, p)
        if self.transforms:
            data = self.transforms(data)
        return data

    def __len__(self) -> int:
        return len(self.files)


