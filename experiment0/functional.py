from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import Compose


def compose_transforms_from_list(input_transforms: List[str]) -> Compose:
    available_transforms = {
        "NormalizeCoordinates": NormalizeCoordinates,
        "NormalizeFeatures": NormalizeFeatures,
        "RemapClassification": RemapClassification,
    }

    transforms = []
    for transform in input_transforms:
        if transform["name"] in available_transforms:
            transform_fn = available_transforms[transform["name"]]
            transforms.append(transform_fn(**transform.get("params", {})))
        else:
            raise ValueError(
                f"Transform {transform['name']} not found. Available transforms are: {list(available_transforms.keys())}"
            )
    return Compose(transforms=transforms)


class BaseTransform:
    def __call__(self, data: Data) -> Data:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class NormalizeCoordinates(BaseTransform):
    def __init__(self, normalization_value=10.0):
        super().__init__()
        self.normalization_value = normalization_value

    def __call__(self, data: Data) -> Data:
        coordinates = getattr(data, "xyz", None)
        if coordinates is None:
            coordinates = getattr(data, "pos", None)

        if coordinates is None:
            raise AttributeError(
                "NormalizeCoordinates expected 'xyz' or 'pos' on Data; "
                f"found keys: {list(data.keys())}"
            )

        centroid = coordinates.mean(0)
        coordinates = (coordinates - centroid) / self.normalization_value
        data.pos = coordinates.float()

        # Optional: only delete xyz if present
        if hasattr(data, "xyz"):
            del data.xyz

        return data


class NormalizeFeatures(BaseTransform):
    def __init__(self, feature_names):
        super().__init__()
        self.feature_names = feature_names

    def __call__(self, data: Data) -> Data:
        features = []
        if "intensity" in self.feature_names:
            intensity = (data.intensity.float() / np.iinfo("uint16").max - 0.5).reshape(
                -1, 1
            )
            features.append(intensity)
        if "return_number" in self.feature_names:
            n_classes = 5
            rn = (data.return_number - 1).clamp(0, n_classes - 1).long()
            rn = F.one_hot(rn, num_classes=n_classes).float()
            features.append(rn)
        if "number_of_returns" in self.feature_names:
            n_classes = 5
            nor = (data.number_of_returns - 1).clamp(0, n_classes - 1).long()
            nor = F.one_hot(nor, num_classes=n_classes).float()
            features.append(nor)
        if "coordinates" in self.feature_names:
            coordinates = data.pos.float()
            features.append(coordinates)
        # if "colors" in self.feature_names:
        #     rgb = (data.rgb / np.iinfo("uint16").max - 0.5).reshape(-1, 3)
        #     features.append(rgb)
        if "colors" in self.feature_names:
            if getattr(data, "rgb", None) is not None:
                rgb = data.rgb.float() / np.iinfo("uint16").max - 0.5
                rgb = rgb.view(-1, 3)
            else:
                rgb = torch.zeros((data.pos.shape[0], 3), dtype=torch.float32)
            features.append(rgb)

        if not features:
            raise ValueError("NormalizeFeatures was called with empty feature list.")

        data.x = torch.cat(features, dim=-1).float()
        return data


class RemapClassification(BaseTransform):
    def __init__(self, class_mapping):
        super().__init__()
        self.class_mapping = class_mapping

    def __call__(self, data: Data) -> Data:
        if self.class_mapping:
            new_y = torch.zeros_like(data.classification)
            for map_from, map_to in self.class_mapping.items():
                new_y.masked_fill_(data.classification == map_from, map_to)
            data.classification = new_y
            data.y = new_y
        return data
