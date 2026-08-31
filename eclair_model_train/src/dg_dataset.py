from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .label_maps import ECLAIR_CLASS_NAMES_11


@dataclass(frozen=True)
class DGDatasetContract:
    name: str
    num_classes: int
    class_names: Tuple[str, ...]
    utility_class_ids: Tuple[int, ...]
    frozen_base_config: str
    full_epochs: int
    scheduler_step_epoch: int
    pilot_epochs: int
    pilot_resume_epoch: int
    final_test_eval_factor: float


_CONTRACTS: Dict[str, DGDatasetContract] = {
    "dales": DGDatasetContract(
        name="dales",
        num_classes=8,
        class_names=(
            "ground",
            "vegetation",
            "cars",
            "trucks",
            "buildings",
            "poles",
            "power_lines",
            "fences",
        ),
        utility_class_ids=(5, 6),
        frozen_base_config="configs/m0_dales_frozen_29364.yaml",
        full_epochs=200,
        scheduler_step_epoch=20,
        pilot_epochs=21,
        pilot_resume_epoch=2,
        final_test_eval_factor=4.0,
    ),
    "eclair": DGDatasetContract(
        name="eclair",
        num_classes=11,
        class_names=tuple(ECLAIR_CLASS_NAMES_11),
        utility_class_ids=(5, 6, 7, 8),
        frozen_base_config="configs/m0_eclair_frozen_29306.yaml",
        full_epochs=100,
        scheduler_step_epoch=10,
        pilot_epochs=21,
        pilot_resume_epoch=2,
        final_test_eval_factor=1.25,
    ),
}


def get_dg_dataset_contract(dataset: str) -> DGDatasetContract:
    key = str(dataset).strip().lower()
    if key not in _CONTRACTS:
        raise ValueError(f"Unsupported DG dataset {dataset!r}; expected one of {sorted(_CONTRACTS)}")
    return _CONTRACTS[key]
