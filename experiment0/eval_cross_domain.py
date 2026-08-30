import torch
from datasets import EclairTiles, GenericLasFolder
from functional import compose_transforms_from_list
from model import MinkUNet14C
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader as PyGDataLoader
from train_baseline import evaluate_split


def eval_cross_domain(
    eclair_dir: str,
    dales_dir: str,
    checkpoint: str,
    config_file: str = "./configs/train_e0.yaml",
):
    cfg = OmegaConf.load(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # datasets with eval transforms
    eclair_val = EclairTiles(
        root=eclair_dir,
        split="val",
        transforms=compose_transforms_from_list(cfg.eval_transforms),
    )
    dales_test = GenericLasFolder(
        root=dales_dir,
        transforms=compose_transforms_from_list(cfg.eval_transforms),
    )

    val_loader = PyGDataLoader(
        eclair_val,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    dales_loader = PyGDataLoader(
        dales_test,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # model
    model = MinkUNet14C(cfg.num_features, cfg.num_classes_native).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # mappings
    eclair2common = OmegaConf.to_container(
        OmegaConf.load(cfg.mapping_eclair_to_common), resolve=True
    )
    dales2common = OmegaConf.to_container(
        OmegaConf.load(cfg.mapping_dales_to_common), resolve=True
    )
    num_common = cfg.num_classes_common

    # ECLAIR
    val_results = evaluate_split(
        model,
        val_loader,
        device,
        cfg.voxel_size,
        cfg.num_classes_native,
        cfg.ignore_ids_native,
        pred_common_mapping=eclair2common,
        gt_common_mapping=eclair2common,
        num_common=num_common,
        important_native=cfg.important_native,
    )

    # DALES
    dales_results = evaluate_split(
        model,
        dales_loader,
        device,
        cfg.voxel_size,
        cfg.num_classes_dales_native,
        ignore_ids_native=cfg.ignore_ids_dales,
        pred_common_mapping=eclair2common,
        gt_common_mapping=dales2common,
        num_common=num_common,
        important_native=list(range(cfg.num_classes_dales_native)),
    )

    delta_miou = val_results["common"]["mIoU"] - dales_results["common"]["mIoU"]

    print("ECLAIR common mIoU:", val_results["common"]["mIoU"])
    print("DALES  common mIoU:", dales_results["common"]["mIoU"])
    print("ΔmIoU (ECLAIR→DALES):", delta_miou)


if __name__ == "__main__":
    import fire

    fire.Fire(eval_cross_domain)
