"""Finite pilot contract, without importing the GPU runtime."""

from __future__ import annotations

import hashlib

ARMS = {
    "reference": (1, 1, 1, 1, 1, 1, 1, 1),
    "context": (1, 1, 2, 1, 1, 1, 1, 1),
    "fine": (1, 1, 1, 1, 1, 1, 1, 2),
}


def block_counts(cfg):
    values = (cfg or {}).get("model", {}).get("layers", ARMS["reference"])
    if not isinstance(values, (list, tuple)) or len(values) != 8:
        raise ValueError("model.layers must contain eight positive integers")
    if any(type(v) is not int or v < 1 for v in values):
        raise ValueError("model.layers must contain eight positive integers")
    return tuple(values)


def initialize_matched(model, cfg):
    """Keep all shared weights identical and both extra blocks identically initialized.

    Build a legacy reference under a forked RNG, then copy its matching tensors.
    Extra block names are canonicalized so context/fine additions share initial
    tensors. Never reset the caller's data RNG or mutate class-level LAYERS.
    """
    import MinkowskiEngine as ME
    import torch

    from .model import MinkUNet14C

    seed = int(cfg["run"]["seed"])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = MinkUNet14C(
            int(cfg["model"]["in_channels"]),
            int(cfg["model"]["out_channels"]),
            D=int(cfg["model"].get("D", 3)),
        )
        state = model.state_dict()
        for name, value in reference.state_dict().items():
            if name not in state or state[name].shape != value.shape:
                raise ValueError(f"Shared initialization incompatible at {name}")
            state[name].copy_(value)
        for name, module in model.named_modules():
            if not name.startswith(("block3.1", "block8.1")):
                continue
            if isinstance(module, ME.MinkowskiConvolution):
                canonical = name.split(".", 1)[1]
                digest = hashlib.sha256(f"{seed}:extra:{canonical}".encode()).digest()
                torch.random.default_generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
                ME.utils.kaiming_normal_(module.kernel, mode="fan_out", nonlinearity="relu")
        del reference
