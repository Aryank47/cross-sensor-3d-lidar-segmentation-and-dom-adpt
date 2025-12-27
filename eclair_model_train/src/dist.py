from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class DistEnv:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0


def init_distributed(backend: str = "nccl") -> DistEnv:
    """
    Initialize torch.distributed from torchrun env vars.

    If not running under torchrun, returns disabled env.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistEnv(enabled=False)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    torch.cuda.set_device(local_rank)
    return DistEnv(
        enabled=True, rank=rank, world_size=world_size, local_rank=local_rank
    )


def is_main_process(env: DistEnv) -> bool:
    return (not env.enabled) or env.rank == 0


def barrier(env: DistEnv) -> None:
    if env.enabled:
        dist.barrier()


@torch.no_grad()
def all_reduce_sum(env: DistEnv, value: torch.Tensor) -> torch.Tensor:
    if env.enabled:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value
