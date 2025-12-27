from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism is often slower; enable only if you really want bitwise repeatability.
    torch.backends.cudnn.benchmark = True


@dataclass
class StepTimer:
    t0: float = time.time()
    last: float = time.time()
    steps: int = 0

    def tick(self) -> float:
        now = time.time()
        dt = now - self.last
        self.last = now
        self.steps += 1
        return dt

    def elapsed(self) -> float:
        return time.time() - self.t0


class CSVLogger:
    """Append-only CSV logger (one row per epoch)."""

    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._initialized = self.path.exists()
        if not self._initialized:
            with self.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            self._initialized = True

    def log(self, row: Dict[str, Any]) -> None:
        # Ensure stable columns
        out = {k: row.get(k, "") for k in self.fieldnames}
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(out)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def atomic_save_torch(state: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def format_seconds(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    r = s - 60 * m
    if m < 60:
        return f"{m}m{r:.0f}s"
    h = m // 60
    m = m % 60
    return f"{h}h{m}m"


def env_or(default: str, env_key: str) -> str:
    return os.environ.get(env_key, default)
