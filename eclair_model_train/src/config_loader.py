from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

_VAR_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")


def _substitute_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _substitute_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_env(v) for v in obj]
    if isinstance(obj, str):

        def repl(m: re.Match) -> str:
            key = m.group(1)
            if key not in os.environ:
                raise KeyError(f"Config references env var {key} but it's not set")
            return os.environ[key]

        return _VAR_RE.sub(repl, obj)
    return obj


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    return _substitute_env(cfg)
