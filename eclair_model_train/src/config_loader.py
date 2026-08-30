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


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_with_extends(path: Path, stack: tuple[Path, ...]) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved in stack:
        chain = " -> ".join(str(x) for x in (*stack, resolved))
        raise ValueError(f"Config extends cycle: {chain}")
    raw = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Top-level YAML must be a mapping: {resolved}")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    base = _load_with_extends(parent_path, (*stack, resolved))
    return _deep_merge(base, raw)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    cfg = _load_with_extends(Path(path), ())
    return _substitute_env(cfg)
