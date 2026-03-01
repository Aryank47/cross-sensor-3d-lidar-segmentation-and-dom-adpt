#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import torch

# Ensure repo root is on PYTHONPATH so "import src.*" works
REPO_ROOT = Path(__file__).resolve().parents[1]  # .../eclair_model_train
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ----------------------------
# Small helpers (robustness)
# ----------------------------
def _expand_env_vars_in_text(s: str) -> str:
    """
    Expands ${VAR} patterns using os.environ (like your YAMLs use).
    Leaves unknown vars unchanged.
    """
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def repl(m: re.Match) -> str:
        k = m.group(1)
        return os.environ.get(k, m.group(0))

    return pattern.sub(repl, s)


def _load_yaml_with_env(path: Path) -> Dict[str, Any]:
    import yaml

    raw = path.read_text()
    raw = _expand_env_vars_in_text(raw)
    cfg = yaml.safe_load(raw)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a YAML mapping/dict, got: {type(cfg)}")
    return cfg


class DotDict(dict):
    """
    Minimal dot-access wrapper so cfg.data.xxx works even if you don't use OmegaConf here.
    """

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def _to_dotdict(x: Any) -> Any:
    if isinstance(x, dict):
        dd = DotDict()
        for k, v in x.items():
            dd[k] = _to_dotdict(v)
        return dd
    if isinstance(x, list):
        return [_to_dotdict(v) for v in x]
    return x


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # handles DDP checkpoints saved with "module." prefix
    if not state:
        return state
    keys = list(state.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state.items()}
    return state


def _pick_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for k in ("model_state", "state_dict", "model", "net"):
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    # sometimes checkpoints are saved as raw state_dict
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # type: ignore[return-value]
    raise KeyError(f"Could not find model state dict in checkpoint keys={list(ckpt.keys())}")


def _call_with_supported_kwargs(fn, **kwargs):
    """
    Calls fn(**filtered_kwargs) where filtered_kwargs are only those supported by fn's signature.
    This makes the script resilient to minor signature differences.
    """
    import inspect

    sig = inspect.signature(fn)
    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return fn(**accepted)


def _is_state_dict(d):
    return (
        isinstance(d, (dict, OrderedDict)) and len(d) > 0 and all(isinstance(k, str) and torch.is_tensor(v) for k, v in d.items())
    )


def _strip_prefix_if_present(sd, prefix="module."):
    if not _is_state_dict(sd):
        return sd
    if any(k.startswith(prefix) for k in sd.keys()):
        return OrderedDict((k[len(prefix) :] if k.startswith(prefix) else k, v) for k, v in sd.items())
    return sd


def _extract_state_dict(ckpt_obj):
    """
    Supports:
      - raw state_dict (OrderedDict[str, Tensor])
      - dict checkpoints with keys: model_state, state_dict, model, net, etc.
    Returns (state_dict, used_key)
    """
    if _is_state_dict(ckpt_obj):
        return ckpt_obj, "__root__"

    if not isinstance(ckpt_obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt_obj)}")

    # common keys across repos
    key_candidates = [
        "model_state",
        "model_state_dict",
        "state_dict",
        "model",
        "net",
        "network",
        "module",
    ]
    for k in key_candidates:
        if k in ckpt_obj and _is_state_dict(ckpt_obj[k]):
            return ckpt_obj[k], k

    # fallback: find first state_dict-like entry
    for k, v in ckpt_obj.items():
        if _is_state_dict(v):
            return v, k

    raise KeyError(f"Could not find a model state_dict in checkpoint. Keys={list(ckpt_obj.keys())[:30]}")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config used to build dataset/model.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt).")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to evaluate.")
    ap.add_argument(
        "--eval_mode",
        type=str,
        default=None,
        help="Override cfg.eval.mode (e.g. 'point' or 'voxel_windowed'). If not set, uses cfg.eval.mode.",
    )
    ap.add_argument("--out", type=str, default=None, help="Output JSON path. Default: next to ckpt.")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--no_amp", action="store_true", help="Disable autocast amp for eval.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    ckpt_path = Path(args.ckpt)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    cfg_raw = _load_yaml_with_env(cfg_path)
    if args.eval_mode is not None:
        cfg_raw.setdefault("eval", {})
        cfg_raw["eval"]["mode"] = args.eval_mode

    cfg = _to_dotdict(cfg_raw)

    # Import your existing training/eval utilities from train.py
    # (This is the key: we reuse your exact implementation.)
    import train as train_mod
    from src.dist import init_distributed

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = (not args.no_amp) and bool(cfg_raw.get("run", {}).get("amp", True)) and (device.type == "cuda")
    dist_env = init_distributed()

    # Build loaders (this should create correct DALES/ECLAIR val/test datasets)
    train_loader, val_loader, test_loader = _call_with_supported_kwargs(
        train_mod.build_dataloaders,
        cfg=cfg,
        dist_env=dist_env,
    )

    if args.split == "train":
        loader = train_loader
    elif args.split == "val":
        loader = val_loader
    else:
        loader = test_loader

    # Build model
    model = _call_with_supported_kwargs(train_mod.build_model, cfg=cfg, in_channels=10, out_channels=8)
    model.to(device)
    model.eval()

    # Load checkpoint
    # ckpt = torch.load(str(ckpt_path), map_location="cpu")
    # state = _pick_state_dict(ckpt)
    # state = _strip_module_prefix(state)
    # missing, unexpected = model.load_state_dict(state, strict=False)
    # if missing or unexpected:
    #     print(f"[ckpt] load_state_dict strict=False | missing={len(missing)} unexpected={len(unexpected)}")
    #     if len(missing) < 50:
    #         print("  missing:", missing)
    #     if len(unexpected) < 50:
    #         print("  unexpected:", unexpected)

    ckpt_obj = torch.load(args.ckpt, map_location="cpu")
    sd, used_key = _extract_state_dict(ckpt_obj)
    sd = _strip_prefix_if_present(sd, "module.")

    missing, unexpected = model.load_state_dict(sd, strict=False)

    print(f"[ckpt] loaded key='{used_key}' tensors={len(sd)} missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("[ckpt] missing (head):", missing[:20])
    if len(unexpected) > 0:
        print("[ckpt] unexpected (head):", unexpected[:20])

    # Decide eval mode
    eval_mode = cfg_raw.get("eval", {}).get("mode", "point")
    if isinstance(eval_mode, str):
        eval_mode = eval_mode.lower()
    else:
        eval_mode = "point"

    out_path = Path(args.out) if args.out else ckpt_path.with_name(f"{ckpt_path.stem}_{args.split}_{eval_mode}.json")
    criterion_cpu = train_mod.build_loss(cfg).cpu()
    
    print(f"Starting evaluation | split={args.split} eval_mode={eval_mode} amp={amp_enabled} device={device} loss={type(criterion_cpu).__name__}")
    # Run evaluation using your train.py eval functions
    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            if eval_mode == "point":
                metrics = _call_with_supported_kwargs(
                    train_mod.evaluate_pointwise,
                    dataset_obj=loader.dataset,
                    model=model,
                    device=device,
                    cfg=cfg,
                    split=args.split,
                    dist_env=dist_env,
                    criterion_cpu=criterion_cpu,
                    num_classes=8,
                    ignore_index=-100,
                    amp=True,
                )
            elif eval_mode == "voxel_windowed":
                metrics = _call_with_supported_kwargs(
                    train_mod.evaluate_voxel_windowed,
                    dataset_obj=loader.dataset,
                    model=model,
                    device=device,
                    cfg=cfg,
                    split=args.split,
                    dist_env=dist_env,
                    criterion_cpu=criterion_cpu,
                    num_classes=8,
                    ignore_index=-100,
                    amp=True,
                )
            else:
                raise ValueError(f"Unsupported eval_mode='{eval_mode}'. Use 'point' or 'voxel_windowed'.")

    # Normalize to JSON-serializable
    def _jsonify(x):
        if isinstance(x, (int, float, str)) or x is None:
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, (list, tuple)):
            return [_jsonify(v) for v in x]
        if isinstance(x, dict):
            return {str(k): _jsonify(v) for k, v in x.items()}
        return x

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonify(metrics), indent=2))
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()
    # main()
