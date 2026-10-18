"""Bounded telemetry, completed-epoch resume and storage-aware study saves."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import signal
import time
from pathlib import Path

import numpy as np

from .study_metrics import common_confusion, common_histogram, summary


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def storage_guard(directory, estimated_write_bytes, cfg):
    """statvfs does NOT expose user quota. Enforce the separate admitted run cap."""
    directory = Path(directory)
    root = next(
        (p for p in (directory, *directory.parents) if (p / "config_resolved.yaml").is_file()),
        directory.parent if directory.name == "checkpoints" else directory,
    )
    files = [p for p in root.rglob("*") if p.is_file()]
    used = sum(p.stat().st_size for p in files)
    diagnostics = sum(p.stat().st_size for p in files if p.suffix != ".pt")
    if diagnostics > int(cfg.get("diagnostics_budget_bytes", 128 * 1024**2)):
        raise OSError(f"Compact-diagnostic budget exceeded: {diagnostics} bytes")
    reserve = int(cfg.get("reserve_bytes", 512 * 1024**2))
    cap = int(cfg.get("run_budget_bytes", 2 * 1024**3))
    free = shutil.disk_usage(directory).free
    inodes = None
    if hasattr(os, "statvfs"):
        inodes = os.statvfs(directory).f_favail
    if free < estimated_write_bytes + reserve or used + estimated_write_bytes > cap:
        raise OSError(
            f"Storage headroom insufficient: free={free}, run_used={used}, "
            f"next_write={estimated_write_bytes}, run_cap={cap}, reserve={reserve}"
        )
    if inodes is not None and inodes < 32:
        raise OSError(f"Insufficient available inodes: {inodes}")
    return {
        "filesystem_free_bytes": free,
        "run_used_bytes": used,
        "next_write_bytes": estimated_write_bytes,
        "quota_is_not_measured_by_statvfs": True,
    }


def trim_jsonl_after(path, completed_epoch):
    path = Path(path)
    if not path.exists():
        return
    rows = []
    contents = path.read_text()
    lines = contents.splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not contents.endswith("\n"):
                continue  # discard only an interrupted trailing write
            raise ValueError(f"Corrupt completed telemetry row in {path} at line {index + 1}")
        if int(row["epoch"]) <= completed_epoch:
            rows.append(json.dumps(row, allow_nan=False))
    path.write_text("".join(row + "\n" for row in rows))


class StudyRuntime:
    def __init__(self, cfg, dist_env, device, out_dir):
        import torch

        self.torch, self.cfg, self.dist, self.device = torch, cfg, dist_env, device
        self.options = cfg.get("study", {})
        self.enabled = bool(self.options.get("enabled", False))
        self.out = Path(out_dir)
        self.started = time.monotonic()
        self.deadline = float(os.environ.get("STUDY_DEADLINE_UNIX", "inf"))
        self.stop_requested = False
        self.epoch_times = []
        self.best_utility = None
        self.updates = 0
        self.skips = 0
        self.last_completed = 0
        self.native_counts = np.zeros(int(cfg["model"]["out_channels"]), dtype=np.int64)
        if self.enabled:
            for sig in (signal.SIGTERM, signal.SIGUSR1):
                signal.signal(sig, self.request_stop)

    def request_stop(self, _signum, _frame):
        self.stop_requested = True  # finish this epoch; checkpoint before returning

    def state(self):
        t = self.torch
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": t.get_rng_state(),
            "torch_cuda": t.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
            "updates": self.updates,
            "skips": self.skips,
            "best_utility": self.best_utility,
        }

    def restore(self, states, epoch):
        if len(states) != self.dist.world_size:
            raise ValueError("Study resume requires the original DDP world size")
        state = states[self.dist.rank]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        self.torch.set_rng_state(state["torch_cpu"].cpu())
        if state["torch_cuda"] is not None:
            self.torch.cuda.set_rng_state(state["torch_cuda"].cpu(), self.device)
        self.updates, self.skips = int(state["updates"]), int(state["skips"])
        self.best_utility = state["best_utility"]
        self.last_completed = epoch
        if self.dist.rank == 0:
            for name in ("study_epochs.jsonl", "loss_gradient_audit.jsonl"):
                trim_jsonl_after(self.out / name, epoch)
            # A killed post-checkpoint tail may have logged beyond the resume epoch.
            csv_path = self.out / "metrics.csv"
            if csv_path.exists():
                import csv

                with csv_path.open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    fields = reader.fieldnames
                    rows = [r for r in reader if int(r["epoch"]) <= epoch]
                with csv_path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)

    def begin_epoch(self, epoch, loader):
        self.epoch = epoch
        self.native_counts[:] = 0
        self.presence = np.zeros_like(self.native_counts)
        self.samples = self.microbatches = 0
        self.reordered_batches = 0
        self.meta = {}
        # Per-epoch worker seeds are independent of model initialization and resume.
        loader.generator.manual_seed(int(self.cfg["run"]["seed"]) + epoch * 1009 + self.dist.rank)

    def observe_batch(self, batch):
        labels = batch["labels"].detach().cpu().numpy()
        coords = batch["coords"].detach().cpu().numpy()
        nclasses = len(self.native_counts)
        valid = (labels >= 0) & (labels < nclasses)
        self.native_counts += np.bincount(labels[valid], minlength=nclasses)
        for item in np.unique(coords[:, 0]):
            ids = labels[(coords[:, 0] == item) & valid]
            self.presence += np.bincount(ids, minlength=nclasses) > 0
            self.samples += 1
        self.microbatches += 1
        for key in (
            "meta_crop_size_xy_m",
            "meta_shrink_steps",
            "meta_resample_attempt",
            "meta_used_fallback",
            "meta_used_rare_center",
            "meta_n_vox",
            "meta_n_points",
            "meta_accepted_over_soft",
        ):
            if key in batch:
                values = batch[key].detach().double().cpu().numpy().reshape(-1)
                old = self.meta.setdefault(key, [0.0, 0.0, 0.0])
                old[0] += float(values.sum())
                old[1] += len(values)
                old[2] += float((values > 0).sum())

    def audit_loss(self, logits, labels, criterion, model, step):
        """Bounded classifier-kernel gradients; no .grad mutation or random sampling."""
        t = self.torch
        self.reordered_batches += int(getattr(model, "_study_label_reordered", False))
        if step > int(self.options.get("gradient_batches_per_epoch", 1)):
            return
        valid = labels >= 0
        row = {
            "epoch": self.epoch,
            "step": step,
            "rank": self.dist.rank,
            "focal_variant": "alpha_inside_cross_entropy_before_exp",
            "classifier_gradient_norms": {},
            "class_focal_loss_means": {},
            "class_support": {},
        }
        focal = criterion.focal if hasattr(criterion, "focal") else criterion
        with t.autocast(device_type=self.device.type, enabled=bool(self.cfg["run"].get("amp", True))):
            lf = focal(logits, labels)
            row["focal"] = float(lf.detach())
            weight = float(criterion.current_weight()) if hasattr(criterion, "current_weight") else 0.0
            row["lovasz_weight"] = weight
            row["lovasz_unweighted"] = float(criterion.lovasz(logits, labels).detach()) if weight else 0.0
            features = model._study_classifier_features
            if features is None or features.shape[0] != logits.shape[0]:
                raise RuntimeError("Classifier feature/label alignment missing")
            audit_logits = logits.detach().float().requires_grad_(True)
            for cls in range(len(self.native_counts)):
                mask = valid & (labels == cls)
                row["class_support"][str(cls)] = int(mask.sum())
                if not bool(mask.any()):
                    row["classifier_gradient_norms"][str(cls)] = None
                    row["class_focal_loss_means"][str(cls)] = None
                    continue
                # Local class-conditioned focal objective, not a decomposition of Lovasz.
                value = focal(audit_logits[mask], labels[mask])
                row["class_focal_loss_means"][str(cls)] = float(value.detach())
                grad_logits = t.autograd.grad(value, audit_logits, retain_graph=True)[0]
                with t.autocast(device_type=self.device.type, enabled=False):
                    grad = features.detach().float().T @ grad_logits.float()
                row["classifier_gradient_norms"][str(cls)] = float(grad.detach().float().norm())
        rows = [None] * self.dist.world_size
        if self.dist.enabled:
            t.distributed.all_gather_object(rows, row)
        else:
            rows = [row]
        if self.dist.rank == 0:
            with (self.out / "loss_gradient_audit.jsonl").open("a") as stream:
                for item in rows:
                    stream.write(json.dumps(item, allow_nan=False) + "\n")

    def optimizer_step(self, scaler, optimizer):
        before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) < before:
            self.skips += 1
        else:
            self.updates += 1

    def end_epoch(self, val_metrics, seconds, model):
        t = self.torch
        local = {
            "native_counts": self.native_counts.tolist(),
            "presence": self.presence.tolist(),
            "samples": self.samples,
            "microbatches": self.microbatches,
            "coordinate_reordered_batches": self.reordered_batches,
            "metadata": self.meta,
            "optimizer_updates_cumulative": self.updates,
            "amp_skips_cumulative": self.skips,
        }
        gathered = [None] * self.dist.world_size
        if self.dist.enabled:
            t.distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]
        validation = None
        if val_metrics.get("confusion_matrix") is not None:
            validation = summary(common_confusion(val_metrics["confusion_matrix"]))
            score = validation["utility_miou"]
            if self.best_utility is None or score > self.best_utility["score"]:
                self.best_utility = {"epoch": self.epoch, "score": score, "weights_retained": False}
        self.epoch_times.append(seconds)
        if self.dist.rank == 0:
            counts = np.sum([g["native_counts"] for g in gathered], axis=0)
            common = common_histogram(counts)
            row = {
                "epoch": self.epoch,
                "source_validation": validation,
                "accepted_native_counts": counts.tolist(),
                "accepted_common_counts": common.tolist(),
                "ranks": gathered,
                "best_source_utility": self.best_utility,
                "train_seconds": seconds,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            with (self.out / "study_epochs.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        # Reject nonfinite objectives before writing a supposedly healthy checkpoint.
        if not math.isfinite(float(val_metrics["miou"])) and val_metrics.get("confusion_matrix") is not None:
            raise RuntimeError("Nonfinite validation score")

    def save(self, common, resumable, is_best):
        from .utils import atomic_save_torch

        t = self.torch
        states = [None] * self.dist.world_size
        if self.dist.enabled:
            t.distributed.all_gather_object(states, self.state())
        else:
            states = [self.state()]
        resumable["study_rank_states"] = states
        resumable["study_world_size"] = self.dist.world_size
        error = [None]
        if self.dist.rank == 0:
            try:
                cp = self.out / "checkpoints"
                cp.mkdir(exist_ok=True)
                model_bytes = sum(v.numel() * v.element_size() for v in common["model_state"].values())
                estimate = int(model_bytes * 3.5 + 16 * 1024**2)
                storage_guard(cp, estimate, self.options.get("storage", {}))
                atomic_save_torch(resumable, cp / "last.pt")
                if is_best:
                    storage_guard(cp, model_bytes + 16 * 1024**2, self.options.get("storage", {}))
                    atomic_save_torch(common, cp / "best.pt")
                atomic_json(
                    self.out / "checkpoint_selection.json",
                    {
                        "primary": "source_all_native_validation_miou",
                        "best_epoch": common["best_epoch"],
                        "last_epoch": common["epoch"],
                        "utility_best": self.best_utility,
                        "target_labels_used": False,
                    },
                )
            except Exception as exc:  # noqa: BLE001 -- every rank must receive a save failure
                error[0] = f"{type(exc).__name__}: {exc}"
        if self.dist.enabled:
            t.distributed.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(f"Study save failed; previous atomic checkpoint preserved: {error[0]}")
        self.last_completed = common["epoch"]

    def repair_best(self, checkpoint):
        """Recover if a save was interrupted between replacing last and best."""
        error = [None]
        if self.dist.rank == 0 and checkpoint["best_epoch"] == checkpoint["epoch"]:
            try:
                from .utils import atomic_save_torch

                best = self.out / "checkpoints/best.pt"
                existing = (
                    self.torch.load(best, map_location="cpu", weights_only=False) if best.exists() else {}
                )
                if existing.get("epoch") != checkpoint["epoch"]:
                    state = {
                        k: checkpoint[k]
                        for k in (
                            "epoch",
                            "global_step",
                            "model_state",
                            "cfg",
                            "best_val_miou",
                            "best_epoch",
                            "method_contract",
                            "resume_contract_sha256",
                        )
                    }
                    estimate = sum(v.numel() * v.element_size() for v in state["model_state"].values())
                    storage_guard(best.parent, estimate + 16 * 1024**2, self.options.get("storage", {}))
                    atomic_save_torch(state, best)
            except Exception as exc:  # noqa: BLE001 -- coordinate checkpoint recovery across ranks
                error[0] = str(exc)
        if self.dist.enabled:
            self.torch.distributed.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(f"Could not repair interrupted best checkpoint: {error[0]}")

    def should_stop(self):
        t = self.torch
        forecast = max(self.epoch_times[-3:], default=1800.0) * 1.5
        seconds_left = self.deadline - time.time()
        stop = self.stop_requested or (self.out / "STOP_REQUESTED").exists() or seconds_left < forecast + 900
        flag = t.tensor(int(stop), device=self.device)
        if self.dist.enabled:
            t.distributed.all_reduce(flag, op=t.distributed.ReduceOp.MAX)
        return bool(flag.item())

    def finish(self, status):
        if self.dist.rank == 0:
            atomic_json(
                self.out / "training_status.json",
                {
                    "status": status,
                    "last_completed_epoch": self.last_completed,
                    "epochs_required": int(self.cfg["epochs"]),
                    "target_evaluation_pending": True,
                },
            )
