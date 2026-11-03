"""Source-correct, full-distribution guidance from a frozen input-contract mentor.

Project adaptation: not GRC, binary target-only KD, or an architectural claim.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from .study_metrics import ECLAIR_TRAIN_TO_COMMON

ARRAY_FIELDS = (
    "valid_counts",
    "eligible_counts",
    "teacher_entropy_sum",
    "teacher_nontarget_mass_sum",
    "teacher_true_probability_sum",
    "teacher_ignore_mass_sum",
    "kl_sum",
)


def validate_mentor_contract(actual, reference, student, expected_mode):
    current = copy.deepcopy(actual)
    resume = current["run"].pop("resume_from", None)
    saved = copy.deepcopy(reference)
    saved["run"].pop("resume_from", None)
    if (
        resume is not None
        and Path(resume).resolve()
        != Path(current["run"]["out_dir"]).resolve() / "checkpoints/last.pt"
    ):
        raise ValueError("Unexpected mentor resume provenance")
    if current != saved:
        raise ValueError(
            "Mentor checkpoint configuration differs from frozen source reference"
        )
    if current["data"]["features"]["return_input_mode"] != expected_mode:
        raise ValueError("Mentor input mode differs from frozen role")
    features = copy.deepcopy(current["data"]["features"])
    features["return_input_mode"] = "observed"
    if features != student["data"]["features"]:
        raise ValueError("Mentor/student feature bins or ordering differ")
    for key in ("patch", "voxelization", "label_space"):
        if current["data"][key] != student["data"][key]:
            raise ValueError(
                "Mentor/student physical or label contract differs: " + key
            )
    return saved


def guidance_weight(epoch, maximum=0.1, ramp_epochs=10):
    if epoch < 1 or maximum != 0.1 or ramp_epochs != 10:
        raise ValueError("Frozen guidance schedule is epochs1..10 ramp to0.1")
    return maximum * min(epoch / ramp_epochs, 1.0)


def check_mentor_finite(values, out_dir=None):
    import torch

    from .study_runtime import atomic_json

    flag = torch.isfinite(values).all().to(dtype=torch.int32)
    distributed = torch.distributed.is_initialized()
    if distributed:
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    if not bool(flag.item()):
        rank = torch.distributed.get_rank() if distributed else 0
        if rank == 0 and out_dir is not None:
            atomic_json(
                Path(out_dir) / "numerical_failure.json",
                {"error": "Nonfinite frozen mentor output", "automatic_restart": False},
            )
        raise RuntimeError("Nonfinite frozen mentor output on at least one rank")


def distribution_guidance(logits, probabilities, labels):
    """FP32 KL(q||p); native-correct teachers; macro COMMON class normalization.

    Denominators include ALL valid points of each present common class, including
    ineligible points (zero contribution). No confidence/target-data selector.
    """
    import torch
    import torch.nn.functional as F

    if logits.ndim != 2 or logits.shape[1] != 11 or probabilities.shape != logits.shape:
        raise ValueError("Mentor/student must align as N x11 native outputs")
    if labels.shape != (len(logits),) or probabilities.requires_grad:
        raise ValueError(
            "Aligned source labels and detached mentor probabilities required"
        )
    if (
        not bool(torch.isfinite(probabilities).all())
        or bool((probabilities < 0).any())
        or not torch.allclose(
            probabilities.sum(1),
            torch.ones(len(probabilities), device=probabilities.device),
            atol=1e-5,
            rtol=1e-5,
        )
    ):
        raise ValueError("Mentor probabilities must be finite normalized distributions")
    valid = (labels >= 0) & (labels < 11)
    if bool(((labels != -100) & ~valid).any()):
        raise ValueError("Invalid native source label")
    safe = labels.clamp(0, 10)
    lut = torch.tensor(ECLAIR_TRAIN_TO_COMMON, device=labels.device)
    common = lut[safe]
    eligible = valid & (common > 0) & (probabilities.argmax(1) == labels)
    q = probabilities.detach().float()
    logp = F.log_softmax(logits.float(), dim=1)
    point_kl = F.kl_div(logp, q, reduction="none").sum(1)
    entropy = -(q * q.clamp_min(1e-12).log()).sum(1)
    true_probability = q.gather(1, safe[:, None]).squeeze(1)
    stats = {key: [] for key in ARRAY_FIELDS}
    losses = []
    for cls in range(1, 8):
        points = valid & (common == cls)
        selected = points & eligible
        count = int(points.sum().item())
        stats["valid_counts"].append(count)
        stats["eligible_counts"].append(int(selected.sum().item()))
        stats["teacher_entropy_sum"].append(float(entropy[selected].sum().detach()))
        stats["teacher_nontarget_mass_sum"].append(
            float((1 - true_probability[selected]).sum().detach())
        )
        stats["teacher_true_probability_sum"].append(
            float(true_probability[selected].sum().detach())
        )
        stats["teacher_ignore_mass_sum"].append(
            float(q[selected][:, [0, 4]].sum().detach())
        )
        stats["kl_sum"].append(float(point_kl[selected].sum().detach()))
        if count:
            losses.append(point_kl[selected].sum() / count)
    loss = torch.stack(losses).mean() if losses else logits.float().sum() * 0
    stats["loss"] = float(loss.detach())
    return loss, stats


def gradient_diagnostics(guide, supervised, logits, weight):
    import torch

    a = torch.autograd.grad(guide, logits, retain_graph=True)[0].detach().float()
    b = torch.autograd.grad(supervised, logits, retain_graph=True)[0].detach().float()
    an, bn = float(a.norm()), float(b.norm())
    cosine = float((a * b).sum()) / max(an * bn, 1e-30)
    return {
        "weighted_logit_gradient_ratio": weight * an / max(bn, 1e-30),
        "logit_gradient_cosine": cosine,
        "guidance_logit_gradient_norm": an,
        "supervised_logit_gradient_norm": bn,
    }


class FrozenMentor:
    def __init__(self, cfg, device):
        import torch

        from .feature_contract_study import fingerprint, model_digest
        from .features import assert_checkpoint_feature_config
        from .model import build_model

        self.cfg, self.device = cfg, device
        self.spec = copy.deepcopy(cfg["study"]["mentor"])
        path = Path(self.spec["checkpoint"])
        before_cpu = torch.get_rng_state().clone()
        before_gpu = torch.cuda.get_rng_state(device).clone()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        reference_path = (
            Path(cfg["run"]["out_dir"]).resolve().parent
            / "reference"
            / self.spec["reference_arm"]
            / "config_resolved.json"
        )
        reference_cfg = json.loads(reference_path.read_text())
        mentor_cfg = validate_mentor_contract(
            checkpoint["cfg"], reference_cfg, cfg, self.spec["input_mode"]
        )
        assert_checkpoint_feature_config(checkpoint, reference_cfg)
        if (
            mentor_cfg["data"]["features"]["return_input_mode"]
            != self.spec["input_mode"]
        ):
            raise ValueError("Mentor checkpoint input contract differs")
        if (
            checkpoint["epoch"] != self.spec["epoch"]
            or model_digest(checkpoint["model_state"]) != self.spec["model_sha256"]
        ):
            raise ValueError(
                "Mentor checkpoint identity differs from source-selected evidence"
            )
        if (
            mentor_cfg["model"]["layers"] != [1] * 8
            or mentor_cfg["model"]["out_channels"] != 11
        ):
            raise ValueError("Unexpected mentor architecture")
        self.model = build_model(10, 11, cfg=mentor_cfg)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model = self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.identity = {
            "input_mode": self.spec["input_mode"],
            "epoch": self.spec["epoch"],
            "model_sha256": self.spec["model_sha256"],
            "checkpoint": str(path),
            "config_sha256": fingerprint(mentor_cfg),
            "frozen": True,
            "forward_amp": False,
            "target_selection": False,
        }
        del checkpoint
        if not torch.equal(before_cpu, torch.get_rng_state()) or not torch.equal(
            before_gpu, torch.cuda.get_rng_state(device)
        ):
            raise RuntimeError("Mentor setup consumed the training RNG")

    def predict(self, batch):
        import MinkowskiEngine as ME
        import torch

        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Mentor must remain frozen in evaluation mode")
        tick = time.monotonic()
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            values = batch["feats"].to(self.device, non_blocking=True).float()
            if self.spec["input_mode"] == "constant_pair":
                values = torch.zeros_like(values)
                values[:, 0] = values[:, 5] = 1
            coords = batch["coords"].to(self.device, non_blocking=True)
            output = self.model(
                ME.SparseTensor(values, coordinates=coords, device=self.device)
            )
            check_mentor_finite(output.F, self.cfg["run"]["out_dir"])
            result = output.F.float().softmax(1).detach(), output.C.detach()
            torch.cuda.synchronize(self.device)
            self.last_seconds = time.monotonic() - tick
            return result

    def loss(self, prediction, logits, labels, output_coords):
        from .ocons import match_sparse_coordinates

        q, coords = prediction
        indices, coverage = match_sparse_coordinates(
            coords, output_coords, require_all=True
        )
        if coverage != 1.0 or len(indices) != len(logits):
            raise RuntimeError("Mentor/student coordinate alignment incomplete")
        loss, stats = distribution_guidance(
            logits, q.index_select(0, indices.to(q.device)), labels
        )
        stats["mentor_forward_seconds"] = self.last_seconds
        return loss, stats

    def verify_immutable(self):
        from .feature_contract_study import model_digest

        if self.model.training or any(
            p.grad is not None for p in self.model.parameters()
        ):
            raise RuntimeError("Mentor mode/gradients changed")
        current = model_digest(self.model.state_dict())
        if current != self.identity["model_sha256"]:
            raise RuntimeError("Mentor parameters or normalization buffers changed")
        return current
