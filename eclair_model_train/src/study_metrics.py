"""Small, dependency-light common-space metrics; rows are GT, columns prediction."""

from __future__ import annotations

import numpy as np

COMMON_NAMES = ("ignore", "ground", "vegetation", "buildings", "wires", "poles", "fence", "vehicle")
DALES_TRAIN_TO_COMMON = (1, 2, 7, 7, 3, 5, 4, 6)


def common_histogram(native_counts, train_to_common=DALES_TRAIN_TO_COMMON):
    counts = np.asarray(native_counts, dtype=np.int64)
    lut = np.asarray(train_to_common, dtype=np.int64)
    if counts.shape != lut.shape or np.any(counts < 0):
        raise ValueError("Native histogram shape/count mismatch")
    result = np.zeros(8, dtype=np.int64)
    np.add.at(result, lut, counts)
    return result


def common_confusion(native_cm, train_to_common=DALES_TRAIN_TO_COMMON):
    mat = np.asarray(native_cm, dtype=np.int64)
    lut = np.asarray(train_to_common, dtype=np.int64)
    if mat.shape != (len(lut), len(lut)) or np.any(mat < 0):
        raise ValueError("Native confusion shape/count mismatch")
    out = np.zeros((8, 8), dtype=np.int64)
    np.add.at(out, (lut[:, None], lut[None, :]), mat)
    out[0] = 0  # ignored GT is never scored; ignore predictions remain false negatives
    return out


def confusion(gt, pred):
    gt, pred = np.asarray(gt, dtype=np.int64), np.asarray(pred, dtype=np.int64)
    if gt.shape != pred.shape or np.any((pred < 0) | (pred > 7)):
        raise ValueError("Prediction range or alignment mismatch")
    valid = (gt > 0) & (gt < 8)
    return np.bincount(gt[valid] * 8 + pred[valid], minlength=64).reshape(8, 8)


def summary(cm):
    cm = np.asarray(cm, dtype=np.int64).copy()
    if cm.shape != (8, 8) or np.any(cm < 0):
        raise ValueError("Expected a nonnegative 8x8 confusion matrix")
    cm[0] = 0
    tp = cm.diagonal()
    support, predicted = cm.sum(1), cm.sum(0)
    union = support + predicted - tp
    iou = np.divide(tp, union, out=np.zeros(8, dtype=float), where=union > 0)
    precision = np.divide(tp, predicted, out=np.zeros(8, dtype=float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(8, dtype=float), where=support > 0)
    return {
        "all_seven_miou": float(iou[1:].mean()),
        "utility_miou": float(iou[[4, 5]].mean()),
        "per_class_iou": dict(zip(COMMON_NAMES[1:], iou[1:].tolist())),
        "precision": dict(zip(COMMON_NAMES[1:], precision[1:].tolist())),
        "recall": dict(zip(COMMON_NAMES[1:], recall[1:].tolist())),
        "support": support.tolist(),
        "predicted": predicted.tolist(),
        "undefined_union_ids": (np.flatnonzero(union[1:] == 0) + 1).tolist(),
        "confusion_matrix": cm.tolist(),
        "wire_to_pole": int(cm[4, 5]),
        "wire_to_pole_rate": float(cm[4, 5] / max(1, support[4])),
        "pole_fp_wire_fraction": float(cm[4, 5] / max(1, predicted[5] - tp[5])),
    }


def balanced_gate(reference_source, candidate_source, reference_target, candidate_target):
    """Conservative observed screen, not a statistical significance test."""
    per_class = {
        name: candidate_target["per_class_iou"][name] - reference_target["per_class_iou"][name]
        for name in COMMON_NAMES[1:]
    }
    checks = {
        "target_all_seven_improves": candidate_target["all_seven_miou"] > reference_target["all_seven_miou"],
        "target_utility_improves": candidate_target["utility_miou"] > reference_target["utility_miou"],
        "no_target_class_regression": min(per_class.values()) >= 0,
        "source_all_seven_preserved": candidate_source["all_seven_miou"]
        >= reference_source["all_seven_miou"],
    }
    return {
        "passes_observed_screen": all(checks.values()),
        "checks": checks,
        "target_class_deltas": per_class,
        "replication_required": True,
    }
