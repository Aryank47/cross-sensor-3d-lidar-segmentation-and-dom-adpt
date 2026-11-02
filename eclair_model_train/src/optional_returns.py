"""Deterministic train-only return-group restriction; no RNG or geometry changes."""

from __future__ import annotations

import hashlib
import json

import numpy as np

CONSTANT = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)


def sample_is_masked(seed, epoch, filename, probability):
    if not 0 <= probability <= 1 or not np.isfinite(probability):
        raise ValueError("Mask probability must be finite and in [0,1]")
    key = json.dumps(
        [int(seed), int(epoch), str(filename), "optional_returns_v1"],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return value < probability * 2**64


def mask_batch(coords, labels, features, filenames, *, seed, epoch, probability=0.5):
    coords, labels, features = (
        np.asarray(coords),
        np.asarray(labels),
        np.asarray(features),
    )
    if (
        coords.shape != (len(labels), 4)
        or features.shape != (len(labels), 10)
        or features.dtype != np.float32
        or not np.isfinite(features).all()
    ):
        raise ValueError(
            "Expected aligned CPU coordinates, labels and finite float32 ten-channel features"
        )
    ids = np.unique(coords[:, 0]).astype(int)
    if ids.tolist() != list(range(len(filenames))):
        raise ValueError(
            "Collated coordinates and filenames do not have the same sample identities"
        )
    result = features.copy()
    audit = {
        "masked_samples": 0,
        "observed_samples": 0,
        "masked_native_counts": np.zeros(11, np.int64),
        "observed_native_counts": np.zeros(11, np.int64),
    }
    for batch_id, filename in enumerate(filenames):
        rows = coords[:, 0] == batch_id
        masked = sample_is_masked(seed, epoch, filename, probability)
        name = "masked" if masked else "observed"
        values = labels[rows]
        valid = (values >= 0) & (values < 11)
        audit[name + "_samples"] += 1
        audit[name + "_native_counts"] += np.bincount(values[valid], minlength=11)
        if masked:
            result[rows] = CONSTANT
    return result, audit


def validate_update_counters(counters, maximum_skips=13):
    counts = np.asarray(counters, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[1] != 2 or np.any(counts < 0):
        raise RuntimeError("Invalid rank optimizer accounting")
    if not np.all(counts == counts[0]):
        raise RuntimeError("DDP ranks disagree on applied updates or AMP skips")
    if int(counts[0, 1]) > maximum_skips:
        raise RuntimeError(
            "Optional-return numerical-health limit exceeded: more than 13 AMP skips"
        )
