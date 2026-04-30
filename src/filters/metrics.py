from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import auc, roc_curve


def event_keep_ratio(accepted_mask: np.ndarray) -> float:
    if len(accepted_mask) == 0:
        return 0.0
    return float(np.mean(accepted_mask.astype(np.float32)))


def compression_ratio(accepted_mask: np.ndarray) -> float:
    accepted = int(np.sum(accepted_mask))
    if accepted == 0:
        return float("inf")
    return float(len(accepted_mask) / accepted)


def accepted_events_per_sample(accepted_mask: np.ndarray) -> int:
    return int(np.sum(accepted_mask))


def area_under_noise_curve(accuracies: Iterable[float]) -> float:
    values = list(accuracies)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5
    fpr, tpr, _ = roc_curve(y_true.astype(np.int32), scores)
    return fpr, tpr, float(auc(fpr, tpr))


def tpr_at_fixed_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_targets: Iterable[float]) -> dict[float, float]:
    fpr, tpr, _ = safe_roc_auc(y_true, scores)
    results: dict[float, float] = {}
    for target in fpr_targets:
        valid = np.where(fpr <= target)[0]
        results[target] = float(tpr[valid[-1]]) if len(valid) else 0.0
    return results


def evaluate_filter_predictions(
    is_signal: np.ndarray,
    scores: np.ndarray,
    accepted_mask: np.ndarray,
) -> dict[str, float | dict[float, float]]:
    y_true = np.asarray(is_signal).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    accepted_mask = np.asarray(accepted_mask).astype(bool)
    fpr, tpr, roc_auc = safe_roc_auc(y_true, scores)
    return {
        "auc": roc_auc,
        "ekr": event_keep_ratio(accepted_mask),
        "compression_ratio": compression_ratio(accepted_mask),
        "accepted_events": int(np.sum(accepted_mask)),
        "tpr_at_fpr": tpr_at_fixed_fpr(y_true, scores, (0.01, 0.05, 0.10)),
        "roc_points": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
    }


def state_memory_bytes(method_name: str, sensor_size: tuple[int, int, int], t_max: int | None = None) -> int:
    width, height, _ = sensor_size
    if method_name == "proposed_balanced":
        return 15 * width * height
    if method_name == "proposed_lowmem":
        return 2 * (height + width) * 4
    if method_name == "ba_snn":
        return 8 * width * height
    if method_name == "stcf_rc_snn":
        return 2 * (width + height) * 8
    if method_name == "proposed_lowlat":
        return 0
    if method_name == "frame_snn":
        if t_max is None:
            raise ValueError("t_max is required for frame_snn state memory")
        return t_max * 2 * height * width * 4
    return 0
