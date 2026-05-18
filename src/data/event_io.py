from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


EVENT_X = 0
EVENT_Y = 1
EVENT_T = 2
EVENT_P = 3

OUTPUT_CONFIDENCE = 4

EVENT_COLUMN_NAMES = ("x", "y", "t", "p")
OUTPUT_COLUMN_NAMES = ("x", "y", "t", "p", "c")


def _to_numpy(events: np.ndarray | torch.Tensor | Iterable[tuple[int, int, int, int]]) -> np.ndarray:
    if isinstance(events, torch.Tensor):
        return events.detach().cpu().numpy()
    return np.asarray(events)


def _structured_to_matrix(events: np.ndarray) -> np.ndarray:
    if events.dtype.names is None:
        return np.asarray(events)
    if not set(EVENT_COLUMN_NAMES).issubset(set(events.dtype.names)):
        raise ValueError(f"Expected event fields {EVENT_COLUMN_NAMES}, got {events.dtype.names}")
    return np.stack(
        [events["x"], events["y"], events["t"], events["p"]],
        axis=1,
    )


def normalize_polarity(polarity: np.ndarray) -> np.ndarray:
    polarity = polarity.astype(np.int8, copy=False)
    unique_values = set(np.unique(polarity).tolist())
    if unique_values.issubset({0, 1}):
        return (polarity * 2 - 1).astype(np.int8)
    if unique_values.issubset({-1, 1}):
        return polarity
    polarity = np.where(polarity > 0, 1, -1)
    return polarity.astype(np.int8)


def sort_events_stable(events: np.ndarray) -> np.ndarray:
    if len(events) == 0:
        return events.astype(np.int64, copy=False)
    original_index = np.arange(len(events), dtype=np.int64)
    order = np.lexsort((original_index, events[:, EVENT_T]))
    return events[order]


def normalize_events(events: np.ndarray | torch.Tensor | Iterable[tuple[int, int, int, int]]) -> np.ndarray:
    matrix = _structured_to_matrix(_to_numpy(events))
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[1] < 4:
        raise ValueError(f"Expected an event matrix with shape (N, 4+), got {matrix.shape}")
    matrix = matrix[:, :4].astype(np.int64, copy=True)
    matrix = sort_events_stable(matrix)
    if len(matrix) == 0:
        return matrix
    matrix[:, EVENT_P] = normalize_polarity(matrix[:, EVENT_P])
    matrix[:, EVENT_T] -= matrix[0, EVENT_T]
    return matrix


def append_confidence(events: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    if len(events) != len(confidence):
        raise ValueError("events and confidence must have matching lengths")
    confidence = np.asarray(confidence, dtype=np.int64).reshape(-1, 1)
    return np.concatenate([events.astype(np.int64, copy=False), confidence], axis=1)


def empty_output_events() -> np.ndarray:
    return np.zeros((0, 5), dtype=np.int64)

