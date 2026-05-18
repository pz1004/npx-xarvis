from __future__ import annotations

from typing import Any

import numpy as np

from src.common import FilterResult
from src.data.event_io import append_confidence, empty_output_events


def empty_filter_result(stats: dict[str, Any] | None = None) -> FilterResult:
    empty_bool = np.zeros((0,), dtype=bool)
    empty_i8 = np.zeros((0,), dtype=np.int8)
    empty_i16 = np.zeros((0,), dtype=np.int16)
    return FilterResult(
        accepted_mask=empty_bool,
        confidence=empty_i8,
        support=empty_i16,
        pair_flag=empty_i8.copy(),
        output_events=empty_output_events(),
        stats=stats or {},
    )


def output_event_row(x: int, y: int, t: int, p: int, confidence: int) -> np.ndarray:
    return np.asarray([x, y, t, p, confidence], dtype=np.int64)


def stack_output_events(events: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    if len(events) == 0:
        return empty_output_events()
    return append_confidence(events, confidence)


def stack_output_rows(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return empty_output_events()
    return np.vstack(rows)
