from __future__ import annotations

import math

import numpy as np
import torch

from src.common import SlicingConfig
from src.data.event_io import EVENT_P, EVENT_T, EVENT_X, EVENT_Y, OUTPUT_CONFIDENCE


CONFIDENCE_CHANNELS = {
    (1, 1): 0,   # ON low
    (1, 2): 1,   # ON high
    (-1, 1): 2,  # OFF low
    (-1, 2): 3,  # OFF high
}


FRAME_CHANNELS = {
    1: 0,
    -1: 1,
}


def select_time_bin_us(
    durations_us: list[int] | np.ndarray,
    candidates: tuple[int, ...] = (1000, 2000, 5000),
    max_slices: int = 300,
) -> SlicingConfig:
    if max_slices <= 0:
        raise ValueError("max_slices must be positive")
    if not candidates:
        raise ValueError("At least one candidate time bin is required")
    if any(candidate <= 0 for candidate in candidates):
        raise ValueError("Candidate time bins must be positive")

    if len(durations_us) == 0:
        return SlicingConfig(time_bin_us=candidates[0], t_max=1)
    durations = np.asarray(durations_us, dtype=np.int64)
    p99_duration = int(np.ceil(np.percentile(durations, 99)))
    for candidate in candidates:
        t_max = max(1, int(math.ceil(p99_duration / candidate)))
        if t_max <= max_slices:
            return SlicingConfig(time_bin_us=candidate, t_max=t_max)

    # Long recordings such as DVS128 Gesture can exceed the fixed candidate
    # range. Keep the protocol's slice cap by moving to the next coarser
    # millisecond-aligned bin instead of returning thousands of dense slices.
    granularity_us = min(candidates)
    required_bin_us = max(1, int(math.ceil(p99_duration / max_slices)))
    final_candidate = int(math.ceil(required_bin_us / granularity_us) * granularity_us)
    return SlicingConfig(
        time_bin_us=final_candidate,
        t_max=max(1, int(math.ceil(p99_duration / final_candidate))),
    )


def sample_duration_us(events: np.ndarray) -> int:
    if len(events) == 0:
        return 0
    return int(events[-1, EVENT_T] - events[0, EVENT_T])


def output_events_to_event_tensor(
    output_events: np.ndarray,
    sensor_size: tuple[int, int, int],
    slicing: SlicingConfig,
) -> torch.Tensor:
    width, height, _ = sensor_size
    tensor = np.zeros((slicing.t_max, 4, height, width), dtype=np.float32)
    if len(output_events) == 0:
        return torch.from_numpy(tensor)

    time_bins = output_events[:, EVENT_T] // slicing.time_bin_us
    confidence = output_events[:, OUTPUT_CONFIDENCE]
    valid = (time_bins >= 0) & (time_bins < slicing.t_max) & ((confidence == 1) | (confidence == 2))
    if not np.any(valid):
        return torch.from_numpy(tensor)

    filtered = output_events[valid]
    filtered_time_bins = time_bins[valid].astype(np.int64, copy=False)
    filtered_confidence = filtered[:, OUTPUT_CONFIDENCE].astype(np.int64, copy=False)
    channels = np.where(filtered[:, EVENT_P] > 0, filtered_confidence - 1, filtered_confidence + 1)
    np.add.at(
        tensor,
        (
            filtered_time_bins,
            channels,
            filtered[:, EVENT_Y].astype(np.int64, copy=False),
            filtered[:, EVENT_X].astype(np.int64, copy=False),
        ),
        1.0,
    )
    return torch.from_numpy(tensor)


def raw_events_to_frame_tensor(
    events: np.ndarray,
    sensor_size: tuple[int, int, int],
    slicing: SlicingConfig,
) -> torch.Tensor:
    width, height, _ = sensor_size
    tensor = np.zeros((slicing.t_max, 2, height, width), dtype=np.float32)
    if len(events) == 0:
        return torch.from_numpy(tensor)

    time_bins = events[:, EVENT_T] // slicing.time_bin_us
    valid = (time_bins >= 0) & (time_bins < slicing.t_max)
    if not np.any(valid):
        return torch.from_numpy(tensor)

    filtered = events[valid]
    channels = np.where(filtered[:, EVENT_P] > 0, 0, 1)
    np.add.at(
        tensor,
        (
            time_bins[valid].astype(np.int64, copy=False),
            channels.astype(np.int64, copy=False),
            filtered[:, EVENT_Y].astype(np.int64, copy=False),
            filtered[:, EVENT_X].astype(np.int64, copy=False),
        ),
        1.0,
    )
    return torch.from_numpy(tensor)

