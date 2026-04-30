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
    if len(durations_us) == 0:
        return SlicingConfig(time_bin_us=candidates[0], t_max=1)
    durations = np.asarray(durations_us, dtype=np.int64)
    p99_duration = int(np.ceil(np.percentile(durations, 99)))
    for candidate in candidates:
        t_max = max(1, int(math.ceil(p99_duration / candidate)))
        if t_max <= max_slices:
            return SlicingConfig(time_bin_us=candidate, t_max=t_max)
    final_candidate = candidates[-1]
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
    for event in output_events:
        time_bin = int(event[EVENT_T] // slicing.time_bin_us)
        if time_bin < 0 or time_bin >= slicing.t_max:
            continue
        polarity = int(event[EVENT_P])
        confidence = int(event[OUTPUT_CONFIDENCE])
        if confidence == 0:
            continue
        channel = CONFIDENCE_CHANNELS[(polarity, confidence)]
        tensor[time_bin, channel, int(event[EVENT_Y]), int(event[EVENT_X])] += 1.0
    return torch.from_numpy(tensor)


def raw_events_to_frame_tensor(
    events: np.ndarray,
    sensor_size: tuple[int, int, int],
    slicing: SlicingConfig,
) -> torch.Tensor:
    width, height, _ = sensor_size
    tensor = np.zeros((slicing.t_max, 2, height, width), dtype=np.float32)
    for event in events:
        time_bin = int(event[EVENT_T] // slicing.time_bin_us)
        if time_bin < 0 or time_bin >= slicing.t_max:
            continue
        channel = FRAME_CHANNELS[int(event[EVENT_P])]
        tensor[time_bin, channel, int(event[EVENT_Y]), int(event[EVENT_X])] += 1.0
    return torch.from_numpy(tensor)

