from __future__ import annotations

import numpy as np

from src.data.event_io import EVENT_T, EVENT_X, EVENT_Y, sort_events_stable


def translate_events(events: np.ndarray, dx: int, dy: int, width: int, height: int) -> np.ndarray:
    translated = events.copy()
    translated[:, EVENT_X] += dx
    translated[:, EVENT_Y] += dy
    mask = (
        (translated[:, EVENT_X] >= 0)
        & (translated[:, EVENT_X] < width)
        & (translated[:, EVENT_Y] >= 0)
        & (translated[:, EVENT_Y] < height)
    )
    return translated[mask]


def horizontal_flip(events: np.ndarray, width: int) -> np.ndarray:
    flipped = events.copy()
    flipped[:, EVENT_X] = width - 1 - flipped[:, EVENT_X]
    return flipped


def temporal_shift(events: np.ndarray, shift_us: int) -> np.ndarray:
    shifted = events.copy()
    shifted[:, EVENT_T] = np.clip(shifted[:, EVENT_T] + shift_us, 0, None)
    return sort_events_stable(shifted)


def apply_train_augmentation(
    events: np.ndarray,
    sensor_size: tuple[int, int, int],
    dataset_name: str,
    time_bin_us: int,
    rng: np.random.Generator,
) -> np.ndarray:
    width, height, _ = sensor_size
    augmented = events
    if dataset_name != "nmnist" and rng.random() < 0.5:
        augmented = horizontal_flip(augmented, width)

    max_shift = 2 if dataset_name == "nmnist" else 4
    dx = int(rng.integers(-max_shift, max_shift + 1))
    dy = int(rng.integers(-max_shift, max_shift + 1))
    augmented = translate_events(augmented, dx=dx, dy=dy, width=width, height=height)
    temporal_bins = int(rng.integers(-1, 2))
    augmented = temporal_shift(augmented, temporal_bins * time_bin_us)
    return augmented

