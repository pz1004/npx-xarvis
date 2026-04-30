from __future__ import annotations

import numpy as np

from src.common import CorruptedEvents
from src.data.event_io import EVENT_P, EVENT_T, EVENT_X, EVENT_Y, normalize_events, sort_events_stable


def _noise_count(signal_events: np.ndarray, ratio: float, explicit_count: int | None) -> int:
    if explicit_count is not None:
        return int(explicit_count)
    return int(round(len(signal_events) * ratio))


def _event_time_horizon(events: np.ndarray) -> int:
    if len(events) == 0:
        return 1
    return max(int(events[:, EVENT_T].max()), 1)


def _merge_signal_and_noise(signal_events: np.ndarray, noise_events: np.ndarray, source: str, ratio: float) -> CorruptedEvents:
    signal_events = normalize_events(signal_events)
    noise_events = normalize_events(noise_events)
    merged = np.concatenate([signal_events, noise_events], axis=0)
    labels = np.concatenate(
        [
            np.ones(len(signal_events), dtype=bool),
            np.zeros(len(noise_events), dtype=bool),
        ]
    )
    original_index = np.arange(len(merged), dtype=np.int64)
    order = np.lexsort((original_index, merged[:, EVENT_T]))
    return CorruptedEvents(
        events=merged[order],
        is_signal=labels[order],
        source=source,
        ratio=ratio,
    )


def generate_ba_noise(
    signal_events: np.ndarray,
    sensor_size: tuple[int, int, int],
    ratio: float,
    rng: np.random.Generator,
    explicit_count: int | None = None,
) -> np.ndarray:
    count = _noise_count(signal_events, ratio, explicit_count)
    if count <= 0:
        return np.zeros((0, 4), dtype=np.int64)
    width, height, _ = sensor_size
    horizon = _event_time_horizon(signal_events)
    x = rng.integers(0, width, size=count, endpoint=False, dtype=np.int64)
    y = rng.integers(0, height, size=count, endpoint=False, dtype=np.int64)
    t = rng.integers(0, horizon + 1, size=count, endpoint=False, dtype=np.int64)
    p = rng.choice(np.array([-1, 1], dtype=np.int64), size=count)
    return sort_events_stable(np.stack([x, y, t, p], axis=1))


def generate_shot_noise(
    signal_events: np.ndarray,
    sensor_size: tuple[int, int, int],
    ratio: float,
    tau_pair_us: int,
    rng: np.random.Generator,
    explicit_count: int | None = None,
) -> np.ndarray:
    total_events = _noise_count(signal_events, ratio, explicit_count)
    if total_events <= 0:
        return np.zeros((0, 4), dtype=np.int64)
    if total_events % 2 == 1:
        total_events += 1
    width, height, _ = sensor_size
    horizon = _event_time_horizon(signal_events)
    pair_count = total_events // 2
    x = rng.integers(0, width, size=pair_count, endpoint=False, dtype=np.int64)
    y = rng.integers(0, height, size=pair_count, endpoint=False, dtype=np.int64)
    first_p = rng.choice(np.array([-1, 1], dtype=np.int64), size=pair_count)
    # Stage 4 defines shot pairs as having an inter-event lag strictly smaller than tau_pair_us.
    lag_upper_exclusive = max(int(tau_pair_us), 2)
    lags = rng.integers(1, lag_upper_exclusive, size=pair_count, endpoint=False, dtype=np.int64)
    start_max = max(horizon - int(lags.max(initial=0)), 1)
    t0 = rng.integers(0, start_max + 1, size=pair_count, endpoint=False, dtype=np.int64)
    first = np.stack([x, y, t0, first_p], axis=1)
    second = np.stack([x, y, t0 + lags, -first_p], axis=1)
    return sort_events_stable(np.concatenate([first, second], axis=0))


def inject_ba_noise(
    signal_events: np.ndarray,
    sensor_size: tuple[int, int, int],
    ratio: float,
    seed: int,
) -> CorruptedEvents:
    rng = np.random.default_rng(seed)
    noise_events = generate_ba_noise(signal_events, sensor_size, ratio, rng)
    return _merge_signal_and_noise(signal_events, noise_events, source="ba", ratio=ratio)


def inject_shot_noise(
    signal_events: np.ndarray,
    sensor_size: tuple[int, int, int],
    ratio: float,
    tau_pair_us: int,
    seed: int,
) -> CorruptedEvents:
    rng = np.random.default_rng(seed)
    noise_events = generate_shot_noise(signal_events, sensor_size, ratio, tau_pair_us, rng)
    return _merge_signal_and_noise(signal_events, noise_events, source="shot", ratio=ratio)


def inject_mixed_noise(
    signal_events: np.ndarray,
    sensor_size: tuple[int, int, int],
    ratio: float,
    tau_pair_us: int,
    seed: int,
) -> CorruptedEvents:
    rng = np.random.default_rng(seed)
    total_events = _noise_count(signal_events, ratio, explicit_count=None)
    ba_events = int(round(total_events * 0.7))
    shot_events = max(total_events - ba_events, 0)
    ba_noise = generate_ba_noise(signal_events, sensor_size, ratio, rng, explicit_count=ba_events)
    shot_noise = generate_shot_noise(signal_events, sensor_size, ratio, tau_pair_us, rng, explicit_count=shot_events)
    noise_events = np.concatenate([ba_noise, shot_noise], axis=0)
    return _merge_signal_and_noise(signal_events, noise_events, source="mixed", ratio=ratio)


def build_noise_suites(
    signal_events: np.ndarray,
    sensor_size: tuple[int, int, int],
    ratios: tuple[float, ...],
    tau_pair_us: int,
    seed: int,
) -> list[CorruptedEvents]:
    suites: list[CorruptedEvents] = []
    for offset, ratio in enumerate(ratios):
        suites.append(inject_ba_noise(signal_events, sensor_size, ratio, seed + offset * 13))
        suites.append(inject_shot_noise(signal_events, sensor_size, ratio, tau_pair_us, seed + offset * 17))
        suites.append(inject_mixed_noise(signal_events, sensor_size, ratio, tau_pair_us, seed + offset * 19))
    return suites
