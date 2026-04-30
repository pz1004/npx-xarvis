from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.datasets import _materialize_split_indices, resize_events_with_padding
from src.data.event_io import EVENT_T, normalize_events
from src.data.noise_injection import generate_shot_noise, inject_ba_noise, inject_mixed_noise, inject_shot_noise
from src.data.slicing import select_time_bin_us
from src.filters.metrics import evaluate_filter_predictions, state_memory_bytes
from src.experiments.train_eval import _build_scheduler


def _clean_events() -> np.ndarray:
    return np.asarray(
        [
            [0, 0, 0, 1],
            [1, 0, 10, -1],
            [0, 1, 20, 1],
            [1, 1, 30, -1],
        ],
        dtype=np.int64,
    )


def test_normalize_events_stable_sort_and_polarity_mapping() -> None:
    dtype = np.dtype([("x", np.int64), ("y", np.int64), ("t", np.int64), ("p", np.int64)])
    events = np.asarray(
        [
            (1, 0, 20, 1),
            (0, 0, 10, 0),
            (2, 0, 10, 1),
        ],
        dtype=dtype,
    )
    normalized = normalize_events(events)
    assert normalized[:, 2].tolist() == [0, 0, 10]
    assert normalized[:, 0].tolist() == [0, 2, 1]
    assert normalized[:, 3].tolist() == [-1, 1, 1]


def test_resize_events_with_padding_matches_expected_geometry() -> None:
    events = np.asarray(
        [
            [0, 0, 0, 1],
            [239, 179, 1, -1],
        ],
        dtype=np.int64,
    )
    resized = resize_events_with_padding(events, sensor_size=(240, 180, 2), target_size=(128, 128))
    assert resized[0, 0] == 0
    assert resized[0, 1] == 16
    assert resized[1, 0] == 127
    assert resized[1, 1] == 111


def test_select_time_bin_uses_p99_rule() -> None:
    slicing = select_time_bin_us([100_000, 610_000, 620_000])
    assert slicing.time_bin_us == 5000
    assert slicing.t_max <= 300
    assert select_time_bin_us([610_000]).time_bin_us == 5000


def test_materialized_split_indices_are_deterministic(tmp_path: Path) -> None:
    labels = np.asarray(([0] * 10) + ([1] * 10) + ([2] * 10), dtype=np.int64)
    first = _materialize_split_indices("toyset", labels, split_seed=2027, root=tmp_path)
    second = _materialize_split_indices("toyset", labels, split_seed=2027, root=tmp_path)
    assert first == second
    assert len(first["train"]) + len(first["val"]) + len(first["test"]) == len(labels)


def test_noise_injection_preserves_exact_signal_labels_and_metrics() -> None:
    clean = _clean_events()
    ba = inject_ba_noise(clean, sensor_size=(4, 4, 2), ratio=1.0, seed=0)
    shot = inject_shot_noise(clean, sensor_size=(4, 4, 2), ratio=1.0, tau_pair_us=100, seed=1)
    mixed = inject_mixed_noise(clean, sensor_size=(4, 4, 2), ratio=1.0, tau_pair_us=100, seed=2)
    assert int(np.sum(ba.is_signal)) == len(clean)
    assert int(np.sum(shot.is_signal)) == len(clean)
    assert int(np.sum(mixed.is_signal)) == len(clean)

    metrics = evaluate_filter_predictions(
        is_signal=np.asarray([True, True, False, False]),
        scores=np.asarray([0.9, 0.8, 0.2, 0.1]),
        accepted_mask=np.asarray([True, True, False, False]),
    )
    assert metrics["auc"] == 1.0
    assert metrics["ekr"] == 0.5
    assert metrics["compression_ratio"] == 2.0
    assert state_memory_bytes("proposed_balanced", (34, 34, 2)) == 15 * 34 * 34
    assert state_memory_bytes("proposed_lowmem", (34, 34, 2)) == 2 * (34 + 34) * 4


def test_shot_noise_lag_is_strictly_smaller_than_tau_pair() -> None:
    noise = generate_shot_noise(
        signal_events=_clean_events(),
        sensor_size=(4, 4, 2),
        ratio=1.0,
        tau_pair_us=20,
        rng=np.random.default_rng(7),
    )
    lags = np.diff(noise[:, EVENT_T].reshape(-1, 2), axis=1).ravel()
    assert np.all(lags < 20)


def test_scheduler_is_flat_for_first_ten_epochs_then_decays() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = _build_scheduler(optimizer, total_epochs=20, warmup_epochs=10)
    values = []
    for _ in range(12):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    assert values[:10] == [1.0] * 10
    assert values[10] < 1.0
