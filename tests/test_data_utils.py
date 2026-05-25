from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.common import SlicingConfig
from src.data.datasets import (
    CIFAR10DVS_DOWNLOAD_URL,
    DATASET_REGISTRY,
    DVSGESTURE_TEST_DOWNLOAD_URL,
    DVSGESTURE_TRAIN_DOWNLOAD_URL,
    _apply_target_policy,
    _build_raw_datasets,
    _extract_labels,
    _materialize_split_indices,
    resize_events_with_padding,
)
from src.data.event_io import EVENT_T, normalize_events
from src.data.noise_injection import generate_shot_noise, inject_ba_noise, inject_mixed_noise, inject_shot_noise
from src.data.slicing import output_events_to_event_tensor, raw_events_to_frame_tensor, select_time_bin_us
from src.filters.metrics import evaluate_filter_predictions, event_structural_ratio, state_memory_bytes
from src.experiments.train_eval import (
    _build_scheduler,
    _cap_batch_size_for_dense_input,
    _dense_batch_budget_bytes,
    _dense_sample_bytes,
)


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


def test_select_time_bin_falls_back_to_coarser_bin_for_long_recordings() -> None:
    slicing = select_time_bin_us([12_000_000])
    assert slicing.time_bin_us == 40_000
    assert slicing.t_max == 300
    assert select_time_bin_us([12_001_000]).t_max <= 300


def test_dense_batch_cap_keeps_large_snn_inputs_within_budget() -> None:
    budget_bytes = 256 * 1024 * 1024
    batch_size = _cap_batch_size_for_dense_input(
        configured_batch_size=32,
        sensor_size=(128, 128, 2),
        t_max=300,
        max_batch_bytes=budget_bytes,
    )
    assert batch_size == 3
    assert batch_size * _dense_sample_bytes((128, 128, 2), t_max=300) <= budget_bytes
    assert (
        _cap_batch_size_for_dense_input(
            configured_batch_size=128,
            sensor_size=(34, 34, 2),
            t_max=100,
            max_batch_bytes=budget_bytes,
        )
        == 128
    )


def test_dense_batch_budget_is_device_aware() -> None:
    assert _dense_batch_budget_bytes({}, torch.device("cpu")) == 64 * 1024 * 1024
    assert _dense_batch_budget_bytes({}, torch.device("cuda")) == 256 * 1024 * 1024
    assert _dense_batch_budget_bytes({"max_dense_batch_bytes": 123}, torch.device("cpu")) == 123
    assert _dense_batch_budget_bytes({"max_dense_batch_bytes": 123}, torch.device("cuda")) == 123
    assert _dense_batch_budget_bytes({"max_cpu_dense_batch_bytes": 456}, torch.device("cpu")) == 456
    assert _dense_batch_budget_bytes({"max_cuda_dense_batch_bytes": 789}, torch.device("cuda")) == 789


def test_vectorized_raw_tensorizer_accumulates_duplicates_and_ignores_late_events() -> None:
    events = np.asarray(
        [
            [1, 2, 0, 1],
            [1, 2, 1, 1],
            [2, 2, 1, -1],
            [3, 3, 2500, 1],
        ],
        dtype=np.int64,
    )
    tensor = raw_events_to_frame_tensor(events, sensor_size=(4, 4, 2), slicing=SlicingConfig(time_bin_us=1000, t_max=2))
    assert tensor.shape == (2, 2, 4, 4)
    assert tensor[0, 0, 2, 1].item() == 2.0
    assert tensor[0, 1, 2, 2].item() == 1.0
    assert tensor.sum().item() == 3.0


def test_vectorized_output_tensorizer_maps_confidence_channels() -> None:
    output_events = np.asarray(
        [
            [1, 1, 0, 1, 1],
            [1, 1, 10, 1, 2],
            [2, 1, 10, -1, 1],
            [2, 1, 10, -1, 2],
            [3, 1, 10, 1, 0],
            [0, 0, 2500, 1, 1],
        ],
        dtype=np.int64,
    )
    tensor = output_events_to_event_tensor(
        output_events,
        sensor_size=(4, 4, 2),
        slicing=SlicingConfig(time_bin_us=1000, t_max=2),
    )
    assert tensor[0, 0, 1, 1].item() == 1.0
    assert tensor[0, 1, 1, 1].item() == 1.0
    assert tensor[0, 2, 1, 2].item() == 1.0
    assert tensor[0, 3, 1, 2].item() == 1.0
    assert tensor.sum().item() == 4.0


def test_materialized_split_indices_are_deterministic(tmp_path: Path) -> None:
    labels = np.asarray(([0] * 10) + ([1] * 10) + ([2] * 10), dtype=np.int64)
    first = _materialize_split_indices("toyset", labels, split_seed=2027, root=tmp_path)
    second = _materialize_split_indices("toyset", labels, split_seed=2027, root=tmp_path)
    assert first == second
    assert len(first["train"]) + len(first["val"]) + len(first["test"]) == len(labels)


def test_cifar10dvs_download_url_is_patched_before_instantiation(monkeypatch, tmp_path: Path) -> None:
    spec = DATASET_REGISTRY["cifar10dvs"]
    dataset_class = spec["class"]
    original_url = dataset_class.url

    class FakeCIFAR10DVS:
        sensor_size = dataset_class.sensor_size
        url = original_url

        def __init__(self, save_to: Path):
            self.save_to = save_to
            self.url_at_init = self.__class__.url

    monkeypatch.setitem(spec, "class", FakeCIFAR10DVS)
    try:
        raw_train, raw_test = _build_raw_datasets("cifar10dvs", tmp_path)
    finally:
        monkeypatch.setitem(spec, "class", dataset_class)

    assert raw_test is None
    assert raw_train.url_at_init == CIFAR10DVS_DOWNLOAD_URL


def test_dvsgesture_download_urls_are_patched_before_instantiation(monkeypatch, tmp_path: Path) -> None:
    spec = DATASET_REGISTRY["dvsgesture"]
    dataset_class = spec["class"]

    class FakeDVSGesture:
        sensor_size = dataset_class.sensor_size
        train_url = dataset_class.train_url
        test_url = dataset_class.test_url

        def __init__(self, save_to: Path, train: bool = True):
            self.save_to = save_to
            self.train = train
            self.url_at_init = self.__class__.train_url if train else self.__class__.test_url

    monkeypatch.setitem(spec, "class", FakeDVSGesture)
    try:
        raw_train, raw_test = _build_raw_datasets("dvsgesture", tmp_path)
    finally:
        monkeypatch.setitem(spec, "class", dataset_class)

    assert raw_train.url_at_init == DVSGESTURE_TRAIN_DOWNLOAD_URL
    assert raw_test.url_at_init == DVSGESTURE_TEST_DOWNLOAD_URL


def test_target_policy_maps_string_labels_to_stable_integer_labels() -> None:
    class ToyStringLabelDataset:
        targets = ["BACKGROUND_Google", "class_b", "class_a", "class_b"]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int):
            return np.empty((0, 4), dtype=np.int64), self.targets[index]

    mapped = _apply_target_policy(
        ToyStringLabelDataset(),
        {
            "num_classes": 3,
            "map_targets": True,
        },
    )

    assert len(mapped) == 4
    assert mapped.targets == [0, 2, 1, 2]
    assert _extract_labels(mapped).tolist() == [0, 2, 1, 2]
    assert mapped[0][1] == 0
    assert mapped[1][1] == 2


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
    assert metrics["esr"] == 1.0
    assert metrics["compression_ratio"] == 2.0
    assert state_memory_bytes("proposed_balanced", (34, 34, 2)) == 15 * 34 * 34
    assert state_memory_bytes("proposed_lowmem", (34, 34, 2)) == 2 * (34 + 34) * 4


def test_label_based_esr_edge_cases() -> None:
    assert event_structural_ratio(
        np.asarray([True, True, False]),
        np.asarray([True, True, False]),
    ) == 1.0
    assert event_structural_ratio(
        np.asarray([True, True, False]),
        np.asarray([False, False, True]),
    ) == 0.0
    assert event_structural_ratio(
        np.asarray([True, True, True, False, False]),
        np.asarray([True, False, True, True, False]),
    ) == 2.0 / 3.0
    assert event_structural_ratio(
        np.asarray([False, False]),
        np.asarray([True, False]),
    ) == 0.0


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
