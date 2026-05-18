from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, TypeAlias

import numpy as np
import tonic
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from src.common import DatasetMetadata, FilterResult, SlicingConfig
from src.data.augmentation import apply_train_augmentation
from src.data.event_io import EVENT_X, EVENT_Y, normalize_events
from src.data.noise_injection import inject_ba_noise, inject_mixed_noise, inject_shot_noise
from src.data.slicing import sample_duration_us, select_time_bin_us
from src.utils.logging import get_logger
from src.utils.seed import make_generator, seed_worker
from src.utils.serialization import load_json, save_json


LOGGER = get_logger(__name__)
FIGSHARE_NDOWNLOADER_URL = "https://ndownloader.figshare.com/files"
DVSGESTURE_TRAIN_DOWNLOAD_URL = f"{FIGSHARE_NDOWNLOADER_URL}/38022171"
DVSGESTURE_TEST_DOWNLOAD_URL = f"{FIGSHARE_NDOWNLOADER_URL}/38020584"
CIFAR10DVS_DOWNLOAD_URL = f"{FIGSHARE_NDOWNLOADER_URL}/38023437"


DATASET_REGISTRY = {
    "nmnist": {
        "class": tonic.datasets.NMNIST,
        "sensor_size": tonic.datasets.NMNIST.sensor_size,
        "num_classes": 10,
        "official_split": True,
        "pretty_name": "N-MNIST",
    },
    "dvsgesture": {
        "class": tonic.datasets.DVSGesture,
        "class_url_overrides": {
            "train_url": DVSGESTURE_TRAIN_DOWNLOAD_URL,
            "test_url": DVSGESTURE_TEST_DOWNLOAD_URL,
        },
        "sensor_size": tonic.datasets.DVSGesture.sensor_size,
        "num_classes": 11,
        "official_split": True,
        "pretty_name": "DVS128 Gesture",
    },
    "ncaltech101": {
        "class": tonic.datasets.NCALTECH101,
        "sensor_size": (240, 180, 2),
        "num_classes": 101,
        "official_split": False,
        "pretty_name": "N-Caltech101",
        "resize_to": (128, 128),
        "map_targets": True,
    },
    "cifar10dvs": {
        "class": tonic.datasets.CIFAR10DVS,
        "class_url_overrides": {
            "url": CIFAR10DVS_DOWNLOAD_URL,
        },
        "sensor_size": tonic.datasets.CIFAR10DVS.sensor_size,
        "num_classes": 10,
        "official_split": False,
        "pretty_name": "CIFAR10-DVS",
    },
}

EventEncoding: TypeAlias = np.ndarray | FilterResult
EventEncoder: TypeAlias = Callable[[np.ndarray], EventEncoding]
Tensorizer: TypeAlias = Callable[[EventEncoding], "torch.Tensor"]


@dataclass(frozen=True)
class DatasetBundle:
    metadata: DatasetMetadata
    train_raw: Dataset
    val_raw: Dataset
    test_raw: Dataset
    slicing: SlicingConfig


class NormalizedEventDataset(Dataset):
    def __init__(self, base_dataset: Dataset, sensor_size: tuple[int, int, int], resize_to: tuple[int, int] | None = None):
        self.base_dataset = base_dataset
        self.sensor_size = sensor_size
        self.resize_to = resize_to

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        events, label = self.base_dataset[index]
        normalized = normalize_events(events)
        if self.resize_to is not None:
            normalized = resize_events_with_padding(normalized, self.sensor_size, self.resize_to)
        return normalized, int(label)


class EncodedEventDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        encoder: EventEncoder,
        tensorizer: Tensorizer,
        sensor_size: tuple[int, int, int],
        dataset_name: str,
        time_bin_us: int,
        train_mode: bool,
        seed: int,
    ):
        self.base_dataset = base_dataset
        self.encoder = encoder
        self.tensorizer = tensorizer
        self.sensor_size = sensor_size
        self.dataset_name = dataset_name
        self.time_bin_us = time_bin_us
        self.train_mode = train_mode
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        events, label = self.base_dataset[index]
        if self.train_mode:
            rng = np.random.default_rng(self.seed + index * 9973)
            events = apply_train_augmentation(
                events=events,
                sensor_size=self.sensor_size,
                dataset_name=self.dataset_name,
                time_bin_us=self.time_bin_us,
                rng=rng,
            )
        encoded = self.encoder(events)
        return self.tensorizer(encoded), int(label)


class CorruptedEventDataset(Dataset):
    _SOURCE_OFFSETS = {
        "ba": 13,
        "shot": 17,
        "mixed": 19,
    }

    def __init__(
        self,
        base_dataset: Dataset,
        sensor_size: tuple[int, int, int],
        source: str,
        ratio: float,
        tau_pair_us: int,
        seed: int,
    ):
        if source not in self._SOURCE_OFFSETS:
            raise KeyError(f"Unsupported corruption source: {source}")
        self.base_dataset = base_dataset
        self.sensor_size = sensor_size
        self.source = source
        self.ratio = ratio
        self.tau_pair_us = tau_pair_us
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _sample_seed(self, index: int) -> int:
        return self.seed + index * self._SOURCE_OFFSETS[self.source]

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        events, label = self.base_dataset[index]
        sample_seed = self._sample_seed(index)
        if self.source == "ba":
            corrupted = inject_ba_noise(events, self.sensor_size, ratio=self.ratio, seed=sample_seed)
        elif self.source == "shot":
            corrupted = inject_shot_noise(
                events,
                self.sensor_size,
                ratio=self.ratio,
                tau_pair_us=self.tau_pair_us,
                seed=sample_seed,
            )
        else:
            corrupted = inject_mixed_noise(
                events,
                self.sensor_size,
                ratio=self.ratio,
                tau_pair_us=self.tau_pair_us,
                seed=sample_seed,
            )
        return corrupted.events, int(label)


class TargetMappedDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        indices: list[int],
        target_to_index: dict[Any, int],
    ):
        self.base_dataset = base_dataset
        self.indices = indices
        self.target_to_index = target_to_index
        self.targets = [target_to_index[_target_at_index(base_dataset, index)] for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        base_index = self.indices[index]
        events, target = self.base_dataset[base_index]
        return events, self.target_to_index[target]


def resize_events_with_padding(
    events: np.ndarray,
    sensor_size: tuple[int, int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    if len(events) == 0:
        return events
    src_w, src_h, _ = sensor_size
    dst_w, dst_h = target_size
    scale = min(dst_w / src_w, dst_h / src_h)
    scaled_w = src_w * scale
    scaled_h = src_h * scale
    offset_x = (dst_w - scaled_w) / 2.0
    offset_y = (dst_h - scaled_h) / 2.0

    resized = events.copy()
    resized[:, EVENT_X] = np.rint(resized[:, EVENT_X] * scale + offset_x).astype(np.int64)
    resized[:, EVENT_Y] = np.rint(resized[:, EVENT_Y] * scale + offset_y).astype(np.int64)
    resized[:, EVENT_X] = np.clip(resized[:, EVENT_X], 0, dst_w - 1)
    resized[:, EVENT_Y] = np.clip(resized[:, EVENT_Y], 0, dst_h - 1)
    return resized


def _split_cache_path(dataset_name: str, split_seed: int, root: Path) -> Path:
    return root / "results" / "splits" / f"{dataset_name}_seed{split_seed}.json"


def _target_at_index(dataset: Dataset, index: int) -> Any:
    if isinstance(dataset, Subset):
        return _target_at_index(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, NormalizedEventDataset):
        return _target_at_index(dataset.base_dataset, index)
    if isinstance(dataset, _ConcatDataset):
        if index < dataset.offsets[1]:
            return _target_at_index(dataset.datasets[0], index)
        return _target_at_index(dataset.datasets[1], index - dataset.offsets[1])
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return targets[index]
    return dataset[index][1]


def _extract_labels(dataset: Dataset) -> np.ndarray:
    labels = [int(_target_at_index(dataset, index)) for index in range(len(dataset))]
    return np.asarray(labels, dtype=np.int64)


def _apply_target_policy(dataset: Dataset, spec: dict[str, Any]) -> Dataset:
    excluded_targets = set(spec.get("exclude_targets", ()))
    if not excluded_targets and not spec.get("map_targets", False):
        return dataset

    raw_targets = [_target_at_index(dataset, index) for index in range(len(dataset))]
    included_targets = [target for target in raw_targets if target not in excluded_targets]
    target_to_index = {target: index for index, target in enumerate(sorted(set(included_targets)))}

    expected_num_classes = spec["num_classes"]
    if len(target_to_index) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes after target mapping, "
            f"found {len(target_to_index)}."
        )

    indices = [index for index, target in enumerate(raw_targets) if target not in excluded_targets]
    return TargetMappedDataset(dataset, indices, target_to_index)


def _materialize_split_indices(
    dataset_name: str,
    labels: np.ndarray,
    split_seed: int,
    root: Path,
    official_train_len: int | None = None,
) -> dict[str, list[int]]:
    cache_path = _split_cache_path(dataset_name, split_seed, root)
    if cache_path.exists():
        return load_json(cache_path)

    indices = np.arange(len(labels))
    if official_train_len is not None:
        train_indices = indices[:official_train_len]
        test_indices = indices[official_train_len:]
        train_labels = labels[train_indices]
        split_train, split_val = train_test_split(
            train_indices,
            test_size=0.1,
            random_state=split_seed,
            stratify=train_labels,
        )
        payload = {
            "train": split_train.tolist(),
            "val": split_val.tolist(),
            "test": test_indices.tolist(),
        }
    else:
        train_val_indices, test_indices, train_val_labels, _ = train_test_split(
            indices,
            labels,
            test_size=0.1,
            random_state=split_seed,
            stratify=labels,
        )
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=1.0 / 9.0,
            random_state=split_seed,
            stratify=train_val_labels,
        )
        payload = {
            "train": train_indices.tolist(),
            "val": val_indices.tolist(),
            "test": test_indices.tolist(),
        }
    save_json(cache_path, payload)
    return payload


def _build_raw_datasets(dataset_name: str, root: Path) -> tuple[Dataset, Dataset | None]:
    spec = DATASET_REGISTRY[dataset_name]
    dataset_class = spec["class"]
    # Tonic 1.6.0 uses figshare.com/ndownloader for some datasets, which can
    # return an empty AWS WAF challenge response to urllib. Figshare's API
    # reports these direct ndownloader URLs for the same files and checksums.
    for attr_name, download_url in spec.get("class_url_overrides", {}).items():
        setattr(dataset_class, attr_name, download_url)
    dataset_root = root / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    if spec["official_split"]:
        train_raw = _apply_target_policy(dataset_class(save_to=dataset_root, train=True), spec)
        test_raw = _apply_target_policy(dataset_class(save_to=dataset_root, train=False), spec)
        return train_raw, test_raw
    return _apply_target_policy(dataset_class(save_to=dataset_root), spec), None


def _normalized_dataset(
    base_dataset: Dataset,
    sensor_size: tuple[int, int, int],
    resize_to: tuple[int, int] | None,
) -> NormalizedEventDataset:
    return NormalizedEventDataset(
        base_dataset=base_dataset,
        sensor_size=sensor_size,
        resize_to=resize_to,
    )


def _apply_split(dataset: Dataset, split_payload: dict[str, list[int]]) -> tuple[Subset, Subset, Subset]:
    return (
        Subset(dataset, split_payload["train"]),
        Subset(dataset, split_payload["val"]),
        Subset(dataset, split_payload["test"]),
    )


def _metadata_sensor_size(
    sensor_size: tuple[int, int, int],
    resize_to: tuple[int, int] | None,
) -> tuple[int, int, int]:
    if resize_to is None:
        return sensor_size
    return resize_to[0], resize_to[1], sensor_size[2]


def prepare_dataset_bundle(dataset_name: str, root: Path, split_seed: int = 2027) -> DatasetBundle:
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {dataset_name}")

    bundle_start = perf_counter()
    spec = DATASET_REGISTRY[dataset_name]
    LOGGER.info("Building raw datasets: dataset=%s", dataset_name)
    raw_train, raw_test = _build_raw_datasets(dataset_name, root)
    sensor_size = spec["sensor_size"]
    resize_to = spec.get("resize_to")
    LOGGER.info(
        "Raw datasets ready: dataset=%s train=%d test=%s",
        dataset_name,
        len(raw_train),
        "none" if raw_test is None else len(raw_test),
    )

    if raw_test is not None:
        train_labels = _extract_labels(raw_train)
        LOGGER.info("Materializing official split indices: dataset=%s", dataset_name)
        split_payload = _materialize_split_indices(
            dataset_name=dataset_name,
            labels=np.concatenate([train_labels, _extract_labels(raw_test)]),
            split_seed=split_seed,
            root=root,
            official_train_len=len(raw_train),
        )
        full_dataset = _normalized_dataset(_ConcatDataset((raw_train, raw_test)), sensor_size, resize_to)
        train_raw, val_raw, test_raw = _apply_split(full_dataset, split_payload)
    else:
        labels = _extract_labels(raw_train)
        LOGGER.info("Materializing stratified split indices: dataset=%s", dataset_name)
        split_payload = _materialize_split_indices(
            dataset_name=dataset_name,
            labels=labels,
            split_seed=split_seed,
            root=root,
        )
        full_dataset = _normalized_dataset(raw_train, sensor_size, resize_to)
        train_raw, val_raw, test_raw = _apply_split(full_dataset, split_payload)

    LOGGER.info("Computing slicing durations: dataset=%s train_samples=%d", dataset_name, len(train_raw))
    durations: list[int] = []
    duration_start = perf_counter()
    duration_log_interval = max(1, len(train_raw) // 10)
    for index in range(len(train_raw)):
        durations.append(sample_duration_us(train_raw[index][0]))
        if (index + 1) % duration_log_interval == 0 or index + 1 == len(train_raw):
            LOGGER.info(
                "Slicing duration scan: dataset=%s sample=%d/%d elapsed=%.1fs",
                dataset_name,
                index + 1,
                len(train_raw),
                perf_counter() - duration_start,
            )
    slicing = select_time_bin_us(durations)
    metadata = DatasetMetadata(
        name=dataset_name,
        sensor_size=_metadata_sensor_size(sensor_size, resize_to),
        num_classes=spec["num_classes"],
        split_policy="official+val" if spec["official_split"] else "stratified_80_10_10",
        root=root / "dataset",
        resize_to=resize_to,
    )
    LOGGER.info(
        "Dataset bundle complete: dataset=%s train=%d val=%d test=%d slicing=(time_bin_us=%d, t_max=%d) elapsed=%.1fs",
        dataset_name,
        len(train_raw),
        len(val_raw),
        len(test_raw),
        slicing.time_bin_us,
        slicing.t_max,
        perf_counter() - bundle_start,
    )
    return DatasetBundle(
        metadata=metadata,
        train_raw=train_raw,
        val_raw=val_raw,
        test_raw=test_raw,
        slicing=slicing,
    )


def make_calibration_subset(dataset: Dataset, seed: int = 2027, fraction: float = 0.1, cap: int = 1000) -> Subset:
    labels = _extract_labels(dataset)
    indices = np.arange(len(dataset))
    target_size = min(max(int(round(len(dataset) * fraction)), 1), cap, len(dataset))
    if target_size == len(dataset):
        return Subset(dataset, indices.tolist())
    selected_indices, _ = train_test_split(
        indices,
        test_size=len(dataset) - target_size,
        random_state=seed,
        stratify=labels,
    )
    return Subset(dataset, selected_indices.tolist())


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    train_mode: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    effective_batch_size = min(batch_size, max(len(dataset), 1))
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=train_mode,
        drop_last=train_mode and len(dataset) >= effective_batch_size and len(dataset) > 1,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )


class _ConcatDataset(Dataset):
    def __init__(self, datasets: tuple[Dataset, Dataset]):
        self.datasets = datasets
        self.offsets = [0, len(datasets[0]), len(datasets[0]) + len(datasets[1])]

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int):
        if index < self.offsets[1]:
            return self.datasets[0][index]
        return self.datasets[1][index - self.offsets[1]]
