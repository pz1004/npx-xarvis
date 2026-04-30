from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SENSOR_SIZE = tuple[int, int, int]


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    sensor_size: SENSOR_SIZE
    num_classes: int
    split_policy: str
    root: Path
    resize_to: tuple[int, int] | None = None


@dataclass(frozen=True)
class SlicingConfig:
    time_bin_us: int
    t_max: int


@dataclass(frozen=True)
class CorruptedEvents:
    events: np.ndarray
    is_signal: np.ndarray
    source: str
    ratio: float


@dataclass(frozen=True)
class FilterResult:
    accepted_mask: np.ndarray
    confidence: np.ndarray
    support: np.ndarray
    pair_flag: np.ndarray
    output_events: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodConfig:
    name: str
    family: str
    filter_params: dict[str, Any]
    uses_confidence: bool
    frame_mode: bool = False
    stage_variant: str = "conf"
    profile_only: bool = False
