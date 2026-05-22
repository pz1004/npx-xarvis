from __future__ import annotations

import itertools
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from src.common import FilterResult, MethodConfig
from src.data.event_io import normalize_events
from src.data.slicing import output_events_to_event_tensor, raw_events_to_frame_tensor
from src.filters.ba_filter import BAFilter, BAFilterConfig
from src.filters._shared import stack_output_events
from src.filters.proposed_balanced import ProposedBalancedConfig, ProposedBalancedFilter
from src.filters.proposed_lowmem import ProposedLowMemConfig, ProposedLowMemFilter
from src.filters.stcf_rc import STCFRCConfig, STCFRCFilter
from src.stage1 import Stage1Config
from src.utils.serialization import load_json


RESULTS_ROOT = Path("results") / "paper"
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"
EventEncoding: TypeAlias = np.ndarray | FilterResult
EventEncoder: TypeAlias = Callable[[np.ndarray], EventEncoding]
Tensorizer: TypeAlias = Callable[[EventEncoding], torch.Tensor]
FilterApply: TypeAlias = Callable[[np.ndarray], FilterResult]


def _load_config(root: Path, category: str, name: str) -> dict[str, Any]:
    return load_json(root / "configs" / category / f"{name}.json")


def load_training_config(root: Path, name: str = "default") -> dict[str, Any]:
    return _load_config(root, "training", name)


def load_dataset_config(root: Path, dataset_name: str) -> dict[str, Any]:
    return _load_config(root, "datasets", dataset_name)


def load_stage1_config(root: Path, name: str = "default") -> Stage1Config:
    payload = _load_config(root, "stage1", name)
    return Stage1Config(
        mode=payload.get("mode", "offline_noop"),
        nominal_bias=payload.get("nominal_bias", {}),
        low_light_bias=payload.get("low_light_bias", {}),
        calibration_stats=payload.get("calibration_stats", {}),
        manifest_name=name,
    )


def load_method_config(root: Path, method_name: str) -> MethodConfig:
    payload = _load_config(root, "methods", method_name)
    return MethodConfig(
        name=method_name,
        family=payload["family"],
        filter_params=payload.get("filter_params", {}),
        uses_confidence=payload.get("uses_confidence", False),
        frame_mode=payload.get("frame_mode", False),
        stage_variant=payload.get("stage_variant", "conf"),
        profile_only=payload.get("profile_only", False),
    )


def make_run_timestamp() -> str:
    return datetime.now().strftime(RUN_TIMESTAMP_FORMAT)


def result_dir(root: Path, dataset_name: str, method_name: str, seed: int, run_timestamp: str | None = None) -> Path:
    seed_dir = f"seed_{seed}" if run_timestamp is None else f"seed_{seed}__{run_timestamp}"
    return root / RESULTS_ROOT / dataset_name / method_name / seed_dir


def latest_result_dir(root: Path, dataset_name: str, method_name: str, seed: int) -> Path:
    method_dir = root / RESULTS_ROOT / dataset_name / method_name
    timestamped = sorted(method_dir.glob(f"seed_{seed}__*"))
    if timestamped:
        return timestamped[-1]
    return result_dir(root, dataset_name, method_name, seed)


def tuning_result_dir(root: Path, dataset_name: str, method_name: str, run_timestamp: str) -> Path:
    return root / RESULTS_ROOT / dataset_name / method_name / f"tuning_{run_timestamp}"


def latest_tuning_summary_path(root: Path, dataset_name: str, method_name: str) -> Path | None:
    method_dir = root / RESULTS_ROOT / dataset_name / method_name
    timestamped = sorted(method_dir.glob("tuning_*/tuning_summary.json"))
    if timestamped:
        return timestamped[-1]
    legacy = method_dir / "tuning_summary.json"
    return legacy if legacy.exists() else None


def latest_tuning_filter_params(root: Path, dataset_name: str, method_name: str) -> tuple[dict[str, Any], Path] | None:
    tuning_path = latest_tuning_summary_path(root, dataset_name, method_name)
    if tuning_path is None:
        return None
    tuning_summary = load_json(tuning_path)
    selected = tuning_summary["selected"]
    if "filter_params" not in selected:
        return None
    return dict(selected["filter_params"]), tuning_path


def resolve_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def filter_score(result: FilterResult) -> np.ndarray:
    score_mode = result.stats.get("score_mode", "proposed")
    if score_mode == "proposed":
        return result.support.astype(np.float64) + 0.1 * result.confidence.astype(np.float64)
    return result.accepted_mask.astype(np.float64)


def _raw_filter(events: np.ndarray) -> FilterResult:
    events = normalize_events(events)
    accepted_mask = np.ones(len(events), dtype=bool)
    confidence = np.ones(len(events), dtype=np.int8)
    support = np.zeros(len(events), dtype=np.int16)
    pair_flag = np.zeros(len(events), dtype=np.int8)
    return FilterResult(
        accepted_mask=accepted_mask,
        confidence=confidence,
        support=support,
        pair_flag=pair_flag,
        output_events=stack_output_events(events, confidence),
        stats={"accepted_count": int(len(events)), "score_mode": "binary"},
    )


def _resolve_filter_params(
    method: MethodConfig,
    sensor_size: tuple[int, int, int],
    filter_params: dict[str, Any],
) -> FilterApply | None:
    family = method.family
    if family == "raw":
        return _raw_filter
    if family == "frame":
        return None
    if family == "ba":
        filter_instance = BAFilter(sensor_size=sensor_size, config=BAFilterConfig(**filter_params))
        return filter_instance.apply
    if family == "stcf_rc":
        filter_instance = STCFRCFilter(sensor_size=sensor_size, config=STCFRCConfig(**filter_params))
        return filter_instance.apply
    if family == "proposed_balanced":
        filter_instance = ProposedBalancedFilter(
            sensor_size=sensor_size,
            config=ProposedBalancedConfig(stage_variant=method.stage_variant, **filter_params),
        )
        return filter_instance.apply
    if family == "proposed_lowmem":
        filter_instance = ProposedLowMemFilter(sensor_size=sensor_size, config=ProposedLowMemConfig(**filter_params))
        return filter_instance.apply
    if family == "proposed_lowlat":
        return _raw_filter
    raise KeyError(f"Unsupported method family: {family}")


def _frame_tensorizer(sensor_size: tuple[int, int, int], slicing_config) -> Tensorizer:
    def tensorize(events: EventEncoding) -> torch.Tensor:
        if not isinstance(events, np.ndarray):
            raise TypeError(f"Expected raw numpy events for frame tensorizer, got {type(events)!r}")
        return raw_events_to_frame_tensor(events, sensor_size=sensor_size, slicing=slicing_config)

    return tensorize


def _filtered_tensorizer(sensor_size: tuple[int, int, int], slicing_config) -> Tensorizer:
    def tensorize(result: EventEncoding) -> torch.Tensor:
        if not isinstance(result, FilterResult):
            raise TypeError(f"Expected FilterResult for filtered tensorizer, got {type(result)!r}")
        return output_events_to_event_tensor(result.output_events, sensor_size=sensor_size, slicing=slicing_config)

    return tensorize


def build_method_pipeline(
    method: MethodConfig,
    sensor_size: tuple[int, int, int],
    slicing_config,
    filter_params: dict[str, Any] | None = None,
) -> tuple[EventEncoder, Tensorizer, FilterApply | None]:
    resolved_filter_params = filter_params or dict(method.filter_params)
    filter_apply = _resolve_filter_params(method, sensor_size, resolved_filter_params)
    if method.frame_mode:
        return (
            normalize_events,
            _frame_tensorizer(sensor_size, slicing_config),
            None,
        )
    assert filter_apply is not None
    return (
        filter_apply,
        _filtered_tensorizer(sensor_size, slicing_config),
        filter_apply,
    )


def calibration_filter_params(
    method: MethodConfig,
    calibration_events: list[np.ndarray],
    _sensor_size: tuple[int, int, int],
) -> dict[str, Any]:
    filter_params = dict(method.filter_params)
    if method.family == "proposed_balanced" and filter_params.get("r_max_hz") == "auto":
        filter_params["r_max_hz"] = ProposedBalancedFilter.estimate_rmax_hz(calibration_events)
    return filter_params


def candidate_param_grid(method_name: str, preset: str = "paper") -> list[dict[str, Any]]:
    if preset not in {"fast", "paper"}:
        raise ValueError(f"Unsupported candidate grid preset: {preset}")
    if method_name in {"raw_snn", "frame_snn"}:
        return [{}]
    if method_name in {"ba_snn", "stcf_rc_snn"}:
        return [{"delta_t_us": delta_t_us} for delta_t_us in (1000, 2000, 5000)]
    if method_name == "proposed_ref":
        return [{"tau_ref_dig_us": tau_ref} for tau_ref in (500, 1000, 2000)]
    if preset == "fast" and method_name == "proposed_sup":
        return [
            {
                "tau_ref_dig_us": tau_ref,
                "delta_t_us": delta_t_us,
                "k0": k0,
                "gamma": 0,
                "tau_pair_us": 1000,
                "k_high": 2,
                "t_recover_us": 1_000_000,
                "warmup_us": 5_000,
            }
            for tau_ref, delta_t_us, k0 in itertools.product(
                (500, 1000, 2000),
                (1000, 2000, 5000),
                (1, 2),
            )
        ]
    if preset == "fast" and method_name in {"proposed_pol", "proposed_conf"}:
        return [
            {
                "tau_ref_dig_us": tau_ref,
                "delta_t_us": delta_t_us,
                "k0": k0,
                "gamma": gamma,
                "tau_pair_us": 1000,
                "k_high": k_high,
                "t_recover_us": 1_000_000,
                "warmup_us": 5_000,
            }
            for tau_ref, delta_t_us, k0, gamma, k_high in itertools.product(
                (500, 1000, 2000),
                (1000, 2000, 5000),
                (1, 2),
                (0, 1),
                (2, 3),
            )
        ]
    if method_name in {"proposed_sup", "proposed_pol", "proposed_conf"}:
        grid: list[dict[str, Any]] = []
        for tau_ref, delta_t_us, k0, gamma, tau_pair_us, k_high, t_recover_us in itertools.product(
            (500, 1000, 2000),
            (1000, 2000, 5000),
            (1, 2),
            (0, 1, 2),
            (500, 1000, 2000),
            (2, 3, 4),
            (500_000, 1_000_000, 2_000_000),
        ):
            grid.append(
                {
                    "tau_ref_dig_us": tau_ref,
                    "delta_t_us": delta_t_us,
                    "k0": k0,
                    "gamma": gamma,
                    "tau_pair_us": tau_pair_us,
                    "k_high": k_high,
                    "t_recover_us": t_recover_us,
                    "warmup_us": 5_000,
                }
            )
        return grid
    if method_name == "proposed_lowmem":
        if preset == "fast":
            return [
                {
                    "tau_ref_dig_us": tau_ref,
                    "delta_t_us": delta_t_us,
                    "k0": k0,
                    "k_high": k_high,
                    "warmup_us": 5_000,
                }
                for tau_ref, delta_t_us, k0, k_high in itertools.product(
                    (500, 1000, 2000),
                    (1000, 2000, 5000),
                    (1, 2),
                    (2, 3),
                )
            ]
        return [
            {
                "tau_ref_dig_us": tau_ref,
                "delta_t_us": delta_t_us,
                "k0": k0,
                "k_high": k_high,
                "warmup_us": 5_000,
            }
            for tau_ref, delta_t_us, k0, k_high in itertools.product(
                (500, 1000, 2000),
                (1000, 2000, 5000),
                (1, 2),
                (2, 3, 4),
            )
        ]
    if method_name == "proposed_lowlat":
        return [{}]
    raise KeyError(f"Unsupported method for grid search: {method_name}")


def subset_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    indices = list(range(max_samples))
    return Subset(dataset, indices)
