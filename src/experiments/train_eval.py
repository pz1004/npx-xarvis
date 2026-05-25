from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.datasets import (
    CorruptedEventDataset,
    EncodedEventDataset,
    build_dataloader,
    make_calibration_subset,
    prepare_dataset_bundle,
)
from src.experiments.common import (
    EventEncoder,
    Tensorizer,
    build_method_pipeline,
    calibration_filter_params,
    latest_tuning_filter_params,
    load_dataset_config,
    load_method_config,
    load_stage1_config,
    load_training_config,
    make_run_timestamp,
    resolve_device,
    result_dir,
    subset_dataset,
)
from src.filters.metrics import area_under_noise_curve, compression_ratio, event_keep_ratio, state_memory_bytes
from src.models.event_snn import BackboneConfig, EventSNN, spike_accuracy, spike_rate_cross_entropy
from src.models.frame_snn import FrameSNN
from src.models.sop_counter import (
    count_dense_macs,
    count_event_sops,
    measure_peak_activation_bytes,
    parameter_count,
    parameter_memory_bytes,
)
from src.stage1 import run_stage1
from src.utils.logging import get_logger, setup_logging
from src.utils.seed import set_seed
from src.utils.serialization import save_json


LOGGER = get_logger(__name__)
ROBUSTNESS_SOURCES = ("ba", "shot", "mixed")
ROBUSTNESS_BASE_SEED = 100_000
DENSE_INPUT_BUDGET_CHANNELS = 4
FLOAT32_BYTES = 4
DEFAULT_CPU_DENSE_BATCH_BYTES = 64 * 1024 * 1024
DEFAULT_CUDA_DENSE_BATCH_BYTES = 256 * 1024 * 1024


def _transpose_batch(batch: torch.Tensor) -> torch.Tensor:
    return batch.permute(1, 0, 2, 3, 4).contiguous()


def _dense_sample_bytes(
    sensor_size: tuple[int, int, int],
    t_max: int,
    channels: int = DENSE_INPUT_BUDGET_CHANNELS,
) -> int:
    width, height, _ = sensor_size
    return int(max(t_max, 1) * channels * height * width * FLOAT32_BYTES)


def _cap_batch_size_for_dense_input(
    configured_batch_size: int,
    sensor_size: tuple[int, int, int],
    t_max: int,
    max_batch_bytes: int,
) -> int:
    if configured_batch_size <= 0:
        raise ValueError("Configured batch_size must be positive")
    if max_batch_bytes <= 0:
        return configured_batch_size
    sample_bytes = _dense_sample_bytes(sensor_size=sensor_size, t_max=t_max)
    max_by_input = max(1, max_batch_bytes // max(sample_bytes, 1))
    return min(configured_batch_size, int(max_by_input))


def _dense_batch_budget_bytes(training_config: dict[str, Any], device: torch.device) -> int:
    shared_budget = training_config.get("max_dense_batch_bytes")
    if device.type == "cuda":
        return int(
            training_config.get(
                "max_cuda_dense_batch_bytes",
                shared_budget if shared_budget is not None else DEFAULT_CUDA_DENSE_BATCH_BYTES,
            )
        )
    return int(
        training_config.get(
            "max_cpu_dense_batch_bytes",
            shared_budget if shared_budget is not None else DEFAULT_CPU_DENSE_BATCH_BYTES,
        )
    )


def _build_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if total_epochs <= 1 or total_epochs <= warmup_epochs:
            return 1.0
        if epoch <= warmup_epochs:
            return 1.0
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _resolve_dataloader_settings(training_config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    configured_workers = training_config.get("dataloader_num_workers")
    if configured_workers is None:
        num_workers = min(4, os.cpu_count() or 1) if device.type == "cuda" else 0
    else:
        num_workers = max(0, int(configured_workers))
    pin_memory = bool(training_config.get("dataloader_pin_memory", device.type == "cuda"))
    if device.type != "cuda":
        pin_memory = False
    persistent_workers = bool(training_config.get("dataloader_persistent_workers", num_workers > 0))
    if num_workers == 0:
        persistent_workers = False
    prefetch_factor = None
    if num_workers > 0:
        prefetch_factor = int(training_config.get("dataloader_prefetch_factor", 2))
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
    }


def _zero_preprocessing_metrics() -> dict[str, float]:
    return {
        "accepted_event_ratio": 0.0,
        "compression_ratio": 0.0,
        "preprocessing_latency_sec": 0.0,
        "preprocessing_throughput_eps": 0.0,
    }


def _evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_items = 0
    with torch.no_grad():
        for batch, targets in data_loader:
            batch = _transpose_batch(batch).to(device)
            targets = targets.to(device)
            spike_record = model(batch)
            batch_size = targets.size(0)
            total_loss += float(spike_rate_cross_entropy(spike_record, targets).item()) * batch_size
            total_acc += spike_accuracy(spike_record, targets) * batch_size
            total_items += batch_size
    if total_items == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {
        "loss": total_loss / total_items,
        "accuracy": total_acc / total_items,
    }


def _measure_preprocessing(
    raw_dataset: Dataset,
    encoder: EventEncoder,
    tensorizer: Tensorizer,
    max_samples: int,
    warmup: int = 50,
) -> dict[str, float]:
    limit = min(len(raw_dataset), max_samples)
    if limit == 0:
        return _zero_preprocessing_metrics()
    warmup = min(warmup, max(0, limit // 4))
    accepted_ratios: list[float] = []
    compression_ratios: list[float] = []
    elapsed: list[float] = []
    total_events = 0

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for index in range(limit):
            events, _ = raw_dataset[index]
            start = time.perf_counter()
            encoded = encoder(events)
            _ = tensorizer(encoded)
            duration = time.perf_counter() - start
            if hasattr(encoded, "accepted_mask"):
                accepted_ratios.append(event_keep_ratio(encoded.accepted_mask))
                compression_ratios.append(compression_ratio(encoded.accepted_mask))
            else:
                accepted_ratios.append(1.0)
                compression_ratios.append(1.0)
            if index >= warmup:
                elapsed.append(duration)
                total_events += len(events)
    finally:
        torch.set_num_threads(previous_threads)

    median_latency = float(torch.median(torch.tensor(elapsed)).item()) if elapsed else 0.0
    throughput = float(total_events / max(sum(elapsed), 1e-12)) if elapsed else 0.0
    return {
        "accepted_event_ratio": float(sum(accepted_ratios) / max(len(accepted_ratios), 1)),
        "compression_ratio": float(sum(compression_ratios) / max(len(compression_ratios), 1)),
        "preprocessing_latency_sec": median_latency,
        "preprocessing_throughput_eps": throughput,
    }


def _extract_sample_batch(data_loader: DataLoader) -> torch.Tensor | None:
    for batch, _targets in data_loader:
        if batch.size(0) == 0:
            continue
        return _transpose_batch(batch[:1])
    return None


def _measure_inference_latency(
    model: nn.Module,
    sample_batch: torch.Tensor | None,
    device: torch.device,
    warmup: int = 20,
    timed: int = 100,
) -> float:
    if sample_batch is None:
        return 0.0
    timings: list[float] = []
    sample_batch = sample_batch.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_batch)
        for _ in range(timed):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            _ = model(sample_batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
    return float(torch.median(torch.tensor(timings)).item()) if timings else 0.0


def _profile_model(model: nn.Module, sample_batch: torch.Tensor | None, frame_mode: bool) -> dict[str, int]:
    if sample_batch is None:
        compute = 0
        peak_activation = 0
    else:
        compute = count_dense_macs(model, sample_batch) if frame_mode else count_event_sops(model, sample_batch)
        peak_activation = measure_peak_activation_bytes(model, sample_batch)
    return {
        "parameter_count": parameter_count(model),
        "parameter_memory_bytes": parameter_memory_bytes(model),
        "peak_activation_bytes": peak_activation,
        "dense_macs": int(compute) if frame_mode else 0,
        "sops": 0 if frame_mode else int(compute),
    }


def _build_encoded_dataset(
    base_dataset: Dataset,
    encoder: EventEncoder,
    tensorizer: Tensorizer,
    sensor_size: tuple[int, int, int],
    dataset_name: str,
    time_bin_us: int,
    train_mode: bool,
    seed: int,
) -> EncodedEventDataset:
    return EncodedEventDataset(
        base_dataset=base_dataset,
        encoder=encoder,
        tensorizer=tensorizer,
        sensor_size=sensor_size,
        dataset_name=dataset_name,
        time_bin_us=time_bin_us,
        train_mode=train_mode,
        seed=seed,
    )


def _build_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    seed: int,
    loader_settings: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    return (
        build_dataloader(train_dataset, batch_size=batch_size, train_mode=True, seed=seed, **loader_settings),
        build_dataloader(val_dataset, batch_size=batch_size, train_mode=False, seed=seed, **loader_settings),
        build_dataloader(test_dataset, batch_size=batch_size, train_mode=False, seed=seed, **loader_settings),
    )


def _build_model(num_classes: int, frame_mode: bool) -> nn.Module:
    if frame_mode:
        return FrameSNN(num_classes=num_classes)
    return EventSNN(BackboneConfig(num_classes=num_classes))


def _train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
    progress_label: str,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    total_items = 0
    for batch, targets in tqdm(train_loader, desc=progress_label, leave=False):
        batch = _transpose_batch(batch).to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        spike_record = model(batch)
        loss = spike_rate_cross_entropy(spike_record, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()
        running_loss += float(loss.item()) * targets.size(0)
        total_items += targets.size(0)
    return {"loss": running_loss / max(total_items, 1)}


def _calibration_subset_info(dataset: Dataset, split_seed: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    calibration_subset = make_calibration_subset(dataset, seed=split_seed)
    calibration_events = [calibration_subset[index][0] for index in range(len(calibration_subset))]
    return calibration_events, {
        "strategy": "stratified_fraction",
        "seed": split_seed,
        "fraction": 0.1,
        "cap": 1000,
        "size": len(calibration_subset),
    }


def _calibrated_filter_params(
    bundle,
    method_config,
    split_seed: int,
    filter_params_override: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration_events, calibration_info = _calibration_subset_info(bundle.train_raw, split_seed)
    resolved_filter_params = calibration_filter_params(method_config, calibration_events, bundle.metadata.sensor_size)
    if filter_params_override:
        resolved_filter_params.update(filter_params_override)
    return resolved_filter_params, calibration_info


def _resolve_run_purpose(
    result_method_name: str | None,
    method_config,
    epochs_override: int | None,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    force_cpu: bool,
    run_purpose: str | None,
) -> str:
    if run_purpose is not None:
        return run_purpose
    if method_config.profile_only:
        return "paper_profile"
    if result_method_name and "__tune_" in result_method_name:
        return "tuning_stage_b"
    if any(value is not None for value in (epochs_override, max_train_samples, max_val_samples, max_test_samples)) or force_cpu:
        return "custom"
    return "paper_main"


def _software_versions() -> dict[str, str]:
    packages = ("torch", "snntorch", "tonic", "numpy", "pandas", "scikit-learn", "matplotlib", "seaborn")
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _parse_filter_params_json(filter_params_json: str | None) -> dict[str, Any]:
    if filter_params_json is None:
        return {}
    try:
        payload = json.loads(filter_params_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --filter-params JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("--filter-params must be a JSON object")
    return dict(payload)


def _resolve_cli_filter_params(
    *,
    root: Path,
    dataset_name: str,
    method_name: str,
    use_latest_tuning: bool,
    filter_params_json: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    resolved: dict[str, Any] = {}
    source_info: dict[str, Any] = {
        "use_latest_tuning": use_latest_tuning,
        "manual_filter_params": filter_params_json is not None,
        "latest_tuning_path": None,
    }
    if use_latest_tuning:
        latest = latest_tuning_filter_params(root, dataset_name, method_name)
        if latest is None:
            raise FileNotFoundError(
                f"No tuning_summary.json found for dataset={dataset_name} method={method_name}"
            )
        latest_params, tuning_path = latest
        resolved.update(latest_params)
        source_info["latest_tuning_path"] = str(tuning_path)
    manual_params = _parse_filter_params_json(filter_params_json)
    resolved.update(manual_params)
    if not resolved:
        return None, source_info
    source_info["resolved_filter_params"] = resolved
    return resolved, source_info


def _filter_memory_name(method_name: str, family: str) -> str:
    if family == "proposed_balanced":
        return "proposed_balanced"
    return method_name


def _build_corrupted_encoded_dataset(
    raw_dataset: Dataset,
    encoder: EventEncoder,
    tensorizer: Tensorizer,
    sensor_size: tuple[int, int, int],
    dataset_name: str,
    time_bin_us: int,
    source: str,
    ratio: float,
    tau_pair_us: int,
    seed: int,
) -> EncodedEventDataset:
    corrupted_raw = CorruptedEventDataset(
        base_dataset=raw_dataset,
        sensor_size=sensor_size,
        source=source,
        ratio=ratio,
        tau_pair_us=tau_pair_us,
        seed=seed,
    )
    return _build_encoded_dataset(
        base_dataset=corrupted_raw,
        encoder=encoder,
        tensorizer=tensorizer,
        sensor_size=sensor_size,
        dataset_name=dataset_name,
        time_bin_us=time_bin_us,
        train_mode=False,
        seed=seed,
    )


def _evaluate_robustness(
    model: nn.Module,
    raw_dataset: Dataset,
    encoder: EventEncoder,
    tensorizer: Tensorizer,
    bundle,
    dataset_name: str,
    batch_size: int,
    device: torch.device,
    seed: int,
    training_config: dict[str, Any],
    tau_pair_us: int,
    loader_settings: dict[str, Any],
) -> dict[str, Any]:
    ratios = tuple(float(ratio) for ratio in training_config["noise_ratios"])
    results: dict[str, dict[str, Any]] = {}
    for source_index, source in enumerate(ROBUSTNESS_SOURCES):
        accuracies: list[float] = []
        per_ratio: list[dict[str, Any]] = []
        for ratio_index, ratio in enumerate(ratios):
            corruption_seed = ROBUSTNESS_BASE_SEED + seed * 10_000 + source_index * 1_000 + ratio_index * 100
            dataset = _build_corrupted_encoded_dataset(
                raw_dataset=raw_dataset,
                encoder=encoder,
                tensorizer=tensorizer,
                sensor_size=bundle.metadata.sensor_size,
                dataset_name=dataset_name,
                time_bin_us=bundle.slicing.time_bin_us,
                source=source,
                ratio=ratio,
                tau_pair_us=tau_pair_us,
                seed=corruption_seed,
            )
            loader = build_dataloader(dataset, batch_size=batch_size, train_mode=False, seed=seed, **loader_settings)
            accuracy = _evaluate_model(model, loader, device)["accuracy"]
            accuracies.append(accuracy)
            per_ratio.append(
                {
                    "ratio": ratio,
                    "accuracy": accuracy,
                    "seed": corruption_seed,
                }
            )
        results[source] = {
            "ratios": per_ratio,
            "aunc": area_under_noise_curve(accuracies),
        }
    return {
        "sources": results,
        "aunc": float(sum(source["aunc"] for source in results.values()) / max(len(results), 1)),
        "ratios": list(ratios),
        "base_seed": ROBUSTNESS_BASE_SEED + seed * 10_000,
    }


def _make_summary(
    *,
    dataset_name: str,
    method_name: str,
    seed: int,
    device: torch.device,
    bundle,
    resolved_filter_params: dict[str, Any],
    val_metrics: dict[str, float] | None,
    test_metrics: dict[str, float] | None,
    preprocessing_metrics: dict[str, float | None],
    inference_latency_sec: float | None,
    epoch_history: list[dict[str, float]],
    profile_metrics: dict[str, int],
    model: nn.Module | None,
    filter_memory_name: str,
    stage1_summary: dict[str, Any],
    run_metadata: dict[str, Any],
    robustness: dict[str, Any] | None,
) -> dict[str, Any]:
    end_to_end_latency = None
    if preprocessing_metrics["preprocessing_latency_sec"] is not None and inference_latency_sec is not None:
        end_to_end_latency = preprocessing_metrics["preprocessing_latency_sec"] + inference_latency_sec
    return {
        "dataset": dataset_name,
        "method": method_name,
        "seed": seed,
        "run_timestamp": run_metadata.get("run_timestamp"),
        "device": str(device),
        "time_bin_us": bundle.slicing.time_bin_us,
        "t_max": bundle.slicing.t_max,
        "filter_params": resolved_filter_params,
        "val_accuracy": None if val_metrics is None else val_metrics["accuracy"],
        "test_accuracy": None if test_metrics is None else test_metrics["accuracy"],
        "accepted_event_ratio": preprocessing_metrics["accepted_event_ratio"],
        "compression_ratio": preprocessing_metrics["compression_ratio"],
        "preprocessing_latency_sec": preprocessing_metrics["preprocessing_latency_sec"],
        "preprocessing_throughput_eps": preprocessing_metrics["preprocessing_throughput_eps"],
        "inference_latency_sec": inference_latency_sec,
        "end_to_end_latency_sec": end_to_end_latency,
        "filter_state_memory_bytes": state_memory_bytes(filter_memory_name, bundle.metadata.sensor_size, bundle.slicing.t_max),
        "confidence_ratio": float(model.confidence_ratio().detach().cpu().item()) if model is not None and hasattr(model, "confidence_ratio") else None,
        "aunc": None if robustness is None else robustness["aunc"],
        "epoch_history": epoch_history,
        "stage1": stage1_summary,
        "run_metadata": run_metadata,
        **profile_metrics,
    }


def _save_run_artifacts(
    output_dir: Path,
    model: nn.Module | None,
    summary: dict[str, Any],
    robustness: dict[str, Any] | None,
    save_checkpoint: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if model is not None and save_checkpoint:
        torch.save(model.state_dict(), output_dir / "best_model.pt")
    save_json(output_dir / "summary.json", summary)
    if robustness is not None:
        save_json(output_dir / "robustness.json", robustness)
    LOGGER.info("Saved summary to %s", output_dir / "summary.json")


def _analytical_lowlat_summary(
    *,
    root: Path,
    dataset_name: str,
    method_name: str,
    seed: int,
    result_method_name: str | None,
    bundle,
    resolved_filter_params: dict[str, Any],
    device: torch.device,
    stage1_summary: dict[str, Any],
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    summary = _make_summary(
        dataset_name=dataset_name,
        method_name=method_name,
        seed=seed,
        device=device,
        bundle=bundle,
        resolved_filter_params=resolved_filter_params,
        val_metrics=None,
        test_metrics=None,
        preprocessing_metrics={
            "accepted_event_ratio": None,
            "compression_ratio": None,
            "preprocessing_latency_sec": 1e-8,
            "preprocessing_throughput_eps": 100_000_000.0,
        },
        inference_latency_sec=None,
        epoch_history=[],
        profile_metrics={
            "parameter_count": 0,
            "parameter_memory_bytes": 0,
            "peak_activation_bytes": 0,
            "dense_macs": 0,
            "sops": 0,
        },
        model=None,
        filter_memory_name=method_name,
        stage1_summary=stage1_summary,
        run_metadata=run_metadata,
        robustness=None,
    )
    output_dir = result_dir(
        root,
        dataset_name,
        result_method_name or method_name,
        seed,
        run_timestamp=run_metadata.get("run_timestamp"),
    )
    summary["run_metadata"]["output_dir"] = str(output_dir)
    _save_run_artifacts(output_dir, None, summary, robustness=None)
    return summary


def run_training_experiment(
    root: Path,
    dataset_name: str,
    method_name: str,
    seed: int,
    *,
    filter_params_override: dict[str, Any] | None = None,
    filter_params_source: dict[str, Any] | None = None,
    result_method_name: str | None = None,
    epochs_override: int | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    max_slices_override: int | None = None,
    force_cpu: bool = False,
    run_purpose: str | None = None,
    skip_test_evaluation: bool = False,
    skip_posthoc_metrics: bool = False,
    save_checkpoint: bool = True,
    run_timestamp: str | None = None,
) -> dict[str, Any]:
    run_start = time.perf_counter()
    resolved_run_timestamp = run_timestamp or make_run_timestamp()
    set_seed(seed)
    training_config = load_training_config(root)
    dataset_config = load_dataset_config(root, dataset_name)
    stage1_config = load_stage1_config(root)
    method_config = load_method_config(root, method_name)
    device = resolve_device(force_cpu=force_cpu)
    LOGGER.info(
        "Starting training experiment: dataset=%s method=%s seed=%d device=%s cuda_available=%s",
        dataset_name,
        method_name,
        seed,
        device,
        torch.cuda.is_available(),
    )
    resolved_run_purpose = _resolve_run_purpose(
        result_method_name=result_method_name,
        method_config=method_config,
        epochs_override=epochs_override,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        force_cpu=force_cpu,
        run_purpose=run_purpose,
    )

    LOGGER.info("Preparing dataset bundle: dataset=%s split_seed=%s", dataset_name, training_config["split_seed"])
    bundle = prepare_dataset_bundle(
        dataset_name,
        root,
        split_seed=training_config["split_seed"],
        max_slices=int(max_slices_override or 300),
    )
    LOGGER.info(
        "Dataset ready: train=%d val=%d test=%d sensor_size=%s slicing=(time_bin_us=%d, t_max=%d)",
        len(bundle.train_raw),
        len(bundle.val_raw),
        len(bundle.test_raw),
        bundle.metadata.sensor_size,
        bundle.slicing.time_bin_us,
        bundle.slicing.t_max,
    )
    LOGGER.info("Resolving filter parameters from calibration subset")
    resolved_filter_params, calibration_info = _calibrated_filter_params(
        bundle=bundle,
        method_config=method_config,
        split_seed=training_config["split_seed"],
        filter_params_override=filter_params_override,
    )
    LOGGER.info("Filter parameters ready: %s", resolved_filter_params)
    LOGGER.info("Running Stage 1 setup")
    stage1_summary = run_stage1(stage1_config)
    run_metadata = {
        "run_purpose": resolved_run_purpose,
        "run_timestamp": resolved_run_timestamp,
        "is_protocol_run": resolved_run_purpose == "paper_main",
        "effective_epochs": int(epochs_override or dataset_config["epochs"]),
        "sample_caps": {
            "train": max_train_samples,
            "val": max_val_samples,
            "test": max_test_samples,
        },
        "max_slices": int(max_slices_override or 300),
        "evaluation": {
            "skip_test_evaluation": skip_test_evaluation,
            "skip_posthoc_metrics": skip_posthoc_metrics,
            "save_checkpoint": save_checkpoint,
        },
        "filter_params_source": filter_params_source,
        "calibration_subset": calibration_info,
        "software_versions": _software_versions(),
        "hardware": {
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": None if device.type != "cuda" else torch.cuda.get_device_name(),
        },
        "corruption_seed_policy": {
            "event_level_base_seed": training_config["split_seed"],
            "robustness_base_seed": ROBUSTNESS_BASE_SEED + seed * 10_000,
            "source_order": list(ROBUSTNESS_SOURCES),
            "source_stride": 1_000,
            "ratio_stride": 100,
        },
    }

    if method_config.profile_only:
        LOGGER.info("Method is profile-only; producing analytical summary")
        return _analytical_lowlat_summary(
            root=root,
            dataset_name=dataset_name,
            method_name=method_name,
            seed=seed,
            result_method_name=result_method_name,
            bundle=bundle,
            resolved_filter_params=resolved_filter_params,
            device=device,
            stage1_summary=stage1_summary,
            run_metadata=run_metadata,
        )

    encoder, tensorizer, _filter_apply = build_method_pipeline(
        method=method_config,
        sensor_size=bundle.metadata.sensor_size,
        slicing_config=bundle.slicing,
        filter_params=resolved_filter_params,
    )
    LOGGER.info("Method pipeline ready: family=%s frame_mode=%s", method_config.family, method_config.frame_mode)

    train_raw = subset_dataset(bundle.train_raw, max_train_samples)
    val_raw = subset_dataset(bundle.val_raw, max_val_samples)
    test_raw = subset_dataset(bundle.test_raw, max_test_samples)

    train_dataset = _build_encoded_dataset(
        base_dataset=train_raw,
        encoder=encoder,
        tensorizer=tensorizer,
        sensor_size=bundle.metadata.sensor_size,
        dataset_name=dataset_name,
        time_bin_us=bundle.slicing.time_bin_us,
        train_mode=True,
        seed=seed,
    )
    val_dataset = _build_encoded_dataset(
        base_dataset=val_raw,
        encoder=encoder,
        tensorizer=tensorizer,
        sensor_size=bundle.metadata.sensor_size,
        dataset_name=dataset_name,
        time_bin_us=bundle.slicing.time_bin_us,
        train_mode=False,
        seed=seed,
    )
    test_dataset = _build_encoded_dataset(
        base_dataset=test_raw,
        encoder=encoder,
        tensorizer=tensorizer,
        sensor_size=bundle.metadata.sensor_size,
        dataset_name=dataset_name,
        time_bin_us=bundle.slicing.time_bin_us,
        train_mode=False,
        seed=seed,
    )

    configured_batch_size = int(dataset_config["batch_size"])
    dense_batch_budget = _dense_batch_budget_bytes(training_config, device)
    batch_size = _cap_batch_size_for_dense_input(
        configured_batch_size=configured_batch_size,
        sensor_size=bundle.metadata.sensor_size,
        t_max=bundle.slicing.t_max,
        max_batch_bytes=dense_batch_budget,
    )
    if batch_size < configured_batch_size:
        sample_bytes = _dense_sample_bytes(
            sensor_size=bundle.metadata.sensor_size,
            t_max=bundle.slicing.t_max,
        )
        LOGGER.info(
            "Adjusted %s batch_size from %d to %d for dense input budget "
            "(sample=%.1f MiB budget=%.1f MiB)",
            device.type,
            configured_batch_size,
            batch_size,
            sample_bytes / float(1024**2),
            dense_batch_budget / float(1024**2),
        )
    run_metadata["batch_size"] = {
        "configured": configured_batch_size,
        "effective": batch_size,
        "dense_batch_budget_bytes": dense_batch_budget,
        "dense_batch_budget_device": device.type,
        "cuda_dense_batch_budget_bytes": dense_batch_budget if device.type == "cuda" else None,
        "cpu_dense_batch_budget_bytes": dense_batch_budget if device.type == "cpu" else None,
        "dense_input_budget_channels": DENSE_INPUT_BUDGET_CHANNELS,
    }
    epochs = int(epochs_override or dataset_config["epochs"])
    LOGGER.info(
        "Building dataloaders: batch_size=%d configured_batch_size=%d epochs=%d train=%d val=%d test=%d",
        batch_size,
        configured_batch_size,
        epochs,
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )
    loader_settings = _resolve_dataloader_settings(training_config, device)
    run_metadata["dataloader"] = loader_settings
    LOGGER.info("Dataloader settings: %s", loader_settings)
    train_loader, val_loader, test_loader = _build_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        seed=seed,
        loader_settings=loader_settings,
    )

    model = _build_model(num_classes=bundle.metadata.num_classes, frame_mode=method_config.frame_mode)
    model.to(device)
    LOGGER.info("Model initialized on %s: num_classes=%d", device, bundle.metadata.num_classes)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = _build_scheduler(optimizer, total_epochs=epochs, warmup_epochs=int(training_config["warmup_epochs"]))

    best_val_accuracy = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epoch_history: list[dict[str, float]] = []

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        train_metrics = _train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            gradient_clip=float(training_config["gradient_clip"]),
            progress_label=f"{dataset_name}/{method_name}/seed{seed}/epoch{epoch}",
        )
        scheduler.step()
        val_metrics = _evaluate_model(model, val_loader, device=device)
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
        LOGGER.info(
            "Epoch %d/%d complete: train_loss=%.4f val_loss=%.4f val_accuracy=%.4f best_val_accuracy=%.4f elapsed=%.1fs",
            epoch + 1,
            epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            best_val_accuracy,
            time.perf_counter() - epoch_start,
        )

    model.load_state_dict(best_state)
    LOGGER.info("Training epochs complete. Evaluating best checkpoint")
    val_metrics = _evaluate_model(model, val_loader, device=device)
    test_metrics = None if skip_test_evaluation else _evaluate_model(model, test_loader, device=device)

    if skip_posthoc_metrics:
        LOGGER.info("Skipping posthoc profiling and robustness metrics for this run")
        preprocessing_metrics = {
            "accepted_event_ratio": None,
            "compression_ratio": None,
            "preprocessing_latency_sec": None,
            "preprocessing_throughput_eps": None,
        }
        inference_latency_sec = None
        profile_metrics = {
            "parameter_count": parameter_count(model),
            "parameter_memory_bytes": parameter_memory_bytes(model),
            "peak_activation_bytes": 0,
            "dense_macs": 0,
            "sops": 0,
        }
        robustness = None
    else:
        sample_batch = _extract_sample_batch(test_loader)
        LOGGER.info("Measuring preprocessing cost")
        preprocessing_metrics = _measure_preprocessing(
            raw_dataset=test_raw,
            encoder=encoder,
            tensorizer=tensorizer,
            max_samples=int(training_config["profile_samples"]),
        )
        LOGGER.info("Measuring inference latency")
        inference_latency_sec = _measure_inference_latency(
            model=model,
            sample_batch=sample_batch,
            device=device,
            warmup=int(training_config["inference_warmup"]),
            timed=int(training_config["inference_runs"]),
        )
        profile_metrics = _profile_model(model=model, sample_batch=sample_batch, frame_mode=method_config.frame_mode)
        LOGGER.info("Evaluating robustness suites")
        robustness = _evaluate_robustness(
            model=model,
            raw_dataset=test_raw,
            encoder=encoder,
            tensorizer=tensorizer,
            bundle=bundle,
            dataset_name=dataset_name,
            batch_size=batch_size,
            device=device,
            seed=seed,
            training_config=training_config,
            tau_pair_us=int(resolved_filter_params.get("tau_pair_us", 1000)),
            loader_settings=loader_settings,
        )

    summary = _make_summary(
        dataset_name=dataset_name,
        method_name=method_name,
        seed=seed,
        device=device,
        bundle=bundle,
        resolved_filter_params=resolved_filter_params,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        preprocessing_metrics=preprocessing_metrics,
        inference_latency_sec=inference_latency_sec,
        epoch_history=epoch_history,
        profile_metrics=profile_metrics,
        model=model,
        filter_memory_name=_filter_memory_name(method_name, method_config.family),
        stage1_summary=stage1_summary,
        run_metadata=run_metadata,
        robustness=robustness,
    )

    output_dir = result_dir(root, dataset_name, result_method_name or method_name, seed, run_timestamp=resolved_run_timestamp)
    summary["run_metadata"]["output_dir"] = str(output_dir)
    _save_run_artifacts(output_dir, model, summary, robustness, save_checkpoint=save_checkpoint)
    LOGGER.info(
        "Training experiment complete: val_accuracy=%s test_accuracy=%s elapsed=%.1fs artifacts=%s",
        "none" if summary["val_accuracy"] is None else f"{summary['val_accuracy']:.4f}",
        "none" if summary["test_accuracy"] is None else f"{summary['test_accuracy']:.4f}",
        time.perf_counter() - run_start,
        output_dir,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate manuscript benchmark models.")
    parser.add_argument("--dataset", required=True, choices=("nmnist", "dvsgesture", "ncaltech101", "cifar10dvs"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--skip-posthoc-metrics", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    parser.add_argument("--run-timestamp", type=str, default=None)
    parser.add_argument("--use-latest-tuning", action="store_true")
    parser.add_argument("--filter-params", type=str, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    setup_logging()
    filter_params_override, filter_param_source = _resolve_cli_filter_params(
        root=args.root,
        dataset_name=args.dataset,
        method_name=args.method,
        use_latest_tuning=args.use_latest_tuning,
        filter_params_json=args.filter_params,
    )
    if filter_params_override is not None:
        LOGGER.info("Using filter parameter override: %s", filter_param_source)
    summary = run_training_experiment(
        root=args.root,
        dataset_name=args.dataset,
        method_name=args.method,
        seed=args.seed,
        filter_params_override=filter_params_override,
        filter_params_source=filter_param_source,
        epochs_override=args.epochs_override,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        max_slices_override=args.max_slices,
        force_cpu=args.force_cpu,
        skip_test_evaluation=args.skip_test_evaluation,
        skip_posthoc_metrics=args.skip_posthoc_metrics,
        save_checkpoint=not args.no_save_checkpoint,
        run_timestamp=args.run_timestamp,
    )
    print(summary)


if __name__ == "__main__":
    main()
