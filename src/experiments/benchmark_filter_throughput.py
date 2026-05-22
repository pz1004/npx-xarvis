from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.experiments.common import (
    RESULTS_ROOT,
    build_method_pipeline,
    calibration_filter_params,
    latest_tuning_filter_params,
    load_method_config,
    load_training_config,
    make_run_timestamp,
)
from src.data.datasets import prepare_dataset_bundle
from src.experiments.train_eval import _software_versions
from src.utils.logging import setup_logging
from src.utils.serialization import save_json


def _parse_filter_params(payload: str | None) -> dict[str, Any]:
    if payload is None:
        return {}
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("--filter-params must be a JSON object")
    return dict(parsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark event-filter preprocessing throughput.")
    parser.add_argument("--dataset", required=True, choices=("nmnist", "dvsgesture", "ncaltech101", "cifar10dvs"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--use-latest-tuning", action="store_true")
    parser.add_argument("--filter-params", type=str, default=None)
    parser.add_argument("--run-timestamp", type=str, default=None)
    args = parser.parse_args()

    setup_logging()
    run_timestamp = args.run_timestamp or make_run_timestamp()
    training_config = load_training_config(args.root)
    method_config = load_method_config(args.root, args.method)
    bundle = prepare_dataset_bundle(args.dataset, args.root, split_seed=training_config["split_seed"])

    sample_count = min(args.max_samples, len(bundle.train_raw))
    samples = [bundle.train_raw[index][0] for index in range(sample_count)]
    filter_params = calibration_filter_params(method_config, samples, bundle.metadata.sensor_size)
    tuning_path = None
    if args.use_latest_tuning:
        latest = latest_tuning_filter_params(args.root, args.dataset, args.method)
        if latest is None:
            raise FileNotFoundError(f"No tuning summary found for {args.dataset}/{args.method}")
        latest_params, tuning_path = latest
        filter_params.update(latest_params)
    filter_params.update(_parse_filter_params(args.filter_params))

    _, _, filter_apply = build_method_pipeline(
        method=method_config,
        sensor_size=bundle.metadata.sensor_size,
        slicing_config=bundle.slicing,
        filter_params=filter_params,
    )
    if filter_apply is None:
        raise ValueError(f"Method {args.method} does not expose an event filter to benchmark")

    if samples:
        filter_apply(samples[0])

    elapsed_sec: list[float] = []
    total_events = 0
    accepted_events = 0
    for events in samples:
        start = time.perf_counter()
        result = filter_apply(events)
        elapsed = time.perf_counter() - start
        elapsed_sec.append(elapsed)
        total_events += int(len(events))
        accepted_events += int(np.sum(result.accepted_mask))

    elapsed_total = float(sum(elapsed_sec))
    payload = {
        "dataset": args.dataset,
        "method": args.method,
        "run_timestamp": run_timestamp,
        "sample_count": sample_count,
        "event_count": total_events,
        "accepted_event_count": accepted_events,
        "accepted_event_ratio": float(accepted_events / total_events) if total_events else 0.0,
        "median_latency_sec": float(np.median(elapsed_sec)) if elapsed_sec else 0.0,
        "mean_latency_sec": float(np.mean(elapsed_sec)) if elapsed_sec else 0.0,
        "total_elapsed_sec": elapsed_total,
        "throughput_eps": float(total_events / max(elapsed_total, 1e-12)),
        "filter_params": filter_params,
        "filter_params_source": {
            "use_latest_tuning": args.use_latest_tuning,
            "latest_tuning_path": None if tuning_path is None else str(tuning_path),
            "manual_filter_params": args.filter_params is not None,
        },
        "software_versions": _software_versions(),
        "hardware": {
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
    }
    output_path = args.root / RESULTS_ROOT / f"filter_throughput_{args.dataset}_{args.method}_{run_timestamp}.json"
    save_json(output_path, payload)
    print(output_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
