from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.common import latest_result_dir
from src.experiments.train_eval import run_training_experiment
from src.utils.logging import setup_logging
from src.utils.serialization import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile preprocessing, latency, memory, and compute metrics.")
    parser.add_argument("--dataset", required=True, choices=("nmnist", "dvsgesture", "ncaltech101", "cifar10dvs"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    setup_logging()
    summary_path = latest_result_dir(args.root, args.dataset, args.method, args.seed) / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
    elif args.train_if_missing:
        summary = run_training_experiment(
            root=args.root,
            dataset_name=args.dataset,
            method_name=args.method,
            seed=args.seed,
            epochs_override=args.epochs_override,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
            force_cpu=args.force_cpu,
        )
        summary_path = Path(summary["run_metadata"]["output_dir"]) / "summary.json"
    else:
        raise FileNotFoundError(f"Missing run summary: {summary_path}")

    profile = {
        "dataset": summary["dataset"],
        "method": summary["method"],
        "seed": summary["seed"],
        "accepted_event_ratio": summary["accepted_event_ratio"],
        "compression_ratio": summary["compression_ratio"],
        "preprocessing_latency_sec": summary["preprocessing_latency_sec"],
        "preprocessing_throughput_eps": summary["preprocessing_throughput_eps"],
        "inference_latency_sec": summary["inference_latency_sec"],
        "end_to_end_latency_sec": summary["end_to_end_latency_sec"],
        "filter_state_memory_bytes": summary["filter_state_memory_bytes"],
        "parameter_count": summary["parameter_count"],
        "parameter_memory_bytes": summary["parameter_memory_bytes"],
        "peak_activation_bytes": summary["peak_activation_bytes"],
        "sops": summary["sops"],
        "dense_macs": summary["dense_macs"],
    }
    output_path = summary_path.parent / "profile.json"
    save_json(output_path, profile)
    print(profile)


if __name__ == "__main__":
    main()
