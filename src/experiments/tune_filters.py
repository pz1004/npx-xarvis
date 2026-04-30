from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from torch.utils.data import Subset

from src.data.datasets import make_calibration_subset, prepare_dataset_bundle
from src.data.noise_injection import build_noise_suites
from src.experiments.common import (
    build_method_pipeline,
    calibration_filter_params,
    candidate_param_grid,
    load_method_config,
    load_training_config,
    result_dir,
)
from src.filters.metrics import evaluate_filter_predictions
from src.experiments.common import filter_score
from src.experiments.train_eval import run_training_experiment
from src.utils.logging import setup_logging
from src.utils.serialization import save_json


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _restrict_subset(calibration_subset: Subset, max_samples: int | None) -> Subset:
    if max_samples is None or len(calibration_subset) <= max_samples:
        return calibration_subset
    return Subset(calibration_subset.dataset, calibration_subset.indices[:max_samples])


def _detailed_event_metrics(
    calibration_subset: Subset,
    filter_apply,
    sensor_size: tuple[int, int, int],
    noise_ratios: tuple[float, ...],
    tau_pair_us: int,
    split_seed: int,
) -> dict[str, Any]:
    suite_payload: dict[str, list[dict[str, Any]]] = {}
    for suite_name in ("ba", "shot", "mixed"):
        ratio_metrics: list[dict[str, Any]] = []
        source_aucs: list[float] = []
        source_ekrs: list[float] = []
        for ratio in noise_ratios:
            y_true_all: list[bool] = []
            score_all: list[float] = []
            accepted_all: list[bool] = []
            for sample_index in range(len(calibration_subset)):
                events, _ = calibration_subset[sample_index]
                suites = build_noise_suites(
                    signal_events=events,
                    sensor_size=sensor_size,
                    ratios=(ratio,),
                    tau_pair_us=tau_pair_us,
                    seed=split_seed + sample_index * 31,
                )
                for suite in suites:
                    if suite.source != suite_name:
                        continue
                    result = filter_apply(suite.events)
                    y_true_all.extend(suite.is_signal.tolist())
                    score_all.extend(filter_score(result).tolist())
                    accepted_all.extend(result.accepted_mask.tolist())
            metrics = evaluate_filter_predictions(
                is_signal=y_true_all,
                scores=score_all,
                accepted_mask=accepted_all,
            )
            source_aucs.append(float(metrics["auc"]))
            source_ekrs.append(float(metrics["ekr"]))
            ratio_metrics.append(
                {
                    "ratio": ratio,
                    "auc": float(metrics["auc"]),
                    "ekr": float(metrics["ekr"]),
                    "compression_ratio": float(metrics["compression_ratio"]),
                    "accepted_events": int(metrics["accepted_events"]),
                    "tpr_at_fpr": metrics["tpr_at_fpr"],
                    "roc_points": metrics["roc_points"],
                }
            )
        suite_payload[suite_name] = {
            "ratios": ratio_metrics,
            "mean_auc": _mean(source_aucs),
            "mean_ekr": _mean(source_ekrs),
        }
    return {
        "sources": suite_payload,
        "noise_ratios": list(noise_ratios),
        "base_seed": split_seed,
    }


def _evaluate_candidate_auc(
    calibration_subset: Subset,
    filter_apply,
    sensor_size: tuple[int, int, int],
    noise_ratios: tuple[float, ...],
    tau_pair_us: int,
    split_seed: int,
) -> tuple[float, float]:
    aucs: list[float] = []
    ekrs: list[float] = []
    for sample_index in range(len(calibration_subset)):
        events, _ = calibration_subset[sample_index]
        suites = build_noise_suites(
            signal_events=events,
            sensor_size=sensor_size,
            ratios=noise_ratios,
            tau_pair_us=tau_pair_us,
            seed=split_seed + sample_index * 31,
        )
        for suite in suites:
            result = filter_apply(suite.events)
            metrics = evaluate_filter_predictions(
                is_signal=suite.is_signal,
                scores=filter_score(result),
                accepted_mask=result.accepted_mask,
            )
            aucs.append(float(metrics["auc"]))
            ekrs.append(float(metrics["ekr"]))
    return _mean(aucs), _mean(ekrs)


def _select_stage_b_candidate(
    top_candidates: list[dict[str, Any]],
    stage_b_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not stage_b_results:
        return top_candidates[0]
    return max(stage_b_results, key=lambda item: item["val_accuracy"])


def evaluate_candidates(
    root: Path,
    dataset_name: str,
    method_name: str,
    max_calibration_samples: int | None = None,
    max_grid: int | None = None,
) -> list[dict[str, Any]]:
    training_config = load_training_config(root)
    method_config = load_method_config(root, method_name)
    bundle = prepare_dataset_bundle(dataset_name, root, split_seed=training_config["split_seed"])
    calibration_subset = make_calibration_subset(bundle.train_raw, seed=training_config["split_seed"])
    calibration_subset = _restrict_subset(calibration_subset, max_calibration_samples)

    calibration_events = [calibration_subset[index][0] for index in range(len(calibration_subset))]
    base_params = calibration_filter_params(method_config, calibration_events, bundle.metadata.sensor_size)

    candidates = candidate_param_grid(method_name)
    if max_grid is not None:
        candidates = candidates[:max_grid]

    ranked: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        resolved = dict(base_params)
        resolved.update(candidate)
        _, _, filter_apply = build_method_pipeline(
            method=method_config,
            sensor_size=bundle.metadata.sensor_size,
            slicing_config=bundle.slicing,
            filter_params=resolved,
        )
        if filter_apply is None:
            ranked.append(
                {
                    "candidate_index": candidate_index,
                    "filter_params": resolved,
                    "mean_auc": 0.5,
                    "mean_ekr": 1.0,
                }
            )
            continue

        mean_auc, mean_ekr = _evaluate_candidate_auc(
            calibration_subset=calibration_subset,
            filter_apply=filter_apply,
            sensor_size=bundle.metadata.sensor_size,
            noise_ratios=tuple(training_config["noise_ratios"]),
            tau_pair_us=int(resolved.get("tau_pair_us", 1000)),
            split_seed=training_config["split_seed"],
        )
        ranked.append(
            {
                "candidate_index": candidate_index,
                "filter_params": resolved,
                "mean_auc": mean_auc,
                "mean_ekr": mean_ekr,
            }
        )

    ranked.sort(key=lambda item: item["mean_auc"], reverse=True)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune filter hyperparameters with synthetic-noise AUC.")
    parser.add_argument("--dataset", required=True, choices=("nmnist", "dvsgesture", "ncaltech101", "cifar10dvs"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-calibration-samples", type=int, default=None)
    parser.add_argument("--max-grid", type=int, default=None)
    parser.add_argument("--skip-stage-b", action="store_true")
    parser.add_argument("--stage-b-epochs", type=int, default=None)
    parser.add_argument("--stage-b-max-train-samples", type=int, default=None)
    parser.add_argument("--stage-b-max-val-samples", type=int, default=None)
    parser.add_argument("--stage-b-max-test-samples", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    setup_logging()
    training_config = load_training_config(args.root)
    method_config = load_method_config(args.root, args.method)
    bundle = prepare_dataset_bundle(args.dataset, args.root, split_seed=training_config["split_seed"])
    ranked = evaluate_candidates(
        root=args.root,
        dataset_name=args.dataset,
        method_name=args.method,
        max_calibration_samples=args.max_calibration_samples,
        max_grid=args.max_grid,
    )
    top_candidates = ranked[: args.top_k]
    stage_b_results: list[dict[str, Any]] = []

    if not args.skip_stage_b:
        for rank, candidate in enumerate(top_candidates):
            summary = run_training_experiment(
                root=args.root,
                dataset_name=args.dataset,
                method_name=args.method,
                seed=args.seed,
                filter_params_override=candidate["filter_params"],
                result_method_name=f"{args.method}__tune_{rank}",
                run_purpose="tuning_stage_b",
                epochs_override=args.stage_b_epochs,
                max_train_samples=args.stage_b_max_train_samples,
                max_val_samples=args.stage_b_max_val_samples,
                max_test_samples=args.stage_b_max_test_samples,
                force_cpu=args.force_cpu,
            )
            stage_b_results.append(
                {
                    "rank": rank,
                    "filter_params": candidate["filter_params"],
                    "val_accuracy": summary["val_accuracy"],
                    "test_accuracy": summary["test_accuracy"],
                    "end_to_end_latency_sec": summary["end_to_end_latency_sec"],
                }
            )

    selected = _select_stage_b_candidate(top_candidates, stage_b_results)

    selected_filter_params = selected["filter_params"]
    _, _, selected_filter_apply = build_method_pipeline(
        method=method_config,
        sensor_size=bundle.metadata.sensor_size,
        slicing_config=bundle.slicing,
        filter_params=selected_filter_params,
    )
    event_metrics = None
    if selected_filter_apply is not None:
        calibration_subset = _restrict_subset(
            make_calibration_subset(bundle.train_raw, seed=training_config["split_seed"]),
            args.max_calibration_samples,
        )
        event_metrics = _detailed_event_metrics(
            calibration_subset=calibration_subset,
            filter_apply=selected_filter_apply,
            sensor_size=bundle.metadata.sensor_size,
            noise_ratios=tuple(training_config["noise_ratios"]),
            tau_pair_us=int(selected_filter_params.get("tau_pair_us", 1000)),
            split_seed=training_config["split_seed"],
        )

    tuning_summary = {
        "dataset": args.dataset,
        "method": args.method,
        "top_candidates": top_candidates,
        "stage_b_results": stage_b_results,
        "selected": selected,
    }
    output_dir = result_dir(args.root, args.dataset, args.method, args.seed).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "tuning_summary.json", tuning_summary)
    if event_metrics is not None:
        save_json(output_dir / "event_metrics.json", event_metrics)
    print(tuning_summary)


if __name__ == "__main__":
    main()
