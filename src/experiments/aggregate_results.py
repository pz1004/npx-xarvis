from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import Subset

from src.data.datasets import make_calibration_subset, prepare_dataset_bundle
from src.data.noise_injection import build_noise_suites
from src.experiments.common import (
    build_method_pipeline,
    calibration_filter_params,
    latest_tuning_filter_params,
    latest_tuning_summary_path,
    load_method_config,
    load_training_config,
)
from src.experiments.common import filter_score
from src.filters.metrics import safe_roc_auc
from src.utils.logging import setup_logging
from src.utils.serialization import load_json


SUMMARY_GLOB = Path("results") / "paper"
ROC_METHODS = ("ba_snn", "stcf_rc_snn", "proposed_ref", "proposed_sup", "proposed_pol", "proposed_conf")
MAIN_METHODS = ("raw_snn", "frame_snn", "ba_snn", "stcf_rc_snn", "proposed_ref", "proposed_sup", "proposed_pol", "proposed_conf")
EFFICIENCY_METHODS = MAIN_METHODS + ("proposed_lowmem", "proposed_lowlat")
ABLATION_METHODS = ("raw_snn", "proposed_ref", "proposed_sup", "proposed_pol", "proposed_conf")
METHOD_LABELS = {
    "raw_snn": "Raw Events",
    "frame_snn": "Frame Baseline",
    "ba_snn": "BA Filter",
    "stcf_rc_snn": "STCF",
    "proposed_ref": "Proposed +REF",
    "proposed_sup": "Proposed +SUP",
    "proposed_pol": "Proposed +OPP",
    "proposed_conf": "Proposed +CONF",
    "proposed_lowmem": "Proposed Low-Mem",
    "proposed_lowlat": "Proposed Low-Lat",
}


def _run_purpose(record: dict[str, Any]) -> str | None:
    return record.get("run_metadata", {}).get("run_purpose")


def _load_summaries(root: Path, allowed_purposes: tuple[str, ...]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for summary_path in (root / SUMMARY_GLOB).rglob("summary.json"):
        record = load_json(summary_path)
        if _run_purpose(record) not in allowed_purposes:
            continue
        records.append(record)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)
    frame["method_label"] = frame["method"].map(METHOD_LABELS).fillna(frame["method"])
    return frame


def _load_event_metrics(root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    results_root = root / SUMMARY_GLOB
    if not results_root.exists():
        return pd.DataFrame()
    for dataset_dir in sorted(path for path in results_root.iterdir() if path.is_dir() and path.name != "aggregated"):
        for method_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            tuning_summary_path = latest_tuning_summary_path(root, dataset_dir.name, method_dir.name)
            if tuning_summary_path is None:
                continue
            event_metrics_path = tuning_summary_path.with_name("event_metrics.json")
            if not event_metrics_path.exists():
                continue
            payload = load_json(event_metrics_path)
            run_timestamp = tuning_summary_path.parent.name.removeprefix("tuning_")
            for source, source_payload in payload.get("sources", {}).items():
                for ratio_payload in source_payload.get("ratios", []):
                    records.append(
                        {
                            "dataset": dataset_dir.name,
                            "method": method_dir.name,
                            "method_label": METHOD_LABELS.get(method_dir.name, method_dir.name),
                            "run_timestamp": run_timestamp,
                            "source": source,
                            "ratio": ratio_payload.get("ratio"),
                            "auc": ratio_payload.get("auc"),
                            "ekr": ratio_payload.get("ekr"),
                            "esr": ratio_payload.get("esr"),
                            "compression_ratio": ratio_payload.get("compression_ratio"),
                            "accepted_events": ratio_payload.get("accepted_events"),
                            "mean_auc": source_payload.get("mean_auc"),
                            "mean_ekr": source_payload.get("mean_ekr"),
                            "mean_esr": source_payload.get("mean_esr"),
                        }
                    )
    return pd.DataFrame.from_records(records)


def _format_mean_std(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "n/a"
    mean = float(values.mean())
    std = float(values.std())
    return f"{mean:.4f} +- {std:.4f}"


def _format_metric_table(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(["method_label", "dataset"])[value_col]
        .apply(_format_mean_std)
        .reset_index(name="value")
    )
    return grouped.pivot(index="method_label", columns="dataset", values="value").fillna("")


def _efficiency_table(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accepted_event_ratio",
        "compression_ratio",
        "preprocessing_latency_sec",
        "end_to_end_latency_sec",
        "filter_state_memory_bytes",
        "peak_activation_bytes",
        "sops",
        "dense_macs",
        "confidence_ratio",
        "aunc",
    ]
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["dataset", "method", "method_label"], dropna=False)
    for (dataset, method, method_label), group in grouped:
        row = {
            "dataset": dataset,
            "method": method,
            "method_label": method_label,
        }
        for metric in metrics:
            row[metric] = _format_mean_std(group[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "method_label"]).reset_index(drop=True)


def _save_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=not {"dataset", "method", "method_label"}.issubset(table.columns))
    path.with_suffix(".md").write_text(table.to_markdown(index=not {"dataset", "method", "method_label"}.issubset(table.columns)))


def _selected_filter_params(root: Path, dataset_name: str, method_name: str) -> dict[str, Any]:
    latest = latest_tuning_filter_params(root, dataset_name, method_name)
    if latest is not None:
        filter_params, _tuning_path = latest
        return filter_params
    method_config = load_method_config(root, method_name)
    return dict(method_config.filter_params)


def _restrict_subset(calibration_subset: Subset, max_samples: int) -> Subset:
    if len(calibration_subset) <= max_samples:
        return calibration_subset
    return Subset(calibration_subset.dataset, calibration_subset.indices[:max_samples])


def _roc_payload_for_suite(
    calibration_subset: Subset,
    filter_apply,
    sensor_size: tuple[int, int, int],
    noise_ratios: tuple[float, ...],
    tau_pair_us: int,
    split_seed: int,
    suite_name: str,
) -> tuple[list[float], list[float], float]:
    y_true_all: list[bool] = []
    score_all: list[float] = []
    for sample_index in range(len(calibration_subset)):
        events, _ = calibration_subset[sample_index]
        suites = build_noise_suites(
            signal_events=events,
            sensor_size=sensor_size,
            ratios=noise_ratios,
            tau_pair_us=tau_pair_us,
            seed=split_seed + sample_index * 41,
        )
        for suite in suites:
            if suite.source != suite_name:
                continue
            result = filter_apply(suite.events)
            y_true_all.extend(suite.is_signal.tolist())
            score_all.extend(filter_score(result).tolist())
    fpr, tpr, roc_auc = safe_roc_auc(y_true_all, score_all)
    return fpr.tolist(), tpr.tolist(), roc_auc


def _compute_column(row: pd.Series) -> float:
    dense = pd.to_numeric(pd.Series([row["dense_macs"]]), errors="coerce").iloc[0]
    sops = pd.to_numeric(pd.Series([row["sops"]]), errors="coerce").iloc[0]
    if pd.notna(dense) and float(dense) > 0:
        return float(dense)
    if pd.notna(sops):
        return float(sops)
    return float("nan")


def _generate_roc_plot(root: Path, dataset_name: str, output_dir: Path, max_calibration_samples: int) -> None:
    training_config = load_training_config(root)
    bundle = prepare_dataset_bundle(dataset_name, root, split_seed=training_config["split_seed"])
    calibration_subset = make_calibration_subset(bundle.train_raw, seed=training_config["split_seed"])
    calibration_subset = _restrict_subset(calibration_subset, max_calibration_samples)
    calibration_events = [calibration_subset[index][0] for index in range(len(calibration_subset))]

    suite_payloads: dict[str, list[tuple[str, list[float], list[float], float]]] = {"ba": [], "shot": [], "mixed": []}
    for method_name in ROC_METHODS:
        method_config = load_method_config(root, method_name)
        filter_params = calibration_filter_params(method_config, calibration_events, bundle.metadata.sensor_size)
        filter_params.update(_selected_filter_params(root, dataset_name, method_name))
        _, _, filter_apply = build_method_pipeline(
            method=method_config,
            sensor_size=bundle.metadata.sensor_size,
            slicing_config=bundle.slicing,
            filter_params=filter_params,
        )
        assert filter_apply is not None
        for suite_name in suite_payloads:
            fpr, tpr, roc_auc = _roc_payload_for_suite(
                calibration_subset=calibration_subset,
                filter_apply=filter_apply,
                sensor_size=bundle.metadata.sensor_size,
                noise_ratios=tuple(training_config["noise_ratios"]),
                tau_pair_us=int(filter_params.get("tau_pair_us", 1000)),
                split_seed=training_config["split_seed"],
                suite_name=suite_name,
            )
            suite_payloads[suite_name].append((METHOD_LABELS.get(method_name, method_name), fpr, tpr, roc_auc))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for axis, suite_name in zip(axes, ("ba", "shot", "mixed")):
        for method_name, fpr, tpr, roc_auc in suite_payloads[suite_name]:
            axis.plot(fpr, tpr, label=f"{method_name} (AUC={roc_auc:.3f})")
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        axis.set_title(f"{dataset_name} / {suite_name.upper()}")
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.legend(fontsize=8)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"roc_{dataset_name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate run summaries into tables and plots.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--regen-roc", action="store_true")
    parser.add_argument("--roc-dataset", choices=("nmnist", "dvsgesture", "ncaltech101", "cifar10dvs"), default="nmnist")
    parser.add_argument("--roc-calibration-samples", type=int, default=64)
    args = parser.parse_args()

    setup_logging()
    main_frame = _load_summaries(args.root, allowed_purposes=("paper_main",))
    efficiency_frame = _load_summaries(args.root, allowed_purposes=("paper_main", "paper_profile"))
    event_metrics_frame = _load_event_metrics(args.root)
    if main_frame.empty and efficiency_frame.empty and event_metrics_frame.empty:
        raise RuntimeError("No protocol run summaries found under results/paper")

    output_dir = args.root / SUMMARY_GLOB / "aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not event_metrics_frame.empty:
        _save_table(event_metrics_frame.sort_values(["dataset", "method_label", "source", "ratio"]), output_dir / "event_metrics.csv")

    if not main_frame.empty:
        accuracy_frame = main_frame[main_frame["method"].isin(MAIN_METHODS)]
        accuracy_table = _format_metric_table(accuracy_frame, "test_accuracy")
        _save_table(accuracy_table, output_dir / "main_accuracy.csv")

        ablation_frame = main_frame[main_frame["method"].isin(ABLATION_METHODS)]
        ablation_table = _format_metric_table(ablation_frame, "test_accuracy")
        _save_table(ablation_table, output_dir / "ablation.csv")
    else:
        accuracy_table = pd.DataFrame()

    if efficiency_frame.empty:
        efficiency_table = pd.DataFrame()
    else:
        filtered_efficiency = efficiency_frame[efficiency_frame["method"].isin(EFFICIENCY_METHODS)]
        efficiency_table = _efficiency_table(filtered_efficiency)
    _save_table(efficiency_table, output_dir / "efficiency.csv")

    if not main_frame.empty:
        confidence_table = _format_metric_table(main_frame[main_frame["method"] == "proposed_conf"], "confidence_ratio")
        _save_table(confidence_table, output_dir / "confidence_ratio.csv")
        aunc_table = _format_metric_table(main_frame[main_frame["method"].isin(MAIN_METHODS)], "aunc")
        _save_table(aunc_table, output_dir / "aunc.csv")

        sns.set_theme(style="whitegrid")
        scatter_frame = main_frame[main_frame["method"].isin(MAIN_METHODS)].copy()
        scatter_frame["method_label"] = scatter_frame["method"].map(METHOD_LABELS).fillna(scatter_frame["method"])
        scatter_frame["compute"] = scatter_frame.apply(_compute_column, axis=1)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=scatter_frame, x="compute", y="test_accuracy", hue="method_label", style="dataset", s=90)
        plt.xscale("log")
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_vs_compute.png", dpi=200, bbox_inches="tight")
        plt.close()

    if args.regen_roc:
        _generate_roc_plot(
            root=args.root,
            dataset_name=args.roc_dataset,
            output_dir=output_dir,
            max_calibration_samples=args.roc_calibration_samples,
        )

    print(f"Aggregated outputs written to {output_dir}")


if __name__ == "__main__":
    main()
