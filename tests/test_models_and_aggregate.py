from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import pandas as pd

from src.experiments.common import candidate_param_grid, latest_result_dir, latest_tuning_filter_params, result_dir, tuning_result_dir
from src.experiments.aggregate_results import _selected_filter_params
from src.experiments.train_eval import _resolve_cli_filter_params
from src.experiments.tune_filters import (
    FAST_DEFAULT_METHODS,
    _default_methods_for_preset,
    _select_stage_b_candidate,
    _stage_b_training_kwargs,
)
from src.models.event_snn import BackboneConfig, EventSNN, spike_rate_cross_entropy
from src.models.frame_snn import FrameSNN
from src.models.sop_counter import count_dense_macs, count_event_sops, measure_peak_activation_bytes


def _write_summary_record(root: Path, record: dict[str, float | int | str]) -> None:
    summary_dir = root / "results" / "paper" / record["dataset"] / record["method"] / f'seed_{record["seed"]}'
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.json").write_text(json.dumps(record))


def test_event_model_forward_backward_and_profile() -> None:
    model = EventSNN(BackboneConfig(num_classes=3))
    sample = torch.zeros(4, 2, 4, 8, 8)
    sample[:, :, 0, 1, 1] = 1.0
    sample[:, :, 2, 2, 2] = 1.0
    targets = torch.tensor([0, 1], dtype=torch.long)
    spike_record = model(sample)
    assert spike_record.shape == (4, 2, 3)
    loss = spike_rate_cross_entropy(spike_record, targets)
    loss.backward()
    assert count_event_sops(model, sample) > 0
    assert measure_peak_activation_bytes(model, sample) > 0


def test_frame_model_forward_and_dense_macs() -> None:
    model = FrameSNN(num_classes=5)
    sample = torch.zeros(4, 2, 2, 8, 8)
    sample[:, :, 0, 1, 1] = 1.0
    spike_record = model(sample)
    assert spike_record.shape == (4, 2, 5)
    assert count_dense_macs(model, sample) > 0
    assert measure_peak_activation_bytes(model, sample) > 0


def test_aggregate_results_from_stub_summaries(tmp_path: Path) -> None:
    records = [
        {
            "dataset": "nmnist",
            "method": "raw_snn",
            "seed": 0,
            "test_accuracy": 0.8,
            "aunc": 0.7,
            "accepted_event_ratio": 1.0,
            "compression_ratio": 1.0,
            "confidence_ratio": None,
            "preprocessing_latency_sec": 0.001,
            "end_to_end_latency_sec": 0.01,
            "filter_state_memory_bytes": 0,
            "peak_activation_bytes": 1024,
            "sops": 100,
            "dense_macs": 0,
            "run_metadata": {"run_purpose": "paper_main"},
        },
        {
            "dataset": "nmnist",
            "method": "proposed_conf",
            "seed": 1,
            "test_accuracy": 0.9,
            "aunc": 0.8,
            "accepted_event_ratio": 0.5,
            "compression_ratio": 2.0,
            "confidence_ratio": 1.7,
            "preprocessing_latency_sec": 0.002,
            "end_to_end_latency_sec": 0.02,
            "filter_state_memory_bytes": 100,
            "peak_activation_bytes": 1024,
            "sops": 80,
            "dense_macs": 0,
            "run_metadata": {"run_purpose": "paper_main"},
        },
        {
            "dataset": "nmnist",
            "method": "proposed_conf",
            "seed": 2,
            "test_accuracy": 0.1,
            "aunc": 0.1,
            "accepted_event_ratio": 0.1,
            "compression_ratio": 10.0,
            "confidence_ratio": 1.0,
            "preprocessing_latency_sec": 1.0,
            "end_to_end_latency_sec": 2.0,
            "filter_state_memory_bytes": 100,
            "peak_activation_bytes": 1024,
            "sops": 80,
            "dense_macs": 0,
            "run_metadata": {"run_purpose": "custom"},
        },
    ]
    for record in records:
        _write_summary_record(tmp_path, record)
    tuning_dir = tmp_path / "results" / "paper" / "nmnist" / "proposed_conf" / "tuning_20260522_120000_000000"
    tuning_dir.mkdir(parents=True)
    (tuning_dir / "tuning_summary.json").write_text(json.dumps({"selected": {"filter_params": {"k0": 1}}}))
    (tuning_dir / "event_metrics.json").write_text(
        json.dumps(
            {
                "sources": {
                    "ba": {
                        "mean_auc": 0.9,
                        "mean_ekr": 0.5,
                        "mean_esr": 0.8,
                        "ratios": [
                            {
                                "ratio": 1.0,
                                "auc": 0.9,
                                "ekr": 0.5,
                                "esr": 0.8,
                                "compression_ratio": 2.0,
                                "accepted_events": 12,
                            }
                        ],
                    }
                }
            }
        )
    )

    subprocess.run(
        [sys.executable, "-m", "src.experiments.aggregate_results", "--root", str(tmp_path)],
        check=True,
    )
    output_dir = tmp_path / "results" / "paper" / "aggregated"
    assert (output_dir / "main_accuracy.csv").exists()
    assert (output_dir / "efficiency.csv").exists()
    assert (output_dir / "aunc.csv").exists()
    assert (output_dir / "confidence_ratio.csv").exists()
    assert (output_dir / "event_metrics.csv").exists()
    assert (output_dir / "accuracy_vs_compute.png").exists()
    accuracy = pd.read_csv(output_dir / "main_accuracy.csv", index_col=0)
    assert "Proposed +CONF" in accuracy.index
    event_metrics = pd.read_csv(output_dir / "event_metrics.csv")
    assert event_metrics.loc[0, "esr"] == 0.8


def test_stage_b_selection_keeps_stage_a_order_on_accuracy_tie() -> None:
    top_candidates = [
        {"rank": 0, "filter_params": {"tau_ref_dig_us": 500}},
        {"rank": 1, "filter_params": {"tau_ref_dig_us": 1000}},
    ]
    stage_b_results = [
        {"rank": 0, "filter_params": {"tau_ref_dig_us": 500}, "val_accuracy": 0.8},
        {"rank": 1, "filter_params": {"tau_ref_dig_us": 1000}, "val_accuracy": 0.8},
    ]
    selected = _select_stage_b_candidate(top_candidates, stage_b_results)
    assert selected["filter_params"]["tau_ref_dig_us"] == 500


def test_fast_tuning_defaults_exclude_untuned_singletons_and_shrink_grids() -> None:
    assert _default_methods_for_preset("fast") == FAST_DEFAULT_METHODS
    assert "raw_snn" not in FAST_DEFAULT_METHODS
    assert "frame_snn" not in FAST_DEFAULT_METHODS
    assert len(candidate_param_grid("proposed_conf", preset="fast")) < len(
        candidate_param_grid("proposed_conf", preset="paper")
    )
    assert len(candidate_param_grid("proposed_sup", preset="fast")) == 18
    assert len(candidate_param_grid("proposed_conf", preset="fast")) == 72
    assert len(candidate_param_grid("proposed_conf", preset="paper")) == 1458
    assert {candidate["k0"] for candidate in candidate_param_grid("proposed_conf", preset="fast")} == {1, 2}
    assert {candidate["k0"] for candidate in candidate_param_grid("proposed_conf", preset="paper")} == {1, 2}
    assert {candidate["warmup_us"] for candidate in candidate_param_grid("proposed_conf", preset="fast")} == {5000}


def test_stage_b_lightweight_kwargs_skip_expensive_posthoc_work() -> None:
    assert _stage_b_training_kwargs(lightweight=True) == {
        "skip_test_evaluation": True,
        "skip_posthoc_metrics": True,
        "save_checkpoint": False,
    }
    assert _stage_b_training_kwargs(lightweight=False) == {
        "skip_test_evaluation": False,
        "skip_posthoc_metrics": False,
        "save_checkpoint": True,
    }


def test_result_directories_can_be_timestamped(tmp_path: Path) -> None:
    legacy = result_dir(tmp_path, "nmnist", "raw_snn", seed=0)
    timestamped = result_dir(tmp_path, "nmnist", "raw_snn", seed=0, run_timestamp="20260522_091500_123456")
    assert legacy.name == "seed_0"
    assert timestamped.name == "seed_0__20260522_091500_123456"

    timestamped.mkdir(parents=True)
    assert latest_result_dir(tmp_path, "nmnist", "raw_snn", seed=0) == timestamped
    assert tuning_result_dir(tmp_path, "nmnist", "ba_snn", "20260522_091500_123456").name == (
        "tuning_20260522_091500_123456"
    )


def test_selected_filter_params_prefers_latest_timestamped_tuning_summary(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "results" / "paper" / "nmnist" / "ba_snn"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "tuning_summary.json").write_text(
        json.dumps({"selected": {"filter_params": {"delta_t_us": 1000}}})
    )
    older_dir = tuning_result_dir(tmp_path, "nmnist", "ba_snn", "20260522_090000_000000")
    newer_dir = tuning_result_dir(tmp_path, "nmnist", "ba_snn", "20260522_100000_000000")
    older_dir.mkdir(parents=True)
    newer_dir.mkdir(parents=True)
    (older_dir / "tuning_summary.json").write_text(
        json.dumps({"selected": {"filter_params": {"delta_t_us": 2000}}})
    )
    (newer_dir / "tuning_summary.json").write_text(
        json.dumps({"selected": {"filter_params": {"delta_t_us": 5000}}})
    )

    assert _selected_filter_params(tmp_path, "nmnist", "ba_snn") == {"delta_t_us": 5000}
    latest = latest_tuning_filter_params(tmp_path, "nmnist", "ba_snn")
    assert latest is not None
    assert latest[0] == {"delta_t_us": 5000}


def test_train_eval_cli_filter_params_merge_latest_and_manual_override(tmp_path: Path) -> None:
    tuning_dir = tuning_result_dir(tmp_path, "nmnist", "proposed_conf", "20260522_100000_000000")
    tuning_dir.mkdir(parents=True)
    (tuning_dir / "tuning_summary.json").write_text(
        json.dumps(
            {
                "selected": {
                    "filter_params": {
                        "tau_ref_dig_us": 500,
                        "delta_t_us": 1000,
                        "k0": 0,
                    }
                }
            }
        )
    )

    params, source = _resolve_cli_filter_params(
        root=tmp_path,
        dataset_name="nmnist",
        method_name="proposed_conf",
        use_latest_tuning=True,
        filter_params_json='{"k0": 1, "gamma": 0}',
    )

    assert params == {"tau_ref_dig_us": 500, "delta_t_us": 1000, "k0": 1, "gamma": 0}
    assert source["latest_tuning_path"].endswith("tuning_summary.json")
    assert source["manual_filter_params"] is True


def test_main_tex_uses_conservative_empirical_claims() -> None:
    text = Path("main.tex").read_text()
    forbidden = (
        "state-of-the-art",
        "Comprehensive evaluations demonstrate",
        "88.5",
        "over 5M",
        "5M eps",
        "Empirical evaluations confirm",
    )
    for phrase in forbidden:
        assert phrase not in text
