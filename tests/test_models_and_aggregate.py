from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import pandas as pd

from src.experiments.tune_filters import _select_stage_b_candidate
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

    subprocess.run(
        [sys.executable, "-m", "src.experiments.aggregate_results", "--root", str(tmp_path)],
        check=True,
    )
    output_dir = tmp_path / "results" / "paper" / "aggregated"
    assert (output_dir / "main_accuracy.csv").exists()
    assert (output_dir / "efficiency.csv").exists()
    assert (output_dir / "aunc.csv").exists()
    assert (output_dir / "confidence_ratio.csv").exists()
    assert (output_dir / "accuracy_vs_compute.png").exists()
    accuracy = pd.read_csv(output_dir / "main_accuracy.csv", index_col=0)
    assert "Proposed +CONF" in accuracy.index


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
