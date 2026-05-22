# npx-xarvis Project Manual

Last updated: 2026-05-22

This is the single canonical Markdown document at the repository root. It
consolidates the former root notes, guides, audits, revision plans, and
implementation specification into one current English manual.

`main.tex` is the authority for the manuscript's scientific argument. This file
keeps only the information that is still useful for running, validating, and
maintaining the codebase beside the paper.

## 1. Scope

The project studies event-domain denoising for direct SNN ingestion without
frame reconstruction in the proposed event-domain path. The implementation
supports:

- deterministic event-domain filters and ablations,
- direct-event and frame-based SNN baselines,
- fast hyperparameter tuning for filter selection,
- full training/evaluation runs for paper-grade results,
- timestamped result artifacts for distinguishable experiments,
- aggregation of accuracy, robustness, efficiency, and confidence-use metrics.

The full theory, related work, algorithms, equations, and evaluation questions
are already in `main.tex`; they are intentionally not repeated here.

## 2. Current Status

Implemented and verified:

- `proposed_balanced` keeps raw same-pixel history separate from accepted
  support history.
- Raw state is updated for every event; accepted state is updated only for
  accepted events.
- Stage 4 is implemented as a polarity-conditioned threshold increase.
- `proposed_balanced` and `proposed_lowmem` use explicit `warmup_us` seeding
  so strict `k0 >= 1` filters can populate accepted history at sample start.
- Confidence-coded event output is supported for `proposed_conf`.
- The proposed filter resets its mutable state per sample, which is required
  for reproducible multi-worker loading.
- The core proposed balanced filter and the low-memory approximation have
  Numba JIT paths with Python fallbacks.
- `tune_filters` defaults to `--preset fast` and writes timestamped tuning
  directories.
- `train_eval` can load the latest timestamped tuning parameters with
  `--use-latest-tuning`.
- Manual filter parameter JSON passed through `--filter-params` overrides the
  latest tuning values when both are provided.
- Dataset slicing metadata and result directories are timestamped or cached so
  repeated runs are distinguishable and less wasteful.
- ESR is reported as retained signal events divided by total signal events in
  synthetic-noise tuning metrics and aggregate event-metric tables.
- Filter throughput can be measured with a timestamped benchmark artifact;
  `proposed_conf` N-MNIST throughput was refreshed on 2026-05-22.
- The current test suite passes after the runtime and tuning handoff fixes.

Open alignment items:

- Treat refreshed fast tuning as parameter-selection evidence only; full
  paper-grade `train_eval` artifacts are still required before making final
  quantitative manuscript claims.
- If citing an exact preprocessing throughput value, cite the timestamped
  benchmark artifact that produced it.
- Keep `main.tex` conservative until the full paper-grade result matrix exists.

## 3. Environment Setup

Recommended Linux setup:

```bash
make preinstall
make install_verified
```

Useful alternatives:

```bash
make install_recent
make install-paper
```

The verified install path pins PyTorch, snnTorch, Tonic, and related packages
for reproducible local execution. CUDA is used automatically when available
unless a script is run with `--force-cpu`.

## 4. Supported Datasets and Methods

Datasets:

| CLI name | Dataset |
|---|---|
| `nmnist` | N-MNIST |
| `dvsgesture` | DVS128 Gesture |
| `ncaltech101` | N-Caltech101, resized to 128 x 128 |
| `cifar10dvs` | CIFAR10-DVS |

Methods:

| CLI name | Role |
|---|---|
| `raw_snn` | Direct raw-event SNN baseline |
| `frame_snn` | Dense frame/slice SNN baseline |
| `ba_snn` | Background Activity filter baseline |
| `stcf_rc_snn` | Row/column STCF baseline |
| `proposed_ref` | Proposed Stage 2 refractory/hot-pixel guard |
| `proposed_sup` | Proposed support-count ablation |
| `proposed_pol` | Proposed polarity-penalty ablation |
| `proposed_conf` | Full confidence-coded proposed method |
| `proposed_lowmem` | Low-memory approximation |
| `proposed_lowlat` | Analytical/profile-only low-latency variant |

The low-memory and low-latency methods are resource-footprint variants, not
primary accuracy baselines.

## 5. Fast Validation Commands

Run the test suite:

```bash
/home/user/anaconda3/envs/analog/bin/python -m pytest tests
```

Run the project smoke target:

```bash
make smoke-paper
```

Run a tiny training smoke test directly:

```bash
python3 -m src.experiments.train_eval \
  --dataset nmnist \
  --method proposed_conf \
  --seed 0 \
  --epochs-override 1 \
  --max-train-samples 64 \
  --max-val-samples 32 \
  --max-test-samples 32 \
  --force-cpu
```

## 6. Tuning Workflow

Fast tuning is the default and is intended for parameter selection, not final
paper numbers:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.experiments.tune_filters \
  --datasets nmnist \
  --preset fast
```

Fast preset defaults:

- methods: `ba_snn`, `stcf_rc_snn`, `proposed_ref`, `proposed_sup`,
  `proposed_pol`, `proposed_conf`, `proposed_lowmem`
- `top_k=3`
- `max_calibration_samples=64`
- compact method-aware grids
- proposed-filter grids use `k0 in {1, 2}` with `warmup_us=5000`
- `stage_b_epochs=5`
- `stage_b_max_train_samples=4096`
- `stage_b_max_val_samples=1024`
- `stage_b_max_test_samples=1024`
- `stage_b_max_slices=80`
- lightweight Stage B ranking by validation accuracy
- no Stage B robustness, profiling, or checkpoint saves unless requested

Paper preset is intentionally expensive:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.experiments.tune_filters \
  --datasets nmnist \
  --preset paper
```

Refresh affected proposed-method tuning after bootstrap-grid changes:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.experiments.tune_filters \
  --datasets nmnist \
  --methods proposed_sup proposed_pol proposed_conf proposed_lowmem \
  --preset fast
```

Useful tuning flags:

| Flag | Purpose |
|---|---|
| `--methods <...>` | Limit tuning to selected methods |
| `--max-grid <N>` | Cap the number of grid candidates |
| `--max-calibration-samples <N>` | Cap Stage A calibration samples |
| `--skip-stage-b` | Run filter-only Stage A tuning |
| `--full-stage-b-eval` | Enable full Stage B evaluation |
| `--stage-b-for-untuned` | Force Stage B for untuned singleton methods |
| `--run-timestamp <str>` | Use a fixed result timestamp |
| `--force-cpu` | Disable CUDA |

## 7. Training Workflow

Use latest tuning parameters for tunable methods:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.experiments.train_eval \
  --dataset nmnist \
  --method proposed_conf \
  --seed 0 \
  --use-latest-tuning
```

Use explicit manual parameters:

```bash
python -m src.experiments.train_eval \
  --dataset nmnist \
  --method proposed_conf \
  --seed 0 \
  --filter-params '{"k0": 0, "gamma": 1}'
```

If `--use-latest-tuning` and `--filter-params` are both present, manual JSON
values override the latest tuning values. The resolved source is written to
`summary.json` under `run_metadata.filter_params_source`.

For raw and frame baselines, omit `--use-latest-tuning` unless a tuning summary
exists for that method.

Common training flags:

| Flag | Purpose |
|---|---|
| `--epochs-override <N>` | Override dataset default epochs |
| `--max-train-samples <N>` | Limit training samples |
| `--max-val-samples <N>` | Limit validation samples |
| `--max-test-samples <N>` | Limit test samples |
| `--max-slices <N>` | Override temporal slice cap |
| `--skip-test-evaluation` | Skip test evaluation |
| `--skip-posthoc-metrics` | Skip robustness/profiling work |
| `--no-save-checkpoint` | Do not write `best_model.pt` |
| `--run-timestamp <str>` | Use a fixed result timestamp |
| `--force-cpu` | Disable CUDA |

## 8. Full Paper Run Order

Use this order for paper-grade results:

1. Tune filter parameters for each dataset.
2. Freeze selected parameters.
3. Train all primary methods with seeds `0`, `1`, and `2`.
4. Profile efficiency for the trained methods.
5. Aggregate tables and plots.
6. Update `main.tex` only from saved artifacts.

For final submission, either use `--preset paper` for exhaustive documented
selection or explicitly state that the faster selector was used only to choose
parameters before full training. The training/evaluation runs, not Stage B fast
tuning, are the source of manuscript accuracy numbers.

Tune one dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.experiments.tune_filters \
  --datasets nmnist \
  --preset fast
```

Train tunable/filter methods with latest tuning:

```bash
for SEED in 0 1 2; do
  for METHOD in ba_snn stcf_rc_snn proposed_ref proposed_sup proposed_pol proposed_conf; do
    CUDA_VISIBLE_DEVICES=0 python -m src.experiments.train_eval \
      --dataset nmnist \
      --method "$METHOD" \
      --seed "$SEED" \
      --use-latest-tuning
  done
done
```

Or run the executable workflow script:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_train_workflow.sh
```

Useful workflow overrides:

```bash
CUDA_VISIBLE_DEVICES=0 METHOD_SET=all_trainable scripts/run_train_workflow.sh
CUDA_VISIBLE_DEVICES=0 METHODS="proposed_conf proposed_lowmem" SEEDS="0 1 2" scripts/run_train_workflow.sh
CUDA_VISIBLE_DEVICES=0 DATASETS="nmnist dvsgesture" METHOD_SET=primary scripts/run_train_workflow.sh
DRY_RUN=1 scripts/run_train_workflow.sh
```

Train non-tuned baselines:

```bash
for SEED in 0 1 2; do
  for METHOD in raw_snn frame_snn; do
    CUDA_VISIBLE_DEVICES=0 python -m src.experiments.train_eval \
      --dataset nmnist \
      --method "$METHOD" \
      --seed "$SEED"
  done
done
```

Repeat with `--dataset dvsgesture`, `--dataset ncaltech101`, and
`--dataset cifar10dvs` when N-MNIST is validated.

## 9. Profiling and Aggregation

Profile a trained method:

```bash
python -m src.experiments.profile_efficiency \
  --dataset nmnist \
  --method proposed_conf \
  --seed 0
```

Train automatically if the summary is missing:

```bash
python -m src.experiments.profile_efficiency \
  --dataset nmnist \
  --method proposed_conf \
  --seed 0 \
  --train-if-missing
```

Aggregate current protocol runs:

```bash
python -m src.experiments.aggregate_results \
  --regen-roc \
  --roc-dataset nmnist
```

Expected aggregate outputs under `results/paper/aggregated/`:

- `main_accuracy.csv`
- `ablation.csv`
- `efficiency.csv`
- `confidence_ratio.csv`
- `aunc.csv`
- `event_metrics.csv`
- `accuracy_vs_compute.png`
- `roc_<dataset>.png`

Benchmark filter preprocessing throughput:

```bash
python -m src.experiments.benchmark_filter_throughput \
  --dataset nmnist \
  --method proposed_conf \
  --use-latest-tuning \
  --max-samples 200
```

## 10. Result Artifacts

Tuning outputs:

```text
results/paper/<dataset>/<method>/tuning_<timestamp>/tuning_summary.json
results/paper/tuning_global_summary_<timestamp>.json
```

Training outputs:

```text
results/paper/<dataset>/<method>/seed_<seed>__<timestamp>/
  summary.json
  robustness.json
  best_model.pt
```

Stage B tuning may also create temporary method directories such as:

```text
results/paper/<dataset>/<method>__tune_<rank>/
```

Split and slicing metadata is cached under:

```text
results/splits/
```

Every new experiment should be identified by its timestamped directory. Use
`--run-timestamp` only when a fixed identifier is needed for a controlled smoke
test or reproducibility check.

## 11. Configuration Reference

Global training configuration: `configs/training/default.json`

| Setting | Current value |
|---|---|
| `learning_rate` | `0.002` |
| `weight_decay` | `0.0001` |
| `gradient_clip` | `1.0` |
| `warmup_epochs` | `10` |
| `seeds` | `[0, 1, 2]` |
| `split_seed` | `2027` |
| `noise_ratios` | `[0.5, 1.0, 2.0, 5.0, 10.0]` |
| `profile_samples` | `200` |
| `inference_warmup` | `20` |
| `inference_runs` | `100` |

Dataset defaults:

| Dataset | Batch size | Epochs | Sensor size |
|---|---:|---:|---|
| N-MNIST | 128 | 100 | 34 x 34 x 2 |
| DVS128 Gesture | 32 | 150 | 128 x 128 x 2 |
| N-Caltech101 | 32 | 150 | 128 x 128 x 2 after resize |
| CIFAR10-DVS | 64 | 150 | 128 x 128 x 2 |

Default proposed-filter bootstrap settings:

| Setting | Current value |
|---|---|
| `k0` | `1` in method defaults; tuning grids use `{1, 2}` |
| `warmup_us` | `5000` |
| `gamma` | method-specific, e.g. `1` for `proposed_conf` |
| `tau_pair_us` | `1000` |
| `t_recover_us` | `1000000` |

The seed count is intentionally `0,1,2` because that is what `main.tex` and the
current config specify. Do not revive the old five-seed requirement unless the
paper and configs are changed together.

## 12. Paper-to-Code Alignment Rules

Keep these invariants synchronized with `main.tex`:

- Raw and accepted histories are distinct.
- Raw history updates on every incoming event.
- Accepted support history updates only on accepted events.
- Support is computed from accepted same-polarity neighbor history.
- The polarity stage raises the support threshold by `gamma * Q_i`; it is not a
  veto after a failed support test.
- Confidence `c=2` means the accepted event reached the high-support threshold.
- `proposed_lowmem` is an approximation.
- `proposed_lowlat` is analytical/profile-only in this software codebase.
- Full manuscript claims must be backed by saved artifacts, not by one-off
  console output.

## 13. Documentation Consolidation Audit

The former root Markdown files were consolidated as follows:

| Former file | Action |
|---|---|
| `summary.md` | Removed; the paper summary duplicates `main.tex`. |
| `literature_comparison_report.md` | Removed; related-work content belongs in `main.tex`. |
| `main_tex_revision_plan.md` | Removed; its proposed edits are either already in `main.tex` or tracked as open alignment items above. |
| `revision_plan.md` | Removed; stale implementation plan. |
| `performance_improvement_items.md` | Removed; useful runtime items are merged here, stale speculative items are omitted. |
| `code_audit.md` | Removed; current alignment rules are retained here without old line-number claims. |
| `implementation.md` | Removed; superseded by the implemented code, this manual, and `main.tex`. |
| `EXPERIMENT_GUIDE.md` | Removed; current commands are merged here in English. |
| `final_summary_report.md` | Removed; stale report-style claims are not kept as operating documentation. |

Do not use deleted root Markdown content as an authority. For science, use
`main.tex`; for execution, use this manual; for exact behavior, inspect the
code and saved artifacts.
