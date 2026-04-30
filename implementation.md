# Implementation Specification: Event-Domain Denoising for Direct SNN Ingestion

This document supersedes the previous `implementation.md`.

Its purpose is to make the revised manuscript implementable end to end:

1. implement the proposed algorithm exactly as defined in `main.tex`,
2. implement the mandatory baselines and ablations,
3. run multi-seed, multi-dataset experiments, and
4. generate the accuracy and efficiency evidence needed for the paper.

If this document conflicts with `main.tex`, `main.tex` wins.

## 1. Deliverables

The implementation must produce all of the following.

- A software implementation of the primary filter, named `proposed_balanced`.
- A software implementation of the low-memory approximation, named `proposed_lowmem`.
- Event-domain SNN training and evaluation code.
- Mandatory baselines:
  - `raw_snn`
  - `frame_snn`
  - `ba_snn`
  - `stcf_rc_snn`
  - `proposed_ref`
  - `proposed_sup`
  - `proposed_pol`
  - `proposed_conf`
- Event-level synthetic-noise evaluation for ROC/AUC.
- System-level multi-seed experiments on four datasets.
- Efficiency reports covering filter memory, model memory, preprocessing latency, end-to-end latency, and SNN compute.
- Final aggregated tables and plots ready to support the manuscript.

## 2. Non-Negotiable Alignment with the Paper

The implementation must preserve these rules exactly.

- Raw same-pixel history and accepted support history are separate states.
- Raw state is updated for every event, including rejected events.
- Accepted support state is updated only when an event is accepted.
- Stage 4 is a polarity-conditioned threshold increase, not a veto on `S_i = 0`.
- Confidence is derived from support after the revised acceptance rule.
- The primary implementation target is the balanced digital variant.
- The low-memory variant is an approximation and must be reported as such.
- The mixed-signal low-latency variant is analytical only and is not a software benchmark target.

## 3. Software Stack

Use the following stack.

- Python 3.10+
- PyTorch 2.2+
- snnTorch
- Tonic
- NumPy
- Numba
- pandas
- scikit-learn
- matplotlib
- seaborn
- psutil

Recommended install:

```bash
pip install torch torchvision torchaudio snntorch tonic numpy numba pandas scikit-learn matplotlib seaborn psutil
```

## 4. Project Layout

Use this layout.

```text
src/
  data/
    datasets.py
    event_io.py
    slicing.py
    augmentation.py
    noise_injection.py
  filters/
    proposed_balanced.py
    proposed_lowmem.py
    ba_filter.py
    stcf_rc.py
    metrics.py
  models/
    event_snn.py
    frame_snn.py
    sop_counter.py
  experiments/
    tune_filters.py
    train_eval.py
    aggregate_results.py
    profile_efficiency.py
  utils/
    seed.py
    logging.py
    serialization.py
configs/
  datasets/
  methods/
  training/
results/
```

The file and module names are fixed so experiment scripts can rely on them.

## 5. Common Data Interface

### 5.1 Event Record

All filters must use the same event representation.

```python
Event = {
    "x": uint16,
    "y": uint16,
    "t": int64,   # microseconds
    "p": int8,    # +1 or -1
}
```

Rules:

- timestamps are in microseconds,
- polarities are converted to `{-1, +1}`,
- all events are sorted by `(t, original_index)` using a stable sort,
- timestamps are shifted so each sample starts at `t = 0`,
- duplicate timestamps are allowed and must not be removed,
- no frame reconstruction is permitted in the proposed method or event-domain baselines.

### 5.2 Dataset List

The mandatory datasets are:

- N-MNIST
- DVS128 Gesture
- N-Caltech101
- DVS-CIFAR10

Use Tonic loaders where available.

### 5.3 Spatial Resolution Rules

Use native resolution unless explicitly resized below.

- N-MNIST: keep `34 x 34`
- DVS128 Gesture: keep `128 x 128`
- DVS-CIFAR10: keep `128 x 128`
- N-Caltech101: resize events to `128 x 128` with aspect-ratio-preserving scaling and zero-centered padding

The same spatial resolution must be used for all methods on a given dataset.

### 5.4 Train / Val / Test Splits

Use the official train/test split when available.

Create a validation split from the training split as follows:

- stratified by class,
- 10% of the official training split,
- fixed once using seed `2027`,
- reused for every method and every training seed.

Hyperparameter selection is done only on train/val. Test data is touched only once per seed after method settings are frozen.

## 6. Temporal Slicing for SNN Input

The proposed method remains event-domain until the SNN input interface. For software training, sparse event-count slices are allowed.

Use this procedure per dataset.

1. Compute sample durations on the training split after timestamp normalization.
2. Choose `time_bin_us` from `{1000, 2000, 5000}` as the smallest value such that:
   - `ceil(P99_duration / time_bin_us) <= 300`
3. Set:
   - `T_max = ceil(P99_duration / time_bin_us)`
4. During training and evaluation:
   - bin events into sparse slices of width `time_bin_us`,
   - pad shorter samples with empty slices,
   - truncate longer samples after `T_max`.

This procedure is fixed per dataset and must be computed once from the training split before any model training.

## 7. Primary Algorithm: `proposed_balanced`

### 7.1 State Definition

The primary implementation must use these states exactly.

Raw state:

```text
M_raw[x, y] = (
  T_raw_last[x, y],
  p_raw_last[x, y],
  R[x, y],
  U[x, y],
  H[x, y]
)
```

Accepted state:

```text
M_acc[x, y] = (
  T_acc_pos[x, y],
  T_acc_neg[x, y]
)
```

State dtypes:

- `T_raw_last`, `T_acc_pos`, `T_acc_neg`: `int64`
- `p_raw_last`: `int8`
- `R`: `float32`
- `U`: `uint16`
- `H`: `bool`

Initialization:

- timestamps: `-10**18`
- polarity: `0`
- `R = 0`
- `U = 0`
- `H = False`

### 7.2 Exact Per-Event Update Rule

For each event `e_i = (x_i, y_i, t_i, p_i)`:

1. Compute raw same-pixel lag:
   - `dt_raw = t_i - T_raw_last[y_i, x_i]`
2. Compute pair flag:
   - `Q_i = 1` if `p_i != p_raw_last[y_i, x_i]` and `dt_raw < tau_pair_us`, else `0`
3. Update rate estimate:
   - `R[y_i, x_i] = alpha * (1 / max(dt_raw, eps_us)) + (1 - alpha) * R[y_i, x_i]`
4. Stage 2 guard:
   - if `H[y_i, x_i]` is `True` or `dt_raw < tau_ref_dig_us`, then set `S_i = 0`, `A_i = 0`
5. Else Stage 3 support:
   - count same-polarity accepted neighbors in a `3 x 3` neighborhood excluding the center pixel
   - use accepted timestamps only
   - `S_i = number of neighbors with 0 < t_i - T_acc_p[ny, nx] < delta_t_us`
6. Stage 4 thresholding:
   - `A_i = 1` iff `S_i >= K0 + gamma * Q_i`
7. Unsupported streak:
   - if `S_i == 0`, increment `U[y_i, x_i]`
   - else reset `U[y_i, x_i] = 0`
8. Hot-pixel update:
   - if `R[y_i, x_i] > R_max_hz` and `U[y_i, x_i] >= U_hot`, set `H[y_i, x_i] = True`
9. Stage 5 confidence:
   - if `A_i == 0`, `c_i = 0`
   - if `A_i == 1` and `S_i < K_high`, `c_i = 1`
   - if `A_i == 1` and `S_i >= K_high`, `c_i = 2`
10. Accepted-state update:
   - if `A_i == 1` and `p_i == +1`, set `T_acc_pos[y_i, x_i] = t_i`
   - if `A_i == 1` and `p_i == -1`, set `T_acc_neg[y_i, x_i] = t_i`
11. Raw-state update:
   - always set `T_raw_last[y_i, x_i] = t_i`
   - always set `p_raw_last[y_i, x_i] = p_i`

This order is mandatory.

### 7.3 Neighborhood and Edge Handling

Use a `3 x 3` neighborhood with radius `1`.

Rules:

- clip neighborhood queries to valid image bounds,
- never wrap around image borders,
- exclude the center pixel from support,
- count only timestamps with strictly positive lag,
- use same-polarity accepted timestamps only.

### 7.4 Primary Hyperparameters

These are the primary search parameters for the balanced implementation.

- `tau_ref_dig_us`: `[500, 1000, 2000]`
- `delta_t_us`: `[1000, 2000, 5000]`
- `K0`: `[1, 2]`
- `gamma`: `[0, 1, 2]`
- `tau_pair_us`: `[500, 1000, 2000]`
- `K_high`: `[2, 3, 4]`

These are fixed defaults unless validation shows clear gains.

- `alpha = 0.01`
- `eps_us = 1`
- `U_hot = 32`

Set `R_max_hz` per dataset from the training split:

1. run the filter once on a small calibration subset with `H=False` for all pixels,
2. collect per-pixel raw instantaneous rate estimates,
3. set `R_max_hz = clip(P99.9_nonzero_rate, 1000, 20000)`.

Do not tune `alpha`, `eps_us`, or `U_hot` unless there is a demonstrated failure mode.

### 7.5 Stage 1 in Offline Dataset Experiments

Stage 1 is sensor-side bias shaping. That cannot be retroactively applied to prerecorded benchmark datasets.

Therefore:

- the software benchmark implementation must expose a `stage1_mode`,
- for offline dataset experiments, `stage1_mode = "offline_noop"`,
- this mode does not delete events,
- it only records calibration statistics for reporting and for selecting the initial filter search range,
- the paper's main benchmark claims are based on Stages 2 to 6.

If live camera experiments are added later, Stage 1 may be implemented as real bias control, but that is not required for the paper benchmark.

## 8. Low-Memory Approximation: `proposed_lowmem`

This variant is required for the memory-efficiency table.

It is not the primary accuracy model.

### 8.1 State Definition

Use row/column state only.

```text
T_row_raw[H], T_col_raw[W]
T_row_acc[H], T_col_acc[W]
```

No exact same-pixel polarity is stored.

Consequences:

- Stage 2 is approximate,
- Stage 3 is approximate,
- Stage 4 is unavailable,
- exact hot-pixel logic is unavailable,
- Stage 5 remains available but is based on approximate support.

### 8.2 Update Rule

For each event:

1. approximate guard lag:
   - `dt_raw = min(t_i - T_row_raw[y_i], t_i - T_col_raw[x_i])`
2. reject if `dt_raw < tau_ref_dig_us`
3. approximate support:
   - `S_rc = sum of indicators over row and column neighbors`
   - use `y_i-1`, `y_i+1`, `x_i-1`, `x_i+1` where valid
   - count recent accepted row/column timestamps within `delta_t_us`
4. accept iff `S_rc >= K0`
5. confidence:
   - `0`, `1`, `2` using `K_high` on `S_rc`
6. update accepted row/column timestamps only on accepted events
7. update raw row/column timestamps on every event

Use this method only in:

- the memory-focused comparison table,
- the supplementary system-level experiment,
- the low-memory ablation.

Do not use it as the main proposed method in the accuracy table.

## 9. Mandatory Baselines

Only the baselines fully specified here are mandatory.

### 9.1 `raw_snn`

- no filtering,
- direct event-to-SNN ingestion,
- binary confidence only,
- same SNN backbone and training schedule as the proposed method.

### 9.2 `frame_snn`

This is the conventional dense baseline.

- accumulate raw events into dense ON/OFF count frames using the same `time_bin_us` and `T_max` as the event-domain methods,
- use the same temporal backbone depth and channel counts,
- dense inputs are float32 ON/OFF counts,
- no event-domain filtering or confidence coding.

This baseline is mandatory for the representation comparison and the memory/latency comparison.

### 9.3 `ba_snn`

Classical any-polarity background-activity filter plus the same downstream SNN.

Definition:

- same-pixel refractory is not included,
- maintain one raw timestamp map `T_last[y, x]`,
- for each event, count any-polarity neighbors in the `3 x 3` neighborhood,
- accept iff neighbor count `>= 1`,
- update `T_last` for every event,
- binary output only, no confidence coding.

### 9.4 `stcf_rc_snn`

Row/column spatiotemporal correlation filter plus the same downstream SNN.

Definition:

- maintain row and column raw timestamps only,
- accept iff at least one neighboring row or column has recent activity within `delta_t_us`,
- no polarity awareness,
- no confidence coding,
- update row/column state for every event.

This is the low-memory classical baseline.

### 9.5 Proposed Ablations

These are mandatory.

- `proposed_ref`
  - Stage 2 only
  - no support thresholding
  - no confidence coding
- `proposed_sup`
  - Stages 2 and 3
  - accept iff `S_i >= K0`
  - equivalent to `gamma = 0`
- `proposed_pol`
  - Stages 2, 3, 4
  - accept iff `S_i >= K0 + gamma * Q_i`
  - no confidence coding
- `proposed_conf`
  - full proposed method
  - this is the primary model reported in the main accuracy table

## 10. Input Encoding for the SNN

### 10.1 Event-Domain Methods

All event-domain methods use the same encoder.

For each time slice, build four sparse maps:

- ON, confidence 1
- ON, confidence 2
- OFF, confidence 1
- OFF, confidence 2

For methods without confidence coding:

- put accepted ON events into the ON, confidence 1 map,
- put accepted OFF events into the OFF, confidence 1 map,
- leave the confidence 2 maps empty.

Then collapse to two channels with learnable input weights:

```text
X_on  = w_low * ON_c1  + w_high * ON_c2
X_off = w_low * OFF_c1 + w_high * OFF_c2
```

`w_low` and `w_high` are trainable scalar parameters with initialization:

- `w_low = 1.0`
- `w_high = 1.5`

This matches the manuscript's dual-population input with confidence-modulated synaptic weights.

### 10.2 Frame Baseline

The frame baseline uses dense `float32` ON/OFF counts per time bin:

```text
X_frame[t, 0, y, x] = ON count
X_frame[t, 1, y, x] = OFF count
```

No confidence channels are used.

## 11. SNN Models

### 11.1 Event-Domain Backbone

Use the same backbone for:

- `raw_snn`
- `ba_snn`
- `stcf_rc_snn`
- `proposed_ref`
- `proposed_sup`
- `proposed_pol`
- `proposed_conf`
- `proposed_lowmem`

Architecture:

1. input weighting layer:
   - learnable `w_low`, `w_high`
2. coincidence-and-inhibition front end:
   - `Conv2d(2, 16, kernel_size=3, padding=1, bias=False)`
   - `LIF(beta=0.9, inhibition=True)`
3. feature block 1:
   - `Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)`
   - `BatchNorm2d(32)`
   - `LIF(beta=0.9)`
4. feature block 2:
   - `Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)`
   - `BatchNorm2d(64)`
   - `LIF(beta=0.9)`
5. pooling:
   - `AdaptiveAvgPool2d((4, 4))`
6. classifier:
   - `Linear(64 * 4 * 4, num_classes, bias=False)`
   - output LIF layer

Loss:

- cross-entropy on output spike rate

Prediction:

- class with largest summed output spike count across time

### 11.2 Frame Baseline Backbone

Use the same architecture depth and channel counts as the event backbone, but consume dense ON/OFF frame slices.

This keeps the model class comparable while exposing the dense representation cost.

## 12. Training Protocol

### 12.1 Seeds

Use exactly five training seeds:

```text
3407, 3413, 3421, 3433, 3449
```

Set all of the following from the same seed:

- Python random
- NumPy
- PyTorch CPU
- PyTorch CUDA
- dataloader worker seeds

### 12.2 Optimizer and Schedule

Use the same optimizer schedule for all methods on a given dataset.

- optimizer: `AdamW`
- learning rate: `2e-3`
- weight decay: `1e-4`
- gradient clipping: `1.0`
- scheduler: cosine decay with 10-epoch warmup

Epochs:

- N-MNIST: `100`
- DVS128 Gesture: `150`
- N-Caltech101: `150`
- DVS-CIFAR10: `150`

Early stopping:

- monitor validation accuracy,
- patience `20`,
- restore best validation checkpoint.

### 12.3 Batch Size

Use these defaults.

- N-MNIST: `128`
- DVS128 Gesture: `32`
- N-Caltech101: `32`
- DVS-CIFAR10: `64`

Reduce batch size only if memory requires it. If changed, apply the same rule to all methods on that dataset.

### 12.4 Data Augmentation

Use event-domain augmentation before slicing.

Mandatory augmentations:

- random horizontal flip for DVS128 Gesture, N-Caltech101, DVS-CIFAR10
- random spatial translation up to `+/- 4` pixels
- random temporal jitter up to `+/- 1` time bin

N-MNIST:

- translation up to `+/- 2` pixels
- no horizontal flip

Augmentations must be identical across methods on the same dataset.

## 13. Hyperparameter Selection

Use this exact selection procedure.

### 13.1 Filter Search Subset

Create a calibration subset from the training split:

- 10% of training samples,
- capped at 1000 samples,
- stratified by class,
- fixed once with seed `2027`.

### 13.2 Two-Stage Selection

Stage A: filter-only pruning

- run event-level synthetic-noise evaluation on the calibration subset,
- keep the top 5 parameter settings per filter family by mean ROC AUC across noise suites.

Stage B: system-level selection

- train one model per surviving setting using seed `3407`,
- choose the setting with the best validation accuracy,
- if two settings are within `0.1` accuracy points, choose the one with lower end-to-end latency,
- freeze the chosen settings before the five-seed test runs.

The same selection budget must be used for every method with tunable filter parameters.

## 14. Event-Level Validation

Event-level validation is mandatory because it supports the filter claims independently of the downstream classifier.

### 14.1 Synthetic Noise Suites

Generate three noise suites on top of real signal samples from the calibration subset.

BA-only suite:

- sample independent noise events uniformly over pixel, time, and polarity,
- total noise-to-signal ratios: `{0.5, 1, 2, 5}`.

Shot-only suite:

- sample same-pixel alternating-polarity pairs,
- choose first polarity uniformly,
- choose second polarity as the opposite,
- choose short lag uniformly in `[50, tau_pair_us]`,
- total pair-event ratios: `{0.5, 1, 2, 5}`.

Mixed suite:

- 70% BA noise events,
- 30% shot-pair events,
- total noise-to-signal ratios: `{0.5, 1, 2, 5}`.

Every injected event has a known label, so event-level ground truth is exact.

### 14.2 Mandatory Event-Level Metrics

These metrics are mandatory.

- ROC curve
- AUC
- true positive rate at fixed false positive rates `{1%, 5%, 10%}`
- accepted-event ratio under each noise suite

ESR and NIR/RIN may be added later, but they are not required for the first complete implementation because the manuscript does not fully define them.

## 15. System-Level Benchmark Matrix

Run the following methods on every dataset and every seed.

- `raw_snn`
- `frame_snn`
- `ba_snn`
- `stcf_rc_snn`
- `proposed_ref`
- `proposed_sup`
- `proposed_pol`
- `proposed_conf`
- `proposed_lowmem`

This yields:

- 4 datasets
- 9 methods
- 5 seeds
- 180 training runs

Do not shrink the matrix for the final paper results.

## 16. Mandatory System-Level Metrics

Report these metrics for every dataset, method, and seed.

Accuracy metrics:

- top-1 test accuracy
- validation accuracy of the selected checkpoint

Event-efficiency metrics:

- accepted-event ratio
- compression ratio
- average accepted events per sample

Filter efficiency metrics:

- algorithmic filter state memory in bytes
- measured preprocessing throughput in events/second
- measured preprocessing latency per sample

Model efficiency metrics:

- parameter count
- parameter memory in bytes
- peak activation memory in bytes
- SNN synaptic operations
- inference latency per sample

End-to-end metrics:

- preprocessing latency + inference latency
- total peak memory

## 17. How to Measure Compute and Memory

### 17.1 Algorithmic Filter Memory

Use algorithmic state memory, not Python object overhead.

Mandatory formulas:

- `proposed_balanced`:
  - `15 * W * H` bytes
- `proposed_lowmem`:
  - report exact row/column array budget from implemented dtypes
- `ba_snn`:
  - `8 * W * H` bytes if implemented with one `int64` timestamp surface
- `stcf_rc_snn`:
  - `2 * (W + H) * 8` bytes for `int64` row/column timestamps
- `frame_snn` input buffer:
  - `T_max * 2 * H * W * 4` bytes

Report the formula and the realized per-dataset values.

### 17.2 Preprocessing Throughput and Latency

Measure on CPU only.

Procedure:

1. single process
2. fixed thread count: `1`
3. warm up on 50 samples
4. time 200 samples
5. use median per-sample latency
6. throughput = total processed events / total elapsed time

### 17.3 Inference Latency

Measure on the same device for all methods on a given run.

Procedure:

1. warm up 20 forward passes
2. time 100 forward passes
3. synchronize CUDA before and after timing if GPU is used
4. report median latency

### 17.4 Synaptic Operations

Count SNN compute with hooks.

For each layer and timestep:

- `Conv2d`: `nnz(input_spikes) * kernel_h * kernel_w * out_channels / groups`
- `Linear`: `nnz(input_spikes) * out_features`

Sum over all timesteps and all layers.

For `frame_snn`, count dense MACs instead and report them separately as dense baseline compute.

### 17.5 Peak Activation Memory

Use forward hooks and tensor shape logging.

Report the maximum live activation bytes during one forward pass at test time.

## 18. Statistical Reporting

For every dataset and method:

- report mean and standard deviation over the 5 seeds,
- keep the per-seed values in the artifact store,
- use paired comparisons against `proposed_conf`.

For the paper:

- use paired t-test across the 5 seeds for accuracy,
- use paired t-test across the 5 seeds for end-to-end latency,
- report p-values in the supplementary material,
- do not claim superiority from a single seed.

## 19. Required Tables and Figures

The implementation must generate these outputs.

Table 1: main accuracy table

- rows: methods
- columns: datasets
- values: test accuracy mean +- std

Table 2: efficiency table

- accepted-event ratio
- preprocessing latency
- end-to-end latency
- filter state memory
- peak activation memory
- SOPs

Table 3: ablation table

- `raw_snn`
- `proposed_ref`
- `proposed_sup`
- `proposed_pol`
- `proposed_conf`

Figure 1: ROC curves on synthetic noise suites

- BA-only
- Shot-only
- Mixed

Figure 2: accuracy versus SOPs

- one point per method per dataset

Figure 3: accuracy versus filter memory

- include `proposed_balanced`, `proposed_lowmem`, `ba_snn`, `stcf_rc_snn`, `frame_snn`

## 20. Result Logging and Artifact Format

Each run must write:

```text
results/{dataset}/{method}/seed_{seed}/
  config.yaml
  metrics.json
  val_history.csv
  test_predictions.pt
  filter_stats.json
  efficiency.json
```

Each `metrics.json` must include:

- dataset
- method
- seed
- best_epoch
- val_accuracy
- test_accuracy
- accepted_event_ratio
- compression_ratio
- sop_count
- filter_memory_bytes
- parameter_memory_bytes
- peak_activation_memory_bytes
- preprocess_latency_ms
- inference_latency_ms
- end_to_end_latency_ms

## 21. Unit and Integration Tests

The code is not complete until these tests pass.

Unit tests:

- rejected events update raw state
- rejected events do not update accepted state
- `gamma > 0` changes acceptance when `Q_i = 1`
- support counts use accepted history only
- confidence levels match the manuscript definition
- low-memory variant never uses same-pixel polarity state

Integration tests:

- deterministic outputs with fixed seeds
- same sample gives identical filter output across repeated runs
- event counts before and after filtering match logged statistics
- algorithmic memory formulas match implemented state allocations

## 22. Execution Order

Follow these steps in order.

1. Implement data loading, normalization, and temporal slicing.
2. Implement `proposed_balanced` exactly.
3. Add unit tests for state semantics.
4. Implement `raw_snn` and `frame_snn`.
5. Implement `ba_snn` and `stcf_rc_snn`.
6. Implement the ablation variants.
7. Implement the event-domain SNN and the frame baseline backbone.
8. Implement event-level synthetic-noise evaluation.
9. Run filter selection on the calibration subset.
10. Freeze method hyperparameters.
11. Run the full 180-run multi-seed benchmark matrix.
12. Run efficiency profiling.
13. Aggregate results and generate tables and figures.

Do not start the full benchmark before step 10 is complete.

## 23. Acceptance Criteria

The work is complete only when all of the following are true.

- The balanced implementation matches the revised manuscript semantics exactly.
- The low-memory approximation is implemented and clearly labeled as approximate.
- All mandatory baselines and ablations are implemented.
- All four datasets are trained and evaluated with 5 seeds each.
- Event-level ROC/AUC results are available for the proposed method and classical baselines.
- System-level tables contain both accuracy and efficiency metrics.
- Every reported number is backed by a saved artifact.
- The manuscript can cite the resulting tables without additional implementation work.
