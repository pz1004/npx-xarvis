#!/usr/bin/env bash
set -Eeuo pipefail

# Multi-dataset, multi-method, multi-seed train_eval workflow.
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0 scripts/run_train_workflow.sh
#   CUDA_VISIBLE_DEVICES=0 DATASETS="nmnist dvsgesture" METHOD_SET=primary scripts/run_train_workflow.sh
#   CUDA_VISIBLE_DEVICES=0 METHODS="proposed_conf proposed_lowmem" SEEDS="0 1 2" scripts/run_train_workflow.sh
#   CUDA_VISIBLE_DEVICES=0 DRY_RUN=1 scripts/run_train_workflow.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/anaconda3/envs/analog/bin/python}"

DATASETS="${DATASETS:-nmnist}"
SEEDS="${SEEDS:-0 1 2}"

# METHOD_SET values:
#   primary: paper-facing trainable methods, excluding resource-only variants
#   tuned: tunable filter methods only
#   baselines: raw/frame/filter baselines only
#   proposed: proposed balanced ablations/full model only
#   all_trainable: all normally trainable methods, including proposed_lowmem
# Override entirely with METHODS="method_a method_b".
METHOD_SET="${METHOD_SET:-primary}"

PRIMARY_METHODS=(
  raw_snn
  frame_snn
  ba_snn
  stcf_rc_snn
  proposed_ref
  proposed_sup
  proposed_pol
  proposed_conf
)
TUNED_METHODS=(
  ba_snn
  stcf_rc_snn
  proposed_ref
  proposed_sup
  proposed_pol
  proposed_conf
  proposed_lowmem
)
BASELINE_METHODS=(
  raw_snn
  frame_snn
  ba_snn
  stcf_rc_snn
)
PROPOSED_METHODS=(
  proposed_ref
  proposed_sup
  proposed_pol
  proposed_conf
)
ALL_TRAINABLE_METHODS=(
  raw_snn
  frame_snn
  ba_snn
  stcf_rc_snn
  proposed_ref
  proposed_sup
  proposed_pol
  proposed_conf
  proposed_lowmem
)
PROFILE_ONLY_METHODS=(
  proposed_lowlat
)

if [[ -n "${METHODS:-}" ]]; then
  read -r -a SELECTED_METHODS <<< "${METHODS}"
else
  case "${METHOD_SET}" in
    primary)
      SELECTED_METHODS=("${PRIMARY_METHODS[@]}")
      ;;
    tuned)
      SELECTED_METHODS=("${TUNED_METHODS[@]}")
      ;;
    baselines)
      SELECTED_METHODS=("${BASELINE_METHODS[@]}")
      ;;
    proposed)
      SELECTED_METHODS=("${PROPOSED_METHODS[@]}")
      ;;
    all_trainable)
      SELECTED_METHODS=("${ALL_TRAINABLE_METHODS[@]}")
      ;;
    *)
      echo "Unknown METHOD_SET='${METHOD_SET}'" >&2
      echo "Use one of: primary, tuned, baselines, proposed, all_trainable" >&2
      exit 2
      ;;
  esac
fi

read -r -a SELECTED_DATASETS <<< "${DATASETS}"
read -r -a SELECTED_SEEDS <<< "${SEEDS}"

TUNING_METHODS=(
  ba_snn
  stcf_rc_snn
  proposed_ref
  proposed_sup
  proposed_pol
  proposed_conf
  proposed_lowmem
)
UNTUNED_METHODS=(
  raw_snn
  frame_snn
)

is_in_list() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

validate_method() {
  local method="$1"
  if is_in_list "${method}" "${ALL_TRAINABLE_METHODS[@]}"; then
    return 0
  fi
  if is_in_list "${method}" "${PROFILE_ONLY_METHODS[@]}"; then
    echo "Method '${method}' is profile-only and is not part of this supervised training workflow." >&2
    echo "Use profile/analytical tooling for '${method}' instead." >&2
    return 1
  fi
  echo "Unknown method '${method}'. Check configs/methods/<method>.json." >&2
  return 1
}

EXTRA_ARGS=()
if [[ -n "${EPOCHS_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--epochs-override "${EPOCHS_OVERRIDE}")
fi
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_VAL_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max-val-samples "${MAX_VAL_SAMPLES}")
fi
if [[ -n "${MAX_TEST_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max-test-samples "${MAX_TEST_SAMPLES}")
fi
if [[ -n "${MAX_SLICES:-}" ]]; then
  EXTRA_ARGS+=(--max-slices "${MAX_SLICES}")
fi
if [[ "${SKIP_TEST_EVALUATION:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-test-evaluation)
fi
if [[ "${SKIP_POSTHOC_METRICS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-posthoc-metrics)
fi
if [[ "${NO_SAVE_CHECKPOINT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-save-checkpoint)
fi
if [[ "${FORCE_CPU:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force-cpu)
fi
if [[ -n "${RUN_TIMESTAMP:-}" ]]; then
  EXTRA_ARGS+=(--run-timestamp "${RUN_TIMESTAMP}")
fi

echo "Root      : ${ROOT_DIR}"
echo "Python    : ${PYTHON_BIN}"
echo "Datasets  : ${SELECTED_DATASETS[*]}"
echo "Methods   : ${SELECTED_METHODS[*]}"
echo "Seeds     : ${SELECTED_SEEDS[*]}"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"

for method in "${SELECTED_METHODS[@]}"; do
  validate_method "${method}"
done

for dataset in "${SELECTED_DATASETS[@]}"; do
  for method in "${SELECTED_METHODS[@]}"; do
    METHOD_ARGS=()
    if is_in_list "${method}" "${TUNING_METHODS[@]}"; then
      METHOD_ARGS+=(--use-latest-tuning)
    fi
    if is_in_list "${method}" "${UNTUNED_METHODS[@]}"; then
      METHOD_ARGS=()
    fi

    for seed in "${SELECTED_SEEDS[@]}"; do
      CMD=(
        "${PYTHON_BIN}"
        -m src.experiments.train_eval
        --root "${ROOT_DIR}"
        --dataset "${dataset}"
        --method "${method}"
        --seed "${seed}"
        "${METHOD_ARGS[@]}"
        "${EXTRA_ARGS[@]}"
      )

      echo
      echo "==> dataset=${dataset} method=${method} seed=${seed}"
      printf '    %q' "${CMD[@]}"
      echo

      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        continue
      fi
      "${CMD[@]}"
    done
  done
done
