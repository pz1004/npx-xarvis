# 실험 프로토콜 실행 가이드

이 문서는 `main.tex` 논문의 "Evaluation Protocol and Reproducibility" 섹션에 명시된 Q1-Q6 실험을 수행하기 위한 상세 가이드라인을 제공합니다. `BA Filter`, `STCF-RC`, 그리고 제안된 필터(`Proposed +REF`, `Proposed +SUP`, `Proposed +OPP`, `Proposed +CONF`)의 다양한 변체들을 대상으로 실험을 진행합니다.

## 1. 전제 조건 및 환경 설정

`MANUAL.md` 파일의 "1. 시스템 요구사항 및 설치" 섹션을 참조하여 프로젝트를 설정하고 필요한 의존성을 설치하십시오.

## 2. 데이터셋 준비

`run_experiment_v2.py`는 합성 노이즈 데이터셋을 사용합니다. 실제 데이터셋(N-MNIST, DVS128 Gesture 등)을 사용하려면 다음 섹션의 `tune_filters.py` 및 `train_eval.py` 사용법을 따르십시오.

## 3. Q1, Q3, Q6 실험: 이벤트 레벨 필터링 성능 평가 (AUC, EKR, ESR)

이 실험은 `run_experiment_v2.py` 스크립트를 사용하여 다양한 노이즈 환경에서 필터들의 이벤트 레벨 성능(AUC, EKR, ESR)을 평가합니다.

### 3.1. `run_experiment_v2.py` 실행

`run_experiment_v2.py` 스크립트는 합성 노이즈(BA, SHOT, MIXED)를 주입한 이벤트 스트림에 대해 `BA Filter`, `STCF-RC`, `Proposed REF`, `Proposed SUP`, `Proposed OPP`, `Proposed CONF` 필터를 적용하고 성능을 측정합니다.

```bash
cd /mnt/desktop/npx-xarvis
python3 run_experiment_v2.py
```

**출력:**
실행이 완료되면 `experiment_results_v2.json` 파일에 모든 실험 결과가 저장됩니다. 콘솔에는 요약된 결과 테이블이 출력됩니다.

### 3.2. 결과 시각화

`plot_results.py` 스크립트를 사용하여 `experiment_results_v2.json` 파일의 결과를 시각화할 수 있습니다.

```bash
cd /mnt/desktop/npx-xarvis
python3 plot_results.py
```

**출력:**
`fig_auc_vs_noise_ratio.png`, `fig_aggregate_comparison.png`, `fig_throughput_speedup.png`와 같은 이미지 파일들이 생성됩니다.

**해석:**
*   **Q1 답변:** `fig_auc_vs_noise_ratio.png` 및 `fig_aggregate_comparison.png`를 통해 각 필터의 노이즈 식별력(AUC)을 비교할 수 있습니다. `Proposed CONF`가 일반적으로 가장 높은 AUC를 보일 것입니다.
*   **Q3 답변:** `Proposed REF`, `Proposed SUP`, `Proposed OPP`, `Proposed CONF` 간의 성능 차이를 비교하여 각 Stage의 점진적인 기여도를 정량화할 수 있습니다.
*   **Q6 답변:** `fig_esr_vs_noise_ratio.png`를 통해 각 필터의 ESR 값을 비교하여 신호 구조 보존 능력을 평가할 수 있습니다. `Proposed CONF`는 높은 AUC를 유지하면서도 합리적인 ESR을 보여줄 것입니다.

## 4. Q2, Q5 실험: 시스템 레벨 SNN 분류 정확도 및 효율성 평가 (실제 데이터셋)

이 질문들은 `src/experiments/train_eval.py` 스크립트를 사용하여 SNN을 학습시키고 평가함으로써 답변할 수 있습니다. 이 스크립트는 다양한 필터링 방법(Raw, Frame, BA, STCF, Proposed variants)을 통해 전처리된 이벤트 스트림을 SNN에 입력하여 분류 정확도 및 기타 시스템 레벨 메트릭을 측정합니다.

### 4.1. 필터 파라미터 튜닝 (Q4 관련)

`train_eval.py`를 실행하기 전에, 각 필터의 최적 파라미터를 결정하기 위한 튜닝 과정이 필요합니다. `src/experiments/tune_filters.py` 스크립트가 이 역할을 수행합니다. `common.py`의 `candidate_param_grid` 함수에 정의된 파라미터 그리드를 참조하여 튜닝을 수행합니다.

```bash
# 예시: Proposed +CONF 필터의 튜닝 (N-MNIST 데이터셋)
cd /mnt/desktop/npx-xarvis
python3 src/experiments/tune_filters.py \
    --dataset NMNIST \
    --method proposed_conf \
    --tuning-metric auc # 또는 ekr, esr 등

# 다른 필터(ba_snn, stcf_rc_snn, proposed_ref, proposed_sup, proposed_pol)에 대해서도 반복
# --dataset 인자를 변경하여 다른 데이터셋(예: DVSGesture)에 대해서도 튜닝을 수행할 수 있습니다.
```

**출력:** 각 튜닝 결과는 `results/paper/<dataset>/<method>/tuning_summary.json` 파일에 저장됩니다. 이 파일에서 `selected` 필드의 `filter_params`를 확인하여 SNN 학습에 사용할 최적 파라미터를 얻을 수 있습니다.

### 4.2. SNN 학습 및 평가

튜닝된 필터 파라미터를 사용하여 각 방법별로 SNN을 학습하고 평가합니다. `train_eval.py` 스크립트는 `--method` 인자를 통해 사용할 필터 방법을 지정하고, `--dataset` 인자를 통해 데이터셋을 지정합니다. `--filter-params` 인자를 사용하여 튜닝된 파라미터를 직접 전달할 수 있습니다.

```bash
cd /mnt/desktop/npx-xarvis

# 1. Raw 이벤트 (노이즈 제거 없음) SNN 학습 및 평가
python3 src/experiments/train_eval.py --dataset NMNIST --method raw_snn

# 2. Frame 기반 SNN 학습 및 평가
python3 src/experiments/train_eval.py --dataset NMNIST --method frame_snn

# 3. BA Filter SNN 학습 및 평가 (튜닝된 파라미터 사용)
# 예시: BA Filter의 최적 delta_t_us가 2000이라고 가정
python3 src/experiments/train_eval.py --dataset NMNIST --method ba_snn --filter-params '{"delta_t_us": 2000}'

# 4. STCF Filter SNN 학습 및 평가 (튜닝된 파라미터 사용)
# 예시: STCF Filter의 최적 delta_t_us가 1000이라고 가정
python3 src/experiments/train_eval.py --dataset NMNIST --method stcf_rc_snn --filter-params '{"delta_t_us": 1000}'

# 5. Proposed +REF SNN 학습 및 평가
# 예시: Proposed +REF의 최적 tau_ref_dig_us가 1000이라고 가정
python3 src/experiments/train_eval.py --dataset NMNIST --method proposed_ref --filter-params '{"tau_ref_dig_us": 1000}'

# 6. Proposed +SUP SNN 학습 및 평가
# 예시: Proposed +SUP의 최적 파라미터 조합을 튜닝 결과에서 가져와 사용
python3 src/experiments/train_eval.py --dataset NMNIST --method proposed_sup --filter-params '{"tau_ref_dig_us": 1000, "delta_t_us": 2000, "k0": 1, "gamma": 0, "tau_pair_us": 1000, "k_high": 2, "t_recover_us": 1000000}'

# 7. Proposed +OPP SNN 학습 및 평가
# 예시: Proposed +OPP의 최적 파라미터 조합을 튜닝 결과에서 가져와 사용
python3 src/experiments/train_eval.py --dataset NMNIST --method proposed_pol --filter-params '{"tau_ref_dig_us": 1000, "delta_t_us": 2000, "k0": 1, "gamma": 1, "tau_pair_us": 1000, "k_high": 2, "t_recover_us": 1000000}'

# 8. Proposed +CONF SNN 학습 및 평가
# 예시: Proposed +CONF의 최적 파라미터 조합을 튜닝 결과에서 가져와 사용
python3 src/experiments/train_eval.py --dataset NMNIST --method proposed_conf --filter-params '{"tau_ref_dig_us": 1000, "delta_t_us": 2000, "k0": 1, "gamma": 1, "tau_pair_us": 1000, "k_high": 3, "t_recover_us": 1000000}'

# 각 데이터셋(DVS128 Gesture, N-Caltech101, CIFAR10-DVS)에 대해서도 위 과정을 반복합니다.
```

**출력:** 각 실행 결과는 `results/paper/<dataset>/<method>/seed_<seed>/metrics.json` 파일에 저장됩니다. 이 파일에는 SNN의 분류 정확도, EKR, CR, SOPs, 메모리 사용량, $w_{\text{high}}/w_{\text{low}}$ 비율 등이 포함됩니다.

**해석:**
*   **Q2 답변:** `metrics.json` 파일에서 각 방법별 SNN 분류 정확도를 비교하여 제안된 노이즈 제거 방법이 Raw 및 Frame 기반 베이스라인 대비 성능 향상을 가져오는지 확인할 수 있습니다. 또한, EKR, CR, SOPs 등을 비교하여 효율성 측면도 평가합니다.
*   **Q5 답변:** `Proposed +CONF` 방법의 `metrics.json` 파일에서 `w_high_w_low_ratio` 값을 확인하여 SNN이 신뢰도 신호를 활용하는지 판단할 수 있습니다. 이 값이 1보다 크다면 SNN이 신뢰도에 따라 가중치를 다르게 학습했음을 의미합니다.

## 5. Q4: 파라미터 설정 및 민감도 분석

**질문:**
*   **Q4:** 튜닝 스윕을 통해 어떤 파라미터 설정이 선택되며, 전체 구성이 이러한 선택에 얼마나 민감한가?

이 질문은 4.1 섹션에서 수행한 필터 튜닝 결과(`tuning_summary.json` 파일)를 분석하여 답변할 수 있습니다.

**해석:**
*   `tuning_summary.json` 파일의 `selected` 필드에서 각 필터의 최적 파라미터 조합을 확인할 수 있습니다.
*   `top_candidates` 필드를 분석하여 최적 파라미터 주변의 다른 파라미터 조합들의 성능을 비교함으로써, 특정 파라미터에 대한 민감도를 정성적 또는 정량적으로 평가할 수 있습니다. 예를 들어, `delta_t_us`가 1000, 2000, 5000일 때 AUC 변화가 크다면 해당 파라미터에 민감하다고 볼 수 있습니다.

## 6. 결과 집계 및 최종 보고 (`aggregate_results.py`)

모든 실험이 완료되면 `src/experiments/aggregate_results.py` 스크립트를 사용하여 전체 결과를 집계하고 최종 보고서를 생성할 수 있습니다.

```bash
cd /mnt/desktop/npx-xarvis
python3 src/experiments/aggregate_results.py
```

**출력:** 이 스크립트는 모든 실험 결과를 종합한 JSON 또는 CSV 파일을 생성할 수 있으며, 이를 바탕으로 최종 논문 작성을 위한 표와 그래프를 구성할 수 있습니다.

## 7. Proposed 필터 명칭 매핑

`main.tex`에 명시된 Proposed 필터 명칭과 코드 상의 `stage_variant` 매핑은 다음과 같습니다.

| 논문 명칭 | `ProposedBalancedConfig`의 `stage_variant` | `run_experiment_v2.py`의 `method_name` |
| :-------------------- | :--------------------------------------- | :------------------------------------- |
| Proposed +REF         | `ref`                                    | `Proposed REF`                         |
| Proposed +SUP         | `sup`                                    | `Proposed SUP`                         |
| Proposed +OPP         | `pol` (gamma=1)                          | `Proposed OPP`                         |
| Proposed +CONF        | `conf`                                   | `Proposed CONF`                        |

`run_experiment_v2.py` 스크립트는 이 명칭 매핑을 반영하여 결과를 출력합니다.

## 8. 트러블슈팅

*   **`ImportError`:** `sys.path.insert(0, ".")`가 스크립트 상단에 있는지 확인하십시오. 또한, `pip3 install -e .` 또는 `pip3 install -r requirements.txt`를 통해 모든 의존성이 설치되었는지 확인하십시오.
*   **`Numba` 관련 오류:** `numba` 패키지가 올바르게 설치되었는지 확인하십시오. `pip3 install numba`.
*   **`UnboundLocalError`:** `proposed_balanced.py`의 `apply` 메서드에서 `output_events` 변수가 모든 코드 경로에서 초기화되는지 확인하십시오. (최근 수정으로 해결됨)
*   **`STCF-RC` 결과 누락:** `run_experiment_v2.py`에 `STCF-RC` 필터가 올바르게 추가되었는지 확인하십시오. (최근 수정으로 해결됨)
*   **`Proposed` 필터 결과 `0.5000` AUC:** `apply_with_warmup` 로직이 `ProposedBalancedFilter`의 상태를 올바르게 전달하고 있는지 확인하십시오. (최근 수정으로 해결됨)

이 가이드를 통해 모든 실험을 성공적으로 수행하시길 바랍니다. 추가적인 문의사항이 있다면 언제든지 질문해 주십시오.
