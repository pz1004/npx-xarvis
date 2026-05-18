# 구체적 수정 계획 (Revision Plan)

본 문서는 제안된 알고리즘의 성능을 개선하고 논문의 완성도를 높이기 위한 구체적인 실행 계획을 정의합니다. 이 계획은 코드 최적화, 실험 설정 일치화, 그리고 평가 메트릭 확장을 포함합니다.

## 1. 코드 최적화: Numba JIT 컴파일 적용

가장 시급한 개선 사항은 순수 Python으로 작성된 필터링 로직의 실행 속도를 높이는 것입니다. `proposed_balanced.py`의 `apply` 메서드와 내부 루프를 Numba를 사용하여 최적화합니다.

**수정 대상 파일:** `src/filters/proposed_balanced.py`

**수정 내용:**
- Numba의 `@njit` 데코레이터를 임포트합니다.
- `apply` 메서드 내부의 이벤트 처리 루프를 별도의 정적 함수(static method)로 분리하고 `@njit`을 적용합니다.
- 상태 배열(`t_raw_last`, `p_raw_last`, `rate_hz`, `unsupported`, `hot`, `t_acc_pos`, `t_acc_neg`)과 이벤트 배열을 Numba 함수에 인자로 전달합니다.
- `_support_count` 메서드에도 `@njit`을 적용하여 호출 오버헤드를 줄입니다.

## 2. 실험 설정 일치화

`implementation.md`와 `main.tex` 간의 시드(seed) 설정 불일치를 해결합니다. 논문(`main.tex`)은 3개의 시드(`{0, 1, 2}`)를 명시하고 있으므로, `implementation.md`의 요구사항(5개 시드) 대신 논문의 설정을 따르도록 코드를 유지하거나, 논문을 5개 시드로 수정해야 합니다. 여기서는 코드의 설정을 논문에 맞추어 명확히 합니다.

**수정 대상 파일:** `configs/training/default.json` 및 `implementation.md` (필요시)

**수정 내용:**
- `configs/training/default.json`의 `seeds` 배열이 `[0, 1, 2]`로 설정되어 있는지 확인하고 유지합니다.
- `implementation.md`의 5개 시드 요구사항 부분을 논문과 일치하도록 3개 시드로 수정하는 것을 고려합니다. (본 작업에서는 코드 실행에 초점을 맞춥니다.)

## 3. 평가 메트릭 확장: ESR (Event Structural Ratio) 추가

단순한 분류 정확도와 ROC/AUC 외에, 필터링 후 객체의 구조적 정보가 얼마나 보존되는지 측정하기 위해 ESR 메트릭을 추가합니다.

**수정 대상 파일:** `src/filters/metrics.py` 및 `src/experiments/tune_filters.py`

**수정 내용:**
- `src/filters/metrics.py`에 ESR을 계산하는 함수를 추가합니다. ESR은 원본 이벤트 스트림과 필터링된 이벤트 스트림 간의 공간적 상관관계를 측정합니다.
- `src/experiments/tune_filters.py`에서 필터 파라미터 튜닝 시 ESR 값을 계산하고 로깅하도록 수정합니다.

## 4. 실행 및 검증 계획

수정 사항을 적용한 후, 다음 단계를 통해 검증을 수행합니다.

1. **단위 테스트 실행:** `pytest tests/` 명령을 실행하여 Numba 최적화 후에도 필터의 논리적 동작이 변경되지 않았는지 확인합니다.
2. **성능 프로파일링:** 최적화 전후의 전처리 지연 시간(preprocessing latency)을 측정하여 속도 향상 폭을 확인합니다.
3. **소규모 실험 실행:** N-MNIST 데이터셋의 캘리브레이션 서브셋에 대해 필터 튜닝 스크립트(`tune_filters.py`)를 실행하여 전체 파이프라인이 정상적으로 동작하는지 확인합니다.

이 계획을 순차적으로 실행하여 알고리즘의 실용성과 논문의 질적 수준을 향상시킬 것입니다.
