# main.tex 수정 계획

`final_summary_report.md`의 제언을 바탕으로 `main.tex`를 다음과 같이 수정합니다.

## 1. Abstract & Introduction
- **Abstract:** Numba JIT 최적화를 통한 5M eps 이상의 실시간 처리량 달성 언급 추가. ESR 메트릭을 통한 구조적 보존 평가 언급 추가.
- **Introduction (Contributions):** 
  - 콜드스타트 문제 해결을 위한 웜업 시딩(warm-up seeding) 메커니즘 기여 추가.
  - Numba JIT 기반의 고성능 소프트웨어 구현 및 ESR 평가 결과 기여 추가.

## 2. Related Work
- **Event-Domain Denoising:** 최근 연구 동향(PCLF, EDmamba 등) 및 E-MLB 벤치마크의 ESR 메트릭 도입 언급 추가.

## 3. Method (Proposed Architecture)
- **Stage 2 & 3 (Split-state semantics):** 콜드스타트(Cold-Start) 문제의 본질적 원인 설명 추가.
- **Warm-up Seeding:** 초기 웜업 기간 동안 $K_0=0$을 사용하여 수락된 상태를 시딩하는 메커니즘을 명시적으로 추가. (Algorithm 1 수정 또는 텍스트 설명 추가)

## 4. Experiments & Results
- **Evaluation Metrics:** ESR (Event Structural Ratio) 메트릭 정의 및 수식 추가.
- **Implementation Details:** Numba JIT 최적화 적용 및 처리량(Throughput) 향상 수치(88.5배, 5M eps) 명시.
- **Results:** 
  - Proposed CONF가 모든 노이즈 환경에서 가장 높은 AUC를 기록했음을 강조.
  - ESR 결과를 포함하여 제안 방법이 구조적 보존 측면에서도 우수함을 입증.
  - 노이즈 비율 증가에 따른 강건성(Robustness) 유지 결과 추가.

## 5. Conclusion
- 콜드스타트 해결, Numba 최적화, ESR 평가 결과를 요약에 반영하여 결론 강화.
