# 코드 감사: 논문(main.tex)과 구현 코드의 일치성 검증

## 감사 범위
- `src/filters/proposed_balanced.py` (핵심 필터 구현)
- `src/models/event_snn.py` (SNN 모델)
- `src/experiments/common.py` (메서드 디스패치)
- `configs/training/default.json` (학습 설정)

---

## 1. `proposed_balanced.py` vs Algorithm 1 (main.tex)

### 일치 항목 (Correct)

| 논문 (Algorithm 1) | 구현 코드 | 상태 |
|---|---|---|
| 원시 상태와 수락된 상태 분리 | `t_raw_last`, `p_raw_last` vs `t_acc_pos`, `t_acc_neg` | ✅ 일치 |
| 원시 상태는 모든 이벤트에서 업데이트 | Line 153-154: 항상 `t_raw_last[y,x]=t`, `p_raw_last[y,x]=p` | ✅ 일치 |
| 수락된 상태는 수락된 이벤트에서만 업데이트 | Line 146-149: `if accepted` 블록 내에서만 | ✅ 일치 |
| Pair flag: $Q_i = \mathbf{1}(p_i \neq p^{\text{raw,last}}) \cdot \mathbf{1}(\Delta_i^{\text{raw}} < \tau_{\text{pair}})$ | Line 92: `pair = int((p != int(p_raw_last[y, x])) and (dt_raw < config.tau_pair_us))` | ✅ 일치 |
| Rate estimate: EWMA | Line 95-96: `alpha * instant_rate_hz + (1-alpha) * rate_hz[y,x]` | ✅ 일치 |
| Hot-pixel recovery | Line 98-101: `if hot[y,x] and dt_raw >= config.t_recover_us` | ✅ 일치 |
| Stage 2 guard: hot OR refractory | Line 103: `hot[y,x] or dt_raw < config.tau_ref_dig_us` | ✅ 일치 |
| Stage 3: same-polarity accepted support (3×3) | `_support_count()`: polarity-specific map, 3×3 excluding center | ✅ 일치 |
| Stage 4: $A_i = \mathbf{1}(S_i \geq K_0 + \gamma Q_i)$ | Line 124: `support_count >= (config.k0 + config.gamma * pair)` | ✅ 일치 |
| Unsupported streak: S=0이면 증가, S>0이면 리셋 | Line 126-129 | ✅ 일치 |
| Hot-pixel flag: R > R_max AND U >= U_hot | Line 131-132 | ✅ 일치 |
| Stage 5: confidence = 1 + 1(S >= K_high) | Line 139: `conf = 1 + int(support_count >= config.k_high)` | ✅ 일치 |
| Guarded 이벤트는 U를 변경하지 않음 | Line 150-151: guarded 경로에서 unsupported 미변경 | ✅ 일치 |

### 불일치/개선 필요 항목

| 항목 | 논문 | 구현 | 심각도 |
|---|---|---|---|
| Hot-pixel recovery 시 rate 리셋 | Algorithm 1 Line 6: `H←0, U←0, R←0` | Line 101: `rate_hz[y,x] = 0.0` | ✅ 일치 |
| Stage variant "ref"에서 accepted state 업데이트 | 논문: Stage 2만 사용 시 support 미계산 | Line 133-134: `accepted=True`, Line 145: `if config.stage_variant != "ref"` → ref에서는 acc 미업데이트 | ✅ 일치 (ref는 support 미사용이므로 acc 업데이트 불필요) |
| `stage_variant == "sup"`에서 gamma 무시 | 논문: proposed_sup = Stages 2-3, gamma=0 | Line 121-122: `accepted = support_count >= config.k0` | ✅ 일치 |

---

## 2. `event_snn.py` vs 논문 Section 5.6 / Table in Section 6

### 일치 항목

| 논문 | 구현 | 상태 |
|---|---|---|
| 4채널 입력 (ON_c1, ON_c2, OFF_c1, OFF_c2) | `collapse_input()`: x.shape[2]==4 | ✅ 일치 |
| w_low, w_high 학습 가능 스칼라 | `nn.Parameter` | ✅ 일치 |
| ON = w_low*ch0 + w_high*ch1 | Line 45 | ✅ 일치 |
| Conv2d(2→16, 3×3, pad=1) + LIF | Lines 27-28 | ✅ 일치 |
| Conv2d(16→32, stride=2) + BN + LIF | Lines 30-32 | ✅ 일치 |
| Conv2d(32→64, stride=2) + BN + LIF | Lines 34-36 | ✅ 일치 |
| AdaptiveAvgPool2d(4,4) + Linear(1024→num_classes) + LIF | Lines 38-40 | ✅ 일치 |
| Spike-rate cross-entropy loss | `spike_rate_cross_entropy()` | ✅ 일치 |
| Confidence ratio = w_high / w_low | `confidence_ratio()` | ✅ 일치 |

### 주의 사항

| 항목 | 설명 | 심각도 |
|---|---|---|
| Winner-Take-All (WTA) coincidence | 논문에서 "coincidence-and-inhibition front end similar to HFirst"라고 기술. 구현에서 `_winner_take_all()`로 구현됨. 논문의 `inhibition=True` 언급과 다소 다른 방식이지만, 기능적으로 유사. | ⚠️ 경미 |

---

## 3. 설정 파일 불일치

| 항목 | 논문/implementation.md | 실제 configs | 심각도 |
|---|---|---|---|
| Training seeds | implementation.md: `[3407, 3413, 3421, 3433, 3449]` (5개) | `configs/training/default.json`: `[0, 1, 2]` (3개) | ⚠️ 중요 |
| 논문 main.tex seeds | `{0, 1, 2}` (3개) | `[0, 1, 2]` | ✅ 일치 (main.tex와 일치) |
| Noise ratios | 논문: `{0.5, 1, 2, 5, 10}` | Config: `[0.5, 1.0, 2.0, 5.0, 10.0]` | ✅ 일치 |

**참고:** `implementation.md`는 5개 시드를 요구하지만, `main.tex`는 3개 시드(`{0,1,2}`)를 명시합니다. `implementation.md`에 "If this document conflicts with main.tex, main.tex wins"라고 명시되어 있으므로, 현재 설정은 main.tex와 일치합니다.

---

## 4. 성능 관련 관찰 사항

| 항목 | 설명 |
|---|---|
| 순수 Python 루프 | `proposed_balanced.py`의 `apply()` 메서드가 이벤트별 Python for-loop로 구현됨. Numba JIT 컴파일이나 벡터화 없이는 대규모 데이터셋에서 매우 느릴 것임. |
| `proposed_lowlat`은 raw filter로 대체 | `common.py` Line 126: `proposed_lowlat`은 `_raw_filter`로 매핑됨 (필터링 없음). 논문의 혼합 신호 변형은 소프트웨어로 구현 불가하므로 적절함. |
| Tuning grid 크기 | `proposed_sup/pol/conf`의 파라미터 그리드: 3×3×2×3×3×3×3 = 1,458개 조합. 대규모 탐색이 필요함. |

---

## 5. 감사 결론

**핵심 알고리즘 구현은 논문(main.tex)의 Algorithm 1과 정확히 일치합니다.** 분리된 상태 의미론, 극성 조건부 임계값, 핫 픽셀 복구, 신뢰도 코딩 등 모든 핵심 요소가 올바르게 구현되어 있습니다. 주요 개선이 필요한 부분은 성능 최적화(Numba JIT 등)와 실험 실행을 위한 인프라 측면입니다.
