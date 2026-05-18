# npx-xarvis 설치 및 실험 매뉴얼

**작성자:** Manus AI
**작성일:** 2026년 5월 11일

본 매뉴얼은 `npx-xarvis` 프로젝트의 설치부터 다양한 인자를 활용한 실험 수행, 결과 집계 및 시각화까지의 전체 파이프라인을 단계별로 안내합니다.

---

## 1. 시스템 요구사항 및 설치

프로젝트는 Python 3.8 ~ 3.10 환경을 권장하며, `Makefile`을 통해 자동화된 설치 스크립트를 제공합니다.

### 1.1 사전 요구사항 설치 (Linux)
Ubuntu/Debian 또는 CentOS/RHEL 환경에서 필요한 시스템 패키지를 설치합니다.
```bash
make preinstall
```

### 1.2 Python 패키지 설치
논문 실험 재현을 위해 검증된 버전의 패키지(PyTorch, snntorch, tonic 등)를 설치합니다.
```bash
make install_verified
```
*참고: 최신 버전의 패키지를 설치하려면 `make install_recent`를, 논문 집계용 추가 패키지만 설치하려면 `make install-paper`를 사용할 수 있습니다.*

---

## 2. 데이터셋 및 지원 메서드

### 2.1 지원되는 데이터셋 (`--dataset`)
실험 스크립트 실행 시 데이터셋은 자동으로 다운로드 및 전처리되어 `dataset/` 디렉토리에 저장되며, 분할(split) 정보는 `results/splits/`에 캐시됩니다.
- `nmnist`: N-MNIST
- `dvsgesture`: DVS128 Gesture
- `ncaltech101`: N-Caltech101 (자동으로 128x128 크기로 조정됨)
- `cifar10dvs`: CIFAR10-DVS

### 2.2 지원되는 메서드 (`--method`)
- **베이스라인:** `raw_snn` (원시 이벤트), `frame_snn` (프레임 기반), `ba_snn` (BA 필터), `stcf_rc_snn` (STCF)
- **제안 방법 (Ablation 포함):**
  - `proposed_ref`: 불응기(Refractory) 가드만 적용
  - `proposed_sup`: 지지(Support) 카운트 적용
  - `proposed_pol`: 극성(Polarity) 페널티 적용
  - `proposed_conf`: **최종 제안 방법 (신뢰도 코딩 적용)**
- **효율성 분석용:** `proposed_lowmem` (저메모리 변형), `proposed_lowlat` (저지연 분석용)

---

## 3. 실험 수행 가이드

프로젝트의 실험은 크게 **1) 하이퍼파라미터 튜닝**, **2) 모델 학습 및 평가**, **3) 효율성 프로파일링**, **4) 결과 집계 및 시각화**의 4단계로 구성됩니다.

### 3.1 하이퍼파라미터 튜닝 (`tune_filters.py`)
Stage-A 필터 하이퍼파라미터를 합성 노이즈 AUC를 기준으로 튜닝하고, 상위 후보에 대해 Stage-B 다운스트림 학습을 수행합니다.

**기본 실행:**
```bash
python3 -m src.experiments.tune_filters --dataset nmnist --method proposed_conf
```

**주요 인자 (Arguments):**
- `--top-k <int>`: Stage-B 학습을 수행할 상위 후보 개수 (기본값: 5)
- `--seed <int>`: 난수 시드 (기본값: 0)
- `--max-calibration-samples <int>`: 튜닝에 사용할 최대 캘리브레이션 샘플 수 (빠른 디버깅용)
- `--max-grid <int>`: 탐색할 파라미터 그리드 최대 개수 (빠른 디버깅용)
- `--skip-stage-b`: Stage-B 다운스트림 학습을 생략하고 AUC 기반 필터 튜닝만 수행
- `--stage-b-epochs <int>`: Stage-B 학습 에포크 수 재정의
- `--force-cpu`: 강제로 CPU만 사용하여 실행

### 3.2 모델 학습 및 평가 (`train_eval.py`)
선택된 데이터셋과 메서드에 대해 SNN 모델을 학습하고, 노이즈 강건성(Robustness)을 평가합니다.

**기본 실행:**
```bash
python3 -m src.experiments.train_eval --dataset nmnist --method proposed_conf --seed 0
```

**주요 인자 (Arguments):**
- `--epochs-override <int>`: 기본 설정된 에포크 수(예: N-MNIST 100, DVSGesture 150)를 무시하고 지정된 에포크만큼 학습
- `--max-train-samples <int>`: 학습에 사용할 최대 샘플 수 제한 (빠른 테스트용)
- `--max-val-samples <int>`: 검증에 사용할 최대 샘플 수 제한
- `--max-test-samples <int>`: 테스트에 사용할 최대 샘플 수 제한
- `--force-cpu`: 강제로 CPU만 사용하여 실행

**출력 결과:**
실험 결과는 `results/paper/<dataset>/<method>/seed_<seed>/` 디렉토리에 저장되며, `summary.json`, `robustness.json`, `best_model.pt` 파일이 생성됩니다.

### 3.3 효율성 프로파일링 (`profile_efficiency.py`)
학습된 모델의 전처리 지연 시간, 처리량, 메모리 사용량, 파라미터 수, MAC/SOP 연산량 등을 프로파일링합니다.

**기본 실행:**
```bash
python3 -m src.experiments.profile_efficiency --dataset nmnist --method proposed_conf --seed 0
```

**주요 인자 (Arguments):**
- `--train-if-missing`: 기존 학습 결과(`summary.json`)가 없을 경우 자동으로 학습을 먼저 수행
- `--epochs-override`, `--max-train-samples` 등: `train_eval.py`와 동일하게 적용 가능

**출력 결과:**
해당 실험 결과 디렉토리에 `profile.json` 파일이 생성됩니다.

### 3.4 결과 집계 및 시각화 (`aggregate_results.py`)
모든 실험 결과를 모아 논문용 표(CSV)와 차트(PNG)를 생성합니다.

**기본 실행:**
```bash
python3 -m src.experiments.aggregate_results
```

**주요 인자 (Arguments):**
- `--regen-roc`: ROC 곡선 차트를 새로 생성
- `--roc-dataset <dataset>`: ROC 곡선을 생성할 대상 데이터셋 (기본값: nmnist)
- `--roc-calibration-samples <int>`: ROC 생성 시 사용할 캘리브레이션 샘플 수 (기본값: 64)

**출력 결과:**
`results/paper/aggregated/` 디렉토리에 다음 파일들이 생성됩니다.
- `main_accuracy.csv`: 주요 메서드별 정확도 표
- `ablation.csv`: 제안 방법의 Ablation 결과 표
- `efficiency.csv`: 효율성(메모리, 연산량 등) 비교 표
- `aunc.csv`: 노이즈 강건성(AUNC) 결과 표
- `accuracy_vs_compute.png`: 연산량 대비 정확도 산점도
- `roc_<dataset>.png`: (옵션) ROC 곡선 차트

---

## 4. 빠른 테스트 (Smoke Test)

설치 및 환경 구성이 올바른지 확인하기 위해 전체 파이프라인을 최소한의 데이터로 빠르게 실행하는 명령어를 제공합니다.

```bash
make smoke-paper
```
이 명령어는 단위 테스트(`pytest`)를 실행한 후, N-MNIST 데이터셋에 대해 `proposed_conf`와 `frame_snn` 메서드를 1 에포크, 소량의 샘플(Train 64, Val 32, Test 32)로 CPU 환경에서 빠르게 학습하여 파이프라인의 정상 동작을 검증합니다.

---

## 5. 전체 논문 실험 재현 (Full Paper Reproduction)

논문에 보고된 모든 결과를 재현하려면 아래 순서를 따릅니다. 기본 학습 설정은 `configs/training/default.json`에 정의되어 있으며, 시드 3개(0, 1, 2)에 대해 반복 실험을 수행합니다.

### 5.1 Step 1: 모든 메서드에 대한 학습 실행
```bash
# 논문 메인 결과 (seeds: 0, 1, 2)
for SEED in 0 1 2; do
  for METHOD in raw_snn frame_snn ba_snn stcf_rc_snn proposed_ref proposed_sup proposed_pol proposed_conf; do
    python3 -m src.experiments.train_eval --dataset nmnist --method $METHOD --seed $SEED
  done
done
```

### 5.2 Step 2: 효율성 프로파일링
```bash
for METHOD in raw_snn frame_snn ba_snn stcf_rc_snn proposed_ref proposed_sup proposed_pol proposed_conf proposed_lowmem proposed_lowlat; do
  python3 -m src.experiments.profile_efficiency --dataset nmnist --method $METHOD --seed 0 --train-if-missing
done
```

### 5.3 Step 3: 결과 집계 및 시각화
```bash
python3 -m src.experiments.aggregate_results --regen-roc --roc-dataset nmnist
```

### 5.4 Step 4: 다른 데이터셋에 대해 반복
위 Step 1~3을 `--dataset dvsgesture`, `--dataset ncaltech101`, `--dataset cifar10dvs`로 변경하여 반복합니다.

---

## 6. 실용적인 실험 시나리오 예제

### 6.1 빠른 디버깅 실행 (5분 이내)
환경 설정 후 파이프라인 동작만 빠르게 확인하고 싶을 때:
```bash
python3 -m src.experiments.train_eval \
  --dataset nmnist \
  --method proposed_conf \
  --seed 0 \
  --epochs-override 2 \
  --max-train-samples 128 \
  --max-val-samples 64 \
  --max-test-samples 64 \
  --force-cpu
```

### 6.2 하이퍼파라미터 빠른 탐색 (Stage-A만)
전체 그리드 탐색은 시간이 오래 걸리므로, 상위 10개 조합만 빠르게 평가:
```bash
python3 -m src.experiments.tune_filters \
  --dataset nmnist \
  --method proposed_conf \
  --max-grid 10 \
  --max-calibration-samples 50 \
  --skip-stage-b \
  --force-cpu
```

### 6.3 특정 시드에 대한 단일 메서드 학습 및 프로파일링
```bash
# 학습
python3 -m src.experiments.train_eval --dataset dvsgesture --method proposed_conf --seed 42

# 프로파일링
python3 -m src.experiments.profile_efficiency --dataset dvsgesture --method proposed_conf --seed 42
```

### 6.4 GPU 사용 시
CUDA가 설치되어 있고 GPU가 사용 가능한 경우, `--force-cpu` 플래그를 제거하면 자동으로 GPU를 사용합니다:
```bash
python3 -m src.experiments.train_eval --dataset cifar10dvs --method proposed_conf --seed 0
```

---

## 7. 프로젝트 디렉토리 구조

```
npx-xarvis/
├── configs/                    # 실험 설정 파일
│   ├── datasets/               # 데이터셋별 학습 설정 (batch_size, epochs)
│   ├── methods/                # 메서드별 필터 파라미터 설정
│   ├── stage1/                 # Stage-1 캘리브레이션 설정
│   └── training/               # 전역 학습 설정 (lr, seeds, noise_ratios)
├── dataset/                    # (자동 생성) 다운로드된 데이터셋
├── results/
│   ├── paper/                  # 실험 결과 저장
│   │   ├── <dataset>/
│   │   │   └── <method>/
│   │   │       ├── seed_<N>/   # 개별 실험 결과
│   │   │       ├── tuning_summary.json
│   │   │       └── event_metrics.json
│   │   └── aggregated/         # 집계된 표 및 차트
│   └── splits/                 # 데이터셋 분할 캐시
├── src/
│   ├── data/                   # 데이터 로딩, 노이즈 주입, 슬라이싱
│   ├── experiments/            # 실험 스크립트 (train_eval, tune, aggregate, profile)
│   ├── filters/                # 필터 구현 (proposed_balanced, ba, stcf_rc)
│   ├── models/                 # SNN 모델 (EventSNN, FrameSNN)
│   └── utils/                  # 유틸리티 (로깅, 시드, 직렬화)
├── tests/                      # 단위 테스트
├── Makefile                    # 빌드/실행 자동화
└── main.tex                    # 논문 원고
```

---

## 8. 주요 설정 파라미터 참조

### 8.1 전역 학습 설정 (`configs/training/default.json`)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `learning_rate` | 0.002 | 학습률 |
| `weight_decay` | 0.0001 | 가중치 감쇠 |
| `gradient_clip` | 1.0 | 그래디언트 클리핑 |
| `warmup_epochs` | 10 | 학습률 웜업 에포크 |
| `seeds` | [0, 1, 2] | 반복 실험 시드 목록 |
| `split_seed` | 2027 | 데이터셋 분할 시드 |
| `noise_ratios` | [0.5, 1.0, 2.0, 5.0, 10.0] | 강건성 평가 노이즈 비율 |
| `profile_samples` | 200 | 전처리 프로파일링 샘플 수 |
| `inference_warmup` | 20 | 추론 지연 측정 웜업 횟수 |
| `inference_runs` | 100 | 추론 지연 측정 반복 횟수 |

### 8.2 제안 방법 필터 파라미터 (`configs/methods/proposed_conf.json`)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `tau_ref_dig_us` | 1000 | 불응기 시간 (마이크로초) |
| `delta_t_us` | 2000 | 지지 계산 시간 윈도우 (마이크로초) |
| `k0` | 1 | 수락 최소 지지 카운트 |
| `gamma` | 1 | 극성 반전 페널티 계수 |
| `tau_pair_us` | 1000 | 극성 쌍 판정 시간 임계값 |
| `k_high` | 2 | 고신뢰도 전환 지지 임계값 |
| `alpha` | 0.01 | 발화율 추정 지수 이동 평균 계수 |
| `r_max_hz` | "auto" | 핫픽셀 판정 최대 발화율 (auto=캘리브레이션) |
| `u_hot` | 32 | 핫픽셀 판정 연속 미지지 횟수 |
| `t_recover_us` | 1000000 | 핫픽셀 복구 대기 시간 (마이크로초) |

### 8.3 데이터셋별 기본 학습 설정

| 데이터셋 | Batch Size | Epochs | 센서 크기 |
|---|---|---|---|
| N-MNIST | 128 | 100 | 34x34x2 |
| DVS128 Gesture | 32 | 150 | 128x128x2 |
| N-Caltech101 | 32 | 150 | 128x128x2 (리사이즈) |
| CIFAR10-DVS | 64 | 150 | 128x128x2 |

---

## 9. 트러블슈팅

### 9.1 데이터셋 다운로드 실패
`tonic` 라이브러리가 자동으로 데이터셋을 다운로드합니다. 네트워크 문제가 발생하면 `dataset/` 디렉토리를 삭제하고 재시도하거나, 수동으로 데이터셋을 다운로드하여 해당 디렉토리에 배치합니다.

### 9.2 Numba JIT 컴파일 지연
첫 실행 시 Numba가 필터 함수를 JIT 컴파일하므로 약간의 지연이 발생합니다. 컴파일된 결과는 `__pycache__`에 캐시되어 이후 실행에서는 즉시 로드됩니다. Numba가 설치되지 않은 환경에서는 자동으로 순수 Python 구현으로 폴백됩니다.

### 9.3 메모리 부족 (OOM)
DVS128 Gesture나 N-Caltech101과 같은 대규모 데이터셋에서 메모리 부족이 발생하면 `--max-train-samples`를 사용하여 샘플 수를 제한하거나, `configs/datasets/` 내 해당 데이터셋의 `batch_size`를 줄입니다.

### 9.4 Windows 환경
Windows에서는 `Makefile` 대신 직접 Python 명령어를 실행합니다:
```cmd
python -m pip install tqdm==4.67.1 tonic==1.6.0
python -m pip install torch==2.4.1+cpu torchvision==0.19.1+cpu torchaudio==2.4.1 --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install snntorch==0.9.1
python -m src.experiments.train_eval --dataset nmnist --method proposed_conf --seed 0 --force-cpu
```
