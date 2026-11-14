# GPU 환경 검증 및 YAMNet 4-fold 가중치 비교

- 실험 완료 여부: **완료** (12/12 folds)
- 최초 GPU 학습·저장·재로딩 검증: **통과**
- 종료 상태: `complete`, exit code `0`
- 시작: 2026-11-09T16:32:08.615537+00:00 / 종료: 2026-11-09T16:36:50.637385+00:00 (UTC)
- 전체 경과 시간: 282.0초
- 이 실행의 패키지 설치: 0건. 환경 프로필 `official221`의 고정 패키지를 확인하여 사용.
- 모델 성능 적합성: 개발 CV와 임시 검증 가드로만 판단하며 제품·현장 적합성을 확정하지 않음.

## 비교 결과의 해석

**공식 GPU 환경에서 실제 정규 학습·저장·재로딩을 포함한 12개 fold를 모두 완료했다. 가중치를 1배에서 2·4배로 높였을 때 국내 성능 개선은 확인되지 않았다.** 국내 OOF 34개 source group의 예측 클래스는 세 설정에서 모두 같았다. 확률 값과 체크포인트까지 동일하다는 뜻은 아니다.

국내 타깃 recall은 모든 설정에서 fold 0~3 순서로 87.50%, 87.50%, 100.00%, 66.67%였다. 평균 85.42%, 표본 표준편차 13.82%p이며, 국내 OOF recall은 86.06%다. Fold 평균과 OOF recall은 집계 방식과 클래스별 그룹 수가 달라 값이 다르다.

국내 OOF에서 화재 알람은 11/13개, 초인종은 14/16개를 맞혔고 나머지 타깃 4개는 모두 `other`로 분류했다. 도어락은 5/5개를 맞혔다. 국내 타깃 두 클래스의 macro precision은 100.00%이며, 세 클래스의 macro-F1은 85.48%다. 국내 도어락 오탐 0/5는 작은 개발 표본에서 얻은 관측값이다.

해외 `other` 오탐은 1배 `[11, 8, 10, 3]`, 2배 `[10, 8, 11, 8]`, 4배 `[10, 6, 11, 7]`개였고, 각 값의 분모는 공통 validation 398개다. 2배는 1배보다 1개 fold에서 감소·1개 동일·2개 증가했고, 4배는 2개 감소·2개 증가했다. 모델별 FPR의 4-fold 평균은 각각 2.01%, 2.32%, 2.14%다. 같은 음원을 재사용하므로 이를 독립 표본으로 합치거나 통계적 유의성을 주장하지 않는다.

검증 가드는 12/12개 fold에서 충족했다. 이 결과는 이번 개발 비교의 실행 성공과 임시 가드 충족을 보여준다. 타깃 누락과 fold별 편차가 남아 있으며, 제품·현장 성능 적합성은 확정하지 않는다. 이번 비교에서는 가중치를 높일 일관된 이득이 확인되지 않았고, 최종 가중치 선택·refit·고정 test 평가는 후속 단계로 남긴다.

## 실행 시간과 사후 검증

- KST 실행 시각: `2026-11-10T01:32:08+09:00` → `2026-11-10T01:36:50+09:00`. CV 제어 프로세스 전체 경과 시간은 282.022초(약 4분 42초)다.
- 가중치별 전체 worker 시간: 1배 112.281초, 2배 75.577초, 4배 85.809초. 비교표의 학습 시간은 각 fold의 데이터 provenance 기록·학습·검증·저장·재로딩을 포함한 시간의 합이며, worker 시작과 공통 캐시 로딩 시간은 제외한다.
- 첫 설정은 GPU 커널 JIT 준비 비용과 캐시 준비 영향을 포함하고 설정마다 조기 종료 시점도 다르다. 이 시간을 가중치 효과나 GPU 대 CPU 속도 비교로 해석하지 않는다. CPU 학습과 속도 비교 실험은 수행하지 않았다.
- `verify_results.py`로 12개 fold의 인자·seed·신규 optimizer 순서·GPU 배치, 임베딩 provenance·동일 split, epoch당 900 draws, 저장 확률에서 재계산한 지표, 국내 OOF 각 그룹 1회 포함, 고정 test 미사용을 검증했다. 이 검증은 TensorFlow를 import하지 않았다.
- 보호 파일 7,154개와 기존 세 환경의 패키지 목록이 실행 전후 일치했다. 선행 GPU 검증의 참조 증거 해시도 유지됐다. 고정 test 임베딩 0개와 test 평가 미실행을 재확인했다.
- 세 GPU worker는 모두 종료 코드 0으로 종료했고 최초 실행 오류는 없다. 종료 후 GPU compute process는 0개, GPU 사용 메모리는 초기와 같은 14 MiB였다.

[12-fold 비교 CSV](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/yamnet/reports/ubuntu_gpu_cv_s42_official221_20261109T163208Z/comparison_folds.csv) · [가중치별 요약 CSV](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/yamnet/reports/ubuntu_gpu_cv_s42_official221_20261109T163208Z/comparison_weights.csv) · [최종 검증](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/yamnet/reports/ubuntu_gpu_cv_s42_official221_20261109T163208Z/FINAL_VERIFICATION.json)

## 검증된 공식 GPU 환경

공식 PyPI 배포 TensorFlow 2.21.0·tf-keras 2.21.0, CUDA 12.8 라이브러리·cuDNN 9.8.0.87을 설치한 `.venv-gpu-official221`을 사용했다. 이전 TensorFlow 2.18.1 후보와 별도 환경이다. TensorFlow Hub 0.16.1·NumPy 1.26.4·setuptools 80.9.0을 유지했다.

선행 최소 GPU 검증 보고서: `/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/yamnet/reports/gpu_training_validation_official221_20261109T135728Z`. 완료·종료 코드 0, 실제 GPU 업데이트, 1 epoch·2 steps 저장/재로딩, 기존 체크포인트 CPU/GPU 수치 일치 증거를 확인했다. 해당 검증의 패키지·보호 입력·GPU 실행 코드가 동일함을 `reused_gpu_validation.json`에 기록하고 최소 학습은 반복하지 않았다. 이번 정규 CV에서는 CPU 분류기 추론이나 CPU 학습을 실행하지 않았다.

## 보존 확인

기존 원본·전처리·임베딩·split·설정·코드·기존 실행 기록 7,154개 파일의 SHA-256 일치: **True**.
기존/별도 환경 패키지 목록 일치: **True**.
고정 test 임베딩: **0개**. 고정 test 특징 추출·평가, refit, DS-CNN, 보드 실행은 하지 않았다.

## 가중치별 비교

국내 recall은 `fire_alarm`과 `doorbell`의 source-group recall 평균이다. 표준편차는 4-fold 표본 표준편차(ddof=1)이며, 신뢰구간이 아니다. precision과 macro-F1도 국내 source-group 기준이다.

| 가중치 | recall 평균 ± SD | 국내 OOF recall | 국내 OOF precision | 국내 OOF macro-F1 | OOF 도어락 FP | 가드 통과 | 학습 시간 |
|---|---|---|---|---|---|---|---|
| 1 | 0.8542 ± 0.1382 | 0.8606 | 1.0000 | 0.8548 | 0/5 | 4/4 | 108.5초 |
| 2 | 0.8542 ± 0.1382 | 0.8606 | 1.0000 | 0.8548 | 0/5 | 4/4 | 73.0초 |
| 4 | 0.8542 ± 0.1382 | 0.8606 | 1.0000 | 0.8548 | 0/5 | 4/4 | 83.2초 |

## Fold별 선택 결과

| 가중치 | fold | 선택/실행 epoch | 국내 recall | 국내 precision | 국내 macro-F1 | 해외 other FP | 국내 도어락 FP | 가드 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 16/24 | 0.8750 | 1.0000 | 0.8857 | 11/398 | 0/2 | 충족 |
| 1 | 1 | 20/28 | 0.8750 | 1.0000 | 0.8413 | 8/398 | 0/1 | 충족 |
| 1 | 2 | 11/19 | 1.0000 | 1.0000 | 1.0000 | 10/398 | 0/1 | 충족 |
| 1 | 3 | 30/30 | 0.6667 | 1.0000 | 0.6667 | 3/398 | 0/1 | 충족 |
| 2 | 0 | 16/24 | 0.8750 | 1.0000 | 0.8857 | 10/398 | 0/2 | 충족 |
| 2 | 1 | 12/20 | 0.8750 | 1.0000 | 0.8413 | 8/398 | 0/1 | 충족 |
| 2 | 2 | 13/21 | 1.0000 | 1.0000 | 1.0000 | 11/398 | 0/1 | 충족 |
| 2 | 3 | 29/30 | 0.6667 | 1.0000 | 0.6667 | 8/398 | 0/1 | 충족 |
| 4 | 0 | 8/16 | 0.8750 | 1.0000 | 0.8857 | 10/398 | 0/2 | 충족 |
| 4 | 1 | 27/30 | 0.8750 | 1.0000 | 0.8413 | 6/398 | 0/1 | 충족 |
| 4 | 2 | 17/25 | 1.0000 | 1.0000 | 1.0000 | 11/398 | 0/1 | 충족 |
| 4 | 3 | 14/22 | 0.6667 | 1.0000 | 0.6667 | 7/398 | 0/1 | 충족 |

가드 미충족은 실행 오류가 아니다. 임시 조건은 해외 other FPR ≤ 0.05, 국내 도어락 오탐 0개이며, 이를 만족하지 못한 fold도 기존 checkpoint 선택 규칙으로 저장했다.

## 개선의 일관성과 오탐 변화

- 2배 대 1배: recall 개선/동일/하락 0/4/0 folds. Fold 0~3 recall 차이 [0.0, 0.0, 0.0, 0.0], 해외 other FP 차이 [-1, 0, 1, 5], 국내 도어락 FP 차이 [0, 0, 0, 0].
- 4배 대 1배: recall 개선/동일/하락 0/4/0 folds. Fold 0~3 recall 차이 [0.0, 0.0, 0.0, 0.0], 해외 other FP 차이 [-1, -2, 1, 4], 국내 도어락 FP 차이 [0, 0, 0, 0].

## 혼동행렬

행=정답, 열=예측, 클래스 순서 `[fire_alarm, doorbell, other]`. 아래는 국내 source-group 기준이다. 전체 validation group 행렬과 클래스별 precision/recall 분자·분모는 `comparison_folds.csv`에 함께 보관한다.

- w1 fold 0: `[[4, 0, 0], [0, 3, 1], [0, 0, 2]]`
- w1 fold 1: `[[3, 0, 0], [0, 3, 1], [0, 0, 1]]`
- w1 fold 2: `[[3, 0, 0], [0, 4, 0], [0, 0, 1]]`
- w1 fold 3: `[[1, 0, 2], [0, 4, 0], [0, 0, 1]]`
- w2 fold 0: `[[4, 0, 0], [0, 3, 1], [0, 0, 2]]`
- w2 fold 1: `[[3, 0, 0], [0, 3, 1], [0, 0, 1]]`
- w2 fold 2: `[[3, 0, 0], [0, 4, 0], [0, 0, 1]]`
- w2 fold 3: `[[1, 0, 2], [0, 4, 0], [0, 0, 1]]`
- w4 fold 0: `[[4, 0, 0], [0, 3, 1], [0, 0, 2]]`
- w4 fold 1: `[[3, 0, 0], [0, 3, 1], [0, 0, 1]]`
- w4 fold 2: `[[3, 0, 0], [0, 4, 0], [0, 0, 1]]`
- w4 fold 3: `[[1, 0, 2], [0, 4, 0], [0, 0, 1]]`
- w1 국내 OOF: `[[11, 0, 2], [0, 14, 2], [0, 0, 5]]`
- w2 국내 OOF: `[[11, 0, 2], [0, 14, 2], [0, 0, 5]]`
- w4 국내 OOF: `[[11, 0, 2], [0, 14, 2], [0, 0, 5]]`

## 조건·증거·한계

가중치 1→2→4, 각 fold 신규 분류기·Adam, seed=42+fold, 최대 30 epoch, patience 8, batch 30, 30 steps/epoch, learning rate 0.001. 모델 구조·샘플링·검증 가드·분할·임베딩은 기존 코드를 사용했다.

GPU 보조 실행기는 Dense, gradient 생성, Adam 연산을 `/GPU:0`에 배치하고 soft device placement를 끈다. 데이터 읽기·샘플링·메트릭·파일 저장은 호스트에서 처리한다. 선행 검증의 기존 체크포인트 CPU/GPU 출력 비교 허용오차는 rtol=1e-4, atol=1e-5다. 각 정규 fold의 저장된 분류기를 GPU에서 재로딩하고 검증 메트릭·혼동행렬 일치를 확인했다.

공통 ESC-50 validation 398개는 여러 fold에서 재사용하므로 독립 OOF 표본 1,592개로 합산하지 않는다. 국내 OOF만 각 국내 개발 그룹의 한 번 검증 예측을 모은다. 작은 국내 개발 표본, 단일 seed, 같은 해외 validation을 사용하는 checkpoint 선택에 따른 한계가 있다. 정규 비교에서 1 epoch·2 steps GPU 검증 실행은 제외한다.

최종 가중치 확정과 refit은 다음 단계로 남긴다. 실제 환경의 false alarms/hour, 이벤트 지연, 고정 test 성능과 보드 동작은 이 결과로 검증되지 않는다.

- `packages_*`, `package_comparison.json`: 전체 패키지·동일성
- `gpu_info.csv`, `environment.json`, `execution_events.json`, `planned_commands.json`: GPU·환경·실행 명령·시간·종료 상태
- `reused_gpu_validation.json` 및 참조된 선행 보고서: 순전파·역전파·업데이트·CPU/GPU 수치·경로 정규화 사본과 텐서 해시·저장/재로딩 증거
- `cv_weight_*.log`, `weight_*/device_evidence.json`, 각 fold의 `gpu_execution.json`: 학습 로그·GPU 배치·fold 시간
- `protected_before.json`, `protected_after.json`, `integrity.json`, `cache_*.json`: 보존·고정 test 0개 확인

GPU 호환성 판단은 실제 학습·업데이트·저장/재로딩 증거에 한정한다. TensorFlow 2.21.0 wheel의 빌드 CUDA 메타데이터와 실제 가상환경에서 로드한 CUDA 12.8 라이브러리를 구분하며, 상세 정보는 `weight_*/tensorflow_gpu.json`과 `weight_*/loaded_cuda_libraries.json`에 있다.
