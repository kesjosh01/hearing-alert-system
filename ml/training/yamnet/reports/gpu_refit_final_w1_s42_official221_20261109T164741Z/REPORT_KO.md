# YAMNet 기준 모델 refit·고정 test 최종 평가

**기준 모델 refit과 고정 test 1회 평가를 완료했다.** 국내 가중치 1배·seed 42·18 epoch를 사용했다. 실행 완료와 제품 성능 적합성은 별도 판단이며, 작은 국내 test와 음원 분류 실험만으로 현장 적합성을 확정하지 않는다.

## 선택 설정과 모델 고정

정규 1·2·4배 4-fold 비교에서 국내 recall과 국내 OOF 34개 그룹의 예측 클래스가 모두 같았다. 사전 규칙 `choose_smallest_if_no_consistent_gain`에 따라 1배를 선택했다. 1배의 fold별 선택 epoch `[16, 20, 11, 30]`의 중앙값인 18 epoch로 refit했다.

신규 분류기·Adam, seed 42, batch 30, 30 steps/epoch, learning rate 0.001, 기존 모델 구조·샘플링·분할·임베딩을 유지했다. 국내 개발 34개 그룹을 train에 포함하고 기존 해외 train 및 보조 타깃을 사용했다. 해외 validation 398개는 학습에서 제외했다. Refit 동안 checkpoint epoch를 다시 선택하지 않고 18 epoch 전체를 학습했다. 총 16,200개 patch draws를 기록했다.

저장·재로딩 validation 398개 확률의 최대 차이는 0이다. Test 특징 추출 전 분류기 SHA-256 `70b9da2319abecf27fb55ef748ca3e65ceb2dcb21207eb9e32f3bb9f446982b2`를 고정했고 평가 후에도 유지됐다. Native patch softmax 평균 → clip → source group 동일 가중치 평균 → argmax 규칙을 유지하며 새 임계값 조정과 test 기반 설정 변경은 없다.

## 수치 검증 기준 변경과 중단 기록

최초 실행은 refit·저장·재로딩과 공식 YAMNet 다운로드를 완료한 뒤 개발 캐시의 중간 WAV가 없는 보조 점검 가정 오류로 중단했다. 원본은 그대로 두고 점검용 개발 WAV 6개를 별도 캐시에 준비했다. 최초 traceback과 종료 코드 1은 `first_error.log`와 `status.json`에 남겼다.

공식 YAMNet SavedModel PB 해시는 기존 개발 캐시 identity와 같았다. GPU 특징의 엄격 비교(rtol=1e-4, atol=1e-5)는 `bsb-1153`에서 최대 절대 차이 0.00050354로 실패했다. 사용자 허가 후 실시한 분류기 출력 비교에서는 개발 6개 모두 예측 클래스가 같았으나 `ESC50-1-100032-A-0`의 clip 확률 차이 0.0000557491447(0.0055749%p)이 같은 엄격 허용오차를 벗어났다. 마지막 음원의 특징 최대 차이는 0.00400817이다.

사용자가 별도로 허가한 **클래스 동일 + 절대 clip 확률 차이 ≤ 0.0001(0.01%p)** 기준을 저장된 개발 6개 출력에 적용하여 진행을 결정했다. 이는 개발 결과를 관측한 뒤 바꾼 수치 수용 기준이다. Test는 이 시점까지 임베딩 0개·평가 0회였고, 설정·가중치·epoch를 바꾸지 않았다. 기존 특징·출력의 엄격 검사 실패 기록을 통과로 바꾸지 않았다. 이 6개 사례의 출력 동등성은 전체 음원의 완전한 수치 동등성을 보장하지 않는다.

원본 PCM 캐시가 없어 디코딩·resampling 차이와 TensorFlow/GPU 산술 차이를 분리하지 못했다. 개발 특징 파일은 덮어쓰지 않았다. 공식 특징 추출은 TF32를 끄고 실행했고 분류기 test 평가 전에 기본 설정으로 복원했다. GPU 학습·추론의 실행 오류나 CPU 대체는 없었다.

## 고정 test 결과

전체 **411 clips / 319 source groups**, 국내 **9 source groups**. 국내 타깃 macro recall **87.50%**, 타깃 macro precision **90.00%**, 세 클래스 macro-F1 **0.9153**.

| 국내 클래스 | 정답/분모 | precision | recall | F1 |
|---|---:|---:|---:|---:|
| fire_alarm | 4/4 | 0.8000 | 1.0000 | 0.8889 |
| doorbell | 3/4 | 1.0000 | 0.7500 | 0.8571 |
| other | 1/1 | 1.0000 | 1.0000 | 1.0000 |

해외 `other` 오탐 **9/402** (2.24%). 국내 도어락 오탐 **0/1**.

| 전체 test source-group 클래스 | 정답/분모 | precision | recall | F1 |
|---|---:|---:|---:|---:|
| fire_alarm | 4/4 | 0.4000 | 1.0000 | 0.5714 |
| doorbell | 3/4 | 0.6000 | 0.7500 | 0.6667 |
| other | 304/311 | 1.0000 | 0.9775 | 0.9886 |

국내 초인종 1개가 화재 알람으로 오분류됐다. 해외 other 오탐 9개는 화재 알람 7개·초인종 2개다. 전체 test source-group 기준 타깃 precision은 화재 40%, 초인종 60%이므로 국내 recall과 함께 해석해야 한다.

해외 other 402개 clip은 310개 원본 그룹에 속한다. Clip 오탐의 분모 402와 전체 source-group 지표의 분모 319를 구분하며, 같은 원본의 여러 clip을 독립 원본으로 간주하지 않는다.

혼동행렬은 행=정답, 열=예측, 순서 `[fire_alarm, doorbell, other]`다.

- 국내 source group: `[[4, 0, 0], [1, 3, 0], [0, 0, 1]]`
- 전체 source group: `[[4, 0, 0], [1, 3, 0], [5, 2, 304]]`
- 전체 clip: `[[4, 0, 0], [1, 3, 0], [7, 2, 394]]`

국내 타깃별 test는 각 4개여서 1개 차이가 해당 클래스 recall 25%p를 바꾼다. 도어락은 1개뿐이며 장기 오탐률을 추정할 수 없다. 해외 other가 많은 전체 정확도는 국내 타깃 성능을 대변하지 않는다. 현장의 false alarms/hour, 이벤트 지연, DS-CNN·보드 실행은 측정하지 않았다. 독립 test 결과를 확인한 뒤 모델·임계값·특징 정책을 수정하지 않았다.

## 실행 시간·보존·결과물

실행 기간(KST): `2026-11-10T01:51:38+09:00` → `2026-11-10T02:06:50+09:00`.

실행 프로세스 경과 시간 합은 118.977초다. 최초 시작부터 최종 종료까지 911.995초이며, 중간 진단·허가 대기 시간을 포함한다. 각 실행 종료 코드는 `[1, 1, 1, 0]`다. 앞선 세 중단과 마지막 성공을 모두 보존했고 refit은 1회, test 평가는 1회다.

- refit/초기 준비: 54.764초, exit 1.
- 엄격 특징 검사: 22.991초, exit 1.
- 승인된 엄격 출력 검사: 9.915초, exit 1.
- 승인된 test 추출·평가: 31.307초, exit 0.

보호 파일 7,436개(원본·기존 CV/체크포인트·설정·split·개발 임베딩·완료한 refit)의 기존 파일 해시와 세 환경의 패키지 목록을 보존했다. 새 test 전처리·특징 1,644개 파일과 평가 5개 파일만 추가했다. Test 임베딩은 최종 평가 요청에 따라 0→411개가 됐다. 모든 GPU 실행 프로세스 종료를 확인했다.

공식 `.venv-gpu-official221`의 TensorFlow 2.21.0·tf-keras 2.21.0·CUDA 12.8 라이브러리·cuDNN 9.8.0.87을 사용했다. 새 패키지 설치·환경 버전 변경·시스템 CUDA/드라이버 변경은 0건이다. 공식 YAMNet 다운로드의 파일 해시·18,275,381 bytes·원본 모델 handle은 `official_yamnet_download.json`에 있다.

설정·체크포인트 고정은 `decision.json`, `frozen_refit.json`, 수치 기준 승인은 `approved_acceptance.json`, `final_acceptance.json`, 비교 실패와 수용 결과는 `development_embedding_probes.json`, `development_output_compatibility.json`, `accepted_development_output_check.json`에 있다. 실제 명령은 `launch_command.sh`, `continue_command.sh`, `approved_command.sh`, `final_command.sh`에 보관한다. 최초 오류 및 각 단계 로그·패키지 목록·GPU 배치·실행 시간·전후 해시를 함께 보관했다.

[Refit 모델](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/yamnet/runs/yamnet/ubuntu_gpu_refit_w1_s42/classifier.keras) · [최종 metrics](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/yamnet/reports/gpu_refit_final_w1_s42_official221_20261109T164741Z/final_test_metrics.json) · [클래스별 CSV](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/yamnet/reports/gpu_refit_final_w1_s42_official221_20261109T164741Z/final_test_class_metrics.csv) · [오탐 CSV](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/yamnet/reports/gpu_refit_final_w1_s42_official221_20261109T164741Z/final_test_false_positives.csv) · [최종 검증](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/yamnet/reports/gpu_refit_final_w1_s42_official221_20261109T164741Z/FINAL_VERIFICATION.json)
