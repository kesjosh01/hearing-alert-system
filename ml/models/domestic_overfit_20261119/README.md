# 국내 목표 음원 과적합 체크포인트 — 2026-11-19

국내 목표 음원 43개를 의도적으로 암기하도록 미세조정한 **실제 학습 체크포인트**다. 학습 파일 전체 분류는 43/43이지만 독립 시험 성능이나 제품 감지 성능이 아니다. 과거 test 타깃 8개도 이번 학습에 포함했다.

- 모델: [`classifier.keras`](classifier.keras), **359,129 bytes**
- SHA-256: `2f58eed947db6fb6d9e59c624f13635ec0a1346c421475500961f9e270c5ad2a`
- 구조: native40 DS-CNN, depthwise/pointwise 6블록, 파라미터 54,291개
- 입력: float32 `[batch,98,40,1]` log-Mel. 16 kHz mono PCM16 → 1초 창·0.5초 이동 → periodic Hann 400/160, FFT 512, 40 HTK power-Mel, 80–7,600 Hz, 자연로그. PCM은 32768로 나누며 peak 정규화하지 않는다.
- 출력: float32 `[batch,3]` softmax, 순서 `fire_alarm`, `doorbell`, `other`. 저장 체크포인트는 **T=1**이다.

이 파일은 PC용 Keras 모델이다. ESP32-S3의 `DscnnRunner`는 int8 TFLite FlatBuffer를 받으므로 `.keras`를 직접 넣을 수 없다. 이 가중치의 int8 모델·C++ 헤더는 아직 없으며, 이전 `native40_r06_v1`의 int8 파일은 다른 가중치다. 이전 int8 양자화 상수와 알림 임계값을 이 T=1 float 출력에 그대로 적용하지 않는다.

## 학습과 결과

기존 비증류 native40 체크포인트에서 시작해 BN을 고정하고 convolution·분류기를 학습했다. BN만 `fused=False` 등가 계산으로 복제했고 초기 가중치 일치와 원본 출력 일치를 확인했다. 증강·증류 없이 43개 파일의 전체 1,029창을 매 epoch 사용했다. Adam 0.0003, 파일 microbatch 3, seed 42, 전체 창의 클래스 존재/부재 MIL + 0.5×파일 평균 확률 분류 손실을 사용했다. 3 epoch 연속 성공 조건을 만족해 **9 epoch·135 optimizer 업데이트**에서 끝났다.

| 평가 정의 | 결과 | 해석 |
|---|---|---|
| 학습 파일 평균 확률로 분류 | 화재 17/17, 국내 초인종 25/25, 국내 스타일 초인종 효과음 1/1 | 학습에 쓴 파일 43/43, 100% |
| 분류와 최대 점수 조건을 함께 만족 | 43/43 | 정답 점수 ≥0.95·반대 목표 점수 <0.5. 0.95는 출력 점수이며 정확도 95%가 아님 |
| 기존 T=4·190/162·2/3 정책의 float 모사 | 화재 15/17, 초인종 26/26, 전체 41/43 | 실제 int8·실시간·보드 결과가 아님. `kr-fa-09`, `kr-fa-10` 미감지 |
| 이번 학습에서 제외한 기존 도어락 자료 | 기타 분류 7/7 → 2/7, float 정책 오알림 0/7 → 5/7 | 7개 파일·고유 녹음 6개. 목표 음원 암기와 함께 비목표 구분이 나빠짐 |

1초 창의 파일 라벨 일치율은 902/1,029(87.7%)다. 무음 구간에도 파일 라벨을 적용한 보조값이며 실제 사건 구간 정답률이 아니다. 작은 도어락 자료의 결과를 일반 생활 소리 오탐률로 확대하지 않는다.

## 함께 보관한 기록

- [`model_metadata.json`](model_metadata.json): 모델 종류·shape·클래스·전처리·배포 미검증 상태
- [`frontend.json`](frontend.json), [`labels.json`](labels.json): 입력 규약과 클래스 순서
- [`training_summary.json`](training_summary.json), [`history.csv`](history.csv): 학습 방법·epoch 이력·평가 범위
- [`per_file_results.csv`](per_file_results.csv): 43개 파일별 분류와 기존 정책 float 모사 결과
- [`controls_summary.json`](controls_summary.json), [`controls_per_file.csv`](controls_per_file.csv): 도어락 부가 진단
- [`reloaded_verification.json`](reloaded_verification.json): 11월 19일 저장 모델 재로딩·원음 재디코딩 검증 기록. 최대 확률 차이 `5.448678694897069e-7`
- [`release_manifest.json`](release_manifest.json): 이 폴더의 파일 SHA-256(자기 자신 제외)
- [마이크·모델·통신 연결 점검](../../../docs/ml/IO_INTEGRATION_REVIEW_2026-11-20.md)

원음과 비공개 백업, 로컬 절대 경로를 담은 manifest는 포함하지 않았다. 기존 체크포인트·int8 모델·통신 코드는 보존했다. 이 공개 묶음 준비 과정에서 추가 학습·추론·양자화·보드 시험을 실행하지 않았다.
