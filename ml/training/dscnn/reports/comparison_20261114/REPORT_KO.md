# 모델 학습·비교 완료 기록 — 2026-11-14

새 공개 시험에서 직전 확장 모델 대비 INT8 목표 발생원 재현율 차이는 +12.50%p, 비목표 파일 오탐률 차이는 -19.40%p다. **새 시험 오탐률 5% 조건은 미충족**이며 실제 기기 교체를 권고하지 않는다. 최초 모델의 새 시험 재현율 58.33%보다 최종 후보의 54.17%가 낮으므로 모든 이전 모델보다 좋아졌다고 결론 내릴 수 없다. 최종 시험은 가중치·집계·float/int8 임계값·시간적 정책을 고정한 뒤 1회 평가했으며, 결과에 맞춘 재선정은 하지 않았다.

## 최종 모델

`dscnn_comparison_20261114_s7_freq24_seed42`. `freq24`, top 50% 창 집계, 비목표 순환 추출, 국내 가중치 4:1, other 비중 1/3, 증류 비중 0.0. 27 epoch 실행 후 선택용 자료로 고른 **19 epoch** 가중치를 전달한다. 매 epoch 30 optimizer steps, 전체 810 steps이며, 저장된 checkpoint까지는 570 steps다. 이후 epoch는 학습 이력에만 남는다. INT8 15,600 bytes, 파라미터 4,707개, 창당 MAC 2,541,600이다.

묶음: [dscnn_comparison_20261114](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/models/dscnn_comparison_20261114/README.md). 가중치 SHA-256: `613bfb837ba42e5b9db65babbb512e3b0a1b3bddca0fe3d263ea617e7fc669b3`. `pipeline_only=false`, `deployment_ready=false`. 공식 YAMNet 고정 백본의 임베딩 위에 3클래스 교사 head를 실제 학습하고, 같은 학습 발생원에서 만든 교사 logits로 50%/25%/0% 증류 비교를 수행했지만, 최종 선택은 증류 0%여서 **최종 모델을 증류 모델이라고 부르지 않는다**.

## 고정 시험 결과

| 자료 | 모델 | 화재경보 발생원 정답 | 초인종 발생원 정답 | 목표 발생원 macro recall | 해외/공개 비목표 파일 FP |
| --- | --- | --- | --- | --- | --- |
| 새 시험 | Initial | 2/4 | 2/3 | 58.33% | 4/67 (5.97%) |
| 새 시험 | Previous KD | 2/4 | 1/3 | 41.67% | 8/67 (11.94%) |
| 새 시험 | Previous expanded | 2/4 | 1/3 | 41.67% | 20/67 (29.85%) |
| 새 시험 | New split baseline | 2/4 | 1/3 | 41.67% | 4/67 (5.97%) |
| 새 시험 | Final comparison | 3/4 | 1/3 | 54.17% | 7/67 (10.45%) |
| 기존 회고 | Initial | 2/4 | 2/4 | 50.00% | 3/394 (0.76%) |
| 기존 회고 | Previous KD | 2/4 | 3/4 | 62.50% | 24/394 (6.09%) |
| 기존 회고 | Previous expanded | 3/4 | 2/4 | 62.50% | 46/394 (11.68%) |
| 기존 회고 | New split baseline | 2/4 | 2/4 | 50.00% | 25/394 (6.35%) |
| 기존 회고 | Final comparison | 2/4 | 3/4 | 62.50% | 18/394 (4.57%) |

목표 지표는 발생원 내 파일 확률을 평균하고 판정한 뒤 화재경보·초인종 재현율을 동등 평균한다. 파일 단위 재현율과 혼동은 아래 원본 CSV에 보존했다. 기존 회고 자료와 새 시험은 난이도·분포가 달라 서로의 점수 차이를 개선량으로 해석하지 않는다. 기존 모델 3개의 회고 예측은 저장된 원본 예측을 재사용했고, 같은 403개 구성과 394개 해외 비목표 분모로 비교했다.

새 시험의 직전 확장 모델 대비 발생원 재현율 차이에 대한 층화 paired bootstrap 기술적 95% percentile 구간은 [+0.00, +37.50]%p다(10,000회, seed 20261114). 양성 발생원이 화재경보 4개·초인종 3개뿐이라 불확실성이 크고, 독립적인 국내 성능 향상이나 모집단 오탐률 5% 이하를 입증하지 않는다.

## 요청한 8단계와 비채택 결과

| 단계 | 실제 수행·결론 |
| --- | --- |
| 1. 최종 가중치 임계값 | 학습/선택/보정을 발생원별 분리. 실제 최종 float와 INT8 출력으로 각각 보정. 보정 점수로 모델을 재선정하지 않음. |
| 2. other 노출 | 선택된 baseline checkpoint는 650/1179개, 최종 checkpoint는 1179/1179개 사용. 순환 추출 채택, 혼동 유형 가중치 2배는 비채택. |
| 3. 짧은 이벤트 희석 | 평균→상위 50%→상위 25%를 한 설정씩 비교. 상위 50% 채택. 세 클래스는 같은 창을 선택해 유효한 확률 분포를 유지. |
| 4. 추가 데이터 | 공개 신규 학습 38개를 별도 비교했으나 비채택. 새 공개 시험 79개 확보. 국내 추가 실녹음은 없었고 Downloads 5개는 기존 파일과 SHA가 같음. |
| 5. 추출 비율 | 국내 가중치 2:1, other 비중 50%를 각각 비교했으나 비채택. 4:1 및 1/3 유지. |
| 6. 증류 | 동일 교사·자료·구조에서 50%, 25%, 0% 비교. 0% 채택. 교사 오류 전파 원인까지 입증한 것은 아님. |
| 7. 구조·epoch | temporal24/freq24 각각 seed42/43/44. 평균 재현율 둘 다 87.50%; 평균 FPR은 0.87% 대 0.52%. 작은 freq24 채택. 최고 seed를 고르지 않고 사전 지정 seed42 전달. plateau 학습률은 별도 비교 후 비채택. epoch 상한30/patience8 유지. |
| 8. 시간적 안정 | single→2-of-3, release0.8, cooldown2s를 각각 비교. 2-of-3는 짧은 초인종 누락과 추가 지연으로 비채택; 나머지도 선택 지표 개선이 없어 비채택. 실제 오디오 시각을 받는 causal state와 별도 재생 임계값 구현. |

### 모든 학생 학습

| run | 단계/seed | 선택 epoch / 실행 epoch | 선택 목표 재현율 | 선택 FPR | 저장 checkpoint의 other 노출 |
| --- | --- | --- | --- | --- | --- |
| s1_baseline | 1/42 | 5 / 13 | 79.17% | 1.56% | 650/1179 |
| s2_coverage | 2/42 | 12 / 20 | 79.17% | 0.52% | 1179/1179 |
| s2_hardneg | 2/42 | 10 / 18 | 79.17% | 0.52% | 1179/1179 |
| s3_top50 | 3/42 | 10 / 18 | 79.17% | 0.00% | 1179/1179 |
| s3_top25 | 3/42 | 3 / 11 | 79.17% | 0.52% | 850/1179 |
| s4_added | 4/42 | 3 / 11 | 79.17% | 1.04% | 871/1210 |
| s5_domestic2 | 5/42 | 2 / 10 | 79.17% | 0.52% | 591/1179 |
| s5_other50 | 5/42 | 3 / 11 | 79.17% | 0.52% | 1179/1179 |
| s6_kd25 | 6/42 | 13 / 21 | 79.17% | 0.52% | 1179/1179 |
| s6_kd0 | 6/42 | 12 / 20 | 91.67% | 2.08% | 1179/1179 |
| s7_temporal24_seed43 | 7/43 | 18 / 26 | 91.67% | 0.52% | 1179/1179 |
| s7_temporal24_seed44 | 7/44 | 3 / 11 | 79.17% | 0.00% | 945/1179 |
| s7_freq24_seed42 | 7/42 | 19 / 27 | 91.67% | 0.00% | 1179/1179 |
| s7_freq24_seed43 | 7/43 | 11 / 19 | 79.17% | 0.00% | 1179/1179 |
| s7_freq24_seed44 | 7/44 | 4 / 12 | 91.67% | 1.56% | 1179/1179 |
| s7_plateau | 7/42 | 19 / 27 | 91.67% | 0.00% | 1179/1179 |

이 표는 작은 선택용 자료에 반복 사용한 모델 선택 점수이며 새로운 교차검증 정확도가 아니다. 원인별 효과는 해당 부모 후보와의 한 설정 비교로만 해석한다. 3개 seed 반복은 구조 비교에만 수행했고, 모든 설정을 다중 seed로 검증한 것은 아니다. 순환 방식 변경은 난수 소비 순서도 바꾸므로 모든 실험의 개별 배치가 같았다고 주장하지 않는다.

## 보정과 시간적 재생

| 실제 최종 모델 | 화재경보/초인종 임계값 | 보정 목표 발생원 재현율 | 보정 FP |
| --- | --- | --- | --- |
| float | {'fire_alarm': 0.0, 'doorbell': 0.55} | 70.83% | 4/198 |
| int8 | {'fire_alarm': 0.0, 'doorbell': 0.6} | 70.83% | 4/198 |

화재경보 임계값 0도 argmax가 화재경보일 때만 통과한다. 보정 자료에서 측정한 점수는 독립적인 시험 성능이 아니다. 임계값 보정은 가중치를 변경하지 않는다.

| 시간적 재생(INT8) | 목표 source-mean file recall | FP 파일/67 | 비목표 알림 수 | 비목표 재생 길이(초) |
| --- | --- | --- | --- | --- |
| single_at_same_thresholds | 41.67% | 9 | 28 | 2347.67 |
| chosen | 41.67% | 9 | 28 | 2347.67 |

파일 집계용 임계값을 매 창 알림에 그대로 쓰지 않는다. 별도 held-out 보정값을 사용한다. 재생 지표는 발생원별 파일 감지 비율을 평균하므로 위 파일 확률 집계 지표와 정의가 다르다. 실제 이벤트 구간 주석이 없으므로 이벤트별 재현율, 진짜 소리 onset부터의 감지 지연은 미측정이다. 파일마다 상태가 초기화되며 짧은 파일은 1초까지 0 padding한다. 마지막 창은 원본 frontend의 end-aligned 시각을 사용한다. 음성 샘플링 시각을 덮어쓰지 않는다. 새 시험 재생 수치는 짧은 공개 녹음 모음의 결과이며 현장 시간당 오탐 보장이 아니다.

## 자료·라벨·보존 범위

기존 개발 자료 역할별 구성: `{"train": {"clips": {"doorbell": 44, "fire_alarm": 36, "other": 1179}, "groups": {"fire_alarm": 25, "doorbell": 24, "other": 907}}, "selection": {"clips": {"fire_alarm": 9, "doorbell": 5, "other": 193}, "groups": {"fire_alarm": 6, "doorbell": 4, "other": 146}}, "calibration": {"clips": {"doorbell": 4, "fire_alarm": 10, "other": 199}, "groups": {"fire_alarm": 6, "doorbell": 4, "other": 145}}}`. 신규 공개 자료: `{"train": {"fire_alarm": {"clips": 4, "groups": 3}, "doorbell": {"clips": 3, "groups": 1}, "other": {"clips": 31, "groups": 8}}, "fresh_test": {"fire_alarm": {"clips": 4, "groups": 4}, "doorbell": {"clips": 8, "groups": 3}, "other": {"clips": 67, "groups": 10}}}`. 총 117개를 원본과 별도로 저장했고 38개 학습 후보·79개 시험으로 발생원 관련 묶음을 분리했다. 시험 79개는 양성 12개(화재경보4·초인종8), 비목표67개다. 관련 파일을 합치면 목표 7개·비목표10개 source family이므로 79개 독립 기기라고 세지 않는다.

국내 사용자가 직접 청취해 수집한 파일 라벨을 사용했다. 이벤트 시간 구간을 사람이 검수했다고 기록하지 않았다. 새 공개 음원은 출처 제목·분류를 확인한 파일 라벨이며 사람의 전 구간 청취나 기기 국가는 검증하지 않았다. 동일 source ID, 원본 bytes SHA, 디코딩 PCM SHA 및 알려진 제작자·기기 계열 중복을 검사했다. 공개 메타데이터만으로 모든 편집본 재사용·기기 독립성·foundation 사전학습 미노출을 보증할 수 없다.

해외 도어벨·화재경보·사이렌은 목표음으로 취급했다. 기존 test의 ESC 사이렌8개는 예측을 보기 전에 기존 기준대로 FP 집계에서 제외했다. 새 공개 사이렌은 fire_alarm 양성이며, 오탐을 줄이려고 틀린 other 예측을 사후 제외하지 않았다.

Commons CDN429 이후 해당 요청을 중단하고 허용된 다른 공개 출처로 진행했다. 발음·노래·장난감·기차 차임·실제 전화벨 계열과 모호한 후보는 자료 정의에 따라 제외했다. 개인 정보·음원을 외부에 업로드하지 않았다. 원본·이전 가중치·기존 실행·환경은 보존하며 앱/패키지/드라이버 설치, 마이크/보드 접근, Git commit/push는 하지 않았다. 공개 출처·저작자·라이선스는 [크레딧](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/data/comparison_20261114/PUBLIC_DATA_CREDITS.md)에 있다.

## 산출물과 검증 경계

- [실행 가능한 모델 묶음](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/models/dscnn_comparison_20261114/README.md)
- [고정 시험 결과 JSON](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/reports/comparison_20261114/evaluation/results.json), [고정 직전 모델·임계값](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/reports/comparison_20261114/evaluation/preregistered_models.json)
- [전체 비교 ledger](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/reports/comparison_20261114/ledger.json), [3-seed 구조 결과](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/reports/comparison_20261114/architecture_replicates.json)
- [자료 freeze](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/data/comparison_20261114/source_freeze.json), [선택 checkpoint 노출 재구성](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/runs/dscnn_comparison_20261114_s7_freq24_seed42/selected_checkpoint_exposure_audit.json)
- [최종 보존·provenance 감사](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/reports/comparison_20261114/final_audit.json), [로컬 CLI 일치 확인](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/reports/comparison_20261114/cli_verification.json)

소프트웨어·실제 오디오 학습·호스트 float/INT8 추론을 검증했다. 실제 마이크, ESP32-S3/TFLM AllocateTensors, tensor arena peak RAM, 보드 처리 지연, 장시간 현장 성능은 수행하지 않았다. 최대 단일 activation은 peak RAM이 아니다. 기존 모델을 덮어쓰거나 실제 기기의 모델을 교체하지 않았다.

## 그래프

![frozen_evaluation](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/reports/comparison_20261114/figures/frozen_evaluation.png)

![selection_runs](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/reports/comparison_20261114/figures/selection_runs.png)

![architecture_seeds](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/reports/comparison_20261114/figures/architecture_seeds.png)

![learning_curves](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/reports/comparison_20261114/figures/learning_curves.png)

![temporal_selection](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/reports/comparison_20261114/figures/temporal_selection.png)

그래프는 기존 시스템 Matplotlib로 생성했다. 별도 설치 없이 PNG/SVG와 원본 CSV를 함께 저장했다. 검정 평균선은 3개 seed 산술평균이며 신뢰구간이 아니다. 학습곡선은 smoothing 없이 실제 epoch 기록을 표시한다. 정확도/FPR 막대는 0 기준, 비교 패널의 같은 축 범위를 사용했다.

## 방법 근거

임계값은 학습과 분리된 자료에서 고정한다: [scikit-learn 공식 가이드](https://scikit-learn.org/stable/modules/classification_threshold.html). 파일 라벨 pooling은 약한 라벨 음향 감지 접근을 참고했다: [Kong et al.](https://arxiv.org/abs/1804.10070). 시간적 후처리는 모델과 별도로 비교했다: [sound event detection postprocessing 연구](https://arxiv.org/abs/2306.15440). 본 결과의 수치는 논문에서 가져오지 않고 이 작업의 실제 로그에서 계산했다.


## 시험 후 오류 해석과 배포 상태

**새 공개 시험 오탐률은 7/67(10.45%)로 5% 목표를 충족하지 못했다. 직전 확장 모델보다 개선됐지만 모든 과거 모델보다 우수한 것은 아니며, 이 묶음은 연구용 비교 후보이고 실제 기기 교체를 권고하지 않는다.**

새 시험에서 어느 모델도 비목표 파일 오탐률 5% 조건을 통과하지 못했다. 최종 후보의 오탐 7건은 시계 알람4·교회 종2·문 삐걱임1이다. 산업용 벨5개는 모두 놓치거나 화재경보로 혼동했으며, 이 5개는 같은 계열 1개 발생원 그룹이다. 학교 화재경보1개와 다른 초인종1개도 놓쳤다. 추가 학습을 검토할 때는 서로 다른 실제 화재경보기·가정용/기계식 초인종, 시계/교회 종·문 소리, 문 너머·먼 거리·생활 소음 녹음을 우선해야 한다. 이번 공개 자료 38개 추가가 선택 지표를 개선하지 못한 사실은 추가 데이터가 불필요하다는 근거가 아니다. 현재 시험 오류 파일은 다음 버전에서 개발에 쓸 경우 더 이상 독립 시험으로 부를 수 없다. 이후 개선을 검증하려면 별도 발생원의 새 시험이 필요하다. 이번 결과를 이용해 모델·임계값·라벨을 재선정하지 않았다.

새 시험에 국내 추가 녹음은 없다. raw 재생 JSON의 국내 재현율 0은 빈 집단의 코드상 기본값이며, 국내 재현율이 0%라고 측정된 것이 아니다. 해당 지표는 N/A로 해석한다. baseline과 최종 학습곡선의 loss는 증류 비중이 달라 서로 다른 목적 함수이며, loss의 절대 크기를 직접 개선량으로 비교하지 않는다.

[오류별 출처와 판정 CSV](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/reports/comparison_20261114/fresh_error_cases.csv), [후속 분석 기록](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/reports/comparison_20261114/fresh_error_analysis.json), [연구용 릴리스 상태](/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/models/dscnn_comparison_20261114/RELEASE_STATUS.json).
