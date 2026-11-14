# YAMNet distillation과 DS-CNN 성능 개선

상태: `training_complete`. 추가 학습 fold: 16/16.

이전 DS-CNN 학습은 파일 라벨 cross entropy만 사용했다. YAMNet은 구간 검수 후보를 제안하는 데 사용됐으며, 이전 학습에 distillation은 적용되지 않았다.

개발 음원 1,640개·1초 창 15,502개를 사용했다. 국내 OOF는 화재경보 13개·초인종 21개·도어락 5개 source group이며, 해외 validation은 fold 간 재사용한다. 기존 test는 411개(타깃 8개, other 403개)다.

이번 비교는 기존 YAMNet backbone과 각 fold의 학습된 3-class head를 고정 teacher로 사용한다. teacher의 train 그룹이 student의 validation과 겹치지 않는지 확인했다. 각 student 1초 창과 같은 시작의 15,600 samples로 teacher 입력을 정렬했다. teacher의 pre-softmax logits을 temperature 2로 softmax한 확률을 사용한다.

손실은 0.5 × 파일 라벨 CE(창 평균 확률) + 0.5 × T² × 파일별 균등 평균 KL(teacher_T || student_T)이다. 사람의 시간 구간 주석이나 창별 정답을 만들지 않았다. teacher는 업데이트하지 않고 student만 학습한다. 파일별 손실은 균등하지만 BN 통계에는 microbatch의 창 분포 영향이 남는다.

기존 24채널 모델의 distillation, 주파수 정보를 남기는 시간축 pooling, 최대 Mel 4개·시간 10 frame의 마스킹, 48채널 모델을 단계적으로 비교한다. 클래스별 판정 임계값은 개발 OOF·기존 해외 validation에서만 고른다. 기존 해외 other 오탐률 5% 이하와 도어락 오탐 0 가드를 유지한다.

| 후보 | 4개 fold 타깃 recall | 평균 ± 표준편차 | fire / bell 임계값 | 파라미터 |
|---|---|---|---|---|
| baseline_w1 | 62.50%, 70.00%, 83.33%, 56.67% | 68.12% ± 11.51%p | 0.4 / 0.6 | 3,339 |
| baseline_w2 | 50.00%, 63.33%, 100.00%, 66.67% | 70.00% ± 21.26%p | 0.4 / 0.4 | 3,339 |
| baseline_w4 | 58.33%, 63.33%, 83.33%, 66.67% | 67.92% ± 10.83%p | 0.5 / 0.0 | 3,339 |
| kd24 | 62.50%, 63.33%, 100.00%, 56.67% | 70.63% ± 19.81%p | 0.5 / 0.6 | 3,339 |
| kd24_freq_aug | 62.50%, 80.00%, 83.33%, 66.67% | 73.13% ± 10.10%p | 0.0 / 0.55 | 4,707 |
| kd24_freq | 75.00%, 63.33%, 83.33%, 66.67% | 72.08% ± 8.96%p | 0.4 / 0.0 | 4,707 |
| kd48_freq_aug | 70.83%, 63.33%, 83.33%, 53.33% | 67.71% ± 12.64%p | 0.0 / 0.4 | 12,867 |

baseline_w1/2/4는 보존된 모델의 공통 임계값을 클래스별 임계값으로 재보정한 대조군이다. 해당 checkpoint는 이전 학습에서 선택됐으므로 KD만의 효과를 완전히 분리한 비교는 아니다. 표의 성능은 모델·임계값 선택에도 쓰인 작은 개발 자료의 수치이며 독립 일반화 성능으로 해석하지 않는다.

동일한 정렬 창에서 frozen fold teacher의 개발 평균 타깃 recall은 78.33%다. 기존 YAMNet의 native patch 평가와는 창·신규 데이터가 다르며 student의 상한을 뜻하지 않는다.

선택: `kd24_freq_aug`. 최종 모델에 distillation 적용: `True`. 고정 refit: 9 epoch. 판정 임계값: `{'fire_alarm': 0.0, 'doorbell': 0.55}`.

사용자 요청으로 해외 siren을 비목표 소리의 오탐 평가에서 제외했다. test의 8개, validation의 8개 전체를 예측과 무관하게 분자·분모에서 제외하며 모델·임계값은 변경하지 않았다. 원래 평가 JSON과 모델 묶음은 보존했다. [추가 데이터·epoch 판단과 최신 평가 정의](evaluation_scope_v2/REPORT_KO.md)를 함께 확인한다.

float: 기존 test 타깃 recall 62.50%; fire_alarm 2/4, doorbell 3/4; 해외 비목표 소리 오탐 26/394 (6.60%, siren 8개 제외); 원래 정의 29/402; 국내 도어락 오탐 0/1.

int8: 기존 test 타깃 recall 62.50%; fire_alarm 2/4, doorbell 3/4; 해외 비목표 소리 오탐 24/394 (6.09%, siren 8개 제외); 원래 정의 27/402; 국내 도어락 오탐 0/1.

Float/int8 파일 판정 일치율 99.51%, 최대 확률 차이 0.033864.

기존 test int8 기준 재현율은 50.00% → 62.50%로 높아졌지만, 해외 비목표 소리 오탐은 3/394 → 24/394 (6.09%)로 늘었다. 개발 fold의 가드는 통과했지만 기존 test의 오탐률은 5%보다 높다. 모든 감지 지표가 개선됐다고 볼 수 없으며 실사용 교체·배포를 권장하지 않는다. 결과는 개선 실험 모델로 보존한다.

기존 test는 이미 YAMNet과 이전 DS-CNN에서 관측했다. 추가 탐색 중에는 읽지 않았으며 선택·refit·양자화 모델 해시를 고정한 뒤 비교한다. 이 결과는 재사용 자료의 회고적 비교이며 새로운 독립 시험이 아니다. test를 보고 다시 튜닝하지 않는다.

실제 distillation CV 16회, optimizer update 8,550회를 검증했다. 선택 모델의 int8 파일은 15,304 bytes다. 원본·분할·YAMNet·기존 DS-CNN 모델과 환경을 보존했다.

신규 초인종의 저장 파일 추론 확인(학습 자료이며 독립 평가가 아님):

| ID | float 판정 | int8 판정 |
|---|---|---|
| kr-user-db-06 | other | other |
| kr-user-db-07 | doorbell | doorbell |
| kr-user-db-08 | doorbell | doorbell |
| kr-user-db-09 | doorbell | doorbell |
| kr-user-db-10 | doorbell | doorbell |

CLI 정상 실행과 올바른 분류를 구분한다.

신규 초인종 4/5개를 doorbell로 판정했다. 초기 baseline은 같은 학습 자료에서 2/5개였으며, 학습 파일 확인 결과를 독립 평가 정확도로 해석하지 않는다.

새 프로젝트·환경·패키지를 설치하지 않았다. 원음은 외부에 업로드하지 않았고 W&B 대신 로컬 기록을 사용했다. 시간 구간 검수·실시간 마이크·ESP-IDF 전체 빌드·보드 RAM/지연·시간당 오탐·통합 알림은 미검증이다. `deployment_ready=false`다.

방법 근거: [Hinton et al.의 distillation](https://arxiv.org/abs/1503.02531), [Keras 공식 구현](https://keras.io/examples/vision/knowledge_distillation/), [TensorFlow YAMNet 전이학습](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio), [SpecAugment](https://arxiv.org/abs/1904.08779). 이 문헌의 성능 수치를 우리 데이터 결과로 옮겨 쓰지 않았다.
