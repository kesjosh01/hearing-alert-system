# ESP32-S3용 1초 DS-CNN

## 최신 결과와 연결 경로

2026-11-14 후속 연구는 실제 optimizer 학습 33회·BN 통계 보정 2건·실제 int8 후보와 새 공개 559개 파일의 고정 평가까지 완료했다. 최신 전체 후보는 [native40_r06_v1](../../models/pursuit_20261114/README.md)이며 88,280-byte int8, 기존 98×40 전처리·확률 출력이다. [통신 담당자 안내](../../../docs/ml/MODEL_HANDOFF.md)에 파일 위치·양자화·시간 판정·이벤트 ID 매핑을, [현재 진행 상황](../../../docs/ml/DEVELOPMENT_STATUS_2026-11-14.md)에 결과와 한계를 정리했다. [전체 체크포인트 색인](../../models/checkpoints_20261114/README.md)에서 당시 저장된 127개 원본 `.keras`를 찾는다.

최신 원본 보고서는 [STAGE2_REVIEW_KO.md](reports/pursuit_20261114/STAGE2_REVIEW_KO.md)다. 실제 제품 목표는 미검증이고 추가 데이터 수집은 중단 상태다. 아래는 앞선 baseline·공개 정답 확장 단계의 상세 규약과 기록이며 최신 후보의 출력·정책은 해당 묶음의 `delivered_output_contract.json`을 우선한다. 학습 재현에는 별도 원음·분할·로컬 데이터 경로가 필요하다.


기준 YAMNet의 refit·최종 평가 완료 기록을 확인한 뒤 개발한 ESP32-S3 대상 경량 모델이다. 세 클래스는 `fire_alarm=0`, `doorbell=1`, `other=2`다. 기존 `ko-source-cv4-20261108-v1` 분할과 고정 test를 보존하고, 2026-11-14에 확인한 국내 초인종 5개를 DS-CNN 전용 개발 확장 `ko-source-cv4-20261114-dscnn-v2`로 추가했다. 원본 manifest·YAMNet 분할은 변경하지 않는다.

## 공개 정답 데이터 확장 재학습

후속 요청으로 한국 공개 자료의 도어벨 24개, CC0 해외 화재·연기·대피경보 9개와 도어벨 6개를 수집했다. [출처·라이선스](../../models/dscnn_expanded_20261114/PUBLIC_DATA_CREDITS.md)와 원음을 로컬에 보관한다. 실제 기기 국가가 확인된 자료라는 뜻은 아니며 새 라벨의 근거는 공개 카탈로그다. 학습용 해외 사이렌 24개는 이 버전에서 `fire_alarm` 정답에 포함했다. 국내 정답 source group은 해외의 **4배** 가중치, 클래스별 학습 추출 확률은 각 **1/3**이다. 최종 학습 자료는 화재경보47개/31그룹·도어벨53개/32그룹이며, 새 공개 자료는 모두 학습 전용이다.

새 라벨로 YAMNet head를 다시 학습하고 기존 구조 `freq24`와 시간축 kernel을 넓힌 `temporal24`를 같은 4-fold에서 비교했다. 개발 평균 recall은 **76.25% 대 76.88%**로 `temporal24`를 선택했다. 시간 평균 전 국소 convolution 범위는15→51 frame이며, 6,003파라미터·창당3,811,680 MAC·int8 **16,896 bytes**다. 최종5epoch·150update를 완료했다. 초기 refit에서 새 파일1개가 미추출된 것을 발견해 test를 보기 전에 학습 타깃100개가 모두 추출되는 시드43으로 최종 refit을 보완했으며 초기 결과도 보존했다.

기존 test의 평균 recall은 **62.5% 유지**, 해외 비목표 오탐은 **24/394(6.09%)→46/394(11.68%) 증가**였다. 최종 float refit validation도49/390(12.56%)로5% 목표를 넘었다. 네 CV checkpoint의 가드 통과와 최종 refit 모델의 가드 통과를 구분한다. 새 모델의 실사용 교체를 권장하지 않는다. [확장 학습 보고서](reports/expansion_20261114/REPORT_KO.md), [곡선](reports/expansion_20261114/TRAINING_REPORT.html), [검증](reports/expansion_20261114/FINAL_VERIFICATION.json), [새 실험 모델 묶음](../../models/dscnn_expanded_20261114/README.md)에 실제 결과를 기록한다. 이 결과를 보고 임계값을 다시 조정하지 않았다. 앱·프로그램·확장 프로그램·패키지·새 환경은 다운로드하거나 설치하지 않았으며 원본·분할·기존 모델·환경은 보존했다.

## 초기 baseline과 이전 distillation 기록

2026-11-14에 사용자가 직접 청취한 파일 라벨로 baseline 학습을 마친 뒤 **YAMNet teacher distillation을 추가 구현·학습·양자화**했다. 당시 개발 자료는 1,640개, 1초 창은 15,502개다. 초기 국내 가중치 1·2·4의 12개 fold와 추가 distillation 후보 4개의 16개 fold를 완료했다. 재부팅 전 완료된 결과와 중단된 실행도 보존했다. 당시 결과는 [distillation 개선 보고서](reports/improvement_20261114/REPORT_KO.md), [곡선](reports/improvement_20261114/TRAINING_REPORT.html), [완료 검증](reports/improvement_20261114/FINAL_VERIFICATION.json)에 있다.

개선 후보의 선택 결과는 `kd24_freq_aug`이며 국내 가중치 1, 고정 refit 9 epoch, fire/bell 임계값 0.0/0.55다. 4-fold 평균 타깃 recall은 **59.79% → 73.13%**로 높아졌다. 기존 test int8의 타깃 recall은 **50.00% → 62.50%**지만 해외 비목표 소리 오탐은 **3/394 → 24/394(6.09%)**로 늘었다. 이는 사용자 요청에 따라 해외 `siren` 8개 전체를 제외한 최신 평가 정의다. 원래 정의의 **6/402 → 27/402(6.72%)**와 원본 평가 JSON은 보존했다. [추가 데이터·epoch 판단과 평가 수정](reports/improvement_20261114/evaluation_scope_v2/REPORT_KO.md)에 제외 목록과 검증을 기록했다. 개발 가드는 통과했지만 기존 test의 오탐률은 5%를 넘었으므로 실사용 교체·배포를 권장하지 않는다. [개선 실험 모델 묶음](../../models/dscnn_improved_20261114/README.md)에 실제 float·int8 모델과 C++ 헤더를 보관한다. 원음·기존 분할·YAMNet·[초기 baseline 모델](../../models/dscnn_filelabels_20261114/README.md)·환경 패키지는 보존했다. `pipeline_only*`는 별도의 합성 검증 모델이다. 실시간 마이크·ESP-IDF 전체 빌드·실보드 동작은 미검증이다.

## 입력과 모델

아래 규약과 초기 구조 비교는 이전 baseline·distillation 기록이다. 최신 `temporal24`는 같은 입력·49×1 pooling을 유지하면서 세 depthwise kernel을9×3으로 변경했고, 6,003파라미터·3,811,680 MAC·int8 16,896 bytes다.

| 항목 | v1 규약 |
|---|---|
| 오디오 | mono signed PCM16, 16 kHz, 16,000 samples(1초) |
| 진폭 | `/32768`, 창별 peak normalization 없음 |
| 긴 녹음 | 8,000 samples hop, 마지막 완전한 창 추가 |
| 짧은 녹음 | 오른쪽 zero padding |
| STFT | 400 samples/25 ms periodic Hann, 160 samples/10 ms hop, FFT 512 |
| FFT | float32 radix-2, 정규화하지 않은 power spectrum |
| Mel | HTK, 40개 비정규화 삼각 필터, 80–7,600 Hz |
| 로그 | 자연로그 `ln(max(mel_power, 1e-6))` |
| 입력 텐서 | `[batch,98,40,1]` NHWC; STFT 끝 padding 없음(마지막 80 samples 미사용) |
| 출력 | `[batch,3]` softmax, 위의 고정 클래스 순서 |

초기 baseline은 `Conv3×3/stride2/24 → BN → ReLU → [Depthwise3×3 → BN → ReLU → Pointwise1×1/24 → BN → ReLU] ×3 → AveragePool49×20 → Dense3 → Softmax`다. 선택된 개선 모델은 pooling을 **49×1**로 바꿔 시간축을 평균하고 주파수 20칸을 최종 분류층까지 보존한다. BN은 int8 변환 시 합쳐진다. 개선 모델은 **4,707 파라미터**, conv·dense 기준 창당 **2,541,600 MAC**이다. 초기 baseline은 3,339 파라미터·2,540,232 MAC이었다.

개선 모델의 실제 int8 flatbuffer는 **15,304 bytes**이고 초기 baseline은 13,656 bytes다. 최대 단일 int8 activation은 23,520 bytes, 입력은 3,920 bytes, C++ FFT workspace는 5,124 bytes다. 이 숫자들은 **보드 전체 RAM이나 tensor arena 크기가 아니다**. PCM 버퍼만도 32,000 bytes이며, 스트리밍 이중 버퍼·TFLM scratch/arena·stack·무선 통신 메모리는 별도다. RAM·처리 시간 목표는 아직 정하지 않았다.

## YAMNet teacher distillation 개선

초기 baseline의 학습 손실에는 distillation이 없었다. 구간 검수 후보 생성에 쓰인 YAMNet과 실제 teacher-student 학습을 구분한다. 이번 추가 학습은 기존 YAMNet backbone과 각 fold의 3-class head를 고정 teacher로 사용했다. teacher의 train 그룹이 해당 student의 validation과 겹치지 않는지 먼저 검사했다. 모든 student 창과 같은 시작의 15,600 PCM16 samples를 teacher에 입력해 정렬하고, pre-softmax logits을 사용한다.

`improvement/model.py`의 실제 손실은 `0.5 × CE(파일 라벨, 창 평균 확률) + 0.5 × T² × mean_file(mean_window(KL(teacher_T || student_T)))`, `T=2`다. teacher는 업데이트하지 않는다. 사람의 창별 정답·시간 구간 검수로 기록하지 않는다. 후보별로 주파수 정보 보존, Mel 최대 4칸·시간 최대 10 frame 마스킹, 48채널 확장을 비교했다. 48채널 모델은 개발 성능이 낮아 선택하지 않았다. teacher 원리와 공식 구현은 개선 보고서에 링크했다.

임계값은 클래스별로 개발 OOF·기존 해외 validation에서 선택했다. 가드와 새 후보·비교 규칙은 실제 학습 전에 `reports/improvement_20261114/execution_plan.json`에 고정했다. 개선 탐색 중에는 test를 읽지 않았으며 최종 모델·정책 해시를 고정한 뒤 기존 test를 회고적으로 비교했다. 이 결과로 재선택하지 않았다. 기존 test는 과거에 관측한 자료이므로 새 독립 평가가 아니다. 이전 checkpoint의 선택 방식도 달라 모든 향상을 KD 하나의 효과로 해석하지 않는다.

## 파일 라벨 학습과 개발 임계값 선택

사용자가 국내 음원을 직접 들으며 수집했다고 확인했고 파일 라벨 방식을 선택했다. 새 초인종 5개도 개인 집에서 녹음한 서로 다른 발생원이라고 확인했다. 이 확인을 로컬 `data/file_label_approval_20261114.json`에 기록한다. 청취 확인을 이벤트 시간 구간 검수로 기록하지 않는다. 초기 baseline은 사람의 파일 라벨만 사용했고, 추가 개선은 위에 설명한 teacher soft-target 손실을 함께 사용한다. teacher 출력을 사람이 검수한 정답 구간으로 기록하지 않는다.

`--supervision file_labels --file-label-approval ...`는 파일의 모든 유효 1초 창을 하나의 bag으로 묶는다. 창별 softmax를 평균한 파일 확률에 cross entropy를 한 번 계산한다. 긴 파일도 파일 하나의 손실 비중을 가지며, 원본 그룹 → 그룹 안의 파일 균등 선택 순서다. 클래스·국내 가중치·other pool의 원본 그룹 확률은 유지한다. 6개 파일 microbatch 5개의 기울기를 누적해 30개 파일마다 Adam을 한 번 갱신한다. 한 epoch는 900개 파일 추출이며 기존 YAMNet의 900개 patch 추출과 학습 단위가 다르다. BN 통계는 microbatch 내부 창 분포의 영향을 받는다.

최대 30 epoch, patience 8, learning rate 0.001, seed `42+fold`다. 각 개발 checkpoint에서 사전 지정한 임계값 grid를 비교하고, 국내 OOF와 각 fold의 기존 해외 validation으로 가중치별 공통 임계값을 선택한다. **해외 other 오탐률 ≤5%, 국내 도어락 오탐 0**이라는 기존 개발 가드를 유지한다. 임계값은 평균 확률의 argmax가 타깃일 때만 적용하며, 확률이 임계값보다 작으면 `other`로 판정한다. 원시 확률을 수정하지 않는다. Source group에서는 파일 확률을 동일 가중치로 평균한 뒤 판정한다.

초기 baseline은 국내 가중치 1·2·4를 동일한 4-fold와 seed로 비교하고 일관된 개선이 없으면 작은 가중치를 유지했다. 추가 개선의 별도 후보·선택 규칙은 위 실행 계획을 따른다. 두 단계 모두 선택한 4개 fold의 best epoch 중앙값을 반올림해 refit epoch로 고정하며 refit에서 epoch를 다시 선택하지 않는다. 각 단계의 임계값·가중치·모델을 고정한 뒤 기존 test 411개를 비교했다. 이 test는 YAMNet에서도 이미 사용했으므로 새 독립 시험이 아니다. 새 국내 음원·감독 방식·학습 단위가 달라 YAMNet과 모델 구조만의 통제 비교도 아니다.

전체 실행은 `python -m dscnn_baseline.study --help`, 단일 fold는 `python -m dscnn_baseline train --help`, 고정 refit은 `python -m dscnn_baseline refit --help`에서 확인한다. 이번 실행은 기존 `.venv`와 지정 GPU UUID를 사용하며 W&B 대신 로컬 CSV·JSON·[HTML 곡선](reports/real_training_20261114/TRAINING_REPORT.html)을 저장한다. 실제 명령은 study `execution_plan.json`과 각 run `run.json`에 기록한다. 최초 오류에서 중단하고 기존 결과를 덮어쓰지 않는다.

## 학습 모델로 저장 음원 추론

Ubuntu PC의 터미널에서 아래 작업 디렉터리로 이동한다. 기존 DS-CNN 환경으로 CPU에서 저장 파일 하나를 추론하며, 새 설치·마이크 접근·학습을 실행하지 않는다. 명령은 foreground로 끝나므로 유지할 서버는 없다. JSON의 `probabilities`, `predicted_label`, `window_count`와 종료 코드 0을 확인한다. 예시 음원은 학습에 사용된 개발 파일이므로 독립 평가가 아니다.

```bash
cd '/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn'
.venv/bin/python -B tools/predict_audio.py \
  --bundle '/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/models/dscnn_expanded_20261114' \
  --audio '/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/yamnet/data/raw/private/doorbell/kr_user_doorbell_06.m4a' \
  --backend int8
```

`--backend float`로 float 모델을 선택할 수 있다. 추론 도구는 묶음의 SHA256과 전처리·라벨 규약을 확인하고 합성 모델을 거부한다. 원음과 calibration 음원은 모델 묶음에 포함하지 않는다. 로컬 학습 곡선은 `TRAINING_REPORT.html` 파일로 열며, 휴대전화의 `127.0.0.1` 주소로 접속할 필요는 없다.

## 구간 주석 학습 경로 (별도 보존)

원본 데이터와 YAMNet 임베딩·모델은 읽기 전용으로 취급한다. `sound_baseline.core.load_splits()`의 기존 검사로 group 누수와 고정 test 멤버십을 검증한 뒤, DS-CNN 개발 경로에는 `train`/`validation` 행만 전달한다. 고정 test 행은 특징 추출 함수에서도 거부한다. 원래 개발 음원 1,635개에 새 초인종 5개를 추가했으며, 4개 fold의 고정 test 411개는 동일하다.

`--supervision event_intervals` 경로에서만 개발 타깃 음원 **37개**의 구간 주석이 필요하다. 파일 라벨 경로에서는 이 주석을 사용하지 않는다. 빈 양식은 [event_intervals.template.csv](annotations/event_intervals.template.csv)다. 한 이벤트당 한 행이며, `source_sha256`는 원본 파일의 SHA256, 시간은 원본 녹음 시작 기준 초 단위다. 실제 구간은 채우지 않았다. 양식은 헤더만 포함하며, 자동 후보는 검수된 주석으로 취급하지 않는다.

```csv
sample_id,start_s,end_s,label,source_sha256
```

`other`의 원래 파일 라벨은 유지한다. 타깃 학습 창은 주석 이벤트와의 겹침이 `0.8 × min(이벤트 길이, 1초)` 이상일 때만 사용한다. 짧은 도어벨을 허용하기 위한 개발 규칙이며, 겹치는 주석은 먼저 합친다. 기준은 설정에 기록되어 있다. 주석 밖의 타깃 창을 임의로 `other`로 바꾸지 않는다. 주석 누락·원본 해시 불일치·라벨 변경·녹음 길이 초과는 오류다.

원본 그룹 → 창을 추출하는 기존 `GroupSampler`를 재사용하므로 긴 파일의 창 수 때문에 그룹 확률이 커지지 않는다. 클래스 균등, `other` pool 0.6/0.3/0.1, 국내 가중치 1, seed `42+fold`를 사용한다. 국내 가중치 1은 기존 YAMNet 선택값을 초기 설정으로 옮긴 것이며 DS-CNN에서 최적이라고 검증한 값은 아니다.

학습 코드는 fold마다 새 모델·Adam을 생성하고 최대 30 epoch, patience 8, batch 30, 30 steps/epoch, lr 0.001을 기본값으로 쓴다. 검증은 타깃 구간으로 창을 골라내지 않고 파일의 모든 창 확률 평균 → 원본 그룹 내 파일 확률 평균으로 집계하여 파일·그룹 분모를 보존한다. 기존 개발 가드와 선택 함수를 재사용한다. 타깃 학습 감독이 YAMNet의 weak file label과 달라진 점은 이후 비교에서 밝혀야 한다.

구간 주석 학습 경로는 주석·검수 영수증 선검사를 통과해야 TensorFlow/오디오를 읽는다. 별도의 파일 라벨 학습 경로는 사용자 확인과 데이터 identity를 먼저 검사한다. `refit`은 완료된 4-fold summary를 요구하며, 기존 test 비교는 고정 모델 해시를 요구하는 `dscnn_baseline.evaluation` 모듈로 분리했다.

## W&B 학습 그래프 연결

DS-CNN 전용 환경은 이 폴더의 `.venv`이며, 현재 연결은 설치된 `wandb==0.31.0`을 기준으로 한다. YAMNet 환경에 패키지를 추가할 필요는 없다. Ubuntu PC의 전용 터미널에서 다음과 같이 활성화하고 인자를 확인한다. 명령은 foreground로 끝나며, 이후 작업하는 동안 같은 터미널의 활성화 상태를 유지한다. 각 명령이 오류 없이 종료되어야 한다.

```bash
cd '/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn'
source .venv/bin/activate
python -m dscnn_baseline train --help
```

`train`은 사용자 요청에 따라 온라인 그래프 기록을 기본으로 사용한다. 기본 저장 대상은 사용자가 제공한 [프로젝트](https://forge.coreweave.com/wandb/jhh0415-korea-university/hearing_alert_system_CNN)이며, 사용자 ID `jhh0415`와 프로젝트 소유자(entity) `jhh0415-korea-university`를 구분한다. `selfcheck`는 계속 CPU와 W&B 비활성화를 기본으로 사용한다.

| 옵션 | 동작 |
|---|---|
| `--wandb-mode disabled` | W&B SDK를 불러오지 않고 기존 로컬 기록만 저장 |
| `--wandb-mode offline` | 선택한 지표를 해당 run 디렉터리에 로컬 저장. 웹으로 전송하지 않음 |
| `--wandb-mode online` | 기본값. 선택한 W&B 프로젝트로 지표 전송. 현재 프로세스의 `WANDB_API_KEY` 필요 |
| `--wandb-project` | 프로젝트 이름. 기본값은 `hearing_alert_system_CNN` |
| `--wandb-entity` | 프로젝트 소유 사용자 또는 팀 ID. 기본값은 `jhh0415-korea-university` |

다음은 현재 기본값을 명시적으로 적은 **인자 예시**이며, 독립적인 학습 실행 명령은 아니다. 앞으로 같은 프로젝트에 기록할 때는 생략할 수 있다. 다른 프로젝트를 선택할 때는 해당 프로젝트의 소유 계정/팀과 공개 범위를 확인한 후 인자를 바꾼다.

```text
--wandb-mode online --wandb-project hearing_alert_system_CNN --wandb-entity jhh0415-korea-university
```

온라인 인증 키는 터미널에서 숨김 입력으로 `WANDB_API_KEY` 환경변수에만 전달한다. 코드·설정 파일이나 CLI 인자로 키를 넣지 않는다. 키가 없으면 자동 로그인 프롬프트 없이 데이터 접근/출력 생성 전에 중단한다. 이 환경변수는 재부팅하거나 터미널을 닫으면 사라지며, 작업을 마칠 때 `unset WANDB_API_KEY`로 제거할 수 있다. 가상환경은 `deactivate`로 해제한다. `wandb status`만으로는 서버 인증 성공이나 run 생성을 확인할 수 없다.

run은 기존 구간 주석 선검사·데이터 준비·모델 초기화가 통과한 후 생성한다. 각 epoch의 `history.csv`를 먼저 저장하고, 다음 지표를 `step=epoch, commit=True`로 기록한다. W&B 그래프의 기본 X축도 `epoch`로 지정한다.

- `epoch`, `train_loss`
- `domestic_target_recall`, `overseas_other_fpr`, `domestic_doorlock_fp`
- `validation_guards_passed`, `saved`

W&B config에는 모델명, fold, 실제 seed, epochs, patience, batch size, steps/epoch, learning rate, 국내 가중치만 보낸다. 원본 오디오·주석·모델 파일을 업로드하는 호출은 추가하지 않았다. 자동 코드·Git·콘솔·시스템 메타데이터/지표·설치 패키지 목록 수집은 비활성화한다. W&B 서비스 자체의 run/SDK 메타데이터는 별개이며, `x_disable_*`, `x_save_requirements`는 내부 SDK 설정이므로 W&B 버전 변경 시 다시 확인해야 한다.

W&B 로컬 파일은 `runs/<run-name>/wandb/` 아래에 생성되며 실제 files 경로와 run ID는 `status.json`의 `wandb` 항목에 남긴다. 온라인 모드에서는 URL도 터미널과 `status.json`에 표시한다. 그 URL을 열면 첫 epoch 기록부터 그래프를 확인할 수 있다. 오프라인 run은 웹에 나타나지 않으며 이 코드에서 자동 동기화하지 않는다.

초기화/epoch 기록 오류는 기존 최초 오류 처리로 중단하고 로컬 결과를 보존한다. 종료 시 W&B 동기화는 최대 30초 기다린다. 종료 동기화 실패는 원래 학습 오류를 덮어쓰지 않고 `status.json`의 `wandb.finish_error_type`과 `wandb.finished=false`로 구분한다. `exit_code=0`만으로 W&B 전송 성공까지 판단하면 안 된다.

연결 변경의 CPU/mock 검사 명령은 다음과 같다. 실제 음원·TensorFlow 모델 실행·GPU·온라인 전송을 수행하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m unittest discover -s tests -v
```

구간 주석 경로의 빈 주석 양식과 GPU 선택/검증 조건은 그대로다. 파일 라벨 경로는 사용자 확인을 별도로 검사한다. W&B 연결 확인과 실제 학습 결과는 각 실행의 기록으로 구분한다.

## 로컬 소프트웨어 검증 재현

호스트는 현재 Ubuntu PC, 아래 터미널의 작업 디렉터리는 DS-CNN 폴더다. 별도 유지 프로세스가 없는 foreground 명령이며 각 명령의 종료 코드 0을 확인한다. 기존 공식 가상환경을 읽어 사용하고 패키지 설치나 activation 스크립트 변경은 없다. 합성 검증은 CPU를 명시적으로 선택한다. 출력 폴더가 이미 있으면 오류로 종료하므로 재현 시 새 이름을 지정한다.

```bash
cd '/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn'
task_python='.venv/bin/python'
"$task_python" -m unittest discover -s tests -v
"$task_python" -m dscnn_baseline audit
"$task_python" tools/generate_frontend_tables.py --check
"$task_python" -m dscnn_baseline selfcheck --output .cache/reproduce/selfcheck_01
"$task_python" tools/check_frontend.py \
  --output .cache/reproduce/frontend_01 \
  --model-metadata .cache/reproduce/selfcheck_01/export/model_metadata.json \
  --tflite .cache/reproduce/selfcheck_01/export/pipeline_only_int8.tflite
```

예상 결과는 단위 테스트 `OK`, `FRONTEND_TABLES_MATCH`, `SYNTHETIC_SOFTWARE_CHECK_PASS`, 전처리 보고의 `passed: true`다. 중간 오류는 자동 재시도하지 말고 해당 로그를 확인한다. `selfcheck`는 저장·재로딩과 합성 확률 오차(사전 기준 max abs ≤0.05)를 검사한다. 이것은 실제 음원 분류 성능이나 양자화 품질 수용 기준이 아니다. float/int8 argmax 일치율도 기록하지만 합성 가짜 라벨의 정확도로 해석하지 않는다.

NumPy 1.26.4, 공식 TensorFlow 2.21.0, tf-keras 2.21.0 및 기존 `g++`만 사용했다. `tf.lite.Interpreter`는 현재 가상환경에 포함된 host reference kernel로 사용하며 deprecation 경고를 로그에 보존했다. 새 LiteRT 패키지를 설치하지 않았다. 패키지가 없는 다른 환경에서는 설치 전 사용자의 허가가 필요하다.

## int8·ESP32-S3 연결

int8 변환은 학습 창만으로 calibration하며 validation/test 사용을 거부한다. 합성 검증은 `synthetic` 역할만 허용한다. 입력·출력 int8, 중간 int8/int32, 연산 whitelist를 검사하고 모델·calibration·frontend 해시와 양자화 scale/zero point를 저장한다. 합성 calibration은 실제 마이크의 대표 분포가 아니다.

[C++ 컴포넌트](../../../firmware/transmitter_s3/dscnn/README.md)에 heap 없는 log-Mel 전처리와 TFLM 실행 연결 코드를 두었다. 생성 모델 헤더의 `kModelPipelineOnly`와 `kModelFrontendSha256`를 runner에 넘긴다. 합성 모델은 runner가 거부하며 frontend 규약 해시가 달라도 중단한다. tensor arena 크기는 호출자가 나중에 예산을 정해 제공한다. 이 컴포넌트는 아직 ESP-IDF에서 빌드·보드 검증하지 않았다.

이전 합성 검증 결과는 [한국어 소프트웨어 개발 보고서](reports/software_v1/REPORT_KO.md)에 보관한다. 실제 파일 라벨 학습 결과는 [실제 음원 학습 보고서](reports/real_training_20261114/REPORT_KO.md)로 구분한다. 이벤트 구간별 감지 성능을 평가하려면 시간 구간 주석이 필요하다. 현장 검증에는 독립적인 실제 마이크 녹음과 보드 RAM·처리 시간 예산이 필요하다. 고정 test는 이미 YAMNet 최종 평가에 사용됐으므로 DS-CNN 개발 선택·수치 허용오차 조정에 사용하지 않는다.

공식 참고: [Espressif TFLM 컴포넌트](https://github.com/espressif/esp-tflite-micro), [LiteRT 전 정수형 양자화](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant). SDK/커널 지원 표시는 보드 실측 결과를 대신하지 않는다.
