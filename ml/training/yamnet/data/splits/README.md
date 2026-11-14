# Split manifest

생성된 split 목록을 이 폴더에 저장합니다. 각 행은 `sample_id,split,split_group_id,split_version`을 포함하고, `split_group_id` 기준으로만 train/validation/test를 배정하세요. split 생성을 위해 사용한 스크립트, seed, 생성일, 그룹 수/클래스별 수를 함께 기록합니다.

같은 `source_recording_id`, `household_id`의 개인화 평가 데이터, 같은 발생원/기기의 파생 crop은 지정한 평가 목적에 맞춰 그룹 전체가 하나의 split을 공유해야 합니다. 이미 공개한 test에 맞춰 튜닝한 경우 기존 test를 더 이상 독립 시험으로 보고하지 마세요.

## 현재 프로토콜: ko-source-cv4-20261108-v1

설정은 [`../../config/training_protocol_v1.json`](../../config/training_protocol_v1.json), 생성기는 [`../../tools/prepare_training_protocol.mjs`](../../tools/prepare_training_protocol.mjs)입니다. 프로젝트 루트에서 `node tools/prepare_training_protocol.mjs`로 재생성합니다. seed는 42이며 test 선택은 클래스와 출처/길이 층을 이용한 결정적 SHA-256 순위입니다. 학습 결과나 음원 청취 결과로 쉬운 test를 고르지 않습니다. 길이는 기존 metadata의 근사값이며 정밀 측정값으로 주장하지 않습니다.

### 분할 목록 사용

- `ko-source-cv4-20261108-v1/fold_0.csv`부터 `fold_3.csv`: 개발 실험의 전체 파일 목록. `split`을 필터링합니다. 매 fold마다 모델·optimizer·early-stopping 상태를 새로 초기화합니다.
- 국내 test: 초인종 4개, 화재경보 4개, 도어락 1개로 고정합니다. 개발 원본은 각각 16/13/5개입니다.
- 각 fold에서 국내 초인종 train 12 / validation 4, 국내 화재경보 train 9~10 / validation 3~4, 국내 도어락 train 3~4 / validation 1~2입니다.
- ESC-50은 모든 개발 실행에서 train 1,200 / validation 398 / test 402로 고정합니다. 국내 교차검증만 회전합니다. 국내 개발 원본은 정확히 한 번씩 검증에 나타나고, ESC-50 validation은 매 실행 재사용합니다. 국내 pooled OOF와 해외 validation 결과를 별도로 보고합니다.
- `test.csv`: 모든 fold에서 동일한 411개. 국내 test 9개 + ESC-50 402개입니다. 해외/효과음 타깃 3개는 train에만 들어갑니다. 전체 accuracy 하나로 국내 성능을 평가하지 않습니다.
- `holdout_lock.json`: test와 국내 CV 배정 및 입력 목록 식별자를 고정합니다. 같은 버전에서 데이터 구성이나 설정을 바꿔 조용히 test를 다시 뽑을 수 없게 검사합니다.
- `refit.csv`: 설정 확정 후 국내 개발 원본 34개 전체와 ESC-50 train 1,200개, 보조 타깃 3개를 학습합니다. Train 1,237 / 해외 validation 398 / test 411입니다. 국내 validation 행은 남아 있지 않습니다.
- 최종 재학습 epoch 수는 개발 4-fold에서 선택된 epoch의 중앙값 등을 미리 정한 규칙으로 결정합니다. 임계값·이벤트 후처리는 국내 개발 OOF와 해외 validation에서 정합니다. 재학습 후 점수 분포가 달라질 수 있지만 test로 보정하지 않습니다. 별도 현장 validation이 확보되면 그 자료로 보정합니다.

### 샘플링 확률 사용

`sampling_groups_fold_*.csv`와 `sampling_groups_refit.csv`는 **원본 그룹별 전체 batch 추출 확률**입니다. `group_probability`의 합은 1이며 train 원본만 포함됩니다.

1. 세 클래스의 목표 추출 비율은 각각 1/3입니다.
2. 타깃 클래스 안에서는 국내 원본 하나가 보조 원본 하나보다 4배 높은 확률을 갖도록 정규화합니다.
3. `other` 안에서는 해외 일반 60%, 해외 혼동음 30%, 국내 도어락 10%입니다. 해외 혼동음은 clock alarm, siren, wood knock, wood creaks, glass breaking입니다. 이 비율도 실험 시작안입니다.
4. 그룹 선택 후 그 그룹의 유효 입력 창 중 하나를 균등 선택합니다. 모든 창을 펼친 sampler라면 각 창에 `group_probability / 그룹의 유효 창 수`를 부여합니다. CSV 확률을 각 창에 그대로 복사하면 긴 원본이 과대표집됩니다.
5. 선택된 입력 창에만 Train 증강을 적용합니다. 혼합 배경도 같은 실행의 Train 소스에서 가져오고, 합성 결과는 평가에 넣지 않습니다.
6. 처음에는 표준 cross-entropy를 사용하고 추가 class/domain loss weight는 적용하지 않습니다. 국내 가중치 1/2/4배를 비교할 때는 test를 고정하고 sampler의 타깃 클래스 내부 확률만 다시 계산합니다. 고정 split 설정을 바꾸지 말고 run별 설정으로 기록합니다.

Batch 30개라면 평균적으로 경보 10개·초인종 10개·other 10개이며, other 중 해외 일반 6개·해외 혼동음 3개·국내 도어락 1개가 됩니다. 독립 확률 추출이므로 실제 batch별 개수가 항상 같다는 뜻은 아닙니다. 예시 epoch는 30 batch, 즉 900회 창 추출로 정의했습니다. 클래스 균형·대체 추출을 사용하므로 저장된 crop 전체를 한 번 순회하는 의미의 epoch와 구분합니다. 이 값은 학습된 최적값이 아닙니다.

### 검증과 한계

국내 타깃 recall 두 값의 평균을 주 선택 지표로 보고, 타깃 precision, macro-F1, 해외 other의 클립 단위 오탐률, 국내 도어락 오탐 개수를 함께 확인합니다. 실제 요구되는 오경보/미검출 기준은 실험 전에 정해야 하며 아직 성능으로 입증된 수치가 없습니다. 국내 4-fold 개별 결과와 평균·표준편차, 원본별 OOF 결과를 모두 남깁니다. 긴 클립이 지표를 지배하지 않도록 원본별 집계 규칙도 고정해야 합니다. 원본에 여러 경보 이벤트가 있더라도 원본 수와 독립 이벤트 수를 혼동하지 않습니다.

현재 test는 타깃 클래스당 원본 4개여서 하나의 오분류로 원본 단위 recall이 25%p 변합니다. 가정·기기 ID와 이벤트 시간 주석은 새로 확인하지 않았습니다. 이 분할이나 oversampling만으로 데이터 다양성 부족이 해결되었다고 보지 않습니다. 모델을 학습·평가하기 전의 데이터 설계이며 양자화·실제 기기 성능을 보장하지 않습니다.

생성기는 파일 존재, 라벨 번호, 알려진 중복 해시, 원본 그룹 분리, test 개수와 train sampler 확률을 검사합니다. 원본 파일과 권리·동의 상태는 변경하지 않습니다. 수정 전 manifest/ESC-50 metadata 백업은 `../raw/private/protocol_backup_20261108_v1/`에 로컬로 보관합니다.
