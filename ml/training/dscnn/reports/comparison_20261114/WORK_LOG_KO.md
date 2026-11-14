# 현재 비교 실험 작업 기록

사용자 목표: 제안 1~8의 순차 비교, 한 번에 한 설정만 변경, 기존 test는 회고적 비교로 유지, 새 발생원 시험 자료 확보, 실제 모델 학습·평가. 앱/패키지/환경 설치 금지. GPU 학습은 현재 goal에서 명시 승인. 로컬 기록만 사용.

기존 코드는 수정하지 않고 `comparison/`, `tests_comparison/`, `data/comparison_20261114/`, `reports/comparison_20261114/`, 새 `runs/dscnn_comparison_20261114_*`만 생성했다. YAMNet 9344파일/3환경 및 기존 DS-CNN 1581파일/환경77배포판을 before snapshot으로 보호했다. 최종 after 대조가 필요하다.

GPU: 5070 Ti Laptop, UUID GPU-10f7f4b0-eee7-9d43-26de-7bbefdbfc7ce. 샌드박스에서는 nvidia-smi 실패하지만 require_escalated 실제 호스트 조회 정상/다른 compute 없음. `comparison.runner`가 기존 .venv CUDA 라이브러리 12개를 자식 프로세스에만 지정하고 GPU 소유권 확인. 설치/영구 환경변경 없음.

기존 확장 자료를 source-disjoint train/selection/calibration으로 재분할. train fire36/doorbell44/other1179, selection fire9/doorbell5/other193(외국negative192+도어락1), calibration fire10/doorbell4/other199(외국negative198+도어락1). 과거 모델에서 학습됐던 양성도 새 비교 모델과 새 teacher에서는 selection/calibration 역할이면 학습하지 않음. 각 모델 실제 checkpoint를 selection 자료에서 고르고 최종 float/int8 threshold는 calibration에서 따로 보정. Calibration 결과로 모델/설정 순위 선택 금지. 새 test는 아직 모델 추론 0회.

6개 CPU 테스트 및 teacher/student gradient/top-k/full-int8 합성 GPU preflight 통과. teacher_base 실제 학습 완료, 16epoch 중8선택, selection target recall91.67%.

초기 순서1~3 coordinator는 `python3 -B -m comparison.study early`로 실행 중. tool session 73081이며 report early_study.log, 각 run status.json 및 process receipts로 확인. baseline checkpoint5: selection79.17%,FPR3/192. 별도 calibration은 보정 전58.33%→70.83%, FPR8/198 유지(float/int8); 독립 시험 결과가 아님. coverage 후보 checkpoint12는 selection79.17%,FPR1/192로 채택. hardneg2배는 비채택. top50은 selection79.17%,FPR0/192로 채택. top25 실행중(상태를 다시 조회). 초기 coordinator는 1~3만 하며 전체 goal 완료가 아님. 이후4~8 late coordinator 구현/실행 필요.

주의: run/sample_exposure.csv와 status.negative_unique_sampled는 전체 실행 epoch의 노출(조기종료 전까지)이므로 선택 checkpoint 노출과 구분해야 한다. `comparison.exposure.audit(run)`는 deterministic sampler replay가 실제 전체 누적 로그와 정확히 일치함을 검증하고 선택 epoch까지의 노출을 따로 저장한다. baseline 선택시 other650/1179, coverage 선택시1179/1179. 이후 모든 후보와 최종 모델도 audit 필요.

새 공개 음원: Commons API+명시적 제작자 설명 검토, 원음 CDN이429를 반환해 반복하지 않고 중단. 성공10파일 보존, 실패 기록 보존. SoundBible CC BY3.0 및 BigSoundBank CC0 대체 수집으로 총117개 성공. source_freeze.json 기준 train fire4/doorbell3/other31=38파일, fresh_test fire4(4그룹)/doorbell8(3그룹)/other67(10보수적그룹)=79파일. 새 test 실제기기국가/물리발생원완전독립/교사foundation 사전학습 미노출은 확인 불가; 신규 publisher recording ID·원본해시·decoded PCM·관련계열 역할 분리까지 확인. 구간 사람검수 아님. domestic 실녹음 추가 없음(사용자 확인). Downloads5개는 이미 kr-user-db06~10으로 학습한 원본과 SHA동일; 새test로 사용하지 않음. 원본 보존. PUBLIC_DATA_CREDITS.md 제공. 신규자료 추가전에 단계4 dataset 한 요인으로 비교하며 새 teacher head도 해당 train만 재학습.

신규 train38파일 CPU feature 준비를 `comparison.prepare features`로 시작했다. GPU aligned YAMNet embeddings(`comparison.prepare embeddings`)는 early coordinator 완료 후 GPU 비었을 때 실행. 이후 teacher_new(BASE+added_training_data=True) 실제 학습 필요.

후속 구현: late study는 early_champion으로부터 added_training_data, domestic_weight2, other_mass.5, kd_weight.25/0, freq24 vs temporal24 seeds42/43/44, plateau LR를 순차/한 요인 비교. Teacher는 데이터별(BASE로 학습한 train-only head) 고정, 학생 weight/KD/pooling 변화 때 teacher를 바꾸지 않음. 구조 비교는 두 구조 각각 같은3seed를 paired로 비교하고 seed42 모델을 final로 전달(최고 seed 사후선택 금지). 마지막 candidate를 먼저 고정한 뒤 calibration 및 int8 export.

순서8은 파일 replay로 single/vote/hysteresis/cooldown을 한 요인씩 비교해야 한다. 정확한 event onset 정답이 없으므로 파일시작 기준/동일파일 single 대비 추가 지연과 미탐/비목표 알림횟수·시간당오탐을 구분하여 기록; 실제 음향 onset 지연 또는 마이크/보드 검증으로 과장 금지. 최종 새 test는 모델/clip/event policies와 해시를 확정한 다음 1회 평가. 이전3모델+이번base+최종모델을 같은 새 test와 기존회고적test에서 비교하고 결과로 재튜닝하지 않음. 모델 묶음, CLI, 학습/비교 그래프, 보고서, requirement별실증 audit, 보호 before/after, 프로세스종료 확인까지 필요. 전체 요구 충족 전 goal complete 금지.


최종 완료: 학생 실제 학습16회·교사 head2회, 요청한 8단계 비교, float/INT8 고정 평가, 파일/causal replay CLI 검증 완료. 새 공개 시험 INT8 목표 발생원 재현율54.17%, FP7/67(10.45%): 직전 확장 모델보다 개선됐으나 5% 목표 미충족. 최초 모델보다 모든 지표가 좋아진 것은 아님. 시험 결과로 재선정하지 않음. 연구용 묶음·보고서·오류 출처·5개 그래프·최종 감사 저장. 보존/provenance23항목 및 실제 호스트 프로세스 종료 확인 통과. 설치0, 마이크/보드0, commit/push0. completion.json 참조.
