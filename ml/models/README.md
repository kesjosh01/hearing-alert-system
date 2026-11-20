# 학습한 경량 모델

## 최신 후보와 체크포인트

- [2026-11-19 국내 목표 음원 과적합 체크포인트](domestic_overfit_20261119/README.md): 실제 학습한 native40 DS-CNN `.keras`, 359,129 bytes. 학습 파일 분류 43/43, 기존 T=4 정책의 float 모사 감지 41/43. 도어락 부가 진단 오알림 5/7로 일반화 한계가 있으며 새 int8·실보드 검증은 없습니다.
- [2026-11-20 마이크·모델·통신 연결 점검](../../docs/ml/IO_INTEGRATION_REVIEW_2026-11-20.md): 현재 마이크 DTW 데모와 제품 통신 앱의 분리, DS-CNN 입력 조건, 이벤트 ID→C6 알림 경로 및 남은 통합 작업.
- [후속 int8 후보 묶음](pursuit_20261114/README.md): 최신 전체 후보 `native40_r06_v1`, 실제 int8 88,280 bytes, 기존 98×40 frontend 입력. 국내 특화·증류·비증류·64-bin 비교를 함께 보존합니다.
- [전체 저장 체크포인트](checkpoints_20261114/README.md): YAMNet 22개·DS-CNN 계열 105개, 총 127개 `.keras`와 해시·요약 기록. smoke/integration 실행을 별도 표시합니다.
- [통신 담당자를 위한 연결 안내](../../docs/ml/MODEL_HANDOFF.md): 모델 선택·입출력·정수 임계값·2/3 시간 정책·고정 UDP 이벤트 ID 매핑.
- [현재 진행 상황과 평가 한계](../../docs/ml/DEVELOPMENT_STATUS_2026-11-14.md): 학습·보정·고정 평가·실보드 미검증 범위.

최신 후보를 실제 제품 성능 달성 모델로 표시하지 않습니다. 기존 모델의 float/int8 파일·정책·원래 manifest는 보존했습니다. 아래는 이전 단계의 묶음이며 각 README의 수치는 해당 단계의 자료·평가 정의를 따릅니다.

## 이전 실험 모델


[dscnn_expanded_20261114/](dscnn_expanded_20261114/README.md)는 공개 정답 음원39개와 화재경보로 변경한 학습용 해외 사이렌24개를 포함한 **확장 학습 실험 모델**입니다. 국내 정답 그룹의 추출 가중치는 해외의4배입니다. 새 YAMNet head로 distillation하고 시간축9×3 구조를 선택했으며, 실제 int8은16,896 bytes입니다. 개발 평균 recall76.88%, 최종5epoch 학습을 완료했습니다. 기존 test 평균 recall은62.5%로 유지됐고 해외 비목표 오탐은46/394(11.68%)로 증가해 실사용 교체를 권장하지 않습니다. [보고서](../training/dscnn/reports/expansion_20261114/REPORT_KO.md)와 [출처](dscnn_expanded_20261114/PUBLIC_DATA_CREDITS.md)를 확인합니다. 앱·프로그램·패키지를 다운로드하거나 설치하지 않았으며 기존 묶음은 보존했습니다.

[dscnn_improved_20261114/](dscnn_improved_20261114/README.md)는 YAMNet distillation·주파수 정보 보존·가벼운 증강을 적용한 **개선 실험 모델**입니다. 4-fold 평균 타깃 recall은 73.13%, int8 모델은 15,304 bytes입니다. 기존 test의 타깃 recall은 62.5%로 높아졌지만 해외 비목표 소리 오탐은 **24/394(6.09%)**로 늘었으므로 실사용 교체·배포를 권장하지 않습니다. 사용자 요청으로 `siren` 8개 전체를 제외한 [최신 평가 수정](../training/dscnn/reports/improvement_20261114/evaluation_scope_v2/REPORT_KO.md)입니다. 모델 묶음 안의 원래 정의 27/402(6.72%)와 SHA-256은 보존했습니다. [개선 보고서](../training/dscnn/reports/improvement_20261114/REPORT_KO.md)를 함께 확인하세요.

[dscnn_filelabels_20261114/](dscnn_filelabels_20261114/README.md)에 실제 파일 라벨로 학습한 float 모델, 전체 int8 모델(13,656 bytes), C++ 헤더, 전처리 규약, 판정 정책과 SHA256 manifest를 보관합니다. 합성 경로 검증 모델과 구분하며 `pipeline_only=false`입니다.

[초기 baseline 보고서](../training/dscnn/reports/real_training_20261114/REPORT_KO.md)와 [저장 음원 추론 안내](../training/dscnn/README.md)를 함께 확인합니다. 초기 baseline의 기존 test 화재경보·초인종 recall은 각각 50%였습니다. 실제 마이크·ESP32-S3 RAM/지연·통합 알림은 두 모델 모두 검증하지 않았고 `deployment_ready=false`입니다. 원음과 calibration 음원은 모델 묶음에 포함하지 않습니다.
