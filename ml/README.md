# 머신러닝

현재 모델·후속 학습 결과는 [2026-11-14 개발 현황](../docs/ml/DEVELOPMENT_STATUS_2026-11-14.md), 파일 위치와 통신 연결은 [모델 인계 안내](../docs/ml/MODEL_HANDOFF.md)를 확인합니다. 최신 int8 후보는 [native40_r06_v1](models/pursuit_20261114/README.md)이며 기존 98×40 전처리와 맞습니다. [체크포인트 127개와 색인](models/checkpoints_20261114/README.md)을 함께 보관합니다. 아래 확장·개선 수치는 이전 단계의 기록입니다.


- `datasets/`: 음향 데이터 및 출처·수집 조건 안내를 관리할 예정입니다.
- `preprocessing/`: 오디오 전처리 코드를 관리할 예정입니다.
- [training/](training/README.md): YAMNet 기준 모델과 ESP32-S3 대상 DS-CNN의 학습·개발 임계값 선택·평가·int8 변환 코드, 설정과 실험 결과를 관리합니다.
- `evaluation/`: 분류 성능 및 경량화 전후 평가를 관리할 예정입니다.
- `export/`: 모델 변환·양자화 및 배포 형식 변환 코드를 관리할 예정입니다.
- [models/](models/README.md): 실제 학습한 DS-CNN float·int8 모델과 판정 정책·출처·해시를 보관합니다.

이전 [공개 정답 확장 학습](training/dscnn/reports/expansion_20261114/REPORT_KO.md)은 공개 음원39개 추가, 학습용 사이렌24개를 화재경보로 변경, 국내 정답 그룹4배 가중치, YAMNet head 재학습과 시간축9×3 CNN 학습을 완료했습니다. 개발 recall76.88%의 [새 실험 모델](models/dscnn_expanded_20261114/README.md)을 보존합니다. 기존 test에서는 평균 recall62.5% 유지·해외 비목표 오탐11.68%로 성능 개선을 입증하지 못해 실사용 교체를 권장하지 않습니다. 앱·프로그램·패키지·새 환경 다운로드/설치는 없습니다.

YAMNet 기준 모델의 학습·기존 test 평가와 [DS-CNN](training/dscnn/README.md)의 초기 baseline 학습을 완료했습니다. 후속 요청으로 YAMNet distillation 후보 4개의 16개 fold 비교·9 epoch 최종 학습·int8 변환도 마쳤습니다. [개선 결과](training/dscnn/reports/improvement_20261114/REPORT_KO.md)는 재현율 상승과 오탐 증가를 함께 기록하며 실사용 배포를 보류합니다. 목표는 ESP32-S3에서 경량 모델을 추론하는 것이며, PC 소프트웨어 결과와 보드 검증을 구분합니다. baseline 재현에 필요한 음원과 기존 체크포인트는 [Ubuntu 안내](training/yamnet/UBUNTU.md)의 데이터 복원 절차를 따릅니다.
