# 머신러닝

- `datasets/`: 음향 데이터 및 출처·수집 조건 안내를 관리할 예정입니다.
- `preprocessing/`: 오디오 전처리 코드를 관리할 예정입니다.
- [training/](training/README.md): YAMNet baseline 학습 코드, 설정과 실험 결과를 관리합니다.
- `evaluation/`: 분류 성능 및 경량화 전후 평가를 관리할 예정입니다.
- `export/`: 모델 변환·양자화 및 배포 형식 변환 코드를 관리할 예정입니다.
- `models/`: 최종 배포용 경량 모델과 관련 설명을 관리할 예정입니다.

YAMNet baseline은 PC에서 개발하며, 향후 ESP32-S3에서 경량 모델 추론을 목표로 합니다. baseline 재현에 필요한 음원과 기존 체크포인트는 [Ubuntu 안내](training/yamnet/UBUNTU.md)의 데이터 복원 절차를 따릅니다.
