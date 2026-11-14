# YAMNet 음향 분류 baseline

화재경보, 초인종, 기타 소리를 분류합니다. FFmpeg가 입력을 16 kHz mono PCM으로 변환하고, 고정 YAMNet이 0.96초 패치에서 1,024차원 특징을 뽑습니다. 이 특징으로 세 클래스 분류기를 학습합니다.

- [Ubuntu에서 이어서 작업하기](UBUNTU.md)
- [학습 설정과 명령 설명](TRAINING_GUIDE.md)
- [수집·라벨·평가 설계](DATASET_PLAN.md)
- [실험 결과와 남은 실험](RESULTS.md)

`sound_baseline/`에 전처리, 특징 추출, 그룹 균형 샘플링, 학습, 평가, 단일 음원 추론 CLI가 있습니다. `config/`는 baseline과 고정 분할 프로토콜, `tools/`는 분할 생성과 공개 출처 메타데이터를 다룹니다.

개인 비공개 데이터 저장소에는 수집 음원, 전체 메타데이터와 고정 분할, 학습 체크포인트 및 상세 실행 기록을 보관합니다. Ubuntu 안내에 따라 프로젝트에 복원할 수 있습니다.
