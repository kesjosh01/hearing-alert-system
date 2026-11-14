# 모델 학습

최신 진행 상황은 [개발·학습 현황](../../docs/ml/DEVELOPMENT_STATUS_2026-11-14.md)에 정리했습니다. 후속 연구 `pursuit_20261114`의 실제 optimizer 학습은 누적 33회이며 실제 distillation·비증류 비교·BN 통계 보정·고정 공개 평가를 완료했습니다. [최신 int8 후보](../models/pursuit_20261114/README.md), [저장 체크포인트 색인](../models/checkpoints_20261114/README.md), [통신 연결 안내](../../docs/ml/MODEL_HANDOFF.md)를 사용합니다. 아래는 이전 baseline·확장 단계의 기록입니다.


[YAMNet baseline](yamnet/README.md)에 전처리, 분류기 학습, 평가 및 추론 코드가 있습니다. 새 환경에서 실행하려면 [Ubuntu 안내](yamnet/UBUNTU.md)를 참고하세요.

이전 [공개 정답 데이터 확장 실험](dscnn/reports/expansion_20261114/REPORT_KO.md)은 공개 음원39개·학습용 사이렌24개의 새 정답 정의·국내 그룹4배 가중치를 적용했습니다. YAMNet head를 다시 학습하고 두 CNN 구조를 4-fold로 비교해 시간축9×3 `temporal24`를 선택했습니다. 개발 recall76.88%, 최종5epoch·int8 변환을 완료했으나 기존 test에서는 recall62.5% 유지·해외 오탐11.68%로 교체를 권장하지 않습니다. [실제 모델](../models/dscnn_expanded_20261114/README.md)과 [곡선](dscnn/reports/expansion_20261114/TRAINING_REPORT.html)을 제공합니다. 이 작업에서 새 앱·프로그램·환경은 다운로드하거나 설치하지 않았습니다.

[ESP32-S3용 DS-CNN](dscnn/README.md)은 1초 log-Mel 경량 모델, 파일 라벨 기반 학습, YAMNet distillation, 별도 구간 주석 경로, 개발 임계값 선택, 4-fold 비교·refit·int8 변환과 portable C++ 전처리를 제공합니다. 2026-11-14에 초기 baseline 12개 fold·19 epoch refit과 추가 distillation 16개 fold·9 epoch refit을 완료했습니다. [이전 개선 결과](dscnn/reports/improvement_20261114/REPORT_KO.md)와 [곡선](dscnn/reports/improvement_20261114/TRAINING_REPORT.html)을 확인합니다. 기존 test에서는 재현율과 함께 오탐도 증가했으므로 실사용 배포는 보류하며 보드·독립 마이크 검증은 별도입니다. [초기 baseline 기록](dscnn/reports/real_training_20261114/REPORT_KO.md)도 보존합니다.
