# 개선 실험 DS-CNN

선택 후보는 `kd24_freq_aug`이며 최종 모델의 distillation 적용 여부는 `True`입니다. refit은 사전 고정 9 epoch입니다. 상세 방법·수치·한계는 `training_provenance.json`과 [개선 보고서](../../training/dscnn/reports/improvement_20261114/REPORT_KO.md)를 확인하세요.

기존 test에서는 재현율과 함께 오탐도 증가했습니다. 개발 가드는 통과했지만 회고적 test의 해외 other 오탐률은 5%를 넘었습니다. 실사용 교체·배포를 권장하지 않으며 연구·검토용 모델로 보존합니다.

`classifier.keras`는 float 모델이며 `dscnn_int8.tflite`와 `.h`는 전체 int8 모델입니다. 전처리는 16 kHz mono PCM16 → 1초 `[98,40,1]` log-Mel입니다. 파일의 모든 창 확률을 평균하고 source group은 파일 확률을 같은 비중으로 평균합니다. 이후 `decision_policy.json`의 클래스별 타깃 임계값을 적용합니다. 이 정책은 보드의 단일 창·시간 누적 정책을 검증한 결과가 아닙니다.

Ubuntu PC의 `/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn`에서 기존 `.venv/bin/python -B tools/predict_audio.py --help`로 저장 음원 추론 옵션을 확인합니다. 이 묶음의 SHA256은 `release_manifest.json`에 있습니다. 원음과 calibration 음원은 포함하지 않습니다.

`pipeline_only=false`인 실제 학습 모델입니다. 기존 test는 과거 관측한 자료여서 독립 시험이 아닙니다. 현장 마이크·시간당 오탐·ESP-IDF 전체 빌드·실보드 RAM/지연·통합 알림은 미검증이며 `deployment_ready=false`입니다.
