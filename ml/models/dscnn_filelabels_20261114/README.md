# 실제 음원으로 학습한 DS-CNN

이 묶음은 실제 파일 라벨 학습·개발 임계값 선택·고정 epoch refit과 int8 변환 결과다. `pipeline_only=false`이며 합성 소프트웨어 검증 모델과 구분한다.

`classifier.keras`는 float 모델, `dscnn_int8.tflite`와 `.h`는 전체 int8 모델이다. 입력은 16 kHz mono PCM16의 1초 창에서 만든 `[1,98,40,1]` log-Mel이다. 전처리 규약과 클래스 순서는 `frontend.json`, `labels.json`을 따른다. 파일 판정은 `decision_policy.json`의 창 평균·그룹 평균·고정 임계값을 따른다.

PC의 기존 DS-CNN 환경에서 `tools/predict_audio.py --help`로 저장된 음원 추론 인자를 확인한다. 실제 마이크·보드 RAM·지연·시간당 오탐·통합 알림은 미검증이며 `deployment_ready=false`다. 기존 test는 YAMNet에서 이미 사용한 자료로 새로운 독립 시험이 아니다. 상세 수치와 해시는 `training_provenance.json`, `release_manifest.json`에 있다. 원본 음원과 calibration 음원은 이 묶음에 포함하지 않는다.
