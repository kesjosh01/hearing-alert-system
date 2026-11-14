# 공개 정답 데이터 확장 DS-CNN

공개 도어벨·경보 음원 39개를 추가하고 학습용 ESC-50 사이렌 24개를 사용자 기준의 `fire_alarm`로 학습했다. 국내 정답 source group 가중치는 해외의 4배이며, 세 클래스의 학습 추출 확률은 각 1/3이다. 새 데이터 라벨과 일치하도록 YAMNet 분류 head를 다시 학습한 뒤 teacher distillation을 수행했다.

기존 test의 int8 평균 타깃 recall은 이전과 같은62.5%이며, 해외 비목표 오탐은24/394(6.09%)에서46/394(11.68%)로 증가했다. 개발 점수의 소폭 상승이 이 자료의 전반적 성능 개선으로 이어지지 않았다. 실사용 모델 교체를 권장하지 않으며 연구·검토용으로 보존한다.

선택 구조는 `temporal24`, 최종 학습은 사전에 고정한 5 epoch다. `temporal24`는 시간축 depthwise kernel을 9×3으로 확장하고, `freq24`는 이전 3×3 구조를 유지한다. 선택은 기존 국내 4-fold와 해외 validation으로만 수행했다. 새 공개 음원은 학습 전용이다.

초기 refit의 무작위 추출에서 새 파일1개가0회인 것을 확인해, test 평가 전에 학습 타깃100개의 포함 여부만으로 최종 시드43을 고정했다. 구조·5epoch·150 update·그룹 가중치·임계값은 유지했고 초기 refit도 보존했다. 성능을 비교해 시드를 선택하지 않았다. 상세 변경은 `training_provenance.json`에 기록한다.

`classifier.keras`는 실제 학습 float 모델이다. `dscnn_int8.tflite`와 `dscnn_int8.h`는 같은 모델의 전체 int8 변환이다. `decision_policy.json`의 클래스별 타깃 임계값은 파일의 창 평균 확률과 source group 평균 확률에 적용한다. 입력은 기존과 동일한 16 kHz mono PCM16, 1초 `[98,40,1]` log-Mel이다. 원음·시간 구간 정답·보드 시간 누적 정책을 포함하지 않는다.

[실험 보고서](../../training/dscnn/reports/expansion_20261114/REPORT_KO.md), [학습 곡선](../../training/dscnn/reports/expansion_20261114/TRAINING_REPORT.html), `training_provenance.json`, [추가 자료 출처](PUBLIC_DATA_CREDITS.md)를 확인한다. 기존 test는 과거에 관측한 자료로 독립 시험이 아니다. 해외 사이렌 8개를 비목표 오탐 모집단에서 제외한 동일한 기준으로 이전 모델과 비교했다.

기존 환경의 `ml/training/dscnn/tools/predict_audio.py`로 저장 음원을 추론할 수 있다. 묶음의 파일 해시는 `release_manifest.json`에 있다. `pipeline_only=false`이며 보드·실시간 마이크·현장 시간당 오탐·실보드 메모리/지연은 미검증이다. `deployment_ready=false`다.
