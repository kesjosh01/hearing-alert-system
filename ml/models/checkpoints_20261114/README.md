# 2026-11-14 학습 체크포인트 보관

로컬에 저장돼 있던 YAMNet 22개, DS-CNN 계열 105개, 총 **127개 `classifier.keras`**를 원본 바이트 그대로 보관한다. 원래 실행 폴더와 체크포인트를 이동하거나 덮어쓰지 않았다. 저장·재로딩 점검용 smoke/integration 실행도 포함되므로 127개를 모두 정식 학습 완료 횟수로 세지 않는다.

[`index.json`](index.json)에 원래 상대 경로, 보관 경로, bytes, SHA-256, 해당 실행의 기록된 상태와 함께 보관한 sidecar 목록이 있다. `status.json`, `run.json`, `history.csv`, 선택 지표 등 존재하는 작은 요약 파일을 함께 보관한다. 원음·특징 tensor·전체 sampling 목록·가상환경·학습 캐시는 포함하지 않는다.

| 찾을 모델 | 보관 위치 |
|---|---|
| 최신 전체 후보의 float 원본 | [`dscnn/pursuit_20261114_r06_native40_supervised_bnrefit/`](dscnn/pursuit_20261114_r06_native40_supervised_bnrefit/classifier.keras) |
| 후속 실제 YAMNet distillation 학생 | [`dscnn/pursuit_20261114_r06_native40_distilled_vs_control/`](dscnn/pursuit_20261114_r06_native40_distilled_vs_control/classifier.keras) |
| 국내 학습 음원 특화 후보의 float 원본 | [`dscnn/pursuit_20261114_domestic_seen_presence/`](dscnn/pursuit_20261114_domestic_seen_presence/classifier.keras) |
| YAMNet 기준 최종 refit head | [`yamnet/ubuntu_gpu_refit_w1_s42/`](yamnet/ubuntu_gpu_refit_w1_s42/classifier.keras) |

`.keras`의 존재나 실행 이름만으로 보드 배포 가능 여부를 판단하지 않는다. YAMNet head에는 별도 backbone이 필요하고, 일부 후속 모델은 custom layer·loss 구현을 요구한다. 펌웨어 연결에는 [int8 후보 묶음](../pursuit_20261114/README.md)의 고정된 모델·전처리·출력 계약을 함께 사용한다. 최신 후보의 float 원본과 int8은 최종 출력 온도 변환이 다르므로 확률 임계값을 공유하지 않는다.
