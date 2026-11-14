# 후속 모델 연구의 int8 후보

최신 전체 후보는 [`candidates/native40_r06_v1/`](candidates/native40_r06_v1/README.md)다. **88,280-byte int8**, 16 kHz PCM16 → 기존 98×40 portable log-Mel 입력이며 `native40_r06_supervised_bnrefit_v1`와 같은 모델·정책이다. 실제 마이크·ESP32-S3·물리 알림 검증은 남아 있고 `deployment_ready=false`다.

처음 연결할 때는 [모델·통신 연결 안내](../../../docs/ml/MODEL_HANDOFF.md)를 따른다. `.h`와 `.tflite`만 가져가지 말고 `model_metadata.json`과 `delivered_output_contract.json`을 함께 확인한다. 서로 다른 모델의 확률·logits·정수 임계값을 섞지 않는다.

| 후보 | 역할 | 입력 | 출력 |
|---|---|---|---|
| `native40_r06_v1` | 공개 시험 전 선택한 최신 전체 후보 | 98×40 | probability, T=4 |
| `native40_r06_supervised_v1` | 비증류 비교 | 98×40 | probability |
| `native40_r06_supervised_bnrefit_v1` | 비증류·BN 통계 보정, 최신 후보와 동일 | 98×40 | probability |
| `native40_r06_distilled_v1` | 실제 KL distillation 학생 | 98×40 | raw logits |
| `native40_r06_distilled_bnrefit_v1` | 증류·BN 통계 보정 비교 | 98×40 | raw logits |
| `domestic_known_v1` | 학습 국내 음원 특화 비교 | 98×40 | probability |
| `domestic_known_aug_v2` | 국내 음원·증강 비교 | 98×40 | probability |
| `domestic_known_enriched_v3` | 국내 음원·배경 확장 비교 | 98×40 | probability, T=4 |
| `general_prefix64_v1` | YAMNet 앞부분 가중치 전이 비교 | **96×64**, 기존 runner와 불일치 | probability |

각 후보 폴더의 원래 파일과 `candidate.json`을 그대로 보관했다. [`release_manifest.json`](release_manifest.json)은 이 묶음 전체의 파일 해시다. float 원본은 [체크포인트 색인](../checkpoints_20261114/index.json)의 `source_model`과 대응하는 실행 이름에서 찾는다.

최신 고정 공개 시험은 서로 다른 화재경보 41개·초인종 6개·기타 512개 파일이며 최신 후보의 감지는 8/41·1/6·오감지 0/512였다. 이 결과를 보고 모델이나 임계값을 다시 고르지 않았다. 기존 국내 특화 후보는 학습 원음 43/43 감지를 기록했지만 새 공개 자료로 일반화하지 못했다. 두 수치는 다른 자료의 결과다. [진행 상황](../../../docs/ml/DEVELOPMENT_STATUS_2026-11-14.md)에 비교·한계·원본 결과를 연결했다.

원본 음원의 출처별 이용 조건과 사전학습 출처는 학습 기록을 따른다. 모델 산출물의 공개가 원음 전체의 재배포나 상용 이용 조건까지 정리됐다는 뜻은 아니다. 이 묶음에는 원음과 대표 양자화 음원을 넣지 않았다.
