# 학습된 비교 모델 묶음

**새 공개 시험 오탐률은 7/67(10.45%)로 5% 목표를 충족하지 못했다. 직전 확장 모델보다 개선됐지만 모든 과거 모델보다 우수한 것은 아니며, 이 묶음은 연구용 비교 후보이고 실제 기기 교체를 권고하지 않는다.**

모델: `dscnn_comparison_20261114_s7_freq24_seed42`. 실제 학습한 checkpoint와 별도 source-held-out 보정셋의 float/int8 판정 정책을 담았다. 기존 모델을 덮어쓰지 않았다.

- 구조: freq24, 선택 epoch: 19
- 증류 손실 비중: 0.0. 0이면 비교 결과로 정답 라벨 학습을 선택한 것이며, 최종 모델을 증류 모델이라고 부르지 않는다.
- 파일 집계: `pooling_fraction=0.5`, 1-p_other가 높은 창을 선택한 뒤 세 확률을 같은 창에서 평균한다.
- 일반 clip 임계값과 streaming/replay 임계값은 별도다. 기존 평균 집계 전용 CLI 대신 아래 CLI를 사용한다.
- 학습/선택/보정/새 시험은 발생원별로 분리했다. 공개 녹음의 실제 기기 국가, 모든 편집본 재사용, foundation 사전학습 미노출까지 검증한 것은 아니다.
- 새 시험 파일 79개는 관련 파일을 포함한 작은 공개 코호트다. 실제 마이크·ESP32-S3 지연/RAM·현장 시간당 오탐은 검증하지 않았으며 deployment_ready=false다.

## 로컬 파일 실행

호스트: 이 노트북. 작업 디렉터리: `/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn`. 기존 환경을 사용하며 설치하지 않는다. 아래 명령은 종료형이며 터미널을 유지할 필요가 없다.

```bash
cd '/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn'
'/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/.venv/bin/python' -B -m comparison.infer --bundle '/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/models/dscnn_comparison_20261114' --audio '/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/data/comparison_20261114/new_public/bsb/raw/bsb-comparison-0107.mp3' --backend int8 --mode clip
'/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/.venv/bin/python' -B -m comparison.infer --bundle '/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/models/dscnn_comparison_20261114' --audio '/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/data/comparison_20261114/new_public/bsb/raw/bsb-comparison-0107.mp3' --backend int8 --mode replay
```

정상 결과는 종료 코드0의 JSON이다. `clip`은 label/probabilities, `replay`는 파일 재생 시각 기준 alerts를 추가한다. replay 시각은 실제 소리의 정답 onset이나 실보드 처리 지연을 뜻하지 않는다. 종료 후 해당 CLI 프로세스가 남지 않는다.

모든 결과와 비채택 비교: `/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn/reports/comparison_20261114`. 출처와 라이선스: PUBLIC_DATA_CREDITS.md. 공개 업로드/커밋/푸시는 하지 않았다.
