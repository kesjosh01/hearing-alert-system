# 국내 음원 과적합 DS-CNN의 ESP32-S3 빌드

[원본 classifier.keras](../classifier.keras)의 학습 가중치를 사용한 88,128바이트 int8 TFLite 모델이다. S3의 INMP441 수집, 98×40 log-Mel 변환, 사건별 재알림 정책과 C6의 UDP 수신·화면·진동·ACK를 그대로 사용한다. 송신기에는 빌드할 때 선택한 모델 하나가 포함된다.

- [기본 송신기 빌드·플래시](../../../../firmware/transmitter_s3/README.md#국내-음원-과적합-모델로-실기기-시험)
- [WS2812B LED 송신기 빌드·플래시](../../../../firmware/transmitter_s3/s3_hardware_control/README.md#국내-음원-과적합-모델-선택)

## 모델과 판정

원본 Keras 모델은 T=1로 학습됐다. 배포본은 마지막 Dense의 가중치와 편향을 4로 나눈 후 softmax를 적용한다. 11월 19일 이 체크포인트에 대해 모의 시험한 T=4 정책을 재현하기 위해서다. 각 창의 점수는 `q + 128`(0~255)이다. 화재경보·초인종 중 해당 클래스가 1위이고 각각 190·162점 이상인 창이 최근 3개 중 2개면 감지한다. 기타는 전송하지 않는다. 입력 양자화 계수는 새 모델의 텐서에서 읽는다. 판정값은 [변환 기록](conversion_report.json)에서 빌드 시 읽는다.

## 호스트 확인

- ESC-50의 프로젝트 학습 분할에서 고른 20개 파일·180창으로 양자화 범위를 보정했다. 파일 목록과 원본 버전은 [recordings.json](recordings.json)에 있다.
- 보정에서 제외한 같은 분할의 별도 녹음 8개·72창에서 T=4 float와 int8의 1위 클래스는 71/72창, 화재·초인종 2-of-3 판정은 72/72창에서 일치했다. 세 클래스의 확률 절대 오차 평균은 0.0050이었다.
- 이 비교는 변환 전후의 일치도를 측정한다. 국내 원음 43개는 재배포되지 않아 int8로 재평가하지 않았다. 원본의 학습 파일 43/43과 과거 float 정책 모사 41/43은 별도 결과다. 기존 float 정책에서는 도어락 7개 중 5개에도 알림이 발생했다. 마이크 감지율, S3 지연과 C6 실물 출력은 보드에서 확인해야 한다.

## 변환 재현

이미 포함된 `model_int8.tflite`로 펌웨어를 빌드할 수 있다. 변환을 다시 수행하려면 Python 3.11과 TensorFlow·tf-keras 2.21.0, SciPy, ai-edge-litert를 준비한다. 저장소 루트에서 아래 명령을 실행한다. `ESC50_AUDIO`는 recordings.json에 고정한 ESC-50 버전의 `audio/` 디렉터리다. 두 번째 인수에는 새 출력 폴더를 지정한다.

```bash
python ml/models/domestic_overfit_20261119/esp32s3/export_model.py "$ESC50_AUDIO" build/domestic-export
```
