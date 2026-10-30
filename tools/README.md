# 개발 보조 도구

`audio_capture/`에서 Ubuntu PC용 음향 데이터 수집 도구를 관리할 예정입니다. 현재는 안내 문서만 있으며 오디오 처리 코드는 없습니다.

## 통신 모니터·분석

`communication/monitor.sh`는 S3·C3용 Terminator 좌우 모니터를 실행하며, `communication/analyze_log.py`는 S3 로그의 절전·이벤트·ACK를 분석합니다. [도구 안내](communication/README.md)와 [실행 방법](../communication_tests/softap_udp/README.md#c3-절전-빌드와-모니터)을 참고합니다.
