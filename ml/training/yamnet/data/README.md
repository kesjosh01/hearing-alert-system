# 학습 데이터 경로

[Ubuntu 복원 안내](../UBUNTU.md)에 따라 비공개 데이터 저장소의 `data/`를 이 디렉터리에 합칩니다. 상대 경로를 그대로 유지합니다.

- `raw/private/`에는 수집 원본 47개가 있습니다. 같은 SHA-256인 도어락 두 파일은 원본 보존을 위해 남겼고, 한 파일만 분할에 포함됩니다.
- `metadata/`의 전체 manifest와 source registry에는 개별 출처, 라벨과 권리 상태가 기록되어 있습니다. 공개 출처의 ESC-50 manifest와 FSD50K 후보 목록은 이 프로젝트에도 있습니다.
- `splits/ko-source-cv4-20261108-v1/`에는 고정 test, 네 개발 fold, refit, 그룹 샘플러 파일이 있습니다. [분할 안내](splits/README.md)를 참고하세요.
- `cache/ESC-50/ESC-50-master/audio/`는 공식 ESC-50 원본을 복원하는 위치입니다.
- `processed/yamnet_v1/embeddings/`의 개발용 특징과 메타데이터를 비공개 저장소에서 복원하면 학습에 바로 사용할 수 있습니다. 캐시를 다시 만들 때는 `python -m sound_baseline prepare`를 실행합니다.

공개 팀 저장소의 변경사항을 올릴 때는 원본 음원, 개별 데이터·분할 기록, 가공 데이터와 학습 실행 폴더를 Git에서 제외합니다.
