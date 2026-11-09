# Ubuntu에서 YAMNet 작업 이어가기

Ubuntu 24.04 x86-64의 Python 3.12와 CPU를 기준으로 작성했습니다. 아래 명령은 Ubuntu 터미널에서 실행합니다. 데이터 저장소를 복제할 때 `HwanheeJu415-rgb` 계정의 GitHub 인증이 필요합니다.

```bash
sudo apt update
sudo apt install -y git ffmpeg python3 python3-venv curl
git clone --branch codex/hwanhee-model-training --single-branch \
  https://github.com/kesjosh01/hearing-alert-system.git
git clone https://github.com/HwanheeJu415-rgb/hearing-alert-dataset.git
```

## 데이터와 기존 체크포인트 복원

두 저장소를 같은 상위 디렉터리에 복제한 상태에서 실행합니다. 이미 복제했다면 각 저장소에서 `git pull`로 갱신하세요. 체크섬은 원본 음원, 고정 split, 임베딩, 체크포인트와 실행 기록의 바이트를 검증합니다.

```bash
(cd hearing-alert-dataset && sha256sum -c SHA256SUMS)
cd hearing-alert-system/ml/training/yamnet
dataset=../../../../hearing-alert-dataset
cp -a "$dataset/data/." data/
mkdir -p runs
cp -a "$dataset/runs/." runs/
```

ESC-50 음원 2,000개는 공식 저장소에서 가져옵니다. 아래 커밋의 오디오와 원본 메타데이터는 기존 학습 환경의 사본과 바이트 단위로 일치합니다. 이미 같은 버전이 지정된 경로에 있으면 다운로드를 건너뜁니다.

```bash
esc50_commit=33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6
mkdir -p data/cache/ESC-50
(
  set -e
  cd data/cache/ESC-50
  curl -fL --retry 3 "https://codeload.github.com/karolpiczak/ESC-50/tar.gz/$esc50_commit" \
    -o ESC-50.tar.gz
  tar -xzf ESC-50.tar.gz
  mv -T "ESC-50-$esc50_commit" ESC-50-master
)
```

최종 오디오 경로는 `data/cache/ESC-50/ESC-50-master/audio/`입니다. 학습 코드가 캐시된 임베딩의 출처를 원본 음원 해시로 확인하므로 캐시를 복원해도 ESC-50 원본이 필요합니다. FSD50K는 후보 메타데이터만 있고 baseline 학습에는 쓰지 않았습니다. 기존 결과를 재평가하려면 복원한 split의 바이트를 유지하세요. 코드가 각 run에 저장된 split SHA-256과 현재 파일을 대조합니다.

## 환경 설치와 동작 확인

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-yamnet.txt
python -m unittest discover -s tests
python -m sound_baseline doctor
```

`doctor`에서 네 fold, fold당 2,046개, 개발 1,635개, 고정 test 411개를 확인합니다. 복원한 `data/processed/yamnet_v1/embeddings/`에는 개발 데이터의 `.npy`와 대응하는 JSON이 있습니다. 바로 아래 학습 명령을 실행하면 저장된 특징을 사용합니다. `prepare`는 캐시를 새로 만들거나 원본이 바뀌었을 때 사용하세요.

## 체크포인트 사용과 다음 실험

기존 첫 fold 체크포인트는 `runs/yamnet/first_fold0_w4/classifier.keras`에 있습니다. 로컬 음원 하나를 분류하려면 `--audio`에 실제 파일 경로를 넣습니다.

```bash
python -m sound_baseline predict --run runs/yamnet/first_fold0_w4 --audio /path/to/recording.wav
```

다음 개발 실험은 같은 네 fold와 시드로 국내 타깃 샘플링 가중치 1·2·4배를 비교합니다. 각 `train` 실행은 새 분류기를 초기화하고 별도 폴더에 저장합니다. 기존 run은 결과 확인과 추론에 사용할 수 있습니다.

```bash
for weight in 1 2 4; do
  python -m sound_baseline train --fold all --seed 42 --domestic-weight "$weight" \
    --run-name "ubuntu_cv_w${weight}" || break
done
```

검증 결과로 설정을 고른 뒤 [학습 안내](TRAINING_GUIDE.md)의 재학습·고정 test 평가 절차를 따릅니다. 안내의 PowerShell 예제는 Ubuntu에서 `python -m sound_baseline ...` 형식으로 실행하세요. 기존 1-epoch CV와 refit은 파이프라인 통합 검사 기록입니다.
