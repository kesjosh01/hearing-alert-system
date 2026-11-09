# YAMNet 첫 학습 실행 안내

이 문서의 명령은 Windows PowerShell 기준입니다. Ubuntu의 환경 설치와 기존 데이터·체크포인트 복원은 [Ubuntu 안내](UBUNTU.md)를 따릅니다.

이 프로젝트는 기존 원본 그룹 분할과 국내 가중치를 읽어 오디오를 전처리하고, 고정된 YAMNet에서 1024차원 특징을 추출한 뒤 작은 분류기를 학습합니다. 처음에는 fold 0으로 실행 과정을 익히고, 설정 비교에는 4-fold 교차검증을 사용합니다. 고정 test는 최종 모델을 정한 뒤 별도 명령으로 평가합니다.

현재 PC에는 가상환경·필수 패키지·YAMNet 모델 캐시와 개발 음원 1,635개의 전처리/특징 캐시가 준비되어 있습니다. `runs/yamnet/first_fold0_w4/`는 기본 설정으로 실제 수행한 첫 fold 결과이며, 16 epoch 실행 후 epoch 8의 모델이 선택되었습니다. 검증 국내 경보 4/4, 초인종 3/4, 도어락 2/2를 맞혔고 해외 other 398개 중 10개에서 오탐했습니다. 이 수치는 단일 개발 fold의 결과이며 고정 test 성능이 아닙니다. `smoke_*`, `integration_*` 실행은 소프트웨어 연결 검사 용도이므로 모델 선택이나 최종 성능 보고에 사용하지 않습니다. 본격적인 4-fold/가중치 비교와 고정 test 평가는 아직 수행하지 않았습니다.

## 1 실행 구조와 준비물

```text
원본 MP3 / M4A / WAV
  → FFmpeg로 16 kHz mono PCM16 WAV 생성
  → float32 파형으로 읽기, 값은 PCM16 / 32768
  → YAMNet 고정 특징 추출
  → 시간축의 1024차원 embedding들을 .npy로 저장
  → 원본 그룹별 가중치로 embedding을 추출해 작은 분류기 학습
  → 검증 결과와 classifier.keras 저장
```

- Windows x64, Python 3.11 또는 3.12, FFmpeg가 필요합니다. GPU 없이 CPU로 시작할 수 있습니다.
- 검증 조합은 `requirements-yamnet.txt`에 고정했습니다. TensorFlow 2.18.1, tf-keras 2.18.0, TensorFlow Hub 0.16.1, NumPy 1.26.4를 사용합니다. 임의로 일부만 업그레이드하지 않습니다.
- YAMNet은 Keras 2를 사용하므로 코드에서 `tf_keras`를 명시적으로 사용합니다. 분류기 초기화와 dropout에도 seed를 지정합니다.
- 원본 음원은 수정하지 않습니다. 변환 WAV와 embedding은 `data/processed/yamnet_v1/`, 학습 결과는 `runs/yamnet/`에 저장하며 Git에서 제외합니다.
- Windows 한글 경로에 대한 TensorFlow 네이티브 파일 입출력 문제를 피하기 위해, 현재 프로젝트에서는 모델 다운로드·TF 임시 파일·실시간 TensorBoard 로그를 `%LOCALAPPDATA%\CapstoneSoundCache`에 저장합니다. 모델과 로그 사본은 프로젝트의 실행 결과 폴더에도 저장합니다. `doctor`가 실제 캐시 위치를 출력합니다.
- 최초 패키지 설치와 YAMNet 다운로드에는 인터넷이 필요합니다. 캐시가 준비되면 기본 학습은 로컬에서 실행됩니다. W&B는 기본 비활성화이며 오디오 업로드 코드는 없습니다.

## 2 PowerShell에서 프로젝트 폴더 열기

아래 명령을 한 줄씩 실행합니다.

```powershell
Set-Location -LiteralPath 'C:\path\to\hearing-alert-system\ml\training\yamnet'
```

명령의 `python`은 다른 설치와 섞이지 않도록 항상 프로젝트 가상환경의 실행 파일로 지정합니다. 가상환경 활성화는 필요하지 않습니다.

## 3 실행 환경 설치 또는 확인

환경이 이미 준비되었다면 먼저 확인합니다.

```powershell
& .\.venv\Scripts\python.exe -m sound_baseline doctor
```

현재 데이터 구성에서는 다음 정보가 나와야 합니다.

```text
validated_folds: 4
records_per_fold: 2046
development_records: 1635
test_records: 411
labels: fire_alarm=0, doorbell=1, other=2
```

새 PC 또는 설치가 필요한 경우 다음을 실행합니다. 실행 정책 변경은 이 PowerShell 프로세스에만 적용됩니다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\setup_yamnet.ps1
```

Python을 자동으로 찾지 못하면 Python 3.12 x64를 설치한 뒤 실제 설치 경로를 지정합니다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\setup_yamnet.ps1 -Python "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
```

FFmpeg가 없다는 안내가 나오면 설치하고 PowerShell을 다시 엽니다.

```powershell
winget install --id Gyan.FFmpeg -e
```

Python 버전이나 설치 위치가 다르면 `-Python`에 자신의 경로를 넣습니다. `python` 명령이 Microsoft Store를 여는 경우에도 가상환경의 `python.exe`를 직접 실행하면 됩니다.

## 4 YAMNet 다운로드와 소규모 실행 검사

```powershell
& .\.venv\Scripts\python.exe -m sound_baseline doctor --load-yamnet
& .\.venv\Scripts\python.exe -m sound_baseline smoke
```

`smoke`는 test를 제외한 실제 음원 12개로 특징 추출, 두 epoch의 짧은 학습, 모델 저장과 재로딩을 검사합니다. 성공하면 `SMOKE PASS`와 결과 폴더를 출력합니다. 이 결과의 정확도는 본 실험 성능으로 사용하지 않습니다.

처음 다운로드할 때 잠시 출력이 없을 수 있습니다. `WARNING`, `deprecated`, `pkg_resources` 안내만 출력되고 명령이 정상 종료되면 실패가 아닙니다. `ERROR` 또는 traceback으로 종료되면 마지막 오류를 확인합니다.

## 5 개발 데이터 전처리와 특징 추출

전처리와 특징 추출을 한 번에 실행하려면 다음 명령을 사용합니다.

```powershell
& .\.venv\Scripts\python.exe -m sound_baseline prepare
```

기본 대상은 모든 fold에서 공통인 비시험 자료 1,635개입니다. Train과 validation의 역할은 학습 시 각 fold 목록으로 결정합니다. 특징 추출은 고정 모델로 각 파일에 독립적으로 수행하며, test의 특징은 기본 생성하지 않습니다.

두 단계를 따로 확인하려면 다음과 같이 실행할 수도 있습니다.

```powershell
& .\.venv\Scripts\python.exe -m sound_baseline preprocess
& .\.venv\Scripts\python.exe -m sound_baseline extract
```

중단되었다면 같은 명령을 다시 실행합니다. 원본 해시·전처리 설정·캐시 파일 해시가 일치하는 결과는 재사용합니다. 빠른 확인만 원하면 `prepare --limit 3`을 사용할 수 있지만, 전체 학습 전에 제한 없는 `prepare`를 완료해야 합니다.

결과는 다음 위치에 있습니다.

```text
data/processed/yamnet_v1/
  audio/<sample_id>.wav           # 16 kHz mono PCM16
  audio/<sample_id>.json          # 실제 디코딩 길이, RMS, 해시 등
  embeddings/<sample_id>.npy      # [패치 수, 1024] float32
  embeddings/<sample_id>.json     # 원본/특징 해시와 설정
  extract_development.csv         # 개발 데이터 특징 목록
  yamnet_identity.json            # 사용한 사전학습 모델 식별 정보
```

YAMNet의 기본 0.96초 패치와 0.48초 이동 간격을 사용합니다. 첫 패치 계산에 필요한 파형 길이를 고려해 0.975초보다 짧은 입력은 오른쪽에 0을 채웁니다. 별도의 peak 음량 정규화나 무음 제거는 하지 않습니다.

현재 정답은 사용자가 검수한 **파일 단위 라벨**입니다. 첫 baseline은 해당 파일의 각 YAMNet 패치에 그 라벨을 상속합니다. 이 구현이 별도의 이벤트 시작·종료 시각을 만들어내는 것은 아닙니다. 타깃 파일 내부의 긴 비타깃 구간이 있다면 이후 구간 주석을 바탕으로 입력을 제한하는 실험이 필요합니다.

## 6 첫 번째 fold 학습

```powershell
& .\.venv\Scripts\python.exe -m sound_baseline train --fold 0 --run-name my_first_fold0_w4
```

실험 이름이 이미 존재하면 덮어쓰지 않습니다. 새 이름을 쓰거나 `--run-name`을 생략하면 시각을 포함한 새 폴더를 만듭니다.

기본 설정은 다음과 같습니다.

| 항목 | 기본값 |
|---|---|
| 특징 추출기 | YAMNet 고정, 가중치 업데이트 없음 |
| 분류기 | Dense 128 ReLU → Dropout 0.2 → 세 클래스 logits |
| Optimizer와 loss | Adam, learning rate 0.001, sparse cross-entropy |
| Batch | 30개 |
| Epoch | 30 batch, 즉 900회 패치 추출 |
| 최대 epoch | 30 |
| Early stopping | 검증 checkpoint 순위가 8 epoch 동안 개선되지 않으면 종료 |
| 클래스 추출 비중 | 경보 1/3, 초인종 1/3, other 1/3 |
| 타깃 클래스 내 원본 가중치 | 국내 4, 보조 1 |
| other 내부 비중 | 해외 일반 60%, 해외 혼동음 30%, 국내 도어락 10% |

먼저 원본 그룹을 고르고 그 원본의 패치를 균등 추출합니다. 긴 파일이 더 많은 패치를 만든다는 이유로 전체 학습 비중이 증가하지 않습니다. 대체 추출을 하므로 여기서 epoch는 원본 전체를 한 번 읽는다는 의미가 아닙니다.

첫 baseline에는 파형 증강과 추가 class/domain loss 가중치를 적용하지 않습니다. 특징 추출기를 고정한 상태의 분류 가능성을 먼저 확인합니다. Dropout은 분류기 학습 중 적용됩니다.

검증에서는 각 패치의 softmax를 파일별로 평균하고, 같은 원본 그룹에 여러 파일이 있으면 파일 점수를 다시 균등 평균해 원본별로 평가합니다. 세 클래스 중 최대 점수를 선택하며 이벤트 임계값·연속 감지·알림 지연은 아직 구현하지 않았습니다.

## 7 학습 결과 읽기

`runs/yamnet/my_first_fold0_w4/`를 확인합니다.

| 파일 | 확인할 내용 |
|---|---|
| `classifier.keras` | 선택된 작은 분류기. YAMNet 본체는 포함하지 않음 |
| `run.json` | 라벨, 설정, 패키지 버전, 코드·분할·데이터 식별 정보 |
| `result.json` | 선택 epoch, 검증 지표, 검증 조건 충족 여부 |
| `history.csv` | epoch별 학습 loss와 검증 성능 |
| `validation_clips.csv` | 파일별 정답, 예측, 세 클래스 점수 |
| `validation_groups.csv` | 원본 그룹별 예측 |
| `validation_metrics.json` | confusion matrix, 클래스별 precision·recall·F1 등 |
| `sampling_groups.csv` | 실제 사용한 그룹별 추출 확률과 패치 수 |
| `sample_exposure.csv` | 학습 중 실제로 뽑힌 파일별 횟수 |
| `environment.txt` | 해당 실행의 전체 설치 패키지 목록 |

PowerShell에서 요약을 읽을 수도 있습니다.

```powershell
Get-Content -Raw -Encoding UTF8 .\runs\yamnet\my_first_fold0_w4\result.json
```

Checkpoint 선택에는 국내 경보·초인종의 원본 단위 recall 평균을 주 지표로 사용합니다. 임시 검증 조건으로 해외 other 클립 오탐률 5% 이하와 국내 도어락 오탐 0개를 두었습니다. 이는 제품 요구 성능이나 안전 기준으로 입증한 값이 아닙니다. 조건을 만족한 epoch를 우선하고, 그 안에서 타깃 recall·국내 macro-F1·검증 loss로 순위를 정합니다. 만족한 epoch가 없다면 위반이 적은 후보를 저장하고 `validation_guards_passed=false`로 표시합니다.

전체 accuracy 하나보다 `domestic_groups.classes.fire_alarm.recall`, `doorbell.recall`, 타깃 precision과 other 오탐을 함께 읽습니다. 짧은 선별 클립의 오탐률은 실제 환경의 시간당 오경보가 아닙니다.

## 8 네 fold로 설정 평가

```powershell
& .\.venv\Scripts\python.exe -m sound_baseline train --fold all --domestic-weight 4 --seed 42 --run-name cv_w4_s42
```

각 fold마다 새 분류기와 optimizer로 시작합니다. 결과 폴더에는 `fold_0/`부터 `fold_3/`과 다음 요약이 생깁니다.

- `cv_folds.csv`: fold별 검증 성능과 선택 epoch
- `cv_summary.json`: 국내 타깃 recall 평균·표준편차, 최종 재학습 epoch 수
- `oof_domestic_clips.csv`: 각 국내 개발 원본이 정확히 한 번씩 검증에 등장한 예측
- `oof_domestic_metrics.json`: 국내 OOF 성능

ESC-50 validation은 네 실행에서 공통으로 사용합니다. 이를 서로 다른 독립 OOF 표본처럼 네 배로 합산하지 않습니다.

## 9 국내 가중치 1배와 2배 비교

4배 실행과 같은 fold·seed를 사용합니다. 분할 파일을 재생성하거나 protocol JSON을 수정하지 않습니다.

```powershell
& .\.venv\Scripts\python.exe -m sound_baseline train --fold all --domestic-weight 1 --seed 42 --run-name cv_w1_s42
& .\.venv\Scripts\python.exe -m sound_baseline train --fold all --domestic-weight 2 --seed 42 --run-name cv_w2_s42
```

세 실행의 `cv_summary.json`, `cv_folds.csv`, OOF 지표를 비교합니다. 국내 경보·초인종 recall의 개선이 특정 fold에만 나타나는지, 타깃 precision이나 other 오탐이 나빠지는지 함께 봅니다. 4배가 자동으로 최선이라는 전제는 두지 않습니다. 비교에 test 결과를 사용하지 않습니다.

## 10 선택한 설정으로 최종 재학습

예를 들어 개발 결과로 4배를 선택했다면 다음을 실행합니다. 1배나 2배를 선택했다면 해당 CV 폴더 경로를 넣습니다.

```powershell
& .\.venv\Scripts\python.exe -m sound_baseline refit --cv-run .\runs\yamnet\cv_w4_s42 --run-name final_w4_s42
```

국내 개발 원본 34개 전체와 해외 보조 타깃 3개, ESC-50 train 1,200개를 사용합니다. 학습 epoch는 4-fold의 선택 epoch 중앙값을 반올림한 값으로 고정합니다. 국내 validation을 재학습에 포함했으므로 국내 검증 성능으로 다시 early stopping하지 않습니다. 해외 validation은 결과 확인용으로 남습니다.

이 baseline의 최종 분류 규칙도 파일별 평균 점수의 argmax입니다. 이벤트용 임계값과 후처리 최적화는 별도 단계입니다.

## 11 고정 test 평가

모델과 설정을 확정한 뒤에 실행합니다. 앞선 개발·학습 명령은 test를 평가하지 않습니다.

```powershell
& .\.venv\Scripts\python.exe -m sound_baseline prepare --include-test
& .\.venv\Scripts\python.exe -m sound_baseline evaluate --run .\runs\yamnet\final_w4_s42 --split test
```

`prepare --include-test`는 기존 개발 캐시를 재사용하고 빠진 test 특징을 추가합니다. `evaluate --split test`는 최종 refit 모델에서만 허용합니다. 결과는 `test_evaluation/metrics.json`과 예측 CSV에 저장합니다. 같은 결과가 이미 있으면 기존 평가를 읽어 보여줍니다.

국내 test는 타깃별 4개뿐입니다. 표본 수와 정답 개수를 함께 보고합니다. Test 결과로 설정을 바꾸었다면 해당 결과를 독립적인 최종 성능으로 다시 주장하지 않습니다.

## 12 새로운 음원 하나 분류

`--audio`에는 자신의 새 파일 경로를 넣습니다. 아래 `new_audio.m4a`는 예시 파일명입니다.

```powershell
& .\.venv\Scripts\python.exe -m sound_baseline predict --run .\runs\yamnet\final_w4_s42 --audio 'C:\경로\new_audio.m4a'
```

예측 클래스와 세 클래스 점수를 출력합니다. `classifier.keras`는 1024차원 특징용 분류기이므로, 오디오 파일을 직접 분류할 때는 이 명령처럼 동일한 전처리와 YAMNet을 함께 사용합니다. 점수는 보정된 정답 확률이나 실제 화재 발생 확률이 아닙니다.

## 13 TensorBoard와 선택적인 W&B

TensorBoard 로그는 기본 생성됩니다. 한글 경로 문제를 피하도록 실제 영문 캐시 로그 경로를 사용합니다.

```powershell
$tfLogDir = & .\.venv\Scripts\python.exe -c "from sound_baseline.features import runtime_cache; print(runtime_cache() / 'logs')"
& .\.venv\Scripts\python.exe -m tensorboard.main --logdir "$tfLogDir"
```

출력되는 로컬 주소를 브라우저에서 엽니다. 학습 종료 시 로그 사본은 해당 run의 `tensorboard/`에도 저장됩니다.

W&B가 필요하면 별도 설치 후 로컬 offline 기록부터 시작할 수 있습니다.

```powershell
& .\.venv\Scripts\python.exe -m pip install wandb
& .\.venv\Scripts\python.exe -m sound_baseline train --fold 0 --wandb-mode offline --run-name fold0_wandb_offline
```

`offline`은 로컬에 기록합니다. 클라우드에 기록하려면 계정 로그인을 구성하고 명시적으로 `--wandb-mode online`을 사용합니다. 코드에서는 수치 지표와 실험 설정만 전달하며 원본 오디오를 기록하지 않습니다.

## 14 오류 해결과 구현 검증

- `Missing embeddings`: 제한 없는 `prepare`를 먼저 완료합니다.
- `Stale/Corrupt embedding cache`: 원본이나 캐시가 변경되었습니다. `prepare`를 다시 실행해 해당 파일을 갱신합니다.
- `FileExistsError`가 run 폴더를 가리킴: 실험 이름을 바꾸거나 `--run-name`을 생략합니다.
- `Empty other training pool`: 분할이나 pool 태그가 변경되었습니다. 해외 일반·해외 혼동음·국내 도어락의 Train 존재 여부를 확인합니다.
- `is not a directory` 등 TensorFlow 경로 오류: `doctor`의 캐시가 영문 경로인지 확인합니다. 필요한 경우 자신에게 쓰기 권한이 있는 영문 경로를 지정합니다.

```powershell
$env:CAPSTONE_TF_CACHE = Join-Path $env:LOCALAPPDATA 'CapstoneSoundCache'
```

- 시스템의 새 `python`으로 실행했을 때 import 오류: 프로젝트의 `.venv\Scripts\python.exe`를 직접 사용합니다.
- CPU만 표시됨: 이 안내의 기본 환경에서 정상입니다. GPU 사용은 별도 환경 구성 후 비교합니다.

자동 검사는 다음으로 실행합니다.

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

테스트는 실제 FFmpeg의 stereo 44.1 kHz → mono 16 kHz 변환, 원본 변경 시 캐시 무효화, 긴 원본에 편향되지 않는 국내 가중치, test 제외, 원본별 평가와 checkpoint 조건을 확인합니다. 전체 end-to-end 확인은 실제 YAMNet을 사용하는 `smoke` 명령으로 수행합니다.

## 구현 파일

- `config/training_protocol_v1.json`: 기존 고정 분할·국내 가중치 설계
- `config/yamnet_baseline_v1.json`: 오디오·특징·분류기 기본 설정
- `sound_baseline/core.py`: 분할 검증, 그룹 sampler, 평가 지표
- `sound_baseline/features.py`: 디코딩, YAMNet, 해시 기반 캐시, Windows 경로 호환
- `sound_baseline/training.py`: 학습, 검증, checkpoint, CV와 refit
- `sound_baseline/__main__.py`: 실행 명령
- `tools/setup_yamnet.ps1`: Windows 환경 준비

## 공식 참고 자료

- [YAMNet 공식 구현과 입력 특징 설명](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
- [YAMNet 전이학습 설명](https://blog.tensorflow.org/2021/03/transfer-learning-for-audio-data-with-yamnet.html)
- [TensorFlow 설치 안내](https://www.tensorflow.org/install/pip)
- [FFmpeg 오디오 옵션](https://www.ffmpeg.org/ffmpeg.html#Audio-Options)
