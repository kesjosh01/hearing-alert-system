# 공식 배포 환경의 실제 GPU 학습 검증 — 통과

**이 노트북의 NVIDIA GeForce RTX 5070 Ti Laptop GPU에서, 공식 PyPI TensorFlow 2.21.0 기반의 기존 분류기 학습·저장·재로딩을 검증했다.** 성공 범위는 새 `.venv-gpu-official221` 환경에 한정한다. 기존 TensorFlow 2.18.1 환경과 실패 기록은 그대로 보존했다.

| 판정 대상 | 결과 |
|---|---|
| 실제 GPU 순전파·역전파·Adam 업데이트 | 통과; 네 trainable tensor가 모두 변경됐고 optimizer iteration=1 확인 |
| 실제 fold 0 학습 | 1 epoch·2 steps·batch 30 = 총 60 draws, GPU 강제, 새 Adam 사용 |
| 모델 저장·재로딩 | 통과; validation 408개 예측 최대 절대 차이 0.0 |
| 기존 Windows 체크포인트 CPU·GPU 비교 | 통과; rtol=1e-4, atol=1e-5, 최대 절대 차이 1.9073486328125e-6 |
| 기존 원본·체크포인트·split·임베딩·코드 보호 | 7,137개 파일 전후 SHA-256 일치 |
| 기존 `.venv`·`.venv-gpu` 및 새 검증 환경 패키지 보존 | 검증 전후 패키지 목록 동일 |
| 정규 12-fold 비교·OOF·최종 성능 적합성 | 이번 범위에서 미실행·미판정 |

## 사용한 공식 환경

- 환경: `/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/yamnet/.venv-gpu-official221`
- Python 3.12.15, TensorFlow **2.21.0**, tf-keras **2.21.0**, TensorFlow Hub **0.16.1**, NumPy **1.26.4**, setuptools **80.9.0**.
- CUDA 12.8 계열의 명시적 고정 패키지, cuDNN **9.8.0.87**, NCCL **2.27.7**.
- 공식 PyPI 출처만 사용했다. 비공식 TensorFlow wheel·RC·nightly·CUDA 13 패키지는 설치하지 않았다.
- 기존 GPU 환경 대비 변경한 패키지는 TensorFlow, tf-keras, h5py(3.14.0), protobuf(6.33.6), cuDNN, NCCL의 6개다. 나머지는 기존 버전을 유지했다. 전체 52개 목록과 해시는 설치 기록 및 `packages_gpu_freeze.txt`에 있다.
- `pip check` 및 기존 단위 테스트 7/7 통과. 개발 캐시 1,635개·17,062개 패치의 해시·shape·유한값 검증 통과.

기존 `.venv`와 `.venv-gpu`에는 설치하지 않았다. 새 가상환경 안에만 패키지를 설치했고 환경변수는 해당 실행 셸에만 적용했다. 새 가상환경 내부의 `.gitignore`는 설치 산출물이 Git에 포함되지 않도록 설정했다. Docker·시스템 CUDA·드라이버·전역 설정 변경은 없었다.

공식 TensorFlow 2.21.0 wheel의 build metadata는 CUDA 12.5.1, compute_90 PTX를 표시하며 native sm_120은 포함하지 않는다. 실제 로드한 NVIDIA 사용자 라이브러리는 모두 새 가상환경의 CUDA 12.8 계열·cuDNN 9.8이었다. `/usr/lib`의 기존 NVIDIA 드라이버만 시스템 라이브러리로 사용했다. `loaded_cuda_libraries.json`의 unexpected 항목은 비어 있다. CUDA 12.5.1은 TensorFlow 바이너리의 빌드 메타데이터이며, 이번에 CUDA 12.5 런타임 패키지를 재설치했다는 뜻이 아니다.

## 실제 GPU 연산 근거

`gpu_validation/device_evidence.json`에서 Dense의 MatMul 2개와 gradient MatMul 3개가 `/device:GPU:0`에 배치된 것을 확인했다. 실행된 네 gradient tensor의 장치도 `/job:localhost/replica:0/task:0/device:GPU:0`이었다. soft device placement=false, GPU memory growth=true로 실행했다. 파라미터 변경과 Adam iteration 증가를 확인해 단순 장치 인식 이상의 실행 증거를 남겼다.

TensorFlow 로그에는 PTX JIT 경고가 남아 있으나, 이번에는 XLA cluster 컴파일 및 GPU 연산이 성공했다. native sm_120 포함 여부만으로 실제 학습 가능성을 단정할 수 없으며, 이 결과는 필요한 연산을 실제 실행한 증거에 따른다. 개별 패키지의 효과를 분리한 실험은 아니므로 TensorFlow 한 패키지만 변경하면 항상 해결된다고 일반화하지 않는다. 재사용 시에는 전체 환경 고정 목록을 기준으로 한다.

실행 시각은 **2026-11-09 22:57:28~22:58:08 KST**다. 설치는 **249.028초**, 검증 전체는 **40.825초**(60분 제한 이내), 실제 fold 0 학습·검증·저장·재로딩은 **2.721초**였다. 이 시간은 작은 검증 실행의 경과 시간이며 CPU 대비 속도 향상이나 정규 학습 소요 시간을 측정한 값이 아니다. 종료 코드는 0이다.

## 실험 완료와 모델 성능의 구분

GPU 학습 가능성 검증은 완료했다. 이 1 epoch·2 steps 실행은 정규 CV와 모델 선택에서 제외한다. 짧은 실행의 국내 타깃 recall은 0.5, 국내 macro-F1은 약 0.4274였고, 해외 other 오탐은 216/398, 국내 도어락 오탐은 1/2로 임시 검증 가드를 충족하지 못했다. 이는 학습·저장·재로딩 파이프라인 검증의 실패를 뜻하지 않으며, 이 점수로 모델 적합성이나 가중치를 결정하지 않는다.

정규 가중치 1·2·4 × 4 folds, OOF 집계, refit, 고정 test 특징 추출·평가, DS-CNN, 보드 실행은 하지 않았다. 고정 test 임베딩은 전후 **0개**다. 공통 ESC-50 validation 398개를 중복 합산한 비교 결과도 생성하지 않았다. 장시간 학습 안정성과 12-fold 완료, 모델의 제품 적합성은 다음 단계다.

## 실행 명령과 산출물

노트북 호스트의 위 작업 디렉터리에서 실행한 검증 명령이다. 현재 프로세스는 종료됐다.

```bash
export CAPSTONE_GPU_VENV="$PWD/.venv-gpu-official221"
export CAPSTONE_GPU_CACHE="$PWD/.cache/gpu_training_validation_official221_20261109_attempt1"
source tools/activate_gpu.sh && exec .venv-gpu-official221/bin/python -u tools/gpu_cv.py validate --profile official221
```

이 캐시와 run은 보존 중이며 같은 경로를 덮어쓰는 재실행은 허용하지 않는다. 실제 하위 명령·PID·경과 시간·종료 상태는 `execution_events.json`에 있다. 설치 명령과 패키지 해결 결과·출처·SHA-256은 `../official_gpu_candidate_review_20261109T134648Z/installation_events.json`, `pip_resolution.json`, `pip_installation.json`에 기록했다. 사전 검토 단계에서는 wheel의 작은 HTTP 범위로 build_info.py만 읽었고, 실제 설치 단계에서 PyPI 배포 wheel을 다운로드했다.

검증 모델은 `runs/yamnet/ubuntu_gpu_validation_1e2s_s42_20261109T135728Z/classifier.keras`이며 SHA-256은 `41ae1ef23a3446fb36aff9fa7ada1a49d0555d9f8d8d99f0726d8360f16c569e`다. Windows 원본 체크포인트는 수정하지 않았고, 별도 정규화 사본의 네 텐서 해시를 원본과 대조했다.

프로젝트의 기존 sound_baseline/·설정·split·임베딩은 수정하지 않았다. GPU 전용 활성화와 검증/집계 보조 코드만 확장했고, 수정 전 사본과 `helper_profile_changes.diff`, 실행 당시 `executed_helpers/`를 보관했다. 이 공식 2.21 프로필은 보조 실행기에서 최소 검증 전용으로 제한되어 있어 정규 CV를 자동으로 시작하지 않는다.

공식 출처: [TensorFlow 2.21.0 PyPI](https://pypi.org/project/tensorflow/2.21.0/), [tf-keras 2.21.0 PyPI](https://pypi.org/project/tf-keras/2.21.0/). 실제 검증 근거는 아래 상세 기록과 원시 로그에 있다.

---

# 실제 GPU 학습 가능성 검증

실제 GPU 학습 검증: **통과**. 종료 코드 `0`.
검증 범위는 기존 분류기의 GPU 순전파·역전파·Adam 업데이트, 실제 개발 캐시를 쓰는 fold 0의 1 epoch·2 steps 학습 및 저장·재로딩, 기존 체크포인트 CPU·GPU 출력 비교다. 정규 12-fold 비교와 모델 성능 적합성 평가는 이 실행에 포함하지 않는다.

시작 2026-11-09T13:57:28.036771+00:00, 종료 2026-11-09T13:58:08.861602+00:00 (UTC; KST=UTC+9). 전체 경과 시간 40.825초. 최초 검증 제한은 3,600초다.

| 검증 | 결과 |
|---|---|
| gpu_forward_backward_adam | 통과 |
| fold0_1epoch_2steps_save_reload | 통과 |
| existing_checkpoint_cpu_gpu_parity | 통과 |
| originals_and_packages_preserved | 통과 |

## 실제 실행 증거

- GPU: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`, UUID `GPU-10f7f4b0-eee7-9d43-26de-7bbefdbfc7ce`.
- Dense·gradient·Adam을 `/GPU:0`에 배치하고 soft device placement=false로 CPU 학습 대체를 금지했다. 데이터 읽기·샘플링·메트릭·파일 저장은 호스트에서 처리한다.
- 개발 캐시: 1635개 임베딩, 17062개 패치.
- GPU 업데이트: loss=1.0993397235870361, optimizer iterations=1, trainable tensor 변화=[True, True, True, True].
- 실제 fold 0 실행: `/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/yamnet/runs/yamnet/ubuntu_gpu_validation_1e2s_s42_20261109T135728Z`; 1 epoch·2 steps·batch 30, 국내 가중치 4, seed 42, learning rate 0.001, 기존 구조·샘플링·가드 유지.
- 재로딩 후 validation 408개 예측의 최대 절대 차이: 0.0; 체크포인트 SHA-256 `41ae1ef23a3446fb36aff9fa7ada1a49d0555d9f8d8d99f0726d8360f16c569e`.
- 이 짧은 학습은 파이프라인 검증 전용이며 정규 CV와 모델 선택에서 제외한다.
- 기존 체크포인트 CPU·GPU 출력: rtol=0.0001, atol=1e-05, 최대 절대 차이=1.9073486328125e-06. CPU 분류기 실행은 이 수치 비교에 한정했다. TF32는 이 비교 중에만 비활성화했다.

## 원본과 환경 보존

- 보호 파일 7137개 실행 전후 SHA-256 일치: True.
- 기존 `.venv` 및 `.venv-gpu`와 선택 환경 패키지 목록 보존: True. 공식 PyPI 패키지만 설치한 별도 `.venv-gpu-official221`에서 검증했다. 설치 기록은 프로젝트 reports/official_gpu_candidate_review_20261109T134648Z/installation_status.json 및 pip_installation.json에 있다.
- Windows 체크포인트 원본 SHA-256 `8a2f37c6a58cdd93013759cd5c8b10fbc800f4f082dca96e85aa34f9ad1f465a`. 별도 경로 정규화 사본의 텐서 해시 일치: True.
- 고정 test 임베딩 0개. test 특징 추출·평가, refit, DS-CNN, 보드 실행은 하지 않았다.
- 이전 실패 기록과 원본 학습 코드·설정·split·임베딩·기존 체크포인트를 보존했다. GPU 전용 환경변수와 캐시는 별도 프로세스·프로젝트 경로에서만 설정했다.

## 판단 범위와 기록

통과는 이 GPU·패키지 조합에서 기존 분류기의 짧은 실제 학습과 저장·재로딩이 가능하다는 뜻이다. 장시간 학습 안정성, 12-fold 완료, GPU 속도 이득, 최종 모델 성능·제품 적합성은 별도 검증 대상이다. 미통과인 경우 최초 종료 오류와 수행된 단계까지만 판단한다.

실행 당시 보조 코드 사본은 `executed_helpers/`에 있다. 공식 2.21 프로필 추가 전 사본은 프로젝트 reports/official_gpu_candidate_review_20261109T134648Z/helpers_before/에, 앞선 수정 전 사본은 .cache/20261109_validation_retry/helper_before/에 보관했다. `validation_summary.json`, `validation_checks.csv`, `gpu_validation/`, `execution_events.json`, `packages_*`, `environment.json`, `checkpoint_normalization.json`, `protected_before.json`, `protected_after.json`, `integrity.json`에 상세 증거를 보관한다.

## 종료 확인

호스트에서 검증 worker PID `32319`가 사라졌고 GPU 계산 프로세스가 없음을 확인했다. GPU 메모리는 실행 전과 같은 14 MiB였다. 별도 종료 신호는 보내지 않았다. `termination_check.json`, `gpu_processes_after.csv`, `gpu_info_after.csv`에 보관했다.
