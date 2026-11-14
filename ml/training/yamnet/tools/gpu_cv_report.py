"""Descriptive CV aggregation; ESC-50 validation is never pooled as OOF."""
from __future__ import annotations

import json
from pathlib import Path
import statistics

from sound_baseline.core import LABELS, aggregate_groups, read_csv, read_json, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]


def make_validation_report(output, status):
    def artifact(name):
        path = output / name
        return read_json(path) if path.is_file() else {}

    update = artifact("gpu_validation/gpu_update.json")
    reload = artifact("gpu_validation/validation_pass.json")
    parity = artifact("gpu_validation/cpu_gpu_parity.json")
    integrity = artifact("integrity.json")
    cache = artifact("cache_before.json")
    normalization = artifact("checkpoint_normalization.json")
    gpu = artifact("gpu_validation/tensorflow_gpu.json")
    result = {"scope": "gpu_validation_only", "overall_passed": status["gpu_validation_passed"],
              "update": update, "real_fold0_and_reload": reload, "cpu_gpu_parity": parity,
              "integrity": integrity, "regular_cv_executed": False, "test_evaluated": False}
    write_json(output / "validation_summary.json", result)
    rows = [{"check": "gpu_forward_backward_adam", "passed": bool(update.get("passed"))},
            {"check": "fold0_1epoch_2steps_save_reload", "passed": bool(reload.get("passed"))},
            {"check": "existing_checkpoint_cpu_gpu_parity", "passed": bool(parity.get("passed"))},
            {"check": "originals_and_packages_preserved", "passed": bool(status.get("integrity_passed"))}]
    write_csv(output / "validation_checks.csv", rows)
    logs = output / "gpu_validation.log"
    terminating = []
    if logs.is_file() and not status["gpu_validation_passed"]:
        terminating = [line for line in logs.read_text(errors="replace").splitlines()
                       if any(word in line for word in ("TypeError:", "ValueError:", "RuntimeError:", "InvalidArgumentError:", "InternalError:", "NotFoundError:", "UnimplementedError:", "AssertionError:"))]
    lines = ["# 실제 GPU 학습 가능성 검증", "",
             f"실제 GPU 학습 검증: **{'통과' if status['gpu_validation_passed'] else '미통과 또는 미완료'}**. 종료 코드 `{status['exit_code']}`.",
             "검증 범위는 기존 분류기의 GPU 순전파·역전파·Adam 업데이트, 실제 개발 캐시를 쓰는 fold 0의 1 epoch·2 steps 학습 및 저장·재로딩, 기존 체크포인트 CPU·GPU 출력 비교다. 정규 12-fold 비교와 모델 성능 적합성 평가는 이 실행에 포함하지 않는다.", "",
             f"시작 {status['started_at']}, 종료 {status['finished_at']} (UTC; KST=UTC+9). 전체 경과 시간 {status['elapsed_seconds']:.3f}초. 최초 검증 제한은 3,600초다.", "",
             "| 검증 | 결과 |", "|---|---|"]
    for row in rows:
        lines.append(f"| {row['check']} | {'통과' if row['passed'] else '미통과 또는 미완료'} |")
    lines.extend(["", "## 실제 실행 증거", "",
                  f"- GPU: `{gpu.get('details', {}).get('device_name', '미기록')}`, UUID `{gpu.get('selected_uuid', '미기록')}`.",
                  "- Dense·gradient·Adam을 `/GPU:0`에 배치하고 soft device placement=false로 CPU 학습 대체를 금지했다. 데이터 읽기·샘플링·메트릭·파일 저장은 호스트에서 처리한다.",
                  f"- 개발 캐시: {cache.get('development_embeddings', '미확인')}개 임베딩, {cache.get('development_patches', '미확인')}개 패치."])
    if update:
        lines.append(f"- GPU 업데이트: loss={update['loss']}, optimizer iterations={update['optimizer_iterations']}, trainable tensor 변화={update['all_trainable_tensors_changed']}.")
    if reload:
        lines.extend([f"- 실제 fold 0 실행: `{reload['run']}`; 1 epoch·2 steps·batch 30, 국내 가중치 4, seed 42, learning rate 0.001, 기존 구조·샘플링·가드 유지.",
                      f"- 재로딩 후 validation {reload['validation_clips']}개 예측의 최대 절대 차이: {reload['reload_max_abs_difference']}; 체크포인트 SHA-256 `{reload['checkpoint_sha256']}`.",
                      "- 이 짧은 학습은 파이프라인 검증 전용이며 정규 CV와 모델 선택에서 제외한다."])
    if parity:
        lines.append(f"- 기존 체크포인트 CPU·GPU 출력: rtol={parity['rtol']}, atol={parity['atol']}, 최대 절대 차이={parity['max_abs_difference']}. CPU 분류기 실행은 이 수치 비교에 한정했다. TF32는 이 비교 중에만 비활성화했다.")
    if terminating or status.get("first_error"):
        lines.extend(["", "## 최초 종료 오류", "", "```text", *(terminating[-4:] or [status.get("first_error", "미기록")]), "```", "",
                      "원시 traceback은 `gpu_validation.log`, 상위 중단 기록은 `first_error.log`에 있다. 오류 뒤 정규 CV, 자동 CPU 전환, TensorFlow 버전 변경, 자동 재시도는 하지 않았다."])
    setup_note = ("공식 PyPI 패키지만 설치한 별도 `.venv-gpu-official221`에서 검증했다. 설치 기록은 프로젝트 reports/official_gpu_candidate_review_20261109T134648Z/installation_status.json 및 pip_installation.json에 있다."
                  if status.get("environment_profile") == "official221" else "이번 실행의 설치는 0건이다.")
    lines.extend(["", "## 원본과 환경 보존", "",
                  f"- 보호 파일 {integrity.get('protected_file_count', '미확인')}개 실행 전후 SHA-256 일치: {integrity.get('protected_hashes_unchanged', False)}.",
                  f"- 기존 `.venv` 및 `.venv-gpu`와 선택 환경 패키지 목록 보존: {integrity.get('package_lists_unchanged', False)}. {setup_note}",
                  f"- Windows 체크포인트 원본 SHA-256 `{normalization.get('source_sha256_before', '미확인')}`. 별도 경로 정규화 사본의 텐서 해시 일치: {normalization.get('tensor_hashes_match', False)}.",
                  f"- 고정 test 임베딩 {integrity.get('fixed_test_embeddings', '미확인')}개. test 특징 추출·평가, refit, DS-CNN, 보드 실행은 하지 않았다.",
                  "- 이전 실패 기록과 원본 학습 코드·설정·split·임베딩·기존 체크포인트를 보존했다. GPU 전용 환경변수와 캐시는 별도 프로세스·프로젝트 경로에서만 설정했다.", "",
                  "## 판단 범위와 기록", "",
                  "통과는 이 GPU·패키지 조합에서 기존 분류기의 짧은 실제 학습과 저장·재로딩이 가능하다는 뜻이다. 장시간 학습 안정성, 12-fold 완료, GPU 속도 이득, 최종 모델 성능·제품 적합성은 별도 검증 대상이다. 미통과인 경우 최초 종료 오류와 수행된 단계까지만 판단한다.", "",
                  "실행 당시 보조 코드 사본은 `executed_helpers/`에 있다. 공식 2.21 프로필 추가 전 사본은 프로젝트 reports/official_gpu_candidate_review_20261109T134648Z/helpers_before/에, 앞선 수정 전 사본은 .cache/20261109_validation_retry/helper_before/에 보관했다. `validation_summary.json`, `validation_checks.csv`, `gpu_validation/`, `execution_events.json`, `packages_*`, `environment.json`, `checkpoint_normalization.json`, `protected_before.json`, `protected_after.json`, `integrity.json`에 상세 증거를 보관한다.", ""])
    (output / "REPORT_KO.md").write_text("\n".join(lines), encoding="utf-8")


def fold_row(weight, fold, directory):
    result = read_json(directory / "result.json")
    metrics = result["validation"]
    domestic = metrics["domestic_groups"]
    row = {"weight": weight, "fold": fold, "run_name": f"ubuntu_gpu_cv_w{weight}_s42",
           "selected_epoch": result["best_epoch"], "epochs_completed": result["epochs_completed"],
           "domestic_target_macro_recall": metrics["domestic_target_macro_recall"],
           "domestic_target_macro_precision": statistics.mean(domestic["classes"][label]["precision"] for label in ("fire_alarm", "doorbell")),
           "domestic_macro_f1": domestic["macro_f1"], "all_validation_group_macro_f1": metrics["groups"]["macro_f1"],
           "domestic_confusion_matrix": json.dumps(domestic["confusion_matrix"]),
           "all_validation_group_confusion_matrix": json.dumps(metrics["groups"]["confusion_matrix"]),
           "overseas_other_fp": metrics["overseas_other_clip_false_positives"],
           "overseas_other_count": metrics["overseas_other_clip_count"],
           "overseas_other_fpr": metrics["overseas_other_clip_fpr"],
           "domestic_doorlock_fp": metrics["domestic_doorlock_false_positive_groups"],
           "domestic_doorlock_count": metrics["domestic_doorlock_group_count"],
           "guards_passed": result["validation_guards_passed"],
           "elapsed_seconds": read_json(directory/"gpu_execution.json")["elapsed_seconds"]}
    for label in LABELS:
        for field in ("recall", "precision", "f1", "support", "correct", "predicted"):
            row[f"domestic_{label}_{field}"] = domestic["classes"][label][field]
    return row


def make_report(output):
    output = Path(output)
    status = read_json(output / "status.json")
    if status.get("scope") == "gpu_validation_only":
        return make_validation_report(output, status)
    folds, weights = [], []
    for weight in (1, 2, 4):
        run = ROOT / f"runs/yamnet/ubuntu_gpu_cv_w{weight}_s42"
        # Only include a run actually launched by this invocation.
        if not (output / f"weight_{weight}").exists():
            continue
        for fold in range(4):
            child = run / f"fold_{fold}"
            if (child / "result.json").is_file() and (child / "gpu_execution.json").is_file():
                folds.append(fold_row(weight, fold, child))
        if weight in status["completed_weights"]:
            summary = read_json(run / "cv_summary.json")
            oof = read_json(run / "oof_domestic_metrics.json")
            by_weight = [r for r in folds if r["weight"] == weight]
            row = {"weight": weight, "fold_count": len(by_weight),
                   "domestic_target_recall_mean": summary["domestic_target_recall_mean"],
                   "domestic_target_recall_std_ddof1": summary["domestic_target_recall_std"],
                   "domestic_oof_target_macro_recall": oof["domestic_target_macro_recall"],
                   "domestic_oof_target_macro_precision": statistics.mean(oof["domestic_groups"]["classes"][label]["precision"] for label in ("fire_alarm", "doorbell")),
                   "domestic_oof_macro_f1": oof["domestic_groups"]["macro_f1"],
                   "domestic_oof_groups": oof["domestic_groups"]["count"],
                   "domestic_oof_confusion_matrix": json.dumps(oof["domestic_groups"]["confusion_matrix"]),
                   "domestic_oof_doorlock_fp": oof["domestic_doorlock_false_positive_groups"],
                   "domestic_oof_doorlock_count": oof["domestic_doorlock_group_count"],
                   "guards_passed_folds": sum(r["guards_passed"] for r in by_weight),
                   "overseas_fpr_fold_mean": statistics.mean(r["overseas_other_fpr"] for r in by_weight),
                   "overseas_fpr_fold_min": min(r["overseas_other_fpr"] for r in by_weight),
                   "overseas_fpr_fold_max": max(r["overseas_other_fpr"] for r in by_weight),
                   "elapsed_seconds": sum(r["elapsed_seconds"] for r in by_weight)}
            weights.append(row)
            write_json(output / f"domestic_oof_w{weight}.json", oof)
            clips = read_csv(run / "oof_domestic_clips.csv")
            write_csv(output / f"domestic_oof_clips_w{weight}.csv", clips)
            write_csv(output / f"domestic_oof_groups_w{weight}.csv", aggregate_groups(clips))
    if folds:
        reference = {(r["weight"], r["fold"]): r for r in folds}
        for row in folds:
            base = reference.get((1, row["fold"]))
            for metric in ("domestic_target_macro_recall", "domestic_target_macro_precision", "domestic_macro_f1", "overseas_other_fp", "domestic_doorlock_fp"):
                row[f"delta_vs_w1_{metric}"] = row[metric] - base[metric] if base else None
        write_csv(output / "comparison_folds.csv", folds)
    else:
        (output / "comparison_folds.csv").write_text("weight,fold,status\n", encoding="utf-8")
    if weights:
        write_csv(output / "comparison_weights.csv", weights)
    else:
        (output / "comparison_weights.csv").write_text("weight,fold_count,status\n", encoding="utf-8")
    write_json(output / "comparison.json", {"completed_folds": len(folds), "folds": folds, "weights": weights,
               "overseas_validation_policy": "398 shared ESC-50 clips: report each model separately; never pool 4x as independent OOF",
               "class_order": list(LABELS), "matrix_rows": "true", "matrix_columns": "predicted"})
    lines = ["# GPU 환경 검증 및 YAMNet 4-fold 가중치 비교", "",
             f"- 실험 완료 여부: **{'완료' if status['experiment_completed'] else '미완료'}** ({len(folds)}/12 folds)",
             f"- 최초 GPU 학습·저장·재로딩 검증: **{'통과' if status['gpu_validation_passed'] else '미통과 또는 미완료'}**",
             f"- 종료 상태: `{status['stage']}`, exit code `{status['exit_code']}`",
             f"- 시작: {status['started_at']} / 종료: {status['finished_at']} (UTC)",
             f"- 전체 경과 시간: {status['elapsed_seconds']:.1f}초",
             f"- 이 실행의 패키지 설치: 0건. 환경 프로필 `{status.get('environment_profile', 'pinned218')}`의 고정 패키지를 확인하여 사용.",
             "- 모델 성능 적합성: 개발 CV와 임시 검증 가드로만 판단하며 제품·현장 적합성을 확정하지 않음.", ""]
    if status.get("validated_report"):
        lines.extend(["## 검증된 공식 GPU 환경", "",
                      "공식 PyPI 배포 TensorFlow 2.21.0·tf-keras 2.21.0, CUDA 12.8 라이브러리·cuDNN 9.8.0.87을 설치한 `.venv-gpu-official221`을 사용했다. 이전 TensorFlow 2.18.1 후보와 별도 환경이다. TensorFlow Hub 0.16.1·NumPy 1.26.4·setuptools 80.9.0을 유지했다.", "",
                      f"선행 최소 GPU 검증 보고서: `{status['validated_report']}`. 완료·종료 코드 0, 실제 GPU 업데이트, 1 epoch·2 steps 저장/재로딩, 기존 체크포인트 CPU/GPU 수치 일치 증거를 확인했다. 해당 검증의 패키지·보호 입력·GPU 실행 코드가 동일함을 `reused_gpu_validation.json`에 기록하고 최소 학습은 반복하지 않았다. 이번 정규 CV에서는 CPU 분류기 추론이나 CPU 학습을 실행하지 않았다.", ""])
    if status.get("first_error"):
        lines.extend(["## 최초 오류와 중단 범위", "", f"`{status['first_error']}`", "",
                      "실행 오류 뒤 후속 학습은 시작하지 않았다. 자동 CPU 학습 전환, TensorFlow 버전 변경, 재시도는 하지 않았다.",
                      "`first_error.log`와 실패 단계의 원시 로그를 함께 확인한다.", ""])
    if (output / "integrity.json").is_file():
        integrity = read_json(output / "integrity.json")
        lines.extend(["## 보존 확인", "",
                      f"기존 원본·전처리·임베딩·split·설정·코드·기존 실행 기록 {integrity['protected_file_count']:,}개 파일의 SHA-256 일치: **{integrity['protected_hashes_unchanged']}**.",
                      f"기존/별도 환경 패키지 목록 일치: **{integrity['package_lists_unchanged']}**.",
                      f"고정 test 임베딩: **{integrity['fixed_test_embeddings']}개**. 고정 test 특징 추출·평가, refit, DS-CNN, 보드 실행은 하지 않았다.", ""])
    if weights:
        lines.extend(["## 가중치별 비교", "",
                      "국내 recall은 `fire_alarm`과 `doorbell`의 source-group recall 평균이다. 표준편차는 4-fold 표본 표준편차(ddof=1)이며, 신뢰구간이 아니다. precision과 macro-F1도 국내 source-group 기준이다.", "",
                      "| 가중치 | recall 평균 ± SD | 국내 OOF recall | 국내 OOF precision | 국내 OOF macro-F1 | OOF 도어락 FP | 가드 통과 | 학습 시간 |",
                      "|---|---|---|---|---|---|---|---|"])
        for row in weights:
            lines.append(f"| {row['weight']} | {row['domestic_target_recall_mean']:.4f} ± {row['domestic_target_recall_std_ddof1']:.4f} | {row['domestic_oof_target_macro_recall']:.4f} | {row['domestic_oof_target_macro_precision']:.4f} | {row['domestic_oof_macro_f1']:.4f} | {row['domestic_oof_doorlock_fp']}/{row['domestic_oof_doorlock_count']} | {row['guards_passed_folds']}/4 | {row['elapsed_seconds']:.1f}초 |")
        lines.extend(["", "## Fold별 선택 결과", "",
                      "| 가중치 | fold | 선택/실행 epoch | 국내 recall | 국내 precision | 국내 macro-F1 | 해외 other FP | 국내 도어락 FP | 가드 |",
                      "|---|---|---|---|---|---|---|---|---|"])
        for row in folds:
            lines.append(f"| {row['weight']} | {row['fold']} | {row['selected_epoch']}/{row['epochs_completed']} | {row['domestic_target_macro_recall']:.4f} | {row['domestic_target_macro_precision']:.4f} | {row['domestic_macro_f1']:.4f} | {row['overseas_other_fp']}/{row['overseas_other_count']} | {row['domestic_doorlock_fp']}/{row['domestic_doorlock_count']} | {'충족' if row['guards_passed'] else '미충족'} |")
        lines.extend(["", "가드 미충족은 실행 오류가 아니다. 임시 조건은 해외 other FPR ≤ 0.05, 국내 도어락 오탐 0개이며, 이를 만족하지 못한 fold도 기존 checkpoint 선택 규칙으로 저장했다.", "",
                      "## 개선의 일관성과 오탐 변화", ""])
        for weight in (2, 4):
            rows = [r for r in folds if r["weight"] == weight]
            if len(rows) == 4:
                recall = [r["delta_vs_w1_domestic_target_macro_recall"] for r in rows]
                fp = [r["delta_vs_w1_overseas_other_fp"] for r in rows]
                doorlock = [r["delta_vs_w1_domestic_doorlock_fp"] for r in rows]
                lines.append(f"- {weight}배 대 1배: recall 개선/동일/하락 {sum(x > 0 for x in recall)}/{sum(x == 0 for x in recall)}/{sum(x < 0 for x in recall)} folds. Fold 0~3 recall 차이 {recall}, 해외 other FP 차이 {fp}, 국내 도어락 FP 차이 {doorlock}.")
        lines.extend(["", "## 혼동행렬", "", "행=정답, 열=예측, 클래스 순서 `[fire_alarm, doorbell, other]`. 아래는 국내 source-group 기준이다. 전체 validation group 행렬과 클래스별 precision/recall 분자·분모는 `comparison_folds.csv`에 함께 보관한다.", ""])
        for row in folds:
            lines.append(f"- w{row['weight']} fold {row['fold']}: `{row['domestic_confusion_matrix']}`")
        for row in weights:
            lines.append(f"- w{row['weight']} 국내 OOF: `{row['domestic_oof_confusion_matrix']}`")
        lines.append("")
    lines.extend(["## 조건·증거·한계", "",
                  "가중치 1→2→4, 각 fold 신규 분류기·Adam, seed=42+fold, 최대 30 epoch, patience 8, batch 30, 30 steps/epoch, learning rate 0.001. 모델 구조·샘플링·검증 가드·분할·임베딩은 기존 코드를 사용했다.", "",
                  "GPU 보조 실행기는 Dense, gradient 생성, Adam 연산을 `/GPU:0`에 배치하고 soft device placement를 끈다. 데이터 읽기·샘플링·메트릭·파일 저장은 호스트에서 처리한다. 선행 검증의 기존 체크포인트 CPU/GPU 출력 비교 허용오차는 rtol=1e-4, atol=1e-5다. 각 정규 fold의 저장된 분류기를 GPU에서 재로딩하고 검증 메트릭·혼동행렬 일치를 확인했다.", "",
                  "공통 ESC-50 validation 398개는 여러 fold에서 재사용하므로 독립 OOF 표본 1,592개로 합산하지 않는다. 국내 OOF만 각 국내 개발 그룹의 한 번 검증 예측을 모은다. 작은 국내 개발 표본, 단일 seed, 같은 해외 validation을 사용하는 checkpoint 선택에 따른 한계가 있다. 정규 비교에서 1 epoch·2 steps GPU 검증 실행은 제외한다.", "",
                  "최종 가중치 확정과 refit은 다음 단계로 남긴다. 실제 환경의 false alarms/hour, 이벤트 지연, 고정 test 성능과 보드 동작은 이 결과로 검증되지 않는다.", "",
                  "- `packages_*`, `package_comparison.json`: 전체 패키지·동일성", 
                  "- `gpu_info.csv`, `environment.json`, `execution_events.json`, `planned_commands.json`: GPU·환경·실행 명령·시간·종료 상태",
                  "- `reused_gpu_validation.json` 및 참조된 선행 보고서: 순전파·역전파·업데이트·CPU/GPU 수치·경로 정규화 사본과 텐서 해시·저장/재로딩 증거",
                  "- `cv_weight_*.log`, `weight_*/device_evidence.json`, 각 fold의 `gpu_execution.json`: 학습 로그·GPU 배치·fold 시간",
                  "- `protected_before.json`, `protected_after.json`, `integrity.json`, `cache_*.json`: 보존·고정 test 0개 확인", "",
                  "GPU 호환성 판단은 실제 학습·업데이트·저장/재로딩 증거에 한정한다. TensorFlow 2.21.0 wheel의 빌드 CUDA 메타데이터와 실제 가상환경에서 로드한 CUDA 12.8 라이브러리를 구분하며, 상세 정보는 `weight_*/tensorflow_gpu.json`과 `weight_*/loaded_cuda_libraries.json`에 있다.", ""])
    (output / "REPORT_KO.md").write_text("\n".join(lines), encoding="utf-8")
