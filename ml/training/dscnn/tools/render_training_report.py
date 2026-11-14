"""Render local training curves and audit actual artifacts without model execution."""
from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.common import config, source, write_json
from dscnn_baseline.data import development_identity
from dscnn_baseline.study import code_identity
from dscnn_baseline import decisions


def read(path, fallback=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def rows(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def curve(history, field, title, ceiling=None):
    points = []
    for row in history:
        try:
            points.append((int(row["epoch"]), float(row[field])))
        except (TypeError, ValueError, KeyError):
            pass
    if not points:
        return ""
    upper = ceiling or max(.01, max(v for _, v in points) * 1.05)
    last = max(e for e, _ in points)
    coordinates = [(40 + (e - 1) / max(1, last - 1) * 490, 150 - v / upper * 120) for e, v in points]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
    dots = "".join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5"><title>epoch {epoch}: {value:.6f}</title></circle>'
                   for (epoch, value), (x, y) in zip(points, coordinates))
    return f'''<figure><figcaption>{html.escape(title)}</figcaption><svg viewBox="0 0 560 180" role="img" aria-label="{html.escape(title)}">
<path d="M40 20V150H540" stroke="#8996a2" fill="none"/><text x="4" y="30">{upper:.3f}</text><text x="20" y="154">0</text>
<text x="40" y="174">epoch 1</text><text x="475" y="174">{last}</text><polyline points="{polyline}" fill="none" stroke="#0874aa" stroke-width="2"/>
<g fill="#0874aa">{dots}</g></svg></figure>'''


def audit(study):
    status = read(study / "status.json", {})
    if status.get("stage") != "complete" or status.get("exit_code") != 0:
        raise ValueError("The actual study has not completed")
    identity = development_identity()
    decision = read(study / "decision.json")
    summary = read(study / f"w{decision['selected_weight']}_cv_summary.json")
    refit = Path(status["refit"])
    refit_status, run = read(refit / "status.json"), read(refit / "run.json")
    frozen = read(study / "frozen_model.json")
    holdout_status = read(study / "historical_holdout/status.json")
    verification = {
        "kind": "dscnn_model_development_and_training_audit",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_development_and_training_complete": False,
        "event_intervals_reviewed": False,
        "independent_microphone_validation_complete": False,
        "esp_idf_cross_compile_verified": False,
        "board_executions": 0, "deployment_ready": False,
        "checks": {}, "limitations": [
            "File labels supervise mean window probabilities. Individual event timestamps and window detection accuracy are unverified.",
            "Old YAMNet test was reused for one frozen retrospective comparison, not a fresh independent test.",
            "New domestic audio and different supervision/training units prevent a controlled architecture-only comparison to YAMNet.",
            "Board RAM, latency, real microphone streams, false alarms/hour and end-to-end alerts remain unmeasured."]}
    checks = verification["checks"]
    checks["study_completed_12_folds"] = status["completed_folds"] == 12 and status["completed_weights"] == [1, 2, 4]
    checks["effective_data_identity_unchanged"] = frozen["development_identity"] == identity == run["development_identity"] == summary["development_identity"]
    checks["file_label_supervision_truthful"] = run["supervision"]["mode"] == "file_labels" and run["supervision"]["event_intervals_used"] is False and run["annotations"] is None
    checks["all_selected_fold_guards_passed"] = all(f["validation_guards_passed"] is True for f in summary["folds"])
    checks["decision_policy_frozen_from_development"] = frozen["decision_policy"] == summary["decision_policy"] and decision["decision_threshold"] == refit_status["selected_threshold"] == summary["decision_policy"]["threshold"]
    all_folds = [f for weight in (1, 2, 4) for f in read(study / f"w{weight}_cv_summary.json")["folds"]]
    final_runs = [Path(f["run"]).resolve() for f in all_folds] + [refit.resolve()]
    plan = read(study / "execution_plan.json")
    checks["frozen_training_code_and_configuration_unchanged"] = (
        plan["code_identity"] == code_identity() and plan["config"] == config()[0] and
        all(read(path / "run.json")["code_sha256"] == plan["code_identity"] and
            read(path / "run.json")["config"] == plan["config"] for path in final_runs))
    checks["all_cv_checkpoints_match_recorded_hashes"] = all(source.sha256(Path(f["run"]) / "classifier.keras") == f["model_sha256"] for f in all_folds)
    checks["every_training_run_used_gpu"] = all(
        read(Path(f["run"]) / "status.json")["device"] == "gpu" and
        all("GPU:0" in d for d in read(Path(f["run"]) / "status.json")["model_variable_devices"])
        for f in all_folds) and refit_status["device"] == "gpu" and all(
            "GPU:0" in device for device in refit_status["model_variable_devices"])
    checks["common_threshold_metrics_match_prediction_files"] = all(
        abs(decisions.report(rows(Path(f["run"])/"validation_clips.csv"),f["common_threshold"])["domestic_target_macro_recall"]-f["domestic_target_recall"])<1e-10 and
        abs(decisions.report(rows(Path(f["run"])/"validation_clips.csv"),f["common_threshold"])["overseas_other_clip_fpr"]-f["overseas_other_fpr"])<1e-10
        for f in all_folds)
    _, protocol = config()
    base_folds, _ = source.load_splits(protocol)
    test_ids = {r["sample_id"] for r in base_folds[0] if r["split"] == "test"}
    train_rows = rows(refit / "used_train_validation.csv")
    cv_rows = [rows(Path(f["run"]) / "used_train_validation.csv") for f in all_folds]
    checks["test_excluded_from_all_training_and_validation"] = all(not test_ids.intersection(r["sample_id"] for r in data) for data in [train_rows, *cv_rows])
    expected_epochs = max(1, int(statistics.median(f["selected_epoch"] for f in summary["folds"]) + .5))
    checks["refit_epochs_fixed_from_cv_median"] = expected_epochs == decision["refit_epochs"] == run["options"]["epochs"] == refit_status["epochs_completed"]
    checks["refit_optimizer_steps_match_fixed_budget"] = refit_status["optimizer_steps"] == expected_epochs * run["options"]["steps_per_epoch"]
    exposure = {r["sample_id"]: int(r["draws"]) for r in rows(refit / "sample_exposure.csv")}
    new_ids = {f"kr-user-db-{i:02}" for i in range(6, 11)}
    checks["all_five_new_doorbells_actually_sampled_in_refit"] = all(exposure.get(s, 0) > 0 for s in new_ids)
    calibration = read(refit / "export/calibration_provenance.json")
    training_ids = {r["sample_id"] for r in train_rows if r["split"] == "train"}
    checks["int8_calibration_train_only"] = bool(calibration) and all(r["split"] == "train" and r["sample_id"] in training_ids for r in calibration)
    metadata = read(refit / "export/model_metadata.json")
    checks["trained_int8_artifact"] = metadata["pipeline_only"] is False and metadata["calibration_role"] == "train" and metadata["model_sha256"] == source.sha256(refit / "export/dscnn_int8.tflite")
    checks["int8_operator_and_tensor_contract"] = set(metadata["all_tensor_types"]) <= {"int8", "int32"} and metadata["input"]["shape"] == [1,98,40,1] and metadata["output"]["shape"] == [1,3]
    frontend_host = read(study.parent / "trained_frontend_host/status.json", {})
    checks["trained_int8_frontend_host_parity_passed"] = (
        frontend_host.get("passed") is True and frontend_host.get("exit_code") == 0 and
        frontend_host.get("host_only") is True and frontend_host.get("board_execution") is False and
        len(frontend_host.get("signals", [])) == 27 and
        all(r.get("int8_model_output_identical") is True for r in frontend_host.get("signals", [])) and
        frontend_host.get("quantization") == {"scale": metadata["input"]["scale"], "zero_point": metadata["input"]["zero_point"]})
    checks["model_frozen_before_and_after_holdout"] = source.sha256(refit / "classifier.keras") == frozen["model_sha256"] == holdout_status["model_sha256_before"] == holdout_status["model_sha256_after"] and source.sha256(refit / "export/dscnn_int8.tflite") == frozen["int8_sha256"] == holdout_status["int8_sha256_after"]
    checks["historical_holdout_compared_once_and_labelled_correctly"] = holdout_status["exit_code"] == 0 and holdout_status["test_evaluation_invocations"] == 1 and holdout_status["evaluated_clips"] == len(test_ids) and holdout_status["is_fresh_independent_test"] is False
    integrity = read(study.parent / "integrity.json", {})
    checks["originals_splits_yamnet_and_environments_preserved"] = integrity.get("unchanged") is True
    before = read(study.parent / "dscnn_packages_before.json")
    packages = sorted([{"name":d.metadata["Name"],"version":d.version} for d in distributions()],key=lambda d:d["name"].lower())
    checks["dscnn_environment_packages_unchanged"] = packages == before
    checks["software_unit_tests_passed"] = "\nOK\n" in (study.parent / "calibrated_tests_final.log").read_text()
    verification["new_doorbell_refit_draws"] = {s: exposure.get(s, 0) for s in sorted(new_ids)}
    verification["final_study_real_audio_training_executions"] = len(all_folds) + 1
    earlier = [p.resolve() for p in sorted((ROOT / "runs").glob("dscnn_*20261114*"))
               if (p / "history.csv").is_file() and p.resolve() not in final_runs]
    verification["earlier_development_runs_preserved"] = [str(p) for p in earlier]
    verification["earlier_run_statuses"] = {
        str(p): {"epochs_recorded": len(rows(p / "history.csv")),
                 "exit_code": read(p / "status.json", {}).get("exit_code")}
        for p in earlier}
    verification["completed_cv_folds_reused_after_reboot"] = status.get("reused_folds", 0)
    verification["new_training_executions_after_reboot"] = status.get("new_training_executions")
    verification["real_audio_training_executions"] = len(all_folds) + 1 + len(earlier)
    verification["optimizer_steps"] = sum(read(Path(f["run"]) / "status.json")["optimizer_steps"] for f in all_folds) + refit_status["optimizer_steps"]
    verification["trained_model"] = str(refit / "classifier.keras")
    verification["trained_int8_model"] = str(refit / "export/dscnn_int8.tflite")
    verification["trained_int8_bytes"] = (refit / "export/dscnn_int8.tflite").stat().st_size
    holdout_metrics = read(study / "historical_holdout/int8_metrics.json")
    verification["historical_domestic_target_recall"] = holdout_metrics["domestic_target_macro_recall"]
    verification["performance_note"] = (
        "Training and artifact checks do not establish practical detection readiness. "
        "Report target misses and the small, historically reused holdout separately from completion.")
    verification["model_development_and_training_complete"] = all(checks.values())
    write_json(study.parent / "FINAL_VERIFICATION.json", verification)
    if not verification["model_development_and_training_complete"]:
        raise ValueError(f"Completion audit failed: {[k for k,v in checks.items() if not v]}")
    return verification


def render(study, final=False):
    study = Path(study).resolve()
    status = read(study / "status.json", {})
    verification = audit(study) if final else None
    summaries = [read(study / f"w{weight}_cv_summary.json") for weight in (1,2,4)]
    decision = read(study / "decision.json", {})
    tables, plots = [], []
    for summary in summaries:
        if not summary:
            continue
        for fold in summary["folds"]:
            tables.append(f'<tr><td>{summary["domestic_weight"]}</td><td>{fold["fold"]}</td><td>{fold["selected_epoch"]}</td><td>{percent(fold["domestic_target_recall"])}</td><td>{percent(fold["overseas_other_fpr"])}</td><td>{fold["domestic_doorlock_fp"]}</td><td>{fold["validation_guards_passed"]}</td></tr>')
            path = Path(fold["run"])
            history = rows(path / "history.csv")
            plots.append(f'<article><h2>{html.escape(path.name)}</h2><div class="curves">' +
                         curve(history,"train_loss","学習損失 / file-level cross entropy") +
                         curve(history,"domestic_target_recall","국내 타깃 그룹 recall",1) + '</div></article>')
    if not plots:
        plan=read(study/"execution_plan.json",{})
        available = set((ROOT / "runs").glob(plan.get("run_prefix","dscnn_calibrated_20261114")+"*"))
        available.update(Path(p) for p in plan.get("reused_completed_folds", {}).values())
        for path in sorted(available):
            history = rows(path / "history.csv")
            if history:
                plots.append(f'<article><h2>{html.escape(path.name)}</h2>'+curve(history,"train_loss","파일 단위 학습 손실")+'</article>')
    metrics = {name:read(study / f"historical_holdout/{name}_metrics.json") for name in ("float","int8")}
    parity = read(study / "historical_holdout/float_int8_comparison.json")
    offline = read(study.parent / "offline_bundle_check/status.json")
    metric_rows=[]
    for name,m in metrics.items():
        if m:
            metric_rows.append(f'<tr><td>{name}</td><td>{percent(m["domestic_target_macro_recall"])}</td><td>{m["domestic_groups"]["macro_f1"]:.4f}</td><td>{m["overseas_other_clip_false_positives"]}/{m["overseas_other_clip_count"]}</td><td>{m["domestic_doorlock_false_positive_groups"]}/{m["domestic_doorlock_group_count"]}</td></tr>')
    headline="모델 개발·실제 음원 학습 완료" if verification else "모델 학습 진행 기록"
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DS-CNN 학습 결과</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#172d42;background:#eff4f8;margin:0}}main{{max-width:1150px;margin:auto;padding:28px}}article{{padding:22px;margin:20px 0;background:white;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #dce5ee;text-align:left}}.curves{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}}figure{{margin:10px}}svg{{width:100%;font-size:12px}}code{{overflow-wrap:anywhere}}</style><main>
<h1>{headline}</h1><p>상태: <code>{html.escape(status.get('stage','not_started'))}</code> · 완료 fold {status.get('completed_folds',0)}/12 · 선택 가중치 {decision.get('selected_weight','—')}</p>
<article><strong>파일 라벨 학습</strong><p>각 파일의 모든 1초 창 확률을 평균한 뒤 파일 정답으로 손실을 계산했습니다. 1초 창별 정답이나 이벤트 시간 구간을 만들어내지 않았습니다. 원본 그룹·클래스·other pool 확률을 유지하고 그룹 안에서 파일을 균등 선택했습니다. 긴 파일은 손실에서 파일 하나의 비중을 갖습니다. BN 통계는 각 microbatch 안의 창 분포에 영향을 받습니다.</p><p>타깃 판정 임계값은 개발 OOF·기존 해외 validation에서 선택하고 refit과 test 전에 고정합니다. 원시 확률은 유지하며 선택 임계값에 못 미친 타깃을 other로 판정합니다. 기존 YAMNet과 학습 감독·예산 및 신규 초인종 5개 포함 여부가 다르므로 모델 구조만의 통제 비교가 아닙니다. 보드·실시간 마이크 검증은 별도입니다.</p></article>
<article><h2>4-fold 비교</h2><p>표는 가중치별 공통 임계값의 수치입니다. epoch 곡선은 해당 epoch에서 개발 자료로 선택한 임계값을 사용하므로 표와 판정 조건이 다를 수 있습니다.</p><table><tr><th>국내 가중치</th><th>fold</th><th>선택 epoch</th><th>국내 타깃 recall</th><th>해외 other 오탐률</th><th>국내 도어락 오탐</th><th>개발 가드</th></tr>{''.join(tables)}</table></article>
<article><h2>고정 모델의 기존 test 비교</h2><p>이 자료는 YAMNet 평가에서 이미 사용했습니다. 새로운 독립 시험이라고 해석하지 않습니다. 모델을 고정한 뒤 한 번만 비교하며 이 결과로 설정을 다시 고르지 않습니다.</p><table><tr><th>모델</th><th>국내 타깃 recall</th><th>국내 macro-F1</th><th>해외 other 오탐</th><th>국내 도어락 오탐</th></tr>{''.join(metric_rows)}</table></article>{''.join(plots)}</main></html>'''
    if metrics["int8"]:
        target = metrics["int8"]["domestic_groups"]["classes"]
        quality = (f"<article><h2>성능 해석</h2><p>기존 test의 정분류는 화재경보 "
                   f"{target['fire_alarm']['correct']}/{target['fire_alarm']['support']}, 초인종 "
                   f"{target['doorbell']['correct']}/{target['doorbell']['support']}입니다. "
                   "학습·변환 완료와 실사용 감지 성능을 구분하며, 보드 배포 준비 완료로 판단하지 않습니다. "
                   "전체 411개 중 403개가 other이므로 전체 정확도만으로 감지 품질을 판단하지 않습니다.</p></article>")
        body = body.replace("<article><strong>파일 라벨 학습</strong>", quality + "<article><strong>파일 라벨 학습</strong>")
    if offline and offline.get("passed"):
        offline_rows = "".join(
            f"<tr><td>{html.escape(r['sample_id'])}</td><td>{r['window_count']}</td><td>{html.escape(r['float_label'])}</td><td>{html.escape(r['int8_label'])}</td></tr>"
            for r in offline["records"])
        body = body.replace("</main></html>",
            "<article><h2>모델 묶음의 저장 음원 추론 확인</h2><p>신규 초인종 5개는 학습에 사용한 자료입니다. float/int8 CLI 10회가 정상 실행됐다는 기술 확인이며 독립 성능 평가가 아닙니다. 잘못 분류한 결과도 그대로 기록합니다.</p>"
            "<table><tr><th>학습 자료 ID</th><th>창 수</th><th>float 판정</th><th>int8 판정</th></tr>" + offline_rows + "</table></article></main></html>")
    body=body.replace("学習損失 / ","학습 손실 / ")
    (study.parent / "TRAINING_REPORT.html").write_text(body,encoding="utf-8")
    markdown=[f"# {headline}","",f"상태: `{status.get('stage')}`. 완료 fold: {status.get('completed_folds',0)}/12.","",
              "사용자가 직접 청취한 파일 라벨로 학습했다. 시간 구간 검수와 창별 라벨은 사용하지 않았다. 파일 안의 모든 창 평균 확률에 대해 파일 단위 cross entropy를 계산하고 6개 파일 microbatch의 기울기를 누적해 30개 파일 batch를 구성했다.","",
              "기존 test는 이미 YAMNet에서 평가한 자료이므로 새로운 독립 시험이 아니다. 모델·설정을 고정한 뒤 한 번만 비교하고 test 결과로 재선택하지 않는다.",""]
    if decision:
        markdown += [f"선택 국내 가중치: {decision['selected_weight']}. Refit epoch: {decision['refit_epochs']}. 개발 자료로 선택한 타깃 판정 임계값: {decision['decision_threshold']}.",""]
    if any(summaries):
        markdown += ["| 국내 가중치 | fold별 국내 타깃 recall | 평균 ± 표준편차 | 공통 임계값 |",
                     "|---|---|---|---|"]
        for summary in summaries:
            if summary:
                markdown.append(f"| {summary['domestic_weight']} | " +
                                ", ".join(percent(f["domestic_target_recall"]) for f in summary["folds"]) +
                                f" | {percent(summary['domestic_target_recall_mean'])} ± {100 * summary['domestic_target_recall_std']:.2f}%p | {summary['decision_policy']['threshold']} |")
        markdown += ["", "개발 가드는 각 fold의 해외 other 오탐률 5% 이하와 국내 도어락 오탐 0이다. 큰 국내 가중치는 4개 중 3개 이상 fold 개선, 나머지 fold 비악화, 타깃 precision 및 국내 macro-F1 비악화를 모두 만족할 때만 선택한다.", ""]
    for name,m in metrics.items():
        if m:
            markdown += [f"{name}: 국내 타깃 recall {percent(m['domestic_target_macro_recall'])}, 국내 macro-F1 {m['domestic_groups']['macro_f1']:.4f}, 해외 other 오탐 {m['overseas_other_clip_false_positives']}/{m['overseas_other_clip_count']}, 국내 도어락 오탐 {m['domestic_doorlock_false_positive_groups']}/{m['domestic_doorlock_group_count']}.", ""]
            classes = m["domestic_groups"]["classes"]
            markdown += ["국내 클래스별 정분류: " + ", ".join(
                f"{label} {classes[label]['correct']}/{classes[label]['support']} (recall {percent(classes[label]['recall'])})"
                for label in ("fire_alarm", "doorbell", "other")) + ".", ""]
    if parity:
        markdown += [f"Float/int8 파일 판정 일치율: {percent(parity['clips']['declared_decision_agreement'])}. 최대 파일 확률 차이: {parity['clips']['probability_max_abs']:.6f}. 양자화 차이는 관측값이며 test를 보고 허용 기준을 새로 정하거나 모델을 다시 선택하지 않았다.", ""]
    if metrics["int8"]:
        markdown += ["학습·변환 절차의 완료가 실사용 감지 성능 충족을 의미하지는 않는다. 현재 타깃 누락이 있어 성능 개선이 필요하다. 전체 411개 중 403개가 other이므로 전체 정확도를 감지 품질의 대표 수치로 사용하지 않는다. test 결과를 본 뒤 추가 튜닝이나 재학습은 진행하지 않았다.", ""]
    if verification:
        markdown += [f"실제 음원 학습 실행 총 {verification['real_audio_training_executions']}회: 최종 비교 {verification['final_study_real_audio_training_executions']}회와 보존된 초기 개발 실행을 포함한다. 최종 비교의 optimizer steps: {verification['optimizer_steps']}.",f"실제 학습 int8 모델: `{verification['trained_int8_model']}` ({verification['trained_int8_bytes']:,} bytes).","", "원본·기존 분할·YAMNet 결과·환경 패키지 보존 검증을 통과했다. 모든 신규 초인종이 refit에서 실제 추출된 기록을 확인했다.",""]
        markdown += [f"재부팅 전 정상 완료된 CV {verification['completed_cv_folds_reused_after_reboot']}개를 재사용했다. optimizer 상태가 없는 중단 fold는 원본을 보존하고 같은 seed로 새 경로에서 재학습했다. 중단 실행도 위 총 실행 횟수에 포함한다.", ""]
        markdown += [f"기존 YAMNet 보호 파일 {read(study.parent / 'integrity.json')['protected_yamnet_files']:,}개와 원음·메타데이터·분할 보호 파일 {read(study.parent / 'integrity.json')['raw_and_metadata_files']}개의 SHA256 및 환경 패키지 목록을 비교했다.", ""]
    if offline and offline.get("passed"):
        markdown += ["학습 모델 묶음의 저장 음원 CLI를 신규 초인종 5개 × float/int8로 총 10회 실행했다. SHA256, 원본 identity, 학습 전처리와 창 수 일치를 확인했다. 이 5개는 학습 자료이며 독립 평가가 아니다.", "",
                     "| 학습 자료 ID | float 판정 | int8 판정 |",
                     "|---|---|---|"]
        markdown += [f"| {r['sample_id']} | {r['float_label']} | {r['int8_label']} |" for r in offline["records"]]
        correct = sum(r["int8_label"] == "doorbell" for r in offline["records"])
        markdown += ["", f"신규 초인종 중 {correct}/{len(offline['records'])}개를 doorbell로 판정했다. 정상적인 CLI 실행과 올바른 분류를 구분한다. 기록은 `offline_bundle_check/status.json`에 있다.", ""]
    markdown += ["원음은 GitHub·Drive·W&B로 업로드하지 않았다. 새 패키지·환경·프로젝트 설치 없이 기존 DS-CNN 전용 환경을 사용했다.","", "현장 독립 마이크·이벤트 지연·시간당 오탐·ESP-IDF 전체 빌드·보드 RAM/처리 시간·통합 알림은 미검증이다. `deployment_ready=false`를 유지한다.","", "곡선과 fold별 수치: `TRAINING_REPORT.html`. 실제 명령과 설정: study의 `execution_plan.json` 및 각 run의 `run.json`. 실패와 보류된 구간 주석 방식의 기록도 보존했다."]
    (study.parent / "REPORT_KO.md").write_text("\n".join(markdown)+"\n",encoding="utf-8")
    print("LOCAL_TRAINING_REPORT",study.parent/"TRAINING_REPORT.html")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--final",action="store_true")
    args=parser.parse_args()
    render(args.study,args.final)
