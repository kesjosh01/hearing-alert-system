"""Local Korean report for the separately frozen distillation comparison."""
from __future__ import annotations

import argparse
import html
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.common import source
from improvement.protocol import REPORT
from render_training_report import curve, rows


def read(path, fallback=None):
    return source.read_json(path) if Path(path).is_file() else fallback


def pct(value):
    return f"{(Decimal(str(value)) * 100).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}%"


def render(final=False):
    verification = read(REPORT / "FINAL_VERIFICATION.json")
    if final:
        from audit_improvement import audit
        verification = audit()
    status = read(REPORT / "status.json", {})
    decision = read(REPORT / "decision.json", {})
    revision = {r["evaluation"]: r for r in read(REPORT / "evaluation_scope_v2/results.json", [])}
    revision_note = ("사용자 요청으로 해외 siren을 비목표 소리의 오탐 평가에서 제외했다. "
                     "test의 8개, validation의 8개 전체를 예측과 무관하게 분자·분모에서 제외하며 모델·임계값은 변경하지 않았다. "
                     "원래 평가 JSON과 모델 묶음은 보존했다. [추가 데이터·epoch 판단과 최신 평가 정의](evaluation_scope_v2/REPORT_KO.md)를 함께 확인한다.")
    summaries = [source.read_json(p) for p in sorted(REPORT.glob("*_cv_summary.json"))]
    markdown = ["# YAMNet distillation과 DS-CNN 성능 개선", "",
                f"상태: `{status.get('stage')}`. 추가 학습 fold: {status.get('completed_folds', 0)}/16.", "",
                "이전 DS-CNN 학습은 파일 라벨 cross entropy만 사용했다. YAMNet은 구간 검수 후보를 제안하는 데 사용됐으며, 이전 학습에 distillation은 적용되지 않았다.", "",
                "개발 음원 1,640개·1초 창 15,502개를 사용했다. 국내 OOF는 화재경보 13개·초인종 21개·도어락 5개 source group이며, 해외 validation은 fold 간 재사용한다. 기존 test는 411개(타깃 8개, other 403개)다.", "",
                "이번 비교는 기존 YAMNet backbone과 각 fold의 학습된 3-class head를 고정 teacher로 사용한다. teacher의 train 그룹이 student의 validation과 겹치지 않는지 확인했다. 각 student 1초 창과 같은 시작의 15,600 samples로 teacher 입력을 정렬했다. teacher의 pre-softmax logits을 temperature 2로 softmax한 확률을 사용한다.", "",
                "손실은 0.5 × 파일 라벨 CE(창 평균 확률) + 0.5 × T² × 파일별 균등 평균 KL(teacher_T || student_T)이다. 사람의 시간 구간 주석이나 창별 정답을 만들지 않았다. teacher는 업데이트하지 않고 student만 학습한다. 파일별 손실은 균등하지만 BN 통계에는 microbatch의 창 분포 영향이 남는다.", "",
                "기존 24채널 모델의 distillation, 주파수 정보를 남기는 시간축 pooling, 최대 Mel 4개·시간 10 frame의 마스킹, 48채널 모델을 단계적으로 비교한다. 클래스별 판정 임계값은 개발 OOF·기존 해외 validation에서만 고른다. 기존 해외 other 오탐률 5% 이하와 도어락 오탐 0 가드를 유지한다.", "",
                "| 후보 | 4개 fold 타깃 recall | 평균 ± 표준편차 | fire / bell 임계값 | 파라미터 |",
                "|---|---|---|---|---|"]
    table = []
    plots = []
    for summary in summaries:
        name = summary["candidate"]
        folds = ", ".join(pct(f["domestic_target_recall"]) for f in summary["folds"])
        thresholds = summary["decision_policy"]["thresholds"]
        mean_std = f"{pct(summary['domestic_target_recall_mean'])} ± {100 * summary['domestic_target_recall_std']:.2f}%p"
        threshold_text = f"{thresholds['fire_alarm']} / {thresholds['doorbell']}"
        markdown.append(f"| {name} | {folds} | {mean_std} | {threshold_text} | {summary['parameters']:,} |")
        highlight = ' style="background:#e4f2e9"' if name == decision.get("candidate") else ""
        table.append(f"<tr{highlight}><td>{html.escape(name)}</td><td>{folds}</td><td>{mean_std}</td><td>{threshold_text}</td><td>{summary['parameters']:,}</td></tr>")
        if name.startswith("kd"):
            for fold in summary["folds"]:
                path = Path(fold["run"])
                history = rows(path / "history.csv")
                plots.append(f"<article><h2>{html.escape(name)} · fold {fold['fold']}</h2><div class=curves>" +
                             curve(history, "hard_label_loss", "파일 라벨 손실") +
                             curve(history, "distillation_loss_T_squared", "T²를 적용한 distillation KL") +
                             curve(history, "domestic_target_recall", "개발 타깃 recall", 1) + "</div></article>")
    markdown += ["", "baseline_w1/2/4는 보존된 모델의 공통 임계값을 클래스별 임계값으로 재보정한 대조군이다. 해당 checkpoint는 이전 학습에서 선택됐으므로 KD만의 효과를 완전히 분리한 비교는 아니다. 표의 성능은 모델·임계값 선택에도 쓰인 작은 개발 자료의 수치이며 독립 일반화 성능으로 해석하지 않는다.", ""]
    teacher = read(REPORT / "teacher_oof_benchmark.json")
    if teacher:
        markdown += [f"동일한 정렬 창에서 frozen fold teacher의 개발 평균 타깃 recall은 {pct(teacher['mean_domestic_target_recall'])}다. 기존 YAMNet의 native patch 평가와는 창·신규 데이터가 다르며 student의 상한을 뜻하지 않는다.", ""]
    if decision:
        markdown += [f"선택: `{decision['candidate']}`. 최종 모델에 distillation 적용: `{decision['distillation_selected']}`. 고정 refit: {decision['refit_epochs']} epoch. 판정 임계값: `{decision['decision_policy']['thresholds']}`.", ""]
    metric_html = []
    if revision:
        markdown += [revision_note, ""]
        metric_html.append("<p>" + html.escape(revision_note.split(" [추가 데이터")[0]) +
                           ' <a href="evaluation_scope_v2/REPORT_KO.md">추가 데이터·epoch 판단과 평가 수정</a></p>')
    for name in ("float", "int8"):
        metrics = read(REPORT / f"historical_holdout/{name}_metrics.json")
        if not metrics:
            continue
        classes = metrics["domestic_groups"]["classes"]
        revised = revision.get(f"improved_{name}_historical")
        negative_text = (f"해외 비목표 소리 오탐 {revised['false_positives']}/{revised['eligible_negative_clips']} "
                         f"({pct(revised['fpr'])}, siren {revised['excluded_clips']}개 제외); "
                         f"원래 정의 {metrics['overseas_other_clip_false_positives']}/{metrics['overseas_other_clip_count']}"
                         if revised else f"해외 other 오탐 {metrics['overseas_other_clip_false_positives']}/{metrics['overseas_other_clip_count']}")
        text = (f"{name}: 기존 test 타깃 recall {pct(metrics['domestic_target_macro_recall'])}; "
                f"fire_alarm {classes['fire_alarm']['correct']}/{classes['fire_alarm']['support']}, "
                f"doorbell {classes['doorbell']['correct']}/{classes['doorbell']['support']}; "
                f"{negative_text}; "
                f"국내 도어락 오탐 {metrics['domestic_doorlock_false_positive_groups']}/{metrics['domestic_doorlock_group_count']}.")
        markdown += [text, ""]
        metric_html.append("<p>" + html.escape(text) + "</p>")
    parity = read(REPORT / "historical_holdout/float_int8_comparison.json")
    if parity:
        markdown += [f"Float/int8 파일 판정 일치율 {pct(parity['clips']['declared_decision_agreement'])}, 최대 확률 차이 {parity['clips']['probability_max_abs']:.6f}.", ""]
    old_test = read(ROOT / "reports/real_training_20261114/study_calibrated_v2_resumed/historical_holdout/int8_metrics.json")
    new_test = read(REPORT / "historical_holdout/int8_metrics.json")
    if old_test and new_test:
        old_negative = revision.get("baseline_int8_historical", {"false_positives": old_test["overseas_other_clip_false_positives"], "eligible_negative_clips": old_test["overseas_other_clip_count"], "fpr": old_test["overseas_other_clip_fpr"]})
        new_negative = revision.get("improved_int8_historical", {"false_positives": new_test["overseas_other_clip_false_positives"], "eligible_negative_clips": new_test["overseas_other_clip_count"], "fpr": new_test["overseas_other_clip_fpr"]})
        tradeoff = (f"기존 test int8 기준 재현율은 {pct(old_test['domestic_target_macro_recall'])} → {pct(new_test['domestic_target_macro_recall'])}로 높아졌지만, "
                    f"해외 비목표 소리 오탐은 {old_negative['false_positives']}/{old_negative['eligible_negative_clips']} → "
                    f"{new_negative['false_positives']}/{new_negative['eligible_negative_clips']} "
                    f"({pct(new_negative['fpr'])})로 늘었다. 개발 fold의 가드는 통과했지만 기존 test의 오탐률은 5%보다 높다. "
                    "모든 감지 지표가 개선됐다고 볼 수 없으며 실사용 교체·배포를 권장하지 않는다. 결과는 개선 실험 모델로 보존한다.")
        markdown += [tradeoff, ""]
        metric_html.insert(0, "<p><strong>" + html.escape(tradeoff) + "</strong></p>")
    markdown += ["기존 test는 이미 YAMNet과 이전 DS-CNN에서 관측했다. 추가 탐색 중에는 읽지 않았으며 선택·refit·양자화 모델 해시를 고정한 뒤 비교한다. 이 결과는 재사용 자료의 회고적 비교이며 새로운 독립 시험이 아니다. test를 보고 다시 튜닝하지 않는다.", ""]
    if verification:
        markdown += [f"실제 distillation CV {verification['actual_distillation_cv_runs']}회, optimizer update {verification['actual_distillation_cv_optimizer_steps']:,}회를 검증했다. 선택 모델의 int8 파일은 {verification['trained_int8_bytes']:,} bytes다. 원본·분할·YAMNet·기존 DS-CNN 모델과 환경을 보존했다.", ""]
    offline = read(REPORT / "offline_bundle_check/status.json")
    if offline and offline.get("passed"):
        markdown += ["신규 초인종의 저장 파일 추론 확인(학습 자료이며 독립 평가가 아님):", "",
                     "| ID | float 판정 | int8 판정 |", "|---|---|---|"]
        markdown += [f"| {r['sample_id']} | {r['float_label']} | {r['int8_label']} |" for r in offline["records"]]
        markdown += ["", "CLI 정상 실행과 올바른 분류를 구분한다.", ""]
        count = sum(r["int8_label"] == "doorbell" for r in offline["records"])
        markdown += [f"신규 초인종 {count}/{len(offline['records'])}개를 doorbell로 판정했다. 초기 baseline은 같은 학습 자료에서 2/5개였으며, 학습 파일 확인 결과를 독립 평가 정확도로 해석하지 않는다.", ""]
    markdown += ["새 프로젝트·환경·패키지를 설치하지 않았다. 원음은 외부에 업로드하지 않았고 W&B 대신 로컬 기록을 사용했다. 시간 구간 검수·실시간 마이크·ESP-IDF 전체 빌드·보드 RAM/지연·시간당 오탐·통합 알림은 미검증이다. `deployment_ready=false`다.", "",
                 "방법 근거: [Hinton et al.의 distillation](https://arxiv.org/abs/1503.02531), [Keras 공식 구현](https://keras.io/examples/vision/knowledge_distillation/), [TensorFlow YAMNet 전이학습](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio), [SpecAugment](https://arxiv.org/abs/1904.08779). 이 문헌의 성능 수치를 우리 데이터 결과로 옮겨 쓰지 않았다.", ""]
    (REPORT / "REPORT_KO.md").write_text("\n".join(markdown), encoding="utf-8")
    intro = """<article><h2>학습 방식과 해석</h2><p>이전에는 distillation이 없었습니다. 이번에는 각 fold 검증 그룹을 학습하지 않은 YAMNet teacher의 정렬 logits을 사용해 실제 KL 손실로 student를 학습합니다. 시간 구간 검수로 기록하지 않습니다.</p><p>정답 파일 CE 0.5 + T² KL 0.5, T=2입니다. 주파수 정보 보존·가벼운 SpecAugment·채널 수를 단계적으로 비교합니다. 모든 선택은 개발 자료로 수행하며 기존 test는 독립 시험이 아닙니다.</p></article>"""
    body = f"""<!doctype html><html lang=ko><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>DS-CNN distillation 개선</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#eff4f8;color:#183148}}main{{max-width:1200px;margin:auto;padding:24px}}article{{background:white;border-radius:10px;padding:20px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:left;border-bottom:1px solid #dde6ec;padding:8px}}.scroll{{overflow-x:auto}}.curves{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}}svg{{width:100%;font-size:12px}}</style><main><h1>YAMNet distillation과 DS-CNN 성능 개선</h1><p>{html.escape(status.get('stage',''))} · {status.get('completed_folds',0)}/16 fold · 선택 {html.escape(decision.get('candidate','진행 중'))}</p>{intro}<article class=scroll><h2>개발 4-fold 비교</h2><table><tr><th>후보</th><th>fold별 타깃 recall</th><th>평균 ± 표준편차</th><th>fire / bell 임계값</th><th>파라미터</th></tr>{''.join(table)}</table><p>기존 checkpoint의 클래스별 임계값 재보정과 KD 비교에는 checkpoint 선택 방식의 차이도 포함됩니다. 작은 개발 자료에서 선택한 수치이며 독립 일반화 성능을 입증하지 않습니다.</p></article><article><h2>고정 모델의 기존 test 회고적 비교</h2>{''.join(metric_html)}<p>이 자료는 이미 관측했습니다. 새로운 독립 시험이 아니며 이 결과로 모델·임계값을 다시 선택하지 않습니다.</p></article>{''.join(plots)}</main></html>"""
    (REPORT / "TRAINING_REPORT.html").write_text(body, encoding="utf-8")
    print("DISTILLATION_IMPROVEMENT_REPORT", REPORT / "REPORT_KO.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    render(parser.parse_args().final)
