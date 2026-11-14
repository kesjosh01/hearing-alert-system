"""Render local evidence, source credits and learning curves for this revision."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from expansion.protocol import CANDIDATES, DATA, REPORT, RUNS
from render_training_report import curve, rows


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def pct(value):
    return "—" if value is None else f"{(Decimal(str(value))*100).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)}%"


def render():
    dataset = read(DATA / "dataset.json")
    status = read(REPORT / "status.json", {})
    decision = read(REPORT / "decision.json", {})
    audit = read(REPORT / "FINAL_VERIFICATION.json", {})
    complete = audit.get("model_development_and_training_complete") is True
    md = ["# 공개 정답 데이터 확장·국내 가중치·모델 구조 비교", "",
          f"상태: `{status.get('stage','준비')}`. 최종 납품 검증: `{complete}`. teacher CV {status.get('completed_teacher_folds',0)}/4, student CV {status.get('completed_student_folds',0)}/8.", "",
          "## 데이터 수집과 라벨", "",
          "사용자 요청에 따라 학습용 해외 사이렌 24개를 이 버전의 `fire_alarm` 정답으로 변경했다. 개발 validation의 사이렌 8개도 같은 정답 정의로 평가한다. 이 클래스는 엄밀한 화재경보기뿐 아니라 사용자가 지정한 사이렌을 포함한다. 원본 ESC-50 manifest와 기존 모델·분할·평가 기록은 바꾸지 않았다.", "",
          "공유마당에서 CC BY 4.0 도어벨·인터폰벨 24개를 수집했다. 검색 결과에서 화재경보로 명확히 분류된 국내 자료가 부족해, CC0 해외 연기감지기·화재/대피경보 9개와 추가 도어벨 6개를 보완했다. 일반 알람·시계·전화 소리를 제목이 비슷하다는 이유로 정답에 넣지 않았다.", "",
          "국내 자료라는 표기는 한국 공개 카탈로그 출처를 뜻한다. 해당 녹음의 실제 기기 국가·장소를 확인했다고 주장하지 않는다. 새 파일 라벨의 근거는 제작자의 제목·분류이며, 사용자가 직접 청취한 기존 국내 자료와 구분한다. 사람의 이벤트 구간 검수나 창별 정답은 만들지 않았다.", "",
          f"추가 공개 음원 {dataset['accepted_new_public_files']}개, 기존과 합친 개발 음원 1,679개·1초 창 16,406개다. 최종 refit 학습 자료는 화재경보 **47개/31그룹**, 도어벨 **53개/32그룹**, other **1,181개/909그룹**이다. 새 공개 파일은 모두 학습 전용이다. 같은 저작자의 연속 작품과 같은 기기의 여러 길이·변형은 보수적으로 묶었으며 파일 수를 독립 발생원 수로 세지 않았다.", "",
          "세 클래스의 학습 추출 확률은 **각 1/3**이다. 각 타깃 클래스 안에서는 **국내 정답 source group 가중치 4 : 해외 1**로 추출한다. 손실에 국내 가중치를 다시 곱하지 않는다. 클래스 비율·그룹 가중치는 실제 sampler CSV와 CPU 검사로 검증한다. 같은 그룹 안의 파일은 균등 선택하고 파일의 모든 창 평균 확률로 파일 라벨 손실을 계산한다.", "",
          "Chrome 연결 도구 오류 뒤 사용자가 허용한 대체 공개 검색·기존 Python·ffmpeg로 수집했다. 앱·프로그램·확장 프로그램·패키지·새 환경은 다운로드하거나 설치하지 않았다. 첫 합성 GPU 호출은 기존 CUDA 라이브러리의 프로세스 경로 지정 누락으로 GPU0개에서 종료됐고, 기존 라이브러리 경로만 자식 프로세스에 지정한 뒤 GPU/int8 합성 검증을 통과했다. 설치·드라이버·영구 설정 변경은 없다. 실패 로그도 보존했다.", "",
          "## 구조 판단과 학습 방법", "",
          "기존 마지막 convolution의 국소 시간 관측 범위는 시간 평균 전 약 15 frame(165 ms)이다. 같은 24채널에서 depthwise kernel을 3×3에서 9×3으로 바꾸면 약 51 frame(525 ms)의 패턴을 볼 수 있다. 이는 커널·stride로 계산한 국소 범위이며 최종 입력은 두 모델 모두 1초다. 경보 반복음·도어벨 감쇠 패턴에 도움이 될 수 있다는 가설을 같은 데이터에서 비교한다. 새로운 구조가 무조건 좋다고 가정하지 않는다.", "",
          "| 구조 | depthwise kernel | 파라미터 | conv·dense MAC/1초 창 |", "|---|---|---:|---:|",
          "| freq24 | 3×3 | 4,707 | 2,541,600 |", "| temporal24 | 9×3 | 6,003 | 3,811,680 |", "",
          "입력 `[98,40,1]`, 주파수 정보를 남기는 49×1 pooling, 출력 3클래스와 int8 연산 종류는 같다. 큰 시간 커널은 계산량이 약 50% 늘지만 최대 단일 activation 크기는 그대로다. 실보드의 전체 RAM·지연을 측정했다는 뜻은 아니다.", "",
          "기존 teacher는 사이렌을 `other`로 배웠으므로 그대로 쓰지 않는다. 공식 frozen YAMNet backbone은 재사용하되, 각 fold의 새 학습 자료·새 라벨로 3-class head를 다시 학습한다. 같은 student의 validation group은 teacher 학습에서 제외한다. 새로 학습한 head의 정렬된 창 logits으로 student를 distillation한다.", "",
          "Student 손실은 `0.5 × CE(파일의 창 평균 확률) + 0.5 × T² × 파일별 균등 평균 KL(teacher_T || student_T)`, `T=2`다. 두 구조에 같은 SpecAugment, CV seed42+fold, learning rate0.001, 최대30 epoch, patience8, epoch당900개 파일 추출을 적용한다. 각 refit epoch는 4개 최적 epoch의 중앙값을 반올림해 먼저 고정한다.", "",
          "초기 최종 refit(seed42)에서 추가 음원 `bsb-new-0268`이 무작위 추출0회인 것을 발견해 그 모델을 보존했다. test 평가 전에 순수 메타데이터로 추출 순서를 재현하고, 학습 타깃100개가 모두 추출되는 첫 다음 정수 시드43을 고정해 같은 구조·5epoch·150 update·가중치·임계값으로 refit을 한 번 더 했다. 시드는 모델 성능으로 고르지 않았다. 실제 추출 횟수와 사전 예측 횟수를 파일별로 대조한다. [사전 변경 기록](refit_coverage_amendment.json)을 보존한다.", "",
          "선택은 기존 국내 OOF 타깃 recall을 우선하되, 네 fold 모두 해외 실제 비목표 소리의 FPR≤5%와 국내 도어락 오탐0을 통과해야 한다. 그 뒤 pooled 국내 macroF1·precision·작은 모델·낮은 FPR 순으로 비교한다. 해외 validation은 같은 390개 비목표 파일을 fold마다 재사용하므로 독립 자료4배로 합치지 않는다.", ""]
    summaries = [read(REPORT / f"{name}_cv_summary.json") for name in CANDIDATES]
    table = []
    if any(summaries):
        md += ["## 동일 확장 데이터의 4-fold 구조 비교", "",
               "| 후보 | fold0·1·2·3 타깃 recall | 평균 | 모든 개발 가드 |", "|---|---|---:|---|"]
    for summary in summaries:
        if not summary:
            continue
        scores = ", ".join(pct(f["domestic_target_recall"]) for f in summary["folds"])
        guard = all(f["validation_guards_passed"] for f in summary["folds"])
        md.append(f"| {summary['candidate']} | {scores} | {pct(summary['domestic_target_recall_mean'])} | {guard} |")
        table.append(f"<tr><td>{summary['candidate']}</td><td>{scores}</td><td>{pct(summary['domestic_target_recall_mean'])}</td><td>{guard}</td></tr>")
    if decision:
        md += ["", f"선택: **{decision['candidate']}**, refit **{decision['refit_epochs']} epoch**, 임계값 **{decision['decision_policy']['thresholds']}**. 선택된 후보·임계값·모델 해시를 test 비교 전에 고정했다.", ""]
        md += ["두 구조의 개발 평균 차이는0.63%p로 작다. 같은 자료로 모델과 임계값을 선택했으므로 이 차이를 독립 일반화 성능의 확정적 향상으로 해석하지 않는다.", ""]
        frozen = read(REPORT / "frozen_model_coverage.json")
        if frozen:
            refit_validation = read(Path(frozen["refit"]) / "validation_metrics.json")
            md += [f"**최종 고정 float refit의 validation 해외 비목표 오탐은 {refit_validation['overseas_other_clip_false_positives']}/{refit_validation['overseas_other_clip_count']} ({pct(refit_validation['overseas_other_clip_fpr'])})로5% 목표를 넘었다.** 위의 네 CV checkpoint가 가드를 통과한 것과, 전체 개발 자료로 다시 학습한 최종 모델의 가드 통과는 다르다. 전달 모델이 이 개발 가드를 통과했다고 주장하지 않는다.", ""]
    teacher = read(REPORT / "teacher_cv_summary.json")
    if teacher:
        md += [f"새 YAMNet head의 개발 평균 타깃 recall은 {pct(teacher['domestic_target_recall_mean'])}다. Teacher의 4-fold 학습 뒤 refit epoch는 {teacher['refit_epochs']}회로 고정했다. 이 수치는 작은 개발 자료에서 선택한 결과이며 student의 성능 상한을 뜻하지 않는다.", ""]
    metric_html = []
    historical = read(REPORT / "historical_holdout/int8_metrics.json")
    if historical:
        md[4:4] = ["**재학습과 산출물 검증의 완료 여부와 성능 개선을 구분한다. 기존 test의 int8 평균 타깃 recall은62.5%로 유지됐고, 해외 비목표 오탐은24/394(6.09%)→46/394(11.68%)로 증가했다. 최종 float refit validation의 오탐률12.56%도5% 목표를 넘었다. 새 모델의 실사용 교체를 권장하지 않는다.**", ""]
        metric_html.append("<p><strong>기존 test에서 평균 recall은62.5%로 유지됐고, 해외 오탐은6.09%→11.68%로 증가했습니다. 최종 float refit validation의 오탐12.56%도5%를 넘습니다. 실사용 교체를 권장하지 않습니다.</strong></p>")
        md += ["## 기존 test의 회고적 비교", "",
               "아래는 새 독립 시험이 아니다. 과거에 관측한 test에서 해외 사이렌8개 전체를 비목표 모집단에서 제외했다. 제외는 예측 결과와 무관하다. 원래411개 멤버십과 라벨은 보존했고 평가한 파일은403개, 그중 해외 비목표 소리는394개다. 이 결과로 다시 튜닝하지 않는다.", "",
               "| 모델 | 국내 타깃 macro recall | 해외 비목표 오탐 | 국내 도어락 오탐 |", "|---|---:|---:|---:|"]
        previous = read(REPORT / "historical_holdout/previous_models_same_scope.json", [])
        for item in previous:
            if item["backend"] != "int8":
                continue
            label = "초기 int8" if item["model"] == "initial" else "이전 distillation int8"
            md.append(f"| {label} | {pct(item['domestic_target_macro_recall'])} | {item['false_positives']}/{item['foreign_negative_count']} ({pct(item['fpr'])}) | 0/1 |")
        for backend in ("float", "int8"):
            metric = read(REPORT / f"historical_holdout/{backend}_metrics.json")
            value = f"{metric['overseas_other_clip_false_positives']}/{metric['overseas_other_clip_count']} ({pct(metric['overseas_other_clip_fpr'])})"
            md.append(f"| 이번 {backend} | {pct(metric['domestic_target_macro_recall'])} | {value} | {metric['domestic_doorlock_false_positive_groups']}/{metric['domestic_doorlock_group_count']} |")
            metric_html.append(f"<p>{backend}: 국내 recall {pct(metric['domestic_target_macro_recall'])}, 해외 오탐 {value}.</p>")
        classes = historical["domestic_groups"]["classes"]
        md += ["", f"이번 int8의 국내 화재경보는 {classes['fire_alarm']['correct']}/{classes['fire_alarm']['support']}, 도어벨은 {classes['doorbell']['correct']}/{classes['doorbell']['support']}개를 맞혔다. 각 타깃이4개뿐이므로 한 파일의 변화가 클래스 recall25%p를 바꾼다.", ""]
        parity = read(REPORT / "historical_holdout/float_int8_comparison.json")["clips"]
        md += [f"Float/int8 판정 일치율은 {pct(parity['declared_decision_agreement'])}, 최대 확률 차이는 {parity['probability_max_abs']:.6f}다. 이 차이에 맞춰 임계값을 바꾸지 않았다.", ""]
        previous_selected = next(x for x in previous if x["model"] == "previous_distilled" and x["backend"] == "int8")
        recall_gain = historical["domestic_target_macro_recall"] >= previous_selected["domestic_target_macro_recall"]
        fpr_gain = historical["overseas_other_clip_fpr"] <= previous_selected["fpr"]
        md += [f"이전 distillation int8 대비 회고적 recall 비하락 `{recall_gain}`, 오탐률 비증가 `{fpr_gain}`다. 개발 점수 상승만으로 모든 지표나 현장 일반화가 개선됐다고 해석하지 않는다.", ""]
    if audit:
        md += ["## 산출물과 검증", "", f"최종 검증 {sum(audit['checks'].values())}/{len(audit['checks'])}개 통과. 수정된 teacher5회, student distillation10회(8개 CV와 추출 범위 보완 전후2개 refit)의 실제 데이터 학습을 기록했다. 추가39개 공개 파일과 학습 사이렌24개의 최종 추출 이력도 확인했다. 실제 int8 크기는 {audit['trained_int8_bytes']:,} bytes다.", "",
               "[학습 모델 묶음](../../../../models/dscnn_expanded_20261114/README.md), [완료 검증](FINAL_VERIFICATION.json), [새 데이터](../../data/expansion_20261114/dataset.json), [파일 라벨 변경](../../data/expansion_20261114/label_overrides.csv), [공개 출처](../../data/expansion_20261114/PUBLIC_DATA_CREDITS.md), [학습 곡선](TRAINING_REPORT.html)을 제공한다.", ""]
    offline = read(REPORT / "offline_bundle_check/status.json", {})
    if offline.get("passed"):
        md += ["저장 파일 CLI는 float/int8 합계10회 정상 실행했다. 아래5개는 학습에 사용한 자료로 독립 시험이 아니며, 정상 실행과 정답 분류를 구분한다.", "",
               "| 학습용 도어벨 ID | float | int8 |", "|---|---|---|"]
        md += [f"| {item['sample_id']} | {item['float_label']} | {item['int8_label']} |" for item in offline["records"]]
        md += ["", "이5개 중 이번 int8의 doorbell 판정은3개다. 이전 distillation 모델은 같은 학습 자료에서4개였다. 이 숫자를 독립 정확도로 해석하지 않는다.", ""]
    md += ["새 공개 파일을 독립 평가에 사용하지 않았다. 자료를 자르거나 변형한 파일을 새로운 독립 발생원으로 세지 않았지만, 공개 카탈로그와 기존 인터넷 음원의 모든 부분 재사용·출처 관계까지 확인한 것은 아니다. 동일 발생원 분리는 확인 가능한 메타데이터와 보수적 그룹 기준에 따른다. 실시간 마이크·보드 전체 메모리/지연·시간당 오탐·현장 통합 알림은 미검증이며 `deployment_ready=false`다.", "",
           "방법 근거: [TensorFlow DepthwiseConv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D), [작은 음향 모델의 시간 상관과 receptive field](https://www.isca-archive.org/interspeech_2020/xu20d_interspeech.pdf), [distillation](https://arxiv.org/abs/1503.02531). 문헌의 성능을 본 프로젝트 성능으로 사용하지 않았다.", ""]
    REPORT.joinpath("REPORT_KO.md").write_text("\n".join(md), encoding="utf-8")
    plots = []
    for directory in sorted(RUNS.glob("dscnn_expanded_20261114_*")):
        history = rows(directory / "history.csv")
        if not history:
            continue
        plots.append(f"<article><h2>{html.escape(directory.name)}</h2><div class=curves>" +
                     curve(history, "hard_label_loss", "파일 라벨 손실") +
                     curve(history, "distillation_loss_T_squared", "T² distillation KL") +
                     curve(history, "domestic_target_recall", "개발 타깃 recall", 1) +
                     curve(history, "overseas_other_fpr", "개발 해외 비목표 오탐률") + "</div></article>")
    page = f"""<!doctype html><html lang=ko><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>공개 정답 데이터 확장 학습</title><style>body{{font:16px/1.6 system-ui,sans-serif;background:#eef3f7;color:#173048}}main{{max-width:1200px;margin:auto;padding:24px}}article{{background:white;border-radius:10px;padding:20px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #dce5ed;text-align:left}}.curves{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}}svg{{width:100%}}</style><main><h1>공개 정답 데이터 확장·모델 구조 비교</h1><p>{html.escape(status.get('stage','준비'))} · student {status.get('completed_student_folds',0)}/8 fold · 최종 검증 {complete}</p><article><p>공개 음원39개 추가, 학습용 사이렌24개 fire_alarm, 국내 정답 그룹4배 가중치, 클래스별1/3 추출. 새 데이터는 학습 전용이며 기존 test는 독립 시험이 아닙니다. 앱·프로그램·패키지를 다운로드하거나 설치하지 않았습니다.</p><p>두 구조는 시간 평균 전 국소 관측 범위를15→51 frame으로 비교합니다. Teacher head도 새 라벨로 학습했습니다.</p><p><a href="REPORT_KO.md">방법·결과·출처·한계 전체 보고서</a></p></article><article><h2>같은 데이터의 구조 비교</h2><table><tr><th>후보</th><th>fold별 recall</th><th>평균</th><th>모든 개발 가드</th></tr>{''.join(table)}</table></article><article><h2>고정 모델의 회고적 비교</h2>{''.join(metric_html)}<p>해외 사이렌8개를 제외한 같은 모집단이며 이 결과로 재선택하지 않습니다.</p></article>{''.join(plots)}</main></html>"""
    REPORT.joinpath("TRAINING_REPORT.html").write_text(page, encoding="utf-8")
    credits = ["# 추가 공개 음원 출처", "", "원음을 재배포할 때 각 이용조건을 지킨다. 한국 출처 표기는 공개 카탈로그 출처이며 실제 기기 국가 확인은 아니다.", ""]
    for item in read(DATA / "accepted_public.json"):
        credits.append(f"- **{item['sample_id']}** [{item['title']}]({item['source_url']}), {item['author']}, [{item['license_id']}]({item['license_url']}); {item['duration_s']:.3f}초; 원음 `{item['relative_path']}`; SHA-256 `{item['source_sha256']}`.")
    DATA.joinpath("PUBLIC_DATA_CREDITS.md").write_text("\n".join(credits) + "\n", encoding="utf-8")
    print("EXPANSION_REPORT_RENDERED", REPORT / "REPORT_KO.md", flush=True)


if __name__ == "__main__":
    render()
