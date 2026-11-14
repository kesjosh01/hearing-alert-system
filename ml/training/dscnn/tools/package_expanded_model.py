"""Ship the audited expansion model and public-data credits as a new bundle."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.common import config, source, write_json
from expansion.protocol import DATA, REPORT


def package():
    audit = source.read_json(REPORT / "MODEL_VERIFICATION.json")
    if not audit["model_development_and_training_complete"]:
        raise ValueError("Requirement-level model audit must pass first")
    frozen = source.read_json(REPORT / "frozen_model_coverage.json")
    run = Path(frozen["refit"])
    if source.sha256(run / "classifier.keras") != frozen["model_sha256"] or source.sha256(run / "export/dscnn_int8.tflite") != frozen["int8_sha256"]:
        raise ValueError("Frozen model hash differs")
    output = ROOT.parents[1] / "models/dscnn_expanded_20261114"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(run / "classifier.keras", output / "classifier.keras")
    for name in ("dscnn_int8.tflite", "dscnn_int8.h", "model_metadata.json"):
        shutil.copy2(run / "export" / name, output / name)
    write_json(output / "decision_policy.json", frozen["decision"]["decision_policy"])
    write_json(output / "labels.json", config()[0]["labels"])
    write_json(output / "frontend.json", config()[0]["frontend"])
    write_json(output / "training_provenance.json", {"study": str(REPORT), "model_audit": audit,
               "frozen_model": frozen, "training_run": source.read_json(run / "run.json"),
               "dataset": source.read_json(DATA / "dataset.json"),
               "float_historical_metrics": source.read_json(REPORT / "historical_holdout/float_metrics.json"),
               "int8_historical_metrics": source.read_json(REPORT / "historical_holdout/int8_metrics.json")})
    credits = ["# 추가 공개 음원 출처", "", "24개 CC BY 4.0 한국 공개 자료와 15개 CC0 해외 자료다. 원음은 이 모델 묶음에 포함하지 않는다. 공개 자료의 물리적 기기 국가나 실제 독립 발생원을 확인했다고 주장하지 않는다. 기존 ESC-50 등 이전 데이터의 이용 조건도 별도로 보존한다.", ""]
    for item in source.read_json(DATA / "accepted_public.json"):
        credits.append(f"- {item['sample_id']}: [{item['title']}]({item['source_url']}), {item['author']}; [{item['license_id']}]({item['license_url']}). SHA-256 `{item['source_sha256']}`.")
    (output / "PUBLIC_DATA_CREDITS.md").write_text("\n".join(credits) + "\n", encoding="utf-8")
    model = frozen["decision"]["candidate"]
    text = f"""# 공개 정답 데이터 확장 DS-CNN

공개 도어벨·경보 음원 39개를 추가하고 학습용 ESC-50 사이렌 24개를 사용자 기준의 `fire_alarm`로 학습했다. 국내 정답 source group 가중치는 해외의 4배이며, 세 클래스의 학습 추출 확률은 각 1/3이다. 새 데이터 라벨과 일치하도록 YAMNet 분류 head를 다시 학습한 뒤 teacher distillation을 수행했다.

기존 test의 int8 평균 타깃 recall은 이전과 같은62.5%이며, 해외 비목표 오탐은24/394(6.09%)에서46/394(11.68%)로 증가했다. 개발 점수의 소폭 상승이 이 자료의 전반적 성능 개선으로 이어지지 않았다. 실사용 모델 교체를 권장하지 않으며 연구·검토용으로 보존한다.

선택 구조는 `{model}`, 최종 학습은 사전에 고정한 {frozen['decision']['refit_epochs']} epoch다. `temporal24`는 시간축 depthwise kernel을 9×3으로 확장하고, `freq24`는 이전 3×3 구조를 유지한다. 선택은 기존 국내 4-fold와 해외 validation으로만 수행했다. 새 공개 음원은 학습 전용이다.

초기 refit의 무작위 추출에서 새 파일1개가0회인 것을 확인해, test 평가 전에 학습 타깃100개의 포함 여부만으로 최종 시드43을 고정했다. 구조·5epoch·150 update·그룹 가중치·임계값은 유지했고 초기 refit도 보존했다. 성능을 비교해 시드를 선택하지 않았다. 상세 변경은 `training_provenance.json`에 기록한다.

`classifier.keras`는 실제 학습 float 모델이다. `dscnn_int8.tflite`와 `dscnn_int8.h`는 같은 모델의 전체 int8 변환이다. `decision_policy.json`의 클래스별 타깃 임계값은 파일의 창 평균 확률과 source group 평균 확률에 적용한다. 입력은 기존과 동일한 16 kHz mono PCM16, 1초 `[98,40,1]` log-Mel이다. 원음·시간 구간 정답·보드 시간 누적 정책을 포함하지 않는다.

[실험 보고서](../../training/dscnn/reports/expansion_20261114/REPORT_KO.md), [학습 곡선](../../training/dscnn/reports/expansion_20261114/TRAINING_REPORT.html), `training_provenance.json`, [추가 자료 출처](PUBLIC_DATA_CREDITS.md)를 확인한다. 기존 test는 과거에 관측한 자료로 독립 시험이 아니다. 해외 사이렌 8개를 비목표 오탐 모집단에서 제외한 동일한 기준으로 이전 모델과 비교했다.

기존 환경의 `ml/training/dscnn/tools/predict_audio.py`로 저장 음원을 추론할 수 있다. 묶음의 파일 해시는 `release_manifest.json`에 있다. `pipeline_only=false`이며 보드·실시간 마이크·현장 시간당 오탐·실보드 메모리/지연은 미검증이다. `deployment_ready=false`다.
"""
    (output / "README.md").write_text(text, encoding="utf-8")
    write_json(output / "release_manifest.json", {"kind": "public_target_expanded_trained_dscnn_bundle",
               "pipeline_only": False, "distillation_applied": True, "public_audio_added": 39,
               "training_sirens_positive": 24, "usage": "experimental_review_only", "deployment_ready": False,
               "files": {p.name: source.sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print("EXPANDED_MODEL_BUNDLE_READY", output, flush=True)


if __name__ == "__main__":
    package()
