"""Package only the audited development-selected improved model."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.common import config, source, write_json
from improvement.protocol import REPORT


def package():
    verification = source.read_json(REPORT / "FINAL_VERIFICATION.json")
    if verification["model_development_and_training_complete"] is not True:
        raise ValueError("Improvement audit required")
    frozen = source.read_json(REPORT / "frozen_model.json")
    refit = Path(frozen["refit"])
    if source.sha256(refit / "classifier.keras") != frozen["model_sha256"] or source.sha256(refit / "export/dscnn_int8.tflite") != frozen["int8_sha256"]:
        raise ValueError("Frozen model changed before package")
    destination = ROOT.parents[1] / "models/dscnn_improved_20261114"
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("classifier.keras", "decision_policy.json"):
        shutil.copy2(refit / name, destination / name)
    for name in ("dscnn_int8.tflite", "dscnn_int8.h", "model_metadata.json"):
        shutil.copy2(refit / "export" / name, destination / name)
    write_json(destination / "labels.json", config()[0]["labels"])
    write_json(destination / "frontend.json", config()[0]["frontend"])
    write_json(destination / "training_provenance.json", {"study": str(REPORT), "frozen": frozen,
               "verification": verification, "training_run": source.read_json(refit / "run.json"),
               "float_historical_holdout": source.read_json(REPORT / "historical_holdout/float_metrics.json"),
               "int8_historical_holdout": source.read_json(REPORT / "historical_holdout/int8_metrics.json")})
    decision = frozen["decision"]
    text = f"""# 개선 실험 DS-CNN\n\n선택 후보는 `{decision['candidate']}`이며 최종 모델의 distillation 적용 여부는 `{decision['distillation_selected']}`입니다. refit은 사전 고정 {decision['refit_epochs']} epoch입니다. 상세 방법·수치·한계는 `training_provenance.json`과 [개선 보고서](../../training/dscnn/reports/improvement_20261114/REPORT_KO.md)를 확인하세요.\n\n기존 test에서는 재현율과 함께 오탐도 증가했습니다. 개발 가드는 통과했지만 회고적 test의 해외 other 오탐률은 5%를 넘었습니다. 실사용 교체·배포를 권장하지 않으며 연구·검토용 모델로 보존합니다.\n\n`classifier.keras`는 float 모델이며 `dscnn_int8.tflite`와 `.h`는 전체 int8 모델입니다. 전처리는 16 kHz mono PCM16 → 1초 `[98,40,1]` log-Mel입니다. 파일의 모든 창 확률을 평균하고 source group은 파일 확률을 같은 비중으로 평균합니다. 이후 `decision_policy.json`의 클래스별 타깃 임계값을 적용합니다. 이 정책은 보드의 단일 창·시간 누적 정책을 검증한 결과가 아닙니다.\n\nUbuntu PC의 `/home/jhh0415/Documents/ChatGPT/캡스톤/hearing-alert-system/ml/training/dscnn`에서 기존 `.venv/bin/python -B tools/predict_audio.py --help`로 저장 음원 추론 옵션을 확인합니다. 이 묶음의 SHA256은 `release_manifest.json`에 있습니다. 원음과 calibration 음원은 포함하지 않습니다.\n\n`pipeline_only=false`인 실제 학습 모델입니다. 기존 test는 과거 관측한 자료여서 독립 시험이 아닙니다. 현장 마이크·시간당 오탐·ESP-IDF 전체 빌드·실보드 RAM/지연·통합 알림은 미검증이며 `deployment_ready=false`입니다.\n"""
    (destination / "README.md").write_text(text, encoding="utf-8")
    write_json(destination / "release_manifest.json", {"kind": "improved_trained_dscnn_local_bundle",
               "pipeline_only": False, "distillation_applied": decision["distillation_selected"],
               "usage": "experimental_review_only",
               "deployment_ready": False, "files": {p.name: source.sha256(p) for p in sorted(destination.iterdir()) if p.is_file()}})
    print("IMPROVED_TRAINED_MODEL_BUNDLE", destination)


if __name__ == "__main__":
    package()
