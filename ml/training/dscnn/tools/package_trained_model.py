"""Copy a completed, audited trained model into a self-contained local bundle."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dscnn_baseline.common import config,source,write_json


def package(study,destination):
    study,destination=Path(study).resolve(),Path(destination).resolve()
    verification=source.read_json(study.parent/"FINAL_VERIFICATION.json")
    if verification["model_development_and_training_complete"] is not True:
        raise ValueError("Package requires a completed artifact audit")
    status=source.read_json(study/"status.json")
    refit=Path(status["refit"])
    frozen=source.read_json(study/"frozen_model.json")
    if status.get("exit_code") != 0 or Path(verification["trained_model"]).resolve() != (refit/"classifier.keras").resolve():
        raise ValueError("Completion audit does not belong to this finished study")
    if source.sha256(refit/"classifier.keras") != frozen["model_sha256"] or source.sha256(refit/"export/dscnn_int8.tflite") != frozen["int8_sha256"]:
        raise ValueError("Frozen trained artifacts changed before packaging")
    mapping={"classifier.keras":refit/"classifier.keras",
             "dscnn_int8.tflite":refit/"export/dscnn_int8.tflite",
             "dscnn_int8.h":refit/"export/dscnn_int8.h",
             "model_metadata.json":refit/"export/model_metadata.json",
             "decision_policy.json":refit/"decision_policy.json"}
    if not all(path.is_file() for path in mapping.values()):
        raise ValueError("A required trained artifact is missing")
    destination.mkdir(parents=True,exist_ok=False)
    for name,path in mapping.items():shutil.copy2(path,destination/name)
    write_json(destination/"frontend.json",config()[0]["frontend"])
    write_json(destination/"labels.json",config()[0]["labels"])
    write_json(destination/"training_provenance.json",{
        "study":str(study),"refit":str(refit),"supervision":source.read_json(refit/"run.json")["supervision"],
        "decision":source.read_json(study/"decision.json"),"frozen":source.read_json(study/"frozen_model.json"),
        "verification":verification,"historical_holdout_float":source.read_json(study/"historical_holdout/float_metrics.json"),
        "historical_holdout_int8":source.read_json(study/"historical_holdout/int8_metrics.json")})
    readme="""# 실제 음원으로 학습한 DS-CNN\n\n이 묶음은 실제 파일 라벨 학습·개발 임계값 선택·고정 epoch refit과 int8 변환 결과다. `pipeline_only=false`이며 합성 소프트웨어 검증 모델과 구분한다.\n\n`classifier.keras`는 float 모델, `dscnn_int8.tflite`와 `.h`는 전체 int8 모델이다. 입력은 16 kHz mono PCM16의 1초 창에서 만든 `[1,98,40,1]` log-Mel이다. 전처리 규약과 클래스 순서는 `frontend.json`, `labels.json`을 따른다. 파일 판정은 `decision_policy.json`의 창 평균·그룹 평균·고정 임계값을 따른다.\n\nPC의 기존 DS-CNN 환경에서 `tools/predict_audio.py --help`로 저장된 음원 추론 인자를 확인한다. 실제 마이크·보드 RAM·지연·시간당 오탐·통합 알림은 미검증이며 `deployment_ready=false`다. 기존 test는 YAMNet에서 이미 사용한 자료로 새로운 독립 시험이 아니다. 상세 수치와 해시는 `training_provenance.json`, `release_manifest.json`에 있다. 원본 음원과 calibration 음원은 이 묶음에 포함하지 않는다.\n"""
    (destination/"README.md").write_text(readme,encoding="utf-8")
    files={p.name:source.sha256(p) for p in sorted(destination.iterdir()) if p.is_file()}
    write_json(destination/"release_manifest.json",{"kind":"trained_dscnn_local_bundle","pipeline_only":False,
               "deployment_ready":False,"files":files})
    print("TRAINED_DSCNN_BUNDLE",destination)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args();package(args.study,args.destination)
