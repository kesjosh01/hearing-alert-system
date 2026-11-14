"""Package actual selected weights and frozen decisions, preserving older bundles."""
import json
import shutil
import shlex
import subprocess
from pathlib import Path
from dscnn_baseline.common import config
from .protocol import ROOT,REPO,DATA,REPORT
from .dataset import sha,write_json


def main():
    frozen=json.loads((REPORT/'final_candidate.json').read_text());run=Path(frozen['champion']['run']);status=json.loads((run/'status.json').read_text())
    evaluation=json.loads((REPORT/'evaluation/status.json').read_text());assert status['complete'] and evaluation['complete']
    assert sha(run/'classifier.keras')==frozen['model_sha256']
    out=REPO/'ml/models/dscnn_comparison_20261114';out.mkdir(exist_ok=False)
    for name in ['classifier.keras']:
        shutil.copy2(run/name,out/name)
    for name in ['dscnn_int8.tflite','dscnn_int8.h']:
        shutil.copy2(run/'export'/name,out/name)
    shutil.copy2(DATA/'PUBLIC_DATA_CREDITS.md',out/'PUBLIC_DATA_CREDITS.md')
    shutil.copy2(REPORT/'temporal_replay/frozen_policy.json',out/'temporal_policy.json')
    calibrated=json.loads((run/'calibrated_policy.json').read_text())
    write_json(out/'decision_policy.json',dict(float_thresholds=calibrated['float'],int8_thresholds=calibrated['int8'],
               pooling_fraction=calibrated['pooling_fraction'],source=calibrated['source']))
    info=json.loads((run/'export/model_metadata.json').read_text());meta=json.loads((run/'run.json').read_text());o=meta['options']
    info.update(trained=True,pipeline_only=False,version='dscnn-comparison-20261114-v1',architecture=o['architecture'],
                pooling_fraction=o['pooling_fraction'],clip_aggregation='Mean of top fraction of windows ranked by1-p_other; all windows when fraction1',
                knowledge_distillation_applied=o['kd_weight']>0,kd_weight=o['kd_weight'],selected_epoch=status['selected_epoch'],
                complexity=status['complexity'],deployment_ready=False,physical_board_or_microphone_validated=False,
                observed_fresh_test_fpr_le_5_percent=evaluation['observed_new_test_fpr_le_5_percent'])
    write_json(out/'model_metadata.json',info);write_json(out/'frontend.json',config()[0]['frontend']);write_json(out/'labels.json',config()[0]['labels'])
    write_json(out/'training_provenance.json',dict(frozen=frozen,run=meta,status=status,
        selected_checkpoint_exposure=json.loads((run/'selected_checkpoint_exposure_audit.json').read_text()),
        calibration=json.loads((REPORT/'calibration'/run.name/'results.json').read_text()),
        temporal_selection=json.loads((REPORT/'temporal_replay/selection.json').read_text()),
        evaluation=evaluation,source_freeze=json.loads((DATA/'source_freeze.json').read_text()),
        code_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        local_changes_not_committed_or_pushed=True))
    commands=[]
    from .dataset import read_csv
    sample=read_csv(DATA/'fresh_test.csv')[0];audio=ROOT/sample['relative_path']
    for mode in ['clip','replay']:
        commands.append(' '.join(shlex.quote(str(x)) for x in [ROOT/'.venv/bin/python','-B','-m','comparison.infer','--bundle',out,'--audio',audio,'--backend','int8','--mode',mode]))
    text=f'''# 학습된 비교 모델 묶음

모델: `{run.name}`. 실제 학습한 checkpoint와 별도 source-held-out 보정셋의 float/int8 판정 정책을 담았다. 기존 모델을 덮어쓰지 않았다.

- 구조: {o['architecture']}, 선택 epoch: {status['selected_epoch']}
- 증류 손실 비중: {o['kd_weight']}. 0이면 비교 결과로 정답 라벨 학습을 선택한 것이며, 최종 모델을 증류 모델이라고 부르지 않는다.
- 파일 집계: `pooling_fraction={o['pooling_fraction']}`, 1-p_other가 높은 창을 선택한 뒤 세 확률을 같은 창에서 평균한다.
- 일반 clip 임계값과 streaming/replay 임계값은 별도다. 기존 평균 집계 전용 CLI 대신 아래 CLI를 사용한다.
- 학습/선택/보정/새 시험은 발생원별로 분리했다. 공개 녹음의 실제 기기 국가, 모든 편집본 재사용, foundation 사전학습 미노출까지 검증한 것은 아니다.
- 새 시험 파일 79개는 관련 파일을 포함한 작은 공개 코호트다. 실제 마이크·ESP32-S3 지연/RAM·현장 시간당 오탐은 검증하지 않았으며 deployment_ready=false다.

## 로컬 파일 실행

호스트: 이 노트북. 작업 디렉터리: `{ROOT}`. 기존 환경을 사용하며 설치하지 않는다. 아래 명령은 종료형이며 터미널을 유지할 필요가 없다.

```bash
cd {shlex.quote(str(ROOT))}
{commands[0]}
{commands[1]}
```

정상 결과는 종료 코드0의 JSON이다. `clip`은 label/probabilities, `replay`는 파일 재생 시각 기준 alerts를 추가한다. replay 시각은 실제 소리의 정답 onset이나 실보드 처리 지연을 뜻하지 않는다. 종료 후 해당 CLI 프로세스가 남지 않는다.

모든 결과와 비채택 비교: `{REPORT}`. 출처와 라이선스: PUBLIC_DATA_CREDITS.md. 공개 업로드/커밋/푸시는 하지 않았다.
'''
    (out/'README.md').write_text(text)
    files={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    write_json(out/'release_manifest.json',dict(files=files,trained_model_sha256=sha(out/'classifier.keras'),trained_int8_sha256=sha(out/'dscnn_int8.tflite'),
               source_run=str(run),actual_training_verified=True,board_or_microphone_verified=False))
    write_json(REPORT/'packaging.json',dict(bundle=str(out),files=files,source_model_sha256=frozen['model_sha256'],older_bundles_overwritten=False))
    print('COMPARISON_BUNDLE_PACKAGED',out,flush=True)


if __name__=='__main__':main()
