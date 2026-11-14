"""Live, evidence-bound status; no historical results or goal gates overwritten."""
import json
from collections import Counter
from datetime import datetime,timezone
from comparison.dataset import write_json,sha
from .paths import ROOT,DATA,REPORT


def main():
    runs=[]
    for directory in sorted((ROOT/'runs').glob('pursuit_20261114_*')):
        p=directory/'status.json'
        if not p.exists():continue
        j=json.loads(p.read_text());meta=json.loads((directory/'run.json').read_text()) if (directory/'run.json').exists() else {}
        metric=j.get('selected_metrics') if j.get('complete') else None
        runs.append(dict(name=directory.name,status_path=str(p),status_sha256=sha(p),complete=j.get('complete',False),
            first_error=j.get('first_error'),epoch=j.get('epoch'),selected_epoch=j.get('selected_epoch'),
            actual_optimizer_steps=j.get('actual_optimizer_steps',j.get('optimizer_steps',0)),
            dataset=meta.get('spec',{}).get('dataset'),selected_metrics=metric,
            target_results_training_seen=j.get('target_results_training_seen',meta.get('target_positive_probes_training_seen',False)),
            fresh_test_used=j.get('fresh_test_used',False),product_targets_verified=j.get('product_targets_verified',False)))
    fsd={}
    for bucket in ['targets','hard_negatives','background']:
        receipts=[json.loads(p.read_text()) for p in (REPORT/'transport'/bucket).glob('*_complete.json')]
        fsd[bucket]=dict(verified_shards=len(receipts),verified_files=sum(len(j['paths']) for j in receipts))
    sources={}
    for name,file in [('realdesed_official','realdesed_recovery/status.json'),('realdesed_priority','realdesed_priority_complete.json'),
        ('korean_speech','fleurs_intake.json'),('additional_audioset','audioset_strong_intake/status.json')]:
        p=REPORT/file
        if p.exists():sources[name]=json.loads(p.read_text())
    cohorts={}
    for d in DATA.iterdir():
        if d.is_dir() and (d/'freeze.json').exists():
            j=json.loads((d/'freeze.json').read_text());cohorts[d.name]=dict(freeze_sha256=sha(d/'freeze.json'),counts=j.get('counts'))
    integer=REPORT/'int8_readout_trials/pursuit_20261114_r02_freq24_aug/canonical_delivered_output_analysis.json'
    domestic_integer={}
    for tag in ['domestic_seen_integer','domestic_seen_integer_full_positive']:
        p=REPORT/tag/'status.json'
        if p.exists():
            j=json.loads(p.read_text())
            domestic_integer[tag]=dict(status_path=str(p),status_sha256=sha(p),complete=j['complete'],
                representative_windows=j['representative_windows'],results=[dict(kind=x['kind'],
                after_calibration=x['selection_after_calibration'],calibration=x['calibration_after_threshold_fit']) for x in j['results']])
    stress_path=REPORT/'domestic_known_v1_stress.json'
    stress=json.loads(stress_path.read_text()) if stress_path.exists() else None
    queue_path=REPORT/'domestic_aug_queue_status.json'
    queue=json.loads(queue_path.read_text()) if queue_path.exists() else None
    public_path=REPORT/'final_public_benchmark_v1/results.json'
    public=json.loads(public_path.read_text()) if public_path.exists() else None
    delivered={}
    for name in ['domestic_aug_delivered_v1','domestic_enriched_delivered_v1','general_delivered_v1']:
        p=REPORT/name/'status.json'
        if p.exists():delivered[name]=json.loads(p.read_text())
    record=dict(updated_utc=datetime.now(timezone.utc).isoformat(),runs=runs,
        completed_models=sum(r['complete'] for r in runs),ongoing_models=[r['name'] for r in runs if not r['complete'] and not r['first_error']],
        verified_fsd_collection=fsd,other_source_status=sources,cohorts=cohorts,domestic_integer=domestic_integer,
        domestic_frozen_candidate_stress=stress,domestic_augmentation_queue=queue,
        delivered_candidates=delivered,final_fixed_public_recording_benchmark=public,
        integer_delivered_output_analysis_path=str(integer) if integer.exists() else None,
        objective=dict(fire_actual_event_recall=.999,doorbell_actual_event_recall=.95,false_notifications_per_device_30_days=1),
        actual_hardware_available=False,actual_hardware_recording_or_notification_trials=0,
        actual_product_criteria_verified=False,overall_goal_complete=False,
        evidence_limits=['Selection thresholds use maximum non-target scores, so zero selection errors are true by construction, not independent reliability evidence.',
            'File-level labels and source-mean file detection are not actual activation-event ground truth.',
            'Domestic seen probes were intentionally used in training and checkpoint selection.',
            'No fresh-test features or predictions used for optimization in this campaign.',
            'Procedural and speech recordings are not independent alarm devices or household device-month exposure.',
            'Research datasets include noncommercial terms; no commercial training-data clearance claimed.'])
    write_json(REPORT/'progress_current.json',record)
    lines=['# 공개 음원 보강·모델 학습 현재 상태','',
        f"갱신 UTC: {record['updated_utc']}",'',
        '**제품 기준 달성은 아직 확인되지 않았습니다.** 마이크·알림 기기 실험은 0회입니다. 아래 값은 모델과 임계값을 선택하는 자료의 파일 감지 대리지표이며, 실제 경보 발생 재현율이나 30일 오알림률이 아닙니다. 선택 자료에서 오분류 0이 되도록 임계값을 정했으므로, 0이라는 수치 자체는 독립 검증 근거가 아닙니다.','',
        '서로 다른 데이터 구성의 숫자를 직접 비교하면 안 됩니다. 국내 과적합 행은 학습에 사용한 음원에 대한 결과입니다.','',
        '| 모델 | 데이터 | 완료 epoch / 선택 epoch | 화재경보 | 초인종 | 자료 성격 |',
        '|---|---|---:|---:|---:|---|']
    for r in runs:
        m=r['selected_metrics']
        if m:
            scope='학습 국내 파일과 합성 변형' if 'training_seen_condition_results' in m else '학습에 사용한 국내 파일' if r['target_results_training_seen'] else '선택 자료, 출처별 평균'
            lines.append(f"| {r['name'].replace('pursuit_20261114_','')} | {r['dataset']} | {r['epoch']} / {r['selected_epoch']} | {m['fire_recall']:.1%} | {m['doorbell_recall']:.1%} | {scope} |")
        else:lines.append(f"| {r['name'].replace('pursuit_20261114_','')} | {r['dataset']} | {r['epoch']} / {r['selected_epoch']} | 진행 중 | 진행 중 | {r['first_error'] or '미완료'} |")
    for tag,j in domestic_integer.items():
        for result in j['results']:
            c=result['after_calibration']['classes'];fa=c['fire_alarm'];db=c['doorbell']
            lines+=['',f"국내 학습 음원 int8 변환 `{tag}` / `{result['kind']}`: 화재경보 {fa['detected_files']}/{fa['training_seen_files']}, 초인종 {db['detected_files']}/{db['training_seen_files']}. 이 값도 학습에 사용한 음원이며 실제 기기 검증이 아닙니다."]
    if stress:
        lines+=['','동결한 국내 int8 후보의 합성 음향 변화 시험입니다. 같은 43개 학습 음원의 변형이며, 새로운 기기·방·발생 사건이 아닙니다. 변형을 적용한 뒤 임계값을 다시 맞추지 않았습니다.','',
            '| 합성 조건 | 화재경보 | 초인종 |','|---|---:|---:|']
        for kind,c in stress['conditions'].items():
            a=c['fire_alarm'];b=c['doorbell'];lines.append(f"| {kind} | {a['detected_files']}/{a['files']} | {b['detected_files']}/{b['files']} |")
    lines+=['','새 목표의 독립 시험 자료는 아직 모델 선택에 사용하지 않았습니다. 원본 음원·기존 체크포인트·기존 환경을 보존하며, 새 작업은 `pursuit_20261114` 경로와 새 실험 디렉터리에 기록합니다.','',
        'FSD50K 다운로드 검증 수: '+', '.join(f"{k} {v['verified_files']:,}개" for k,v in fsd.items())+'.','',
        '숫자의 원본과 데이터 경계·해시·실험별 상태 파일 경로는 같은 폴더의 `progress_current.json`에 기록했습니다.']
    if public:
        lines+=['','학습·임계값 선택과 분리한 최종 공개 평가를 완료했습니다. 위 표는 선택 자료 또는 학습 국내 음원 결과입니다. 별도의 공개 시험과 제품 목표 판단은 `FINAL_REVIEW_KO.md` 및 `final_public_benchmark_v1/results.json`을 확인하세요. 공개 파일 감지율은 실제 발생 감지율·30일 기기 오알림률이 아닙니다.']
    (REPORT/'progress_current.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('PURSUIT_CURRENT_PROGRESS_UPDATED',record['completed_models'],record['ongoing_models'],flush=True)


if __name__=='__main__':main()
