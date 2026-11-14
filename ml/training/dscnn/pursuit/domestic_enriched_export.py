"""Export known-signal robustness candidates; choose readout before calibration."""
import json
import shutil
import numpy as np
from comparison.dataset import read_csv,write_csv,write_json,sha
from dscnn_baseline.common import runtime,config
from .paths import ROOT,DATA,REPORT
from .data import feature,CACHE
from .augmentation import DEST as AUG_CACHE,KINDS
from . import delivered_integer as di
from .cpu_integer import validated_interpreter

NAME='pursuit_20261114_domestic_seen_enriched_presence'
POLICY=((1,1),(1,1))


def inputs(row):
    m=feature(row);kind=row.get('probe_condition') or 'original'
    if kind=='original':directory=CACHE/m['source_sha256']
    else:
        assert kind in KINDS;directory=AUG_CACHE/m['source_sha256']/kind
        r=json.loads((directory/'metadata.json').read_text());assert r['source_sha256']==m['source_sha256'];m=r
    assert sha(directory/'features.npy')==m['feature_sha256']
    return np.load(directory/'features.npy',mmap_mode='r',allow_pickle=False),m


def predict(interpreter,rows):
    return {r['sample_id']:di.predict_q(interpreter,inputs(r)[0]) for r in rows}


def main():
    run=ROOT/'runs'/NAME;status=json.loads((run/'status.json').read_text());assert status['complete'] and status['target_results_training_seen']
    model_path=run/'classifier.keras';assert sha(model_path)==status['model_sha256']
    out=REPORT/'domestic_enriched_delivered_v1';out.mkdir(exist_ok=False)
    rows=read_csv(DATA/'domestic_seen_enriched_v1/development.csv');lookup={r['sample_id']:r for r in rows}
    probes=read_csv(run/'known_positive_probes.csv');assert len(probes)==172 and all(r['role']=='train' for r in probes)
    original=read_csv(DATA/'domestic_seen_enriched_v1/known_positive_probes.csv');original_ids={r['sample_id'] for r in original}
    negative_ids=[r['sample_id'] for r in rows if r['role']=='train' and r['label']=='other'];assert len(negative_ids)==8696
    representative=[];provenance=[]
    for row in probes+[lookup[k] for k in negative_ids]:
        assert row['role']=='train';a,m=inputs(row)
        indices=range(len(a)) if row['label']!='other' else np.unique([len(a)//2,int(np.argmax(np.mean(a,axis=(1,2,3))))])
        for i in indices:
            representative.append(a[i]);provenance.append(dict(split='train',sample_id=row['sample_id'],
                original_sample_id=row.get('original_sample_id',row['sample_id']),condition=row.get('probe_condition') or 'original',
                source_sha256=m['source_sha256'],feature_sha256=m['feature_sha256'],window_index=int(i)))
    x=np.stack(representative);np.save(out/'representative_training.npy',x,allow_pickle=False);write_json(out/'representative_training_provenance.json',provenance)
    selection=[r for r in rows if r['role']=='selection']+probes;calibration=[r for r in rows if r['role']=='calibration']
    assert all(r['label']=='other' for r in rows if r['role'] in ('selection','calibration'))
    write_json(out/'export_plan.json',dict(source_model=NAME,source_model_sha256=sha(model_path),
        temperatures=[1.,4.],output_kind='probability',policy=POLICY,
        selection='Worst training-seen acoustic condition first, using fixed 1/1 readout; source-held-out other recordings',
        calibration='Only raise thresholds after delivered variant selected; no calibration ranking',
        original_signal_recordings=43,transformed_training_probes=172,independent_target_test_recordings=0,
        representative_windows=len(x),representative_role='train',
        code_identity={p:sha(ROOT/p) for p in ['pursuit/domestic_enriched_export.py','pursuit/delivered_integer.py']}))
    tf,keras=runtime('cpu');assert not tf.config.list_physical_devices('GPU')
    model=keras.models.load_model(str(model_path),compile=False);variants=[]
    for temperature in [1.,4.]:
        directory=out/('probability_t'+str(int(temperature)))
        interpreter,meta=di.export(tf,keras,model,x,provenance,directory,config()[0]['frontend'],'probability',temperature)
        interpreter=validated_interpreter(tf,directory/'model_int8.tflite',x,directory/'cpu_kernel_gate.json',64)
        values=predict(interpreter,selection);cuts=di.fit_cuts(selection,values,'probability',meta['output']['zero_point'],POLICY)
        metric=di.measure(selection,values,'probability',meta['output']['zero_point'],POLICY,cuts)
        np.savez(directory/'selection_outputs_q.npz',**values)
        write_json(directory/'selection_analysis.json',dict(metric=metric,output_values_are_actual_int8=True,
            selection_zero_errors_by_construction=True,target_sources_training_seen=True,fresh_test_used=False))
        variants.append(dict(temperature=temperature,directory=directory,interpreter=interpreter,meta=meta,values=values,cuts=cuts,metric=metric))
        print('DOMESTIC_AUG_DELIVERED_SELECTION',temperature,metric,flush=True)
    selected=max(variants,key=lambda r:di.key(r['metric']))
    write_json(out/'chosen_before_calibration.json',dict(directory=str(selected['directory']),temperature=selected['temperature'],
        source_tflite_sha256=selected['meta']['model_sha256'],selection_metric=selected['metric'],
        calibration_predictions_used_to_choose=False,independent_target_validation_recordings=0))
    meta=selected['meta'];zero=meta['output']['zero_point'];cal=predict(selected['interpreter'],calibration)
    cuts=di.fit_cuts(calibration,cal,'probability',zero,POLICY,selected['cuts'])
    metric=di.measure(selection,selected['values'],'probability',zero,POLICY,cuts)
    cal_metric=di.measure(calibration,cal,'probability',zero,POLICY,cuts)
    combined=dict(selected['values'],**cal);np.savez(out/'chosen_selection_and_calibration_outputs_q.npz',**combined)
    assert sha(model_path)==status['model_sha256']
    result=dict(complete=True,source_model=NAME,source_model_sha256=status['model_sha256'],
        representative_windows=len(x),selected_temperature=selected['temperature'],
        selected_tflite_sha256=meta['model_sha256'],selection_after_calibration=metric,calibration_after_fit=cal_metric,
        calibration_is_threshold_fitting_not_independent_test=True,known_sources_training_seen=True,
        fresh_test_used=False,actual_product_targets_verified=False)
    write_json(out/'status.json',result)
    bundle=REPORT/'candidates/domestic_known_enriched_v3';bundle.mkdir(exist_ok=False)
    for filename in ['model_int8.tflite','model_int8.h','model_metadata.json']:shutil.copyfile(selected['directory']/filename,bundle/filename)
    config_dict=dict(output_semantics='Quantized softmax probabilities',class_order=di.LABELS,
        output_scale=meta['output']['scale'],output_zero_point=zero,policy=POLICY,
        per_class_int8_minimum_q=[int(k+zero) for k in cuts],
        unsigned_probability_cuts=[int(k) for k in cuts],
        argmax_required_per_contributing_window=True,target_argmax_required=True,
        threshold_semantics='Strictly above every observed nonmatching selection/calibration envelope; uint score 0 is inactive',
        tie_break='First maximum, matching numpy.argmax',
        input_shape=meta['input']['shape'],export_logit_temperature=selected['temperature'],
        multiple_notifications_per_recording_controlled=False,actual_notification_controller_tested=False,
        old_firmware_thresholds_automatically_updated=False,board_validated=False)
    write_json(bundle/'delivered_output_contract.json',config_dict)
    write_json(bundle/'readout_config.json',dict(config_dict,thresholds=[float(k*meta['output']['scale']) for k in cuts],
        calibration_only_raises_selected_thresholds=True,actual_product_targets_verified=False,deployment_ready=False))
    write_json(bundle/'analysis.json',result);write_csv(bundle/'known_training_condition_probes.csv',probes)
    write_json(bundle/'candidate.json',dict(kind='DOMESTIC_TRAINING_SEEN_AUGMENTED_RESEARCH_CANDIDATE',
        source_model=NAME,source_model_sha256=status['model_sha256'],original_target_recordings=43,training_condition_probes=172,
        independent_target_test_recordings=0,actual_device_days=0,deployment_ready=False,actual_product_targets_verified=False,
        files_sha256={p.name:sha(p) for p in bundle.iterdir() if p.is_file()},
        representative_provenance_path=str(out/'representative_training_provenance.json'),
        research_terms='Legacy noncommercial and original source restrictions retained; commercial clearance not claimed'))
    lines=['# 국내 학습 음원과 합성 변형을 사용한 int8 후보','',
        '서로 다른 원본 신호는 43개이며 모두 학습에 사용했습니다. 아래는 원음과 학습에 포함한 합성 변형의 파일 감지 결과입니다. 새로운 기기·방·실제 발생 사건을 검증한 결과로 해석할 수 없습니다.','',
        '| 조건 | 화재경보 | 초인종 |','|---|---:|---:|---|']
    lines[-1]='|---|---:|---:|'
    for condition,c in metric['conditions'].items():
        a=c['fire_alarm'];b=c['doorbell'];lines.append(f"| {condition} | {a['detected_files']}/{a['files']} | {b['detected_files']}/{b['files']} |")
    lines+=['',f"원래 모델의 마지막 Dense 출력을 온도 {selected['temperature']:g}로 조정한 뒤 정수화했습니다. 원본 체크포인트는 보존했습니다. 출력은 softmax 확률이며, 클래스별 int8 최소값은 {config_dict['per_class_int8_minimum_q']}입니다.",'',
        '다른 소리의 선택·보정 자료는 임계값을 정하는 데 사용했습니다. 그 자료에서 오류가 0이라는 수치로 30일 오알림률을 증명할 수 없습니다. SNR은 전체 파일 RMS 기준이며 이벤트 구간 SNR과 실제 마이크 응답은 미측정입니다.','',
        '실제 기기 사용 기간은 0일이며 제품 목표 달성 상태는 false입니다. 기존 배포 모델·펌웨어는 교체하지 않았습니다. 별도로 등록한 새로운 합성 조합 시험과 실제 기기 시험의 결과를 구분해 기록합니다.']
    (bundle/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('DOMESTIC_AUG_TRAINED_RESEARCH_BUNDLE_CREATED',bundle,metric,flush=True)


if __name__=='__main__':main()
