"""Versioned full-positive representative calibration; same trained model.

Known positive probes remain TRAINING-SEEN. Readout selection uses these
probes and source-held-out other recordings. Calibration-other recordings
can only raise the frozen thresholds; no fresh recordings are used here.
"""
import json
import numpy as np
from comparison.dataset import read_csv,write_json,sha
from dscnn_baseline.common import runtime
from dscnn_baseline.export import export_int8,interpreter_for,predict_int8
from .paths import ROOT,DATA,REPORT
from .data import feature,CACHE
from . import operating as op
from .integer_readout_audit import score

NAME='pursuit_20261114_domestic_seen_presence'


def predict(interpreter,rows):
    out={}
    for r in rows:
        m=feature(r);a=np.load(CACHE/m['source_sha256']/'features.npy',mmap_mode='r',allow_pickle=False)
        out[r['sample_id']]=predict_int8(interpreter,a)
    return out


def fixed(rows,values,mode,policy,thresholds):
    scores={r['sample_id']:op.score(values[r['sample_id']],mode,1.,policy).max(0) for r in rows}
    flags={k:(v>=thresholds)&(v>0) for k,v in scores.items()};counts={}
    for k,label in enumerate(['fire_alarm','doorbell']):
        rr=[r for r in rows if r['label']==label]
        counts[label]=dict(training_seen_files=len(rr),detected_files=sum(bool(flags[r['sample_id']][k]) for r in rr))
    others=[r for r in rows if r['label']=='other']
    return dict(classes=counts,other_files=len(others),other_alert_files=sum(bool(flags[r['sample_id']].any()) for r in others),
        wrong_target_alert_files=sum(bool(flags[r['sample_id']][1-k]) for k,label in enumerate(['fire_alarm','doorbell']) for r in rows if r['label']==label),
        independent_target_test_files=0,actual_monthly_false_rate_proven=False)


def main():
    run=ROOT/'runs'/NAME;status=json.loads((run/'status.json').read_text());assert status['complete'] and status['target_results_training_seen']
    d=DATA/'domestic_seen_v1';freeze=json.loads((d/'freeze.json').read_text());assert all(sha(d/k)==v for k,v in freeze['files'].items())
    rows=read_csv(d/'development.csv');probes=read_csv(d/'known_positive_probes.csv')
    training=[r for r in rows if r['role']=='train'];selection=[r for r in rows if r['role']=='selection']
    calibration=[r for r in rows if r['role']=='calibration'];assert all(r['label']=='other' for r in selection+calibration)
    out=REPORT/'domestic_seen_integer_full_positive';out.mkdir(exist_ok=False);representative=[];provenance=[];rng=np.random.default_rng(42)
    rr=probes+[r for r in training if r['label']=='other']
    rr=probes+[rr[i] for i in rng.permutation(len(rr)) if rr[i]['label']=='other'][:256]
    for r in rr:
        assert r['role']=='train';m=feature(r);a=np.load(CACHE/m['source_sha256']/'features.npy',mmap_mode='r',allow_pickle=False)
        for i in (range(len(a)) if r['label']!='other' else np.unique(np.linspace(0,len(a)-1,min(len(a),2)).astype(int))):
            representative.append(a[i]);provenance.append(dict(split='train',sample_id=r['sample_id'],source_sha256=m['source_sha256'],window_index=int(i)))
    x=np.stack(representative);tf,keras=runtime('cpu');assert not tf.config.list_physical_devices('GPU')
    p=run/'classifier.keras';assert sha(p)==status['model_sha256']
    with tf.device('/CPU:0'):
        model=keras.models.load_model(str(p),compile=False);logits=keras.Model(model.input,model.get_layer('class_logits').output)
    op.score=score;records=[]
    for kind,net in [('probability',model),('logits',logits)]:
        directory=out/kind;path,meta=export_int8(tf,net,x,directory,provenance,pipeline_only=False)
        meta.update(output_semantics=kind,training_seen_positive_model=True,source_trained_model_sha256=sha(p),
            deployment_ready=False,actual_board_or_microphone_tested=False,
            probability_firmware_compatible_output=kind=='probability',
            new_readout_required=kind=='logits',actual_product_targets_verified=False)
        write_json(directory/'model_metadata.json',meta)
        if kind=='logits':
            h=directory/'dscnn_int8.h';h.write_text('// OUTPUT IS RAW LOGITS. Use the explicit int16 margin readout; probability thresholds are invalid.\n'+h.read_text())
        i=interpreter_for(tf,path);selected=predict(i,selection+probes);cal=predict(i,calibration)
        if kind=='logits':
            scale=meta['output']['scale'];zero=meta['output']['zero_point']
            selected={k:np.rint(v/scale+zero).astype(np.int16) for k,v in selected.items()}
            cal={k:np.rint(v/scale+zero).astype(np.int16) for k,v in cal.items()};mode='margin'
        else:mode='delivered_probability'
        results=[op.measure(selection+probes,selected,mode,1.,(fire,bell)) for fire in [(1,1),(2,3)] for bell in [(1,1),(2,3)]]
        best=max(results,key=op.key);thresholds=np.array(best['thresholds']);policy=best['policy']
        if kind=='logits':thresholds=np.floor(thresholds)+1
        maxima=np.stack([op.score(cal[r['sample_id']],mode,1.,policy).max(0) for r in calibration]).max(0)
        cal_thresholds=np.floor(maxima)+1 if kind=='logits' else np.nextafter(maxima,np.inf)
        thresholds=np.maximum(thresholds,cal_thresholds)
        merged=dict(selected,**cal);np.savez(directory/'selected_and_calibration_outputs.npz',**merged)
        config=dict(output_semantics=kind,score_mode=mode,policy=policy,thresholds=thresholds.tolist(),
            margin_units='int16 quantized-output differences' if kind=='logits' else 'Delivered dequantized probabilities',
            threshold_rule='At least one above-zero causal score >= threshold; selection/calibration maxima excluded strictly',
            firmware_frontend='Existing portable 98x40',input_shape=[1,98,40,1],
            trained_model_sha256=sha(p),tflite_sha256=sha(path),
            target_files_training_seen=True,independent_domestic_target_test_files=0,
            calibration_only_raises_selected_thresholds=True,calibration_other_files=len(calibration),
            notification_cooldown_implemented=False,board_tested=False,deployment_ready=False)
        write_json(directory/'readout_config.json',config)
        result=dict(kind=kind,selection_before_calibration=best,
            selection_after_calibration=fixed(selection+probes,merged,mode,policy,thresholds),
            calibration_after_threshold_fit=fixed(calibration,merged,mode,policy,thresholds),
            policy=config,calibration_is_threshold_fit_not_independent_test=True,
            fresh_test_used=False,product_targets_verified=False)
        records.append(result);write_json(directory/'analysis.json',result)
        print('DOMESTIC_SEEN_FULL_POSITIVE_INT8_ANALYSIS_COMPLETE',kind,result['selection_after_calibration'],flush=True)
    write_json(out/'status.json',dict(complete=True,source_model=NAME,source_model_sha256=sha(p),
        representative_windows=len(x),representative_role='train',results=records,
        known_target_probes_training_seen=True,fresh_test_used=False,product_targets_verified=False,
        deployment_ready=False,existing_firmware_or_model_bundle_replaced=False))


if __name__=='__main__':main()
