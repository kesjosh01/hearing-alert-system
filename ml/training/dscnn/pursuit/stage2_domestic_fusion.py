"""Frozen native-CNN probability fusion; fit only on development/known probes."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
from comparison.dataset import read_csv, write_json, sha
from dscnn_baseline.common import runtime
from .paths import ROOT, DATA, REPORT
from .stage2_stream import FeatureStore
from .stage2_native_export import predict
from .cpu_integer import open_interpreter
from .stage2_known_probe import PROBES, OUT as KNOWN_OUT
from .final_public_benchmark import summary
from . import delivered_integer as di

PLAN = REPORT / 'stage2_domestic_fusion_plan.json'
OUT = REPORT / 'stage2_domestic_fusion_v1'
COHORT = DATA / 'round06_local_regions'


def probability(q, meta):
    scale, zero = meta['output']['scale'], meta['output']['zero_point']
    p = np.maximum((q.astype(np.float64)-zero)*scale, scale*.5)
    return p/p.sum(1, keepdims=True)


def raw_scores(q1, q2, meta1, meta2, alpha):
    assert q1.shape == q2.shape and q1.shape[1] == 3
    z = alpha*np.log(probability(q1, meta1))+(1-alpha)*np.log(probability(q2, meta2))
    arg = z.argmax(1)
    return np.stack([np.where(arg == k, np.floor(np.maximum(z[:, k]-np.delete(z, k, axis=1).max(1), 0.)*128), 0)
                     for k in (0, 1)], axis=1).astype(np.int32)


def temporal(raw, policy):
    out = np.zeros_like(raw)
    for k, (need, width) in enumerate(policy):
        if (need, width) == (1, 1):
            out[:, k] = raw[:, k]
        else:
            assert (need, width) == (2, 3)
            a = np.pad(raw[:, k], (2, 0))
            out[:, k] = np.sort(np.lib.stride_tricks.sliding_window_view(a, 3), axis=1)[:, 1]
    return out


def metrics(rows, maxima, cuts):
    values = {r['sample_id']:maxima[r['sample_id']][None] for r in rows}
    # di.measure uses its score function; already-computed positive margins can
    # instead be represented as q=[margin0,margin1,0] with one active class at a
    # time only at frame level. File envelopes may contain both target classes,
    # so aggregate them directly rather than re-applying argmax at file level.
    flags = np.stack([maxima[r['sample_id']] >= cuts for r in rows])
    classes, conditions = {}, {}
    for k, label in enumerate(di.LABELS[:2]):
        grouped = {}
        for i, row in enumerate(rows):
            if row['label'] == label: grouped.setdefault(row['split_group_id'], []).append(bool(flags[i, k]))
        classes[label] = dict(files=sum(r['label'] == label for r in rows),
            detected_files=sum(bool(flags[i, k]) for i,r in enumerate(rows) if r['label'] == label),
            source_groups=len(grouped), source_mean_file_recall=float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None)
    for condition in sorted({r.get('probe_condition') or 'original' for r in rows if r['label'] != 'other'}):
        conditions[condition] = {}
        for k, label in enumerate(di.LABELS[:2]):
            mask = [i for i,r in enumerate(rows) if r['label'] == label and (r.get('probe_condition') or 'original') == condition]
            conditions[condition][label] = dict(files=len(mask), detected_files=sum(bool(flags[i,k]) for i in mask))
    return dict(fire_recall=classes['fire_alarm']['source_mean_file_recall'],
        doorbell_recall=classes['doorbell']['source_mean_file_recall'], classes=classes, conditions=conditions,
        other_files=sum(r['label']=='other' for r in rows),
        other_false_alert_files=sum(bool(flags[i].any()) for i,r in enumerate(rows) if r['label']=='other'),
        wrong_target_alert_files=sum(bool(flags[i,1-di.LABELS.index(r['label'])]) for i,r in enumerate(rows) if r['label']!='other'),
        cuts_integer_units=cuts.tolist(), actual_event_or_monthly_rate_verified=False)


def fit_cuts(rows, maxima, lower=None):
    cut = np.array([max(int(maxima[r['sample_id']][k]) for r in rows if r['label'] != di.LABELS[k])+1 for k in (0,1)], np.int32)
    return np.maximum(cut, lower) if lower is not None else cut


def choice_key(known, general):
    worst = [min(v[label]['detected_files']/v[label]['files'] for v in known['conditions'].values())
             for label in di.LABELS[:2]]
    return (min(worst[0]/.999, worst[1]/.95), sum(worst)/2,
            min(general['fire_recall']/.999, general['doorbell_recall']/.95),
            (general['fire_recall']+general['doorbell_recall'])/2)


def fit():
    assert not OUT.exists(), 'Preserve the completed or partial fusion study'
    spec = json.loads(PLAN.read_text())
    names = [spec['specialist'], spec['general']]
    bundles = [REPORT/'candidates'/n for n in names]
    metas = [json.loads((p/'model_metadata.json').read_text()) for p in bundles]
    assert all(m['output_semantics']=='probability' for m in metas)
    for name,p in zip(names,bundles):
        assert sha(p/'model_int8.tflite') == spec['source_model_sha256'][name]
        assert metas[names.index(name)]['input']['shape']==[1,98,40,1]
    rows = read_csv(COHORT/'development.csv')
    selected = [r for r in rows if r['role']=='selection']
    calibration = [r for r in rows if r['role']=='calibration']
    known = [dict(r,sample_id='known__'+r['sample_id']) for r in read_csv(PROBES)]
    assert len(known)==172 and all(r['role']=='train' for r in known)
    OUT.mkdir()
    write_json(OUT/'design_freeze.json',dict(plan_sha256=sha(PLAN),code_sha256=sha(Path(__file__)),
        data_freeze_sha256=sha(COHORT/'freeze.json'), source_models_sha256=spec['source_model_sha256'],
        new_public_outputs_read=False,new_public_outputs_used_to_choose=False))
    tf,_=runtime('cpu');assert not tf.config.list_physical_devices('GPU')
    gate=json.loads((REPORT/'final_public_benchmark_v1'/(names[0]+'_cpu_kernel_gate.json')).read_text())
    assert gate['source_model_sha256']==spec['source_model_sha256'][names[0]]
    interpreter=open_interpreter(tf,bundles[0]/'model_int8.tflite',bool(gate['passed']),2)
    store=FeatureStore(rows,tf,kind='portable40',budget_bytes=1536*2**20)
    first=predict(interpreter,selected+calibration,store)
    identities=json.loads((KNOWN_OUT/'freeze.json').read_text())['read_only_feature_identities']
    for row in known:
        key=row['sample_id'].removeprefix('known__');info=identities[key]
        assert sha(Path(info['path']))==info['sha256']
        x=np.load(info['path'],mmap_mode='r',allow_pickle=False)
        first[row['sample_id']]=di.predict_q(interpreter,x)
    with np.load(REPORT/'stage2_native40_delivered_v1/chosen_selection_and_calibration_q.npz',allow_pickle=False) as d:
        second={k:d[k] for k in d.files}
    alias=json.loads((REPORT/'stage2_paired_integer_v1/status.json').read_text())['global_bundle_equivalent_to']
    with np.load(KNOWN_OUT/(alias+'_q.npz'),allow_pickle=False) as d:
        second.update({'known__'+k:d[k] for k in d.files})
    for row in selected+calibration+known:
        assert first[row['sample_id']].shape==second[row['sample_id']].shape
    np.savez_compressed(OUT/'specialist_development_q.npz',**first)
    np.savez_compressed(OUT/'general_development_q.npz',**second)
    trials=[]
    for alpha in spec['weights']:
        raw={r['sample_id']:raw_scores(first[r['sample_id']],second[r['sample_id']],*metas,alpha) for r in selected+known}
        for policy in spec['policies']:
            maxima={k:temporal(v,policy).max(0) for k,v in raw.items()}
            cuts=fit_cuts(selected+known,maxima)
            km,gm=metrics(known,maxima,cuts),metrics(selected,maxima,cuts)
            trials.append(dict(alpha=alpha,policy=policy,cuts=cuts.tolist(),known=km,selection=gm,key=choice_key(km,gm)))
        print('FUSION_DEVELOPMENT_ALPHA_COMPLETE',alpha,flush=True)
    chosen=max(trials,key=lambda t:tuple(t['key']))
    write_json(OUT/'chosen_before_calibration.json',dict(chosen=chosen,trials=trials,
        calibration_or_new_public_outputs_used_to_choose=False))
    alpha,policy=chosen['alpha'],chosen['policy']
    all_max={r['sample_id']:temporal(raw_scores(first[r['sample_id']],second[r['sample_id']],*metas,alpha),policy).max(0) for r in selected+calibration+known}
    cuts=fit_cuts(calibration,all_max,np.array(chosen['cuts'],np.int32))
    km,gm,cm=metrics(known,all_max,cuts),metrics(selected,all_max,cuts),metrics(calibration,all_max,cuts)
    result=dict(complete=True,plan_sha256=sha(PLAN),code_sha256=sha(Path(__file__)),alpha=alpha,policy=policy,
        source_names=names,source_models_sha256=spec['source_model_sha256'],output_metadata=[m['output'] for m in metas],
        cuts_integer_units=cuts.tolist(),known_after_calibration=km,selection_after_calibration=gm,calibration_after_fit=cm,
        prior_selected=chosen,new_public_test_used=False,feature_arrays_written=False,actual_product_targets_verified=False)
    assert cm['other_false_alert_files']==0 and cm['wrong_target_alert_files']==0
    write_json(OUT/'status.json',result)
    print('DOMESTIC_FUSION_DEVELOPMENT_COMPLETE',alpha,policy,cuts.tolist(),km['conditions']['original'],flush=True)


def test():
    fixed=json.loads((OUT/'status.json').read_text());assert fixed['complete'] and not fixed['new_public_test_used']
    assert sha(Path(__file__))==fixed['code_sha256'] and sha(PLAN)==fixed['plan_sha256']
    assert not (OUT/'new_public_results.json').exists()
    base=REPORT/'stage2_public_benchmark_v1';frozen=json.loads((base/'freeze.json').read_text())
    assert frozen['global_winner_alias']==json.loads((REPORT/'stage2_paired_integer_v1/status.json').read_text())['global_bundle_equivalent_to']
    rows=read_csv(COHORT/'fresh_test.csv');assert len(rows)==559
    for name,expected in fixed['source_models_sha256'].items():assert sha(REPORT/'candidates'/name/'model_int8.tflite')==expected
    write_json(OUT/'test_freeze.json',dict(fusion_status_sha256=sha(OUT/'status.json'),base_test_freeze_sha256=sha(base/'freeze.json'),
        base_manifest_sha256=sha(COHORT/'fresh_test.csv'),readout_frozen_before_opening_public_q_traces=True,
        no_test_based_tuning=True))
    metas=[dict(output=m) for m in fixed['output_metadata']];maxima={};traces={}
    for i,row in enumerate(rows,1):
        record=json.loads((base/'recordings'/(row['sample_id']+'.json')).read_text())
        assert record['complete'] and record['freeze_sha256']==sha(base/'freeze.json')
        path=base/'q_traces'/record['trace_filename'];assert sha(path)==record['trace_sha256']
        with np.load(path,allow_pickle=False) as q:
            a=q[fixed['source_names'][0]];b=q[frozen['global_winner_alias']]
        scores=temporal(raw_scores(a,b,*metas,fixed['alpha']),fixed['policy'])
        maxima[row['sample_id']]=scores.max(0);traces[row['sample_id']]=scores
        if i%100==0:print('FROZEN_FUSION_PUBLIC_RECORDINGS',i,len(rows),flush=True)
    np.savez_compressed(OUT/'new_public_fusion_integer_scores.npz',**traces)
    metric=metrics(rows,maxima,np.array(fixed['cuts_integer_units'],np.int32))
    write_json(OUT/'new_public_results.json',dict(complete=True,metric=metric,readout_status_sha256=sha(OUT/'status.json'),
        original_eight_model_results_unchanged=True,test_based_model_or_threshold_tuning=False,
        true_activation_recall_or_device_monthly_false_rate_verified=False,actual_product_targets_verified=False))
    print('FROZEN_FUSION_NEW_PUBLIC_RESULT',metric,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['fit','test']);args=parser.parse_args()
    fit() if args.mode=='fit' else test()
