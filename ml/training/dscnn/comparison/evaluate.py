"""One frozen evaluation on new public sources and retrospective historical data."""
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from dscnn_baseline.common import runtime,source,config
from dscnn_baseline.evaluation import HistoricalHoldoutStore,compare
from dscnn_baseline.export import interpreter_for,predict_int8
from expansion.data import ExpandedFeatureStore
from .protocol import DATA,REPORT,ROOT,REPO
from .dataset import sha,write_json,write_csv,read_csv
from .metrics import LABELS,assess,apply,pool,arrays,grouped,decide
from .train import predict
from .temporal import evaluate as replay_metrics,added_delay,BASE_POLICY


class FreshStore(ExpandedFeatureStore):
    def __init__(self,rows):
        super().__init__();self.allowed={r['sample_id']:dict(r) for r in rows}
        assert rows and all(r['role']=='fresh_test' and r['split']=='test' for r in rows)
    def load(self,row):
        if self.allowed.get(row['sample_id'])!=row:raise ValueError('Fresh-test inventory differs from frozen membership')
        if sha(ROOT/row['relative_path'])!=row['source_sha256']:raise ValueError('Fresh-test source bytes changed')
        # This explicit evaluation-only boundary adapts the existing decoder;
        # the development loader continues to reject test-role rows.
        return super().load(dict(row,split='validation'))


def policy(path):
    j=json.loads(path.read_text())
    return tuple(j['thresholds'][k] for k in LABELS[:2]) if 'thresholds' in j else (j['threshold'],j['threshold'])


def registry():
    models=[]
    for tag,name in [('initial','dscnn_filelabels_20261114'),('previous_distilled','dscnn_improved_20261114'),('previous_expanded','dscnn_expanded_20261114')]:
        p=REPO/'ml/models'/name;t=policy(p/'decision_policy.json')
        models.append(dict(name=tag,float_path=str(p/'classifier.keras'),int8_path=str(p/'dscnn_int8.tflite'),float_thresholds=t,int8_thresholds=t,pooling_fraction=1.,historical_saved=True))
    frozen=json.loads((REPORT/'final_candidate.json').read_text())
    candidates=[('comparison_baseline',ROOT/'runs/dscnn_comparison_20261114_s1_baseline'),('final',Path(frozen['champion']['run']))]
    for tag,p in candidates:
        j=json.loads((REPORT/'calibration'/p.name/'results.json').read_text());o=json.loads((p/'run.json').read_text())['options']
        models.append(dict(name=tag,float_path=str(p/'classifier.keras'),int8_path=str(p/'export/dscnn_int8.tflite'),
                           float_thresholds=tuple(j['float_thresholds'][k] for k in LABELS[:2]),int8_thresholds=tuple(j['int8_thresholds'][k] for k in LABELS[:2]),pooling_fraction=o['pooling_fraction'],historical_saved=False))
    for m in models:m.update(float_sha256=sha(m['float_path']),int8_sha256=sha(m['int8_path']))
    return models


def any_target_recall(rows,thresholds):
    g=grouped(rows);p,y=arrays(g);pred=decide(p,thresholds)
    rates=[float((pred[y==k]!=2).mean()) for k in (0,1) if (y==k).any()]
    return float(np.mean(rates)) if len(rates)==2 else None


def paired_source_bootstrap(a,b,ta,tb):
    # Descriptive source-stratified uncertainty; tiny/related source counts do
    # not establish physical independence or a population false-alarm bound.
    ag=grouped(a);bg=grouped(b);assert [r['split_group_id'] for r in ag]==[r['split_group_id'] for r in bg]
    ap,y=arrays(ag);bp,_=arrays(bg);ac=decide(ap,ta)==y;bc=decide(bp,tb)==y
    rng=np.random.default_rng(20261114);delta=np.zeros(10000)
    for k in (0,1):
        mask=y==k;x=(bc[mask].astype(float)-ac[mask].astype(float));ids=rng.integers(len(x),size=(10000,len(x)));delta+=x[ids].mean(1)/2
    return dict(target_recall_difference=float(np.mean([(bc[y==k].mean()-ac[y==k].mean()) for k in (0,1)])),
                descriptive_95_percentile_interval=np.quantile(delta,[.025,.975]).tolist(),source_group_support={LABELS[k]:int((y==k).sum()) for k in (0,1)},
                resamples=10000,small_source_count=True,physical_source_independence_not_proven=True)


def evaluate():
    out=REPORT/'evaluation'
    if out.exists():raise FileExistsError('Fresh test already entered; do not retune or silently repeat')
    frozen=json.loads((DATA/'source_freeze.json').read_text())
    assert all(sha(DATA/k)==v for k,v in frozen['files'].items())
    event_policy=json.loads((REPORT/'temporal_replay/frozen_policy.json').read_text())
    models=registry()
    candidate=json.loads((REPORT/'final_candidate.json').read_text())
    assert candidate['dataset_freeze_sha256']==sha(DATA/'source_freeze.json')
    assert models[-1]['float_sha256']==candidate['model_sha256']==event_policy['model_sha256']
    assert models[-1]['int8_sha256']==event_policy['int8_sha256']
    out.mkdir();write_json(out/'preregistered_models.json',dict(models=models,dataset_freeze_sha256=sha(DATA/'source_freeze.json'),event_policy=event_policy,
                        thresholds_and_models_fixed_before_test=True,retune_from_test_forbidden=True))
    fresh=read_csv(DATA/'fresh_test.csv');store=FreshStore(fresh);fresh_features={};durations={}
    for r in fresh:
        x,m=store.load(r);fresh_features[r['sample_id']]=x;durations[r['sample_id']]=m['samples']/16000
    folds,_=source.load_splits(config()[1]);historical=[r for r in folds[0] if r['split']=='test']
    exclusions=set(json.loads((ROOT/'reports/expansion_20261114/historical_holdout/evaluation_scope.json').read_text())['excluded_ids'])
    historical=[r for r in historical if r['sample_id'] not in exclusions];assert len(historical)==403
    hs=HistoricalHoldoutStore(historical,models[-1]['float_sha256']);historical_features={r['sample_id']:hs.load(r)[0] for r in historical}
    tf,keras=runtime('gpu');results=[];fresh_predictions={};final_windows=None
    for m in models:
        keras.backend.clear_session()
        with tf.device('/GPU:0'):net=keras.models.load_model(m['float_path'],compile=False)
        integer=interpreter_for(tf,m['int8_path'])
        for scope,rr,features in [('fresh_test',fresh,fresh_features),('historical',historical,historical_features)]:
            dest=out/scope/m['name'];dest.mkdir(parents=True)
            if scope=='historical' and m['historical_saved']:
                roots={'initial':ROOT/'reports/real_training_20261114/study_calibrated_v2_resumed/historical_holdout',
                       'previous_distilled':ROOT/'reports/improvement_20261114/historical_holdout',
                       'previous_expanded':ROOT/'reports/expansion_20261114/historical_holdout'}
                allowed={r['sample_id'] for r in historical}
                def restore_metadata(path):
                    saved={r['sample_id']:r for r in read_csv(path) if r['sample_id'] in allowed}
                    assert set(saved)==allowed,'Historical predictions do not match the frozen cohort'
                    for r in historical:
                        assert saved[r['sample_id']]['label']==r['label']
                        assert saved[r['sample_id']]['split_group_id']==r['split_group_id']
                    # Older prediction CSVs omit sampler_pool. Restore the
                    # original metadata so the domestic doorlock never enters
                    # the foreign-negative denominator, and preserve row order.
                    return [dict(saved[r['sample_id']],**r) for r in historical]
                floating=restore_metadata(roots[m['name']]/'float_clips.csv')
                quantized=restore_metadata(roots[m['name']]/'int8_clips.csv')
            else:
                with tf.device('/GPU:0'):
                    floating,fw=predict(net,rr,features,m['pooling_fraction'],return_windows=True)
                quantized=[];iw={};started=time.monotonic();windows_count=0
                for r in rr:
                    p=predict_int8(integer,features[r['sample_id']]);v=pool(p,m['pooling_fraction']);windows_count+=len(p)
                    quantized.append(dict(r,**{'p_'+k:float(vv) for k,vv in zip(LABELS,v)},patch_count=len(p)))
                    if scope=='fresh_test' and m['name']=='final':iw[r['sample_id']]=p
                write_json(dest/'integer_host_timing.json',dict(windows=windows_count,total_seconds=time.monotonic()-started,backend='Existing host TFLite reference kernels, not board latency'))
                if scope=='fresh_test' and m['name']=='final':final_windows=dict(float=fw,int8=iw)
            assert len(floating)==len(quantized)==len(rr)
            for backend,pred in [('float',floating),('int8',quantized)]:
                t=m[backend+'_thresholds'];metrics=assess(pred,t);metrics['any_target_alert_macro_recall']=any_target_recall(pred,t)
                assert metrics['foreign_negative_clips']==(394 if scope=='historical' else 67)
                write_csv(dest/(backend+'_clips.csv'),apply(pred,t));write_json(dest/(backend+'_metrics.json'),metrics)
                results.append(dict(scope=scope,model=m['name'],backend=backend,metrics=metrics))
                if scope=='fresh_test':fresh_predictions[m['name'],backend]=pred
            write_json(dest/'float_int8_comparison.json',compare(apply(floating,m['float_thresholds']),apply(quantized,m['int8_thresholds'])))
            print('FROZEN_EVALUATION',scope,m['name'],'int8_recall',results[-1]['metrics']['target_macro_group_recall'],'fpr',results[-1]['metrics']['foreign_fpr'],flush=True)
    replay={}
    for backend in ('float','int8'):
        thr=event_policy[backend+'_thresholds'];policy_value=event_policy['policy'];w=final_windows[backend]
        replay[backend]=dict(single_at_same_thresholds=replay_metrics(fresh,w,durations,thr,BASE_POLICY),chosen=replay_metrics(fresh,w,durations,thr,policy_value),additional_delay=added_delay(fresh,w,thr,policy_value,durations))
        np.savez(out/('fresh_final_'+backend+'_windows.npz'),**w)
    write_json(out/'fresh_temporal_replay.json',replay)
    old=next(m for m in models if m['name']=='previous_expanded');new=next(m for m in models if m['name']=='final')
    uncertainty=paired_source_bootstrap(fresh_predictions['previous_expanded','int8'],fresh_predictions['final','int8'],old['int8_thresholds'],new['int8_thresholds'])
    write_json(out/'fresh_paired_source_uncertainty.json',uncertainty)
    write_json(out/'results.json',results)
    final=next(r['metrics'] for r in results if r['scope']=='fresh_test' and r['model']=='final' and r['backend']=='int8')
    previous=next(r['metrics'] for r in results if r['scope']=='fresh_test' and r['model']=='previous_expanded' and r['backend']=='int8')
    write_json(out/'status.json',dict(complete=True,models=len(models),fresh_test_clips=len(fresh),historical_clips=len(historical),excluded_historical_sirens=sorted(exclusions),
               no_test_driven_reselection=True,observed_new_test_fpr_le_5_percent=final['foreign_fpr']<=.05,
               fresh_target_recall_not_lower=final['target_macro_group_recall']>=previous['target_macro_group_recall'],fresh_fpr_not_higher=final['foreign_fpr']<=previous['foreign_fpr'],
               actual_event_onset_latency_unverified=True,real_microphone_or_board_runs=0,field_deployment_ready=False,
               fresh_test_independence_basis='Publisher recording identities + related-source families + exact source and PCM checks; physical device/country/full excerpt reuse not certified'))
    print('FROZEN_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':evaluate()
