"""Compare readout operating points on selection recordings only.

Logit margins and softened probabilities avoid treating rounded softmax 1.0
values as infinitely strong evidence. No setting is tuned on fresh test data.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from dscnn_baseline.common import runtime
from comparison.dataset import read_csv,write_csv,write_json,sha
from .paths import ROOT,DATA,REPORT
from .data import feature,CACHE
from .context import Context

LABELS=['fire_alarm','doorbell','other']


def softmax(z,temperature=1.):
    a=np.asarray(z,dtype=np.float64)/temperature;a-=a.max(1,keepdims=True);e=np.exp(a);return e/e.sum(1,keepdims=True)


def score(z,mode,temperature,policy):
    z=np.asarray(z,dtype=np.float64);arg=z.argmax(1);p=softmax(z,temperature);a=np.zeros((len(z),2),np.float64)
    for k in (0,1):
        raw=p[:,k] if mode=='probability' else z[:,k]-np.delete(z,k,axis=1).max(1)
        raw=np.where(arg==k,raw,0.);need,width=policy[k];history=[]
        for i,value in enumerate(raw):
            history.append(value);history=history[-width:];a[i,k]=sorted(history,reverse=True)[need-1] if len(history)>=need else 0
    return a


def measure(rows,logits,mode='probability',temperature=1.,policy=((1,1),(1,1))):
    maximum=np.stack([score(logits[r['sample_id']],mode,temperature,policy).max(0) for r in rows]);y=np.array([LABELS.index(r['label']) for r in rows]);thresholds=[]
    for k in (0,1):thresholds.append(float(np.nextafter(max(0.,float(maximum[y!=k,k].max())),np.inf)))
    flags=(maximum>=thresholds)&(maximum>0);recall=[];classes={}
    for k,label in enumerate(LABELS[:2]):
        grouped=defaultdict(list)
        for i,r in enumerate(rows):
            if y[i]==k:grouped[r['split_group_id']].append(float(flags[i,k]))
        rate=float(np.mean([np.mean(v) for v in grouped.values()]));recall.append(rate)
        classes[label]=dict(files=int((y==k).sum()),detected_files=int(flags[y==k,k].sum()),source_groups=len(grouped),source_mean_file_recall=rate,
             wrong_target_or_negative_alert_files=int(flags[y!=k,k].sum()))
    return dict(mode=mode,temperature=temperature,policy=policy,thresholds=thresholds,fire_recall=recall[0],doorbell_recall=recall[1],
       macro_recall=float(np.mean(recall)),classes=classes,other_false_alert_files=int(flags[y==2].any(1).sum()),other_files=int((y==2).sum()),
       cross_or_negative_alert_files=sum(c['wrong_target_or_negative_alert_files'] for c in classes.values()),
       selection_zero_errors_by_construction=True,true_event_or_monthly_rate_verified=False)


def key(m):return (min(m['fire_recall']/.999,m['doorbell_recall']/.95),m['macro_recall'])


def main(dataset,names):
    out=REPORT/'operating'/dataset;out.mkdir(parents=True,exist_ok=True);d=DATA/dataset;freeze=json.loads((d/'freeze.json').read_text());assert all(sha(d/k)==v for k,v in freeze['files'].items())
    rows=[r for r in read_csv(d/'development.csv') if r['role']=='selection'];assert rows and all(r['role']=='selection' for r in rows)
    # Explicit host-CPU analysis runs alongside the owned GPU training process.
    tf,keras=runtime('cpu');assert not tf.config.list_physical_devices('GPU');ledger=[]
    for name in names:
        target=out/(name+'.json')
        if target.exists():raise FileExistsError('Preserve completed operating-point analysis')
        run=ROOT/'runs'/name;status=json.loads((run/'status.json').read_text());assert status['complete'] and not status.get('target_results_training_seen',False)
        training=[r for r in read_csv(run/'used_train_selection.csv') if r['role']=='train'];assert {r['split_group_id'] for r in training}.isdisjoint(r['split_group_id'] for r in rows)
        model_path=run/'classifier.keras';mh=sha(model_path);assert mh==status['model_sha256']
        with tf.device('/CPU:0'):model=keras.models.load_model(str(model_path),compile=False);net=keras.Model(model.input,model.get_layer('class_logits').output)
        assert len(net.input_shape) in (2,4),'Expected compact CNN or aligned YAMNet head'
        logits={};provenance=[]
        for r in rows:
            if len(net.input_shape)==4 and tuple(net.input_shape[1:])==(96,64,1):
                from .yamnet_frontend import feature as official_feature,CACHE as OFFICIAL_CACHE
                m=official_feature(r);a=np.load(OFFICIAL_CACHE/m['source_sha256']/'features.npy',mmap_mode='r',allow_pickle=False)
            elif len(net.input_shape)==2:
                m=feature(r)
                from .embeddings import load as aligned_embeddings
                a,_=aligned_embeddings(r)
            else:
                m=feature(r)
                a=np.load(CACHE/m['source_sha256']/'features.npy',mmap_mode='r',allow_pickle=False)
                if net.input_shape[-1]>1:a=Context(a,net.input_shape[-1])
            with tf.device('/CPU:0'):z=np.concatenate([net(a[i:i+128],training=False).numpy() for i in range(0,len(a),128)])
            logits[r['sample_id']]=z;provenance.append(dict(sample_id=r['sample_id'],source_sha256=m['source_sha256'],feature_sha256=m['feature_sha256']))
        cache=out/(name+'_selection_logits.npz');np.savez(cache,**logits);results=[]
        for mode,temperatures in [('probability',[1.,2.,4.,8.]),('margin',[1.])]:
            for temperature in temperatures:
                for fire in [(1,1),(2,3)]:
                    for bell in [(1,1),(2,3)]:results.append(measure(rows,logits,mode,temperature,(fire,bell)))
        best=max(results,key=key);assert sha(model_path)==mh
        record=dict(complete=True,model=name,model_sha256=mh,dataset=dataset,dataset_freeze_sha256=sha(d/'freeze.json'),execution='Explicit host CPU, no GPU context',
           best=best,results=results,feature_provenance=provenance,logit_cache_sha256=sha(cache),fresh_test_used=False,product_targets_verified=False)
        write_json(target,record);ledger.append(dict(model=name,**best));write_csv(out/'summary.csv',ledger)
        print('OPERATING_SELECTION_COMPLETE',name,best,flush=True);keras.backend.clear_session()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('names',nargs='+');a=p.parse_args();main(a.dataset,a.names)
