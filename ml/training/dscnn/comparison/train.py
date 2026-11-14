"""Real GPU training, selection-only early stop, immutable per-run provenance."""
import argparse
import json
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np
from dscnn_baseline.common import runtime
from dscnn_baseline.model import complexity
from expansion.model import build_teacher_head
from .protocol import ROOT,RUNS,REPORT,BASE
from .dataset import sha,write_csv,write_json
from .features import rows,load,categories
from .sampling import Sampler
from .model import build,update_function
from .metrics import LABELS,pool,calibrate,rank


def predict(model,rr,features,fraction=1.,teacher=False,return_windows=False):
    outputs=[];windows={}
    for r in rr:
        x=features[r['sample_id']];parts=[]
        for start in range(0,len(x),128):parts.append(model(np.asarray(x[start:start+128]),training=False).numpy())
        p=np.concatenate(parts)
        if teacher:
            p=np.exp(p-p.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        if not np.isfinite(p).all():raise ValueError('Non-finite model output')
        v=pool(p,fraction);outputs.append(dict(r,**{'p_'+k:float(x) for k,x in zip(LABELS,v)},patch_count=len(p)))
        if return_windows:windows[r['sample_id']]=p
    return (outputs,windows) if return_windows else outputs


def train(spec):
    name=spec['name'];assert Path(name).name==name
    out=RUNS/name
    if out.exists():raise FileExistsError('Preserve prior run; inspect process identity instead of restarting')
    out.mkdir();started=time.monotonic();status=dict(complete=False,exit_code=None,actual_optimizer_steps=0,fresh_test_used=False,calibration_used_for_selection=False)
    write_json(out/'status.json',status)
    try:
        o=dict(BASE,**spec.get('options',{}));seed=spec.get('seed',42);teacher=spec.get('role')=='teacher'
        rr=categories(rows(o['added_training_data']));training=[r for r in rr if r['role']=='train'];validation=[r for r in rr if r['role']=='selection']
        assert training and validation
        features,provenance=load(rr,teacher=teacher);write_json(out/'feature_provenance.json',provenance);write_csv(out/'used_train_selection.csv',rr)
        code={str(p.relative_to(ROOT)):sha(p) for folder in ('comparison','expansion','improvement','dscnn_baseline') for p in (ROOT/folder).glob('*.py')}
        write_json(out/'run.json',dict(spec=spec,options=o,source_identity=sha(ROOT/'data/comparison_20261114/development.csv'),code_identity=code,
                   supervision='file-label bags; no reviewed event timestamps',test_used=False,calibration_used=False))
        for folder in ('comparison',):shutil.copytree(ROOT/folder,out/'source_snapshot'/folder,ignore=shutil.ignore_patterns('__pycache__'))
        tf,keras=runtime('gpu');tf.config.experimental.enable_op_determinism();keras.utils.set_random_seed(seed)
        targets={};teacher_receipt=None
        if not teacher and o['kd_weight']>0:
            headpath=Path(spec['teacher_run'])/'classifier.keras'
            with tf.device('/GPU:0'):head=keras.models.load_model(str(headpath),compile=False)
            head.trainable=False;before=sha(headpath);emb,embprov=load(training,teacher=True)
            for r in training:targets[r['sample_id']]=head(np.asarray(emb[r['sample_id']]),training=False).numpy().astype(np.float32)
            assert sha(headpath)==before
            headmeta=json.loads((headpath.parent/'run.json').read_text())
            headrows=json.loads((headpath.parent/'feature_provenance.json').read_text())
            assert {r['sample_id'] for r in training}=={r['sample_id'] for r in headrows if r['role']=='train'},'Teacher/student training inventories differ'
            np.savez(out/'training_teacher_logits.npz',**targets)
            teacher_receipt=dict(run=str(headpath.parent),model_sha256=before,logits_sha256=sha(out/'training_teacher_logits.npz'),same_train_only_sources=True)
            write_json(out/'teacher_provenance.json',teacher_receipt);del head,emb
            keras.backend.clear_session();keras.utils.set_random_seed(seed)
        with tf.device('/GPU:0'):
            model=build_teacher_head(keras,seed) if teacher else build(keras,o,seed)
            optimizer=keras.optimizers.Adam(o['learning_rate'])
        update=update_function(tf,keras,model,optimizer,o,teacher)
        sampler=Sampler(training,features,o,seed);best=None;stale=0;lr_stale=0;history=[];selected=None
        for epoch in range(1,o['epochs']+1):
            losses=[]
            for _ in range(o['steps_per_epoch']):
                batches=[]
                for x,w,y,ids in sampler.batches():
                    t=np.zeros((len(x),3),np.float32) if teacher or o['kd_weight']==0 else np.concatenate([targets[s] for s in ids])
                    batches.append((x,w,y,t))
                v=update(batches).numpy()
                if not np.isfinite(v).all():raise ValueError('Non-finite optimizer loss')
                losses.append(v)
            with tf.device('/GPU:0'):raw=predict(model,validation,features,1. if teacher else o['pooling_fraction'],teacher)
            thresholds,m=calibrate(raw);key=rank(m);improved=best is None or key>best
            if improved:
                best=key;stale=0;lr_stale=0;selected=epoch
                model.save(str(out/'classifier.keras'));write_csv(out/'selection_raw.csv',raw);write_json(out/'selection_metrics.json',m)
                write_json(out/'selection_policy.json',dict(thresholds=dict(zip(LABELS[:2],thresholds)),selection_only=True))
            else:stale+=1;lr_stale+=1
            mean=np.mean(losses,axis=0)
            history.append(dict(epoch=epoch,train_loss=float(mean[0]),hard_label_loss=float(mean[1]),distillation_loss_T_squared=float(mean[2]),
                target_macro_group_recall=m['target_macro_group_recall'],domestic_target_macro_group_recall=m['domestic_target_macro_group_recall'],
                foreign_fpr=m['foreign_fpr'],doorlock_fp=m['domestic_doorlock_fp'],guards_passed=m['guards_passed'],
                learning_rate=float(keras.backend.get_value(optimizer.learning_rate)),saved=improved))
            write_csv(out/'history.csv',history)
            write_csv(out/'sample_exposure.csv',[dict(sample_id=r['sample_id'],label=r['label'],split_group_id=r['split_group_id'],draws=sampler.exposure[r['sample_id']]) for r in training])
            status.update(epoch=epoch,selected_epoch=selected,actual_optimizer_steps=int(optimizer.iterations.numpy()),elapsed_seconds=time.monotonic()-started,latest_selection=m)
            write_json(out/'status.json',status);print('COMPARISON_EPOCH',name,epoch,history[-1],flush=True)
            if o['lr_schedule']=='plateau' and lr_stale>=3:
                old=float(keras.backend.get_value(optimizer.learning_rate));keras.backend.set_value(optimizer.learning_rate,max(old*.5,1e-5));lr_stale=0
            if stale>=o['patience']:break
        with tf.device('/GPU:0'):restored=keras.models.load_model(str(out/'classifier.keras'),compile=False)
        if any('GPU:0' not in v.device for v in restored.variables):raise RuntimeError('Restored model variables not on selected GPU')
        pm=predict(restored,validation,features,1. if teacher else o['pooling_fraction'],teacher)
        _,check=calibrate(pm);saved=json.loads((out/'selection_metrics.json').read_text())
        if rank(check)!=rank(saved):raise ValueError('Save/reload prediction mismatch')
        status.update(complete=True,exit_code=0,selected_epoch=selected,epochs_completed=len(history),model_sha256=sha(out/'classifier.keras'),
                      selected_metrics=saved,complexity=None if teacher else complexity(restored,keras),
                      all_training_positives_sampled=all(sampler.exposure[r['sample_id']]>0 for r in training if r['label']!='other'),
                      negative_files=len([r for r in training if r['label']=='other']),negative_unique_sampled=sum(sampler.exposure[r['sample_id']]>0 for r in training if r['label']=='other'),
                      elapsed_seconds=time.monotonic()-started)
        write_json(out/'status.json',status);print('COMPARISON_TRAIN_COMPLETE',name,selected,flush=True)
    except Exception as e:
        status.update(exit_code=1,first_error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started);write_json(out/'status.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec',type=Path);a=p.parse_args();train(json.loads(a.spec.read_text()))
