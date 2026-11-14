"""Matched random-vs-YAMNet-initialized prefix CNN, same native frontend."""
import argparse
import json
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
from dscnn_baseline.common import runtime
from .cost import complexity
from .yamnet_prefix import build_model
from comparison.protocol import BASE
from comparison.model import update_function
from comparison.sampling import Sampler
from comparison.dataset import read_csv,write_csv,write_json,sha
from .paths import ROOT,DATA,REPORT
from .data import CACHE,feature
from .metrics import calibrate,rank
from .teacher import load as teacher_targets
from .strong import StrongView


def predict(model,rr,features):
    result={}
    for r in rr:
        a=features[r['sample_id']];result[r['sample_id']]=np.concatenate([model(a[i:i+128],training=False).numpy() for i in range(0,len(a),128)])
    return result


def train(spec):
    name=spec['name'];assert Path(name).name==name
    out=ROOT/'runs'/name;out.mkdir(exist_ok=False);status=dict(complete=False,exit_code=None,actual_optimizer_steps=0,fresh_test_used=False,product_targets_verified=False)
    write_json(out/'status.json',status);started=time.monotonic()
    try:
        o=dict(BASE,negative_sampler='coverage_cycle',pooling_fraction=.5,kd_weight=0.,epochs=60,patience=12);o.update(spec.get('options',{}))
        channels=spec.get('channels',24);policy=spec.get('policy',[[1,1],[1,1]])
        d=DATA/spec['dataset'];freeze=json.loads((d/'freeze.json').read_text());assert all(sha(d/k)==v for k,v in freeze['files'].items())
        rr=[r for r in read_csv(d/'development.csv') if r['role'] in ('train','selection')];assert all(r['role']!='fresh_test' for r in rr)
        train=[r for r in rr if r['role']=='train'];selection=[r for r in rr if r['role']=='selection'];features={};provenance=[]
        assert {r['split_group_id'] for r in train}.isdisjoint(r['split_group_id'] for r in selection)
        for r in rr:
            m=feature(r);features[r['sample_id']]=np.load(CACHE/m['source_sha256']/'features.npy',mmap_mode='r',allow_pickle=False);provenance.append(dict(sample_id=r['sample_id'],role=r['role'],**m))
        assert not spec.get('teacher') and o['kd_weight']==0,'Strong cropped bags require matching teacher index mapping before KD'
        strong_inputs=[]
        for row in train:
            if row.get('event_intervals_json'):
                intervals=json.loads(row['event_intervals_json'])
                if intervals:
                    m=feature(row);view=StrongView(features[row['sample_id']],m['starts'],intervals);features[row['sample_id']]=view
                    strong_inputs.append(dict(sample_id=row['sample_id'],role='train',feature_window_indices=view.indices.tolist(),source_intervals=intervals,full_source_windows=m['window_count']))
        write_json(out/'strong_training_windows.json',strong_inputs)
        targets={};teacher_receipts=[]
        if spec.get('teacher'):
            for row in train:
                logits,receipt=teacher_targets(row,spec['teacher']);targets[row['sample_id']]=logits;teacher_receipts.append(dict(sample_id=row['sample_id'],**receipt))
        elif o['kd_weight']!=0:raise ValueError('KD needs a hash-bound trained teacher')
        write_json(out/'teacher_provenance.json',teacher_receipts)
        # Native decoding may reveal an exact duplicate not identifiable by
        # source ID/container bytes. Stop before any optimizer update.
        pcm_roles={}
        for r in provenance:
            pcm_roles.setdefault(r['decoded_pcm_sha256'],set()).add(r['role'])
        assert all(len(v)==1 for v in pcm_roles.values()),'Decoded PCM crosses fitting/selection roles'
        write_csv(out/'used_train_selection.csv',rr);write_json(out/'feature_provenance.json',provenance)
        write_json(out/'run.json',dict(spec=spec,options=o,channels=channels,policy=policy,dataset_freeze_sha256=sha(d/'freeze.json'),
          code_identity={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'pursuit').glob('*.py')},supervision='Weak file-label bags; no human event timestamps',
          selection_proxy='Zero observed other and wrong-target-class alerts, maximize smaller recall/target ratio',calibration_used=False,fresh_test_used=False))
        shutil.copytree(ROOT/'pursuit',out/'source_snapshot/pursuit',ignore=shutil.ignore_patterns('__pycache__'))
        tf,keras=runtime('gpu');tf.config.experimental.enable_op_determinism();keras.utils.set_random_seed(spec.get('seed',42))
        with tf.device('/GPU:0'):
            model=build_model(keras,dict(prefix_layers=spec['prefix_layers'],pretrained=spec['pretrained']),spec.get('seed',42))
            if spec.get('initialize_from'):
                checkpoint=ROOT/'runs'/spec['initialize_from']/'classifier.keras';previous=keras.models.load_model(str(checkpoint),compile=False)
                assert json.loads((checkpoint.parent/'status.json').read_text())['complete']
                model.set_weights(previous.get_weights());write_json(out/'initialization.json',dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),optimizer_restarted=True))
            optimizer=keras.optimizers.Adam(o['learning_rate'])
        update=update_function(tf,keras,model,optimizer,o,False);sampler=Sampler(train,features,o,spec.get('seed',42));best=None;stale=0;history=[];selected=None
        for epoch in range(1,o['epochs']+1):
            loss=[]
            for _ in range(o['steps_per_epoch']):
                batches=[(x,w,y,np.concatenate([targets[s] for s in ids]) if targets else np.zeros((len(x),3),np.float32)) for x,w,y,ids in sampler.batches()]
                value=update(batches).numpy();assert np.isfinite(value).all();loss.append(value)
            with tf.device('/GPU:0'):windows=predict(model,selection,features)
            thresholds,metric=calibrate(selection,windows,policy,o['pooling_fraction']);key=rank(metric)
            coverage_ready=all(sampler.exposure[r['sample_id']]>=spec.get('minimum_negative_coverage',0) for r in train if r['label']=='other')
            improved=coverage_ready and (best is None or key>best)
            if improved:
                best=key;stale=0;selected=epoch;model.save(str(out/'classifier.keras'));write_json(out/'selection_metrics.json',metric)
                write_csv(out/'selected_checkpoint_exposure.csv',[dict(sample_id=r['sample_id'],label=r['label'],draws=sampler.exposure[r['sample_id']]) for r in train])
                write_json(out/'selection_policy.json',dict(thresholds=thresholds,policy=policy,selection_only=True,not_product_rate_evidence=True))
            elif coverage_ready:stale+=1
            history.append(dict(epoch=epoch,train_loss=float(np.mean(loss,axis=0)[0]),hard_loss=float(np.mean(loss,axis=0)[1]),kd_loss=float(np.mean(loss,axis=0)[2]),learning_rate=float(keras.backend.get_value(optimizer.learning_rate)),saved=improved,negative_coverage_ready=coverage_ready,
                  selection_fire_recall=metric['fire_recall'],selection_doorbell_recall=metric['doorbell_recall'],selection_proxy_macro=metric['target_macro_source_file_recall'],
                  selection_positive_margin=metric['positive_margin_mean'],selection_loss=metric['loss'],cross_or_negative_alerts=metric['total_cross_class_or_negative_alert_files']))
            write_csv(out/'history.csv',history);write_csv(out/'sample_exposure.csv',[dict(sample_id=r['sample_id'],label=r['label'],role=r['role'],draws=sampler.exposure[r['sample_id']]) for r in train])
            status.update(epoch=epoch,selected_epoch=selected,actual_optimizer_steps=int(optimizer.iterations.numpy()),elapsed_seconds=time.monotonic()-started,latest_selection=metric)
            write_json(out/'status.json',status);print('PURSUIT_EPOCH',name,epoch,history[-1],flush=True)
            if stale>=o['patience']:break
        assert selected is not None,'Training budget insufficient for required negative coverage'
        with tf.device('/GPU:0'):restored=keras.models.load_model(str(out/'classifier.keras'),compile=False)
        assert all('GPU:0' in v.device for v in restored.variables)
        _,metric=calibrate(selection,predict(restored,selection,features),policy,o['pooling_fraction']);assert rank(metric)==rank(json.loads((out/'selection_metrics.json').read_text()))
        status.update(complete=True,exit_code=0,selected_epoch=selected,epochs_completed=len(history),model_sha256=sha(out/'classifier.keras'),
                      selected_metrics=metric,complexity=complexity(restored,keras),negative_files=sum(r['label']=='other' for r in train),
                      negative_unique_executed=sum(sampler.exposure[r['sample_id']]>0 for r in train if r['label']=='other'),elapsed_seconds=time.monotonic()-started)
        write_json(out/'status.json',status);print('PURSUIT_TRAIN_COMPLETE',name,selected,flush=True)
    except Exception as e:
        status.update(exit_code=1,first_error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started);write_json(out/'status.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec',type=Path);a=p.parse_args();train(json.loads(a.spec.read_text()))
