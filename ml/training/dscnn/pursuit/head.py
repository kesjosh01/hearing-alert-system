"""Full frozen-YAMNet reference head with sparse weak-label pooling.

This reference is not a microcontroller deployment claim. Its purpose is to
test whether pretrained semantic features resolve the compact CNN's errors.
"""
import argparse
import json
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
from dscnn_baseline.common import runtime
from comparison.dataset import read_csv,write_csv,write_json,sha
from comparison.sampling import Sampler
from comparison.protocol import BASE
from .paths import ROOT,DATA,REPORT
from .embeddings import load
from .train import predict
from .metrics import calibrate,rank


def make_update(tf,keras,model,optimizer,fraction):
    @tf.function(input_signature=[tf.TensorSpec([None,1024],tf.float32),tf.TensorSpec([None,None],tf.float32),tf.TensorSpec([None],tf.int64)])
    def differentiate(x,w,y):
        with tf.device('/GPU:0'),tf.GradientTape() as tape:
            p=model(x,training=True)
            def one(weight):
                n=tf.reduce_sum(tf.cast(weight>0,tf.int32));k=tf.maximum(1,tf.cast(tf.math.ceil(tf.cast(n,tf.float32)*fraction),tf.int32))
                score=tf.where(weight>0,1-p[:,2],tf.constant(-1.,tf.float32));i=tf.math.top_k(score,k=k).indices
                return tf.scatter_nd(i[:,None],tf.ones([k],tf.float32)/tf.cast(k,tf.float32),tf.shape(weight))
            selected=w if fraction==1 else tf.map_fn(one,w,fn_output_signature=tf.TensorSpec([None],tf.float32))
            loss=tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y,tf.matmul(selected,p)))
        return loss,tape.gradient(loss,model.trainable_variables)
    @tf.function
    def apply(g):optimizer.apply_gradients(zip(g,model.trainable_variables))
    def update(batches):
        batches=list(batches);count=sum(len(b[2]) for b in batches);g=[tf.zeros_like(v) for v in model.trainable_variables];total=tf.constant(0.)
        for x,w,y,ids in batches:
            loss,gg=differentiate(x,w,y);weight=len(y)/count;g=[a+b*weight for a,b in zip(g,gg)];total+=loss*weight
        apply(g);return float(total.numpy())
    return update


def train(name,dataset,fraction):
    out=ROOT/'runs'/name;out.mkdir(exist_ok=False);status=dict(complete=False,exit_code=None,optimizer_steps=0,model_type='frozen_yamnet_mlp128',product_targets_verified=False)
    write_json(out/'status.json',status);started=time.monotonic()
    try:
        d=DATA/dataset;freeze=json.loads((d/'freeze.json').read_text());assert all(sha(d/k)==v for k,v in freeze['files'].items())
        rows=[r for r in read_csv(d/'development.csv') if r['role'] in ('train','selection')];training=[r for r in rows if r['role']=='train'];selection=[r for r in rows if r['role']=='selection']
        features={};provenance=[]
        for r in rows:
            a,m=load(r);features[r['sample_id']]=a;provenance.append(dict(sample_id=r['sample_id'],role=r['role'],**m))
        assert all(r['role']!='fresh_test' for r in rows)
        write_csv(out/'used_train_selection.csv',rows);write_json(out/'feature_provenance.json',provenance)
        o=dict(BASE,negative_sampler='coverage_cycle',kd_weight=0.,pooling_fraction=fraction,epochs=60,patience=12)
        write_json(out/'run.json',dict(name=name,dataset=dataset,options=o,model_type='Frozen official YAMNet + Dense128/Dropout0.2/3class',
          dataset_freeze_sha256=sha(d/'freeze.json'),fresh_test_used=False,backbone_fine_tuned=False,not_board_deployable_as_a_full_backbone=True,
          code_identity={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'pursuit').glob('*.py')}))
        shutil.copytree(ROOT/'pursuit',out/'source_snapshot/pursuit',ignore=shutil.ignore_patterns('__pycache__'))
        tf,keras=runtime('gpu');tf.config.experimental.enable_op_determinism();keras.utils.set_random_seed(42)
        with tf.device('/GPU:0'):
            inputs=keras.Input((1024,));x=keras.layers.Dense(128,activation='relu')(inputs);x=keras.layers.Dropout(.2)(x)
            logits=keras.layers.Dense(3,name='class_logits')(x);model=keras.Model(inputs,keras.layers.Softmax()(logits));optimizer=keras.optimizers.Adam(.001)
        update=make_update(tf,keras,model,optimizer,fraction);sampler=Sampler(training,features,o,42);best=None;stale=0;history=[];selected=None
        for epoch in range(1,o['epochs']+1):
            losses=[update(sampler.batches()) for _ in range(o['steps_per_epoch'])];assert np.isfinite(losses).all()
            with tf.device('/GPU:0'):windows=predict(model,selection,features)
            thresholds,metric=calibrate(selection,windows,fraction=fraction);key=rank(metric);improved=best is None or key>best
            if improved:
                best=key;stale=0;selected=epoch;model.save(str(out/'classifier.keras'));write_json(out/'selection_metrics.json',metric)
                write_json(out/'selection_policy.json',dict(thresholds=thresholds,policy=[[1,1],[1,1]],selection_only=True))
            else:stale+=1
            history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),fire_recall=metric['fire_recall'],doorbell_recall=metric['doorbell_recall'],
               positive_margin=metric['positive_margin_mean'],cross_or_negative_alerts=metric['total_cross_class_or_negative_alert_files'],saved=improved))
            write_csv(out/'history.csv',history);write_csv(out/'sample_exposure.csv',[dict(sample_id=r['sample_id'],label=r['label'],draws=sampler.exposure[r['sample_id']]) for r in training])
            status.update(epoch=epoch,selected_epoch=selected,optimizer_steps=int(optimizer.iterations.numpy()),latest_selection=metric,elapsed_seconds=time.monotonic()-started)
            write_json(out/'status.json',status);print('PURSUIT_HEAD_EPOCH',name,epoch,history[-1],flush=True)
            if stale>=o['patience']:break
        with tf.device('/GPU:0'):restored=keras.models.load_model(str(out/'classifier.keras'),compile=False)
        _,metric=calibrate(selection,predict(restored,selection,features),fraction=fraction);assert rank(metric)==rank(json.loads((out/'selection_metrics.json').read_text()))
        status.update(complete=True,exit_code=0,selected_epoch=selected,epochs_completed=len(history),model_sha256=sha(out/'classifier.keras'),selected_metrics=metric,
                      parameters=int(restored.count_params()),elapsed_seconds=time.monotonic()-started,backbone_weights_changed=False)
        write_json(out/'status.json',status);print('PURSUIT_HEAD_TRAIN_COMPLETE',name,selected,flush=True)
    except Exception as e:
        status.update(exit_code=1,first_error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());write_json(out/'status.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--dataset',default='round01');p.add_argument('--fraction',type=float,default=.01);a=p.parse_args()
    train(a.name,a.dataset,a.fraction)
