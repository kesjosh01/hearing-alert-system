"""Train-only BatchNorm moment refit; convolution/classifier weights preserved."""
import argparse
import json
import shutil
import time
import traceback
import numpy as np
from comparison.dataset import read_csv,write_json,write_csv,sha
from comparison.sampling import Sampler
from dscnn_baseline.common import runtime
from .paths import ROOT,DATA,REPORT
from .stage2_stream import FeatureStore,infer_all
from .operating import measure,key

PLAN=REPORT/'stage2_native40_bn_moment_study_plan.json'
SOURCES=dict(supervised='pursuit_20261114_r06_native40_supervised',distilled='pursuit_20261114_r06_native40_distilled_vs_control')


def register():
    assert not PLAN.exists()
    stop=json.loads((REPORT/'collection_stopped_by_user.json').read_text());assert stop['collection_disabled_by_user']
    write_json(PLAN,dict(created_at_unix=time.time(),dataset='round06_local_regions',
        dataset_freeze_sha256=sha(DATA/'round06_local_regions/freeze.json'),sources=SOURCES,
        hypothesis='Moment estimates from masked, hard-mined microbatches may differ from natural recording windows; re-estimate moments without changing learned kernels, gamma or beta.',
        source_admission='Completed real-data source checkpoint; source model SHA verified before and after transform. Inherit its learned weights; no optimizer updates.',
        batches=100,windows_per_batch=128,momentum=.9,sampler_seed=841,window_seed=842,
        data_rule='Same source-balanced training sampler as source fit; one unaugmented window uniformly from each eligible training file/view. No selection/calibration/test inputs for moment updates.',
        initialization='Reset moving mean to zero and variance to one before fixed-momentum updates; restore original trainability/momentum configuration for inference.',
        comparison_readout=dict(mode='margin',temperature=1.,policy=[[1,1],[1,1]]),
        paired_control_and_kd_use_identical_source_and_window_sampling=True,
        feature_arrays_saved=False,collection_allowed=False,calibration_or_fresh_used=False,
        disk_reserve_bytes=8*2**30,actual_product_targets_verified=False,
        code_identity={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'pursuit/stage2_bn_refit.py',ROOT/'pursuit/stage2_stream.py',ROOT/'comparison/sampling.py']}))
    print('TRAIN_ONLY_BN_MOMENT_STUDY_REGISTERED',flush=True)


def evaluate(kind):
    spec=json.loads(PLAN.read_text());assert all(sha(ROOT/k)==v for k,v in spec['code_identity'].items())
    assert json.loads((REPORT/'collection_stopped_by_user.json').read_text())['collection_disabled_by_user']
    directory=DATA/spec['dataset'];assert sha(directory/'freeze.json')==spec['dataset_freeze_sha256']
    frozen=json.loads((directory/'freeze.json').read_text());assert all(sha(directory/k)==v for k,v in frozen['files'].items())
    source=ROOT/'runs'/spec['sources'][kind];parent=json.loads((source/'status.json').read_text())
    assert parent['complete'] and parent['actual_optimizer_steps']>0 and not parent['calibration_used'] and not parent['fresh_test_used']
    model_path=source/'classifier.keras';assert sha(model_path)==parent['model_sha256']
    run=json.loads((source/'run.json').read_text());assert run['dataset_freeze_sha256']==spec['dataset_freeze_sha256']
    rows=[r for r in read_csv(directory/'development.csv') if r['role'] in ('train','selection')]
    training=[r for r in rows if r['role']=='train'];selection=[r for r in rows if r['role']=='selection']
    out=ROOT/'runs'/(source.name+'_bnrefit');out.mkdir(exist_ok=False);started=time.monotonic()
    status=dict(complete=False,exit_code=None,actual_optimizer_steps=0,parent_model_sha256=parent['model_sha256'],
        inherited_real_trained_checkpoint=True,bn_moment_reestimation=True,fresh_test_used=False,calibration_used=False,
        feature_arrays_saved=False,actual_product_targets_verified=False)
    write_json(out/'status.json',status)
    try:
        tf,keras=runtime('cpu');assert not tf.config.list_physical_devices('GPU');tf.config.experimental.enable_op_determinism();keras.utils.set_random_seed(42)
        with tf.device('/CPU:0'):
            original=keras.models.load_model(str(model_path),compile=False)
            model=keras.models.clone_model(original);model.set_weights(original.get_weights())
        assert tuple(model.input_shape[1:])==(98,40,1)
        assert all(a is not b for a,b in zip(original.weights,model.weights))
        stable=[(v,v.numpy().copy()) for v in model.weights if 'moving_' not in v.name]
        moment=[(v,v.numpy().copy()) for v in model.weights if 'moving_' in v.name]
        flags={l.name:l.trainable for l in model.layers};momenta={}
        for layer in model.layers:
            layer.trainable=isinstance(layer,keras.layers.BatchNormalization)
            if layer.trainable:
                momenta[layer.name]=layer.momentum;layer.momentum=spec['momentum']
                layer.moving_mean.assign(tf.zeros_like(layer.moving_mean));layer.moving_variance.assign(tf.ones_like(layer.moving_variance))
        @tf.function(input_signature=[tf.TensorSpec([None,98,40,1],tf.float32)])
        def adapt(x):
            with tf.device('/CPU:0'):return model(x,training=True)
        store=FeatureStore(rows,tf,kind='portable40',budget_bytes=1536*2**20)
        sampler=Sampler(training,store,run['options'],spec['sampler_seed']);rng=np.random.default_rng(spec['window_seed']);draws=[]
        for batch in range(spec['batches']):
            rr=sampler.draw_ids(spec['windows_per_batch']);images=[]
            for row in rr:
                x,idx=store.get(row);chosen=int(rng.integers(len(x)));images.append(x[chosen])
                draws.append(dict(sample_id=row['sample_id'],source_sha256=row['source_sha256'],global_window_index=int(idx[chosen]),role='train'))
            result=adapt(np.stack(images)).numpy();assert np.isfinite(result).all()
            if (batch+1)%10==0:
                status.update(bn_batches=batch+1,elapsed_seconds=time.monotonic()-started);write_json(out/'status.json',status)
                print('BN_MOMENT_TRAIN_ONLY_BATCH',kind,batch+1,spec['batches'],flush=True)
        assert all(np.array_equal(value,v.numpy()) for v,value in stable),'A learned weight changed during moment-only adaptation'
        changed=sum(not np.array_equal(value,v.numpy()) for v,value in moment);assert changed>0
        assert all(np.isfinite(v.numpy()).all() for v,_ in moment)
        for layer in model.layers:
            layer.trainable=flags[layer.name]
            if layer.name in momenta:layer.momentum=momenta[layer.name]
        assert shutil.disk_usage(ROOT).free>spec['disk_reserve_bytes']+16*2**20
        model.save(str(out/'classifier.keras'))
        net=keras.Model(model.input,model.get_layer('class_logits').output)
        @tf.function(input_signature=[tf.TensorSpec([None,98,40,1],tf.float32)])
        def call(x):
            with tf.device('/CPU:0'):return net(x,training=False)
        logits=infer_all(model,selection,store,keras,tf,call=call);metric=measure(selection,logits,**spec['comparison_readout'])
        np.savez(out/'selection_logits.npz',**logits);write_json(out/'selection_metrics.json',metric)
        write_json(out/'bn_training_window_receipts.json',draws);write_json(out/'feature_stream_provenance.json',list(store.provenance.values()))
        write_csv(out/'used_train_selection.csv',rows)
        write_json(out/'run.json',dict(specification_sha256=sha(PLAN),parent_source=source.name,parent_model_sha256=parent['model_sha256'],
            dataset_freeze_sha256=spec['dataset_freeze_sha256'],options=run['options'],actual_optimizer_steps=0,
            parent_selected_epoch=parent['selected_epoch'],parent_selected_optimizer_steps=parent['selected_epoch']*run['options']['steps_per_epoch'],
            stable_learned_arrays_preserved=len(stable),moving_statistic_arrays_changed=changed,
            source_sampling_receipts_sha256=sha(out/'bn_training_window_receipts.json')))
        assert sha(model_path)==parent['model_sha256']
        status.update(complete=True,exit_code=0,bn_batches=spec['batches'],model_sha256=sha(out/'classifier.keras'),
            selected_metrics=metric,parent_selection=parent['selected_metrics'],selection_better_than_parent=key(metric)>key(parent['selected_metrics']),
            elapsed_seconds=time.monotonic()-started,stable_learned_arrays_preserved=len(stable),moving_statistic_arrays_changed=changed)
        write_json(out/'status.json',status);print('BN_MOMENT_REFIT_COMPLETE',kind,status['selection_better_than_parent'],metric['fire_recall'],metric['doorbell_recall'],flush=True)
    except Exception as error:
        status.update(exit_code=1,first_error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        write_json(out/'status.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','supervised','distilled']);a=p.parse_args()
    register() if a.mode=='register' else evaluate(a.mode)
