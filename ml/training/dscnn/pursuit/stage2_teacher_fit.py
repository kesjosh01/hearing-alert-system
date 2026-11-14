"""Full YAMNet teacher with local-only source/window supervision.

At most two hard-mined windows per file participate in a backward pass.
Mining uses an unaugmented inference pass; masked/dropout training afterwards
is an explicitly approximate, memory-bounded MIL update with frozen BN.
"""
import argparse
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
from comparison.dataset import read_csv, write_csv, write_json, sha
from comparison.protocol import BASE
from comparison.sampling import Sampler
from dscnn_baseline.common import runtime
from dscnn_baseline.weak_labels import bag_arrays
from .paths import ROOT, DATA, REPORT
from .stage2_stream import FeatureStore, encoder_teacher, configure_encoder, logit_predictor, infer_all
from .presence_mil import update_function
from .operating import measure, key


STUDY = REPORT / 'stage2_teacher_study_plan.json'
UNFUSED_STUDY = REPORT / 'stage2_teacher_unfused_study_plan.json'


def register():
    assert not STUDY.exists(), 'Preserve the pre-fit experiment choices'
    directory = DATA / 'round06_local_regions'
    frozen = json.loads((directory / 'freeze.json').read_text())
    assert all(sha(directory/k) == v for k,v in frozen['files'].items())
    gate = REPORT / 'stage2_memory_stream_cpu_gate.json'
    assert json.loads(gate.read_text())['passed']
    options = dict(BASE, kd_weight=0., negative_sampler='coverage_cycle', other_mass=1/3,
        batch_size=36, microbatch_files=6, steps_per_epoch=40, max_frequency_mask=4, max_time_mask=10,
        domestic_weight=4., hard_negative_multiplier=1., epochs=60, patience=12)
    specs = [dict(name='pursuit_20261114_r06_full_yamnet_head',phase='head',learning_rate=.001,epochs=24,patience=8),
             dict(name='pursuit_20261114_r06_full_yamnet_refined',phase='refine',learning_rate=.00003,epochs=60,patience=12)]
    write_json(STUDY,dict(created_at_unix=time.time(),dataset='round06_local_regions',
        dataset_freeze_sha256=sha(directory/'freeze.json'),cpu_gate_sha256=sha(gate),seed=42,options=options,runs=specs,
        selection_readout=dict(mode='margin',temperature=1.,policy=[[1,1],[1,1]],thresholds='Strictly above all nonmatching selection recording scores'),
        source_balanced_recording_metric='Maximize min(fire/.999,doorbell/.95), then macro source-mean recording recall; selection zero errors by fitted thresholds is not a monthly-rate result.',
        positive_published_views='Sample one published-positive window uniformly and the worst absent-class window.',
        weak_or_negative_views='Mine the maximum log-odds window for each alert class, at most two windows per file.',
        mining_approximation='Frozen-BN unaugmented hard mining followed by masked/dropout MIL gradients; not exact full-bag augmented gradients.',
        refinement='Start from the best frozen-encoder head; train convolution/head weights, keep every BN layer frozen.',
        representative_baseline='pursuit_20261114_r04_prefix64_pretrained',
        old_selection_matched_comparison='Use only common parent selection IDs and the same margin policy; no old public-test scoring or tuning.',
        distillation_gate='Only a selection-validated stronger teacher; compare a matched native40 supervised control with distillation before any new fresh-test inference.',
        disk_reserve_bytes=8*2**30,feature_arrays_saved=False,collection_allowed=False,
        fresh_test_used_for_selection=False,calibration_used_for_selection=False,actual_product_targets_verified=False,
        code_identity={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'pursuit/stage2_stream.py',ROOT/'pursuit/presence_mil.py']}))
    print('FULL_YAMNET_TEACHER_STUDY_REGISTERED',flush=True)


def odds(z):
    z=np.asarray(z,dtype=np.float64)
    return np.stack([z[:,k]-np.logaddexp(z[:,(k+1)%3],z[:,(k+2)%3]) for k in (0,1)],axis=1)


def register_unfused():
    assert not UNFUSED_STUDY.exists()
    study=json.loads(STUDY.read_text())
    failed=ROOT/'runs'/study['runs'][1]['name']/'status.json';failure=json.loads(failed.read_text())
    assert failure['exit_code']==1 and failure['actual_optimizer_steps']==0
    assert 'deterministic GPU implementation of fused batch-norm backprop' in failure['first_error']
    gates={device:REPORT/f'stage2_unfused_bn_{device}_gate.json' for device in ['cpu','gpu']}
    for path in gates.values():
        g=json.loads(path.read_text());assert g['passed'] and g['deterministic_ops_enabled']
        assert g['code_sha256']==sha(ROOT/'pursuit/stage2_unfused_bn.py')
    study.update(created_at_unix=time.time(),amends_study_plan_sha256=sha(STUDY),failed_run_status_sha256=sha(failed),
        amendment='Frozen fused BN backward lacks a deterministic GPU implementation. Clone only BN fused=False; preserve all values, frozen statistics and deterministic operations. No optimizer update occurred in the failed attempt.',
        unfused_bn_gate_sha256={device:sha(path) for device,path in gates.items()})
    study['runs'][1]=dict(study['runs'][1],name='pursuit_20261114_r06_full_yamnet_refined_unfused_bn',phase='refine_unfused')
    study['code_identity']={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'pursuit/stage2_stream.py',ROOT/'pursuit/presence_mil.py',ROOT/'pursuit/stage2_unfused_bn.py']}
    write_json(UNFUSED_STUDY,study);print('UNFUSED_BN_CORRECTION_REGISTERED_WITHOUT_ENVIRONMENT_CHANGE',flush=True)


def mined_batches(rows,store,call,rng,microbatch_files):
    for start in range(0,len(rows),microbatch_files):
        batch=rows[start:start+microbatch_files];bags=[]
        for row in batch:
            x,_=store.get(row)
            z=np.concatenate([call(x[i:i+128]).numpy() for i in range(0,len(x),128)])
            values=odds(z);label=int(row['label_id'])
            if row.get('training_region_kind')=='published_target_present':
                selected=[int(rng.integers(len(x))),int(values[:,1-label].argmax())]
            else:selected=[int(values[:,k].argmax()) for k in (0,1)]
            selected=sorted(set(selected));bags.append(x[selected])
        x,w=bag_arrays(bags)
        yield x,w,np.asarray([int(r['label_id']) for r in batch],np.int64),np.zeros((len(x),3),np.float32)


def train(phase,study_path=STUDY):
    study=json.loads(study_path.read_text())
    assert all(sha(ROOT/k)==v for k,v in study['code_identity'].items()),'Preserve registered fitting code'
    assert json.loads((REPORT/'collection_stopped_by_user.json').read_text())['collection_disabled_by_user']
    spec=next(s for s in study['runs'] if s['phase']==phase);o=dict(study['options']);o.update({k:spec[k] for k in ['learning_rate','epochs','patience']})
    directory=DATA/study['dataset'];assert sha(directory/'freeze.json')==study['dataset_freeze_sha256']
    frozen=json.loads((directory/'freeze.json').read_text());assert all(sha(directory/k)==v for k,v in frozen['files'].items())
    gate=REPORT/'stage2_memory_stream_cpu_gate.json';assert sha(gate)==study['cpu_gate_sha256'] and json.loads(gate.read_text())['passed']
    if phase=='refine_unfused':
        assert all(sha(REPORT/f'stage2_unfused_bn_{device}_gate.json')==digest for device,digest in study['unfused_bn_gate_sha256'].items())
    rows=[r for r in read_csv(directory/'development.csv') if r['role'] in ('train','selection')]
    training=[r for r in rows if r['role']=='train'];selection=[r for r in rows if r['role']=='selection']
    assert {r['canonical_pcm_sha256'] for r in training}.isdisjoint(r['canonical_pcm_sha256'] for r in selection)
    out=ROOT/'runs'/spec['name'];out.mkdir(exist_ok=False)
    started=time.monotonic();status=dict(complete=False,exit_code=None,phase=phase,actual_optimizer_steps=0,
        fresh_test_used=False,calibration_used=False,actual_product_targets_verified=False,features_written_to_disk=False)
    write_json(out/'status.json',status)
    try:
        tf,keras=runtime('gpu');tf.config.experimental.enable_op_determinism();keras.utils.set_random_seed(study['seed'])
        with tf.device('/GPU:0'):
            if phase=='head':model=encoder_teacher(keras,study['seed'])
            else:
                previous=ROOT/'runs'/study['runs'][0]['name'];ps=json.loads((previous/'status.json').read_text())
                assert ps['complete'] and sha(previous/'classifier.keras')==ps['model_sha256']
                model=keras.models.load_model(str(previous/'classifier.keras'),compile=False)
                if phase=='refine_unfused':
                    from .stage2_unfused_bn import clone
                    model=clone(model,keras)
                configure_encoder(model,keras,True)
                write_json(out/'initial_checkpoint.json',dict(path=str(previous/'classifier.keras'),sha256=ps['model_sha256']))
            optimizer=keras.optimizers.Adam(o['learning_rate'])
        assert all('GPU:0' in v.device for v in model.variables)
        store=FeatureStore(rows,tf,budget_bytes=1536*2**20)
        call=logit_predictor(model,keras,tf);update=update_function(tf,keras,model,optimizer,o,False)
        sampler=Sampler(training,store,o,study['seed']);rng=np.random.default_rng(study['seed']+619)
        write_csv(out/'used_train_selection.csv',rows)
        write_json(out/'run.json',dict(spec=spec,options=o,study_plan_sha256=sha(study_path),
            dataset_freeze_sha256=study['dataset_freeze_sha256'],teacher=True,distillation=False,
            source_weights_modified=False,parameters=int(model.count_params()),
            trainable_parameters=int(sum(np.prod(v.shape) for v in model.trainable_variables)),
            all_batch_normalization_frozen=True,features_on_disk=False,source_code_identity=study['code_identity']))
        shutil.copytree(ROOT/'pursuit',out/'source_snapshot/pursuit',ignore=shutil.ignore_patterns('__pycache__'))
        history=[];best=None;stale=0;selected=None
        for epoch in range(1,o['epochs']+1):
            losses=[]
            for step in range(o['steps_per_epoch']):
                sampled=sampler.draw_ids(o['batch_size'])
                loss=update(mined_batches(sampled,store,call,rng,o['microbatch_files'])).numpy()
                assert np.isfinite(loss).all();losses.append(float(loss[0]))
                if (step+1)%10==0:
                    status.update(epoch=epoch,step=step+1,actual_optimizer_steps=int(optimizer.iterations.numpy()),elapsed_seconds=time.monotonic()-started)
                    write_json(out/'status.json',status)
                    print('FULL_TEACHER_STEP',phase,epoch,step+1,float(np.mean(losses)),flush=True)
            logits=infer_all(model,selection,store,keras,tf,call=call)
            metric=measure(selection,logits,**{k:v for k,v in study['selection_readout'].items() if k!='thresholds'})
            value=key(metric);improved=best is None or value>best
            if improved:
                assert shutil.disk_usage(ROOT).free>study['disk_reserve_bytes']+64*2**20,'Preserve the user storage reserve'
                best=value;stale=0;selected=epoch
                temp=out/'classifier_temporary.keras';model.save(str(temp));os.replace(temp,out/'classifier.keras')
                write_json(out/'selection_metrics.json',metric)
                np.savez(out/'selection_logits.npz',**logits)
            else:stale+=1
            record=dict(epoch=epoch,train_loss=float(np.mean(losses)),saved=improved,learning_rate=float(keras.backend.get_value(optimizer.learning_rate)),
                selection_fire_recall=metric['fire_recall'],selection_doorbell_recall=metric['doorbell_recall'],
                selection_macro_recall=metric['macro_recall'],selection_other_false_alert_files=metric['other_false_alert_files'],
                smaller_target_ratio=float(value[0]))
            history.append(record);write_csv(out/'history.csv',history)
            write_csv(out/'sample_exposure.csv',[dict(sample_id=r['sample_id'],label=r['label'],draws=sampler.exposure[r['sample_id']]) for r in training])
            status.update(epoch=epoch,step=o['steps_per_epoch'],selected_epoch=selected,actual_optimizer_steps=int(optimizer.iterations.numpy()),
                elapsed_seconds=time.monotonic()-started,latest_selection=metric)
            write_json(out/'status.json',status);print('FULL_TEACHER_EPOCH',phase,record,flush=True)
            if stale>=o['patience']:break
        with tf.device('/GPU:0'):restored=keras.models.load_model(str(out/'classifier.keras'),compile=False)
        final_logits=infer_all(restored,selection,store,keras,tf)
        final_metric=measure(selection,final_logits,**{k:v for k,v in study['selection_readout'].items() if k!='thresholds'})
        saved=json.loads((out/'selection_metrics.json').read_text())
        assert key(final_metric)==key(saved)
        write_json(out/'feature_stream_provenance.json',list(store.provenance.values()))
        status.update(complete=True,exit_code=0,epochs_completed=len(history),selected_epoch=selected,
            actual_optimizer_steps=int(optimizer.iterations.numpy()),model_sha256=sha(out/'classifier.keras'),
            selected_metrics=final_metric,elapsed_seconds=time.monotonic()-started,
            negative_training_views=sum(r['label']=='other' for r in training),
            negative_unique_views_executed=sum(sampler.exposure[r['sample_id']]>0 for r in training if r['label']=='other'),
            negative_region_views_are_not_independent_recordings=True)
        write_json(out/'status.json',status);print('FULL_YAMNET_TEACHER_FIT_COMPLETE',phase,selected,flush=True)
    except Exception as e:
        status.update(exit_code=1,first_error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        write_json(out/'status.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','head','refine','register_unfused','refine_unfused']);a=p.parse_args()
    if a.mode=='register':register()
    elif a.mode=='register_unfused':register_unfused()
    else:train(a.mode,UNFUSED_STUDY if a.mode=='refine_unfused' else STUDY)
