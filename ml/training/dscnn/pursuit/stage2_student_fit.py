"""Matched native40 supervised/KD students; existing local data only."""
import argparse
import json
import os
import shutil
import time
import traceback

import numpy as np
from comparison.dataset import read_csv,write_csv,write_json,sha
from comparison.protocol import BASE
from comparison.sampling import Sampler
from dscnn_baseline.common import runtime
from dscnn_baseline.model import complexity
from dscnn_baseline.weak_labels import bag_arrays
from .paths import ROOT,DATA,REPORT
from .stage2_stream import FeatureStore,logit_predictor,infer_all
from .stage2_student_model import build
from .stage2_student_loss import make_update
from .stage2_teacher_fit import odds
from .operating import measure,key,softmax

PLAN=REPORT/'stage2_native40_student_study_plan.json'


def register(plan_path=PLAN,comparison=REPORT/'stage2_teacher_matched_comparison.json',distilled_name='pursuit_20261114_r06_native40_distilled'):
    plan_path=plan_path.resolve();comparison=comparison.resolve()
    assert plan_path.resolve().parent==REPORT.resolve() and comparison.resolve().parent==REPORT.resolve()
    assert not plan_path.exists(),'Preserve registered student settings'
    assert ROOT.joinpath('runs',distilled_name).parent==ROOT/'runs'
    teacher=json.loads(comparison.read_text())
    assert teacher['complete'] and not teacher['calibration_or_fresh_used']
    gate=REPORT/'stage2_student_mil_kd_cpu_gate_v2.json';assert json.loads(gate.read_text())['passed']
    gpu_gate=REPORT/'stage2_student_mil_kd_gpu_gate_v2.json';gpu_check=json.loads(gpu_gate.read_text());assert gpu_check['passed']
    assert all(sha(ROOT/k)==v for k,v in gpu_check['code_identity'].items())
    directory=DATA/'round06_local_regions';frozen=json.loads((directory/'freeze.json').read_text())
    assert all(sha(directory/k)==v for k,v in frozen['files'].items())
    options=dict(BASE,negative_sampler='coverage_cycle',other_mass=1/3,domestic_weight=4.,hard_negative_multiplier=1.,
        batch_size=36,microbatch_files=6,steps_per_epoch=40,learning_rate=.0007,epochs=100,patience=20,
        max_frequency_mask=4,max_time_mask=10,temperature=2.,kd_weight=0.)
    specs=[dict(name='pursuit_20261114_r06_native40_supervised',kd_weight=0.)]
    if teacher['distillation_gate_passed']:specs.append(dict(name=distilled_name,kd_weight=.25))
    write_json(plan_path,dict(created_at_unix=time.time(),dataset='round06_local_regions',dataset_freeze_sha256=sha(directory/'freeze.json'),
        teacher_comparison_path=str(comparison.relative_to(ROOT)),
        teacher_comparison_sha256=sha(comparison),teacher_name=teacher['chosen_teacher'],teacher_model_sha256=teacher['chosen_model_sha256'],
        stronger_teacher_gate_passed=teacher['distillation_gate_passed'],cpu_gate_sha256=sha(gate),gpu_gate_sha256=sha(gpu_gate),options=options,runs=specs,seed=42,
        architecture='Native 98x40x1 six-block DS-CNN, time receptive field 117 frames, frequency-preserving final time average; time-only downsampling uses average pooling',
        source_sampler_and_initialization_seeds_matched=True,negative_training_views_must_all_be_seen_before_checkpoint_eligibility=True,
        weak_bag_mining='Native student maximum log-odds per alert class; published positives also include one uniform published-positive window.',
        kd='Add alpha * T^2 KL to the unscaled hard presence loss. Teacher sees original aligned official64 windows; student sees masked native40 windows.',
        kd_mask='Teacher argmax matches the file/view label and untempered probability >= 0.7. Masked-out targets contribute no KL and do not reduce hard-loss strength.',
        student_mining_may_diverge_after_distillation=True,
        selection_readout=dict(mode='margin',temperature=1.,policy=[[1,1],[1,1]]),
        minimum_disk_reserve_bytes=8*2**30,no_audio_or_metadata_collection=True,no_feature_arrays_saved=True,
        calibration_or_new_fresh_used_for_training=False,actual_product_targets_verified=False,
        code_identity={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'pursuit/stage2_student_fit.py',ROOT/'pursuit/stage2_student_model.py',ROOT/'pursuit/stage2_student_loss.py',ROOT/'pursuit/stage2_stream.py',ROOT/'pursuit/stage2_teacher_fit.py']}))
    print('NATIVE40_MATCHED_STUDY_REGISTERED','KD_ALLOWED',teacher['distillation_gate_passed'],flush=True)


def predictor(model,keras,tf):
    net=keras.Model(model.input,model.get_layer('class_logits').output)
    @tf.function(input_signature=[tf.TensorSpec([None,98,40,1],tf.float32)])
    def call(x):
        with tf.device('/GPU:0'):return net(x,training=False)
    return call


def batches(rows,store,call,rng,microbatch,teacher_store=None,teacher_call=None,teacher_cache=None):
    for start in range(0,len(rows),microbatch):
        rr=rows[start:start+microbatch];bags=[];specs=[];pending={}
        for row in rr:
            x,indices=store.get(row)
            z=np.concatenate([call(x[i:i+128]).numpy() for i in range(0,len(x),128)])
            value=odds(z);label=int(row['label_id'])
            chosen=[int(rng.integers(len(x))),int(value[:,1-label].argmax())] if row.get('training_region_kind')=='published_target_present' else [int(value[:,k].argmax()) for k in (0,1)]
            chosen=sorted(set(chosen));bags.append(x[chosen]);keys=[]
            if teacher_store is not None:
                teacher_x=teacher_store.full(row)
                assert store.provenance[row['source_sha256']]['starts']==teacher_store.provenance[row['source_sha256']]['starts']
                for i in chosen:
                    index=int(indices[i]);cache_key=(row['source_sha256'],index);keys.append(cache_key)
                    if cache_key not in teacher_cache:pending[cache_key]=teacher_x[index]
            specs.append((keys,label))
        if pending:
            keys=list(pending);values=teacher_call(np.stack([pending[k] for k in keys])).numpy()
            assert values.shape==(len(keys),3) and np.isfinite(values).all()
            teacher_cache.update(zip(keys,values))
        target=[]
        for bag,(keys,label) in zip(bags,specs):
            if teacher_store is None:target.append(np.zeros((len(bag),4),np.float32));continue
            logits=np.stack([teacher_cache[k] for k in keys]);probability=softmax(logits)
            mask=((logits.argmax(1)==label)&(probability.max(1)>=.7)).astype(np.float32)
            target.append(np.concatenate([logits,mask[:,None]],axis=1).astype(np.float32))
        x,w=bag_arrays(bags)
        yield x,w,np.asarray([int(r['label_id']) for r in rr],np.int64),np.concatenate(target)


def train(kind,plan_path=PLAN):
    plan_path=plan_path.resolve()
    assert plan_path.resolve().parent==REPORT.resolve()
    study=json.loads(plan_path.read_text());assert all(sha(ROOT/k)==v for k,v in study['code_identity'].items())
    assert json.loads((REPORT/'collection_stopped_by_user.json').read_text())['collection_disabled_by_user']
    if kind=='distilled':assert study['stronger_teacher_gate_passed']
    spec=next(s for s in study['runs'] if ('distilled' in s['name'])==(kind=='distilled'))
    options=dict(study['options'],kd_weight=spec['kd_weight'])
    directory=DATA/study['dataset'];assert sha(directory/'freeze.json')==study['dataset_freeze_sha256']
    frozen=json.loads((directory/'freeze.json').read_text());assert all(sha(directory/k)==v for k,v in frozen['files'].items())
    assert sha(ROOT/study['teacher_comparison_path'])==study['teacher_comparison_sha256']
    assert sha(REPORT/'stage2_student_mil_kd_cpu_gate_v2.json')==study['cpu_gate_sha256']
    assert sha(REPORT/'stage2_student_mil_kd_gpu_gate_v2.json')==study['gpu_gate_sha256']
    rows=[r for r in read_csv(directory/'development.csv') if r['role'] in ('train','selection')]
    training=[r for r in rows if r['role']=='train'];selection=[r for r in rows if r['role']=='selection']
    assert {r['canonical_pcm_sha256'] for r in training}.isdisjoint(r['canonical_pcm_sha256'] for r in selection)
    out=ROOT/'runs'/spec['name'];out.mkdir(exist_ok=False);started=time.monotonic()
    status=dict(complete=False,exit_code=None,actual_optimizer_steps=0,fresh_test_used=False,calibration_used=False,
        distillation=kind=='distilled',features_written_to_disk=False,actual_product_targets_verified=False)
    write_json(out/'status.json',status)
    try:
        tf,keras=runtime('gpu');tf.config.experimental.enable_op_determinism();keras.utils.set_random_seed(study['seed'])
        teacher_store=None;teacher_call=None;teacher_cache={}
        with tf.device('/GPU:0'):
            model=build(keras,study['seed']);optimizer=keras.optimizers.Adam(options['learning_rate'])
            if kind=='distilled':
                teacher_path=ROOT/'runs'/study['teacher_name']/'classifier.keras';assert sha(teacher_path)==study['teacher_model_sha256']
                teacher=keras.models.load_model(str(teacher_path),compile=False);teacher.trainable=False
                teacher_call=logit_predictor(teacher,keras,tf);teacher_store=FeatureStore(rows,tf,budget_bytes=1024*2**20)
        assert all('GPU:0' in v.device for v in model.variables)
        store=FeatureStore(rows,tf,kind='portable40',budget_bytes=1536*2**20);call=predictor(model,keras,tf)
        update=make_update(tf,keras,model,optimizer,options);sampler=Sampler(training,store,options,study['seed'])
        rng=np.random.default_rng(study['seed']+619);negative=[r for r in training if r['label']=='other']
        write_csv(out/'used_train_selection.csv',rows)
        write_json(out/'run.json',dict(spec=spec,options=options,study_plan_sha256=sha(plan_path),dataset_freeze_sha256=study['dataset_freeze_sha256'],
            teacher_name=study['teacher_name'] if kind=='distilled' else None,teacher_model_sha256=study['teacher_model_sha256'] if kind=='distilled' else None,
            architecture='native40_long_receptive_dscnn',complexity=complexity(model,keras),feature_arrays_saved=False))
        shutil.copytree(ROOT/'pursuit',out/'source_snapshot/pursuit',ignore=shutil.ignore_patterns('__pycache__'))
        history=[];best=None;selected=None;stale=0
        for epoch in range(1,options['epochs']+1):
            losses=[]
            for step in range(options['steps_per_epoch']):
                rr=sampler.draw_ids(options['batch_size'])
                loss=update(batches(rr,store,call,rng,options['microbatch_files'],teacher_store,teacher_call,teacher_cache)).numpy()
                assert np.isfinite(loss).all();losses.append(loss)
                if (step+1)%10==0:
                    status.update(epoch=epoch,step=step+1,actual_optimizer_steps=int(optimizer.iterations.numpy()),elapsed_seconds=time.monotonic()-started)
                    write_json(out/'status.json',status);print('NATIVE40_STUDENT_STEP',kind,epoch,step+1,np.mean(losses,axis=0).tolist(),flush=True)
            logits=infer_all(model,selection,store,keras,tf,call=call);metric=measure(selection,logits,**study['selection_readout'])
            coverage=all(sampler.exposure[r['sample_id']]>0 for r in negative)
            improved=coverage and (best is None or key(metric)>best)
            if improved:
                assert shutil.disk_usage(ROOT).free>study['minimum_disk_reserve_bytes']+32*2**20
                best=key(metric);selected=epoch;stale=0;temp=out/'classifier_temporary.keras'
                model.save(str(temp));os.replace(temp,out/'classifier.keras');write_json(out/'selection_metrics.json',metric);np.savez(out/'selection_logits.npz',**logits)
            elif coverage:stale+=1
            loss=np.mean(losses,axis=0)
            history.append(dict(epoch=epoch,train_loss=float(loss[0]),hard_loss=float(loss[1]),kd_loss=float(loss[2]),saved=improved,
                full_negative_coverage=coverage,selection_fire_recall=metric['fire_recall'],selection_doorbell_recall=metric['doorbell_recall'],selection_macro_recall=metric['macro_recall']))
            write_csv(out/'history.csv',history);write_csv(out/'sample_exposure.csv',[dict(sample_id=r['sample_id'],label=r['label'],draws=sampler.exposure[r['sample_id']]) for r in training])
            status.update(epoch=epoch,step=options['steps_per_epoch'],selected_epoch=selected,actual_optimizer_steps=int(optimizer.iterations.numpy()),
                latest_selection=metric,full_negative_training_view_coverage=coverage,elapsed_seconds=time.monotonic()-started)
            write_json(out/'status.json',status);print('NATIVE40_STUDENT_EPOCH',kind,history[-1],flush=True)
            if coverage and stale>=options['patience']:break
        assert selected is not None,'No checkpoint after complete negative-view coverage'
        with tf.device('/GPU:0'):restored=keras.models.load_model(str(out/'classifier.keras'),compile=False)
        final_logits=infer_all(restored,selection,store,keras,tf,call=predictor(restored,keras,tf))
        final_metric=measure(selection,final_logits,**study['selection_readout']);assert key(final_metric)==key(json.loads((out/'selection_metrics.json').read_text()))
        if kind=='distilled':assert sha(teacher_path)==study['teacher_model_sha256']
        write_json(out/'feature_stream_provenance.json',dict(native40=list(store.provenance.values()),official64=list(teacher_store.provenance.values()) if teacher_store else [],feature_arrays_saved=False))
        status.update(complete=True,exit_code=0,model_sha256=sha(out/'classifier.keras'),selected_metrics=final_metric,epochs_completed=len(history),
            actual_optimizer_steps=int(optimizer.iterations.numpy()),elapsed_seconds=time.monotonic()-started,
            negative_training_views=len(negative),negative_unique_views_executed=sum(sampler.exposure[r['sample_id']]>0 for r in negative))
        write_json(out/'status.json',status);print('NATIVE40_STUDENT_FIT_COMPLETE',kind,selected,flush=True)
    except Exception as error:
        status.update(exit_code=1,first_error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        write_json(out/'status.json',status);raise


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','supervised','distilled'])
    p.add_argument('--plan',type=Path,default=PLAN);p.add_argument('--comparison',type=Path,default=REPORT/'stage2_teacher_matched_comparison.json')
    p.add_argument('--distilled-name',default='pursuit_20261114_r06_native40_distilled');a=p.parse_args()
    register(a.plan,a.comparison,a.distilled_name) if a.mode=='register' else train(a.mode,a.plan)
