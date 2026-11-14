"""Choose actual int8 native40 outputs on selection, then raise cuts on calibration."""
import argparse
import hashlib
import itertools
import json
import shutil
import time
import traceback
import numpy as np
from comparison.dataset import read_csv,write_json,sha
from dscnn_baseline.common import runtime,config
from .paths import ROOT,DATA,REPORT
from .stage2_stream import FeatureStore
from .native import DIRECTORY
from .cpu_integer import validated_interpreter
from . import delivered_integer as di

PLAN=REPORT/'stage2_native40_export_plan.json'
OUT=REPORT/'stage2_native40_delivered_v1'
NAMES=['pursuit_20261114_r06_native40_supervised','pursuit_20261114_r06_native40_distilled_vs_control',
       'pursuit_20261114_r06_native40_supervised_bnrefit','pursuit_20261114_r06_native40_distilled_vs_control_bnrefit']


def register():
    assert not PLAN.exists()
    write_json(PLAN,dict(created_at_unix=time.time(),dataset='round06_local_regions',
        dataset_freeze_sha256=sha(DATA/'round06_local_regions/freeze.json'),source_names=NAMES,
        source_admission='Completed real trained student, or its documented moment-only clone; source checkpoint SHA checked before and after export.',
        variants=[['probability',1.],['probability',4.],['probability',8.],['logits',1.]],
        policies=list(itertools.product([(1,1),(2,3)],repeat=2)),
        representative='All train-positive file/views and 512 seeded train-negative views; eligible midpoint and maximum mean-log-Mel windows. Native features retained in memory only.',
        representative_negative_seed=843,selection='Maximize min(source_mean_fire/.999,source_mean_doorbell/.95), then source-mean macro recall in actual int8 output units.',
        calibration='Only raise the chosen cuts using held-out calibration source groups; never re-rank candidates using calibration.',
        native_binary_sha256=sha(DIRECTORY/'libpursuit_frontend.so'),frontend=config()[0]['frontend'],
        disk_reserve_bytes=8*2**30,feature_arrays_saved=False,collection_allowed=False,
        new_fresh_test_used=False,actual_product_targets_verified=False,
        code_identity={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'pursuit/stage2_native_export.py',ROOT/'pursuit/stage2_stream.py',ROOT/'pursuit/delivered_integer.py',ROOT/'pursuit/cpu_integer.py']}))
    print('ACTUAL_INT8_NATIVE40_EXPORT_RULES_REGISTERED',flush=True)


def selection_key(metric):
    fire=metric['fire_recall'] or 0.;bell=metric['doorbell_recall'] or 0.
    return min(fire/.999,bell/.95),(fire+bell)/2


def predict(interpreter,rows,store):
    values={}
    for i,row in enumerate(rows,1):
        assert row['role'] in ('selection','calibration') and not row.get('training_region_kind')
        x,_=store.get(row);values[row['sample_id']]=di.predict_q(interpreter,x)
        if i%200==0:print('NATIVE40_INT8_RECORDINGS',row['role'],i,len(rows),flush=True)
    return values


def evaluate():
    spec=json.loads(PLAN.read_text());assert all(sha(ROOT/k)==v for k,v in spec['code_identity'].items())
    assert json.loads((REPORT/'collection_stopped_by_user.json').read_text())['collection_disabled_by_user']
    assert sha(DIRECTORY/'libpursuit_frontend.so')==spec['native_binary_sha256']
    directory=DATA/spec['dataset'];assert sha(directory/'freeze.json')==spec['dataset_freeze_sha256']
    frozen=json.loads((directory/'freeze.json').read_text());assert all(sha(directory/k)==v for k,v in frozen['files'].items())
    sources={}
    for name in spec['source_names']:
        run=ROOT/'runs'/name;status=json.loads((run/'status.json').read_text());meta=json.loads((run/'run.json').read_text())
        assert status['complete'] and not status['fresh_test_used'] and not status['calibration_used']
        assert sha(run/'classifier.keras')==status['model_sha256'] and meta['dataset_freeze_sha256']==spec['dataset_freeze_sha256']
        if status['actual_optimizer_steps']==0:
            assert status['inherited_real_trained_checkpoint'] and status['bn_moment_reestimation']
            assert sha(ROOT/'runs'/meta['parent_source']/'classifier.keras')==status['parent_model_sha256']
        sources[name]=dict(status=status,meta=meta,model_path=run/'classifier.keras')
    rows=read_csv(directory/'development.csv');training=[r for r in rows if r['role']=='train']
    selection=[r for r in rows if r['role']=='selection'];calibration=[r for r in rows if r['role']=='calibration']
    assert all(r['role'] in ('train','selection','calibration') for r in rows)
    OUT.mkdir(exist_ok=False);started=time.monotonic();state=dict(complete=False,exit_code=None,fresh_test_used=False,actual_product_targets_verified=False)
    write_json(OUT/'status.json',state)
    try:
        tf,keras=runtime('cpu');assert not tf.config.list_physical_devices('GPU')
        store=FeatureStore(rows,tf,kind='portable40',budget_bytes=1536*2**20)
        negatives=[r for r in training if r['label']=='other'];rng=np.random.default_rng(spec['representative_negative_seed'])
        chosen_negative=[negatives[int(i)] for i in rng.choice(len(negatives),512,replace=False)]
        representatives=[];provenance=[]
        for row in [r for r in training if r['label']!='other']+chosen_negative:
            x,idx=store.get(row)
            for i in np.unique([len(x)//2,int(np.argmax(np.mean(x,axis=(1,2,3))))]):
                representatives.append(x[i]);provenance.append(dict(split='train',sample_id=row['sample_id'],
                    source_sha256=row['source_sha256'],canonical_pcm_sha256=row['canonical_pcm_sha256'],window_index=int(idx[i])))
        representative=np.stack(representatives)
        write_json(OUT/'representative_provenance.json',provenance)
        write_json(OUT/'representative_memory_receipt.json',dict(shape=list(representative.shape),dtype=str(representative.dtype),
            values_sha256=hashlib.sha256(representative.tobytes()).hexdigest(),feature_array_saved=False,role='train'))
        variants=[]
        for name,source in sources.items():
            assert shutil.disk_usage(ROOT).free>spec['disk_reserve_bytes']+64*2**20
            model=keras.models.load_model(str(source['model_path']),compile=False);assert tuple(model.input_shape[1:])==(98,40,1)
            source_directory=OUT/name;source_directory.mkdir(exist_ok=False)
            for kind,temperature in spec['variants']:
                target=source_directory/(kind+'_t'+str(int(temperature)))
                _,meta=di.export(tf,keras,model,representative,provenance,target,spec['frontend'],kind,temperature)
                interpreter=validated_interpreter(tf,target/'model_int8.tflite',representative,target/'cpu_kernel_gate.json',64)
                values=predict(interpreter,selection,store);np.savez(target/'selection_outputs_q.npz',**values);analyses=[]
                for policy in spec['policies']:
                    cuts=di.fit_cuts(selection,values,kind,meta['output']['zero_point'],policy)
                    metric=di.measure(selection,values,kind,meta['output']['zero_point'],policy,cuts)
                    analyses.append(dict(policy=policy,cuts=cuts.tolist(),metric=metric))
                    variants.append(dict(name=name,directory=target,kind=kind,temperature=temperature,meta=meta,
                        interpreter=interpreter,values=values,policy=policy,cuts=cuts,metric=metric))
                write_json(target/'selection_analysis.json',dict(results=analyses,selection_zero_errors_by_construction=True,
                    calibration_or_fresh_used=False,actual_product_targets_verified=False))
                state.update(source=name,output_kind=kind,temperature=temperature,variants_completed=len(variants)//len(spec['policies']),elapsed_seconds=time.monotonic()-started)
                write_json(OUT/'status.json',state);print('NATIVE40_ACTUAL_INT8_VARIANT_COMPLETE',name,kind,temperature,flush=True)
        selected=max(variants,key=lambda v:selection_key(v['metric']))
        choice=dict(source_model=selected['name'],source_model_sha256=sources[selected['name']]['status']['model_sha256'],
            selected_tflite_sha256=selected['meta']['model_sha256'],directory=str(selected['directory']),kind=selected['kind'],
            temperature=selected['temperature'],policy=selected['policy'],cuts_integer_units=selected['cuts'].tolist(),selection=selected['metric'])
        write_json(OUT/'chosen_before_calibration.json',dict(**choice,calibration_or_fresh_used_to_rank=False,
            comparison=[dict(name=v['name'],kind=v['kind'],temperature=v['temperature'],policy=v['policy'],metric=v['metric']) for v in variants]))
        cal_values=predict(selected['interpreter'],calibration,store)
        cuts=di.fit_cuts(calibration,cal_values,selected['kind'],selected['meta']['output']['zero_point'],selected['policy'],selected['cuts'])
        selection_after=di.measure(selection,selected['values'],selected['kind'],selected['meta']['output']['zero_point'],selected['policy'],cuts)
        cal_metric=di.measure(calibration,cal_values,selected['kind'],selected['meta']['output']['zero_point'],selected['policy'],cuts)
        np.savez(OUT/'chosen_selection_and_calibration_q.npz',**dict(selected['values'],**cal_values))
        assert all(sha(v['model_path'])==v['status']['model_sha256'] for v in sources.values())
        result=dict(complete=True,exit_code=0,source_model=selected['name'],source_model_sha256=choice['source_model_sha256'],
            selected_tflite_sha256=selected['meta']['model_sha256'],output_kind=selected['kind'],temperature=selected['temperature'],policy=selected['policy'],
            cuts_integer_units=cuts.tolist(),selection_before_calibration=selected['metric'],selection_after_calibration=selection_after,
            calibration_after_fit=cal_metric,calibration_is_threshold_fitting_not_independent_test=True,
            export_plan_sha256=sha(PLAN),feature_arrays_saved=False,fresh_test_used=False,board_validated=False,
            actual_device_days=0,actual_product_targets_verified=False,elapsed_seconds=time.monotonic()-started)
        write_json(OUT/'status.json',result)
        bundle=REPORT/'candidates/native40_r06_v1';bundle.mkdir(exist_ok=False)
        for filename in ['model_int8.tflite','model_int8.h','model_metadata.json']:shutil.copyfile(selected['directory']/filename,bundle/filename)
        write_json(bundle/'delivered_output_contract.json',dict(class_order=di.LABELS,output_kind=selected['kind'],
            output_scale=selected['meta']['output']['scale'],output_zero_point=selected['meta']['output']['zero_point'],
            policy=selected['policy'],cuts_integer_units=cuts.tolist(),input_shape=[1,98,40,1],frontend=spec['frontend'],
            score_semantics='Argmax-gated unsigned probability' if selected['kind']=='probability' else 'Argmax-gated positive logit margin to strongest other class',
            target_argmax_required=True,threshold_comparison='score>=cut and score>0',tie_break='First maximum',
            existing_portable_frontend_compatible=True,notification_controller_on_actual_hardware_validated=False,board_validated=False))
        write_json(bundle/'analysis.json',result)
        (bundle/'README.md').write_text('Native40 trained research candidate. Uses the existing portable 98x40 frontend. '
            'Output interpretation, integer cuts and temporal policy are in delivered_output_contract.json. '
            'Real board memory/latency, microphone and notification reliability remain unverified. '
            'Underlying recording rights and model pretraining exposure are recorded in source manifests; commercial clearance is unverified.\n')
        write_json(bundle/'candidate.json',dict(kind='NATIVE40_R06_RESEARCH_CANDIDATE',source_model=selected['name'],
            source_model_sha256=choice['source_model_sha256'],actual_device_days=0,deployment_ready=False,actual_product_targets_verified=False,
            files_sha256={p.name:sha(p) for p in bundle.iterdir() if p.is_file()}))
        print('NATIVE40_ACTUAL_INT8_BUNDLE_COMPLETE',selected['name'],selection_after['fire_recall'],selection_after['doorbell_recall'],flush=True)
    except Exception as error:
        state.update(exit_code=1,first_error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        write_json(OUT/'status.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','evaluate']);a=p.parse_args();register() if a.mode=='register' else evaluate()
