"""Test a frozen teacher retaining original YAMNet alert-class classifiers.

Only existing local weights/class metadata and frozen selection audio are used.
This is pretrained semantic transfer with zero current-task optimizer updates.
"""
import argparse
import csv
import json
import shutil
import time
import traceback
import numpy as np
from comparison.dataset import read_csv,write_json,sha
from dscnn_baseline.common import runtime
from expansion.protocol import BACKBONE
from .paths import ROOT,DATA,REPORT
from .yamnet_prefix import network,assign,DIRECTORY
from .yamnet_frontend import transform
from .stage2_stream import FeatureStore,logit_predictor
from .operating import measure,key

PLAN=REPORT/'stage2_semantic_teacher_plan.json'
NAME='pursuit_20261114_r06_yamnet_semantic_prior'
CLASS_MAP=BACKBONE/'assets/yamnet_class_map.csv'
FIRE={'/m/012n7d','/m/012ndj','/m/01y3hg','/m/03kmc9','/m/04qvtq','/m/0c3f7m','/m/0dgbq'}
BELL={'/m/03wwcy','/m/07r67yg'}


def class_indices():
    with CLASS_MAP.open() as f:rows=list(csv.DictReader(f))
    groups=[[r for r in rows if r['mid'] in mids] for mids in [FIRE,BELL]]
    assert [len(g) for g in groups]==[7,2]
    return groups


def register():
    assert not PLAN.exists()
    stop=REPORT/'collection_stopped_by_user.json';assert json.loads(stop.read_text())['collection_disabled_by_user']
    directory=DATA/'round06_local_regions';f=json.loads((directory/'freeze.json').read_text())
    assert all(sha(directory/k)==v for k,v in f['files'].items())
    verification=json.loads((DIRECTORY/'verification.json').read_text())
    assert verification['passed'] and sha(DIRECTORY/'frozen_variables.npz')==verification['archive_sha256']
    write_json(PLAN,dict(created_at_unix=time.time(),name=NAME,dataset='round06_local_regions',
        dataset_freeze_sha256=sha(directory/'freeze.json'),class_map_sha256=sha(CLASS_MAP),class_groups=class_indices(),
        original_weight_archive_sha256=verification['archive_sha256'],
        hypothesis='Retaining original alert-specific classifier weights may generalize better than the newly fitted 3-class heads on the available source-held-out recordings.',
        architecture='Original full YAMNet embedding; select original 7 fire/siren and 2 doorbell Dense classifiers by official MID; class logits are group maxima and a constant zero other reference, followed by 3-class softmax.',
        current_task_optimizer_steps=0,all_weights_frozen=True,
        selection_readout=dict(mode='margin',temperature=1.,policy=[[1,1],[1,1]]),
        stronger_teacher_gate='No worse on either alert class, strictly higher target-normalized minimum than the old best checkpoint/readout on identical parent selection IDs.',
        minimum_disk_reserve_bytes=8*2**30,feature_arrays_saved=False,collection_allowed=False,
        calibration_or_fresh_used=False,foundation_pretraining_exposure_certified_absent=False,actual_product_targets_verified=False,
        code_identity={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'pursuit/stage2_semantic_teacher.py',ROOT/'pursuit/stage2_stream.py',ROOT/'pursuit/yamnet_prefix.py',ROOT/'pursuit/operating.py']}))
    print('FROZEN_SEMANTIC_TEACHER_HYPOTHESIS_REGISTERED',flush=True)


def build(keras,groups):
    base=network(keras,(96,64,1),14,True,42)
    with np.load(DIRECTORY/'frozen_variables.npz',allow_pickle=False) as h:values={k:h[k] for k in h.files}
    assert len(assign(base,values,True))==110
    embedding=base.get_layer('official_embedding').output
    kernel,bias=base.get_layer('dense').get_weights();outputs=[]
    for label,group in zip(['fire','doorbell'],groups):
        indices=[int(r['index']) for r in group]
        dense=keras.layers.Dense(len(indices),trainable=False,name=label+'_original_class_logits')
        z=dense(embedding);dense.set_weights([kernel[:,indices],bias[indices]])
        z=keras.layers.Reshape((len(indices),1),name=label+'_class_axis')(z)
        z=keras.layers.MaxPooling1D(len(indices),name=label+'_class_max')(z)
        outputs.append(keras.layers.Flatten(name=label+'_score')(z))
    other=keras.layers.Dense(1,trainable=False,kernel_initializer='zeros',bias_initializer='zeros',name='other_zero_reference')(embedding)
    logits=keras.layers.Concatenate(name='class_logits')([*outputs,other])
    model=keras.Model(base.input,keras.layers.Softmax(name='class_probability')(logits),name='yamnet_frozen_semantic_prior')
    model.trainable=False
    return model,keras.Model(base.input,base.get_layer('dense').output)


def evaluate():
    spec=json.loads(PLAN.read_text());assert all(sha(ROOT/k)==v for k,v in spec['code_identity'].items())
    assert sha(CLASS_MAP)==spec['class_map_sha256'] and class_indices()==spec['class_groups']
    assert sha(DIRECTORY/'frozen_variables.npz')==spec['original_weight_archive_sha256']
    directory=DATA/spec['dataset'];assert sha(directory/'freeze.json')==spec['dataset_freeze_sha256']
    frozen=json.loads((directory/'freeze.json').read_text());assert all(sha(directory/k)==v for k,v in frozen['files'].items())
    rows=[r for r in read_csv(directory/'development.csv') if r['role']=='selection']
    out=ROOT/'runs'/NAME;out.mkdir(exist_ok=False);started=time.monotonic()
    status=dict(complete=False,exit_code=None,actual_optimizer_steps=0,pretrained_semantic_transfer=True,
        trained_on_current_dataset=False,fresh_test_used=False,calibration_used=False,feature_arrays_saved=False,actual_product_targets_verified=False)
    write_json(out/'status.json',status)
    try:
        tf,keras=runtime('cpu');assert not tf.config.list_physical_devices('GPU');keras.utils.set_random_seed(42)
        with tf.device('/CPU:0'):model,original_logits=build(keras,spec['class_groups'])
        rng=np.random.default_rng(842)
        signals=np.stack([np.zeros(15600,np.float32),(.1*np.sin(np.arange(15600)*2*np.pi*520/16000)).astype(np.float32),rng.normal(0,.04,15600).astype(np.float32)])
        with tf.device('/CPU:0'):
            x=transform(tf,signals);reference=original_logits(x,training=False).numpy()
            actual=keras.Model(model.input,model.get_layer('class_logits').output)(x,training=False).numpy()
        expected=np.stack([reference[:,[int(r['index']) for r in group]].max(1) for group in spec['class_groups']]+[np.zeros(3)],axis=1)
        error=float(np.max(np.abs(expected-actual)));assert error<2e-4,error
        assert not model.trainable_variables
        write_json(out/'original_classifier_mapping_gate.json',dict(passed=True,synthetic_only=True,real_audio_files=0,
            original_dense_group_logit_max_abs=error,all_original_weight_values_preserved=True,class_map_sha256=sha(CLASS_MAP),optimizer_steps=0))
        assert shutil.disk_usage(ROOT).free>spec['minimum_disk_reserve_bytes']+32*2**20
        model.save(str(out/'classifier.keras'))
        store=FeatureStore(rows,tf,budget_bytes=1536*2**20);call=logit_predictor(model,keras,tf,device='/CPU:0');logits={}
        for i,row in enumerate(rows,1):
            x,_=store.get(row);logits[row['sample_id']]=np.concatenate([call(x[j:j+128]).numpy() for j in range(0,len(x),128)])
            if i%100==0:
                status.update(selection_recordings_scored=i,elapsed_seconds=time.monotonic()-started);write_json(out/'status.json',status)
                print('SEMANTIC_PRIOR_CPU_SELECTION',i,len(rows),flush=True)
        metric=measure(rows,logits,**spec['selection_readout']);np.savez(out/'selection_logits.npz',**logits)
        write_json(out/'selection_metrics.json',metric);write_json(out/'feature_stream_provenance.json',list(store.provenance.values()))
        write_json(out/'run.json',dict(specification_sha256=sha(PLAN),dataset_freeze_sha256=spec['dataset_freeze_sha256'],
            class_groups=spec['class_groups'],parameters=int(model.count_params()),source_weights_modified=False,actual_optimizer_steps=0))
        parent=[r for r in read_csv(DATA/'round04_enriched/development.csv') if r['role']=='selection']
        assert {r['sample_id'] for r in parent}.issubset(logits)
        matched=measure(parent,logits,**spec['selection_readout'])
        bp=REPORT/'operating/round04_enriched/pursuit_20261114_r04_prefix64_pretrained.json';baseline=json.loads(bp.read_text())
        assert sha(ROOT/'runs'/baseline['model']/'classifier.keras')==baseline['model_sha256']
        b=baseline['best'];stronger=matched['fire_recall']>=b['fire_recall'] and matched['doorbell_recall']>=b['doorbell_recall'] and key(matched)>key(b)
        model_hash=sha(out/'classifier.keras')
        write_json(REPORT/'stage2_semantic_teacher_matched_comparison.json',dict(complete=True,common_selection_files=len(parent),
            chosen_teacher=NAME,chosen_model_sha256=model_hash,teacher_type='Frozen original class-specific pretrained weights, zero new-task optimizer steps',
            selection_only=True,baseline_model=baseline['model'],baseline_previous_best_readout=b,
            matched_parent_selection=matched,full_frozen_selection=metric,distillation_gate_passed=bool(stronger),
            calibration_or_fresh_used=False,repeated_selection_is_not_independent_test=True,actual_product_targets_verified=False))
        status.update(complete=True,exit_code=0,model_sha256=model_hash,selected_metrics=metric,matched_parent_selection=matched,
            distillation_gate_passed=bool(stronger),selection_recordings_scored=len(rows),elapsed_seconds=time.monotonic()-started)
        write_json(out/'status.json',status);print('SEMANTIC_PRIOR_TEACHER_SELECTION_COMPLETE','KD_ALLOWED',stronger,matched['fire_recall'],matched['doorbell_recall'],flush=True)
    except Exception as error:
        status.update(exit_code=1,first_error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        write_json(out/'status.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','evaluate']);a=p.parse_args()
    register() if a.mode=='register' else evaluate()
