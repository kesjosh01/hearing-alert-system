"""Verify real train-only native/official window alignment for masked KD."""
import json
import numpy as np
from comparison.dataset import read_csv,write_json,sha
from dscnn_baseline.common import runtime
from .paths import ROOT,DATA,REPORT
from .stage2_stream import FeatureStore,logit_predictor
from .stage2_student_model import build
from .stage2_student_fit import batches


def main():
    target=REPORT/'stage2_dual_frontend_training_alignment_cpu_gate.json';assert not target.exists()
    directory=DATA/'round06_local_regions';freeze=json.loads((directory/'freeze.json').read_text());assert all(sha(directory/k)==v for k,v in freeze['files'].items())
    rows=[r for r in read_csv(directory/'development.csv') if r['role']=='train'];lookup={r['sample_id']:r for r in rows}
    negative=next(r for r in rows if r.get('training_region_kind')=='published_target_absent')
    positive=lookup[negative['original_source_sample_id']]
    domestic=next(r for r in rows if r['sampler_pool']=='domestic_target' and not r.get('training_region_kind'))
    selected=[positive,negative,domestic];tf,keras=runtime('cpu');assert not tf.config.list_physical_devices('GPU')
    path=ROOT/'runs/pursuit_20261114_r06_full_yamnet_refined_unfused_bn/classifier.keras';original=sha(path)
    with tf.device('/CPU:0'):
        native=build(keras,42);net=keras.Model(native.input,native.get_layer('class_logits').output)
        teacher=keras.models.load_model(str(path),compile=False)
    @tf.function(input_signature=[tf.TensorSpec([None,98,40,1],tf.float32)])
    def native_call(x):
        with tf.device('/CPU:0'):return net(x,training=False)
    teacher_call=logit_predictor(teacher,keras,tf,device='/CPU:0')
    store=FeatureStore(selected,tf,kind='portable40');teacher_store=FeatureStore(selected,tf);cache={}
    pieces=list(batches(selected,store,native_call,np.random.default_rng(661),3,teacher_store,teacher_call,cache))
    assert len(pieces)==1;x,w,y,targets=pieces[0]
    assert x.shape[1:]==(98,40,1) and w.shape==(3,len(x)) and targets.shape==(len(x),4)
    assert np.isfinite(x).all() and np.isfinite(targets).all() and set(targets[:,3]).issubset({0.,1.})
    assert np.allclose(w.sum(1),1.)
    errors=[];by_digest={r['source_sha256']:r for r in selected}
    for (digest,index),value in cache.items():
        row=by_digest[digest];full=teacher_store.full(row);expected=teacher_call(full[index:index+1]).numpy()[0]
        error=float(np.max(np.abs(value-expected)));assert error<5e-4;errors.append(error)
        assert store.provenance[digest]['starts']==teacher_store.provenance[digest]['starts']
    assert errors and sha(path)==original
    write_json(target,dict(passed=True,execution='Explicit CPU; real training inputs, no optimizer updates',
        training_views=[r['sample_id'] for r in selected],unique_canonical_recordings=len({r['canonical_pcm_sha256'] for r in selected}),
        includes_published_positive_guarded_negative_and_domestic_weak_views=True,
        native_and_official_frontend_global_start_indices_identical=True,
        actual_teacher_logits_match_independent_single_window_reference=True,maximum_logit_difference=max(errors),
        source_checkpoint_sha256=original,source_checkpoint_modified=False,feature_arrays_saved=False,fresh_test_inputs=0,
        actual_product_targets_verified=False,
        code_identity={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'pursuit/stage2_alignment_gate.py',ROOT/'pursuit/stage2_student_fit.py',ROOT/'pursuit/stage2_stream.py']}))
    print('DUAL_FRONTEND_REAL_TRAINING_KD_ALIGNMENT_CPU_GATE_PASS',max(errors),flush=True)


if __name__=='__main__':main()
