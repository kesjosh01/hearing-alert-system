"""Verify the memory-only loader and teacher on CPU before any new GPU fit."""
import hashlib
import io
import json
import numpy as np
from comparison.dataset import read_csv, write_json, sha
from dscnn_baseline.common import runtime
from expansion.protocol import BACKBONE
from .paths import DATA, REPORT
from .data import CACHE as NATIVE_CACHE
from .yamnet_frontend import CACHE as LEGACY_CACHE, CONTRACT
from .stage2_stream import FeatureStore, encoder_teacher, configure_encoder
from .stage2_regions import present_indices, absent_indices


def main():
    target = REPORT / 'stage2_memory_stream_cpu_gate.json'
    assert not target.exists(), 'Preserve completed or failed gate'
    tf, keras = runtime('cpu')
    assert not tf.config.list_physical_devices('GPU')
    directory = DATA / 'round05_train_targets'
    freeze = json.loads((directory / 'freeze.json').read_text())
    assert all(sha(directory / k) == v for k,v in freeze['files'].items())
    rows = read_csv(directory / 'development.csv')
    candidates = []
    for row in rows:
        p = LEGACY_CACHE / row['source_sha256'] / 'metadata.json'
        if p.exists(): candidates.append((json.loads(p.read_text())['window_count'], row, p))
    short = min(candidates, key=lambda a:a[0])
    medium = min(candidates, key=lambda a:abs(a[0]-19))
    long = min((a for a in candidates if a[0]>128), key=lambda a:a[0])
    selected = [short, medium, long]
    store = FeatureStore([a[1] for a in selected], tf, budget_bytes=8*2**20)
    checks = []
    for n,row,p in selected:
        before = p.read_bytes(); expected = json.loads(before)
        x,idx = store.get(row)
        serialized = io.BytesIO(); np.save(serialized, x, allow_pickle=False)
        digest = hashlib.sha256(serialized.getvalue()).hexdigest()
        assert expected['contract'] == CONTRACT and digest == expected['feature_sha256']
        assert p.read_bytes() == before and len(idx) == n and store.bytes <= store.budget
        checks.append(dict(sample_id=row['sample_id'], window_count=n, recomputed_npy_sha256=digest,
            exact_existing_feature_bytes_reconstructed=True, legacy_feature_array_read=False))
    row = dict(short[1], role='fresh_test')
    rejected = False
    try: store.get(row)
    except ValueError: rejected = True
    assert rejected
    rejected_constructor = False
    try: FeatureStore([row], tf)
    except ValueError: rejected_constructor = True
    assert rejected_constructor
    native = FeatureStore([short[1]], tf, kind='portable40')
    x,_ = native.get(short[1]); serialized = io.BytesIO(); np.save(serialized,x,allow_pickle=False)
    m = json.loads((NATIVE_CACHE/short[1]['source_sha256']/'metadata.json').read_text())
    assert hashlib.sha256(serialized.getvalue()).hexdigest() == m['feature_sha256']
    starts = np.array([0,8000,16000,24000,32000,40000,48000])
    positive = present_indices(starts, [[1.,2.]])
    negative = absent_indices(starts, [[1.,2.]], [[3.,4.]], .25)
    assert positive == [2,3] and negative == [], (positive,negative)
    # A longer signal exposes genuinely separate, guarded negative windows.
    negative = absent_indices(np.arange(0,8*16000,8000), [[1.,2.]], [[3.,4.]], .25)
    assert negative and all(i*.5>=4.25 for i in negative)
    import tensorflow_hub as hub
    with tf.device('/CPU:0'):
        source = hub.load(str(BACKBONE))
        model = encoder_teacher(keras)
        embedding = keras.Model(model.input,model.get_layer('official_embedding').output)
        signal = (.1*np.sin(np.arange(15600)*2*np.pi*520/16000)).astype(np.float32)
        _,expected,mel = source(signal)
        actual = embedding(mel[None,:96,:,None],training=False).numpy()
        error = float(np.max(np.abs(actual-expected.numpy())))
        assert error < 2e-3
        assert sum(np.prod(v.shape) for v in model.trainable_variables) == 131587
        configure_encoder(model,keras,True)
        assert all(not l.trainable for l in model.layers if isinstance(l,keras.layers.BatchNormalization))
        net = keras.Model(model.input,model.get_layer('class_logits').output)
        with tf.GradientTape() as tape:
            logits = net(mel[None,:96,:,None],training=True)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[0],logits=logits)[0]
        gradients = tape.gradient(loss,model.trainable_variables)
        assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
        assert any(np.any(g.numpy()!=0) for g in gradients)
    write_json(target,dict(passed=True,execution='Explicit CPU only',source_real_audio_feature_checks=checks,
        official_embedding_max_abs=error, full_teacher_parameters=int(model.count_params()),
        refinable_parameters=int(sum(np.prod(v.shape) for v in model.trainable_variables)),
        full_teacher_finite_conv_and_head_gradients=True, all_batch_normalization_frozen=True,
        unseen_row_rejected_before_io=rejected, unseen_constructor_rejected=rejected_constructor,
        portable40_existing_feature_bytes_reconstructed=True, published_present_and_guarded_absent_window_rules_passed=True,
        feature_arrays_written_to_disk=False, source_model_weights_modified=False,
        gpu_optimizer_steps=0, actual_product_targets_verified=False))
    print('STAGE2_MEMORY_STREAM_AND_FULL_TEACHER_CPU_GATE_PASS', checks, flush=True)


if __name__ == '__main__': main()
