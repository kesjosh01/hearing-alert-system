"""Keep BN frozen and determinism enabled; use supported primitive BN math."""
import argparse
import json
import numpy as np
from comparison.dataset import write_json, sha
from comparison.protocol import BASE
from dscnn_baseline.common import runtime
from .paths import ROOT, REPORT
from .stage2_stream import configure_encoder
from .yamnet_frontend import transform
from .presence_mil import update_function


def clone(model,keras):
    def layer_copy(layer):
        config=layer.get_config()
        if isinstance(layer,keras.layers.BatchNormalization):config['fused']=False
        return layer.__class__.from_config(config)
    result=keras.models.clone_model(model,clone_function=layer_copy)
    result.set_weights(model.get_weights())
    assert all(np.array_equal(a,b) for a,b in zip(model.get_weights(),result.get_weights()))
    assert all(l.fused is False for l in result.layers if isinstance(l,keras.layers.BatchNormalization))
    return result


def main(device):
    target=REPORT/f'stage2_unfused_bn_{device}_gate.json';assert not target.exists()
    parent=ROOT/'runs/pursuit_20261114_r06_full_yamnet_head';status=json.loads((parent/'status.json').read_text())
    source=parent/'classifier.keras';source_hash=sha(source)
    assert status['complete'] and source_hash==status['model_sha256']
    tf,keras=runtime(device);tf.config.experimental.enable_op_determinism();keras.utils.set_random_seed(42)
    where='/GPU:0' if device=='gpu' else '/CPU:0'
    if device=='cpu':assert not tf.config.list_physical_devices('GPU')
    rng=np.random.default_rng(828)
    signals=np.stack([np.zeros(15600,np.float32),(.1*np.sin(np.arange(15600)*2*np.pi*520/16000)).astype(np.float32),rng.normal(0,.04,15600).astype(np.float32)])
    with tf.device('/CPU:0'):x=transform(tf,signals).numpy()
    with tf.device(where):
        original=keras.models.load_model(str(source),compile=False)
        model=clone(original,keras);configure_encoder(model,keras,True)
        before=[v.numpy().copy() for v in model.trainable_variables]
        bn={l.name:[a.copy() for a in l.get_weights()] for l in model.layers if isinstance(l,keras.layers.BatchNormalization)}
        reference=keras.Model(original.input,original.get_layer('class_logits').output)(x,training=False).numpy()
        actual=keras.Model(model.input,model.get_layer('class_logits').output)(x,training=False).numpy()
        error=float(np.max(np.abs(actual-reference)));assert error<5e-4,error
        update=update_function(tf,keras,model,keras.optimizers.Adam(.00003),dict(BASE,kd_weight=0.),False,device=where)
        losses=[]
        for _ in range(2):
            value=update([(x,np.eye(3,dtype=np.float32),np.arange(3,dtype=np.int64),np.zeros((3,3),np.float32))]).numpy()
            assert np.isfinite(value).all();losses.append(value.tolist())
        changed=sum(not np.array_equal(a,v.numpy()) for a,v in zip(before,model.trainable_variables))
        assert changed>0,'No trainable tensor changed in the synthetic update'
        assert all(v.device.endswith(where.lstrip('/')) for v in model.variables),'Unexpected variable device'
        assert all(all(np.array_equal(a,b) for a,b in zip(bn[l.name],l.get_weights())) for l in model.layers if l.name in bn)
    assert sha(source)==source_hash
    write_json(target,dict(passed=True,device=device,synthetic_only=True,real_audio_files=0,
        source_model_sha256=source_hash,code_sha256=sha(ROOT/'pursuit/stage2_unfused_bn.py'),
        source_weights_and_frozen_bn_values_identical_before_update=True,
        sample_forward_logits_max_abs=error,sample_forward_tolerance=5e-4,
        deterministic_ops_enabled=True,all_batch_normalization_fused_false=True,
        batch_normalization_trainability_or_values_changed=False,synthetic_optimizer_steps=2,
        trainable_arrays_changed=changed,finite_losses=losses,original_checkpoint_modified=False,
        packages_or_environment_changed=False,actual_product_targets_verified=False))
    print('FROZEN_UNFUSED_BN_DETERMINISTIC_GATE_PASS',device,error,changed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('device',choices=['cpu','gpu']);a=p.parse_args();main(a.device)
