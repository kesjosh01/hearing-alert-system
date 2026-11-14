# Architecture follows the Apache-2.0 TensorFlow YAMNet reference:
# https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
# Copyright 2019 The TensorFlow Authors. See that source's Apache-2.0 notice.
"""Transfer existing YAMNet filters; leave its saved weights unchanged."""
import argparse
import json
from pathlib import Path
import numpy as np
from dscnn_baseline.common import runtime
from comparison.dataset import write_json,sha
from expansion.protocol import BACKBONE
from .paths import ROOT,REPORT

DIRECTORY=REPORT/'yamnet_transfer'
DEFS=[(32,2),(64,1),(128,2),(128,1),(256,2),(256,1),(512,2),(512,1),(512,1),(512,1),(512,1),(512,1),(1024,2),(1024,1)]


def network(keras,input_shape=(98,40,1),layers=8,official=False,seed=42):
    keras.utils.set_random_seed(seed);inputs=keras.Input(input_shape,name='log_mel');x=inputs
    for i,(width,stride) in enumerate(DEFS[:layers],1):
        def bn(value,name):return keras.layers.BatchNormalization(name=name,center=True,scale=False,epsilon=1e-4)(value)
        if i==1:
            x=keras.layers.Conv2D(width,3,strides=stride,padding='same',use_bias=False,name=f'layer{i}/conv')(x)
            x=bn(x,f'layer{i}/conv/bn');x=keras.layers.ReLU(name=f'layer{i}/relu')(x)
        else:
            x=keras.layers.DepthwiseConv2D(3,strides=stride,padding='same',use_bias=False,name=f'layer{i}/depthwise_conv')(x)
            x=bn(x,f'layer{i}/depthwise_conv/bn');x=keras.layers.ReLU(name=f'layer{i}/depthwise_conv/relu')(x)
            x=keras.layers.Conv2D(width,1,padding='same',use_bias=False,name=f'layer{i}/pointwise_conv')(x)
            x=bn(x,f'layer{i}/pointwise_conv/bn');x=keras.layers.ReLU(name=f'layer{i}/pointwise_conv/relu')(x)
    if official:
        embedding=keras.layers.GlobalAveragePooling2D(name='official_embedding')(x)
        logits=keras.layers.Dense(521,name='dense')(embedding);prob=keras.layers.Activation('sigmoid')(logits)
        return keras.Model(inputs,[prob,embedding],name='verified_official_yamnet_replica')
    x=keras.layers.AveragePooling2D((int(x.shape[1]),1),name='global_average')(x);x=keras.layers.Flatten(name='flatten')(x)
    logits=keras.layers.Dense(3,name='class_logits')(x)
    return keras.Model(inputs,keras.layers.Softmax(name='class_probability')(logits),name='yamnet_prefix'+str(layers)+'_native40')


def assign(model,values,official=False):
    copied=[]
    for layer in model.layers:
        if layer.name=='class_logits' and not official:continue
        for variable in layer.weights:
            name=layer.name+'/'+variable.name.rsplit('/',1)[-1]
            assert name in values and list(variable.shape)==list(values[name].shape),(name,list(variable.shape))
            variable.assign(values[name]);copied.append(name)
    return copied


def build_model(keras,options,seed):
    model=network(keras,layers=options['prefix_layers'],seed=seed)
    if options['pretrained']:
        receipt=json.loads((DIRECTORY/'verification.json').read_text());assert receipt['passed']
        archive=DIRECTORY/'frozen_variables.npz';assert sha(archive)==receipt['archive_sha256']
        with np.load(archive,allow_pickle=False) as h:values={k:h[k] for k in h.files}
        copied=assign(model,values);assert len(copied)==4+8*(options['prefix_layers']-1)
    return model


def extract_and_verify():
    DIRECTORY.mkdir(exist_ok=True);target=DIRECTORY/'verification.json';archive=DIRECTORY/'frozen_variables.npz'
    if target.exists() or archive.exists():raise FileExistsError('Preserve completed or partial transfer stage')
    tf,keras=runtime('cpu');import tensorflow_hub as hub
    with tf.device('/CPU:0'):backbone=hub.load(str(BACKBONE))
    variables=backbone.signatures['serving_default'].variables;values={v.name:v.numpy() for v in variables};assert len(values)==len(variables)==110
    with archive.open('xb') as h:np.savez(h,**values)
    with tf.device('/CPU:0'):
        full=network(keras,(96,64,1),14,True);copied=assign(full,values,True)
    assert len(copied)==110
    rng=np.random.default_rng(42);signals=[np.zeros(15600,np.float32),(.1*np.sin(np.arange(15600)*2*np.pi*520/16000)).astype(np.float32),rng.normal(0,.05,15600).astype(np.float32)];checks=[]
    for i,signal in enumerate(signals):
        with tf.device('/CPU:0'):
            expected_score,expected_embedding,mel=backbone(signal);actual_score,actual_embedding=full(mel[None,:96,:,None],training=False)
        errors=dict(score_max_abs=float(np.max(np.abs(actual_score.numpy()-expected_score.numpy()))),embedding_max_abs=float(np.max(np.abs(actual_embedding.numpy()-expected_embedding.numpy()))))
        assert errors['score_max_abs']<2e-4 and errors['embedding_max_abs']<2e-3,errors;checks.append(dict(synthetic_signal=i,**errors))
    write_json(target,dict(passed=True,synthetic_only=True,real_audio_files=0,all_official_variables_assigned_by_name_and_shape=True,variable_count=len(copied),archive_sha256=sha(archive),
        source_saved_model_pb_sha256=sha(BACKBONE/'saved_model.pb'),source_variable_file_hashes={str(p.relative_to(BACKBONE)):sha(p) for p in (BACKBONE/'variables').glob('*')},
        full_official_96x64_replica_checks=checks,native40_prefix_semantic_equivalence_claimed=False,native40_experiment='Fine-tune on existing portable 98x40 power-log-Mel; BN statistics adapt during train only',
        source_weights_modified=False,execution='Explicit CPU, no GPU or real audio',official_reference_source='https://github.com/tensorflow/models/tree/master/research/audioset/yamnet'))
    print('OFFICIAL_YAMNET_ALL_WEIGHT_MAPPING_AND_REPLICA_PASS',checks,flush=True)


if __name__=='__main__':extract_and_verify()
