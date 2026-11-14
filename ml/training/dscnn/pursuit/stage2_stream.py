"""Bounded in-memory training features; never reads the legacy 64-bin cache."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import subprocess
import wave
import numpy as np
from comparison.dataset import sha
from dscnn_baseline.frontend import windows
from .data import audio_path,CACHE as NATIVE_CACHE
from .native import NativeFrontend
from .yamnet_frontend import transform,CONTRACT


class FeatureStore:
    def __init__(self,rows,tf,kind='official64',budget_bytes=1536*2**20):
        if kind not in ('official64','portable40'):raise ValueError('Unknown feature contract')
        if any(r['role'] not in ('train','selection','calibration') for r in rows):
            raise ValueError('Unseen/historical test inputs are prohibited in this fitting loader')
        assert len({r['sample_id'] for r in rows})==len(rows)
        self.rows={r['sample_id']:dict(r) for r in rows};self.tf=tf;self.kind=kind;self.budget=budget_bytes
        self.arrays=OrderedDict();self.bytes=0;self.checked={};self.provenance={};self.native=None

    def validate_row(self,row):
        if row['role'] not in ('train','selection','calibration'):
            raise ValueError('Test waveforms cannot enter the fitting feature loader')
        if row['sample_id'] not in self.rows or row!=self.rows[row['sample_id']]:
            raise ValueError('Feature request is not an exact frozen development row')

    def pcm(self,row):
        self.validate_row(row)
        path=audio_path(row);st=path.stat();stamp=(st.st_size,st.st_mtime_ns,st.st_ino)
        digest=row.get('source_sha256') or sha(path)
        if self.checked.get(digest)!=stamp:
            assert sha(path)==digest;self.checked[digest]=stamp
        cached=NATIVE_CACHE/digest;receipt=cached/'metadata.json';raw=cached/'audio.wav'
        if receipt.exists() and raw.exists():
            m=json.loads(receipt.read_text());assert m['source_sha256']==digest
            with wave.open(str(raw)) as w:
                assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2)
                data=w.readframes(w.getnframes())
            assert hashlib.sha256(data).hexdigest()==m['decoded_pcm_sha256']
        else:
            data=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',str(path),'-ac','1','-ar','16000',
                '-c:a','pcm_s16le','-f','s16le','-'],stderr=subprocess.PIPE)
        assert data and len(data)%2==0
        pcm_sha=hashlib.sha256(data).hexdigest();expected=row.get('canonical_pcm_sha256') or row.get('decoded_pcm_sha256')
        if expected:assert expected==pcm_sha
        return np.frombuffer(data,dtype='<i2'),digest,pcm_sha

    def full(self,row):
        self.validate_row(row)
        digest=row.get('source_sha256') or sha(audio_path(row));key=(digest,self.kind)
        if key in self.arrays:
            self.arrays.move_to_end(key);return self.arrays[key]
        pcm,digest,pcm_sha=self.pcm(row);pairs=list(windows(pcm))
        if self.kind=='official64':
            with self.tf.device('/CPU:0'):
                x=np.concatenate([transform(self.tf,np.stack([a[:15600].astype(np.float32)/32768. for _,a in pairs[i:i+128]])).numpy()
                    for i in range(0,len(pairs),128)])
            shape=(96,64,1)
        else:
            if self.native is None:self.native=NativeFrontend()
            x=np.stack([self.native(np.ascontiguousarray(a)) for _,a in pairs]);shape=(98,40,1)
        assert x.shape==(len(pairs),*shape) and np.isfinite(x).all()
        self.provenance[digest]=dict(source_sha256=digest,decoded_pcm_sha256=pcm_sha,sample_count=len(pcm),
            starts=[s for s,_ in pairs],window_count=len(pairs),feature_kind=self.kind,
            feature_values_sha256=hashlib.sha256(x.tobytes()).hexdigest(),features_written_to_disk=False)
        while self.arrays and self.bytes+x.nbytes>self.budget:
            _,old=self.arrays.popitem(last=False);self.bytes-=old.nbytes
        if x.nbytes<=self.budget:self.arrays[key]=x;self.bytes+=x.nbytes
        return x

    def indices(self,row,length):
        kind=row.get('training_region_kind') or ''
        if kind:
            assert row['role']=='train' and kind in ('published_target_present','published_target_absent')
            idx=np.asarray(json.loads(row['training_window_indices_json']),np.int64)
            assert len(idx)>0 and len(np.unique(idx))==len(idx) and idx.min()>=0 and idx.max()<length
            if kind=='published_target_absent':assert row['label']=='other' and row.get('original_source_sample_id')
            return idx
        assert not row.get('training_window_indices_json')
        return np.arange(length)

    def get(self,row):
        x=self.full(row);idx=self.indices(row,len(x));return x[idx],idx

    def __getitem__(self,sample_id):return self.get(self.rows[sample_id])[0]


def encoder_teacher(keras,seed=42,refine=False):
    from .yamnet_prefix import network,assign,DIRECTORY
    keras.utils.set_random_seed(seed)
    verification=json.loads((DIRECTORY/'verification.json').read_text());archive=DIRECTORY/'frozen_variables.npz'
    assert verification['passed'] and sha(archive)==verification['archive_sha256']
    base=network(keras,(96,64,1),14,True,seed)
    with np.load(archive,allow_pickle=False) as h:values={k:h[k] for k in h.files}
    assert len(assign(base,values,True))==110
    x=keras.layers.Dense(128,activation='relu',name='projection')(base.get_layer('official_embedding').output)
    x=keras.layers.Dropout(.2,name='head_dropout')(x)
    logits=keras.layers.Dense(3,name='class_logits')(x)
    model=keras.Model(base.input,keras.layers.Softmax(name='class_probability')(logits),name='full_yamnet_teacher_strong_views')
    configure_encoder(model,keras,refine)
    return model


def configure_encoder(model,keras,refine):
    for layer in model.layers:
        if layer.name.startswith('layer'):
            layer.trainable=bool(refine) and not isinstance(layer,keras.layers.BatchNormalization)
    assert all(not l.trainable for l in model.layers if isinstance(l,keras.layers.BatchNormalization))


def logit_predictor(model,keras,tf,device='/GPU:0'):
    net=keras.Model(model.input,model.get_layer('class_logits').output)
    @tf.function(input_signature=[tf.TensorSpec([None,96,64,1],tf.float32)])
    def call(x):
        with tf.device(device):return net(x,training=False)
    return call


def infer_all(model,rows,store,keras,tf,chunk=128,call=None):
    """Batch whole-recording inference; no temporal annotations used in selection."""
    if call is None:call=logit_predictor(model,keras,tf)
    result={}
    for row in rows:
        assert row['role']=='selection' and not row.get('training_region_kind')
        x,idx=store.get(row)
        result[row['sample_id']]=np.concatenate([call(x[i:i+chunk]).numpy() for i in range(0,len(x),chunk)])
    return result
