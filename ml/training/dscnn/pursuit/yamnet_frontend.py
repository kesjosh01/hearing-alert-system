"""Versioned official YAMNet magnitude-log-Mel inputs, CPU preparation only.

This is a separate 96x64 frontend, not a drop-in 98x40 firmware artifact.
Windows retain the existing causal recording starts and take 15600 samples.
"""
import argparse
import hashlib
import json
import shutil
import wave
from pathlib import Path
import numpy as np
from comparison.dataset import read_csv, write_json, sha
from dscnn_baseline.common import runtime
from dscnn_baseline.frontend import windows
from expansion.protocol import BACKBONE
from .paths import ROOT, DATA, REPORT
from .data import feature as native_feature, CACHE as NATIVE_CACHE

CACHE=ROOT/'.cache/pursuit_yamnet_logmel64_v1'
CONTRACT=dict(sample_rate=16000,sample_count=15600,frame_length=400,frame_step=160,
    fft_length=512,hann_periodic=True,spectrum='magnitude',mel_bins=64,
    mel_lower_hz=125.,mel_upper_hz=7500.,log_offset=.001,shape=[96,64,1],
    waveform_scale=32768.,window_start_grid='existing 1-second/0.5-second-hop, final end-aligned')


def transform(tf,pcm):
    x=tf.convert_to_tensor(pcm,tf.float32)
    s=tf.abs(tf.signal.stft(x,400,160,512,pad_end=False))
    bank=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    return tf.math.log(tf.linalg.matmul(s,bank)+.001)[...,None]


def verify():
    target=REPORT/'yamnet_frontend64_gate.json'
    if target.exists():raise FileExistsError('Preserve completed frontend verification')
    tf,_=runtime('cpu');import tensorflow_hub as hub
    with tf.device('/CPU:0'):backbone=hub.load(str(BACKBONE))
    rng=np.random.default_rng(190);checks=[]
    signals=[np.zeros(15600,np.float32),
        (.1*np.sin(np.arange(15600)*2*np.pi*520/16000)).astype(np.float32),
        rng.normal(0,.04,15600).astype(np.float32)]
    for i,x in enumerate(signals):
        with tf.device('/CPU:0'):
            _,_,expected=backbone(x);actual=transform(tf,x).numpy()[:,:,0]
        error=float(np.max(np.abs(actual-expected.numpy()[:96])))
        assert actual.shape==(96,64) and error<2e-5,(i,error)
        checks.append(dict(synthetic_signal=i,max_abs=error))
    write_json(target,dict(passed=True,contract=CONTRACT,checks=checks,
        source_saved_model_sha256=sha(BACKBONE/'saved_model.pb'),
        real_audio_files=0,execution='Explicit CPU',firmware_frontend_replaced=False,
        installed_software=False,board_validated=False))
    print('OFFICIAL_YAMNET_FRONTEND64_PARITY_PASS',checks,flush=True)


def feature(row):
    if row['role']=='fresh_test':raise ValueError('Fresh-test feature gate remains closed')
    native=native_feature(row);directory=CACHE/native['source_sha256']
    receipt=directory/'metadata.json';p=directory/'features.npy'
    if not receipt.exists():raise FileNotFoundError('Prepare official 64-bin features on CPU before the GPU run')
    r=json.loads(receipt.read_text())
    assert r['contract']==CONTRACT and r['source_sha256']==native['source_sha256']
    assert r['decoded_pcm_sha256']==native['decoded_pcm_sha256'] and r['starts']==native['starts']
    assert sha(p)==r['feature_sha256']
    return r


def prepare(name):
    gate=json.loads((REPORT/'yamnet_frontend64_gate.json').read_text());assert gate['passed'] and gate['contract']==CONTRACT
    target=REPORT/(name+'_frontend64_features.json')
    if target.exists():raise FileExistsError('Preserve completed preparation stage')
    d=DATA/name;freeze=json.loads((d/'freeze.json').read_text());assert all(sha(d/k)==v for k,v in freeze['files'].items())
    rows=read_csv(d/'development.csv');assert all(r['role']!='fresh_test' for r in rows)
    tf,_=runtime('cpu');CACHE.mkdir(exist_ok=True);records=[]
    for i,row in enumerate(rows,1):
        native=native_feature(row);directory=CACHE/native['source_sha256'];receipt=directory/'metadata.json'
        if not receipt.exists():
            n=native['window_count']
            if shutil.disk_usage(CACHE).free < n*96*64*4+12*2**30:raise OSError('Preserve data and free-capacity reserve')
            directory.mkdir(exist_ok=True);p=directory/'features.npy'
            if p.exists():raise FileExistsError('Preserve partial feature output; inspect before resuming')
            with wave.open(str(NATIVE_CACHE/native['source_sha256']/'audio.wav')) as w:
                pcm=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2')
            chunks=[chunk[:15600].astype(np.float32)/32768. for _,chunk in windows(pcm)]
            with tf.device('/CPU:0'):
                a=np.concatenate([transform(tf,np.stack(chunks[k:k+128])).numpy() for k in range(0,n,128)])
            assert a.shape==(n,96,64,1) and np.isfinite(a).all()
            with p.open('xb') as f:np.save(f,a,allow_pickle=False)
            write_json(receipt,dict(source_sha256=native['source_sha256'],decoded_pcm_sha256=native['decoded_pcm_sha256'],
                feature_sha256=sha(p),sample_count=native['sample_count'],duration_s=native['duration_s'],
                window_count=n,starts=native['starts'],contract=CONTRACT,
                source_frontend_gate_sha256=sha(REPORT/'yamnet_frontend64_gate.json'),
                frontend='Official YAMNet magnitude log-Mel, 96x64, separately verified CPU implementation'))
        records.append(dict(sample_id=row['sample_id'],role=row['role'],**feature(row)))
        if i%200==0:print('YAMNET_FRONTEND64_PREPARED',i,len(rows),flush=True)
    roles={}
    for r in records:roles.setdefault(r['decoded_pcm_sha256'],set()).add(r['role'])
    assert all(len(v)==1 for v in roles.values()),'Decoded waveform crosses roles'
    write_json(target,dict(complete=True,files=len(records),receipts=records,
        fresh_test_files=0,contract=CONTRACT,decoded_pcm_role_disjoint=True,execution='Explicit CPU'))
    print('YAMNET_FRONTEND64_PREPARATION_COMPLETE',name,len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','prepare']);p.add_argument('name',nargs='?');a=p.parse_args()
    verify() if a.mode=='verify' else prepare(a.name)
