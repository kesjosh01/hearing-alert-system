"""Deterministic training-only waveform variants, preserving original sources."""
import argparse
import hashlib
import json
import wave
from collections import Counter
import numpy as np
from comparison.dataset import read_csv,write_json,sha
from dscnn_baseline.frontend import windows
from .paths import ROOT,DATA,REPORT
from .data import feature,CACHE
from .native import NativeFrontend

DEST=ROOT/'.cache/pursuit_waveform_aug_v1'
KINDS=('quiet','room_snr20','room_snr10')


def pcm(row):
    m=feature(row)
    with wave.open(str(CACHE/m['source_sha256']/'audio.wav')) as h:a=np.frombuffer(h.readframes(h.getnframes()),dtype='<i2').astype(np.float32)/32768
    return a,m


def variant(row,kind,negatives):
    assert row['role']=='train' and row['label']!='other' and kind in KINDS
    a,m=pcm(row);out=DEST/m['source_sha256']/kind;meta=out/'metadata.json'
    if meta.exists():
        r=json.loads(meta.read_text());assert r['source_sha256']==m['source_sha256'] and sha(out/'features.npy')==r['feature_sha256']
        if r['noise']:
            noise_row=next(x for x in negatives if x['sample_id']==r['noise']['sample_id'])
            assert noise_row['role']=='train' and feature(noise_row)['source_sha256']==r['noise']['source_sha256']
        return r
    if out.exists():raise FileExistsError('Preserve unreceipted augmentation directory')
    seed=int(hashlib.sha256((m['source_sha256']+kind).encode()).hexdigest()[:16],16);rng=np.random.default_rng(seed)
    noise_record=None
    if kind=='quiet':b=a*10**(-12/20)
    else:
        noise_row=negatives[int(rng.integers(len(negatives)))];assert noise_row['role']=='train' and noise_row['label']=='other'
        noise,nm=pcm(noise_row);offset=int(rng.integers(max(1,len(noise))));noise=np.roll(noise,-offset)
        noise=np.tile(noise,int(np.ceil(len(a)/max(len(noise),1))))[:len(a)]
        b=a.copy()
        for delay,gain in [(0.027,.34),(.063,.21),(.119,.12),(.193,.06)]:
            lag=round(delay*16000)
            if lag<len(a):b[lag:]+=gain*a[:-lag]
        snr=20 if kind=='room_snr20' else 10
        signal_rms=max(float(np.sqrt(np.mean(b.astype(np.float64)**2))),1e-8);noise_rms=max(float(np.sqrt(np.mean(noise.astype(np.float64)**2))),1e-8)
        b=b+noise*(signal_rms/noise_rms)*10**(-snr/20)
        b*=10**(-float(rng.uniform(0,9))/20)
        noise_record=dict(sample_id=noise_row['sample_id'],source_sha256=nm['source_sha256'],role='train',snr_db=snr,offset_samples=offset)
    peak=float(np.max(np.abs(b))) if len(b) else 0
    if peak>.98:b*=.98/peak
    q=np.clip(np.rint(b*32768),-32768,32767).astype('<i2');assert len(q)==len(a)
    out.mkdir(parents=True)
    with wave.open(str(out/'audio.wav'),'wb') as h:h.setnchannels(1);h.setsampwidth(2);h.setframerate(16000);h.writeframes(q.tobytes())
    pairs=list(windows(q));native=NativeFrontend();x=np.stack([native(np.ascontiguousarray(chunk)) for _,chunk in pairs]);assert len(x)==m['window_count'] and np.isfinite(x).all()
    with (out/'features.npy').open('xb') as h:np.save(h,x,allow_pickle=False)
    r=dict(source_sha256=m['source_sha256'],role='train',kind=kind,seed=seed,noise=noise_record,starts=[s for s,_ in pairs],window_count=len(x),
       feature_sha256=sha(out/'features.npy'),audio_sha256=sha(out/'audio.wav'),label_basis='Same weak source label, synthetic waveform transformation; no added independent event/device',
       human_reviewed_event_onsets=False,transform='-12dB gain' if kind=='quiet' else 'Synthetic sparse early room reflections + training-only real background at fixed SNR and random gain')
    write_json(meta,r);return r


class Choices:
    """Choose a whole-recording variant per training bag, never per test file."""
    def __init__(self,base,row):
        self.arrays=[base];self.names=['original'];self.counts=Counter();self.shape=base.shape;self.provenance=[]
        self.rng=np.random.default_rng(int(hashlib.sha256(row['sample_id'].encode()).hexdigest()[:16],16)^42)
        for kind in KINDS:
            d=DEST/sha_source(row)/kind;r=json.loads((d/'metadata.json').read_text());assert sha(d/'features.npy')==r['feature_sha256']
            assert r['source_sha256']==sha_source(row) and r['role']=='train'
            self.provenance.append(dict(metadata_path=str(d/'metadata.json'),metadata_sha256=sha(d/'metadata.json'),**r))
            a=np.load(d/'features.npy',mmap_mode='r',allow_pickle=False);assert a.shape==base.shape
            self.arrays.append(a);self.names.append(kind)
    def __len__(self):return len(self.arrays[0])
    def __getitem__(self,item):
        index=int(self.rng.integers(len(self.arrays)));self.counts[self.names[index]]+=1
        return self.arrays[index][item]


def sha_source(row):return feature(row)['source_sha256']


def prepare(dataset):
    d=DATA/dataset;freeze=json.loads((d/'freeze.json').read_text());assert all(sha(d/k)==v for k,v in freeze['files'].items())
    rows=[r for r in read_csv(d/'development.csv') if r['role']=='train'];positive=[r for r in rows if r['label']!='other'];negative=[r for r in rows if r['label']=='other'];assert negative
    receipts=[]
    for index,row in enumerate(positive,1):
        for kind in KINDS:receipts.append(dict(sample_id=row['sample_id'],**variant(row,kind,negative)))
        if index%25==0:print('PURSUIT_WAVEFORM_AUGMENTATION',index,len(positive),flush=True)
    write_json(REPORT/(dataset+'_waveform_augmentation.json'),dict(complete=True,source_files=len(positive),variants=len(receipts),receipts=receipts,
       independent_new_sources=0,fresh_test_or_calibration_audio_used=False,original_audio_changed=False))
    print('PURSUIT_TRAIN_ONLY_WAVEFORM_AUGMENTATION_COMPLETE',dataset,len(receipts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');a=p.parse_args();prepare(a.dataset)
