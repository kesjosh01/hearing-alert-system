"""Campaign-specific manifests and portable-frontend cache; originals read-only."""
import argparse
import hashlib
import json
import re
import subprocess
import wave
import threading
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from dscnn_baseline.common import YAMNET
from dscnn_baseline.frontend import windows
from comparison.dataset import read_csv,write_csv,write_json,sha
from .paths import ROOT,DATA,REPORT
from .native import NativeFrontend,DIRECTORY

CACHE=ROOT/'.cache/pursuit_features_v1'
_LOCKS={}
_REGISTRY_LOCK=threading.Lock()


def confirmed_new():
    plan=json.loads((DATA/'fsd_collection_plan.json').read_text());valid={}
    for receipt in (REPORT/'transport').rglob('*_complete.json'):
        j=json.loads(receipt.read_text());assert j['revision']==plan['revision']
        for k,v in j['sha256'].items():valid[k]=v
    out=[]
    for r in plan['rows']:
        if r['hf_path'] not in valid:continue
        path=DATA/'fsd_audio'/r['hf_path'];expected=valid[r['hf_path']]
        metadata=path.parent.parent.parent/'.cache/huggingface/download'/r['hf_path']
        metadata=Path(str(metadata)+'.metadata');lines=metadata.read_text().splitlines()
        assert lines[0]==plan['revision'] and lines[1]==expected and sha(path)==expected
        # An omitted/incomplete target label must never silently make a
        # target-sounding title into an other training/evaluation example.
        text=re.sub(r'[_\-.]+',' ',(r['title']+' '+' '.join(r['tags'])).lower())
        if r['label']=='other' and re.search(r'\b(door\s*bell|fire\s*alarm|smoke\s*(alarm|detector)|siren|feueralarm)\b',text):continue
        out.append(dict(r,audio_root='dscnn',relative_path=str(path.relative_to(ROOT)),source_sha256=expected))
    return out


def build(name,hard_shards=4):
    out=DATA/name
    if out.exists():raise FileExistsError('Preserve round membership and its test boundary')
    p=REPORT/'transport/targets/status.json'
    if not p.exists() or not json.loads(p.read_text())['complete']:raise RuntimeError('All target shards must finish before round freeze')
    # A broad background queue remains active; this explicit development round
    # is a versioned intermediate model, never the full-goal completion.
    eligible=confirmed_new();hard_paths=set()
    for index in range(hard_shards):
        p=REPORT/'transport/hard_negatives'/f'{index:04d}_complete.json'
        if not p.exists():raise RuntimeError(f'Need verified hard-negative shard {index} before this round')
        hard_paths.update(json.loads(p.read_text())['paths'])
    new=[r for r in eligible if r['bucket']=='targets' or r['hf_path'] in hard_paths]
    old=read_csv(ROOT/'data/comparison_20261114/development.csv')
    # The previously observed public test becomes development only in this new
    # campaign. Its old manifest/predictions remain frozen and retrospective.
    for file in ['new_training.csv','fresh_test.csv']:
        old += [dict(r,role='train',split='train',previous_campaign_role=r['role']) for r in read_csv(ROOT/'data/comparison_20261114'/file)]
    plan=json.loads((DATA/'fsd_collection_plan.json').read_text());author_role={r['uploader']:r['role'] for r in plan['rows']}
    info={}
    for split in ('dev','eval'):info.update(json.loads((DATA/f'metadata/FSD50K.metadata/{split}_clips_info_FSD50K.json').read_text()))
    for r in old:
        if r['source_dataset']=='ESC-50' and r['source_recording_id'] in info:
            user=info[r['source_recording_id']]['uploader']
            if user in author_role:
                role=author_role[user];assert role!='fresh_test'
                r.update(role=role,split='train' if role=='train' else 'validation',creator_group_id='fsd50k-uploader-'+hashlib.sha256(user.encode()).hexdigest()[:16])
    rr=old+new;seen_ids=set();roles=defaultdict(set);creators=defaultdict(set);hashes=defaultdict(set)
    for r in rr:
        assert r['sample_id'] not in seen_ids;seen_ids.add(r['sample_id']);roles[r['split_group_id']].add(r['role'])
        if r.get('creator_group_id'):creators[r['creator_group_id']].add(r['role'])
        if r.get('source_sha256'):hashes[r['source_sha256']].add(r['role'])
    assert all(len(v)==1 for m in [roles,creators,hashes] for v in m.values()),'Source/known creator/exact bytes cross roles'
    out.mkdir();write_csv(out/'inventory.csv',rr)
    dev=[r for r in rr if r['role']!='fresh_test'];fresh=[r for r in rr if r['role']=='fresh_test']
    write_csv(out/'development.csv',dev);write_csv(out/'fresh_test.csv',fresh)
    counts={role:dict(Counter(r['label'] for r in rr if r['role']==role)) for role in ['train','selection','calibration','fresh_test']}
    write_json(out/'freeze.json',dict(name=name,files={k:sha(out/k) for k in ['inventory.csv','development.csv','fresh_test.csv']},
       counts=counts,new_fsd_files=len(new),hard_negative_shards=hard_shards,model_scores_used=False,
       fresh_test_features_or_model_scores_created=False,previous_observed_public_test_now_development=True,
       labels='File-level labels; no claimed human event onset annotations',independence='Known recording IDs, exact bytes and FSD creator families; all physical device identity or foundation pretraining exposure not certified'))
    print('PURSUIT_ROUND_FROZEN',name,counts,flush=True)


def audio_path(row):
    base=YAMNET if row.get('audio_root','yamnet')=='yamnet' else ROOT
    p=(base/row['relative_path']).resolve()
    assert base in p.parents and p.is_file()
    return p


def feature(row):
    digest=sha(audio_path(row))
    with _REGISTRY_LOCK:lock=_LOCKS.setdefault(digest,threading.Lock())
    with lock:return _feature_unlocked(row)


def _feature_unlocked(row):
    if row['role']=='fresh_test':raise ValueError('Fresh-test feature gate is closed during model development')
    path=audio_path(row);digest=sha(path)
    if row.get('source_sha256'):assert row['source_sha256']==digest
    directory=CACHE/digest;receipt=directory/'metadata.json';feature=directory/'features.npy'
    if receipt.exists():
        meta=json.loads(receipt.read_text());assert meta['source_sha256']==digest and meta['feature_sha256']==sha(feature)
        return meta
    directory.mkdir(parents=True,exist_ok=True)
    # The resampled waveform and features belong only to this new cache.
    pcm_path=directory/'audio.wav'
    if not pcm_path.exists():
        subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(path),'-ac','1','-ar','16000','-c:a','pcm_s16le',str(pcm_path)],check=True,capture_output=True)
    with wave.open(str(pcm_path)) as h:
        assert h.getframerate()==16000 and h.getnchannels()==1 and h.getsampwidth()==2
        pcm=np.frombuffer(h.readframes(h.getnframes()),dtype='<i2')
    native=NativeFrontend();pairs=list(windows(pcm));x=np.stack([native(np.ascontiguousarray(chunk)) for _,chunk in pairs]);np.save(feature,x,allow_pickle=False)
    meta=dict(source_sha256=digest,decoded_pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),feature_sha256=sha(feature),
              sample_count=len(pcm),duration_s=len(pcm)/16000,window_count=len(x),starts=[s for s,_ in pairs],
              native_binary_sha256=sha(DIRECTORY/'libpursuit_frontend.so'),frontend='Original portable float32 radix2 log-Mel, host shared wrapper; tested parity')
    write_json(receipt,meta);return meta


def prepare(name):
    d=DATA/name;frozen=json.loads((d/'freeze.json').read_text());assert all(sha(d/k)==v for k,v in frozen['files'].items())
    rr=read_csv(d/'development.csv');assert all(r['role']!='fresh_test' for r in rr)
    # Same raw bytes only prepare once; final manifest-level dedup follows PCM.
    unique={str(audio_path(r)):r for r in rr};records=[]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i,(r,m) in enumerate(zip(unique.values(),executor.map(feature,unique.values())),1):
            records.append(dict(sample_id=r['sample_id'],role=r['role'],**m))
            if i%100==0:print('PURSUIT_NATIVE_FEATURES',i,len(unique),flush=True)
    write_json(REPORT/(name+'_features.json'),dict(complete=True,files=len(records),windows=sum(x['window_count'] for x in records),
               fresh_test_files=0,receipts=records))
    print('PURSUIT_FEATURE_PREPARATION_COMPLETE',name,len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','prepare']);p.add_argument('name');p.add_argument('--hard-shards',type=int,default=4);a=p.parse_args()
    build(a.name,a.hard_shards) if a.mode=='freeze' else prepare(a.name)
