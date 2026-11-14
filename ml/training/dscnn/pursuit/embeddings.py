"""Frozen official YAMNet reference features, never reading fresh-test audio."""
import argparse
import json
import time
import wave
from pathlib import Path
import numpy as np
from dscnn_baseline.common import runtime
from dscnn_baseline.frontend import windows
from expansion.protocol import BACKBONE,EMBEDDINGS
from expansion.runtime import backbone_embed
from comparison.dataset import read_csv,write_json,sha
from .paths import ROOT,DATA,REPORT
from .data import feature,CACHE

DEST=ROOT/'.cache/pursuit_yamnet_embeddings_v1'


def load(row):
    m=feature(row);receipt=DEST/(m['source_sha256']+'.json');r=json.loads(receipt.read_text());p=Path(r['embedding_path'])
    assert r['source_sha256']==m['source_sha256'] and r['window_count']==m['window_count'] and sha(p)==r['embedding_sha256']
    assert r['backbone_pb_sha256']==sha(BACKBONE/'saved_model.pb')
    return np.load(p,mmap_mode='r',allow_pickle=False),r


def prepare(name):
    output=REPORT/(name+'_yamnet_embeddings.json')
    if output.exists():raise FileExistsError('Preserve completed feature stage')
    d=DATA/name;frozen=json.loads((d/'freeze.json').read_text());assert all(sha(d/k)==v for k,v in frozen['files'].items())
    rr=read_csv(d/'development.csv');assert all(r['role']!='fresh_test' for r in rr)
    tf,_=runtime('gpu');tf.config.experimental.enable_op_determinism();_,embed=backbone_embed(tf);bh=sha(BACKBONE/'saved_model.pb')
    DEST.mkdir(parents=True,exist_ok=True);receipts=[];started=time.monotonic();reused=0
    for index,r in enumerate(rr,1):
        m=feature(r);target=DEST/(m['source_sha256']+'.npy');meta=target.with_suffix('.json')
        if not meta.exists():
            previous=EMBEDDINGS/(r['sample_id']+'.npy');oldmeta=previous.with_suffix('.json')
            if previous.exists() and oldmeta.exists():
                old=json.loads(oldmeta.read_text());assert old['source_sha256']==m['source_sha256'] and old['window_count']==m['window_count'] and old['backbone_pb_sha256']==bh and sha(previous)==old['embedding_sha256']
                embedding_path=previous;reused+=1
            else:
                if target.exists():raise FileExistsError('Preserve partial feature file; inspect before resuming')
                with wave.open(str(CACHE/m['source_sha256']/'audio.wav')) as h:pcm=np.frombuffer(h.readframes(h.getnframes()),dtype='<i2')
                x=np.stack([embed(chunk[:15600].astype(np.float32)/32768).numpy() for _,chunk in windows(pcm)])
                assert x.shape==(m['window_count'],1024) and np.isfinite(x).all()
                with target.open('xb') as h:np.save(h,x,allow_pickle=False)
                embedding_path=target
            write_json(meta,dict(source_sha256=m['source_sha256'],embedding_path=str(embedding_path),embedding_sha256=sha(embedding_path),
                       window_count=m['window_count'],starts=m['starts'],backbone_pb_sha256=bh,frontend='Frozen official YAMNet .975sec per aligned one-second window',
                       existing_cache_read_only=embedding_path==previous if previous.exists() else False))
        _,record=load(r);receipts.append(dict(sample_id=r['sample_id'],role=r['role'],**record))
        if index%100==0:print('PURSUIT_YAMNET_FEATURES',index,len(rr),'old_cache_reused',reused,flush=True)
    write_json(output,dict(complete=True,files=len(receipts),receipts=receipts,reused_previous_read_only_embeddings=reused,
                         fresh_test_files=0,backbone_weights_unchanged=sha(BACKBONE/'saved_model.pb')==bh,elapsed_seconds=time.monotonic()-started))
    print('PURSUIT_YAMNET_PREPARATION_COMPLETE',name,len(receipts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();prepare(a.name)
