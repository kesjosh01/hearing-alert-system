"""Prepare only newly accepted training sources; no fresh-test model access."""
import argparse
import hashlib
import json
import wave
import numpy as np
from dscnn_baseline.common import runtime,source
from dscnn_baseline.frontend import windows
from expansion.data import ExpandedFeatureStore
from expansion.runtime import backbone_embed
from expansion.protocol import BACKBONE,EMBEDDINGS
from .protocol import ROOT,DATA,REPORT
from .dataset import read_csv,write_json,sha


def features():
    rows=read_csv(DATA/'new_training.csv');store=ExpandedFeatureStore();receipts=[]
    for r in rows:
        assert r['role']=='train'
        a,m=store.load(r);receipts.append(dict(sample_id=r['sample_id'],source_sha256=m['source_sha256'],window_count=len(a),feature_sha256=m['feature_sha256']))
        print('NEW_TRAIN_FEATURES',r['sample_id'],len(a),flush=True)
    write_json(REPORT/'new_training_features.json',dict(complete=True,files=len(rows),windows=sum(r['window_count'] for r in receipts),receipts=receipts,fresh_test_files=0))


def embeddings():
    tf,_=runtime('gpu');tf.config.experimental.enable_op_determinism()
    _,embed=backbone_embed(tf);backbone_hash=sha(BACKBONE/'saved_model.pb');store=ExpandedFeatureStore();receipts=[]
    for r in read_csv(DATA/'new_training.csv'):
        assert r['role']=='train';a,m=store.load(r);target=EMBEDDINGS/(r['sample_id']+'.npy')
        if target.exists():
            meta=json.loads(target.with_suffix('.json').read_text())
            assert meta['embedding_sha256']==sha(target) and meta['source_sha256']==m['source_sha256'] and meta['backbone_pb_sha256']==backbone_hash
        else:
            directory=store.cache/source.fingerprint([m['source_sha256'],store.signature])
            with wave.open(str(directory/'audio.wav')) as h:pcm=np.frombuffer(h.readframes(h.getnframes()),dtype='<i2')
            x=np.stack([embed(chunk[:15600].astype(np.float32)/32768).numpy() for _,chunk in windows(pcm)])
            assert x.shape==(len(a),1024) and np.isfinite(x).all()
            np.save(target,x,allow_pickle=False)
            meta=dict(sample_id=r['sample_id'],source_sha256=m['source_sha256'],decoded_pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                      student_feature_sha256=m['feature_sha256'],window_starts=m['starts'],embedding_sha256=sha(target),backbone_pb_sha256=backbone_hash,window_count=len(x),role='train',event_annotation=False)
            write_json(target.with_suffix('.json'),meta)
        receipts.append(meta);print('NEW_TRAIN_EMBEDDING',r['sample_id'],len(a),flush=True)
    write_json(REPORT/'new_training_embeddings.json',dict(complete=True,files=len(receipts),receipts=receipts,fresh_test_files=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['features','embeddings']);a=p.parse_args()
    features() if a.mode=='features' else embeddings()
