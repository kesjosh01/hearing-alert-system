"""Only development roles are available here. Fresh test has a separate gate."""
import json
from pathlib import Path
import numpy as np
from expansion.data import ExpandedFeatureStore
from .dataset import read_csv,sha
from .protocol import DATA,ROOT,HARD_NEGATIVE_CATEGORIES


def rows(added=False,roles=('train','selection')):
    rr=read_csv(DATA/'development.csv')
    extra=DATA/'new_training.csv'
    if added and extra.exists():rr+=read_csv(extra)
    return [r for r in rr if r['role'] in roles]


def load(rr,teacher=False):
    assert all(r['role'] in ('train','selection','calibration') for r in rr),'Fresh-test features unavailable during development'
    features={};receipts=[];store=ExpandedFeatureStore()
    for r in rr:
        if teacher:
            p=ROOT/'.cache/distillation_aligned_v1'/(r['sample_id']+'.npy')
            receipt=json.loads(p.with_suffix('.json').read_text())
            if receipt['embedding_sha256']!=sha(p):raise ValueError('Aligned embedding changed')
            a=np.load(p,mmap_mode='r',allow_pickle=False)
            if r.get('source_sha256') and receipt['source_sha256']!=r['source_sha256']:raise ValueError('Embedding/source mismatch')
            m={'source_sha256':receipt['source_sha256'],'feature_sha256':receipt['embedding_sha256'],'window_count':len(a)}
        else:
            a,m=store.load(r);m=dict(m,window_count=len(a))
        features[r['sample_id']]=a;receipts.append(dict(sample_id=r['sample_id'],role=r['role'],source_sha256=m['source_sha256'],feature_sha256=m['feature_sha256'],window_count=len(a)))
    return features,receipts


def categories(rr):
    import re
    meta=read_csv(ROOT.parent/'yamnet/data/metadata/esc50_manifest.csv')
    cats={}
    for r in meta:
        m=re.search(r'Original category=([^;]+)',r.get('notes',''));cats[r['sample_id']]=m[1] if m else ''
    return [dict(r,source_category=r.get('source_category') or cats.get(r['sample_id'],'')) for r in rr]
