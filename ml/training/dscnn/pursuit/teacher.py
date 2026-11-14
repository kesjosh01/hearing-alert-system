"""Hash-bound, aligned frozen teacher logits on training sources only."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from dscnn_baseline.common import runtime
from comparison.dataset import read_csv,write_json,sha
from .paths import ROOT,DATA,REPORT
from .embeddings import load as embeddings
from .data import feature

DEST=ROOT/'.cache/pursuit_teacher_v1'


def load(row,teacher):
    if row['role']!='train':raise ValueError('Distillation targets belong to training only')
    model=ROOT/'runs'/teacher/'classifier.keras';mh=sha(model);m=feature(row)
    p=DEST/mh/(m['source_sha256']+'.npy');r=json.loads(p.with_suffix('.json').read_text())
    assert r['teacher_model_sha256']==mh and r['source_sha256']==m['source_sha256'] and r['starts']==m['starts'] and sha(p)==r['logits_sha256']
    a=np.load(p,mmap_mode='r',allow_pickle=False);assert a.shape==(m['window_count'],3) and np.isfinite(a).all();return a,r


def prepare(dataset,teacher):
    d=DATA/dataset;f=json.loads((d/'freeze.json').read_text());assert all(sha(d/k)==v for k,v in f['files'].items())
    run=ROOT/'runs'/teacher;status=json.loads((run/'status.json').read_text());assert status['complete'] and status['model_type']=='frozen_yamnet_mlp128'
    model_path=run/'classifier.keras';mh=sha(model_path);assert status['model_sha256']==mh
    rr=[r for r in read_csv(d/'development.csv') if r['role']=='train'];directory=DEST/mh;directory.mkdir(parents=True,exist_ok=True)
    tf,keras=runtime('gpu');tf.config.experimental.enable_op_determinism()
    with tf.device('/GPU:0'):
        model=keras.models.load_model(str(model_path),compile=False);logit_model=keras.Model(model.input,model.get_layer('class_logits').output)
    started=time.monotonic();receipts=[]
    for i,row in enumerate(rr,1):
        x,meta=embeddings(row);m=feature(row);p=directory/(m['source_sha256']+'.npy')
        if not p.with_suffix('.json').exists():
            if p.exists():raise FileExistsError('Preserve unreceipted target array')
            with tf.device('/GPU:0'):a=np.concatenate([logit_model(x[j:j+256],training=False).numpy() for j in range(0,len(x),256)])
            assert a.shape==(m['window_count'],3) and np.isfinite(a).all()
            with p.open('xb') as h:np.save(h,a,allow_pickle=False)
            write_json(p.with_suffix('.json'),dict(source_sha256=m['source_sha256'],starts=m['starts'],window_count=m['window_count'],teacher=teacher,
                teacher_model_sha256=mh,embedding_sha256=meta['embedding_sha256'],logits_sha256=sha(p),role='train',pseudo_window_labels_not_human_event_annotations=True))
        _,receipt=load(row,teacher);receipts.append(dict(sample_id=row['sample_id'],**receipt))
        if i%200==0:print('PURSUIT_TEACHER_TARGETS',i,len(rr),flush=True)
    write_json(REPORT/(dataset+'_'+teacher+'_targets.json'),dict(complete=True,teacher=teacher,teacher_model_sha256=mh,files=len(rr),receipts=receipts,
        dataset_freeze_sha256=sha(d/'freeze.json'),fresh_test_or_calibration_audio=False,elapsed_seconds=time.monotonic()-started))
    print('PURSUIT_ALIGNED_TEACHER_TARGETS_COMPLETE',dataset,teacher,len(rr),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('teacher');a=p.parse_args();prepare(a.dataset,a.teacher)
